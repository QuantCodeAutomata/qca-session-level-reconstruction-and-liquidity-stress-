import calendar

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class SyntheticConfig:
    n_sessions: int = 20
    session_duration_hours: float = 5.0
    base_price: float = 80.0
    tick_size: float = 0.01
    random_seed: int = 42
    n_hourly_contracts: int = 20


def _generate_contract_messages(
    rng: np.random.RandomState,
    contract_id: str,
    delivery_start: pd.Timestamp,
    delivery_end: pd.Timestamp,
    session_start: pd.Timestamp,
    session_end: pd.Timestamp,
    base_price: float,
    tick_size: float,
) -> List[dict]:
    """Generate L3 messages for a single contract within a session window."""
    session_duration_ms = int(
        (session_end - session_start).total_seconds() * 1000
    )

    n_submit = rng.randint(50, 201)

    # Submissions arrive via a (near-)Poisson process: uniform draws on [0, T]
    submit_offsets_ms = np.sort(
        rng.randint(0, session_duration_ms, n_submit).astype(np.int64)
    )

    # Unique base integers for order IDs in this contract/session
    id_base = rng.randint(10_000_000, 99_999_999, n_submit)

    messages: List[dict] = []

    for i in range(n_submit):
        t_submit = session_start + pd.Timedelta(milliseconds=int(submit_offsets_ms[i]))
        order_id = f"{contract_id}_{id_base[i]}"
        side = "bid" if rng.random() < 0.5 else "ask"

        # Bid prices below, ask prices above base_price
        # Spread between best bid and best ask: ~0.1-3.0
        half_spread = rng.uniform(0.05, 1.5)
        price_noise = rng.normal(0.0, 0.25)
        if side == "bid":
            price = base_price - half_spread + price_noise
        else:
            price = base_price + half_spread + price_noise

        # Round to nearest tick
        price = round(round(price / tick_size) * tick_size, 4)
        quantity = round(float(rng.exponential(1.0)) + 0.1, 1)

        messages.append({
            "timestamp": t_submit,
            "contract_id": contract_id,
            "delivery_start": delivery_start,
            "delivery_end": delivery_end,
            "order_id": order_id,
            "msg_type": "submit",
            "side": side,
            "price": price,
            "quantity": quantity,
        })

        # Remaining session time in ms after this submission
        remaining_ms = session_duration_ms - int(submit_offsets_ms[i])
        if remaining_ms <= 1:
            continue  # no time for a follow-up message

        fate = rng.random()
        if fate < 0.30:
            # Cancel: uniform time after submission within remaining window
            delta_ms = rng.randint(1, remaining_ms)
            t_cancel = session_start + pd.Timedelta(
                milliseconds=int(submit_offsets_ms[i]) + delta_ms
            )
            messages.append({
                "timestamp": t_cancel,
                "contract_id": contract_id,
                "delivery_start": delivery_start,
                "delivery_end": delivery_end,
                "order_id": order_id,
                "msg_type": "cancel",
                "side": side,
                "price": price,
                "quantity": quantity,
            })
        elif fate < 0.50:
            # Execute (full fill): uniform time after submission
            delta_ms = rng.randint(1, remaining_ms)
            t_exec = session_start + pd.Timedelta(
                milliseconds=int(submit_offsets_ms[i]) + delta_ms
            )
            messages.append({
                "timestamp": t_exec,
                "contract_id": contract_id,
                "delivery_start": delivery_start,
                "delivery_end": delivery_end,
                "order_id": order_id,
                "msg_type": "execute",
                "side": side,
                "price": price,
                "quantity": quantity,
            })
        # else: order remains active (50% probability)

    return messages


def generate_l3_messages(
    config: SyntheticConfig,
    month: str = "2024-01",
) -> pd.DataFrame:
    """Generate synthetic L3 messages that mimic XBID structure.

    Returns DataFrame with columns:
    timestamp (datetime64[ms]), contract_id, delivery_start, delivery_end,
    order_id, msg_type ('submit'/'cancel'/'execute'),
    side ('bid'/'ask'), price (float), quantity (float)
    """
    rng = np.random.RandomState(config.random_seed)
    year, mon = map(int, month.split("-"))

    days_in_month = calendar.monthrange(year, mon)[1]
    # All possible (day, delivery_hour) pairs; skip hours 0-5 for realism
    all_pairs = [
        (d, h) for d in range(1, days_in_month + 1) for h in range(6, 23)
    ]

    n_sessions = min(config.n_sessions, len(all_pairs))
    chosen_indices = rng.choice(len(all_pairs), n_sessions, replace=False)
    chosen_pairs = sorted(chosen_indices.tolist())

    all_messages: List[dict] = []

    for idx in chosen_pairs:
        day, delivery_hour = all_pairs[idx]

        delivery_start = pd.Timestamp(year=year, month=mon, day=day, hour=delivery_hour)
        delivery_end = delivery_start + pd.Timedelta(hours=1)
        session_start = delivery_start - pd.Timedelta(hours=config.session_duration_hours)
        session_end = delivery_start

        date_str = delivery_start.strftime("%Y%m%d")
        hour_str = f"{delivery_hour:02d}"

        # Hourly contract
        hourly_id = f"H_{date_str}_{hour_str}"

        # 4 Quarter-hourly contracts
        quarter_ids = [f"Q_{date_str}_{hour_str}_Q{q}" for q in range(1, 5)]
        quarter_del_starts = [
            delivery_start + pd.Timedelta(minutes=15 * (q - 1)) for q in range(1, 5)
        ]
        quarter_del_ends = [
            delivery_start + pd.Timedelta(minutes=15 * q) for q in range(1, 5)
        ]

        # Hourly contract messages
        all_messages.extend(
            _generate_contract_messages(
                rng, hourly_id, delivery_start, delivery_end,
                session_start, session_end,
                config.base_price, config.tick_size,
            )
        )

        # Quarter-hourly contract messages
        for q_id, q_start, q_end in zip(quarter_ids, quarter_del_starts, quarter_del_ends):
            all_messages.extend(
                _generate_contract_messages(
                    rng, q_id, q_start, q_end,
                    session_start, session_end,
                    config.base_price, config.tick_size,
                )
            )

    if not all_messages:
        return pd.DataFrame(columns=[
            "timestamp", "contract_id", "delivery_start", "delivery_end",
            "order_id", "msg_type", "side", "price", "quantity",
        ])

    df = pd.DataFrame(all_messages)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
    df["delivery_start"] = pd.to_datetime(df["delivery_start"]).astype("datetime64[ms]")
    df["delivery_end"] = pd.to_datetime(df["delivery_end"]).astype("datetime64[ms]")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def generate_session_dataset(
    months: list = ["2024-01", "2024-04", "2024-07", "2024-10"],
    n_sessions_per_month: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a full synthetic session dataset for testing."""
    frames: List[pd.DataFrame] = []

    for i, month in enumerate(months):
        config = SyntheticConfig(
            n_sessions=n_sessions_per_month,
            random_seed=seed + i * 1000,
        )
        df = generate_l3_messages(config, month=month)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("timestamp").reset_index(drop=True)
    return result
