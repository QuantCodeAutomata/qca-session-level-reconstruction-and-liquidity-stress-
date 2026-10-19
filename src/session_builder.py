import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SESSION_DURATION_HOURS = 5.0


@dataclass
class Session:
    session_id: str
    hourly_contract_id: str
    quarter_contract_ids: List[str]   # [Q1, Q2, Q3, Q4]
    delivery_start: pd.Timestamp
    delivery_end: pd.Timestamp
    session_start: pd.Timestamp       # delivery_start - 5h
    session_end: pd.Timestamp         # = hourly gate closure (delivery_start)
    month: str
    delivery_hour: int
    is_weekend: bool

    # Filled after book reconstruction
    h_events: Optional[List[float]] = None   # seconds from session_start
    q_events: Optional[List[float]] = None   # pooled Q1-Q4, seconds from session_start
    state_path: Optional[pd.DataFrame] = None  # columns: time_sec, regime
    lsi_series: Optional[pd.DataFrame] = None  # columns: timestamp, lsi, regime, …
    stressed_fraction: Optional[float] = None
    is_retained: bool = False
    exclusion_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Session identification
# ---------------------------------------------------------------------------

def identify_sessions(
    messages: pd.DataFrame,
    session_duration_hours: float = SESSION_DURATION_HOURS,
) -> List[Session]:
    """Identify all valid Hourly + QuarterHourly session groups.

    For each Hourly contract that has all 4 matching Quarter-Hourly contracts,
    build a Session with a 5-hour window ending at delivery_start.
    """
    contract_ids = set(messages["contract_id"].unique())

    hourly_ids = sorted(c for c in contract_ids if c.startswith("H_"))
    sessions: List[Session] = []

    for h_id in hourly_ids:
        parts = h_id.split("_")
        if len(parts) != 3:
            continue
        _, date_str, hour_str = parts

        q_ids = [f"Q_{date_str}_{hour_str}_Q{q}" for q in range(1, 5)]
        if not all(q in contract_ids for q in q_ids):
            continue

        h_msgs = messages.loc[messages["contract_id"] == h_id]
        if h_msgs.empty:
            continue

        delivery_start = pd.Timestamp(h_msgs["delivery_start"].iloc[0])
        delivery_end = pd.Timestamp(h_msgs["delivery_end"].iloc[0])
        session_start = delivery_start - pd.Timedelta(hours=session_duration_hours)
        session_end = delivery_start

        month = f"{delivery_start.year}-{delivery_start.month:02d}"

        sessions.append(
            Session(
                session_id=f"S_{h_id}",
                hourly_contract_id=h_id,
                quarter_contract_ids=q_ids,
                delivery_start=delivery_start,
                delivery_end=delivery_end,
                session_start=session_start,
                session_end=session_end,
                month=month,
                delivery_hour=delivery_start.hour,
                is_weekend=delivery_start.weekday() >= 5,
            )
        )

    return sessions


# ---------------------------------------------------------------------------
# Event-stream construction
# ---------------------------------------------------------------------------

def build_event_streams(
    session: Session,
    books: dict,  # contract_id -> OrderBookReconstructor
) -> Tuple[List[float], List[float]]:
    """Build H and Q event streams for a session.

    Only submission events that fall within [session_start, session_end] are
    included.  Times are expressed as float seconds from session_start.

    Returns (h_events, q_events) as sorted lists of floats.
    """
    session_start = session.session_start
    session_end = session.session_end

    def _to_float_seconds(ts: pd.Timestamp) -> float:
        return (ts - session_start).total_seconds()

    def _collect(contract_id: str) -> List[float]:
        rec = books.get(contract_id)
        if rec is None:
            return []
        return sorted(
            _to_float_seconds(t)
            for t in rec.get_submission_stream()
            if session_start <= t <= session_end
        )

    h_events = _collect(session.hourly_contract_id)

    q_events_raw: List[float] = []
    for q_id in session.quarter_contract_ids:
        q_events_raw.extend(_collect(q_id))
    q_events = sorted(q_events_raw)

    return h_events, q_events


# ---------------------------------------------------------------------------
# State-path construction
# ---------------------------------------------------------------------------

def build_state_path(
    session: Session,
    regime_series: pd.Series,  # timestamp-indexed integer regime (0/1)
) -> pd.DataFrame:
    """Build the predictable (t^-) state path for a session.

    Returns a DataFrame with columns:
        time_sec (float)  – seconds from session_start
        regime   (int)    – last observed regime strictly before each event
    """
    session_start = session.session_start
    time_secs = [
        (t - session_start).total_seconds() for t in regime_series.index
    ]
    df = pd.DataFrame(
        {
            "time_sec": time_secs,
            "regime": regime_series.values,
        }
    )
    return df.sort_values("time_sec").reset_index(drop=True)


def get_state_at_time(state_path: pd.DataFrame, t: float) -> int:
    """Return the predictable regime S(t^-) at *t* seconds from session_start.

    Uses the last observation whose time_sec is strictly less than *t*.
    Returns 0 (Normal) if no prior observation exists.
    """
    if state_path is None or state_path.empty:
        return 0

    ts_arr = state_path["time_sec"].values
    # bisect_left gives the index of the first element >= t;
    # one step left is the last element strictly < t.
    idx = bisect.bisect_left(ts_arr, t) - 1
    if idx < 0:
        return 0

    val = state_path["regime"].iloc[idx]
    if pd.isna(val):
        return 0
    return int(val)


# ---------------------------------------------------------------------------
# Session quality filtering
# ---------------------------------------------------------------------------

def filter_retained_sessions(
    sessions: List[Session],
    min_h_events: int = 5,
    min_q_events: int = 5,
) -> List[Session]:
    """Retain only sessions that satisfy all quality criteria.

    A session is retained when:
    - h_events and q_events are both populated
    - len(h_events) >= min_h_events and len(q_events) >= min_q_events
    - lsi_series is available
    - state_path contains both Normal (0) and Stressed (1) observations
    """
    retained: List[Session] = []

    for s in sessions:
        if s.h_events is None or s.q_events is None:
            s.exclusion_reason = "missing_event_streams"
            continue

        if len(s.h_events) < min_h_events:
            s.exclusion_reason = (
                f"insufficient_h_events: {len(s.h_events)} < {min_h_events}"
            )
            continue

        if len(s.q_events) < min_q_events:
            s.exclusion_reason = (
                f"insufficient_q_events: {len(s.q_events)} < {min_q_events}"
            )
            continue

        if s.lsi_series is None:
            s.exclusion_reason = "missing_lsi_series"
            continue

        if s.state_path is not None and not s.state_path.empty:
            unique_regimes = set(s.state_path["regime"].dropna().unique())
            if not ({0, 1} <= unique_regimes):
                s.exclusion_reason = "single_regime_only"
                continue

        s.is_retained = True
        retained.append(s)

    return retained


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_threshold_transportability(
    sessions: List[Session],
    global_threshold: float = 0.264,
) -> pd.DataFrame:
    """Compute threshold-transportability diagnostics for each session.

    For every session that has an lsi_series, computes:
    - session_q75_threshold : session-specific 75th percentile of LSI
    - stressed_fraction_global  : fraction of obs >= global_threshold
    - stressed_fraction_session : fraction of obs >= session_specific threshold

    Returns a DataFrame with one row per session.
    """
    rows = []
    for s in sessions:
        if s.lsi_series is None:
            continue
        lsi_col = "lsi" if "lsi" in s.lsi_series.columns else s.lsi_series.columns[0]
        lsi_vals = s.lsi_series[lsi_col].dropna()
        if lsi_vals.empty:
            continue

        session_q75 = float(np.percentile(lsi_vals.values, 75))
        stressed_global = float((lsi_vals >= global_threshold).mean())
        stressed_session = float((lsi_vals >= session_q75).mean())

        rows.append(
            {
                "session_id": s.session_id,
                "month": s.month,
                "delivery_hour": s.delivery_hour,
                "is_weekend": s.is_weekend,
                "session_q75_threshold": session_q75,
                "stressed_fraction_global": stressed_global,
                "stressed_fraction_session": stressed_session,
                "n_obs": int(len(lsi_vals)),
            }
        )

    return pd.DataFrame(rows)


def summarize_sessions_by_period(sessions: List[Session]) -> pd.DataFrame:
    """Summarise retained sessions by month, delivery hour, and weekday/weekend."""
    rows = []
    for s in sessions:
        if not s.is_retained:
            continue
        rows.append(
            {
                "session_id": s.session_id,
                "month": s.month,
                "delivery_hour": s.delivery_hour,
                "is_weekend": s.is_weekend,
                "stressed_fraction": s.stressed_fraction,
                "n_h_events": len(s.h_events) if s.h_events else 0,
                "n_q_events": len(s.q_events) if s.q_events else 0,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "month", "delivery_hour", "is_weekend",
                "n_sessions", "mean_stressed_fraction",
                "mean_n_h_events", "mean_n_q_events",
            ]
        )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["month", "delivery_hour", "is_weekend"], observed=True)
        .agg(
            n_sessions=("session_id", "count"),
            mean_stressed_fraction=("stressed_fraction", "mean"),
            mean_n_h_events=("n_h_events", "mean"),
            mean_n_q_events=("n_q_events", "mean"),
        )
        .reset_index()
    )
    return summary
