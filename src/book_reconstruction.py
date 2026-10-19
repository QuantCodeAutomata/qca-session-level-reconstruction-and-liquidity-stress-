import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Order:
    order_id: str
    side: str        # 'bid' or 'ask'
    price: float
    quantity: float
    timestamp: pd.Timestamp


@dataclass
class BookState:
    """Snapshot of the active order book at a point in time."""
    timestamp: pd.Timestamp
    bids: Dict[float, float]          # price -> total quantity
    asks: Dict[float, float]          # price -> total quantity
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid_price: Optional[float]
    spread: Optional[float]
    total_bid_qty: float
    total_ask_qty: float
    queue_imbalance: Optional[float]  # |(bid_qty - ask_qty) / (bid_qty + ask_qty)|


class OrderBookReconstructor:
    """Reconstructs the active LOB from a Level-3 message stream.

    Processes messages in arrival order: 'submit' adds an order,
    'cancel' removes it, 'execute' reduces or removes it.
    All submission timestamps are stored for event-stream construction.
    """

    def __init__(self, contract_id: str):
        self.contract_id = contract_id
        self.active_orders: Dict[str, Order] = {}
        self.submission_times: List[pd.Timestamp] = []
        self.book_snapshots: List[BookState] = []
        # Parallel list of snapshot timestamps for efficient binary search
        self._snapshot_ts: List[pd.Timestamp] = []

    # ------------------------------------------------------------------
    # Public processing interface
    # ------------------------------------------------------------------

    def process_message(self, msg: pd.Series) -> Optional[BookState]:
        """Process one L3 message and return the updated book state."""
        msg_type = str(msg["msg_type"])
        order_id = str(msg["order_id"])
        timestamp = pd.Timestamp(msg["timestamp"])

        if msg_type == "submit":
            order = Order(
                order_id=order_id,
                side=str(msg["side"]),
                price=float(msg["price"]),
                quantity=float(msg["quantity"]),
                timestamp=timestamp,
            )
            self.active_orders[order_id] = order
            self.submission_times.append(timestamp)

        elif msg_type == "cancel":
            self.active_orders.pop(order_id, None)

        elif msg_type == "execute":
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                new_qty = order.quantity - float(msg["quantity"])
                if new_qty <= 1e-9:
                    del self.active_orders[order_id]
                else:
                    order.quantity = new_qty

        state = self._compute_book_metrics(timestamp)
        self.book_snapshots.append(state)
        self._snapshot_ts.append(timestamp)
        return state

    def process_messages(self, messages: pd.DataFrame) -> List[BookState]:
        """Process all L3 messages for this contract (must be time-sorted)."""
        snapshots: List[BookState] = []
        for _, msg in messages.iterrows():
            snapshots.append(self.process_message(msg))
        return snapshots

    # ------------------------------------------------------------------
    # Point-in-time query
    # ------------------------------------------------------------------

    def get_book_state(self, timestamp: pd.Timestamp) -> Optional[BookState]:
        """Return the book state at *timestamp* using t^- information.

        Returns the last snapshot whose timestamp is <= *timestamp*, or
        None if no message has been processed before that time.
        """
        if not self._snapshot_ts:
            return None

        # bisect_right gives the insertion point after all equal timestamps,
        # so index - 1 is the last snapshot at or before *timestamp*.
        idx = bisect.bisect_right(self._snapshot_ts, timestamp) - 1
        if idx < 0:
            return None
        return self.book_snapshots[idx]

    # ------------------------------------------------------------------
    # Event-stream accessor
    # ------------------------------------------------------------------

    def get_submission_stream(self) -> List[pd.Timestamp]:
        """Return a sorted list of all submission timestamps."""
        return sorted(self.submission_times)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_book_metrics(self, timestamp: pd.Timestamp) -> BookState:
        """Build a BookState from the current active-order set."""
        bid_levels: Dict[float, float] = defaultdict(float)
        ask_levels: Dict[float, float] = defaultdict(float)

        for order in self.active_orders.values():
            if order.side == "bid":
                bid_levels[order.price] += order.quantity
            else:
                ask_levels[order.price] += order.quantity

        bids = dict(bid_levels)
        asks = dict(ask_levels)

        total_bid_qty = sum(bids.values())
        total_ask_qty = sum(asks.values())

        best_bid: Optional[float] = max(bids) if bids else None
        best_ask: Optional[float] = min(asks) if asks else None

        if best_bid is not None and best_ask is not None:
            mid_price: Optional[float] = (best_bid + best_ask) / 2.0
            spread: Optional[float] = best_ask - best_bid
        else:
            mid_price = None
            spread = None

        # Queue imbalance: |bid_qty - ask_qty| / (bid_qty + ask_qty)
        # When book is one-sided total == one side, so result is always 1.0.
        total = total_bid_qty + total_ask_qty
        if total > 1e-12:
            queue_imbalance: Optional[float] = (
                abs(total_bid_qty - total_ask_qty) / total
            )
        elif total_bid_qty > 0 or total_ask_qty > 0:
            queue_imbalance = 1.0   # degenerate one-sided book
        else:
            queue_imbalance = None  # empty book

        return BookState(
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            total_bid_qty=total_bid_qty,
            total_ask_qty=total_ask_qty,
            queue_imbalance=queue_imbalance,
        )


# ---------------------------------------------------------------------------
# Session-level reconstruction
# ---------------------------------------------------------------------------

def reconstruct_session_books(
    messages: pd.DataFrame,
    hourly_contract_id: str,
    quarter_contract_ids: List[str],
) -> Dict[str, OrderBookReconstructor]:
    """Reconstruct LOBs for all 5 contracts in a session.

    Messages are time-sorted before processing so that each reconstructor
    sees events in strict chronological order.

    Returns a dict mapping contract_id -> OrderBookReconstructor.
    """
    all_ids = [hourly_contract_id] + list(quarter_contract_ids)
    books: Dict[str, OrderBookReconstructor] = {}

    for contract_id in all_ids:
        mask = messages["contract_id"] == contract_id
        contract_msgs = messages.loc[mask].sort_values("timestamp")
        reconstructor = OrderBookReconstructor(contract_id)
        if not contract_msgs.empty:
            reconstructor.process_messages(contract_msgs)
        books[contract_id] = reconstructor

    return books
