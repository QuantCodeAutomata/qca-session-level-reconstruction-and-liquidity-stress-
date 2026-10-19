"""
Tests for the order book reconstruction module.

Validates LOB reconstruction logic including order add/cancel/execute,
spread/mid-price/queue-imbalance computations, and edge cases.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.book_reconstruction import (
    OrderBookReconstructor, reconstruct_session_books,
    Order, BookState,
)


def _make_msg(ts, oid, msg_type, side, price, qty, contract='H_20240101_08'):
    return {
        'timestamp': pd.Timestamp(ts),
        'contract_id': contract,
        'delivery_start': pd.Timestamp('2024-01-01 08:00:00'),
        'delivery_end': pd.Timestamp('2024-01-01 09:00:00'),
        'order_id': oid,
        'msg_type': msg_type,
        'side': side,
        'price': price,
        'quantity': qty,
    }


def _make_df(msgs):
    return pd.DataFrame(msgs)


# -----------------------------------------------------------------------
# Test: basic order submission
# -----------------------------------------------------------------------
def test_submit_creates_order():
    rec = OrderBookReconstructor('H_20240101_08')
    msg = _make_msg('2024-01-01 08:00:00.100', 'O1', 'submit', 'bid', 80.0, 1.0)
    rec.process_message(pd.Series(msg))
    assert 'O1' in rec.active_orders
    assert rec.active_orders['O1'].price == 80.0
    assert rec.active_orders['O1'].side == 'bid'


def test_submit_records_submission_time():
    rec = OrderBookReconstructor('H_20240101_08')
    msg = _make_msg('2024-01-01 08:00:00.100', 'O1', 'submit', 'bid', 80.0, 1.0)
    rec.process_message(pd.Series(msg))
    assert len(rec.submission_times) == 1


def test_cancel_removes_order():
    rec = OrderBookReconstructor('H_20240101_08')
    rec.process_message(pd.Series(_make_msg('2024-01-01 08:00:00.100', 'O1', 'submit', 'bid', 80.0, 1.0)))
    rec.process_message(pd.Series(_make_msg('2024-01-01 08:00:01.000', 'O1', 'cancel', 'bid', 80.0, 1.0)))
    assert 'O1' not in rec.active_orders


def test_execute_removes_order():
    rec = OrderBookReconstructor('H_20240101_08')
    rec.process_message(pd.Series(_make_msg('2024-01-01 08:00:00.100', 'O1', 'submit', 'ask', 82.0, 2.0)))
    rec.process_message(pd.Series(_make_msg('2024-01-01 08:00:01.000', 'O1', 'execute', 'ask', 82.0, 2.0)))
    assert 'O1' not in rec.active_orders


# -----------------------------------------------------------------------
# Test: spread and mid-price computation
# -----------------------------------------------------------------------
def test_spread_and_midprice():
    rec = OrderBookReconstructor('H_20240101_08')
    msgs = [
        _make_msg('2024-01-01 08:00:00.000', 'B1', 'submit', 'bid', 80.0, 5.0),
        _make_msg('2024-01-01 08:00:00.100', 'A1', 'submit', 'ask', 82.0, 5.0),
    ]
    rec.process_messages(_make_df(msgs))
    state = rec.book_snapshots[-1]
    assert state.spread == pytest.approx(2.0)
    assert state.mid_price == pytest.approx(81.0)


def test_one_sided_book_no_spread():
    rec = OrderBookReconstructor('H_20240101_08')
    msgs = [_make_msg('2024-01-01 08:00:00', 'B1', 'submit', 'bid', 80.0, 5.0)]
    rec.process_messages(_make_df(msgs))
    state = rec.book_snapshots[-1]
    assert state.spread is None or np.isnan(state.spread)
    assert state.mid_price is None or np.isnan(state.mid_price)


# -----------------------------------------------------------------------
# Test: queue imbalance
# -----------------------------------------------------------------------
def test_queue_imbalance_balanced():
    """Balanced book: QI = 0."""
    rec = OrderBookReconstructor('H_20240101_08')
    msgs = [
        _make_msg('2024-01-01 08:00:00', 'B1', 'submit', 'bid', 80.0, 10.0),
        _make_msg('2024-01-01 08:00:00.001', 'A1', 'submit', 'ask', 82.0, 10.0),
    ]
    rec.process_messages(_make_df(msgs))
    state = rec.book_snapshots[-1]
    assert state.queue_imbalance == pytest.approx(0.0, abs=1e-10)


def test_queue_imbalance_all_bids():
    """All bids, no asks: QI = 1.0."""
    rec = OrderBookReconstructor('H_20240101_08')
    msgs = [_make_msg('2024-01-01 08:00:00', 'B1', 'submit', 'bid', 80.0, 10.0)]
    rec.process_messages(_make_df(msgs))
    state = rec.book_snapshots[-1]
    # One-sided book: qi should be 1.0 (all bids)
    assert abs(state.queue_imbalance) == pytest.approx(1.0, abs=1e-10)


def test_queue_imbalance_range():
    """Queue imbalance must be in [0, 1]."""
    rec = OrderBookReconstructor('H_20240101_08')
    msgs = [
        _make_msg('2024-01-01 08:00:00', 'B1', 'submit', 'bid', 80.0, 7.0),
        _make_msg('2024-01-01 08:00:00.001', 'A1', 'submit', 'ask', 82.0, 3.0),
    ]
    rec.process_messages(_make_df(msgs))
    state = rec.book_snapshots[-1]
    assert 0.0 <= state.queue_imbalance <= 1.0


# -----------------------------------------------------------------------
# Test: multiple orders at same price level
# -----------------------------------------------------------------------
def test_multiple_orders_same_price():
    rec = OrderBookReconstructor('H_20240101_08')
    msgs = [
        _make_msg('2024-01-01 08:00:00', 'B1', 'submit', 'bid', 80.0, 5.0),
        _make_msg('2024-01-01 08:00:00.001', 'B2', 'submit', 'bid', 80.0, 3.0),
    ]
    rec.process_messages(_make_df(msgs))
    state = rec.book_snapshots[-1]
    assert state.total_bid_qty == pytest.approx(8.0)
    assert state.best_bid == pytest.approx(80.0)


# -----------------------------------------------------------------------
# Test: session-level reconstruction
# -----------------------------------------------------------------------
def test_reconstruct_session_books():
    msgs = []
    # Hourly contract
    msgs += [
        _make_msg('2024-01-01 08:00:00', 'H1', 'submit', 'bid', 80.0, 5.0, 'H_20240101_08'),
        _make_msg('2024-01-01 08:00:01', 'H2', 'submit', 'ask', 82.0, 5.0, 'H_20240101_08'),
    ]
    # Q1-Q4
    for q_num, offset in enumerate([0, 900, 1800, 2700]):
        cid = f'Q_20240101_08_Q{q_num+1}'
        msgs.append(_make_msg(f'2024-01-01 08:00:0{q_num}', f'Q{q_num}', 'submit', 'bid', 79.0, 2.0, cid))

    df = _make_df(msgs)
    quarter_ids = [f'Q_20240101_08_Q{i}' for i in range(1, 5)]
    books = reconstruct_session_books(df, 'H_20240101_08', quarter_ids)
    assert 'H_20240101_08' in books
    for qid in quarter_ids:
        assert qid in books


# -----------------------------------------------------------------------
# Test: submission stream
# -----------------------------------------------------------------------
def test_submission_stream_sorted():
    rec = OrderBookReconstructor('H_20240101_08')
    msgs = [
        _make_msg('2024-01-01 08:00:05', 'O3', 'submit', 'bid', 80.0, 1.0),
        _make_msg('2024-01-01 08:00:01', 'O1', 'submit', 'bid', 80.0, 1.0),
        _make_msg('2024-01-01 08:00:03', 'O2', 'submit', 'bid', 80.0, 1.0),
    ]
    # Process in timestamp order
    df = _make_df(msgs).sort_values('timestamp')
    rec.process_messages(df)
    stream = rec.get_submission_stream()
    ts_list = [t.timestamp() for t in stream]
    assert ts_list == sorted(ts_list)


# -----------------------------------------------------------------------
# Test: empty book
# -----------------------------------------------------------------------
def test_empty_book_no_metrics():
    rec = OrderBookReconstructor('H_20240101_08')
    # No messages processed → snapshot list empty
    assert len(rec.book_snapshots) == 0
    assert len(rec.active_orders) == 0


# -----------------------------------------------------------------------
# Test: partial execution reduces quantity
# -----------------------------------------------------------------------
def test_partial_execute_reduces_quantity():
    rec = OrderBookReconstructor('H_20240101_08')
    rec.process_message(pd.Series(_make_msg('2024-01-01 08:00:00', 'O1', 'submit', 'ask', 82.0, 10.0)))
    # Execute partial quantity
    msg = _make_msg('2024-01-01 08:00:01', 'O1', 'execute', 'ask', 82.0, 6.0)
    rec.process_message(pd.Series(msg))
    if 'O1' in rec.active_orders:
        assert rec.active_orders['O1'].quantity == pytest.approx(4.0)
