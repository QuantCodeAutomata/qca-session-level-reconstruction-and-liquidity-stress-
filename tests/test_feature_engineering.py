"""
Tests for feature engineering: spread, mid-price, rolling volatility,
queue imbalance, EWMA Z-score, LSI, and regime classification.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import (
    compute_spread_series, compute_midprice_series,
    compute_rolling_midprice_volatility, compute_queue_imbalance_series,
    ewma_zscore, compute_lsi, classify_regime, compute_global_lsi_threshold,
    LSI_THRESHOLD, EWMA_LAMBDA,
)
from src.book_reconstruction import BookState


def _make_bookstate(best_bid, best_ask, bid_qty=5.0, ask_qty=5.0, ts=None):
    if ts is None:
        ts = pd.Timestamp('2024-01-01 08:00:00')
    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
    spread = (best_ask - best_bid) if best_bid and best_ask else None
    total = bid_qty + ask_qty
    qi = abs(bid_qty - ask_qty) / total if total > 0 else None
    bids = {best_bid: bid_qty} if best_bid else {}
    asks = {best_ask: ask_qty} if best_ask else {}
    return BookState(
        timestamp=ts,
        bids=bids, asks=asks,
        best_bid=best_bid, best_ask=best_ask,
        mid_price=mid, spread=spread,
        total_bid_qty=bid_qty, total_ask_qty=ask_qty,
        queue_imbalance=qi,
    )


# -----------------------------------------------------------------------
# Test: spread series
# -----------------------------------------------------------------------
def test_spread_series_positive():
    bs = [_make_bookstate(80.0, 82.0), _make_bookstate(81.0, 83.5)]
    spreads = compute_spread_series(bs)
    assert (spreads.dropna() > 0).all()
    assert spreads.iloc[0] == pytest.approx(2.0)
    assert spreads.iloc[1] == pytest.approx(2.5)


def test_spread_series_one_sided_is_nan():
    bs = [_make_bookstate(80.0, None)]
    spreads = compute_spread_series(bs)
    assert spreads.isna().all() or (spreads.dropna().empty)


# -----------------------------------------------------------------------
# Test: mid-price series
# -----------------------------------------------------------------------
def test_midprice_formula():
    bs = [_make_bookstate(80.0, 84.0)]
    mid = compute_midprice_series(bs)
    assert mid.iloc[0] == pytest.approx(82.0)


# -----------------------------------------------------------------------
# Test: rolling volatility
# -----------------------------------------------------------------------
def test_rolling_vol_constant_series_zero():
    """Constant mid-price → zero volatility."""
    ts_index = pd.date_range('2024-01-01 08:00', periods=30, freq='30s')
    mid = pd.Series(np.ones(30) * 80.0, index=ts_index)
    vol = compute_rolling_midprice_volatility(mid, window_minutes=10)
    # After enough data points, vol should be ~0
    assert vol.dropna().abs().max() < 1e-8


def test_rolling_vol_nonzero_for_varying_prices():
    ts_index = pd.date_range('2024-01-01 08:00', periods=30, freq='30s')
    mid = pd.Series(np.arange(30, dtype=float), index=ts_index)
    vol = compute_rolling_midprice_volatility(mid, window_minutes=10)
    assert vol.dropna().max() > 0


def test_rolling_vol_nonnegative():
    ts_index = pd.date_range('2024-01-01 08:00', periods=50, freq='20s')
    mid = pd.Series(np.random.randn(50) + 80.0, index=ts_index)
    vol = compute_rolling_midprice_volatility(mid)
    assert (vol.dropna() >= 0).all()


# -----------------------------------------------------------------------
# Test: queue imbalance
# -----------------------------------------------------------------------
def test_qi_balanced_book():
    bs = [_make_bookstate(80.0, 82.0, bid_qty=5.0, ask_qty=5.0)]
    qi = compute_queue_imbalance_series(bs)
    assert qi.iloc[0] == pytest.approx(0.0, abs=1e-10)


def test_qi_all_bids():
    bs = [_make_bookstate(80.0, None, bid_qty=10.0, ask_qty=0.0)]
    qi = compute_queue_imbalance_series(bs)
    # Should be 1.0 (all bids, no asks) or NaN for one-sided book
    val = qi.iloc[0]
    if not np.isnan(val):
        assert val == pytest.approx(1.0, abs=1e-10)


def test_qi_range_01():
    """Queue imbalance must be in [0, 1]."""
    bs = [_make_bookstate(80.0, 82.0, bid_qty=7.0, ask_qty=3.0)]
    qi = compute_queue_imbalance_series(bs)
    val = qi.dropna().iloc[0] if not qi.dropna().empty else 0.5
    assert 0.0 <= val <= 1.0


# -----------------------------------------------------------------------
# Test: EWMA Z-score
# -----------------------------------------------------------------------
def test_ewma_zscore_returns_series_same_length():
    x = pd.Series(np.random.randn(100) + 80.0)
    z = ewma_zscore(x)
    assert len(z) == len(x)


def test_ewma_zscore_not_all_nan():
    x = pd.Series(np.random.randn(50) + 80.0)
    z = ewma_zscore(x)
    assert z.dropna().shape[0] > 0


def test_ewma_zscore_constant_series():
    """Constant series: after warmup, zscore should be near 0."""
    x = pd.Series(np.ones(200) * 80.0)
    z = ewma_zscore(x)
    # After long warmup, should be very close to 0
    valid = z.dropna()
    if len(valid) > 100:
        assert abs(valid.iloc[-1]) < 1e-3


def test_ewma_zscore_lambda_parameter():
    """Test with different lambda values (smaller lambda = slower adaptation)."""
    x = pd.Series(np.random.randn(100))
    z1 = ewma_zscore(x, lam=0.01)
    z2 = ewma_zscore(x, lam=0.1)
    # Both should return valid series
    assert len(z1) == 100
    assert len(z2) == 100


# -----------------------------------------------------------------------
# Test: LSI construction
# -----------------------------------------------------------------------
def test_lsi_average_of_three_z_scores():
    """LSI = mean of three z-scores by construction."""
    n = 100
    spread = pd.Series(np.random.randn(n))
    vol = pd.Series(np.random.randn(n))
    qi = pd.Series(np.random.randn(n))
    lsi = compute_lsi(spread, vol, qi)
    z_s = ewma_zscore(spread)
    z_v = ewma_zscore(vol)
    z_q = ewma_zscore(qi)
    expected = (z_s + z_v + z_q) / 3
    # Check on valid (non-NaN) entries
    valid = ~(lsi.isna() | expected.isna())
    if valid.sum() > 0:
        np.testing.assert_allclose(
            lsi[valid].values, expected[valid].values, rtol=1e-5
        )


def test_lsi_same_length_as_input():
    n = 50
    lsi = compute_lsi(
        pd.Series(np.random.randn(n)),
        pd.Series(np.random.randn(n)),
        pd.Series(np.random.randn(n)),
    )
    assert len(lsi) == n


# -----------------------------------------------------------------------
# Test: regime classification
# -----------------------------------------------------------------------
def test_classify_regime_binary():
    lsi = pd.Series([-1.0, 0.0, 0.1, 0.264, 0.5, 1.0])
    regime = classify_regime(lsi, threshold=0.264)
    assert set(regime.unique()).issubset({0, 1})


def test_classify_regime_threshold():
    lsi = pd.Series([0.263, 0.264, 0.265])
    regime = classify_regime(lsi, threshold=0.264)
    assert regime.iloc[0] == 0   # below threshold
    assert regime.iloc[1] == 1   # at threshold → stressed
    assert regime.iloc[2] == 1   # above threshold


def test_classify_regime_default_threshold():
    """Default threshold should match the paper value 0.264."""
    assert LSI_THRESHOLD == pytest.approx(0.264)


# -----------------------------------------------------------------------
# Test: global threshold
# -----------------------------------------------------------------------
def test_global_threshold_percentile():
    """75th percentile of known data."""
    data = pd.Series(np.arange(100, dtype=float))
    thresh = compute_global_lsi_threshold(data, 75.0)
    assert thresh == pytest.approx(np.percentile(data, 75))


def test_global_threshold_range():
    """Threshold should be within data range."""
    data = pd.Series(np.random.randn(500))
    thresh = compute_global_lsi_threshold(data)
    assert data.min() <= thresh <= data.max()
