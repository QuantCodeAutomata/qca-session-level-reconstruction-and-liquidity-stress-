import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.book_reconstruction import BookState, OrderBookReconstructor

EWMA_LAMBDA = 0.01               # slow baseline decay
ROLLING_VOL_WINDOW_MINUTES = 10  # trailing window for mid-price volatility
LSI_THRESHOLD = 0.264            # global 75th-percentile threshold (paper value)


# ---------------------------------------------------------------------------
# Raw series extractors
# ---------------------------------------------------------------------------

def compute_spread_series(book_snapshots: List[BookState]) -> pd.Series:
    """Extract spread time series from book snapshots."""
    timestamps = [s.timestamp for s in book_snapshots]
    values = [s.spread if s.spread is not None else np.nan for s in book_snapshots]
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="spread")


def compute_midprice_series(book_snapshots: List[BookState]) -> pd.Series:
    """Extract mid-price time series from book snapshots."""
    timestamps = [s.timestamp for s in book_snapshots]
    values = [s.mid_price if s.mid_price is not None else np.nan for s in book_snapshots]
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="mid_price")


def compute_queue_imbalance_series(book_snapshots: List[BookState]) -> pd.Series:
    """Extract absolute queue imbalance from book snapshots.

    QI_abs(t) = |(sum_bid_qty - sum_ask_qty) / (sum_bid_qty + sum_ask_qty)|
    across the full active book.
    """
    timestamps = [s.timestamp for s in book_snapshots]
    values = [
        s.queue_imbalance if s.queue_imbalance is not None else np.nan
        for s in book_snapshots
    ]
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="queue_imbalance")


# ---------------------------------------------------------------------------
# Rolling mid-price volatility
# ---------------------------------------------------------------------------

def compute_rolling_midprice_volatility(
    midprice_series: pd.Series,
    window_minutes: float = ROLLING_VOL_WINDOW_MINUTES,
) -> pd.Series:
    """Compute 10-minute trailing mid-price volatility (population RMS deviation).

    sigma_mid(t) = sqrt( mean( (P_mid - mean(P_mid))^2 ) )
    over the trailing window (t - window, t], requiring at least 2 observations.
    """
    window_str = f"{int(window_minutes)}min"

    def _rms_deviation(x: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        mu = np.nanmean(x)
        return float(np.sqrt(np.nanmean((x - mu) ** 2)))

    return midprice_series.rolling(window_str, min_periods=2).apply(
        _rms_deviation, raw=True
    ).rename("volatility")


# ---------------------------------------------------------------------------
# EWMA Z-score normalisation
# ---------------------------------------------------------------------------

def ewma_zscore(series: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """Compute slow EWMA Z-score normalisation.

    Recursive update (applied to non-NaN observations only):
        mu_t  = (1 - lam) * mu_{t-1}  + lam * X_t
        var_t = (1 - lam) * var_{t-1} + lam * (X_t - mu_t)^2
        z_t   = (X_t - mu_t) / sqrt(var_t)

    Initialisation: mu_0 = X_0, var_0 = 1.0.
    Returns NaN while var is effectively zero (< 1e-12).
    """
    arr = series.values.astype(float)
    n = len(arr)
    z = np.full(n, np.nan)

    # Locate first valid observation
    first = next((i for i, v in enumerate(arr) if not np.isnan(v)), None)
    if first is None:
        return pd.Series(z, index=series.index, name=series.name)

    mu = arr[first]
    var = 1.0

    for i in range(first, n):
        x = arr[i]
        if np.isnan(x):
            continue
        mu = (1.0 - lam) * mu + lam * x
        var = (1.0 - lam) * var + lam * (x - mu) ** 2
        if var > 1e-12:
            z[i] = (x - mu) / np.sqrt(var)

    return pd.Series(z, index=series.index, name=series.name)


# ---------------------------------------------------------------------------
# LSI construction and regime classification
# ---------------------------------------------------------------------------

def compute_lsi(
    spread_series: pd.Series,
    volatility_series: pd.Series,
    qi_series: pd.Series,
    lam: float = EWMA_LAMBDA,
) -> pd.Series:
    """Compute the Liquidity Stress Indicator.

    LSI(t) = [ z_spread^slow(t) + z_volatility^slow(t) + z_QI_abs^slow(t) ] / 3
    """
    z_spread = ewma_zscore(spread_series, lam)
    z_vol = ewma_zscore(volatility_series, lam)
    z_qi = ewma_zscore(qi_series, lam)

    lsi = (z_spread + z_vol + z_qi) / 3.0
    lsi.name = "lsi"
    return lsi


def classify_regime(
    lsi_series: pd.Series,
    threshold: float = LSI_THRESHOLD,
) -> pd.Series:
    """Map LSI values to regime labels: Normal = 0, Stressed = 1.

    S(t) = 0 if LSI(t) < threshold
    S(t) = 1 if LSI(t) >= threshold
    NaN LSI -> NaN regime.
    """
    regime = pd.Series(np.nan, index=lsi_series.index, name="regime")
    valid = lsi_series.notna()
    regime.loc[valid] = (lsi_series.loc[valid] >= threshold).astype(int)
    return regime


def compute_global_lsi_threshold(
    all_lsi_values: pd.Series,
    percentile: float = 75.0,
) -> float:
    """Return the *percentile*-th quantile of LSI across all sessions."""
    clean = all_lsi_values.dropna()
    if clean.empty:
        return LSI_THRESHOLD
    return float(np.percentile(clean.values, percentile))


# ---------------------------------------------------------------------------
# Session-level feature table
# ---------------------------------------------------------------------------

def compute_session_features(
    books: Dict[str, OrderBookReconstructor],
    hourly_id: str,
    quarter_ids: list,
    session_start: pd.Timestamp,
    session_end: pd.Timestamp,
    obs_freq_seconds: float = 1.0,
) -> pd.DataFrame:
    """Compute all session features on a regular observation grid.

    For each grid time t, book states from all 5 contracts are queried and
    their spread, mid-price, and queue-imbalance are averaged (ignoring NaN).
    Rolling volatility, EWMA Z-scores, LSI, and regime are then computed on
    the aggregated series.

    Returns a DataFrame with columns:
        timestamp, spread, mid_price, volatility, queue_imbalance, lsi, regime
    """
    all_contract_ids = [hourly_id] + list(quarter_ids)
    n_contracts = len(all_contract_ids)

    # Build the regular observation grid
    total_seconds = (session_end - session_start).total_seconds()
    n_obs = max(1, int(total_seconds / obs_freq_seconds) + 1)
    obs_times = [
        session_start + pd.Timedelta(seconds=i * obs_freq_seconds)
        for i in range(n_obs)
    ]

    # Collect per-contract metrics at each grid point
    spreads = np.full((n_obs, n_contracts), np.nan)
    mid_prices = np.full((n_obs, n_contracts), np.nan)
    queue_imbalances = np.full((n_obs, n_contracts), np.nan)

    for j, cid in enumerate(all_contract_ids):
        rec = books.get(cid)
        if rec is None or not rec.book_snapshots:
            continue
        for i, t in enumerate(obs_times):
            state = rec.get_book_state(t)
            if state is None:
                continue
            if state.spread is not None:
                spreads[i, j] = state.spread
            if state.mid_price is not None:
                mid_prices[i, j] = state.mid_price
            if state.queue_imbalance is not None:
                queue_imbalances[i, j] = state.queue_imbalance

    # Cross-contract averages; NaN where all contracts have no data.
    # np.nanmean emits a RuntimeWarning for all-NaN slices — suppress it
    # because returning NaN in that case is the correct behaviour.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spread_mean = np.nanmean(spreads, axis=1)
        midprice_mean = np.nanmean(mid_prices, axis=1)
        qi_mean = np.nanmean(queue_imbalances, axis=1)

    all_nan_spread = np.all(np.isnan(spreads), axis=1)
    all_nan_mid = np.all(np.isnan(mid_prices), axis=1)
    all_nan_qi = np.all(np.isnan(queue_imbalances), axis=1)

    spread_agg = np.where(all_nan_spread, np.nan, spread_mean)
    midprice_agg = np.where(all_nan_mid, np.nan, midprice_mean)
    qi_agg = np.where(all_nan_qi, np.nan, qi_mean)

    dt_index = pd.DatetimeIndex(obs_times)
    spread_series = pd.Series(spread_agg, index=dt_index, name="spread")
    midprice_series = pd.Series(midprice_agg, index=dt_index, name="mid_price")
    qi_series = pd.Series(qi_agg, index=dt_index, name="queue_imbalance")

    vol_series = compute_rolling_midprice_volatility(
        midprice_series, window_minutes=ROLLING_VOL_WINDOW_MINUTES
    )

    lsi = compute_lsi(spread_series, vol_series, qi_series)
    regime = classify_regime(lsi)

    return pd.DataFrame(
        {
            "timestamp": obs_times,
            "spread": spread_agg,
            "mid_price": midprice_agg,
            "volatility": vol_series.values,
            "queue_imbalance": qi_agg,
            "lsi": lsi.values,
            "regime": regime.values,
        }
    )
