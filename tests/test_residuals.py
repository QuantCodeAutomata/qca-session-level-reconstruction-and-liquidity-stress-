"""
Tests for time-rescaled residuals and diagnostic statistics:
KS test against Exp(1), Ljung-Box, Wasserstein-1 distance, NLL per event.
"""
import sys
import os
import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sdhp_models import KernelType, ExpKernelParams
from src.residuals import (
    compute_compensator_exp, compute_time_rescaled_residuals,
    ks_test_exp1, ljung_box_test, wasserstein_distance_exp1,
    compute_session_residual_diagnostics,
)


def _make_params(n_comp: int = 1) -> ExpKernelParams:
    return ExpKernelParams(
        mu=np.ones((2, 2)) * 0.02,
        alpha=np.ones((2, 2, 2, n_comp)) * 0.01,
        beta=np.ones((2, 2, 2, n_comp)) * 5.0,
        n_components=n_comp,
        kernel_type={1: KernelType.EXP1, 2: KernelType.EXP2, 3: KernelType.EXP3}[n_comp],
    )


def _make_events(n: int = 50, T: float = 3600.0, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    h = np.sort(rng.uniform(1, T - 1, n))
    q = np.sort(rng.uniform(1, T - 1, n))
    sp = np.array([[0.0, 0]], dtype=float)
    return h, q, sp, T


def _build_all_events(h: np.ndarray, q: np.ndarray) -> np.ndarray:
    h_col = np.stack([h, np.zeros(len(h))], axis=1) if len(h) else np.empty((0, 2))
    q_col = np.stack([q, np.ones(len(q))], axis=1) if len(q) else np.empty((0, 2))
    all_ev = np.vstack([h_col, q_col]) if (len(h) + len(q)) > 0 else np.empty((0, 2))
    if len(all_ev) > 0:
        all_ev = all_ev[np.argsort(all_ev[:, 0], kind='stable')]
    return all_ev


# -----------------------------------------------------------------------
# Compensator
# -----------------------------------------------------------------------
def test_compensator_nonnegative():
    h, q, sp, T = _make_events()
    params = _make_params(1)
    all_ev = _build_all_events(h, q)
    comp = compute_compensator_exp(h, all_ev, 0, sp, T, params)
    assert (comp >= 0).all()


def test_compensator_monotone():
    """Compensator is monotonically non-decreasing (integrated intensity)."""
    h, q, sp, T = _make_events()
    params = _make_params(1)
    all_ev = _build_all_events(h, q)
    comp = compute_compensator_exp(h, all_ev, 0, sp, T, params)
    assert (np.diff(comp) >= -1e-10).all()


def test_compensator_length():
    h, q, sp, T = _make_events(n=30)
    params = _make_params(1)
    all_ev = _build_all_events(h, q)
    comp = compute_compensator_exp(h, all_ev, 0, sp, T, params)
    assert len(comp) == len(h)


# -----------------------------------------------------------------------
# Time-rescaled residuals
# -----------------------------------------------------------------------
def test_residuals_nonnegative():
    h, q, sp, T = _make_events()
    params = _make_params(1)
    all_ev = _build_all_events(h, q)
    deltas = compute_time_rescaled_residuals(h, all_ev, 0, sp, T, params, burn_in=0)
    assert (deltas >= 0).all()


def test_residuals_burn_in():
    """Burn-in removes leading observations from residuals."""
    h, q, sp, T = _make_events()
    params = _make_params(1)
    all_ev = _build_all_events(h, q)
    full = compute_time_rescaled_residuals(h, all_ev, 0, sp, T, params, burn_in=0)
    trimmed = compute_time_rescaled_residuals(h, all_ev, 0, sp, T, params, burn_in=5)
    assert len(trimmed) == max(0, len(full) - 5)


def test_residuals_empty_stream():
    """Empty event stream → empty residuals."""
    h = np.array([])
    q = np.array([10.0, 20.0, 30.0])
    sp = np.array([[0.0, 0]])
    T = 3600.0
    params = _make_params(1)
    all_ev = _build_all_events(h, q)
    deltas = compute_time_rescaled_residuals(h, all_ev, 0, sp, T, params, burn_in=0)
    assert deltas.size == 0


# -----------------------------------------------------------------------
# KS test
# -----------------------------------------------------------------------
def test_ks_test_passes_for_exp1():
    """KS test should pass (not reject Exp(1)) for actual Exp(1) samples."""
    rng = np.random.default_rng(0)
    sample = rng.exponential(scale=1.0, size=200)
    ks_stat, p_value, passed = ks_test_exp1(sample)
    assert isinstance(passed, bool)
    assert 0.0 <= p_value <= 1.0
    assert p_value > 0.01


def test_ks_test_rejects_non_exp():
    """KS test should reject for clearly non-Exp(1) distribution."""
    # Constant values — clearly not Exp(1)
    sample = np.ones(500) * 10.0
    _, p_value, passed = ks_test_exp1(sample)
    assert p_value < 0.05


def test_ks_test_empty():
    """Empty sample → result handled gracefully."""
    ks_stat, p_value, passed = ks_test_exp1(np.array([]))
    assert np.isnan(p_value)
    assert isinstance(passed, bool)


# -----------------------------------------------------------------------
# Ljung-Box test
# -----------------------------------------------------------------------
def test_lb_test_returns_tuple():
    rng = np.random.default_rng(42)
    sample = rng.exponential(scale=1.0, size=100)
    lb_stat, p_value, passed = ljung_box_test(sample, lags=10)
    assert isinstance(passed, bool)
    assert np.isfinite(lb_stat)


def test_lb_test_passes_for_iid():
    """Ljung-Box should mostly not reject i.i.d. exponential samples."""
    rng = np.random.default_rng(42)
    rejects = 0
    for _ in range(20):
        sample = rng.exponential(scale=1.0, size=300)
        _, p_value, passed = ljung_box_test(sample, lags=10)
        if p_value < 0.05:
            rejects += 1
    assert rejects <= 8, "Too many rejections for i.i.d. data"


# -----------------------------------------------------------------------
# Wasserstein-1 distance
# -----------------------------------------------------------------------
def test_w1_exp1_is_small():
    """Exp(1) sample should have small W1 distance to Exp(1)."""
    rng = np.random.default_rng(0)
    sample = rng.exponential(1.0, size=2000)
    w1 = wasserstein_distance_exp1(sample)
    assert w1 < 0.15


def test_w1_different_scale():
    """Exp(2) sample has W1 distance > 0 to Exp(1)."""
    rng = np.random.default_rng(0)
    sample = rng.exponential(2.0, size=2000)
    w1 = wasserstein_distance_exp1(sample)
    assert w1 > 0.05


def test_w1_nonnegative():
    rng = np.random.default_rng(42)
    for _ in range(5):
        scale = float(rng.uniform(0.5, 3.0))
        sample = rng.exponential(scale, size=200)
        w1 = wasserstein_distance_exp1(sample)
        assert w1 >= 0


# -----------------------------------------------------------------------
# Full session residual diagnostics
# -----------------------------------------------------------------------
def test_compute_session_residual_diagnostics_runs():
    h, q, sp, T = _make_events()
    params = _make_params(3)
    diag = compute_session_residual_diagnostics(h, q, sp, T, params, 'test_session')
    assert 'session_id' in diag


def test_compute_session_residual_diagnostics_has_metrics():
    h, q, sp, T = _make_events()
    params = _make_params(3)
    diag = compute_session_residual_diagnostics(h, q, sp, T, params, 'test_session')
    assert len(diag) > 1
    # Should contain KS and Ljung-Box fields
    ks_keys = [k for k in diag.keys() if 'ks' in k]
    assert len(ks_keys) > 0
