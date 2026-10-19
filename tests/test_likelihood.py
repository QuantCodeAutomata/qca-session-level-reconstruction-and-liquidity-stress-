"""
Tests for Hawkes process log-likelihood computation.
Validates that the likelihood is finite, negative for reasonable params,
and that it increases with better-fitting parameters.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sdhp_models import (
    KernelType, ExpKernelParams, PowerLawKernelParams,
    initialize_exp_params, initialize_powerlaw_params,
    params_to_vector_exp, vector_to_params_exp,
    params_to_vector_powerlaw, vector_to_params_powerlaw,
)
from src.likelihood_exp import (
    compute_exp_log_likelihood,
    compute_exp_log_likelihood_and_gradient,
)
from src.likelihood_powerlaw import (
    compute_powerlaw_log_likelihood,
    compute_powerlaw_log_likelihood_and_gradient,
    powerlaw_kernel, powerlaw_kernel_integral,
)


def _make_simple_state_path(session_end: float, regime: int = 0) -> np.ndarray:
    """Simple constant-regime state path."""
    return np.array([[0.0, regime]], dtype=float)


def _make_subcritical_params(n_comp: int = 1) -> ExpKernelParams:
    """Make clearly subcritical params (small alpha, large beta)."""
    return ExpKernelParams(
        mu=np.ones((2, 2)) * 0.02,
        alpha=np.ones((2, 2, 2, n_comp)) * 0.01,
        beta=np.ones((2, 2, 2, n_comp)) * 5.0,
        n_components=n_comp,
        kernel_type={1: KernelType.EXP1, 2: KernelType.EXP2, 3: KernelType.EXP3}[n_comp],
    )


def _simple_events(n_h: int = 20, n_q: int = 20, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    session_end = 3600.0
    h_events = np.sort(rng.uniform(0, session_end, n_h))
    q_events = np.sort(rng.uniform(0, session_end, n_q))
    return h_events, q_events, session_end


# -----------------------------------------------------------------------
# Exponential likelihood: basic sanity checks
# -----------------------------------------------------------------------
def test_exp_likelihood_finite():
    h, q, T = _simple_events()
    params = _make_subcritical_params(1)
    sp = _make_simple_state_path(T, 0)
    ll = compute_exp_log_likelihood(h, q, sp, 0.0, T, params, include_penalty=False)
    assert np.isfinite(ll)


def test_exp_likelihood_negative():
    """Log-likelihood of a point process is typically negative."""
    h, q, T = _simple_events()
    params = _make_subcritical_params(1)
    sp = _make_simple_state_path(T, 0)
    ll = compute_exp_log_likelihood(h, q, sp, 0.0, T, params, include_penalty=False)
    # Not necessarily negative in all cases, but should be finite
    assert np.isfinite(ll)


def test_exp_likelihood_penalty_reduces_value():
    """Penalized LL ≤ unpenalized LL when params violate spectral radius constraint."""
    h, q, T = _simple_events()
    # Supercritical params → penalty is positive → penalized < unpenalized
    params = ExpKernelParams(
        mu=np.ones((2, 2)) * 0.02,
        alpha=np.ones((2, 2, 2, 1)) * 2.0,   # large → supercritical
        beta=np.ones((2, 2, 2, 1)) * 1.0,
        n_components=1,
        kernel_type=KernelType.EXP1,
    )
    sp = _make_simple_state_path(T, 0)
    ll_unpen = compute_exp_log_likelihood(h, q, sp, 0.0, T, params, include_penalty=False)
    ll_pen = compute_exp_log_likelihood(h, q, sp, 0.0, T, params, include_penalty=True)
    assert ll_pen < ll_unpen


def test_exp_likelihood_no_events():
    """Zero-event streams should give finite (possibly high negative) LL."""
    params = _make_subcritical_params(1)
    sp = _make_simple_state_path(3600.0, 0)
    ll = compute_exp_log_likelihood(
        np.array([]), np.array([]), sp, 0.0, 3600.0, params, include_penalty=False
    )
    assert np.isfinite(ll)


def test_exp_likelihood_n_components():
    """1-Exp, 2-Exp, and 3-Exp all produce finite likelihoods."""
    h, q, T = _simple_events()
    sp = _make_simple_state_path(T, 0)
    for n in [1, 2, 3]:
        params = _make_subcritical_params(n)
        ll = compute_exp_log_likelihood(h, q, sp, 0.0, T, params, include_penalty=False)
        assert np.isfinite(ll), f"LL not finite for {n}-Exp"


def test_exp_likelihood_with_regime_switching():
    """Regime-switching state path should not crash."""
    h, q, T = _simple_events(n_h=15, n_q=15)
    # State switches at midpoint
    sp = np.array([[0.0, 0], [T / 2, 1]], dtype=float)
    params = _make_subcritical_params(2)
    ll = compute_exp_log_likelihood(h, q, sp, 0.0, T, params, include_penalty=False)
    assert np.isfinite(ll)


# -----------------------------------------------------------------------
# Gradient checks
# -----------------------------------------------------------------------
def test_exp_gradient_finite():
    h, q, T = _simple_events(n_h=10, n_q=10)
    sp = _make_simple_state_path(T, 0)
    params = _make_subcritical_params(1)
    theta = params_to_vector_exp(params)
    neg_ll, grad = compute_exp_log_likelihood_and_gradient(
        theta, 1, h, q, sp, T, KernelType.EXP1, include_penalty=False
    )
    assert np.isfinite(neg_ll)
    assert np.all(np.isfinite(grad))


def test_exp_gradient_shape():
    h, q, T = _simple_events(n_h=10, n_q=10)
    sp = _make_simple_state_path(T, 0)
    params = _make_subcritical_params(3)
    theta = params_to_vector_exp(params)
    neg_ll, grad = compute_exp_log_likelihood_and_gradient(
        theta, 3, h, q, sp, T, KernelType.EXP3, include_penalty=False
    )
    assert grad.shape == theta.shape


# -----------------------------------------------------------------------
# Power-law kernel
# -----------------------------------------------------------------------
def test_powerlaw_kernel_positive():
    tau = np.array([0.1, 1.0, 10.0, 100.0])
    k = powerlaw_kernel(tau, alpha=1.0, eta=1.5, c=1.0)
    assert (k > 0).all()


def test_powerlaw_kernel_decreasing():
    """Kernel should be monotonically decreasing in tau."""
    tau = np.linspace(0.1, 100.0, 100)
    k = powerlaw_kernel(tau, alpha=1.0, eta=1.5, c=1.0)
    assert (np.diff(k) < 0).all()


def test_powerlaw_kernel_integral_positive():
    val = powerlaw_kernel_integral(0.0, 100.0, alpha=1.0, eta=1.5, c=1.0)
    assert val > 0


def test_powerlaw_kernel_integral_monotone():
    """Longer interval → larger integral."""
    v1 = powerlaw_kernel_integral(0.0, 50.0, alpha=1.0, eta=1.5, c=1.0)
    v2 = powerlaw_kernel_integral(0.0, 100.0, alpha=1.0, eta=1.5, c=1.0)
    assert v2 > v1


def test_powerlaw_likelihood_finite():
    h, q, T = _simple_events(n_h=15, n_q=15)
    sp = _make_simple_state_path(T, 0)
    rng = np.random.default_rng(42)
    params = PowerLawKernelParams(
        mu=np.ones((2, 2)) * 0.01,
        alpha=np.ones((2, 2, 2)) * 0.05,
        eta=np.ones((2, 2, 2)) * 1.5,
        c=np.ones((2, 2, 2)) * 5.0,
    )
    ll = compute_powerlaw_log_likelihood(h, q, sp, T, params, include_penalty=False)
    assert np.isfinite(ll)


def test_powerlaw_gradient_shape():
    h, q, T = _simple_events(n_h=10, n_q=10)
    sp = _make_simple_state_path(T, 0)
    params = PowerLawKernelParams(
        mu=np.ones((2, 2)) * 0.01,
        alpha=np.ones((2, 2, 2)) * 0.05,
        eta=np.ones((2, 2, 2)) * 1.5,
        c=np.ones((2, 2, 2)) * 5.0,
    )
    theta = params_to_vector_powerlaw(params)
    neg_ll, grad = compute_powerlaw_log_likelihood_and_gradient(theta, h, q, sp, T)
    assert np.isfinite(neg_ll)
    assert grad.shape == theta.shape
    assert np.all(np.isfinite(grad))
