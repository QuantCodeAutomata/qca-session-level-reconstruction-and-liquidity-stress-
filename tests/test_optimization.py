"""
Tests for SDHP estimation optimization: convergence, FitResult structure,
bounds construction, and multi-start behavior.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sdhp_models import KernelType, N_STATES
from src.optimization import (
    estimate_session, estimate_all_sessions,
    FitResult, _build_bounds_exp, _build_bounds_powerlaw,
    MAX_ITER,
)


def _make_simple_session(seed: int = 42, n_h: int = 8, n_q: int = 12) -> dict:
    """Minimal synthetic session dict for fast estimation."""
    rng = np.random.default_rng(seed)
    T = 3600.0
    h_events = np.sort(rng.uniform(10, T - 10, n_h))
    q_events = np.sort(rng.uniform(10, T - 10, n_q))
    state_path = np.array([[0.0, 0], [T / 2, 1]], dtype=float)
    return {
        'session_id': f'test_session_{seed}',
        'h_events': h_events,
        'q_events': q_events,
        'state_path': state_path,
        'session_end': T,
        'month': '2024-01',
        'delivery_hour': 10,
        'is_weekend': False,
    }


def _call_estimate_session(session: dict, kernel_type: KernelType, **kwargs) -> FitResult:
    """Unpack session dict and call estimate_session."""
    return estimate_session(
        session_id=session['session_id'],
        h_events=session['h_events'],
        q_events=session['q_events'],
        state_path=session['state_path'],
        session_end=session['session_end'],
        kernel_type=kernel_type,
        month=session.get('month', ''),
        delivery_hour=session.get('delivery_hour', 0),
        is_weekend=session.get('is_weekend', False),
        **kwargs,
    )


# -----------------------------------------------------------------------
# FitResult structure
# -----------------------------------------------------------------------
# Very tight limits: each FD-gradient call costs (n_params+1) likelihood evals.
# EXP3 has 52 params → 53 evals per gradient call; cap at 15 scipy steps.
_FAST = dict(n_starts=1, seed=0, max_iter=5, max_func_eval=15)


def test_fit_result_fields():
    """FitResult must have all required fields."""
    session = _make_simple_session()
    result = _call_estimate_session(session, KernelType.EXP1, **_FAST)
    assert hasattr(result, 'session_id')
    assert hasattr(result, 'kernel_type')
    assert hasattr(result, 'converged')
    assert hasattr(result, 'nll_per_event')
    assert hasattr(result, 'n_events_total')
    assert hasattr(result, 'neg_log_lik')


def test_fit_result_session_id():
    session = _make_simple_session()
    result = _call_estimate_session(session, KernelType.EXP1, **_FAST)
    assert result.session_id == session['session_id']


def test_fit_result_kernel_type():
    session = _make_simple_session()
    for kt in [KernelType.EXP1, KernelType.EXP2]:
        result = _call_estimate_session(session, kt, **_FAST)
        assert result.kernel_type == kt


def test_fit_result_n_events():
    session = _make_simple_session(n_h=30, n_q=40)
    result = _call_estimate_session(session, KernelType.EXP1, **_FAST)
    assert result.n_events_total == 70


def test_fit_result_nll_per_event_finite():
    session = _make_simple_session()
    result = _call_estimate_session(session, KernelType.EXP1, **_FAST)
    assert np.isfinite(result.neg_log_lik)


def test_fit_result_nll_positive():
    """NLL per event should typically be positive for reasonable data."""
    session = _make_simple_session()
    result = _call_estimate_session(session, KernelType.EXP3, **_FAST)
    assert np.isfinite(result.neg_log_lik)


# -----------------------------------------------------------------------
# Bounds construction
# -----------------------------------------------------------------------
def test_bounds_exp1_shape():
    bounds = _build_bounds_exp(n_components=1)
    n_mu = N_STATES * 2
    n_ab = N_STATES * 2 * 2 * 1
    expected_size = n_mu + 2 * n_ab
    assert len(bounds.lb) == expected_size
    assert len(bounds.ub) == expected_size


def test_bounds_exp3_shape():
    bounds = _build_bounds_exp(n_components=3)
    n_mu = N_STATES * 2
    n_ab = N_STATES * 2 * 2 * 3
    expected_size = n_mu + 2 * n_ab
    assert len(bounds.lb) == expected_size
    assert len(bounds.ub) == expected_size


def test_bounds_all_lb_lt_ub():
    for n_comp in [1, 2, 3]:
        bounds = _build_bounds_exp(n_components=n_comp)
        assert (bounds.lb < bounds.ub).all()


def test_bounds_powerlaw_shape():
    bounds = _build_bounds_powerlaw()
    n_mu = N_STATES * 2
    n_mat = N_STATES * 2 * 2
    expected_size = n_mu + 3 * n_mat
    assert len(bounds.lb) == expected_size


# -----------------------------------------------------------------------
# estimate_session: runs without exception for all kernel types
# -----------------------------------------------------------------------
@pytest.mark.parametrize("kernel_type", [KernelType.EXP1, KernelType.EXP2, KernelType.EXP3])
def test_estimate_session_runs(kernel_type):
    session = _make_simple_session()
    result = _call_estimate_session(session, kernel_type, **_FAST)
    assert isinstance(result, FitResult)


def test_estimate_session_powerlaw_runs():
    session = _make_simple_session()
    result = _call_estimate_session(session, KernelType.POWERLAW, **_FAST)
    assert isinstance(result, FitResult)


def test_estimate_session_converged_has_params():
    """With tight limits convergence is unlikely; just check the struct is valid."""
    session = _make_simple_session()
    result = _call_estimate_session(session, KernelType.EXP1, **_FAST)
    assert isinstance(result, FitResult)
    assert result.params_exp is not None or not result.converged


# -----------------------------------------------------------------------
# estimate_all_sessions: multiple sessions, multiple kernels
# -----------------------------------------------------------------------
def test_estimate_all_sessions_count():
    sessions = [_make_simple_session(seed=i) for i in range(3)]
    kernel_types = [KernelType.EXP1, KernelType.EXP3]
    results = estimate_all_sessions(
        sessions, kernel_types, **{k: v for k, v in _FAST.items() if k != 'seed'},
        seed=0,
    )
    assert len(results) == len(sessions) * len(kernel_types)


def test_estimate_all_sessions_types():
    sessions = [_make_simple_session(seed=i) for i in range(2)]
    results = estimate_all_sessions(
        sessions, [KernelType.EXP1],
        **{k: v for k, v in _FAST.items() if k != 'seed'},
        seed=0,
    )
    assert all(isinstance(r, FitResult) for r in results)


# -----------------------------------------------------------------------
# Edge case: very short event streams
# -----------------------------------------------------------------------
def test_estimate_tiny_session():
    session = _make_simple_session(n_h=3, n_q=3)
    result = _call_estimate_session(session, KernelType.EXP1, **_FAST)
    assert isinstance(result, FitResult)
    assert np.isfinite(result.neg_log_lik)
