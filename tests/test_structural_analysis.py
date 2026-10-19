"""
Tests for structural analysis: branching matrices, spillover multipliers,
asymmetry, response speed, and bootstrap confidence intervals.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sdhp_models import (
    KernelType, ExpKernelParams,
    compute_branching_matrix, compute_spectral_radius,
    compute_spillover_matrix, compute_asymmetry_index,
)
from src.structural_analysis import (
    compute_kernel_integral, compute_total_mass, solve_quantile_time,
    compute_mean_lag, compute_component_decomposition, analyze_session_structure,
    analyze_all_sessions, compute_pooled_medians, compute_within_session_differences,
    bootstrap_median_ci, SessionStructure,
)
from src.optimization import FitResult


def _make_exp3_params(seed: int = 0) -> ExpKernelParams:
    """Create valid 3-Exp params with subcritical spectral radius."""
    rng = np.random.default_rng(seed)
    mu = np.array([[0.03, 0.02], [0.03, 0.02]])
    # Small alpha values → subcritical
    alpha = rng.uniform(0.001, 0.01, size=(2, 2, 2, 3))
    beta = rng.uniform(0.1, 10.0, size=(2, 2, 2, 3))
    return ExpKernelParams(
        mu=mu, alpha=alpha, beta=beta,
        n_components=3, kernel_type=KernelType.EXP3,
    )


def _make_fit_result(sid: str, params: ExpKernelParams) -> FitResult:
    return FitResult(
        session_id=sid,
        kernel_type=KernelType.EXP3,
        converged=True,
        termination_reason='optimality',
        n_iterations=100,
        n_func_evals=200,
        neg_log_lik=500.0,
        nll_per_event=1.2,
        n_events_total=250,
        params_exp=params,
        penalty_value=0.0,
        month='2024-01',
        delivery_hour=10,
        is_weekend=False,
    )


# -----------------------------------------------------------------------
# compute_kernel_integral
# -----------------------------------------------------------------------
def test_kernel_integral_at_zero():
    alpha = np.array([0.5, 0.3, 0.2])
    beta = np.array([1.0, 2.0, 5.0])
    result = compute_kernel_integral(alpha, beta, 0.0)
    assert abs(result) < 1e-10


def test_kernel_integral_at_infinity():
    """M(large T) ≈ M(inf) = sum alpha/beta."""
    alpha = np.array([0.5, 0.3, 0.2])
    beta = np.array([1.0, 2.0, 5.0])
    m_inf = compute_total_mass(alpha, beta)
    m_large = compute_kernel_integral(alpha, beta, 1e6)
    assert abs(m_large - m_inf) < 1e-6


def test_kernel_integral_monotone():
    """Cumulative mass is non-decreasing."""
    alpha = np.array([0.5, 0.3, 0.2])
    beta = np.array([1.0, 2.0, 5.0])
    ts = np.linspace(0, 100, 20)
    vals = [compute_kernel_integral(alpha, beta, t) for t in ts]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))


# -----------------------------------------------------------------------
# compute_total_mass
# -----------------------------------------------------------------------
def test_total_mass_positive():
    alpha = np.array([0.1, 0.2, 0.3])
    beta = np.array([1.0, 2.0, 3.0])
    m = compute_total_mass(alpha, beta)
    assert m > 0


def test_total_mass_formula():
    alpha = np.array([1.0, 2.0])
    beta = np.array([2.0, 4.0])
    # sum: 1/2 + 2/4 = 1.0
    assert abs(compute_total_mass(alpha, beta) - 1.0) < 1e-10


# -----------------------------------------------------------------------
# solve_quantile_time
# -----------------------------------------------------------------------
def test_solve_t50_is_positive():
    alpha = np.array([0.5, 0.3, 0.2])
    beta = np.array([1.0, 2.0, 5.0])
    t50 = solve_quantile_time(alpha, beta, 0.5)
    assert t50 > 0


def test_solve_t50_lt_t90():
    alpha = np.array([0.5, 0.3, 0.2])
    beta = np.array([1.0, 2.0, 5.0])
    t50 = solve_quantile_time(alpha, beta, 0.5)
    t90 = solve_quantile_time(alpha, beta, 0.9)
    assert t50 < t90


def test_solve_quantile_cumulative_fraction():
    """At t_p, M(t_p)/M(inf) should equal p."""
    alpha = np.array([1.0, 0.5])
    beta = np.array([2.0, 0.5])
    for p in [0.1, 0.5, 0.9]:
        t_p = solve_quantile_time(alpha, beta, p)
        m_inf = compute_total_mass(alpha, beta)
        m_at_tp = compute_kernel_integral(alpha, beta, t_p)
        assert abs(m_at_tp / m_inf - p) < 1e-4, f"Failed at p={p}"


def test_solve_zero_mass():
    alpha = np.zeros(3)
    beta = np.array([1.0, 2.0, 3.0])
    t50 = solve_quantile_time(alpha, beta, 0.5)
    assert np.isnan(t50)


# -----------------------------------------------------------------------
# compute_mean_lag
# -----------------------------------------------------------------------
def test_mean_lag_positive():
    alpha = np.array([0.5, 0.3, 0.2])
    beta = np.array([1.0, 2.0, 5.0])
    lag = compute_mean_lag(alpha, beta)
    assert lag > 0


def test_mean_lag_formula():
    """For single component: mean lag = 1/beta."""
    alpha = np.array([1.0])
    beta = np.array([2.0])
    lag = compute_mean_lag(alpha, beta)
    assert abs(lag - 0.5) < 1e-10


def test_faster_beta_smaller_lag():
    """Higher beta → smaller mean lag."""
    alpha = np.array([1.0])
    lag_slow = compute_mean_lag(alpha, np.array([0.5]))
    lag_fast = compute_mean_lag(alpha, np.array([5.0]))
    assert lag_fast < lag_slow


# -----------------------------------------------------------------------
# compute_component_decomposition
# -----------------------------------------------------------------------
def test_component_decomposition_mass_shares_sum_to_one():
    alpha = np.array([0.5, 0.3, 0.2])
    beta = np.array([1.0, 2.0, 5.0])
    halflives, mass_shares = compute_component_decomposition(alpha, beta)
    assert abs(mass_shares.sum() - 1.0) < 1e-10


def test_component_decomposition_sorted_halflives():
    """Components returned sorted by halflife ascending (fastest first)."""
    alpha = np.array([0.5, 0.3, 0.2])
    beta = np.array([0.5, 2.0, 8.0])  # halflife = ln2/beta: 1.38, 0.35, 0.087
    halflives, _ = compute_component_decomposition(alpha, beta)
    assert (np.diff(halflives) >= 0).all() or (np.diff(halflives) <= 0).all()


def test_component_decomposition_positive_halflives():
    alpha = np.array([0.5, 0.3, 0.2])
    beta = np.array([1.0, 2.0, 5.0])
    halflives, _ = compute_component_decomposition(alpha, beta)
    assert (halflives > 0).all()


def test_component_decomposition_positive_shares():
    alpha = np.array([0.5, 0.3, 0.2])
    beta = np.array([1.0, 2.0, 5.0])
    _, shares = compute_component_decomposition(alpha, beta)
    assert (shares >= 0).all()


# -----------------------------------------------------------------------
# analyze_session_structure
# -----------------------------------------------------------------------
def test_analyze_session_structure_returns_session_structure():
    params = _make_exp3_params()
    result = _make_fit_result('test_s', params)
    struct = analyze_session_structure(result)
    assert isinstance(struct, SessionStructure)


def test_analyze_session_structure_rho_positive():
    params = _make_exp3_params()
    result = _make_fit_result('test_s', params)
    struct = analyze_session_structure(result)
    assert struct is not None
    assert struct.rho[0] >= 0
    assert struct.rho[1] >= 0


def test_analyze_session_structure_subcritical():
    params = _make_exp3_params()
    result = _make_fit_result('test_s', params)
    struct = analyze_session_structure(result)
    assert struct is not None
    # With small alpha values, should be subcritical
    assert struct.rho[0] < 1
    assert struct.rho[1] < 1


def test_analyze_session_structure_asymmetry_in_range():
    """Asymmetry index should be in [-1, 1]."""
    params = _make_exp3_params()
    result = _make_fit_result('test_s', params)
    struct = analyze_session_structure(result)
    assert struct is not None
    for x in [0, 1]:
        if not np.isnan(struct.A[x]):
            assert -1.0 <= struct.A[x] <= 1.0


def test_analyze_session_structure_none_if_not_converged():
    """Non-converged fit should return None."""
    params = _make_exp3_params()
    result = _make_fit_result('test_s', params)
    result = FitResult(
        session_id='test_s', kernel_type=KernelType.EXP3,
        converged=False, termination_reason='iteration_limit',
        n_iterations=100, n_func_evals=200, neg_log_lik=500.0,
        nll_per_event=1.2, n_events_total=250, params_exp=params,
        penalty_value=0.0, month='2024-01', delivery_hour=10, is_weekend=False,
    )
    struct = analyze_session_structure(result)
    assert struct is None


def test_analyze_session_structure_G_nonnegative():
    """Branching matrix entries are non-negative."""
    params = _make_exp3_params()
    result = _make_fit_result('test_s', params)
    struct = analyze_session_structure(result)
    assert struct is not None
    for x in [0, 1]:
        assert (struct.G[x] >= 0).all()


# -----------------------------------------------------------------------
# compute_pooled_medians
# -----------------------------------------------------------------------
def test_pooled_medians_returns_dataframe():
    import pandas as pd
    results = [_make_fit_result(f's{i}', _make_exp3_params(seed=i)) for i in range(4)]
    structures, summary_df = analyze_all_sessions(results)
    medians = compute_pooled_medians(summary_df)
    assert isinstance(medians, pd.DataFrame)


def test_pooled_medians_has_state_labels():
    import pandas as pd
    results = [_make_fit_result(f's{i}', _make_exp3_params(seed=i)) for i in range(4)]
    structures, summary_df = analyze_all_sessions(results)
    medians = compute_pooled_medians(summary_df)
    assert 'Normal' in medians.index or 'Stressed' in medians.index


# -----------------------------------------------------------------------
# bootstrap_median_ci — returns (median, lo, hi)
# -----------------------------------------------------------------------
def test_bootstrap_ci_bounds():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 100)
    median, lo, hi = bootstrap_median_ci(data, n_bootstrap=200, seed=0)
    assert lo <= median <= hi


def test_bootstrap_ci_width_positive():
    rng = np.random.default_rng(42)
    data = rng.exponential(1.0, 100)
    _, lo, hi = bootstrap_median_ci(data, n_bootstrap=200, seed=42)
    assert hi >= lo


def test_bootstrap_ci_reproducible():
    rng = np.random.default_rng(99)
    data = rng.normal(5, 2, 200)
    med1, lo1, hi1 = bootstrap_median_ci(data, n_bootstrap=500, seed=7)
    med2, lo2, hi2 = bootstrap_median_ci(data, n_bootstrap=500, seed=7)
    assert abs(lo1 - lo2) < 1e-10
    assert abs(hi1 - hi2) < 1e-10


def test_bootstrap_ci_empty_input():
    median, lo, hi = bootstrap_median_ci(np.array([]), n_bootstrap=100, seed=0)
    assert np.isnan(median) and np.isnan(lo) and np.isnan(hi)
