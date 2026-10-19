"""
Tests for SDHP model parameterization, branching matrix, spectral radius,
spillover matrix, asymmetry index, and parameter vector transformations.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sdhp_models import (
    KernelType, ExpKernelParams, PowerLawKernelParams,
    compute_branching_matrix, compute_spectral_radius,
    compute_spectral_radius_penalty, compute_spillover_matrix,
    compute_asymmetry_index, params_to_vector_exp, vector_to_params_exp,
    params_to_vector_powerlaw, vector_to_params_powerlaw,
    initialize_exp_params, initialize_powerlaw_params,
    N_STATES, SPECTRAL_RADIUS_TARGET, PENALTY_WEIGHT,
)


def _make_params(n_comp: int, scale: float = 0.1) -> ExpKernelParams:
    rng = np.random.default_rng(42)
    return ExpKernelParams(
        mu=rng.uniform(0.01, 0.05, size=(N_STATES, 2)),
        alpha=rng.uniform(0.001, scale, size=(N_STATES, 2, 2, n_comp)),
        beta=rng.uniform(0.5, 10.0, size=(N_STATES, 2, 2, n_comp)),
        n_components=n_comp,
        kernel_type=KernelType.EXP3 if n_comp == 3 else KernelType.EXP1,
    )


# -----------------------------------------------------------------------
# Branching matrix
# -----------------------------------------------------------------------
def test_branching_matrix_shape():
    params = _make_params(3)
    G = compute_branching_matrix(params, 0)
    assert G.shape == (2, 2)


def test_branching_matrix_formula():
    """G_{ij} = sum_r alpha[x,i,j,r] / beta[x,i,j,r]."""
    params = _make_params(2)
    G = compute_branching_matrix(params, 0)
    for i in range(2):
        for j in range(2):
            expected = float(np.sum(params.alpha[0, i, j, :] / params.beta[0, i, j, :]))
            assert G[i, j] == pytest.approx(expected, rel=1e-10)


def test_branching_matrix_nonnegative():
    params = _make_params(3)
    for x in range(N_STATES):
        G = compute_branching_matrix(params, x)
        assert (G >= 0).all()


def test_branching_matrix_different_states():
    """States 0 and 1 should produce different matrices (different params)."""
    rng = np.random.default_rng(7)
    params = ExpKernelParams(
        mu=rng.uniform(0.01, 0.05, (2, 2)),
        alpha=rng.uniform(0.001, 0.3, (2, 2, 2, 2)),
        beta=rng.uniform(1.0, 5.0, (2, 2, 2, 2)),
        n_components=2,
        kernel_type=KernelType.EXP2,
    )
    G0 = compute_branching_matrix(params, 0)
    G1 = compute_branching_matrix(params, 1)
    assert not np.allclose(G0, G1)


# -----------------------------------------------------------------------
# Spectral radius
# -----------------------------------------------------------------------
def test_spectral_radius_identity():
    """Spectral radius of identity matrix is 1."""
    rho = compute_spectral_radius(np.eye(2))
    assert rho == pytest.approx(1.0)


def test_spectral_radius_zero_matrix():
    rho = compute_spectral_radius(np.zeros((2, 2)))
    assert rho == pytest.approx(0.0)


def test_spectral_radius_nonnegative():
    for _ in range(5):
        G = np.random.rand(2, 2) * 0.5
        rho = compute_spectral_radius(G)
        assert rho >= 0.0


def test_spectral_radius_diagonal():
    """Spectral radius of diagonal matrix = max of diagonal entries."""
    G = np.diag([0.3, 0.7])
    rho = compute_spectral_radius(G)
    assert rho == pytest.approx(0.7)


# -----------------------------------------------------------------------
# Spectral radius penalty
# -----------------------------------------------------------------------
def test_penalty_zero_when_subcritical():
    """No penalty when rho < target for all states."""
    # Use tiny alpha to ensure rho << 0.95
    params = ExpKernelParams(
        mu=np.ones((2, 2)) * 0.01,
        alpha=np.ones((2, 2, 2, 1)) * 1e-5,
        beta=np.ones((2, 2, 2, 1)) * 10.0,
        n_components=1,
        kernel_type=KernelType.EXP1,
    )
    penalty = compute_spectral_radius_penalty(params)
    assert penalty == pytest.approx(0.0, abs=1e-8)


def test_penalty_positive_when_supercritical():
    """Penalty > 0 when rho > target."""
    params = ExpKernelParams(
        mu=np.ones((2, 2)) * 0.01,
        alpha=np.ones((2, 2, 2, 1)) * 2.0,  # large alpha → large G
        beta=np.ones((2, 2, 2, 1)) * 1.0,
        n_components=1,
        kernel_type=KernelType.EXP1,
    )
    penalty = compute_spectral_radius_penalty(params)
    assert penalty > 0.0


def test_penalty_weight_scaling():
    """Doubling omega doubles penalty."""
    params = ExpKernelParams(
        mu=np.ones((2, 2)) * 0.01,
        alpha=np.ones((2, 2, 2, 1)) * 1.5,
        beta=np.ones((2, 2, 2, 1)) * 1.0,
        n_components=1,
        kernel_type=KernelType.EXP1,
    )
    p1 = compute_spectral_radius_penalty(params, omega=1e3)
    p2 = compute_spectral_radius_penalty(params, omega=2e3)
    assert p2 == pytest.approx(2 * p1, rel=1e-10)


# -----------------------------------------------------------------------
# Spillover matrix
# -----------------------------------------------------------------------
def test_spillover_matrix_shape():
    G = np.array([[0.8, 0.02], [0.05, 0.85]])
    Psi = compute_spillover_matrix(G)
    assert Psi.shape == (2, 2)


def test_spillover_matrix_diagonal_dominance():
    """Spillover diagonal > 1 for subcritical G with positive entries."""
    G = np.array([[0.7, 0.05], [0.05, 0.6]])
    Psi = compute_spillover_matrix(G)
    assert Psi[0, 0] > 1.0
    assert Psi[1, 1] > 1.0


def test_spillover_matrix_identity_check():
    """(I - G) Psi = I."""
    G = np.array([[0.8, 0.02], [0.05, 0.85]])
    Psi = compute_spillover_matrix(G)
    product = (np.eye(2) - G) @ Psi
    np.testing.assert_allclose(product, np.eye(2), atol=1e-10)


# -----------------------------------------------------------------------
# Asymmetry index
# -----------------------------------------------------------------------
def test_asymmetry_index_range():
    """Asymmetry index is in (-1, 1)."""
    Psi = np.array([[5.0, 2.0], [3.0, 4.0]])
    A = compute_asymmetry_index(Psi)
    assert -1.0 < A < 1.0


def test_asymmetry_index_symmetric():
    """Equal off-diagonal spillovers → A = 0."""
    Psi = np.array([[5.0, 3.0], [3.0, 4.0]])
    A = compute_asymmetry_index(Psi)
    assert A == pytest.approx(0.0)


def test_asymmetry_index_sign():
    """Psi_HQ > Psi_QH → A > 0."""
    Psi = np.array([[4.0, 8.0], [2.0, 5.0]])
    A = compute_asymmetry_index(Psi)
    assert A > 0.0


# -----------------------------------------------------------------------
# Parameter vector serialization
# -----------------------------------------------------------------------
def test_params_roundtrip_exp1():
    params = _make_params(1)
    params.kernel_type = KernelType.EXP1
    v = params_to_vector_exp(params)
    p2 = vector_to_params_exp(v, 1, KernelType.EXP1)
    np.testing.assert_allclose(p2.mu, params.mu, rtol=1e-6)
    np.testing.assert_allclose(p2.alpha, params.alpha, rtol=1e-6)
    np.testing.assert_allclose(p2.beta, params.beta, rtol=1e-6)


def test_params_roundtrip_exp3():
    params = _make_params(3)
    v = params_to_vector_exp(params)
    p2 = vector_to_params_exp(v, 3, KernelType.EXP3)
    np.testing.assert_allclose(p2.mu, params.mu, rtol=1e-6)
    np.testing.assert_allclose(p2.alpha, params.alpha, rtol=1e-6)
    np.testing.assert_allclose(p2.beta, params.beta, rtol=1e-6)


def test_vector_to_params_exp_positive():
    """Exponentiated params must be positive."""
    rng = np.random.default_rng(99)
    v = rng.normal(0, 1, 100)
    # Truncate to correct size for EXP3
    params_test = _make_params(3)
    v_correct = params_to_vector_exp(params_test)
    p2 = vector_to_params_exp(v_correct, 3, KernelType.EXP3)
    assert (p2.mu > 0).all()
    assert (p2.alpha > 0).all()
    assert (p2.beta > 0).all()


def test_params_roundtrip_powerlaw():
    rng = np.random.default_rng(42)
    params = PowerLawKernelParams(
        mu=rng.uniform(0.01, 0.05, (2, 2)),
        alpha=rng.uniform(0.1, 1.0, (2, 2, 2)),
        eta=rng.uniform(1.5, 3.0, (2, 2, 2)),
        c=rng.uniform(1.0, 60.0, (2, 2, 2)),
    )
    v = params_to_vector_powerlaw(params)
    p2 = vector_to_params_powerlaw(v)
    np.testing.assert_allclose(p2.mu, params.mu, rtol=1e-6)
    np.testing.assert_allclose(p2.alpha, params.alpha, rtol=1e-6)
    np.testing.assert_allclose(p2.eta, params.eta, rtol=1e-6)
    np.testing.assert_allclose(p2.c, params.c, rtol=1e-6)


# -----------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------
def test_initialize_exp_params_correct_shape():
    params = initialize_exp_params(KernelType.EXP3, h_rate=0.01, q_rate=0.02)
    assert params.mu.shape == (N_STATES, 2)
    assert params.alpha.shape == (N_STATES, 2, 2, 3)
    assert params.beta.shape == (N_STATES, 2, 2, 3)


def test_initialize_exp_params_positive():
    params = initialize_exp_params(KernelType.EXP1, h_rate=0.01, q_rate=0.02)
    assert (params.mu > 0).all()
    assert (params.alpha > 0).all()
    assert (params.beta > 0).all()


def test_initialize_exp_params_baseline_ratio():
    """mu[1,e] / mu[0,e] = 1.5 / 0.5 = 3 (stressed baseline = 3x normal)."""
    params = initialize_exp_params(KernelType.EXP3, h_rate=0.01, q_rate=0.02)
    ratio = params.mu[1, :] / params.mu[0, :]
    np.testing.assert_allclose(ratio, [3.0, 3.0], rtol=1e-10)


def test_initialize_powerlaw_params():
    params = initialize_powerlaw_params(h_rate=0.01, q_rate=0.02)
    assert params.mu.shape == (N_STATES, 2)
    assert (params.eta > 1).all()
    assert (params.c > 0).all()
