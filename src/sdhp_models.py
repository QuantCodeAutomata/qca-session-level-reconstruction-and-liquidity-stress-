import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class KernelType(Enum):
    EXP1 = '1-Exp'
    EXP2 = '2-Exp'
    EXP3 = '3-Exp'
    POWERLAW = 'power-law'


EVENT_TYPES = ['H', 'Q']  # Hourly, Quarter-Hourly
N_STATES = 2              # 0=Normal, 1=Stressed
SPECTRAL_RADIUS_TARGET = 0.95
PENALTY_WEIGHT = 1e3

_N_COMPONENTS_MAP = {
    KernelType.EXP1: 1,
    KernelType.EXP2: 2,
    KernelType.EXP3: 3,
}


@dataclass
class ExpKernelParams:
    """Parameters for n-component exponential kernel."""
    # For each state x, source e', target e: alpha[x][e'][e][r], beta[x][e'][e][r]
    # alpha shape: (n_states, 2, 2, n_components)
    # beta shape:  (n_states, 2, 2, n_components)
    mu: np.ndarray      # shape (n_states, 2) - baseline intensities
    alpha: np.ndarray   # shape (n_states, 2, 2, n_components)
    beta: np.ndarray    # shape (n_states, 2, 2, n_components)
    n_components: int
    kernel_type: KernelType


@dataclass
class PowerLawKernelParams:
    """Parameters for power-law kernel."""
    mu: np.ndarray    # shape (n_states, 2)
    alpha: np.ndarray # shape (n_states, 2, 2)
    eta: np.ndarray   # shape (n_states, 2, 2), must be > 1
    c: np.ndarray     # shape (n_states, 2, 2), must be > 0


# ---------------------------------------------------------------------------
# Branching matrix helpers
# ---------------------------------------------------------------------------

def compute_branching_matrix(
    params: ExpKernelParams,
    state: int
) -> np.ndarray:
    """Compute regime-specific branching matrix G^(x).

    G_{e'->e}^(x) = sum_r alpha_{e'->e}^{(x,r)} / beta_{e'->e}^{(x,r)}

    Returns 2x2 matrix where rows=source (e'), cols=target (e).
    """
    # alpha[state]: shape (2, 2, n_components)
    # G[e', e] = sum_r alpha[state, e', e, r] / beta[state, e', e, r]
    ratio = params.alpha[state] / params.beta[state]   # (2, 2, n_components)
    return ratio.sum(axis=-1)                           # (2, 2)


def compute_spectral_radius(
    branching_matrix: np.ndarray
) -> float:
    """Compute spectral radius (largest eigenvalue magnitude) of branching matrix."""
    eigvals = np.linalg.eigvals(branching_matrix)
    return float(np.max(np.abs(eigvals)))


def compute_spectral_radius_penalty(
    params: ExpKernelParams,
    target_rho: float = SPECTRAL_RADIUS_TARGET,
    omega: float = PENALTY_WEIGHT
) -> float:
    """Compute per-state spectral-radius penalty.

    P(theta) = sum_{x=0}^1 omega * (max(0, rho_x - rho*))^2
    """
    penalty = 0.0
    for x in range(N_STATES):
        G_x = compute_branching_matrix(params, x)
        rho_x = compute_spectral_radius(G_x)
        excess = max(0.0, rho_x - target_rho)
        penalty += omega * excess ** 2
    return penalty


def compute_spillover_matrix(
    branching_matrix: np.ndarray
) -> np.ndarray:
    """Compute spillover multiplier matrix Psi = (I - G)^{-1}.

    Verify spectral radius < 1 before inversion.
    Returns 2x2 spillover matrix.
    """
    rho = compute_spectral_radius(branching_matrix)
    if rho >= 1.0:
        raise ValueError(
            f"Spectral radius {rho:.4f} >= 1; process is not stationary."
        )
    return np.linalg.inv(np.eye(2) - branching_matrix)


def compute_asymmetry_index(
    spillover_matrix: np.ndarray
) -> float:
    """Compute asymmetry index A = (Psi_{H->Q} - Psi_{Q->H}) / (Psi_{H->Q} + Psi_{Q->H}).

    Psi_{H->Q} = spillover_matrix[0, 1]  (H is source, Q is target)
    Psi_{Q->H} = spillover_matrix[1, 0]  (Q is source, H is target)
    """
    psi_hq = spillover_matrix[0, 1]
    psi_qh = spillover_matrix[1, 0]
    denom = psi_hq + psi_qh
    if denom == 0.0:
        return 0.0
    return float((psi_hq - psi_qh) / denom)


# ---------------------------------------------------------------------------
# Parameter <-> vector conversions (log-space)
# ---------------------------------------------------------------------------

def params_to_vector_exp(
    params: ExpKernelParams
) -> np.ndarray:
    """Flatten ExpKernelParams to optimization vector (log-transformed).

    Transform: log(mu), log(alpha), log(beta)
    Order: all mu values, then all alpha values, then all beta values.
    Flattened in order: state, event_type for mu;
    state, source, target, component for alpha, beta.
    """
    log_mu    = np.log(params.mu).ravel()       # (n_states*2,)
    log_alpha = np.log(params.alpha).ravel()    # (n_states*2*2*n_components,)
    log_beta  = np.log(params.beta).ravel()     # same shape
    return np.concatenate([log_mu, log_alpha, log_beta])


def vector_to_params_exp(
    vector: np.ndarray,
    n_components: int,
    kernel_type: KernelType
) -> ExpKernelParams:
    """Reconstruct ExpKernelParams from optimization vector (exponentiate)."""
    n_mu    = N_STATES * 2
    n_ab    = N_STATES * 2 * 2 * n_components

    log_mu    = vector[:n_mu]
    log_alpha = vector[n_mu : n_mu + n_ab]
    log_beta  = vector[n_mu + n_ab : n_mu + 2 * n_ab]

    mu    = np.exp(log_mu).reshape(N_STATES, 2)
    alpha = np.exp(log_alpha).reshape(N_STATES, 2, 2, n_components)
    beta  = np.exp(log_beta).reshape(N_STATES, 2, 2, n_components)

    return ExpKernelParams(
        mu=mu,
        alpha=alpha,
        beta=beta,
        n_components=n_components,
        kernel_type=kernel_type,
    )


def params_to_vector_powerlaw(
    params: PowerLawKernelParams
) -> np.ndarray:
    """Flatten PowerLawKernelParams: log(mu), log(alpha), log(eta-1), log(c)."""
    log_mu      = np.log(params.mu).ravel()            # (n_states*2,)
    log_alpha   = np.log(params.alpha).ravel()         # (n_states*2*2,)
    log_eta_m1  = np.log(params.eta - 1.0).ravel()    # (n_states*2*2,)
    log_c       = np.log(params.c).ravel()             # (n_states*2*2,)
    return np.concatenate([log_mu, log_alpha, log_eta_m1, log_c])


def vector_to_params_powerlaw(
    vector: np.ndarray
) -> PowerLawKernelParams:
    """Reconstruct PowerLawKernelParams from optimization vector."""
    n_mu  = N_STATES * 2     # 4
    n_mat = N_STATES * 2 * 2 # 8

    log_mu     = vector[:n_mu]
    log_alpha  = vector[n_mu           : n_mu + n_mat]
    log_eta_m1 = vector[n_mu + n_mat   : n_mu + 2 * n_mat]
    log_c      = vector[n_mu + 2*n_mat : n_mu + 3 * n_mat]

    mu    = np.exp(log_mu).reshape(N_STATES, 2)
    alpha = np.exp(log_alpha).reshape(N_STATES, 2, 2)
    eta   = np.exp(log_eta_m1).reshape(N_STATES, 2, 2) + 1.0
    c     = np.exp(log_c).reshape(N_STATES, 2, 2)

    return PowerLawKernelParams(mu=mu, alpha=alpha, eta=eta, c=c)


# ---------------------------------------------------------------------------
# Parameter initialization
# ---------------------------------------------------------------------------

def initialize_exp_params(
    kernel_type: KernelType,
    h_rate: float,
    q_rate: float,
    rng: np.random.Generator = None
) -> ExpKernelParams:
    """Initialize parameters for exponential kernel.

    Per methodology:
    - mu_{0,e}^(0) = 0.5 * nu_e  (normal state baseline)
    - mu_{1,e}^(0) = 1.5 * nu_e  (stressed state baseline)
    - nu_H = h_rate, nu_Q = q_rate  (empirical event rates in events/second)
    - Draw branching mass b ~ Uniform(0.01, 0.30)
    - Initialize beta in [1e-4, 1000.0], alpha consistent with branching mass
    """
    if rng is None:
        rng = np.random.default_rng()

    n_components = _N_COMPONENTS_MAP[kernel_type]

    # Baseline intensities
    rates = np.array([h_rate, q_rate])
    mu = np.stack([0.5 * rates, 1.5 * rates], axis=0)  # (2, 2)
    mu = np.clip(mu, 1e-4, None)

    # Branching mass shared equally across all (state, source, target, component)
    b = rng.uniform(0.01, 0.30)
    # Each G[e', e] entry gets b / (2*2) mass, split equally over components
    branch_per_entry_per_component = b / (4.0 * n_components)

    # Beta drawn log-uniformly in [1e-4, 1000.0]
    log_beta_lo, log_beta_hi = np.log(1e-4), np.log(1000.0)
    beta_vals = np.exp(
        rng.uniform(log_beta_lo, log_beta_hi, size=(N_STATES, 2, 2, n_components))
    )

    # alpha = (b per entry per component) * beta
    alpha_vals = branch_per_entry_per_component * beta_vals

    return ExpKernelParams(
        mu=mu,
        alpha=alpha_vals,
        beta=beta_vals,
        n_components=n_components,
        kernel_type=kernel_type,
    )


def initialize_powerlaw_params(
    h_rate: float,
    q_rate: float,
    rng: np.random.Generator = None
) -> PowerLawKernelParams:
    """Initialize power-law parameters.

    eta > 1, c > 0; use documented reasonable defaults.
    mu initialized same as exponential.
    """
    if rng is None:
        rng = np.random.default_rng()

    rates = np.array([h_rate, q_rate])
    mu = np.stack([0.5 * rates, 1.5 * rates], axis=0)
    mu = np.clip(mu, 1e-4, None)

    # alpha: small initial branching mass
    b = rng.uniform(0.01, 0.30)
    alpha = np.full((N_STATES, 2, 2), b / 4.0)

    # eta drawn uniformly in (1.5, 3.0); c in [1.0, 60.0] seconds
    eta = rng.uniform(1.5, 3.0, size=(N_STATES, 2, 2))
    c   = np.exp(rng.uniform(np.log(1.0), np.log(60.0), size=(N_STATES, 2, 2)))

    return PowerLawKernelParams(mu=mu, alpha=alpha, eta=eta, c=c)
