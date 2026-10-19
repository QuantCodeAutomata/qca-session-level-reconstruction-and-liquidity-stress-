"""
Structural analysis of converged subcritical 3-Exponential SDHP fits.

Computes branching matrices, spectral radii, spillover multipliers, asymmetry
indices, amplification ratios, response-speed metrics (t50, t90, mean lag),
and kernel-component decompositions. Implements Experiment 3 from the paper.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.sdhp_models import (
    ExpKernelParams, KernelType, N_STATES,
    compute_branching_matrix, compute_spectral_radius, compute_spillover_matrix,
    compute_asymmetry_index,
)
from src.optimization import FitResult

N_BOOTSTRAP = 1000  # bootstrap replications for CI
BOOTSTRAP_SEED = 42

# Channel labels: rows = source, cols = target
# Index 0 = H, 1 = Q
CHANNEL_LABELS = {(0, 0): 'H->H', (0, 1): 'H->Q', (1, 0): 'Q->H', (1, 1): 'Q->Q'}
STATE_LABELS = {0: 'Normal', 1: 'Stressed'}


@dataclass
class SessionStructure:
    """All derived structural quantities for one converged subcritical session."""
    session_id: str
    month: str
    delivery_hour: int
    is_weekend: bool

    # Branching matrices [state] -> 2x2 array
    G: Dict[int, np.ndarray]
    # Spectral radii
    rho: Dict[int, float]
    # Spillover matrices
    Psi: Dict[int, np.ndarray]
    # Asymmetry indices
    A: Dict[int, float]
    # Baseline intensities mu[state, event_type]
    mu: np.ndarray  # shape (2, 2)

    # Response speed metrics: dict[state][source][target]
    t50: Dict[int, Dict[Tuple, float]]
    t90: Dict[int, Dict[Tuple, float]]
    mean_lag: Dict[int, Dict[Tuple, float]]

    # Kernel decomposition per channel/state
    # component_halflives[state][(src,tgt)] -> array shape (n_comp,) sorted
    component_halflives: Dict[int, Dict[Tuple, np.ndarray]]
    component_mass_shares: Dict[int, Dict[Tuple, np.ndarray]]

    # Within-session differences (Stressed - Normal)
    delta_G: np.ndarray   # shape (2,2)
    delta_rho: float
    delta_A: float

    # Amplification ratios Psi/G per channel/state
    amplification: Dict[int, Dict[Tuple, float]]


def compute_kernel_integral(
    alpha: np.ndarray,
    beta: np.ndarray,
    t: float
) -> float:
    """Compute cumulative branching mass M(t) = sum_r (alpha_r/beta_r)*(1-exp(-beta_r*t)).

    # Custom — Context7 found no library equivalent (paper Eq. for cumulative mass)
    """
    return float(np.sum((alpha / beta) * (1.0 - np.exp(-beta * t))))


def compute_total_mass(alpha: np.ndarray, beta: np.ndarray) -> float:
    """M(infinity) = sum_r alpha_r / beta_r."""
    return float(np.sum(alpha / beta))


def solve_quantile_time(
    alpha: np.ndarray,
    beta: np.ndarray,
    p: float,
    upper_bound_multiplier: float = 100.0
) -> float:
    """Solve for t_p such that M(t_p) / M(inf) = p.

    Uses Brent's method on the closed-form cumulative mass.
    # Custom — Context7 found no library equivalent (paper response-speed metric)
    """
    M_inf = compute_total_mass(alpha, beta)
    if M_inf <= 0:
        return np.nan
    target = p * M_inf

    # Adaptive upper bound: expand until cumulative mass exceeds target
    half_life_max = float(np.log(2) / np.min(beta[beta > 0])) if np.any(beta > 0) else 1.0
    upper = upper_bound_multiplier * half_life_max
    for _ in range(20):
        if compute_kernel_integral(alpha, beta, upper) >= target:
            break
        upper *= 2.0

    f = lambda t: compute_kernel_integral(alpha, beta, t) - target
    try:
        return brentq(f, 0.0, upper, xtol=1e-3)
    except ValueError:
        return np.nan


def compute_mean_lag(alpha: np.ndarray, beta: np.ndarray) -> float:
    """Mean lag = [sum_r alpha_r/beta_r^2] / [sum_r alpha_r/beta_r].

    # Custom — Context7 found no library equivalent (paper Eq. for mean lag)
    """
    numerator = float(np.sum(alpha / beta ** 2))
    denominator = float(np.sum(alpha / beta))
    return numerator / denominator if denominator > 0 else np.nan


def compute_component_decomposition(
    alpha: np.ndarray,
    beta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-component half-lives and mass shares, sorted by decay rate.

    Returns (half_lives, mass_shares) each of shape (n_comp,).
    Components sorted ascending by half-life (fastest first).
    """
    half_lives = np.log(2) / beta           # shape (n_comp,)
    masses = alpha / beta                    # shape (n_comp,)
    total_mass = masses.sum()
    mass_shares = masses / total_mass if total_mass > 0 else np.zeros_like(masses)
    order = np.argsort(half_lives)          # sort by half-life ascending
    return half_lives[order], mass_shares[order]


def analyze_session_structure(
    fit: FitResult,
) -> Optional[SessionStructure]:
    """Compute all structural quantities for a single converged subcritical session.

    Returns None if fit is not converged or not subcritical.
    """
    if not fit.converged or fit.params_exp is None:
        return None

    params = fit.params_exp
    G: Dict[int, np.ndarray] = {}
    rho: Dict[int, float] = {}
    Psi: Dict[int, np.ndarray] = {}
    A: Dict[int, float] = {}
    t50: Dict[int, Dict] = {}
    t90: Dict[int, Dict] = {}
    mean_lags: Dict[int, Dict] = {}
    comp_hl: Dict[int, Dict] = {}
    comp_ms: Dict[int, Dict] = {}
    amplification: Dict[int, Dict] = {}

    for x in range(N_STATES):
        G_x = compute_branching_matrix(params, x)
        rho_x = compute_spectral_radius(G_x)
        if rho_x >= 1.0:
            return None   # not subcritical
        G[x] = G_x
        rho[x] = rho_x

        Psi_x = compute_spillover_matrix(G_x)
        Psi[x] = Psi_x
        A[x] = compute_asymmetry_index(Psi_x)

        t50[x] = {}
        t90[x] = {}
        mean_lags[x] = {}
        comp_hl[x] = {}
        comp_ms[x] = {}
        amplification[x] = {}

        for src in range(2):
            for tgt in range(2):
                a = params.alpha[x, src, tgt, :]
                b = params.beta[x, src, tgt, :]
                channel = (src, tgt)
                t50[x][channel] = solve_quantile_time(a, b, 0.5)
                t90[x][channel] = solve_quantile_time(a, b, 0.9)
                mean_lags[x][channel] = compute_mean_lag(a, b)
                hl, ms = compute_component_decomposition(a, b)
                comp_hl[x][channel] = hl
                comp_ms[x][channel] = ms
                g_val = G_x[src, tgt]
                psi_val = Psi_x[src, tgt]
                amplification[x][channel] = psi_val / g_val if g_val > 0 else np.nan

    delta_G = G[1] - G[0]
    delta_rho = rho[1] - rho[0]
    delta_A = A[1] - A[0]

    return SessionStructure(
        session_id=fit.session_id,
        month=fit.month,
        delivery_hour=fit.delivery_hour,
        is_weekend=fit.is_weekend,
        G=G, rho=rho, Psi=Psi, A=A,
        mu=params.mu.copy(),
        t50=t50, t90=t90, mean_lag=mean_lags,
        component_halflives=comp_hl,
        component_mass_shares=comp_ms,
        delta_G=delta_G, delta_rho=delta_rho, delta_A=delta_A,
        amplification=amplification,
    )


def analyze_all_sessions(
    fit_results: List[FitResult],
) -> Tuple[List[SessionStructure], pd.DataFrame]:
    """Analyze all converged subcritical sessions.

    Returns (structures_list, summary_df).
    """
    structures = []
    for fit in fit_results:
        s = analyze_session_structure(fit)
        if s is not None:
            structures.append(s)
    summary = _build_summary_df(structures)
    return structures, summary


def _build_summary_df(structures: List[SessionStructure]) -> pd.DataFrame:
    """Build flat summary DataFrame from list of SessionStructure objects."""
    rows = []
    for s in structures:
        for x in range(N_STATES):
            row = {
                'session_id': s.session_id,
                'month': s.month,
                'delivery_hour': s.delivery_hour,
                'is_weekend': s.is_weekend,
                'state': x,
                'state_label': STATE_LABELS[x],
                'rho': s.rho[x],
                'A': s.A[x],
                'mu_H': s.mu[x, 0],
                'mu_Q': s.mu[x, 1],
                'G_HH': s.G[x][0, 0],
                'G_HQ': s.G[x][0, 1],
                'G_QH': s.G[x][1, 0],
                'G_QQ': s.G[x][1, 1],
                'Psi_HH': s.Psi[x][0, 0],
                'Psi_HQ': s.Psi[x][0, 1],
                'Psi_QH': s.Psi[x][1, 0],
                'Psi_QQ': s.Psi[x][1, 1],
            }
            for (src, tgt), label in CHANNEL_LABELS.items():
                ch = (src, tgt)
                row[f't50_{label}'] = s.t50[x].get(ch, np.nan)
                row[f't90_{label}'] = s.t90[x].get(ch, np.nan)
                row[f'mean_lag_{label}'] = s.mean_lag[x].get(ch, np.nan)
                row[f'ampl_{label}'] = s.amplification[x].get(ch, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_pooled_medians(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pooled medians for key structural quantities by state."""
    key_cols = [
        'rho', 'A', 'mu_H', 'mu_Q',
        'G_HH', 'G_HQ', 'G_QH', 'G_QQ',
        'Psi_HH', 'Psi_HQ', 'Psi_QH', 'Psi_QQ',
        't50_H->H', 't50_H->Q', 't50_Q->H', 't50_Q->Q',
        't90_H->H', 't90_H->Q', 't90_Q->H', 't90_Q->Q',
        'mean_lag_H->H', 'mean_lag_H->Q', 'mean_lag_Q->H', 'mean_lag_Q->Q',
        'ampl_H->H', 'ampl_H->Q', 'ampl_Q->H', 'ampl_Q->Q',
    ]
    available = [c for c in key_cols if c in summary_df.columns]
    return summary_df.groupby('state_label')[available].median()


def compute_within_session_differences(
    structures: List[SessionStructure],
) -> pd.DataFrame:
    """Compute session-level Stressed - Normal differences for each structural qty.

    Used for within-contract regime comparison analysis.
    """
    rows = []
    for s in structures:
        row = {
            'session_id': s.session_id,
            'month': s.month,
            'delivery_hour': s.delivery_hour,
            'is_weekend': s.is_weekend,
            'delta_rho': s.delta_rho,
            'delta_A': s.delta_A,
            'delta_G_HH': s.delta_G[0, 0],
            'delta_G_HQ': s.delta_G[0, 1],
            'delta_G_QH': s.delta_G[1, 0],
            'delta_G_QQ': s.delta_G[1, 1],
            'delta_mu_H': s.mu[1, 0] - s.mu[0, 0],
            'delta_mu_Q': s.mu[1, 1] - s.mu[0, 1],
        }
        # Spillover differences
        for src in range(2):
            for tgt in range(2):
                label = CHANNEL_LABELS[(src, tgt)]
                row[f'delta_Psi_{label}'] = s.Psi[1][src, tgt] - s.Psi[0][src, tgt]
                t50_0 = s.t50[0].get((src, tgt), np.nan)
                t50_1 = s.t50[1].get((src, tgt), np.nan)
                row[f'delta_t50_{label}'] = t50_1 - t50_0
                ml_0 = s.mean_lag[0].get((src, tgt), np.nan)
                ml_1 = s.mean_lag[1].get((src, tgt), np.nan)
                row[f'delta_mean_lag_{label}'] = ml_1 - ml_0
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_median_ci(
    values: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Compute median and percentile bootstrap CI.

    Returns (median, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    median = float(np.median(values))
    boot_medians = np.array([
        np.median(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(boot_medians, 100 * alpha))
    hi = float(np.percentile(boot_medians, 100 * (1 - alpha)))
    return median, lo, hi


def compute_delta_bootstrap_table(
    diff_df: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """Compute bootstrap CI for median within-session differences.

    Returns DataFrame with columns: quantity, median, lower_95, upper_95, pct_positive.
    """
    delta_cols = [c for c in diff_df.columns if c.startswith('delta_')]
    rows = []
    for col in delta_cols:
        vals = diff_df[col].dropna().values
        med, lo, hi = bootstrap_median_ci(vals, n_bootstrap=n_bootstrap)
        pct_pos = float(np.mean(vals > 0)) if len(vals) > 0 else np.nan
        rows.append({
            'quantity': col,
            'median': med,
            'lower_95': lo,
            'upper_95': hi,
            'pct_positive': pct_pos,
        })
    return pd.DataFrame(rows)


def summarize_by_period(
    summary_df: pd.DataFrame,
    group_col: str = 'month',
) -> pd.DataFrame:
    """Summarize structural quantities grouped by a period column."""
    key_cols = ['rho', 'A', 'Psi_HQ', 'Psi_QH', 'G_HH', 'G_QQ']
    available = [c for c in key_cols if c in summary_df.columns]
    return (
        summary_df.groupby(['state_label', group_col])[available]
        .median()
        .reset_index()
    )
