import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from src.sdhp_models import ExpKernelParams, PowerLawKernelParams, KernelType, N_STATES

LJUNG_BOX_LAGS   = 10
SIGNIFICANCE_LEVEL = 0.05

_MIN_INTENSITY = 1e-10


# ---------------------------------------------------------------------------
# Internal helpers (mirrored from likelihood_exp to avoid circular imports)
# ---------------------------------------------------------------------------

def _get_state_at_time(t: float, state_path: np.ndarray) -> int:
    times = state_path[:, 0]
    idx   = int(np.searchsorted(times, t, side='right')) - 1
    return int(state_path[max(idx, 0), 1])


def _get_sub_intervals(
    t_a: float,
    t_b: float,
    state_path: np.ndarray,
) -> List[Tuple[float, float, int]]:
    if t_b <= t_a:
        return []
    times  = state_path[:, 0]
    states = state_path[:, 1].astype(int)
    idx    = max(int(np.searchsorted(times, t_a, side='right')) - 1, 0)
    sub    = []
    t_cur  = t_a
    while t_cur < t_b:
        x_seg = states[idx]
        t_next = float(times[idx + 1]) if idx + 1 < len(times) else t_b
        t_next = min(t_next, t_b)
        if t_next > t_cur:
            sub.append((t_cur, t_next, x_seg))
        t_cur = t_next
        idx  += 1
    return sub


# ---------------------------------------------------------------------------
# Compensator computation
# ---------------------------------------------------------------------------

def compute_compensator_exp(
    events: np.ndarray,        # times of events of one specific type
    all_events: np.ndarray,    # all events (time, type) array
    event_type: int,           # 0=H, 1=Q
    state_path: np.ndarray,    # (n_obs, 2) = [time_sec, state]
    session_end: float,
    params: ExpKernelParams
) -> np.ndarray:
    """Compute compensator Lambda_e(t_k) = integral_0^{t_k} lambda_e(t) dt.

    Returns array of compensator values at each event time in `events`.
    Used to compute time-rescaled residuals: Delta_k = Lambda(t_k) - Lambda(t_{k-1}).
    Under correct model, Delta_k ~ Exp(1) i.i.d.
    """
    if len(events) == 0:
        return np.empty(0)

    n_comp      = params.n_components
    n_type_evs  = len(events)

    # Build sorted combined event list
    all_ev = all_events  # assumed sorted (time, type)

    # Running excitation sums: R[x, e', e, r]
    R             = np.zeros((N_STATES, 2, 2, n_comp))
    compensator_e = 0.0   # cumulative Lambda_e

    result        = np.empty(n_type_evs)
    type_ev_idx   = 0     # pointer into events[]

    t_prev        = 0.0
    n_all         = len(all_ev)

    for k in range(n_all):
        t_k = float(all_ev[k, 0])
        e_k = int(all_ev[k, 1])

        # Process interval [t_prev, t_k]: update compensator and decay R
        for t_a, t_b, x_seg in _get_sub_intervals(t_prev, t_k, state_path):
            seg_dt = t_b - t_a
            # compensator for target stream e = event_type only
            e = event_type
            b  = params.beta[x_seg, :, e, :]   # (2, n_comp)
            rv = R[x_seg, :, e, :]              # (2, n_comp)
            compensator_e += params.mu[x_seg, e] * seg_dt
            compensator_e += float(np.sum(rv / b * (1.0 - np.exp(-b * seg_dt))))
            # Decay ALL R sums
            R *= np.exp(-params.beta * seg_dt)

        # Record compensator if this is a type-e event (compare with events[type_ev_idx])
        if e_k == event_type and type_ev_idx < n_type_evs:
            result[type_ev_idx] = compensator_e
            type_ev_idx += 1

        # Update R for all states
        R[:, e_k, :, :] += params.alpha[:, e_k, :, :]
        t_prev = t_k

    # Handle remaining type-e events after last all_ev (shouldn't happen if all_events complete)
    # Process tail if any type_ev_idx remain (they might be beyond all events in all_ev)
    while type_ev_idx < n_type_evs:
        t_k = float(events[type_ev_idx])
        for t_a, t_b, x_seg in _get_sub_intervals(t_prev, t_k, state_path):
            seg_dt = t_b - t_a
            e = event_type
            b  = params.beta[x_seg, :, e, :]
            rv = R[x_seg, :, e, :]
            compensator_e += params.mu[x_seg, e] * seg_dt
            compensator_e += float(np.sum(rv / b * (1.0 - np.exp(-b * seg_dt))))
            R *= np.exp(-params.beta * seg_dt)
        result[type_ev_idx] = compensator_e
        t_prev = t_k
        type_ev_idx += 1

    return result


def compute_time_rescaled_residuals(
    events: np.ndarray,
    all_events: np.ndarray,
    event_type: int,
    state_path: np.ndarray,
    session_end: float,
    params: ExpKernelParams,
    burn_in: int = 5
) -> np.ndarray:
    """Compute time-rescaled residuals for one event stream.

    Delta_k = Lambda_e(t_k) - Lambda_e(t_{k-1}) for k = 1, ..., n
    Should be ~ Exp(1) i.i.d. under correct specification.

    burn_in: skip first burn_in residuals to avoid initialization effects.
    """
    if len(events) == 0:
        return np.empty(0)

    compensators = compute_compensator_exp(
        events, all_events, event_type, state_path, session_end, params
    )

    # Delta_k = Lambda(t_k) - Lambda(t_{k-1}); Lambda(t_0) = 0
    deltas = np.diff(compensators, prepend=0.0)
    # Clip to non-negative (numerical noise)
    deltas = np.clip(deltas, 0.0, None)

    return deltas[burn_in:]


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def ks_test_exp1(
    residuals: np.ndarray
) -> Tuple[float, float, bool]:
    """KS test against Exp(1).

    Returns (ks_statistic, p_value, pass_bool)
    pass_bool = True if p_value > SIGNIFICANCE_LEVEL (fail to reject H0)
    """
    if len(residuals) < 2:
        return (np.nan, np.nan, False)
    ks_stat, p_val = stats.kstest(residuals, 'expon', args=(0.0, 1.0))
    return float(ks_stat), float(p_val), bool(p_val > SIGNIFICANCE_LEVEL)


def ljung_box_test(
    residuals: np.ndarray,
    lags: int = LJUNG_BOX_LAGS
) -> Tuple[float, float, bool]:
    """Ljung-Box test for residual independence.

    Returns (lb_statistic, p_value, pass_bool)
    pass_bool = True if p_value > SIGNIFICANCE_LEVEL
    """
    if len(residuals) < lags + 2:
        return (np.nan, np.nan, False)
    lb_result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    lb_stat   = float(lb_result['lb_stat'].iloc[-1])
    p_val     = float(lb_result['lb_pvalue'].iloc[-1])
    return lb_stat, p_val, bool(p_val > SIGNIFICANCE_LEVEL)


def wasserstein_distance_exp1(
    residuals: np.ndarray
) -> float:
    """Compute Wasserstein-1 distance between residuals and Exp(1).

    Uses scipy.stats.wasserstein_distance against samples from Exp(1).
    """
    if len(residuals) == 0:
        return np.nan
    rng     = np.random.default_rng(0)
    samples = rng.exponential(scale=1.0, size=len(residuals))
    return float(stats.wasserstein_distance(residuals, samples))


# ---------------------------------------------------------------------------
# Session-level diagnostics
# ---------------------------------------------------------------------------

def compute_session_residual_diagnostics(
    h_events: np.ndarray,
    q_events: np.ndarray,
    state_path: np.ndarray,
    session_end: float,
    params: ExpKernelParams,
    session_id: str
) -> Dict:
    """Compute all residual diagnostics for a session.

    Computes diagnostics for 4 residual streams split by regime:
      H-Normal, H-Stressed, Q-Normal, Q-Stressed

    Returns dict with session_id plus per-stream diagnostic values.
    """
    # Build sorted combined event array
    h_col  = np.stack([h_events, np.zeros(len(h_events))], axis=1) if len(h_events) else np.empty((0, 2))
    q_col  = np.stack([q_events, np.ones(len(q_events))],  axis=1) if len(q_events) else np.empty((0, 2))
    all_ev = np.vstack([h_col, q_col]) if (len(h_events) + len(q_events)) > 0 else np.empty((0, 2))
    if len(all_ev) > 0:
        order  = np.argsort(all_ev[:, 0], kind='stable')
        all_ev = all_ev[order]

    diag = {'session_id': session_id}

    stream_specs = [
        ('H', 0, 0, 'h_normal'),
        ('H', 0, 1, 'h_stressed'),
        ('Q', 1, 0, 'q_normal'),
        ('Q', 1, 1, 'q_stressed'),
    ]

    for label_prefix, event_type, target_state, key_prefix in stream_specs:
        # Events of this type that occurred in target_state
        if event_type == 0:
            type_events = h_events
        else:
            type_events = q_events

        if len(type_events) == 0:
            for suffix in ('ks_stat', 'ks_pval', 'ks_pass', 'lb_stat', 'lb_pval', 'lb_pass', 'w1_dist'):
                diag[f'{key_prefix}_{suffix}'] = np.nan if suffix not in ('ks_pass', 'lb_pass') else False
            continue

        # Compute full residuals for this event type
        full_residuals = compute_time_rescaled_residuals(
            type_events, all_ev, event_type, state_path, session_end, params, burn_in=5
        )

        # Identify which full_residuals correspond to target_state events
        # The residuals are indexed over events[burn_in:]
        # Determine the state at each event time (after burn_in)
        burn_in   = 5
        ev_subset = type_events[burn_in:] if len(type_events) > burn_in else np.empty(0)
        state_mask = np.array(
            [_get_state_at_time(t, state_path) == target_state for t in ev_subset]
        )

        if state_mask.sum() < 2:
            for suffix in ('ks_stat', 'ks_pval', 'ks_pass', 'lb_stat', 'lb_pval', 'lb_pass', 'w1_dist'):
                diag[f'{key_prefix}_{suffix}'] = np.nan if suffix not in ('ks_pass', 'lb_pass') else False
            continue

        regime_residuals = full_residuals[state_mask]

        ks_stat, ks_pval, ks_pass = ks_test_exp1(regime_residuals)
        lb_stat, lb_pval, lb_pass = ljung_box_test(regime_residuals)
        w1_dist                   = wasserstein_distance_exp1(regime_residuals)

        diag[f'{key_prefix}_ks_stat']  = ks_stat
        diag[f'{key_prefix}_ks_pval']  = ks_pval
        diag[f'{key_prefix}_ks_pass']  = ks_pass
        diag[f'{key_prefix}_lb_stat']  = lb_stat
        diag[f'{key_prefix}_lb_pval']  = lb_pval
        diag[f'{key_prefix}_lb_pass']  = lb_pass
        diag[f'{key_prefix}_w1_dist']  = w1_dist

    return diag
