import logging
import numpy as np
from scipy.optimize import minimize, Bounds
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from src.sdhp_models import (
    KernelType, ExpKernelParams, PowerLawKernelParams, N_STATES,
    initialize_exp_params, initialize_powerlaw_params,
    params_to_vector_exp, vector_to_params_exp,
    params_to_vector_powerlaw, vector_to_params_powerlaw,
    compute_spectral_radius_penalty,
)
from src.likelihood_exp import compute_exp_log_likelihood_and_gradient
from src.likelihood_powerlaw import (
    compute_powerlaw_log_likelihood_and_gradient,
    _powerlaw_spectral_radius_penalty,
    DEFAULT_DELTA_MAX,
)

log = logging.getLogger(__name__)

MAX_ITER       = 1000
MAX_FUNC_EVAL  = 5000
N_RANDOM_STARTS = 3

_N_COMPONENTS_MAP = {
    KernelType.EXP1: 1,
    KernelType.EXP2: 2,
    KernelType.EXP3: 3,
}


@dataclass
class FitResult:
    session_id: str
    kernel_type: KernelType
    converged: bool
    termination_reason: str
    n_iterations: int
    n_func_evals: int
    neg_log_lik: float       # final penalized negative log-likelihood
    nll_per_event: float     # NLL per event (unpenalized)
    n_events_total: int
    params_exp: Optional[ExpKernelParams] = None
    params_powerlaw: Optional[PowerLawKernelParams] = None
    param_vector: Optional[np.ndarray] = None
    penalty_value: float = 0.0
    month: str = ''
    delivery_hour: int = 0
    is_weekend: bool = False


# ---------------------------------------------------------------------------
# Bounds construction
# ---------------------------------------------------------------------------

def _build_bounds_exp(
    n_components: int,
    n_states: int = N_STATES,
    n_event_types: int = 2
) -> Bounds:
    """Build L-BFGS-B bounds for exponential kernel in log-space.

    mu     in [1e-4, 10.0]   -> log-space: [-9.21, 2.30]
    alpha  in [1e-4, 500.0]  -> log-space: [-9.21, 6.21]
    beta   in [1e-4, 1000.0] -> log-space: [-9.21, 6.91]
    """
    n_mu  = n_states * n_event_types
    n_ab  = n_states * n_event_types * n_event_types * n_components

    lo_mu    = np.full(n_mu,  np.log(1e-4))
    hi_mu    = np.full(n_mu,  np.log(10.0))
    lo_alpha = np.full(n_ab,  np.log(1e-4))
    hi_alpha = np.full(n_ab,  np.log(500.0))
    lo_beta  = np.full(n_ab,  np.log(1e-4))
    hi_beta  = np.full(n_ab,  np.log(1000.0))

    lb = np.concatenate([lo_mu, lo_alpha, lo_beta])
    ub = np.concatenate([hi_mu, hi_alpha, hi_beta])
    return Bounds(lb, ub)


def _build_bounds_powerlaw(
    n_states: int = N_STATES,
    n_event_types: int = 2
) -> Bounds:
    """Build L-BFGS-B bounds for power-law kernel in log-space.

    mu     in [1e-4, 10.0]   -> log(mu)
    alpha  in [1e-4, 500.0]  -> log(alpha)
    eta-1  in [0.01, 10.0]   -> log(eta-1)  i.e. eta in [1.01, 11.0]
    c      in [1e-3, 3600.0] -> log(c)
    """
    n_mu  = n_states * n_event_types
    n_mat = n_states * n_event_types * n_event_types

    lo_mu     = np.full(n_mu,  np.log(1e-4))
    hi_mu     = np.full(n_mu,  np.log(10.0))
    lo_alpha  = np.full(n_mat, np.log(1e-4))
    hi_alpha  = np.full(n_mat, np.log(500.0))
    lo_eta    = np.full(n_mat, np.log(0.01))   # log(eta-1) lower
    hi_eta    = np.full(n_mat, np.log(10.0))   # log(eta-1) upper
    lo_c      = np.full(n_mat, np.log(1e-3))
    hi_c      = np.full(n_mat, np.log(3600.0))

    lb = np.concatenate([lo_mu, lo_alpha, lo_eta, lo_c])
    ub = np.concatenate([hi_mu, hi_alpha, hi_eta, hi_c])
    return Bounds(lb, ub)


# ---------------------------------------------------------------------------
# Single-session estimation
# ---------------------------------------------------------------------------

def _message_str(msg) -> str:
    """Decode scipy result.message to plain string."""
    if isinstance(msg, bytes):
        return msg.decode('utf-8', errors='replace')
    return str(msg)


def _converged(msg: str) -> bool:
    return 'CONVERGENCE' in msg.upper() or 'successfully' in msg.lower()


def estimate_session(
    session_id: str,
    h_events: np.ndarray,
    q_events: np.ndarray,
    state_path: np.ndarray,
    session_end: float,
    kernel_type: KernelType,
    n_starts: int = N_RANDOM_STARTS,
    seed: int = 42,
    month: str = '',
    delivery_hour: int = 0,
    is_weekend: bool = False,
    max_iter: int = MAX_ITER,
    max_func_eval: int = MAX_FUNC_EVAL,
) -> FitResult:
    """Estimate SDHP for one session with given kernel type.

    Uses L-BFGS-B with multiple random starts.
    Returns best result across starts (highest log-likelihood).

    Convergence is classified as successful if L-BFGS-B terminates
    by first-order optimality (message contains 'CONVERGENCE') or
    relative reduction criterion.
    """
    n_events_total = len(h_events) + len(q_events)
    is_powerlaw    = kernel_type == KernelType.POWERLAW

    if is_powerlaw:
        bounds = _build_bounds_powerlaw()
    else:
        n_comp = _N_COMPONENTS_MAP[kernel_type]
        bounds = _build_bounds_exp(n_comp)

    # Empirical event rates (events/second) for initialization
    h_rate = len(h_events) / session_end if session_end > 0 else 1e-4
    q_rate = len(q_events) / session_end if session_end > 0 else 1e-4

    best_neg_lik = np.inf
    best_result  = None

    master_rng = np.random.default_rng(seed)

    for start_idx in range(n_starts):
        rng = np.random.default_rng(master_rng.integers(0, 2 ** 31))

        try:
            if is_powerlaw:
                params0 = initialize_powerlaw_params(h_rate, q_rate, rng=rng)
                theta0  = params_to_vector_powerlaw(params0)

                def objective(th):
                    return compute_powerlaw_log_likelihood_and_gradient(
                        th, h_events, q_events, state_path,
                        session_end, DEFAULT_DELTA_MAX, include_penalty=True
                    )
            else:
                params0 = initialize_exp_params(kernel_type, h_rate, q_rate, rng=rng)
                theta0  = params_to_vector_exp(params0)

                def objective(th, _nc=n_comp, _kt=kernel_type):
                    return compute_exp_log_likelihood_and_gradient(
                        th, _nc, h_events, q_events, state_path,
                        session_end, _kt, include_penalty=True
                    )

            # Clip initial vector to bounds before optimization
            theta0 = np.clip(theta0, bounds.lb, bounds.ub)

            result = minimize(
                fun=objective,
                x0=theta0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={
                    'maxiter': max_iter,
                    'maxfun':  max_func_eval,
                    'eps':     1e-6,
                    'ftol':    1e-9,
                    'gtol':    1e-6,
                },
            )

            if np.isfinite(result.fun) and result.fun < best_neg_lik:
                best_neg_lik = result.fun
                best_result  = result

        except Exception:
            continue  # skip failed starts

    if best_result is None:
        # All starts failed; return a sentinel FitResult
        return FitResult(
            session_id=session_id,
            kernel_type=kernel_type,
            converged=False,
            termination_reason='all_starts_failed',
            n_iterations=0,
            n_func_evals=0,
            neg_log_lik=np.nan,
            nll_per_event=np.nan,
            n_events_total=n_events_total,
            month=month,
            delivery_hour=delivery_hour,
            is_weekend=is_weekend,
        )

    msg = _message_str(best_result.message)

    # Reconstruct best params
    if is_powerlaw:
        params_pl  = vector_to_params_powerlaw(best_result.x)
        penalty    = _powerlaw_spectral_radius_penalty(params_pl, DEFAULT_DELTA_MAX)
        unpen_nll  = best_neg_lik - penalty
        nll_per_ev = unpen_nll / max(n_events_total, 1)
        return FitResult(
            session_id=session_id,
            kernel_type=kernel_type,
            converged=_converged(msg),
            termination_reason=msg,
            n_iterations=best_result.nit,
            n_func_evals=best_result.nfev,
            neg_log_lik=best_neg_lik,
            nll_per_event=nll_per_ev,
            n_events_total=n_events_total,
            params_powerlaw=params_pl,
            param_vector=best_result.x.copy(),
            penalty_value=penalty,
            month=month,
            delivery_hour=delivery_hour,
            is_weekend=is_weekend,
        )
    else:
        params_exp = vector_to_params_exp(best_result.x, n_comp, kernel_type)
        penalty    = compute_spectral_radius_penalty(params_exp)
        unpen_nll  = best_neg_lik - penalty
        nll_per_ev = unpen_nll / max(n_events_total, 1)
        return FitResult(
            session_id=session_id,
            kernel_type=kernel_type,
            converged=_converged(msg),
            termination_reason=msg,
            n_iterations=best_result.nit,
            n_func_evals=best_result.nfev,
            neg_log_lik=best_neg_lik,
            nll_per_event=nll_per_ev,
            n_events_total=n_events_total,
            params_exp=params_exp,
            param_vector=best_result.x.copy(),
            penalty_value=penalty,
            month=month,
            delivery_hour=delivery_hour,
            is_weekend=is_weekend,
        )


# ---------------------------------------------------------------------------
# Batch estimation
# ---------------------------------------------------------------------------

def estimate_all_sessions(
    sessions_data: List[Dict],
    kernel_types: List[KernelType] = None,
    n_starts: int = N_RANDOM_STARTS,
    seed: int = 42,
    max_iter: int = MAX_ITER,
    max_func_eval: int = MAX_FUNC_EVAL,
) -> List[FitResult]:
    """Estimate SDHP for all sessions and all kernel types.

    sessions_data: list of dicts with keys:
      session_id, h_events, q_events, state_path, session_end,
      month, delivery_hour, is_weekend

    Returns list of FitResult objects.
    """
    if kernel_types is None:
        kernel_types = list(KernelType)

    results = []
    master_rng = np.random.default_rng(seed)
    n_sess = len(sessions_data)
    n_kt = len(kernel_types)
    total = n_sess * n_kt

    for si, sess in enumerate(sessions_data):
        for ki, kt in enumerate(kernel_types):
            idx = si * n_kt + ki + 1
            n_h = len(sess['h_events'])
            n_q = len(sess['q_events'])
            log.info(
                f"[{idx}/{total}] session={sess['session_id']} "
                f"kernel={kt.value} events H={n_h} Q={n_q}"
            )
            sess_seed = int(master_rng.integers(0, 2 ** 31))
            fit = estimate_session(
                session_id   =sess['session_id'],
                h_events     =sess['h_events'],
                q_events     =sess['q_events'],
                state_path   =sess['state_path'],
                session_end  =sess['session_end'],
                kernel_type  =kt,
                n_starts     =n_starts,
                seed         =sess_seed,
                month        =sess.get('month', ''),
                delivery_hour=sess.get('delivery_hour', 0),
                is_weekend   =sess.get('is_weekend', False),
                max_iter     =max_iter,
                max_func_eval=max_func_eval,
            )
            conv_str = "CONVERGED" if fit.converged else "not-converged"
            log.info(
                f"  -> {conv_str}  NLL/ev={fit.nll_per_event:.4f}  "
                f"penalty={fit.penalty_value:.4f}  msg={fit.termination_reason!r}"
            )
            results.append(fit)

    return results
