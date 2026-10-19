import numpy as np
from typing import Tuple
import numba
from numba import njit, float64, int64
from src.sdhp_models import (
    ExpKernelParams, N_STATES, SPECTRAL_RADIUS_TARGET, PENALTY_WEIGHT,
    compute_branching_matrix, compute_spectral_radius,
    compute_spectral_radius_penalty,
    params_to_vector_exp, vector_to_params_exp,
)

_MIN_INTENSITY = 1e-10  # floor for numerical safety


def _get_state_at_time(t: float, state_path: np.ndarray) -> int:
    """Get state S(t^-) via binary search on state_path[:, 0]."""
    times = state_path[:, 0]
    idx = int(np.searchsorted(times, t, side='right')) - 1
    return int(state_path[max(idx, 0), 1])


def _build_merged_timeline(
    h_events: np.ndarray,
    q_events: np.ndarray,
    state_path: np.ndarray,
    session_end: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge event times with state-change times into a single sorted timeline.

    Returns
    -------
    times : float64 array of all milestone times (events + state changes)
    types : int array   −1 = state-change, 0 = H event, 1 = Q event
    states: int array   state active *before* this milestone (i.e. S(t^-))
    """
    sp_times  = state_path[:, 0]
    sp_states = state_path[:, 1].astype(int)

    parts_t, parts_y = [], []
    if len(h_events):
        parts_t.append(h_events);          parts_y.append(np.zeros(len(h_events), int))
    if len(q_events):
        parts_t.append(q_events);          parts_y.append(np.ones(len(q_events),  int))
    # State-change milestones (skip the first which is session_start)
    if len(sp_times) > 1:
        sc_t = sp_times[1:];              sc_y = np.full(len(sc_t), -1, int)
        parts_t.append(sc_t);             parts_y.append(sc_y)

    if not parts_t:
        return np.array([session_end]), np.array([-1]), np.array([sp_states[0]])

    all_t = np.concatenate(parts_t)
    all_y = np.concatenate(parts_y)
    order = np.argsort(all_t, kind='stable')
    all_t = all_t[order]
    all_y = all_y[order]

    # State at each milestone (left-continuous: S(t^-))
    sc_idx = np.searchsorted(sp_times, all_t, side='right') - 1
    sc_idx = np.clip(sc_idx, 0, len(sp_states) - 1)
    all_s  = sp_states[sc_idx]

    return all_t, all_y, all_s


# ---------------------------------------------------------------------------
# Numba-JIT core: LL + gradient in one O(N) pass
# Signature: all arrays are flat float64; indexing done manually.
# Shapes: mu(N_STATES,2), alpha(N_STATES,2,2,C), beta(N_STATES,2,2,C)
# R,W:   (N_STATES,2,2,C)  — flattened to 1-D for numba
# Gradient arrays: same shapes, also flattened.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _exp_ll_grad_jit(
    all_t: np.ndarray,   # (M,) merged times
    all_y: np.ndarray,   # (M,) types: -1=state_change, 0=H, 1=Q
    all_s: np.ndarray,   # (M,) state at t^- for each milestone
    session_end: float,
    mu: np.ndarray,      # (NS, 2)   NS=N_STATES
    alpha: np.ndarray,   # (NS, 2, 2, C)
    beta: np.ndarray,    # (NS, 2, 2, C)
    n_states: int,
    n_comp: int,
    min_lam: float,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (log_lik, d_mu, d_alpha, d_beta) – no penalty, no negative flip."""
    NS = n_states
    C  = n_comp

    # Flat recursive sums: index = x*2*2*C + ep*2*C + e*C + r
    R  = np.zeros(NS * 2 * 2 * C)
    W  = np.zeros(NS * 2 * 2 * C)

    d_mu_score    = np.zeros(NS * 2)
    d_alpha_score = np.zeros(NS * 2 * 2 * C)
    d_beta_score  = np.zeros(NS * 2 * 2 * C)
    d_mu_comp     = np.zeros(NS * 2)
    d_alpha_comp  = np.zeros(NS * 2 * 2 * C)
    d_beta_comp   = np.zeros(NS * 2 * 2 * C)
    compensator   = np.zeros(2)

    EPS = 1e-30
    log_lik_sum = 0.0
    t_prev = 0.0
    M = len(all_t)

    for m in range(M):
        t_k = all_t[m]
        y_k = all_y[m]   # -1=state_change, 0=H event, 1=Q event
        x_k = all_s[m]   # state S(t_k^-)

        # --- Segment [t_prev, t_k]: compensator for ACTIVE x_k; decay ALL ---
        dt = t_k - t_prev
        if dt > EPS:
            # Step 1: compensator + grad for active state x_k only
            for e in range(2):
                compensator[e] += mu[x_k, e] * dt
                d_mu_comp[x_k * 2 + e] += mu[x_k, e] * dt
                for ep in range(2):
                    for r in range(C):
                        i4 = x_k * 4 * C + ep * 2 * C + e * C + r
                        b  = beta[x_k, ep, e, r]
                        a  = alpha[x_k, ep, e, r]
                        sb = b if b > EPS else EPS
                        sa = a if a > EPS else EPS
                        Rv = R[i4]; Wv = W[i4]
                        S0 = Rv / sa
                        eb = np.exp(-b * dt)
                        EM1 = 1.0 - eb
                        EM2 = 1.0 - (1.0 + b * dt) * eb
                        compensator[e] += Rv / sb * EM1
                        d_alpha_comp[i4] += a * S0 / sb * EM1
                        d_beta_comp[i4]  += (-a * b) * (Wv / sb * EM1 + S0 / (sb * sb) * EM2)
            # Step 2: decay ALL states' R and W
            for x in range(NS):
                for ep in range(2):
                    for e in range(2):
                        for r in range(C):
                            i4 = x * 4 * C + ep * 2 * C + e * C + r
                            b  = beta[x, ep, e, r]
                            sa = alpha[x, ep, e, r]
                            sa = sa if sa > EPS else EPS
                            Rv = R[i4]; Wv = W[i4]
                            S0 = Rv / sa
                            eb = np.exp(-b * dt)
                            W[i4] = eb * (Wv + dt * S0)
                            R[i4] = eb * Rv

        # --- If event: intensity log-term + score gradient + impulse ---
        if y_k >= 0:
            e_k = y_k
            lam = mu[x_k, e_k]
            for ep in range(2):
                for r in range(C):
                    lam += R[x_k * 4 * C + ep * 2 * C + e_k * C + r]
            if lam < min_lam:
                lam = min_lam
            inv = 1.0 / lam
            log_lik_sum += np.log(lam)

            d_mu_score[x_k * 2 + e_k] += mu[x_k, e_k] * inv
            for ep in range(2):
                for r in range(C):
                    i4 = x_k * 4 * C + ep * 2 * C + e_k * C + r
                    Rv = R[i4]; Wv = W[i4]
                    a  = alpha[x_k, ep, e_k, r]
                    b  = beta[x_k, ep, e_k, r]
                    d_alpha_score[i4] += Rv * inv
                    d_beta_score[i4]  += (-a * b * Wv) * inv

            # Impulse: ALL states, source=e_k
            for x in range(NS):
                for e in range(2):
                    for r in range(C):
                        R[x * 4 * C + e_k * 2 * C + e * C + r] += alpha[x, e_k, e, r]

        t_prev = t_k

    # --- Final segment [t_prev, session_end] under last active state ---
    dt = session_end - t_prev
    if dt > EPS:
        x_last = all_s[M - 1] if M > 0 else 0
        for e in range(2):
            compensator[e] += mu[x_last, e] * dt
            d_mu_comp[x_last * 2 + e] += mu[x_last, e] * dt
            for ep in range(2):
                for r in range(C):
                    i4 = x_last * 4 * C + ep * 2 * C + e * C + r
                    b  = beta[x_last, ep, e, r]
                    a  = alpha[x_last, ep, e, r]
                    sb = b if b > EPS else EPS
                    sa = a if a > EPS else EPS
                    Rv = R[i4]; Wv = W[i4]
                    S0 = Rv / sa
                    eb = np.exp(-b * dt)
                    EM1 = 1.0 - eb
                    EM2 = 1.0 - (1.0 + b * dt) * eb
                    compensator[e] += Rv / sb * EM1
                    d_alpha_comp[i4] += a * S0 / sb * EM1
                    d_beta_comp[i4]  += (-a * b) * (Wv / sb * EM1 + S0 / (sb * sb) * EM2)
        for x in range(NS):
            for ep in range(2):
                for e in range(2):
                    for r in range(C):
                        i4 = x * 4 * C + ep * 2 * C + e * C + r
                        b  = beta[x, ep, e, r]
                        sa = alpha[x, ep, e, r]
                        sa = sa if sa > EPS else EPS
                        Rv = R[i4]; Wv = W[i4]
                        S0 = Rv / sa
                        eb = np.exp(-b * dt)
                        W[i4] = eb * (Wv + dt * S0)
                        R[i4] = eb * Rv

    log_lik = log_lik_sum - compensator[0] - compensator[1]
    d_mu    = -d_mu_score    + d_mu_comp
    d_alpha = -d_alpha_score + d_alpha_comp
    d_beta  = -d_beta_score  + d_beta_comp
    return log_lik, d_mu, d_alpha, d_beta


def compute_exp_log_likelihood(
    h_events: np.ndarray,
    q_events: np.ndarray,
    state_path: np.ndarray,
    session_start: float,
    session_end: float,
    params: ExpKernelParams,
    include_penalty: bool = True,
) -> float:
    """Compute penalized log-likelihood using the JIT-accelerated kernel.

    Delegates heavy computation to ``_exp_ll_grad_jit``; only LL is returned.
    """
    all_t, all_y, all_s = _build_merged_timeline(h_events, q_events, state_path, session_end)
    ll, _, _, _ = _exp_ll_grad_jit(
        all_t, all_y.astype(np.int64), all_s.astype(np.int64),
        float(session_end),
        params.mu.astype(np.float64),
        params.alpha.astype(np.float64),
        params.beta.astype(np.float64),
        int(N_STATES), int(params.n_components), float(_MIN_INTENSITY),
    )
    if include_penalty:
        ll -= compute_spectral_radius_penalty(params)
    return ll


def _spectral_radius_gradient_2x2(G: np.ndarray) -> np.ndarray:
    """∂ρ/∂G for a 2×2 positive matrix, where ρ is the spectral radius.

    Using the closed-form eigenvalue formula:
      ρ = (trace + sqrt((G_HH−G_QQ)²+4·G_HQ·G_QH)) / 2
    Returns a (2,2) array of partial derivatives ∂ρ/∂G[e',e].
    Falls back to numerical FD for nearly degenerate discriminant.
    """
    G_HH, G_HQ = G[0, 0], G[0, 1]
    G_QH, G_QQ = G[1, 0], G[1, 1]
    disc = (G_HH - G_QQ) ** 2 + 4.0 * G_HQ * G_QH

    if disc < 1e-16:
        # Numerical fallback for degenerate case
        rho0 = np.max(np.abs(np.linalg.eigvals(G)))
        eps  = 1e-7
        dG   = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                Gp = G.copy(); Gp[i, j] += eps
                dG[i, j] = (np.max(np.abs(np.linalg.eigvals(Gp))) - rho0) / eps
        return dG

    sq = np.sqrt(disc)
    return np.array([
        [(1.0 + (G_HH - G_QQ) / sq) / 2.0,  G_QH / sq],
        [G_HQ / sq,  (1.0 - (G_HH - G_QQ) / sq) / 2.0],
    ])  # shape (2,2): rows=source e', cols=target e


def _penalty_gradient_log_space(
    params: ExpKernelParams,
    target_rho: float = SPECTRAL_RADIUS_TARGET,
    omega: float = PENALTY_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Analytical ∂P/∂(log α) and ∂P/∂(log β) for the spectral-radius penalty.

    P = Σ_x ω·max(0, ρ_x−ρ*)²
    G_x[e',e] = Σ_r α[x,e',e,r]/β[x,e',e,r]
    Chain rule in log-space:
      ∂P/∂(log α[x,e',e,r]) = +2ω·excess_x · ∂ρ_x/∂G[e',e] · α/β
      ∂P/∂(log β[x,e',e,r]) = −2ω·excess_x · ∂ρ_x/∂G[e',e] · α/β
    """
    _EPS = 1e-30
    d_alpha = np.zeros_like(params.alpha)   # (N_STATES, 2, 2, n_comp)
    d_beta  = np.zeros_like(params.beta)

    for x in range(N_STATES):
        G_x    = compute_branching_matrix(params, x)
        rho_x  = compute_spectral_radius(G_x)
        excess = rho_x - target_rho
        if excess <= 0:
            continue

        coeff = 2.0 * omega * excess
        drho  = _spectral_radius_gradient_2x2(G_x)           # (2,2)
        ratio = params.alpha[x] / np.maximum(params.beta[x], _EPS)  # (2,2,n_comp)

        contrib = coeff * drho[:, :, np.newaxis] * ratio      # (2,2,n_comp)
        d_alpha[x] += contrib
        d_beta[x]  -= contrib

    return d_alpha, d_beta


def compute_exp_log_likelihood_and_gradient(
    theta: np.ndarray,
    n_components: int,
    h_events: np.ndarray,
    q_events: np.ndarray,
    state_path: np.ndarray,
    session_end: float,
    kernel_type,
    include_penalty: bool = True
) -> Tuple[float, np.ndarray]:
    """Compute negative penalized log-likelihood and ANALYTICAL gradient.

    Replaces the O(n_params × n_events) finite-difference approach with a
    single O(n_events) forward pass that simultaneously accumulates the
    log-likelihood, the compensator, and all gradient terms.

    Two sets of recursive sums are maintained beyond the usual R:
      R[x,e',e,r]  = α·Σ exp(−β·age)   (intensity excitation)
      W[x,e',e,r]  = Σ age·exp(−β·age) (required for ∂/∂β)

    Segment update rules (time step Δ, ALL states):
      S    = R / α                        (per-unit excitation)
      W  ← exp(−βΔ)·(W + Δ·S)
      R  ← R·exp(−βΔ)

    Event update (source type e_k, ALL states x):
      R[x,e_k,:,:] += α[x,e_k,:,:]
      (W unchanged — new event contributes zero at age 0)

    Score terms at event k (state x_k, type e_k, intensity λ):
      ∂log λ/∂(log μ[x_k,e_k])           = μ/λ
      ∂log λ/∂(log α[x_k,e',e_k,r])      = R[x_k,e',e_k,r]/λ
      ∂log λ/∂(log β[x_k,e',e_k,r])      = −α·β·W[x_k,e',e_k,r]/λ

    Compensator gradient per segment (state x_seg, length Δ):
      ∂C/∂(log μ[x_seg,e])               = μ·Δ
      ∂C/∂(log α[x_seg,e',e,r])          = α·(S₀/β)·EM1
      ∂C/∂(log β[x_seg,e',e,r])          = −α·β·(W₀/β·EM1 + S₀/β²·EM2)
    where EM1=1−e^{−βΔ}, EM2=1−(1+βΔ)e^{−βΔ}, S₀=R/α, W₀=W at segment start.

    Returns
    -------
    (neg_log_lik, gradient_of_neg_log_lik)  both as float / 1-D ndarray.
    """
    params = vector_to_params_exp(theta, n_components, kernel_type)

    all_t, all_y, all_s = _build_merged_timeline(
        h_events, q_events, state_path, session_end
    )
    ll, d_mu_flat, d_alpha_flat, d_beta_flat = _exp_ll_grad_jit(
        all_t, all_y.astype(np.int64), all_s.astype(np.int64),
        float(session_end),
        params.mu.astype(np.float64),
        params.alpha.astype(np.float64),
        params.beta.astype(np.float64),
        int(N_STATES), int(n_components), float(_MIN_INTENSITY),
    )

    if include_penalty:
        ll -= compute_spectral_radius_penalty(params)
    neg_ll = -ll

    # Reshape flat gradient arrays back to parameter shapes
    grad_mu    = d_mu_flat.reshape(N_STATES, 2)
    grad_alpha = d_alpha_flat.reshape(N_STATES, 2, 2, n_components)
    grad_beta  = d_beta_flat.reshape(N_STATES, 2, 2, n_components)

    if include_penalty:
        pen_a, pen_b = _penalty_gradient_log_space(params)
        grad_alpha += pen_a
        grad_beta  += pen_b

    grad = np.concatenate([grad_mu.ravel(), grad_alpha.ravel(), grad_beta.ravel()])
    return neg_ll, grad
