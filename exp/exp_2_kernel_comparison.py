"""
Experiment 2: Contract-by-contract kernel comparison for SDHP.

Estimates four SDHP specifications (1-Exp, 2-Exp, 3-Exp, Power-law) for each
retained session and evaluates fit using: penalized log-likelihood, KS/Ljung-Box
non-rejection rates, Wasserstein-1 distance to Exp(1), NLL per event, pairwise
likelihood wins, and convergence shares.

Expected outcomes (pooled):
  KS pass rate:  1-Exp=0.856, 2-Exp=0.922, 3-Exp=0.926, Power-law=0.900
  LB pass rate:  1-Exp=0.221, 2-Exp=0.267, 3-Exp=0.268, Power-law=0.144
  Median W1:     1-Exp=0.0834, 2-Exp=0.0661, 3-Exp=0.0651, Power-law=0.0963
  NLL/event:     1-Exp=1.2708, 2-Exp=1.2067, 3-Exp=1.1950, Power-law=1.2480
  Convergence:   1-Exp=99.5%, 2-Exp=98.1%, 3-Exp=95.6%, Power-law=82.4%
"""
import sys
import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.sdhp_models import KernelType
from src.optimization import estimate_session, estimate_all_sessions, FitResult
from src.residuals import compute_session_residual_diagnostics
from src.diagnostics import (
    build_results_dataframe, compute_convergence_shares, compute_nll_summary,
    compute_pairwise_wins, aggregate_residual_diagnostics, kernel_comparison_table,
    compute_month_breakdown, PAPER_REFERENCE, KERNEL_LABELS,
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

KERNEL_TYPES = [KernelType.EXP1, KernelType.EXP2, KernelType.EXP3, KernelType.POWERLAW]
N_STARTS = 3
SEED = 42


def prepare_session_data(sessions: list) -> list:
    """Convert Session objects to estimation-ready dicts."""
    data = []
    for s in sessions:
        if s.h_events is None or s.q_events is None or s.state_path is None:
            continue
        h_arr = np.array(s.h_events, dtype=float)
        q_arr = np.array(s.q_events, dtype=float)
        session_end = (s.session_end - s.session_start).total_seconds()

        # Build state_path array from session.state_path DataFrame
        if isinstance(s.state_path, pd.DataFrame):
            sp_arr = s.state_path[['time_sec', 'regime']].values.astype(float)
        else:
            sp_arr = np.array([[0.0, 0]])

        if len(h_arr) < 2 or len(q_arr) < 2:
            continue

        data.append({
            'session_id': s.session_id,
            'h_events': h_arr,
            'q_events': q_arr,
            'state_path': sp_arr,
            'session_end': session_end,
            'month': s.month,
            'delivery_hour': s.delivery_hour,
            'is_weekend': s.is_weekend,
        })
    return data


def run_experiment_2(
    sessions: list = None,
    exp1_pkl: str = None,
    n_starts: int = N_STARTS,
    seed: int = SEED,
    max_sessions: int = None,
    save_results: bool = True,
) -> dict:
    """Run Experiment 2: SDHP kernel comparison.

    Parameters
    ----------
    sessions : list, optional
        Session objects from Experiment 1. If None, loads from exp1_pkl.
    exp1_pkl : str, optional
        Path to pickled Experiment 1 results.
    n_starts : int
        Number of random starts per session per kernel.
    seed : int
        Random seed.
    max_sessions : int, optional
        Limit number of sessions (for fast testing).
    save_results : bool
        Whether to save FitResult list to disk.

    Returns
    -------
    dict with keys: fit_results, results_df, comparison_table, pairwise_wins.
    """
    log.info("=== Experiment 2: SDHP Kernel Comparison ===")

    # Load sessions from Experiment 1 if not provided
    if sessions is None:
        if exp1_pkl is None:
            exp1_pkl = os.path.join(RESULTS_DIR, 'exp1_sessions.pkl')
        if os.path.exists(exp1_pkl):
            with open(exp1_pkl, 'rb') as f:
                exp1_data = pickle.load(f)
            sessions = exp1_data.get('sessions', [])
            log.info(f"Loaded {len(sessions)} sessions from {exp1_pkl}")
        else:
            raise FileNotFoundError(
                f"No sessions provided and {exp1_pkl} not found. "
                "Run Experiment 1 first."
            )

    # Prepare data
    session_data = prepare_session_data(sessions)
    log.info(f"Sessions ready for estimation: {len(session_data)}")

    if max_sessions is not None:
        session_data = session_data[:max_sessions]
        log.info(f"Capped to {len(session_data)} sessions for speed")

    # Pre-warm Numba JIT so compilation overhead does not inflate timing.
    log.info("Pre-warming Numba JIT kernels ...")
    from src.likelihood_exp import compute_exp_log_likelihood_and_gradient
    from src.likelihood_powerlaw import _warm_up_nb as _pl_warm_up
    _dummy_h = np.array([1.0, 2.0, 3.0])
    _dummy_q = np.array([1.5, 2.5])
    # state_path: 2-column array (time, state)
    _dummy_sp = np.column_stack([np.array([0.0, 2.0, 3.5, 4.5]),
                                  np.array([0, 1, 0, 1], dtype=np.float64)])
    for _nc in (1, 2, 3):
        _kt = {1: KernelType.EXP1, 2: KernelType.EXP2, 3: KernelType.EXP3}[_nc]
        from src.sdhp_models import initialize_exp_params, params_to_vector_exp
        _p0 = initialize_exp_params(_kt, 0.5, 0.5)
        _th0 = params_to_vector_exp(_p0)
        try:
            compute_exp_log_likelihood_and_gradient(_th0, _nc, _dummy_h, _dummy_q,
                                                    _dummy_sp, 5.0, _kt, False)
        except Exception:
            pass
    _pl_warm_up()  # warm up power-law JIT kernel
    log.info("JIT warm-up complete.")

    # ---- Estimate all sessions × all kernel types ----
    # With Numba JIT (~1.7 ms/call for Exp, ~41 ms/call for PL FD grad):
    #   Exp kernels: 1000 iters × 5000 evals/start × 3 starts → worst-case
    #   ~25 s/session → ~8 min for 19 sessions; typical 1-5 s/session.
    #   Power-law: FD gradient costs 29 inner LL evals per objective call.
    #   300 func-evals × 41 ms = ~12 s/session → ~4 min for 19 sessions.
    EXP_KERNELS = [KernelType.EXP1, KernelType.EXP2, KernelType.EXP3]
    fit_results: list[FitResult] = []

    log.info(f"Estimating 3 Exp kernels × {len(session_data)} sessions (analytical gradient, JIT) ...")
    fit_results += estimate_all_sessions(
        session_data,
        kernel_types=EXP_KERNELS,
        n_starts=n_starts,
        seed=seed,
        max_iter=1000,
        max_func_eval=5000,
    )

    # Power-law: FD gradient, ~41 ms per objective call (29 inner LL evals).
    # Budget: 300 func-evals × 41 ms = ~12 s/session.  Use 1 start.
    log.info(f"Estimating power-law kernel × {len(session_data)} sessions (FD gradient, JIT) ...")
    fit_results += estimate_all_sessions(
        session_data,
        kernel_types=[KernelType.POWERLAW],
        n_starts=1,
        seed=seed,
        max_iter=200,
        max_func_eval=300,
    )
    log.info(f"Total fits completed: {len(fit_results)}")

    # ---- Compute residual diagnostics for all exponential kernel fits ----
    # Power-law fits are skipped: compute_session_residual_diagnostics requires
    # ExpKernelParams; there is no equivalent analytical time-rescaling for PL.
    log.info("Computing residual diagnostics for all Exp-kernel fits ...")
    sd_index = {s['session_id']: s for s in session_data}
    diag_records = []
    for fit in fit_results:
        # Skip unconverged fits and power-law (no ExpKernelParams)
        if not fit.converged or fit.params_exp is None:
            continue
        sd = sd_index.get(fit.session_id)
        if sd is None:
            continue
        try:
            diag = compute_session_residual_diagnostics(
                sd['h_events'], sd['q_events'],
                sd['state_path'], sd['session_end'],
                fit.params_exp, fit.session_id,
            )
            # Tag each record with its kernel so aggregate_residual_diagnostics
            # doesn't need to infer it from a potentially incorrect kernel_map
            diag['kernel'] = KERNEL_LABELS[fit.kernel_type]
            diag_records.append(diag)
        except Exception as exc:
            log.debug(f"Residuals failed for {fit.session_id} ({fit.kernel_type}): {exc}")

    # ---- Aggregate results ----
    results_df = build_results_dataframe(fit_results)
    log.info("\n--- Convergence Shares ---")
    conv = compute_convergence_shares(results_df)
    log.info(conv.to_string())
    log.info("\nPaper targets: 1-Exp=99.5%, 2-Exp=98.1%, 3-Exp=95.6%, PL=82.4%")

    log.info("\n--- NLL per event (converged) ---")
    nll = compute_nll_summary(results_df)
    log.info(nll.to_string())

    log.info("\n--- Pairwise wins vs 3-Exp ---")
    pw = compute_pairwise_wins(results_df)
    log.info(pw.to_string())

    # Aggregate residual diagnostics
    diag_agg = aggregate_residual_diagnostics(diag_records, results_df)
    if not diag_agg.empty:
        log.info("\n--- Residual Diagnostics ---")
        log.info(diag_agg.to_string())

    comp_table = kernel_comparison_table(results_df, diag_agg if not diag_agg.empty else None)
    log.info("\n--- Full Comparison Table ---")
    log.info(comp_table.to_string())

    # ---- Save results ----
    output = {
        'fit_results': fit_results,
        'results_df': results_df,
        'comparison_table': comp_table,
        'pairwise_wins': pw,
        'diag_records': diag_records,
        'diag_agg': diag_agg,
    }
    if save_results:
        pkl_path = os.path.join(RESULTS_DIR, 'exp2_fits.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(output, f)
        log.info(f"Saved fit results to {pkl_path}")

        # Save CSVs
        results_df.to_csv(os.path.join(RESULTS_DIR, 'exp2_results.csv'), index=False)
        comp_table.to_csv(os.path.join(RESULTS_DIR, 'exp2_comparison_table.csv'), index=False)

    # ---- Figures ----
    _plot_nll_boxplot(results_df)
    _plot_convergence_bars(conv)
    _plot_comparison_heatmap(comp_table)

    # ---- Write metrics ----
    _write_exp2_metrics(output)

    return output


def _plot_nll_boxplot(results_df: pd.DataFrame) -> None:
    """Boxplot of NLL per event by kernel (converged only)."""
    conv = results_df[results_df['converged']].copy()
    if conv.empty:
        return
    order = ['1-Exp', '2-Exp', '3-Exp', 'Power-law']
    order = [o for o in order if o in conv['kernel'].unique()]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=conv, x='kernel', y='nll_per_event', order=order,
                palette='Blues', ax=ax, fliersize=2)
    ax.set_xlabel('Kernel', fontsize=13)
    ax.set_ylabel('NLL per event', fontsize=13)
    ax.set_title('Negative Log-Likelihood per Event by Kernel', fontsize=14)
    sns.despine(ax=ax)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp2_nll_boxplot.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved plot: {path}")


def _plot_convergence_bars(conv: pd.Series) -> None:
    """Bar chart of convergence shares."""
    paper = {'1-Exp': 0.995, '2-Exp': 0.981, '3-Exp': 0.956, 'Power-law': 0.824}
    labels = list(conv.index)
    values = conv.values
    paper_vals = [paper.get(l, np.nan) for l in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, values, width=0.35, label='Reproduced', color='steelblue', alpha=0.8)
    ax.bar(x + 0.2, paper_vals, width=0.35, label='Paper target', color='tomato', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Convergence share', fontsize=13)
    ax.set_title('Convergence Shares by Kernel', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp2_convergence.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved plot: {path}")


def _plot_comparison_heatmap(comp_table: pd.DataFrame) -> None:
    """Heatmap of key metrics across kernels."""
    metrics = ['convergence_share', 'nll_median']
    avail = [m for m in metrics if m in comp_table.columns]
    if not avail or 'kernel' not in comp_table.columns:
        return
    hm_data = comp_table.set_index('kernel')[avail]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(hm_data.T, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax,
                linewidths=0.5, cbar=True)
    ax.set_title('Kernel Comparison Metrics', fontsize=14)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp2_comparison_heatmap.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved plot: {path}")


def _write_exp2_metrics(output: dict) -> None:
    """Append Experiment 2 metrics to RESULTS.md."""
    results_df = output['results_df']
    comp_table = output['comparison_table']
    pw = output['pairwise_wins']

    lines = [
        "## Experiment 2: SDHP Kernel Comparison\n\n",
        "### Convergence Shares\n",
    ]
    conv = compute_convergence_shares(results_df)
    for kernel, share in conv.items():
        paper_val = PAPER_REFERENCE.get(kernel, {}).get('conv', float('nan'))
        lines.append(f"- **{kernel}**: {share:.3f}  *(paper: {paper_val:.3f})*\n")

    lines.append("\n### NLL per Event (median, converged fits)\n")
    nll = compute_nll_summary(results_df)
    for kernel in nll.index:
        paper_val = PAPER_REFERENCE.get(kernel, {}).get('nll', float('nan'))
        lines.append(f"- **{kernel}**: {nll.loc[kernel, 'nll_median']:.4f}  *(paper: {paper_val:.4f})*\n")

    if not pw.empty:
        lines.append("\n### Pairwise Win Rates vs 3-Exp Benchmark\n")
        for _, row in pw.iterrows():
            lines.append(f"- **{row['kernel']}** win rate vs 3-Exp: {row.get('win_rate_vs_bm', float('nan')):.3f}\n")

    lines += [
        "\n- **Figures**: exp2_nll_boxplot.png, exp2_convergence.png, exp2_comparison_heatmap.png\n",
        "\n",
    ]

    results_md = os.path.join(RESULTS_DIR, 'RESULTS.md')
    with open(results_md, 'a') as f:
        f.writelines(lines)
    log.info(f"Metrics appended to {results_md}")


if __name__ == '__main__':
    results = run_experiment_2(max_sessions=20)
    print(f"\nExp 2 complete. Total fits: {len(results['fit_results'])}")
