"""
Aggregated diagnostics for model comparison across kernel families.

Computes pooled KS pass rates, Ljung-Box pass rates, Wasserstein-1 medians,
NLL per event, convergence shares, and pairwise likelihood win rates.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.sdhp_models import KernelType
from src.optimization import FitResult


KERNEL_ORDER = [KernelType.EXP1, KernelType.EXP2, KernelType.EXP3, KernelType.POWERLAW]
KERNEL_LABELS = {
    KernelType.EXP1: '1-Exp',
    KernelType.EXP2: '2-Exp',
    KernelType.EXP3: '3-Exp',
    KernelType.POWERLAW: 'Power-law',
}

# Reference values from paper (Table 4 / paper body)
PAPER_REFERENCE = {
    '1-Exp':     {'ks_pass': 0.856, 'lb_pass': 0.221, 'w1': 0.0834, 'nll': 1.2708, 'conv': 0.995},
    '2-Exp':     {'ks_pass': 0.922, 'lb_pass': 0.267, 'w1': 0.0661, 'nll': 1.2067, 'conv': 0.981},
    '3-Exp':     {'ks_pass': 0.926, 'lb_pass': 0.268, 'w1': 0.0651, 'nll': 1.1950, 'conv': 0.956},
    'Power-law': {'ks_pass': 0.900, 'lb_pass': 0.144, 'w1': 0.0963, 'nll': 1.2480, 'conv': 0.824},
}


def build_results_dataframe(fit_results: List[FitResult]) -> pd.DataFrame:
    """Convert list of FitResult to flat DataFrame for analysis.

    One row per (session, kernel_type).
    """
    rows = []
    for r in fit_results:
        rows.append({
            'session_id': r.session_id,
            'kernel': KERNEL_LABELS.get(r.kernel_type, str(r.kernel_type)),
            'converged': r.converged,
            'nll_per_event': r.nll_per_event,
            'neg_log_lik': r.neg_log_lik,
            'penalty': r.penalty_value,
            'n_events': r.n_events_total,
            'month': r.month,
            'delivery_hour': r.delivery_hour,
            'is_weekend': r.is_weekend,
            'termination': r.termination_reason,
        })
    return pd.DataFrame(rows)


def compute_convergence_shares(results_df: pd.DataFrame) -> pd.Series:
    """Compute convergence share per kernel type."""
    return results_df.groupby('kernel')['converged'].mean().rename('convergence_share')


def compute_nll_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute NLL per event summary statistics (converged fits only)."""
    conv = results_df[results_df['converged']]
    return conv.groupby('kernel')['nll_per_event'].agg(['mean', 'median', 'std']).rename(
        columns={'mean': 'nll_mean', 'median': 'nll_median', 'std': 'nll_std'}
    )


def compute_pairwise_wins(
    results_df: pd.DataFrame,
    benchmark: str = '3-Exp',
) -> pd.DataFrame:
    """Compute pairwise NLL win rates vs benchmark kernel.

    A win for kernel K vs benchmark means K has lower NLL per event on the same session.
    Only considers sessions converged for both kernels.
    """
    rows = []
    bm = results_df[results_df['kernel'] == benchmark][['session_id', 'nll_per_event']].rename(
        columns={'nll_per_event': 'nll_bm'}
    )
    for kernel in results_df['kernel'].unique():
        if kernel == benchmark:
            continue
        other = results_df[results_df['kernel'] == kernel][['session_id', 'nll_per_event']].rename(
            columns={'nll_per_event': 'nll_other'}
        )
        merged = bm.merge(other, on='session_id').dropna()
        if len(merged) == 0:
            continue
        win_rate = float((merged['nll_other'] < merged['nll_bm']).mean())
        rows.append({
            'kernel': kernel,
            'vs_benchmark': benchmark,
            'n_sessions': len(merged),
            'win_rate_vs_bm': win_rate,
            'bm_win_rate': 1.0 - win_rate,
        })
    return pd.DataFrame(rows)


def aggregate_residual_diagnostics(
    diag_records: List[Dict],
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge residual diagnostics with fit results and aggregate by kernel.

    diag_records: list of dicts from compute_session_residual_diagnostics.
    Returns DataFrame with columns: kernel, ks_pass_rate, lb_pass_rate, w1_median.
    """
    diag_df = pd.DataFrame(diag_records)
    if diag_df.empty:
        return pd.DataFrame()

    # Only fill missing 'kernel' values from kernel_map; preserve pre-tagged values
    # (exp_2 tags every record with diag['kernel'] = KERNEL_LABELS[fit.kernel_type])
    if 'session_id' in diag_df.columns:
        kernel_map = results_df.groupby('session_id')['kernel'].first().to_dict()
        if 'kernel' not in diag_df.columns:
            diag_df['kernel'] = diag_df['session_id'].map(kernel_map)
        else:
            missing = diag_df['kernel'].isna()
            diag_df.loc[missing, 'kernel'] = diag_df.loc[missing, 'session_id'].map(kernel_map)

    # Identify KS pass and W1 columns (created by residuals module)
    ks_cols = [c for c in diag_df.columns if 'ks_pass' in c]
    lb_cols = [c for c in diag_df.columns if 'lb_pass' in c]
    w1_cols = [c for c in diag_df.columns if 'w1' in c]

    rows = []
    for kernel, grp in diag_df.groupby('kernel'):
        row = {'kernel': kernel}
        if ks_cols:
            row['ks_pass_rate'] = float(grp[ks_cols].values.mean())
        if lb_cols:
            row['lb_pass_rate'] = float(grp[lb_cols].values.mean())
        if w1_cols:
            row['w1_median'] = float(np.nanmedian(grp[w1_cols].values))
        rows.append(row)
    return pd.DataFrame(rows)


def kernel_comparison_table(
    results_df: pd.DataFrame,
    diag_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Produce kernel comparison summary table.

    Columns: kernel, convergence_share, nll_median, ks_pass_rate, lb_pass_rate, w1_median.
    """
    conv = compute_convergence_shares(results_df)
    nll = compute_nll_summary(results_df)
    df = conv.reset_index().merge(nll.reset_index(), on='kernel', how='left')

    if diag_df is not None and not diag_df.empty:
        df = df.merge(diag_df, on='kernel', how='left')

    # Add paper reference values
    ref_rows = []
    for label, vals in PAPER_REFERENCE.items():
        ref_rows.append({'kernel': label, **{f'paper_{k}': v for k, v in vals.items()}})
    ref_df = pd.DataFrame(ref_rows)
    df = df.merge(ref_df, on='kernel', how='left')
    return df


def compute_month_breakdown(
    results_df: pd.DataFrame,
    metric: str = 'nll_per_event',
) -> pd.DataFrame:
    """Summarize a metric by month and kernel (converged fits only)."""
    conv = results_df[results_df['converged']]
    return conv.groupby(['month', 'kernel'])[metric].median().unstack('kernel')
