"""
Experiment 1: Session-level reconstruction and LSI regime classification.

Builds session dataset from synthetic Dutch XBID EPEX SPOT Level 3 data,
reconstructs active limit order books for Hourly and Quarter-Hourly contracts,
computes LSI components (spread, 10-min rolling mid-price volatility, queue
imbalance), normalizes with slow EWMA Z-score (lambda=0.01), constructs LSI,
and classifies Normal/Stressed regimes using the global 75th-percentile threshold.

Validates that the state process S(t) can be reproduced from observable LOB.
Expected outcomes (paper):
  - ~2,915 retained sessions over 4 months
  - Global LSI threshold q_0.75^glob = 0.264
  - Stressed occupancy ~25% globally
"""
import sys
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.synthetic_data import generate_session_dataset, SyntheticConfig
from src.book_reconstruction import reconstruct_session_books
from src.feature_engineering import (
    compute_session_features, compute_global_lsi_threshold,
    LSI_THRESHOLD, EWMA_LAMBDA,
)
from src.session_builder import (
    identify_sessions, build_event_streams, build_state_path,
    filter_retained_sessions, compute_threshold_transportability,
    summarize_sessions_by_period,
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

MONTHS = ['2024-01', '2024-04', '2024-07', '2024-10']
# Synthetic: use 10 sessions/month for fast testing
N_SESSIONS_PER_MONTH = 10
OBS_FREQ_SECONDS = 60.0  # 1-minute observation grid
SEED = 42


def run_experiment_1(
    n_sessions_per_month: int = N_SESSIONS_PER_MONTH,
    obs_freq: float = OBS_FREQ_SECONDS,
    seed: int = SEED,
    save_pickle: bool = True,
) -> dict:
    """Run Experiment 1: LOB reconstruction + LSI regime classification.

    Parameters
    ----------
    n_sessions_per_month : int
        Number of sessions to generate per month (paper target: ~730).
    obs_freq : float
        Observation frequency in seconds for feature grid.
    seed : int
        Random seed for synthetic data generation.
    save_pickle : bool
        Whether to save processed session data to disk.

    Returns
    -------
    dict with keys: sessions, all_lsi, global_threshold, summary_df,
                    transportability_df, period_summary_df.
    """
    log.info("=== Experiment 1: LSI Regime Classification ===")
    log.info(f"Generating synthetic L3 data: {n_sessions_per_month} sessions/month, months={MONTHS}")

    # ---- Step 1-2: Generate and ingest L3 messages ----
    messages = generate_session_dataset(
        months=MONTHS,
        n_sessions_per_month=n_sessions_per_month,
        seed=seed,
    )
    log.info(f"Generated {len(messages):,} L3 messages across {messages['contract_id'].nunique()} contracts")

    # ---- Step 3-5: Identify sessions (Hourly + Q1-Q4 groupings) ----
    sessions = identify_sessions(messages)
    log.info(f"Identified {len(sessions)} candidate sessions")

    # ---- Step 6-7: Reconstruct order books and compute features ----
    all_lsi_values = []
    retained_sessions = []

    for i, session in enumerate(sessions):
        if (i + 1) % 10 == 0:
            log.info(f"  Processing session {i+1}/{len(sessions)}: {session.session_id}")

        contract_ids = [session.hourly_contract_id] + session.quarter_contract_ids
        session_msgs = messages[messages['contract_id'].isin(contract_ids)].copy()

        try:
            # Reconstruct books
            books = reconstruct_session_books(
                session_msgs,
                session.hourly_contract_id,
                session.quarter_contract_ids,
            )

            # Step 7-9: Compute features including LSI
            feat_df = compute_session_features(
                books,
                session.hourly_contract_id,
                session.quarter_contract_ids,
                session.session_start,
                session.session_end,
                obs_freq_seconds=obs_freq,
            )

            if feat_df.empty or feat_df['lsi'].isna().all():
                session.exclusion_reason = 'empty_features'
                continue

            session.lsi_series = feat_df

            # Collect all LSI values for global threshold
            valid_lsi = feat_df['lsi'].dropna()
            all_lsi_values.extend(valid_lsi.tolist())

            # Step 12: Build event streams
            h_events, q_events = build_event_streams(session, books)
            session.h_events = h_events
            session.q_events = q_events

            retained_sessions.append(session)

        except Exception as exc:
            session.exclusion_reason = f'error: {exc}'
            log.debug(f"Session {session.session_id} excluded: {exc}")

    log.info(f"Initial retention after book reconstruction: {len(retained_sessions)}")

    # ---- Step 10: Compute global LSI threshold ----
    all_lsi_arr = np.array(all_lsi_values)
    global_threshold = compute_global_lsi_threshold(pd.Series(all_lsi_arr))
    log.info(f"Global LSI threshold (q75): {global_threshold:.4f}  (paper target: 0.264)")

    # ---- Step 11: Classify regimes with global threshold ----
    for session in retained_sessions:
        feat_df = session.lsi_series
        feat_df['regime'] = (feat_df['lsi'] >= global_threshold).astype(int)
        session.lsi_series = feat_df
        stressed_fraction = feat_df['regime'].mean()
        session.stressed_fraction = float(stressed_fraction)

        # Build state path from classified regimes
        session.state_path = build_state_path(session, feat_df.set_index('timestamp')['regime'])

    # ---- Step 13: Filter retained sessions ----
    final_sessions = filter_retained_sessions(retained_sessions, min_h_events=3, min_q_events=3)
    log.info(f"Final retained sessions: {len(final_sessions)} (paper target: ~2,915)")

    # Month breakdown
    month_counts = {}
    for s in final_sessions:
        month_counts[s.month] = month_counts.get(s.month, 0) + 1
    log.info(f"Session counts by month: {month_counts}")
    log.info("  Paper targets: Jan=741, Apr=707, Jul=738, Oct=729")

    # ---- Step 14: Threshold transportability diagnostics ----
    transport_df = compute_threshold_transportability(final_sessions, global_threshold)
    log.info(f"Global stressed fraction: {transport_df['stressed_fraction_global'].mean():.3f} (target ~0.25)")

    # ---- Step 15: Period summaries ----
    period_df = summarize_sessions_by_period(final_sessions)

    # ---- Save results ----
    results = {
        'sessions': final_sessions,
        'all_lsi': all_lsi_arr,
        'global_threshold': global_threshold,
        'transport_df': transport_df,
        'period_df': period_df,
        'month_counts': month_counts,
        'n_retained': len(final_sessions),
    }

    if save_pickle:
        pkl_path = os.path.join(RESULTS_DIR, 'exp1_sessions.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)
        log.info(f"Saved session data to {pkl_path}")

    # ---- Produce figures ----
    _plot_lsi_distribution(all_lsi_arr, global_threshold)
    _plot_stressed_fractions(transport_df)
    _plot_month_breakdown(month_counts)

    # ---- Write metrics ----
    _write_exp1_metrics(results)

    return results


def _plot_lsi_distribution(lsi_values: np.ndarray, threshold: float) -> None:
    """Plot LSI distribution with threshold line."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(lsi_values[~np.isnan(lsi_values)], bins=80, density=True,
            color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.axvline(threshold, color='crimson', linestyle='--', linewidth=2,
               label=f'Global q75 = {threshold:.3f}')
    ax.set_xlabel('LSI', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Liquidity Stress Indicator — Pooled Distribution', fontsize=14)
    ax.legend(fontsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp1_lsi_distribution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved plot: {path}")


def _plot_stressed_fractions(transport_df: pd.DataFrame) -> None:
    """Plot histogram of per-session stressed fractions."""
    if transport_df.empty or 'stressed_fraction_global' not in transport_df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(transport_df['stressed_fraction_global'].dropna(), bins=30,
            color='darkorange', alpha=0.75, edgecolor='white')
    ax.axvline(0.25, color='navy', linestyle='--', linewidth=2, label='25% reference')
    ax.set_xlabel('Stressed fraction (global threshold)', fontsize=13)
    ax.set_ylabel('Number of sessions', fontsize=13)
    ax.set_title('Per-session Stressed Occupancy (LSI threshold = 0.264)', fontsize=14)
    ax.legend(fontsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp1_stressed_fractions.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved plot: {path}")


def _plot_month_breakdown(month_counts: dict) -> None:
    """Bar chart of retained session counts by month."""
    fig, ax = plt.subplots(figsize=(8, 5))
    months = list(month_counts.keys())
    counts = [month_counts[m] for m in months]
    paper_targets = {'2024-01': 741, '2024-04': 707, '2024-07': 738, '2024-10': 729}
    ax.bar(range(len(months)), counts, color='teal', alpha=0.75, label='Reproduced')
    for j, m in enumerate(months):
        if m in paper_targets:
            ax.bar(j, paper_targets[m], color='none', edgecolor='red',
                   linewidth=2, label='Paper target' if j == 0 else '')
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([m.replace('2024-', '') for m in months], fontsize=12)
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Session count', fontsize=13)
    ax.set_title('Retained Sessions by Month', fontsize=14)
    ax.legend(fontsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp1_session_counts.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved plot: {path}")


def _write_exp1_metrics(results: dict) -> None:
    """Append Experiment 1 metrics to RESULTS.md."""
    global_threshold = results['global_threshold']
    n_retained = results['n_retained']
    month_counts = results['month_counts']
    transport_df = results['transport_df']
    stressed_mean = (
        transport_df['stressed_fraction_global'].mean()
        if not transport_df.empty and 'stressed_fraction_global' in transport_df.columns
        else float('nan')
    )

    lines = [
        "## Experiment 1: LSI Regime Classification\n",
        f"- **Global LSI threshold (q75)**: {global_threshold:.4f}  *(paper: 0.264)*\n",
        f"- **Total retained sessions**: {n_retained}  *(paper target: 2,915)*\n",
        "- **Session counts by month**:\n",
    ]
    for m, cnt in sorted(month_counts.items()):
        paper_map = {'2024-01': 741, '2024-04': 707, '2024-07': 738, '2024-10': 729}
        paper_val = paper_map.get(m, 'N/A')
        lines.append(f"  - {m}: {cnt}  *(paper: {paper_val})*\n")
    lines += [
        f"- **Mean stressed fraction (global threshold)**: {stressed_mean:.3f}  *(target: 0.25)*\n",
        "- **Figures**: exp1_lsi_distribution.png, exp1_stressed_fractions.png, exp1_session_counts.png\n",
        "\n",
    ]

    results_md = os.path.join(RESULTS_DIR, 'RESULTS.md')
    mode = 'a' if os.path.exists(results_md) else 'w'
    with open(results_md, mode) as f:
        if mode == 'w':
            f.write("# Experiment Results\n\n")
        f.writelines(lines)
    log.info(f"Metrics written to {results_md}")


if __name__ == '__main__':
    results = run_experiment_1()
    print(f"\nExp 1 complete. Retained sessions: {results['n_retained']}")
    print(f"Global LSI threshold: {results['global_threshold']:.4f}")
