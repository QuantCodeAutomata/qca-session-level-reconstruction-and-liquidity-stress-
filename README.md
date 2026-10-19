# XBID State-Dependent Hawkes Process Framework

Reproduce three experiments from a paper on Dutch EPEX SPOT XBID
intraday electricity market microstructure, using a **State-Dependent
Hawkes Process (SDHP)** that conditions excitation kernels and baselines
on an observable liquidity-stress regime.

---

## Research overview

| Exp | Title | Key output |
|-----|-------|------------|
| 1 | Session-level LOB reconstruction & LSI regime classification | 19 synthetic sessions, global LSI threshold, stressed fractions |
| 2 | Contract-by-contract kernel comparison (1-/2-/3-Exp, power-law) | 3-Exp is best by NLL, KS pass rate, Wasserstein-1 |
| 3 | Structural analysis: branching matrices, spillovers, asymmetry, response speed | Endogeneity ~ρ 0.95; Q→H dominates in Normal; stress symmetrises cross-product flow |

## Data

Real data (Dutch EPEX SPOT XBID Level-3, Jan/Apr/Jul/Oct 2024) is
**proprietary and not included**.  The `data/synthetic_data.py` module
generates structurally equivalent synthetic sessions for reproducible
testing and CI.

Set `MASSIVE_API_KEY` in your environment if you have access to the Massive
REST API and wish to download real data via the ingest hooks in `data/`.

## Repository structure

```
├── data/
│   ├── __init__.py
│   └── synthetic_data.py    # Synthetic L3 session generator
├── src/
│   ├── book_reconstruction.py   # L3 → active bid/ask limit order book
│   ├── feature_engineering.py   # Spread, mid-price, vol, QI, EWMA Z-score, LSI
│   ├── session_builder.py       # Hourly + Q1-Q4 session windows
│   ├── sdhp_models.py           # SDHP dataclasses, branching matrix, init logic
│   ├── likelihood_exp.py        # O(n) recursive n-Exp log-likelihood
│   ├── likelihood_powerlaw.py   # Truncated power-law log-likelihood
│   ├── optimization.py          # Multi-start L-BFGS-B session estimation
│   ├── residuals.py             # Time-rescaled residuals
│   ├── diagnostics.py           # KS / Ljung-Box / Wasserstein-1 / NLL
│   └── structural_analysis.py   # Branching matrices, Ψ, A, t50/t90, bootstrap
├── exp/
│   ├── exp_1_lsi_regime.py
│   ├── exp_2_kernel_comparison.py
│   └── exp_3_structural_analysis.py
├── tests/                       # 135 pytest tests (all passing)
├── results/                     # Figures, CSVs, PKLs, RESULTS.md
├── AGENTS.md                    # Persistent agent memory
└── README.md
```

## Quickstart

```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn numba

# Exp 1 — reconstruct sessions and classify LSI regimes
python exp/exp_1_lsi_regime.py

# Exp 2 — fit all four kernel families (~4 min on 19 synthetic sessions)
python exp/exp_2_kernel_comparison.py

# Exp 3 — structural analysis of 3-Exp fits
python exp/exp_3_structural_analysis.py

# Run all tests
pytest tests/ -q
```

## Key formulas

**LSI (Liquidity Stress Indicator)**

```
LSI(t) = (z_spread + z_volatility + z_QI_abs) / 3
```

where each `z_x^slow(t)` uses a slow EWMA with λ = 0.01:

```
μ_t = (1-λ)μ_{t-1} + λ X_t
σ²_t = (1-λ)σ²_{t-1} + λ(X_t - μ_t)²
z_t = (X_t - μ_t) / σ_t
```

**Regime classification** (evaluated at t⁻, predictable):

```
S(t) = 1  iff  LSI(t) ≥ q₀.₇₅^{glob} = 0.264
```

**SDHP intensity** for event type e in state x:

```
λ_e(t) = μ_{x,e} + Σ_{e'} Σ_{t_i<t, e_i=e'} κ_{e'→e}^(x)(t - t_i)
```

with 3-Exp kernel: `κ(τ) = Σ_{r=1}^3 α_r exp(-β_r τ)`

**Penalized objective** (spectral-radius constraint):

```
J(θ) = L(θ) − ω · Σ_x max(0, ρ_x − 0.95)²,   ω = 10³
```

**Spillover multiplier**:

```
Ψ^(x) = (I - G^(x))^{-1},   G_{e'e}^(x) = Σ_r α_r/β_r
```

**Asymmetry index**:

```
A^(x) = (Ψ_{H→Q} - Ψ_{Q→H}) / (Ψ_{H→Q} + Ψ_{Q→H})
```

## Results summary

See [`results/RESULTS.md`](results/RESULTS.md) for full metric tables and
figure references.  Key qualitative findings (synthetic data):

- **3-Exp** achieves the lowest NLL and Wasserstein-1 among all kernel families.
- **Power-law** has the lowest convergence rate (63 % vs ≥ 99 % for Exp families).
- Self-excitation dominates the diagonal of G in both regimes.
- Stress regime increases cross-product Q→H total spillover.
- Within-session bootstrap confirms directional signs for most sessions.

## Reproducibility notes

| Unresolved choice | Decision |
|---|---|
| Power-law Δ_max | 1 800 s |
| Residual burn-in | 5 events |
| KS/LB α | 0.05 |
| Ljung-Box lags | 20 |
| Bootstrap reps | 1 000 |
