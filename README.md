# NQFO
My first podium project, great memories and great learnings!!
# Volatility Surface Completion — NQFO 2026

**National Quant Finance Olympiad 2026** · Organised by Finance and Economics Club, IIT Guwahati  
*In collaboration with Jane Street, QRT, AQUA, Matiks, GeeksforGeeks & Spykar Lifestyles Pvt. Ltd.*

> **Result:** 🥈 First Runner-Up (2nd Prize) — Solo Participation

---

## Problem Statement

Given a partially observed implied volatility (IV) surface across multiple strikes, maturities, and option types, predict the missing IV values such that the completed surface is:
- Accurate (minimises RMSE against held-out ground truth)
- Financially valid (free of calendar spread and butterfly arbitrage)

---

## Approach — 5-Phase Hybrid Pipeline

### Phase 1 · Data Loading & Feature Engineering
Parses train/test CSVs, validates schema, and computes derived columns — log-moneyness, total variance `w = σ²τ`, option type indicator.

### Phase 2 · Volatility Regime Detection
Uses **K-Means clustering** (3 regimes: calm / normal / turbulent) on daily surface features — ATM IV, term structure slope, put-call skew, smile curvature, IV standard deviation. Regime labels are used as features in the ML residual model.

### Phase 3 · Arbitrage-Free SVI Surface Fitting
Fits a **Stochastic Volatility Inspired (SVI)** parametric smile to each (date, maturity, option type) slice in total-variance space `(k, w)`. Optimisation uses **SLSQP** with:
- Multiple warm-started initialisations (smart + random)
- Hard constraints enforcing Gatheral's no-arbitrage conditions (ρ bounds, butterfly density positivity)
- Post-fit **calendar spread enforcement** — shifts `a` parameters across maturities to guarantee `w(T₁) ≤ w(T₂)` for T₁ < T₂

### Phase 4 · ML Residual Correction
Trains a **Histogram Gradient Boosting Regressor** on the residuals `iv_observed − iv_svi` using 16 features spanning moneyness polynomials, time-to-expiry, regime labels, and cross-terms. Final prediction: `iv_predicted = iv_svi + residual_ml`.  
Validated via expanding-window cross-validation (5 folds).

### Phase 5 · Arbitrage Enforcement & Wing Correction
- **Iterative arbitrage checker** (up to 5 passes): calendar and butterfly violations in predicted rows are replaced with the arbitrage-free SVI value.
- **Wing extrapolation corrector**: rows outside the observed strike range are corrected using a boundary-anchored blend (`α=0.9 × local_iv + 0.1 × svi_ml`), calibrated from training medians per (maturity, type, moneyness).

---

## Results

| Model | RMSE (%) |
|-------|----------|
| Baseline (SVI only) | — |
| SVI + ML Residual | **~43% reduction vs baseline** |

---

## Repository Structure

```
├── solution.py          # Full pipeline (Phases 1–5)
├── train.csv            # Training data (observed IVs)
├── test.csv             # Test data (IVs to predict)
├── submission.csv       # Model output (row_id, iv_predicted)
├── methodology.pdf      # Detailed writeup of the approach
└── requirements.txt     # Dependencies
```

---

## Setup & Usage

```bash
pip install -r requirements.txt
python solution.py
# or with custom paths:
python solution.py --train train.csv --test test.csv --out submission.csv
```

**Dependencies:** `numpy`, `pandas`, `scipy`, `scikit-learn`

---

## Key Concepts

- **SVI parameterisation** — Gatheral (2004); total variance smile fitting in `(k, w)` space
- **Calendar spread arbitrage** — `w(k, T)` must be non-decreasing in `T` at fixed `k`
- **Butterfly arbitrage** — the Dupire local variance density must remain non-negative
- **Regime-conditional ML** — residual correction conditioned on market volatility regime

---

*Developed independently for NQFO 2026.*
