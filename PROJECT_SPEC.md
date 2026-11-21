# Project Specification: ML Surrogate Modeling for Stacking Fault Energy (SFE)

## 1. Project Goal
Build a physics-informed Machine Learning surrogate model that predicts intrinsic Stacking Fault Energy (SFE, mJ/m²) and deformation regime (TRIP / TWIP / Glide) for homogenized, single-phase FCC multicomponent alloys (austenitic steels, MEAs, and 3d transition-metal HEAs) across 300–600 K.

**Phase Status:**
- **Phase 1 (Setup & CALPHAD Data Generation) — COMPLETED & FROZEN:** 4,000 unique validated FCC alloys × 5 temperatures = 20,000 clean rows saved in `data/processed/sfe_dataset.csv`.
- **Phase 2 (EDA, ML Surrogate, SHAP & Literature Validation) — ACTIVE.**
- **Phase 3 (Streamlit Web App Deployment) — DEFERRED** (do NOT build `app.py` yet).

## 2. Locked Directory Structure

    .
    ├── .venv/                             # Local Python venv (Windows PowerShell)
    ├── .gitignore                         # Ignores .venv/, __pycache__/, raw_sfe_data.csv, logs
    ├── AGENTS.md                          # Multi-AI coordination & ownership guardrails
    ├── CONTRACTS.md                       # Locked schemas, function contracts & live status
    ├── PROJECT_SPEC.md                    # This file
    ├── README.md
    ├── requirements.txt                   # Streamlit Cloud ONLY (MUST EXCLUDE pycalphad)
    ├── requirements_local.txt             # Full local dev dependencies (pycalphad, ipykernel, pyarrow)
    ├── databases/
    │   └── mpea-05.tdb                    # Locked master .tdb (Hallstedt et al. 2023) [FROZEN]
    ├── data/
    │   ├── raw/
    │   │   ├── sample_20_rows.csv         # 100-row smoke test dataset [FROZEN]
    │   │   └── raw_sfe_data.csv           # 20,000-row raw CALPHAD output [FROZEN / gitignored]
    │   └── processed/
    │       ├── sfe_dataset.csv            # 20,000-row QA-validated dataset [FROZEN]
    │       ├── magpie_features_4k.parquet # Cached Magpie features for 4,000 unique formulas
    │       └── literature_bench.csv       # Verified experimental SFE benchmarks (Stage 2D)
    ├── src/
    │   ├── config.py                      # Locked constants & paths [FROZEN]
    │   ├── utils.py                       # Shared VEC, normalization, formula & regime helpers
    │   ├── verify_tdb.py                  # Stage 0 DB & Al check [FROZEN]
    │   ├── generate_data.py               # Stage 1 multi-core CALPHAD generator [FROZEN]
    │   └── clean_data.py                  # Stage 2 QA & EDA distribution script [FROZEN]
    ├── notebooks/
    │   ├── 01_eda_data_check.ipynb        # Stage 2A: Alloy families, T-trends & correlations
    │   └── 02_train_and_shap.ipynb        # Stages 2B–2D: Featurization, CV, XGBoost, SHAP & benchmarks
    ├── models/
    │   ├── xgb_sfe_model.joblib           # Tuned XGBRegressor
    │   ├── feature_columns.joblib         # Exact ordered list of input feature names
    │   ├── metrics.json                   # Benchmark & per-family metrics
    │   ├── fcc_domain_nn.joblib           # 5D NearestNeighbors FCC domain checker
    │   └── domain_config.json             # Domain bounds & p99 distance threshold
    ├── outputs/
    │   ├── generation_stats.json          # Phase 1 screening funnel counts
    │   ├── sfe_distribution.png           # Phase 1 SFE histogram
    │   ├── alloy_family_breakdown.png     # Stage 2A
    │   ├── sfe_vs_temp_trends.png         # Stage 2A
    │   ├── correlation_heatmap.png        # Stage 2A
    │   ├── parity_plot.png                # Stage 2C
    │   ├── residual_plot.png              # Stage 2C
    │   ├── regime_confusion_matrix.png    # Stage 2C
    │   ├── shap_summary.png               # Stage 2D
    │   ├── shap_dependence.png            # Stage 2D
    │   ├── shap_local_examples.png        # Stage 2D
    │   └── benchmark_comparison.csv       # Stage 2D
    └── app.py                             # Placeholder for Phase 3 Streamlit UI

## 3. Phase 1 Physics & Data Record (FROZEN — Reference Only)
- **Database:** `databases/mpea-05.tdb` (Hallstedt et al., *CALPHAD* 83, 2023, DOI: 10.1016/j.calphad.2023.102644).
- **Element Space:** Strictly 5-element substitutional `Co-Cr-Fe-Mn-Ni` (`INCLUDE_AL = False`; Al excluded due to B2_BCC precipitation at 8 at.% and unphysical SFE drop in `mpea-05.tdb`).
- **Screening Funnel (4,000 valid unique alloys / 20,000 rows):**
  - Stratified orders: 10% binary, 25% ternary, 30% quaternary, 35% quinary; integer basis-point allocation guaranteeing active fractions `>= 0.0200` and exact sum `1.0000`.
  - Gate 1: `VEC >= 7.5`.
  - Gate 2: Miscibility-gap-safe single-phase `FCC_A1` equilibrium at `1273 K`, `101325 Pa` (`NP >= 0.99`, exactly 1 active phase).
  - Target SFE: Vectorized Olson-Cohen model across `[300]` + 4 random distinct temperatures in `[301, 600]` K with `a = 3.60e-10 m` and `sigma = 10.0 mJ/m²` (supported by Pierce et al. 2014 range `8.6–11.8 mJ/m²`).
  - Physical bounds: `-30.0 <= SFE <= 150.0 mJ/m²`.
  - Final regime distribution: **15.6% TRIP (`<20`) / 17.8% TWIP (`20–40`) / 66.6% Glide (`>40`)**.

## 4. Phase 2 Specification — ML Surrogate, Interpretability & Validation

### A. Alloy Family Classification (Mutually Exclusive Order of Precedence)
Compute $\Delta S_{\text{mix}} / R = -\sum_{x_i > 0} x_i \ln(x_i)$ for each unique alloy:
1. `Binary`: `n_elements == 2`
2. `Austenitic Steel (Fe-Base)`: `n_elements >= 3` and `Fe >= 0.50`
3. `Ni/Co-Base Solid Solution`: `n_elements >= 3` and (`Ni >= 0.50` or `Co >= 0.50`)
4. `HEA`: $\Delta S_{\text{mix}} / R \ge 1.5$
5. `MEA`: `n_elements >= 3`, no element `>= 0.50`, and $1.0 \le \Delta S_{\text{mix}} / R < 1.5$
6. `Other Multi-Component FCC`: Any remaining valid FCC alloy

### B. Unique-Formula Matminer Featurization
- Extract the 4,000 unique `formula` values from `data/processed/sfe_dataset.csv`.
- Apply `StrToComposition` + `ElementProperty.from_preset("magpie")` once on the 4,000 unique formulas and cache to `data/processed/magpie_features_4k.parquet`.
- Merge back onto the 20,000 rows on `formula` and append `Temperature`.
- Prune NaN and zero-variance columns fitted strictly on the training split.

### C. Leak-Free Splitting, 4-Model Benchmark & Exports
- Outer split: `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` grouped by `formula` (3,200 train alloys / 800 test alloys). Assert zero formula overlap.
- Inner CV: 5-fold `GroupKFold(n_splits=5)` grouped by `formula` on Train.
- Benchmark 4 models: `Pipeline([StandardScaler(), Ridge(alpha=1.0)])`, `RandomForestRegressor(n_estimators=200)`, `ExtraTreesRegressor(n_estimators=200)`, and `XGBRegressor(tree_method="hist")`.
- Tune `XGBoost` with `RandomizedSearchCV` (`n_iter=20`, 5-fold `GroupKFold`, `neg_root_mean_squared_error`).
- Evaluate overall `R²`, `RMSE`, `MAE`, per-family RMSE, and derived `TRIP/TWIP/Glide` accuracy + confusion matrix.
- Export all 5 artifacts locked in `CONTRACTS.md § 4` (including 5D `NearestNeighbors` FCC domain checker).

### D. SHAP & Literature Validation (Stage 2D)
- Run `shap.TreeExplainer(xgb_model)` on 1,500 stratified test rows and save 300-DPI beeswarm, top-3 dependence, and 3 local waterfall plots (`Cantor HEA`, `304-base Steel`, `CoCrNi MEA`).
- Compare `CALPHAD vs. ML Surrogate vs. Experimental` on verified interstitial-free `Co-Cr-Fe-Mn-Ni` literature alloys in `data/processed/literature_bench.csv` and save `outputs/benchmark_comparison.csv`.