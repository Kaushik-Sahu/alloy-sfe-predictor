# CONTRACTS.md — Team Interfaces & Live Handoff Board

## 1. Live Handoff Status (Update & Push After Every Stage)
| Parameter / Artifact | Current Value / Status | Updated By |
| :--- | :--- | :--- |
| Locked `.tdb` file | `databases/mpea-05.tdb` (Hallstedt et al. 2023, DOI: 10.1016/j.calphad.2023.102644) | Person 1 |
| `INCLUDE_AL` flag | `False` (Locked — 5-element `Co-Cr-Fe-Mn-Ni` space) | Person 1 |
| `data/raw/sample_20_rows.csv` | `READY` (100 rows, 20 alloys, min frac >= 0.0200) | Person 1 |
| `data/processed/sfe_dataset.csv` | `READY` (20,000 rows, 4,000 unique alloys; 15.6% TRIP / 17.8% TWIP / 66.6% Glide) | Person 1 |
| `notebooks/01_eda_data_check.ipynb` | `PENDING` (Stage 2A) | Phase 2 Lead |
| `models/xgb_sfe_model.joblib` + domain artifacts | `PENDING` (Stage 2B–2C) | Phase 2 Lead |
| `data/processed/literature_bench.csv` + SHAP plots | `PENDING` (Stage 2D) | Phase 2 Lead |
| `app.py` + Streamlit Cloud URL | `DEFERRED` (Phase 3) | Phase 3 Lead |

---

## 2. Locked Data Schemas (Exact Column Names & Units)

### A. `data/processed/sfe_dataset.csv`
- **Columns:** `formula,Co,Cr,Fe,Mn,Ni,Temperature,VEC,delta_G,SFE`
- **Units:** Elements in **mole fractions** (`0.0` to `1.0`, summing to `1.0000`, non-zero `>= 0.0200`), `Temperature` in **Kelvin (`300–600`)**, `delta_G` in **J/mol**, `SFE` in **mJ/m²**.

### B. `data/processed/literature_bench.csv`
- **Columns:** `alloy_name,formula,Temperature,exp_SFE,exp_method,source_citation`
- Scope: Interstitial-free `Co-Cr-Fe-Mn-Ni` alloys only (no C/N/Al/Si).

---

## 3. Shared Function Contract (`src/utils.py`)
All scripts, notebooks, and `app.py` MUST import these functions:
- `VEC_MAP: dict[str, float] = {"Al": 3, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10}`
- `normalize_composition(comp: dict[str, float]) -> dict[str, float]`
- `compute_vec(comp: dict[str, float]) -> float`
- `make_canonical_formula(comp: dict[str, float]) -> str`
  - Alphabetical formula string with 4-decimal mole fractions, omitting ONLY zero elements (`round(frac, 4) > 0`) so minor 1% additions in benchmarks/app inputs are preserved.
- `classify_sfe_regime(sfe: float) -> str`
  - `< 20.0` -> `"TRIP"`, `20.0 <= sfe <= 40.0` -> `"TWIP"`, `> 40.0` -> `"Glide"`.

---

## 4. Model Export Contract (`models/` — Phase 2 -> Phase 3)
Phase 2 MUST save all five files:
1. `models/xgb_sfe_model.joblib`: Trained sklearn-API `xgboost.XGBRegressor` supporting `.predict(X_df)` and `shap.TreeExplainer(model)`.
2. `models/feature_columns.joblib`: Plain Python `list[str]` of exact ordered input features (pruned Magpie columns + `"Temperature"`).
3. `models/metrics.json`: Train/Test/CV metrics (`r2`, `rmse`, `mae`) for `Ridge`, `RandomForest`, `ExtraTrees`, `XGBoost` (baseline & tuned), plus per-family test RMSE and regime accuracy.
4. `models/fcc_domain_nn.joblib`: `sklearn.neighbors.NearestNeighbors(n_neighbors=5)` fitted on the 4,000 unique valid FCC 5-element mole-fraction vectors (`[Co, Cr, Fe, Mn, Ni]`).
5. `models/domain_config.json`: `{"element_order": ["Co", "Cr", "Fe", "Mn", "Ni"], "p99_distance_threshold": <float>, "vec_min": 7.5, "temp_bounds": [300, 600], "sfe_bounds": [-30.0, 150.0]}`.