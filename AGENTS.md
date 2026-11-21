# AGENTS.md — Shared Multi-AI Coordination Rules

Read this file and `CONTRACTS.md` before creating or editing any code. Multiple teammates (or one lead switching roles across phases) may work on this repo using different AI assistants.

## 1. Phase-Based File Ownership (Transferable)
Always check the **Active Role** declared in the user's prompt.

- **Phase 1 — Data & Physics (FROZEN / READ-ONLY):**
  `databases/mpea-05.tdb`, `src/config.py`, `src/verify_tdb.py`, `src/generate_data.py`, `src/clean_data.py`, `data/raw/`, `data/processed/sfe_dataset.csv`
  *Do NOT overwrite, re-run, or modify these files unless explicitly instructed.*
- **Phase 2 — ML, SHAP & Validation (Person 2 / Active Lead):**
  `notebooks/01_eda_data_check.ipynb`, `notebooks/02_train_and_shap.ipynb`, `data/processed/magpie_features_4k.parquet`, `data/processed/literature_bench.csv`, `models/`, `outputs/*.png`, `outputs/*.csv`
- **Phase 3 — Web App & Deployment (Person 3 / Deferred):**
  `app.py`, `requirements.txt`, `.gitignore`, `README.md`
- **Shared Utility (`src/utils.py`):**
  NEVER change existing function names, parameter types, or return types without updating `CONTRACTS.md`.

## 2. Non-Negotiable Engineering Guardrails
1. **No Mock Data or Fabricated Numbers:** Never fabricate ML metrics, SHAP values, or experimental literature SFE values/citations.
2. **Zero Data Leakage:** Every alloy appears at 5 temperatures. ALWAYS split train/test and run cross-validation using `GroupShuffleSplit` / `GroupKFold` grouped by `formula`. Never use row-level `train_test_split`.
3. **No `pycalphad` in `requirements.txt`:** Keep `requirements.txt` strictly for Streamlit Community Cloud. Put local/dev tools (`pycalphad`, `ipykernel`, `pyarrow`) in `requirements_local.txt`.
4. **Domain & Schema Lock:** 5-element substitutional space (`Co, Cr, Fe, Mn, Ni`), `Temperature` in `300–600 K`, `VEC >= 7.5`, SFE in `[-30.0, 150.0] mJ/m²`.
5. **Exact Feature Alignment:** `models/feature_columns.joblib` is the single source of truth for model input columns and ordering.

## 3. Source of Truth
- Physics & ML engineering specifications -> `PROJECT_SPEC.md`
- Exact schemas, shared contracts & live progress -> `CONTRACTS.md`