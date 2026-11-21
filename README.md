# ML Surrogate Modeling for Stacking Fault Energy (SFE)

Physics-informed Machine Learning surrogate model that predicts intrinsic Stacking Fault Energy (SFE, mJ/m²) and deformation regime (TRIP / TWIP / Glide) for homogenized, single-phase FCC multicomponent alloys (austenitic steels, MEAs, and 3d transition-metal HEAs) across 300–600 K.

## Project Structure

```
.
├── databases/          # CALPHAD .tdb thermodynamic database
├── data/raw/           # Raw generated CSV data
├── data/processed/     # Cleaned & validated dataset
├── src/                # Data generation & utility scripts
├── notebooks/          # EDA & ML training notebooks (Phase 2)
├── models/             # Trained model artifacts (Phase 2)
├── outputs/            # Plots, stats, benchmark tables
└── app.py              # Streamlit web UI (Phase 3)
```

## Quick Start (Local Development)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements_local.txt
python src/verify_tdb.py
python src/generate_data.py --target-alloys 20 --out data/raw/sample_20_rows.csv
python src/clean_data.py
```

## Deployment (Streamlit Community Cloud)

Uses `requirements.txt` only (no `pycalphad`).

## Team

- **Person 1:** Data & Physics Lead (CALPHAD pipeline)
- **Person 2:** ML & SHAP Lead (training, evaluation, explainability)
- **Person 3:** App & Integration Lead (Streamlit UI)
- **Person 4:** Docs & Benchmarking (literature validation)
