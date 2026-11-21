"""
Central configuration for the SFE data-generation pipeline.
Owned by: Person 1 (Data & Physics Lead).
"""
from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Database ────────────────────────────────────────────────────────
TDB_PATH = PROJECT_ROOT / "databases" / "mpea-05.tdb"

# ─── Aluminum Flag (Locked after Gate 2 approval) ────────────────────
INCLUDE_AL = False  # Approved: 5-element Co-Cr-Fe-Mn-Ni space

# ─── Element List (5 core elements, alphabetical order) ──────────────
ELEMENTS = ["Co", "Cr", "Fe", "Mn", "Ni"]

# ─── Disordered Phase Names (verified from mpea-05.tdb in Gate 2) ───
FCC_PHASE = "FCC_A1"
HCP_PHASE = "HCP_A3"

# ─── Temperature Domain Lock ────────────────────────────────────────
TEMP_MIN = 300   # K
TEMP_MAX = 600   # K

# ─── VEC Pre-filter ─────────────────────────────────────────────────
VEC_THRESHOLD = 7.5  # No upper cap

# ─── Sampling Fractions ─────────────────────────────────────────────
SAMPLING_FRACTIONS = {
    2: 0.10,   # 10% binary
    3: 0.25,   # 25% ternary
    4: 0.30,   # 30% quaternary
    5: 0.35,   # 35% quinary
}

# ─── Composition Constraints ────────────────────────────────────────
MIN_FRAC = 0.02        # Minimum mole fraction for any element

# ─── SFE Physical Bounds ────────────────────────────────────────────
# Updated for Gate 3: -30 to 150 mJ/m² retains valid low-SFE TRIP alloys (e.g. 304 steel, CoCrNi)
SFE_MIN = -30.0  # mJ/m²
SFE_MAX = 150.0  # mJ/m²

# ─── Olson-Cohen SFE Model Parameters ───────────────────────────────
# σ_γ/ε (FCC/HCP interfacial energy) — literature: 8–15 mJ/m²
SIGMA_INTERFACE = 10.0  # mJ/m² (mid-range, Olson & Cohen 1976)

# ρ_surf (molar surface density of {111} planes)
# Using a_FCC ≈ 3.59 Å for CoCrFeMnNi → ρ ≈ 2/(√3 * a² * N_A)
# Pre-computed ≈ 2.94e-5 mol/m²
RHO_SURF_DEFAULT = 2.94e-5  # mol/m²

# ─── Generation Parameters ──────────────────────────────────────────
DEFAULT_TARGET_ALLOYS = 2200
TEMPS_PER_ALLOY = 5          # 300 K + 4 random in [301, 600]
CHECKPOINT_INTERVAL = 25     # Flush every N valid alloys
MAX_TASKS_PER_CHILD = 100    # Worker recycling for memory safety

# ─── Locked CSV Schema (CONTRACTS.md § 2.A) ─────────────────────────
DATA_COLUMNS = ["formula", "Co", "Cr", "Fe", "Mn", "Ni", "Temperature", "VEC", "delta_G", "SFE"]

# ─── Output Paths ───────────────────────────────────────────────────
DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR        = PROJECT_ROOT / "outputs"

RAW_CSV_DEFAULT    = DATA_RAW_DIR / "raw_sfe_data.csv"
SAMPLE_CSV         = DATA_RAW_DIR / "sample_20_rows.csv"
CLEAN_CSV          = DATA_PROCESSED_DIR / "sfe_dataset.csv"
STATS_JSON         = OUTPUTS_DIR / "generation_stats.json"
SFE_DIST_PNG       = OUTPUTS_DIR / "sfe_distribution.png"
