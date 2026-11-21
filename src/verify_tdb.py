"""
verify_tdb.py — Stage 0: Database, sublattice, benchmark & Aluminum check.
Owned by: Person 1 (Data & Physics Lead).

Verifies the CALPHAD .tdb database (databases/mpea-05.tdb) by:
  1. Parsing with pycalphad.Database
  2. Listing all elements and phases
  3. Inspecting FCC / HCP sublattice structures
  4. Running 1273 K single-phase FCC equilibrium gate + 300 K Olson-Cohen SFE
     on 4 benchmark alloys
  5. Testing Aluminum (AL) viability in FCC/HCP sublattices
"""
import sys
import os
import math
import warnings

# Suppress the pycalphad order-disorder warnings (known limitation, not relevant here)
warnings.filterwarnings("ignore", message=".*order-disorder.*")

# Ensure src/ is on path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TDB_PATH, SIGMA_INTERFACE, RHO_SURF_DEFAULT
from utils import VEC_MAP, normalize_composition, compute_vec, make_canonical_formula, classify_sfe_regime

import numpy as np
from pycalphad import Database, equilibrium, variables as v


def banner(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def parse_database(tdb_path: str) -> Database:
    """Parse the TDB file. Fail fast on any error."""
    banner("1. PARSING TDB DATABASE")
    print(f"  Path: {tdb_path}")
    try:
        db = Database(str(tdb_path))
        print("  ✓ Database parsed successfully (no syntax/keyword errors).")
        return db
    except Exception as e:
        print(f"  ✗ FATAL: Database parsing failed!")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Message: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def list_elements_and_phases(db: Database) -> bool:
    """Print all elements and phase names. Returns whether AL is present."""
    banner("2. ELEMENTS & PHASES")

    elements = sorted([str(e) for e in db.elements])
    print(f"  Elements ({len(elements)}): {elements}")

    required = {"CO", "CR", "FE", "MN", "NI", "VA"}
    found = set(elements)
    missing = required - found
    if missing:
        print(f"  ✗ MISSING required elements: {missing}")
    else:
        print(f"  ✓ All required elements present (CO, CR, FE, MN, NI, VA)")

    al_present = "AL" in found
    print(f"  AL present in db.elements: {al_present}")

    # Also note other non-core elements
    extra = found - required - {"AL"}
    if extra:
        print(f"  Additional elements in DB: {sorted(extra)}")

    phases = sorted(db.phases.keys())
    print(f"\n  Phases ({len(phases)}):")
    for p in phases:
        print(f"    - {p}")

    return al_present


def inspect_sublattices(db: Database) -> dict:
    """Inspect and print FCC/HCP sublattice structures."""
    banner("3. SUBLATTICE STRUCTURE INSPECTION")

    phase_info = {}
    fcc_candidates = [p for p in db.phases.keys() if "FCC" in p.upper()]
    hcp_candidates = [p for p in db.phases.keys() if "HCP" in p.upper()]
    bcc_candidates = [p for p in db.phases.keys() if "BCC" in p.upper()]

    print(f"  FCC-type phases found: {fcc_candidates}")
    print(f"  HCP-type phases found: {hcp_candidates}")
    print(f"  BCC-type phases found: {bcc_candidates}")

    for phase_name in fcc_candidates + hcp_candidates + bcc_candidates:
        phase_obj = db.phases[phase_name]
        constituents = phase_obj.constituents
        sublattice_model = phase_obj.sublattices
        print(f"\n  Phase: {phase_name}")
        print(f"    Sublattice site ratios: {sublattice_model}")
        for i, subl in enumerate(constituents):
            species_list = sorted([str(s) for s in subl])
            print(f"    Sublattice {i}: {species_list}")

        phase_info[phase_name] = {
            "sublattices": sublattice_model,
            "constituents": [sorted([str(s) for s in subl]) for subl in constituents],
        }

    return phase_info


def find_disordered_phases(phase_info: dict) -> tuple:
    """Identify the disordered FCC and HCP phase names."""
    banner("4. IDENTIFYING DISORDERED FCC & HCP PHASES")

    fcc_phase = None
    hcp_phase = None

    fcc_candidates = [p for p in phase_info if "FCC" in p.upper()]
    for p in fcc_candidates:
        if "A1" in p.upper() and "4SL" not in p.upper():
            fcc_phase = p
            break
    if fcc_phase is None and fcc_candidates:
        fcc_phase = fcc_candidates[0]

    hcp_candidates = [p for p in phase_info if "HCP" in p.upper()]
    for p in hcp_candidates:
        if "A3" in p.upper():
            hcp_phase = p
            break
    if hcp_phase is None and hcp_candidates:
        hcp_phase = hcp_candidates[0]

    print(f"  Selected FCC phase: {fcc_phase}")
    print(f"  Selected HCP phase: {hcp_phase}")

    if fcc_phase is None or hcp_phase is None:
        print("  ✗ FATAL: Could not identify both FCC and HCP phases!")
        sys.exit(1)

    return fcc_phase, hcp_phase


def _alloy_elements(comp: dict) -> list:
    """
    Build the element list for pycalphad equilibrium call.
    Uses ONLY the alloy's elements (uppercase) + VA.
    This avoids the 'degrees of freedom is not zero' error that occurs
    when unused database elements are included without composition constraints.
    """
    els = sorted(set(el.upper() for el in comp.keys()))
    if "VA" not in els:
        els.append("VA")
    return sorted(els)


def check_single_phase_fcc(db: Database, comp: dict, T: float,
                           fcc_phase: str, label: str = "") -> bool:
    """
    Check if the alloy is single-phase FCC at temperature T.
    Returns True if exactly 1 FCC phase entry with NP >= 0.99.
    Miscibility-gap safe: checks all FCC entries.
    """
    comp_upper = {el.upper(): frac for el, frac in comp.items()}
    dependent_el = max(comp_upper, key=comp_upper.get)

    comp_conds = {}
    for el, frac in comp_upper.items():
        if el != dependent_el:
            comp_conds[v.X(el)] = frac

    conds = {v.T: T, v.P: 101325, v.N: 1.0}
    conds.update(comp_conds)

    # Use ONLY alloy elements (not all DB elements)
    alloy_els = _alloy_elements(comp)
    all_phases = list(db.phases.keys())

    try:
        eq = equilibrium(db, alloy_els, all_phases, conds)
    except Exception as e:
        print(f"    [{label}] Equilibrium calc failed at {T} K: {e}")
        return False

    phase_names = eq.Phase.values.squeeze()
    phase_nps = eq.NP.values.squeeze()

    if phase_names.ndim == 0:
        phase_names = np.array([phase_names])
        phase_nps = np.array([phase_nps])

    stable = []
    for pname, npval in zip(phase_names.flat, phase_nps.flat):
        pname_str = str(pname).strip()
        if pname_str == '' or pname_str == 'nan':
            continue
        npval_f = float(npval)
        if npval_f > 1e-6:
            stable.append((pname_str, npval_f))

    fcc_entries = [(pn, npv) for pn, npv in stable if pn == fcc_phase]
    non_fcc = [(pn, npv) for pn, npv in stable if pn != fcc_phase]

    print(f"    [{label}] T={T} K: Stable phases = {stable}")
    print(f"      FCC entries: {len(fcc_entries)}, Non-FCC: {non_fcc}")

    if len(fcc_entries) == 1 and fcc_entries[0][1] >= 0.99 and len(non_fcc) == 0:
        print(f"      ✓ Single-phase FCC gate PASSED (NP={fcc_entries[0][1]:.4f})")
        return True
    else:
        print(f"      ✗ Single-phase FCC gate FAILED")
        return False


def calc_olson_cohen_sfe(db: Database, comp: dict, T: float,
                         fcc_phase: str, hcp_phase: str,
                         label: str = "") -> tuple:
    """
    Calculate SFE via Olson-Cohen model:
      SFE = 2 * ρ * ΔG(FCC→HCP) + 2 * σ
    Returns (sfe_mj, delta_G) or (nan, nan) on failure.
    """
    comp_upper = {el.upper(): frac for el, frac in comp.items()}
    dependent_el = max(comp_upper, key=comp_upper.get)

    comp_conds = {}
    for el, frac in comp_upper.items():
        if el != dependent_el:
            comp_conds[v.X(el)] = frac

    conds = {v.T: T, v.P: 101325, v.N: 1.0}
    conds.update(comp_conds)

    alloy_els = _alloy_elements(comp)

    try:
        eq_fcc = equilibrium(db, alloy_els, [fcc_phase], conds)
        G_fcc = float(eq_fcc.GM.values.squeeze())

        eq_hcp = equilibrium(db, alloy_els, [hcp_phase], conds)
        G_hcp = float(eq_hcp.GM.values.squeeze())
    except Exception as e:
        print(f"    [{label}] SFE calc failed at {T} K: {e}")
        return float('nan'), float('nan')

    delta_G = G_hcp - G_fcc  # J/mol

    # SFE [mJ/m²] = 2 * ρ [mol/m²] * ΔG [J/mol] * 1000 [mJ/J] + 2 * σ [mJ/m²]
    sfe_mj = 2.0 * RHO_SURF_DEFAULT * delta_G * 1000.0 + 2.0 * SIGMA_INTERFACE

    print(f"    [{label}] T={T} K: G_FCC={G_fcc:.2f} J/mol, G_HCP={G_hcp:.2f} J/mol")
    print(f"      ΔG(HCP-FCC) = {delta_G:.2f} J/mol")
    print(f"      SFE = 2×{RHO_SURF_DEFAULT:.2e}×{delta_G:.2f}×1000 + 2×{SIGMA_INTERFACE}")
    print(f"      SFE = {sfe_mj:.2f} mJ/m²")

    return sfe_mj, delta_G


def run_benchmark(db: Database, fcc_phase: str, hcp_phase: str) -> None:
    """Run benchmark alloy calculations."""
    banner("5. BENCHMARK ALLOYS (1273 K FCC gate + 300 K SFE)")

    benchmarks = [
        ("Cantor HEA", {"Co": 0.20, "Cr": 0.20, "Fe": 0.20, "Mn": 0.20, "Ni": 0.20}),
        ("304-base Steel", {"Fe": 0.72, "Cr": 0.18, "Ni": 0.10}),
        ("CoCrNi MEA", {"Co": 0.3333, "Cr": 0.3333, "Ni": 0.3334}),
        ("High-Mn TRIP HEA", {"Fe": 0.50, "Mn": 0.30, "Co": 0.10, "Cr": 0.10}),
    ]

    for name, comp in benchmarks:
        formula = make_canonical_formula(comp)
        print(f"\n  --- {name}: {formula} ---")

        vec = compute_vec(comp)
        print(f"    VEC = {vec:.2f}")

        fcc_ok = check_single_phase_fcc(db, comp, 1273.0, fcc_phase, label=name)
        sfe, delta_G = calc_olson_cohen_sfe(db, comp, 300.0, fcc_phase, hcp_phase, label=name)

        regime = "N/A"
        if not math.isnan(sfe):
            regime = classify_sfe_regime(sfe)
        print(f"    Result: FCC@1273K={'PASS' if fcc_ok else 'FAIL'}, "
              f"SFE@300K={sfe:.2f} mJ/m², Regime={regime}")


def run_aluminum_test(db: Database, fcc_phase: str, hcp_phase: str,
                      phase_info: dict) -> None:
    """Test Aluminum viability in FCC/HCP sublattices."""
    banner("6. ALUMINUM (AL) VIABILITY TEST")

    print("  Checking AL in phase sublattices:")
    al_in_fcc = False
    al_in_hcp = False

    if fcc_phase in phase_info:
        for i, subl in enumerate(phase_info[fcc_phase]["constituents"]):
            if "AL" in subl:
                al_in_fcc = True
                print(f"    ✓ AL found in {fcc_phase} sublattice {i}: {subl}")
                break
        if not al_in_fcc:
            print(f"    ✗ AL NOT found in any {fcc_phase} sublattice")

    if hcp_phase in phase_info:
        for i, subl in enumerate(phase_info[hcp_phase]["constituents"]):
            if "AL" in subl:
                al_in_hcp = True
                print(f"    ✓ AL found in {hcp_phase} sublattice {i}: {subl}")
                break
        if not al_in_hcp:
            print(f"    ✗ AL NOT found in any {hcp_phase} sublattice")

    if not (al_in_fcc and al_in_hcp):
        print("\n  ✗ AL is NOT viable: missing from FCC and/or HCP metallic sublattices.")
        print("  RECOMMENDATION: INCLUDE_AL = False")
        return

    print("\n  AL is present in both FCC and HCP sublattices. Running equilibrium tests...")

    # Base Cantor alloy for comparison
    base_comp = {"Co": 0.20, "Cr": 0.20, "Fe": 0.20, "Mn": 0.20, "Ni": 0.20}
    base_sfe, _ = calc_olson_cohen_sfe(db, base_comp, 300.0, fcc_phase, hcp_phase,
                                        label="Base CoCrFeMnNi")

    # Test 1: Al0.05(CoCrFeMnNi)0.95
    al5_comp = {"Al": 0.05, "Co": 0.19, "Cr": 0.19, "Fe": 0.19, "Mn": 0.19, "Ni": 0.19}
    al5_comp = normalize_composition(al5_comp)

    print(f"\n  --- Al0.05(CoCrFeMnNi)0.95: {make_canonical_formula(al5_comp)} ---")
    vec5 = compute_vec(al5_comp)
    print(f"    VEC = {vec5:.2f}")

    fcc_ok_5 = check_single_phase_fcc(db, al5_comp, 1273.0, fcc_phase, label="Al5%")
    sfe_5, _ = calc_olson_cohen_sfe(db, al5_comp, 300.0, fcc_phase, hcp_phase, label="Al5%")

    # Test 2: Al0.08(CoCrFeMnNi)0.92
    al8_comp = {"Al": 0.08, "Co": 0.184, "Cr": 0.184, "Fe": 0.184, "Mn": 0.184, "Ni": 0.184}
    al8_comp = normalize_composition(al8_comp)

    print(f"\n  --- Al0.08(CoCrFeMnNi)0.92: {make_canonical_formula(al8_comp)} ---")
    vec8 = compute_vec(al8_comp)
    print(f"    VEC = {vec8:.2f}")

    fcc_ok_8 = check_single_phase_fcc(db, al8_comp, 1273.0, fcc_phase, label="Al8%")
    sfe_8, _ = calc_olson_cohen_sfe(db, al8_comp, 300.0, fcc_phase, hcp_phase, label="Al8%")

    # Summary
    print(f"\n  --- AL VIABILITY SUMMARY ---")
    print(f"    Base CoCrFeMnNi SFE@300K = {base_sfe:.2f} mJ/m²")
    print(f"    Al5% alloy: FCC@1273K={'PASS' if fcc_ok_5 else 'FAIL'}, SFE@300K={sfe_5:.2f} mJ/m²")
    print(f"    Al8% alloy: FCC@1273K={'PASS' if fcc_ok_8 else 'FAIL'}, SFE@300K={sfe_8:.2f} mJ/m²")

    if not math.isnan(sfe_5) and not math.isnan(sfe_8) and not math.isnan(base_sfe):
        if sfe_5 > base_sfe:
            print(f"    ✓ AL raises SFE (5%: {sfe_5:.2f} vs base: {base_sfe:.2f} mJ/m²)")
        else:
            print(f"    ✗ AL does NOT raise SFE (5%: {sfe_5:.2f} vs base: {base_sfe:.2f} mJ/m²)")

        if fcc_ok_5 and fcc_ok_8:
            print(f"    ✓ Both Al compositions pass FCC gate at 1273 K")
            print(f"    RECOMMENDATION: INCLUDE_AL = True")
        elif fcc_ok_5:
            print(f"    ⚠ Al8% fails FCC gate — AL viable but cap strictly")
            print(f"    RECOMMENDATION: INCLUDE_AL = True (with strict AL_MAX_FRAC=0.10 cap)")
        else:
            print(f"    ✗ AL alloys fail FCC gate")
            print(f"    RECOMMENDATION: INCLUDE_AL = False")
    else:
        print(f"    ✗ Could not compute SFE for AL alloys")
        print(f"    RECOMMENDATION: INCLUDE_AL = False")


def main():
    print("=" * 70)
    print("  SFE Project — TDB Database Verification Script")
    print(f"  Database: {TDB_PATH}")
    print("=" * 70)

    db = parse_database(TDB_PATH)
    al_in_elements = list_elements_and_phases(db)
    phase_info = inspect_sublattices(db)
    fcc_phase, hcp_phase = find_disordered_phases(phase_info)
    run_benchmark(db, fcc_phase, hcp_phase)

    if al_in_elements:
        run_aluminum_test(db, fcc_phase, hcp_phase, phase_info)
    else:
        banner("6. ALUMINUM (AL) VIABILITY TEST")
        print("  ✗ AL not present in db.elements — skipping AL tests.")
        print("  RECOMMENDATION: INCLUDE_AL = False")

    banner("VERIFICATION COMPLETE")
    print("  Review the output above and approve INCLUDE_AL flag before Gate 3.")


if __name__ == "__main__":
    main()
