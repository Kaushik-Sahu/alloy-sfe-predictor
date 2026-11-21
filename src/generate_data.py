"""
generate_data.py — Stage 1: Multi-core, stratified, resumable CALPHAD SFE data generator.
Owned by: Person 1 (Data & Physics Lead).

Features:
  - Stratified sampling (10% binary, 25% ternary, 30% quaternary, 35% quinary)
  - Simplex sampling with min_frac = 0.02
  - Fast VEC >= 7.5 filter (no upper cap)
  - 1273 K single-phase FCC equilibrium gate (miscibility-gap safe)
  - 5 temperatures per alloy (300 K + 4 random distinct in [301, 600] K)
  - Vectorized Olson-Cohen SFE (-30.0 <= SFE <= 150.0 mJ/m²)
  - Windows-safe ProcessPoolExecutor with worker initializer
  - Resumable CSV checkpointing every 25 alloys
  - Screening funnel metrics saved to outputs/generation_stats.json
"""

import os
import sys
import json
import time
import random
import argparse
import tempfile
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TDB_PATH, ELEMENTS, FCC_PHASE, HCP_PHASE,
    TEMP_MIN, TEMP_MAX, VEC_THRESHOLD,
    SAMPLING_FRACTIONS, MIN_FRAC,
    SFE_MIN, SFE_MAX,
    SIGMA_INTERFACE, RHO_SURF_DEFAULT,
    TEMPS_PER_ALLOY, CHECKPOINT_INTERVAL,
    DATA_COLUMNS, STATS_JSON, RAW_CSV_DEFAULT
)
from utils import compute_vec, make_canonical_formula, normalize_composition


# Global variable for worker processes (never pickled across processes)
_worker_db = None


def _init_worker(tdb_path_str: str):
    """Worker process initializer: sets up environment and loads database once per worker."""
    global _worker_db
    # Avoid matplotlib font cache collision on Windows
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), f"mpl_worker_{os.getpid()}")
    warnings.filterwarnings("ignore", message=".*order-disorder.*")
    from pycalphad import Database
    _worker_db = Database(tdb_path_str)


def _get_candidate_phases(db, alloy_els_set: set) -> list[str]:
    """Filter phases to only those whose sublattices can be populated by the active elements."""
    candidate_phases = []
    for pname, phase in db.phases.items():
        possible = True
        for subl in phase.constituents:
            subl_str = {str(c).upper() for c in subl}
            if not (subl_str & alloy_els_set):
                possible = False
                break
        if possible:
            candidate_phases.append(pname)
    return candidate_phases


def _check_single_phase_fcc(db, comp: dict, T: float = 1273.0) -> bool:
    """
    Check if the alloy is single-phase FCC at 1273 K.
    Returns True if exactly 1 active FCC phase entry with NP >= 0.99
    and total non-FCC phase fraction < 0.01.
    """
    from pycalphad import equilibrium, variables as v

    comp_upper = {el.upper(): frac for el, frac in comp.items()}
    dependent_el = max(comp_upper, key=comp_upper.get)

    comp_conds = {
        v.X(el): frac for el, frac in comp_upper.items() if el != dependent_el
    }
    conds = {v.T: T, v.P: 101325, v.N: 1.0}
    conds.update(comp_conds)

    alloy_els = sorted(set(comp_upper.keys()) | {"VA"})
    candidate_phases = _get_candidate_phases(db, set(alloy_els))

    try:
        eq = equilibrium(db, alloy_els, candidate_phases, conds)
    except Exception:
        return False

    phase_names = eq.Phase.values.squeeze()
    phase_nps = eq.NP.values.squeeze()

    stable = []
    for pname, npval in zip(np.atleast_1d(phase_names).flat, np.atleast_1d(phase_nps).flat):
        pname_str = str(pname).strip()
        if pname_str and pname_str != "nan":
            val = float(npval)
            if val > 1e-5:
                stable.append((pname_str, val))

    fcc_entries = [(pn, npv) for pn, npv in stable if pn == FCC_PHASE]
    non_fcc = [(pn, npv) for pn, npv in stable if pn != FCC_PHASE]

    # Exactly 1 FCC entry with NP >= 0.99 and sum of non-FCC < 0.01
    non_fcc_sum = sum(npv for _, npv in non_fcc)
    if len(fcc_entries) == 1 and fcc_entries[0][1] >= 0.99 and non_fcc_sum < 0.01:
        return True
    return False


def _calc_vectorized_sfe(db, comp: dict, temps: list[int]) -> list[dict] | None:
    """
    Compute Olson-Cohen SFE across all 5 temperatures in vectorized fashion.
    Returns list of row dicts if all temperatures satisfy SFE bounds, else None.
    """
    from pycalphad import equilibrium, variables as v

    comp_upper = {el.upper(): frac for el, frac in comp.items()}
    dependent_el = max(comp_upper, key=comp_upper.get)

    comp_conds = {
        v.X(el): frac for el, frac in comp_upper.items() if el != dependent_el
    }
    conds = {v.T: temps, v.P: 101325, v.N: 1.0}
    conds.update(comp_conds)

    alloy_els = sorted(set(comp_upper.keys()) | {"VA"})

    try:
        eq_fcc = equilibrium(db, alloy_els, [FCC_PHASE], conds)
        eq_hcp = equilibrium(db, alloy_els, [HCP_PHASE], conds)
    except Exception:
        return None

    # Extract Gibbs energies across temperatures
    G_fcc = eq_fcc.GM.values.squeeze()
    G_hcp = eq_hcp.GM.values.squeeze()

    if np.any(np.isnan(G_fcc)) or np.any(np.isnan(G_hcp)):
        return None

    delta_G_arr = G_hcp - G_fcc  # J/mol
    sfe_arr = 2.0 * RHO_SURF_DEFAULT * delta_G_arr * 1000.0 + 2.0 * SIGMA_INTERFACE  # mJ/m²

    # Check bounds for all temperatures
    for sfe in np.atleast_1d(sfe_arr):
        if np.isnan(sfe) or sfe < SFE_MIN or sfe > SFE_MAX:
            return None

    formula = make_canonical_formula(comp)
    vec = compute_vec(comp)

    rows = []
    for t_val, dg_val, sfe_val in zip(temps, np.atleast_1d(delta_G_arr), np.atleast_1d(sfe_arr)):
        row = {
            "formula": formula,
            "Co": round(comp.get("Co", 0.0), 4),
            "Cr": round(comp.get("Cr", 0.0), 4),
            "Fe": round(comp.get("Fe", 0.0), 4),
            "Mn": round(comp.get("Mn", 0.0), 4),
            "Ni": round(comp.get("Ni", 0.0), 4),
            "Temperature": int(t_val),
            "VEC": round(vec, 4),
            "delta_G": round(float(dg_val), 2),
            "SFE": round(float(sfe_val), 2),
        }
        rows.append(row)

    return rows


def _evaluate_candidate(comp: dict, temps: list[int]) -> tuple[str, list[dict] | None]:
    """
    Worker task: evaluates single candidate alloy.
    Returns (status_label, rows_or_none).
    """
    global _worker_db

    # Step 1: 1273 K single-phase FCC gate
    if not _check_single_phase_fcc(_worker_db, comp, T=1273.0):
        return ("fcc_rejected", None)

    # Step 2: Vectorized 5-temperature SFE calculation & bounds check
    rows = _calc_vectorized_sfe(_worker_db, comp, temps)
    if rows is None:
        return ("sfe_rejected", None)

    return ("success", rows)


def sample_candidate_composition(rng: random.Random) -> dict[str, float]:
    """
    Sample a candidate composition according to stratified order distribution:
      10% binary, 25% ternary, 30% quaternary, 35% quinary.
    Fractions strictly satisfy min_frac >= 0.0200 for all active elements,
    are rounded to 4 decimals, and sum to exactly 1.0000.
    All inactive elements are strictly 0.0.
    """
    # Select order k
    r = rng.random()
    if r < 0.10:
        k = 2
    elif r < 0.35:
        k = 3
    elif r < 0.65:
        k = 4
    else:
        k = 5

    # Choose k distinct elements
    active_elements = rng.sample(ELEMENTS, k)

    # Basis points integer allocation: 10,000 points = 1.0000, 200 points = 0.0200
    M_TOTAL = 10000
    M_MIN = 200  # corresponds strictly to 0.0200 mole fraction
    R = M_TOTAL - k * M_MIN  # surplus points to distribute on the simplex

    # Sample surplus points via exponential weights (uniform Dirichlet distribution)
    weights = [rng.expovariate(1.0) for _ in range(k)]
    total_w = sum(weights)
    raw_r = [w / total_w * R for w in weights]
    int_r = [int(round(x)) for x in raw_r]

    # Adjust rounding discrepancy on the largest surplus element
    diff = R - sum(int_r)
    max_idx = int_r.index(max(int_r))
    int_r[max_idx] += diff

    # Final integer points for each active element (guaranteed >= 200 points)
    points = [M_MIN + x for x in int_r]

    # Build composition: active elements >= 0.0200, inactive elements strictly 0.0
    comp = {el: 0.0 for el in ELEMENTS}
    for el, p in zip(active_elements, points):
        comp[el] = round(p / 10000.0, 4)

    # Strict post-condition assertions
    assert all(v == 0.0 or v >= 0.0200 for v in comp.values()), f"Mole fraction < 0.0200 detected: {comp}"
    assert round(sum(comp.values()), 4) == 1.0000, f"Composition sum != 1.0000: {comp}"

    return comp


def sample_temperatures(rng: random.Random) -> list[int]:
    """Return strictly [300] + 4 random distinct integer temperatures in [301, 600]."""
    random_temps = rng.sample(range(301, 601), 4)
    return [300] + sorted(random_temps)


def main():
    parser = argparse.ArgumentParser(description="Multi-core CALPHAD SFE Data Generator")
    parser.add_argument("--target-alloys", type=int, default=2200, help="Target number of unique valid alloys")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default=str(RAW_CSV_DEFAULT), help="Output raw CSV path")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    STATS_JSON.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Determine worker count
    num_cpus = os.cpu_count() or 4
    workers = args.workers if args.workers is not None else max(1, num_cpus - 2)

    print("=" * 70)
    print("  SFE Multi-Core CALPHAD Data Generator")
    print(f"  Target Valid Alloys: {args.target_alloys} (~{args.target_alloys * TEMPS_PER_ALLOY} rows)")
    print(f"  Output CSV:          {out_path}")
    print(f"  Worker Processes:    {workers} (CPUs detected: {num_cpus})")
    print(f"  Random Seed:         {args.seed}")
    print(f"  VEC Pre-filter:      >= {VEC_THRESHOLD}")
    print(f"  SFE Bounds:          [{SFE_MIN}, {SFE_MAX}] mJ/m²")
    print("=" * 70)

    # Resume support: check if output file already exists
    existing_formulas = set()
    buffered_rows = []
    valid_count = 0

    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            df_exist = pd.read_csv(out_path)
            if "formula" in df_exist.columns:
                existing_formulas = set(df_exist["formula"].unique())
                valid_count = len(existing_formulas)
                print(f"  [Resume] Found existing CSV with {len(df_exist)} rows ({valid_count} unique alloys).")
        except Exception as e:
            print(f"  [Resume Warning] Could not parse existing CSV ({e}), starting fresh.")
            existing_formulas = set()
            valid_count = 0
    else:
        # Write CSV header initially
        pd.DataFrame(columns=DATA_COLUMNS).to_csv(out_path, index=False)

    if valid_count >= args.target_alloys:
        print(f"  Target already reached ({valid_count} >= {args.target_alloys})! Nothing to do.")
        return

    # Funnel stats
    stats = {
        "target_alloys": args.target_alloys,
        "total_sampled": 0,
        "vec_passed": 0,
        "vec_rejected": 0,
        "fcc_passed": 0,
        "fcc_rejected": 0,
        "sfe_passed": 0,
        "sfe_rejected": 0,
        "valid_alloys_generated": valid_count,
        "start_time": time.time(),
        "elapsed_seconds": 0.0,
    }

    # Load existing stats if available
    if STATS_JSON.exists():
        try:
            with open(STATS_JSON, "r") as f:
                old_stats = json.load(f)
                stats["total_sampled"] = old_stats.get("total_sampled", 0)
                stats["vec_passed"] = old_stats.get("vec_passed", 0)
                stats["vec_rejected"] = old_stats.get("vec_rejected", 0)
                stats["fcc_passed"] = old_stats.get("fcc_passed", 0)
                stats["fcc_rejected"] = old_stats.get("fcc_rejected", 0)
                stats["sfe_passed"] = old_stats.get("sfe_passed", 0)
                stats["sfe_rejected"] = old_stats.get("sfe_rejected", 0)
        except Exception:
            pass

    start_time = time.time()
    last_log_time = start_time

    # Launch process pool
    print(f"\n  Spawning {workers} worker processes (loading CALPHAD database)...")
    batch_size = workers * 3

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(TDB_PATH),)
    ) as executor:
        print("  Workers initialized. Commencing candidate screening...\n")

        pending_futures = {}

        def submit_candidates():
            """Fill the pending futures queue up to batch_size."""
            while len(pending_futures) < batch_size and (valid_count + len(pending_futures)) < args.target_alloys * 3:
                comp = sample_candidate_composition(rng)
                stats["total_sampled"] += 1

                # Fast VEC filter (instant)
                vec = compute_vec(comp)
                if vec < VEC_THRESHOLD:
                    stats["vec_rejected"] += 1
                    continue

                stats["vec_passed"] += 1
                formula = make_canonical_formula(comp)

                # Skip if already exists or already queued
                if formula in existing_formulas:
                    continue

                existing_formulas.add(formula)
                temps = sample_temperatures(rng)

                fut = executor.submit(_evaluate_candidate, comp, temps)
                pending_futures[fut] = (formula, comp)

        # Initial submission
        submit_candidates()

        while pending_futures and valid_count < args.target_alloys:
            # Process as futures complete
            # Pop a completed future
            done_future = next(as_completed(pending_futures))
            formula, comp = pending_futures.pop(done_future)

            try:
                status_label, rows = done_future.result()
            except Exception as e:
                status_label = "worker_error"
                rows = None

            if status_label == "fcc_rejected":
                stats["fcc_rejected"] += 1
                existing_formulas.discard(formula)
            elif status_label == "sfe_rejected":
                stats["fcc_passed"] += 1
                stats["sfe_rejected"] += 1
                existing_formulas.discard(formula)
            elif status_label == "success" and rows:
                stats["fcc_passed"] += 1
                stats["sfe_passed"] += 1
                valid_count += 1
                stats["valid_alloys_generated"] = valid_count
                buffered_rows.extend(rows)

                # Checkpoint & flush to disk
                if len(buffered_rows) >= CHECKPOINT_INTERVAL * TEMPS_PER_ALLOY:
                    df_batch = pd.DataFrame(buffered_rows)
                    df_batch.to_csv(out_path, mode="a", header=False, index=False)
                    buffered_rows.clear()

                    stats["elapsed_seconds"] = round(time.time() - start_time, 1)
                    with open(STATS_JSON, "w") as f:
                        json.dump(stats, f, indent=2)

                    print(f"  [Checkpoint] Valid alloys: {valid_count}/{args.target_alloys} "
                          f"({valid_count * TEMPS_PER_ALLOY} rows) | "
                          f"Funnel: Sampled={stats['total_sampled']}, "
                          f"VEC_Pass={stats['vec_passed']}, "
                          f"FCC_Pass={stats['fcc_passed']}, "
                          f"SFE_Pass={stats['sfe_passed']}")

            # Periodically replenish the worker queue
            submit_candidates()

            # Logging heartbeat every 20 seconds
            now = time.time()
            if now - last_log_time >= 20:
                elapsed = now - start_time
                rate = valid_count / elapsed if elapsed > 0 else 0
                sec_per_alloy = elapsed / valid_count if valid_count > 0 else float("inf")
                print(f"  [Heartbeat] Valid: {valid_count}/{args.target_alloys} | "
                      f"Rate: {rate:.2f} alloys/sec ({sec_per_alloy:.1f} s/alloy) | "
                      f"Sampled: {stats['total_sampled']}")
                last_log_time = now

        # Flush any remaining buffered rows
        if buffered_rows:
            df_batch = pd.DataFrame(buffered_rows)
            df_batch.to_csv(out_path, mode="a", header=False, index=False)
            buffered_rows.clear()

    total_time = time.time() - start_time
    stats["elapsed_seconds"] = round(total_time, 1)
    stats["valid_alloys_generated"] = valid_count

    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 70)
    print("  DATA GENERATION COMPLETE")
    print(f"  Target Achieved:    {valid_count} unique alloys ({valid_count * TEMPS_PER_ALLOY} rows)")
    print(f"  Total Time:         {total_time:.1f} seconds ({total_time / 60:.2f} minutes)")
    if valid_count > 0:
        print(f"  Speed:              {total_time / valid_count:.2f} seconds per valid alloy")
    print(f"  Saved to:           {out_path}")
    print(f"  Funnel Stats:       {STATS_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
