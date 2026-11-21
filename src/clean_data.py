"""
clean_data.py — Stage 2: Deduplication, QA checks, regime stats, and EDA plots.
Owned by: Person 1 (Data & Physics Lead).

Features:
  - Validates exact schema against CONTRACTS.md § 2.A
  - Deduplicates by (formula, Temperature)
  - Enforces zero NaNs/infs and valid domain ranges
  - Confirms exactly 5 temperatures per unique alloy
  - Computes and prints TRIP / TWIP / Glide regime breakdown
  - Exports cleaned data to data/processed/sfe_dataset.csv
  - Saves publication-quality histogram to outputs/sfe_distribution.png
"""

import os
import sys
import tempfile
import argparse
from pathlib import Path

# Ensure writable matplotlib cache on Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_clean"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_COLUMNS, CLEAN_CSV, RAW_CSV_DEFAULT,
    SAMPLE_CSV, SFE_DIST_PNG, SFE_MIN, SFE_MAX,
    TEMP_MIN, TEMP_MAX, VEC_THRESHOLD,
    TEMPS_PER_ALLOY
)
from utils import classify_sfe_regime, compute_vec


def clean_and_validate(input_csv: Path, output_csv: Path, plot_png: Path) -> pd.DataFrame:
    """Clean, validate, summarize, and plot the SFE dataset."""
    print("=" * 70)
    print("  SFE Data Cleaning & QA Pipeline")
    print(f"  Input:  {input_csv}")
    print(f"  Output: {output_csv}")
    print("=" * 70)

    if not input_csv.exists() or input_csv.stat().st_size == 0:
        raise FileNotFoundError(f"Input file not found or empty: {input_csv}")

    df = pd.read_csv(input_csv)
    raw_row_count = len(df)
    print(f"\n[1] Loaded raw data: {raw_row_count} rows")

    # 1. Exact Column Alignment Check
    print(f"\n[2] Checking column schema against CONTRACTS.md...")
    actual_cols = list(df.columns)
    if actual_cols != DATA_COLUMNS:
        raise ValueError(
            f"Schema mismatch!\n"
            f"Expected: {DATA_COLUMNS}\n"
            f"Actual:   {actual_cols}"
        )
    print(f"    ✓ Column schema strictly matches CONTRACTS.md § 2.A: {actual_cols}")

    # 2. Deduplication Check
    print(f"\n[3] Checking duplicates on ('formula', 'Temperature')...")
    dup_mask = df.duplicated(subset=["formula", "Temperature"], keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        print(f"    ⚠ Dropping {dup_count} duplicate rows.")
        df = df[~dup_mask].copy()
    else:
        print("    ✓ Zero duplicates found.")

    # 3. Missing Value / Inf Check
    print(f"\n[4] Checking for NaNs, nulls, and infinities...")
    nan_counts = df.isna().sum()
    total_nans = int(nan_counts.sum())
    if total_nans > 0:
        print(f"    ✗ Found NaN values:\n{nan_counts[nan_counts > 0]}")
        df = df.dropna().copy()
    else:
        print("    ✓ Zero NaNs found.")

    inf_mask = np.isinf(df[["delta_G", "SFE", "VEC"]]).any(axis=1)
    inf_count = int(inf_mask.sum())
    if inf_count > 0:
        print(f"    ✗ Dropping {inf_count} rows with infinite values.")
        df = df[~inf_mask].copy()
    else:
        print("    ✓ Zero infinite values found.")

    # 4. Domain & Physics Range Validation
    print(f"\n[5] Validating domain boundaries and physical constraints...")
    # SFE bounds
    sfe_invalid = (df["SFE"] < SFE_MIN) | (df["SFE"] > SFE_MAX)
    if sfe_invalid.sum() > 0:
        print(f"    ✗ Dropping {sfe_invalid.sum()} rows with SFE outside [{SFE_MIN}, {SFE_MAX}].")
        df = df[~sfe_invalid].copy()
    else:
        print(f"    ✓ All SFE values within physical bounds [{SFE_MIN}, {SFE_MAX}] mJ/m².")

    # Temperature bounds
    temp_invalid = (df["Temperature"] < TEMP_MIN) | (df["Temperature"] > TEMP_MAX)
    if temp_invalid.sum() > 0:
        print(f"    ✗ Dropping {temp_invalid.sum()} rows with T outside [{TEMP_MIN}, {TEMP_MAX}] K.")
        df = df[~temp_invalid].copy()
    else:
        print(f"    ✓ All temperatures within [{TEMP_MIN}, {TEMP_MAX}] K.")

    # VEC bounds
    vec_invalid = df["VEC"] < VEC_THRESHOLD
    if vec_invalid.sum() > 0:
        print(f"    ✗ Dropping {vec_invalid.sum()} rows with VEC < {VEC_THRESHOLD}.")
        df = df[~vec_invalid].copy()
    else:
        print(f"    ✓ All VEC values >= {VEC_THRESHOLD}.")

    # Composition sum check (sum to 1.0000 ± 0.001)
    elem_cols = ["Co", "Cr", "Fe", "Mn", "Ni"]
    comp_sums = df[elem_cols].sum(axis=1)
    sum_invalid = (comp_sums - 1.0).abs() > 0.001
    if sum_invalid.sum() > 0:
        print(f"    ✗ Dropping {sum_invalid.sum()} rows where composition sum != 1.0.")
        df = df[~sum_invalid].copy()
    else:
        print("    ✓ All compositions sum to 1.0000 ± 0.001.")

    # Minimum non-zero fraction check (strictly >= 0.0200)
    print("\n[5b] Validating minimum non-zero mole fractions (min_frac >= 0.0200)...")
    trace_violations = 0
    for col in elem_cols:
        trace_mask = (df[col] > 0.0) & (df[col] < 0.02)
        count_trace = int(trace_mask.sum())
        if count_trace > 0:
            print(f"    ✗ Found {count_trace} rows with trace mole fraction < 0.02 in '{col}'!")
            trace_violations += count_trace

    if trace_violations > 0:
        raise ValueError(f"Dataset contains {trace_violations} elements violating min_frac >= 0.0200 rule!")

    # Calculate and display the minimum non-zero mole fraction
    nonzero_values = df[elem_cols].values.flatten()
    nonzero_values = nonzero_values[nonzero_values > 0.0]
    min_nonzero_val = float(nonzero_values.min()) if len(nonzero_values) > 0 else 0.0
    print(f"    ✓ All non-zero element fractions are strictly >= 0.0200 (Minimum non-zero = {min_nonzero_val:.4f}).")

    # 5. Temperature Balance Check per Formula
    print(f"\n[6] Validating {TEMPS_PER_ALLOY} temperatures per unique alloy...")
    counts_per_formula = df.groupby("formula")["Temperature"].nunique()
    incomplete = counts_per_formula[counts_per_formula != TEMPS_PER_ALLOY]
    if len(incomplete) > 0:
        print(f"    ⚠ Found {len(incomplete)} alloys with != {TEMPS_PER_ALLOY} temps. Dropping incomplete alloys.")
        valid_formulas = set(counts_per_formula[counts_per_formula == TEMPS_PER_ALLOY].index)
        df = df[df["formula"].isin(valid_formulas)].copy()
    else:
        print(f"    ✓ Every unique alloy has exactly {TEMPS_PER_ALLOY} distinct temperatures.")

    unique_alloys = df["formula"].nunique()
    total_clean_rows = len(df)
    print(f"\n[7] Cleaned dataset summary:")
    print(f"    - Unique alloys: {unique_alloys}")
    print(f"    - Total rows:    {total_clean_rows} ({unique_alloys * TEMPS_PER_ALLOY} expected)")

    # 6. Deformation Regime Classification
    print(f"\n[8] Deformation Regime Breakdown (CONTRACTS.md § 3):")
    df["regime"] = df["SFE"].apply(classify_sfe_regime)
    regime_counts = df["regime"].value_counts()
    for regime in ["TRIP", "TWIP", "Glide"]:
        cnt = regime_counts.get(regime, 0)
        pct = (cnt / total_clean_rows * 100) if total_clean_rows > 0 else 0
        print(f"    - {regime:6s}: {cnt:6d} rows ({pct:5.1f}%)")

    # 7. Export Clean CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    export_df = df[DATA_COLUMNS].copy()
    export_df.to_csv(output_csv, index=False)
    print(f"\n[9] Saved clean dataset to: {output_csv}")

    # 8. Generate Distribution Plot
    plot_png.parent.mkdir(parents=True, exist_ok=True)
    _generate_distribution_plot(df, plot_png)
    print(f"[10] Saved distribution plot to: {plot_png}")

    print("\n" + "=" * 70)
    print("  QA & CLEANING COMPLETE — READY FOR PERSON 2 ML PIPELINE")
    print("=" * 70)

    return df


def _generate_distribution_plot(df: pd.DataFrame, plot_path: Path):
    """Create publication-quality SFE distribution plot with TRIP/TWIP/Glide zones."""
    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    sfe = df["SFE"]

    # Shaded deformation regime zones
    ax_hist.axvspan(SFE_MIN, 20.0, alpha=0.18, color="#3498db", label="TRIP (< 20 mJ/m²)")
    ax_hist.axvspan(20.0, 40.0, alpha=0.18, color="#2ecc71", label="TWIP (20–40 mJ/m²)")
    ax_hist.axvspan(40.0, SFE_MAX, alpha=0.18, color="#e74c3c", label="Glide (> 40 mJ/m²)")

    ax_box.axvspan(SFE_MIN, 20.0, alpha=0.18, color="#3498db")
    ax_box.axvspan(20.0, 40.0, alpha=0.18, color="#2ecc71")
    ax_box.axvspan(40.0, SFE_MAX, alpha=0.18, color="#e74c3c")

    # Histogram & KDE
    n_bins = min(40, max(15, len(df) // 5))
    counts, bins, patches = ax_hist.hist(
        sfe, bins=n_bins, color="#2c3e50", edgecolor="white", alpha=0.85, density=False
    )

    # Threshold divider lines
    for boundary in [20.0, 40.0]:
        ax_hist.axvline(boundary, color="#7f8c8d", linestyle="--", linewidth=1.2)
        ax_box.axvline(boundary, color="#7f8c8d", linestyle="--", linewidth=1.2)

    # Boxplot
    ax_box.boxplot(sfe, orientation="horizontal", widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="#95a5a6", color="#2c3e50"),
                   medianprops=dict(color="#c0392b", linewidth=2))

    # Stats annotations
    mean_val = sfe.mean()
    median_val = sfe.median()
    std_val = sfe.std()
    q25, q75 = sfe.quantile(0.25), sfe.quantile(0.75)

    stats_text = (
        f"Alloys: {df['formula'].nunique()}\n"
        f"Rows: {len(df)}\n"
        f"Mean: {mean_val:.1f} mJ/m²\n"
        f"Median: {median_val:.1f} mJ/m²\n"
        f"Std: {std_val:.1f} mJ/m²\n"
        f"IQR: [{q25:.1f}, {q75:.1f}]"
    )
    ax_hist.text(
        0.97, 0.95, stats_text, transform=ax_hist.transAxes,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#bdc3c7", alpha=0.9),
        fontsize=9, family="monospace"
    )

    ax_hist.set_title(
        "Stacking Fault Energy (SFE) Distribution across Deformation Regimes (300–600 K)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.legend(loc="upper left", framealpha=0.9)
    ax_hist.grid(axis="y", alpha=0.3, linestyle=":")

    ax_box.set_xlabel("Stacking Fault Energy, SFE (mJ/m²)", fontsize=11)
    ax_box.set_yticks([])

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Clean and validate CALPHAD SFE data")
    parser.add_argument("--in-csv", type=str, default=str(RAW_CSV_DEFAULT), help="Input raw CSV path")
    parser.add_argument("--out-csv", type=str, default=str(CLEAN_CSV), help="Output processed CSV path")
    parser.add_argument("--out-png", type=str, default=str(SFE_DIST_PNG), help="Output plot PNG path")
    args = parser.parse_args()

    clean_and_validate(Path(args.in_csv), Path(args.out_csv), Path(args.out_png))


if __name__ == "__main__":
    main()
