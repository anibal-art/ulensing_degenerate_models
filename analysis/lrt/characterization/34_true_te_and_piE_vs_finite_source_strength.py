#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
True tE and true |piE| versus finite-source strength.

NO simulation.
NO fitting.

Goal
----
Provide a diagnostic figure to assess whether the apparent behavior at
large finite-source strength

    R_FS = rho_true / |u0_true|

could simply reflect that those events occupy a different region of
true (tE, |piE|) space.

Population
----------
Use the strictly positive H1-generated population built in

    analysis/lrt/results/characterization/18_positive_truth/
        h1_positive_truth_characterization.parquet

Figure
------
Single figure with three vertically stacked panels:

1. number of events per R_FS bin,
2. true tE distribution versus R_FS,
3. true |piE| distribution versus R_FS.

For panels 2 and 3 we show the median and the central 16-84% interval.

Outputs
-------
Figure:
    analysis/lrt/figures/characterization/
        34_true_te_and_piE_vs_finite_source_strength/
        true_te_and_piE_vs_finite_source_strength.png

Binned table:
    analysis/lrt/results/characterization/
        34_true_te_and_piE_vs_finite_source_strength/
        true_te_and_piE_vs_finite_source_strength_bins.csv

Summary:
    analysis/lrt/results/characterization/
        34_true_te_and_piE_vs_finite_source_strength/
        true_te_and_piE_vs_finite_source_strength_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys as _paper_sys
from pathlib import Path as _PaperPath

_PAPER_LRT_DIR = next(
    parent
    for parent in _PaperPath(__file__).resolve().parents
    if parent.name == "lrt"
)
_paper_sys.path.insert(
    0,
    str(_PAPER_LRT_DIR / "plotting"),
)
from paper_style import (
    apply_paper_style,
    label_panels,
    save_pdf_companion,
)

apply_paper_style()


# ============================================================================
# Configuration
# ============================================================================

INPUT_PATH = Path(
    "analysis/lrt/results/characterization/18_positive_truth/"
    "h1_positive_truth_characterization.parquet"
)

FIGURE_DIR = Path(
    "analysis/lrt/figures/characterization/"
    "34_true_te_and_piE_vs_finite_source_strength"
)

RESULTS_DIR = Path(
    "analysis/lrt/results/characterization/"
    "34_true_te_and_piE_vs_finite_source_strength"
)

FIGURE_PATH = FIGURE_DIR / "true_te_and_piE_vs_finite_source_strength.png"
TABLE_PATH = RESULTS_DIR / "true_te_and_piE_vs_finite_source_strength_bins.csv"
SUMMARY_PATH = RESULTS_DIR / "true_te_and_piE_vs_finite_source_strength_summary.txt"

N_BINS = 18
MIN_COUNT_FOR_STATS = 30


# ============================================================================
# Helpers
# ============================================================================

def wilson_interval(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return np.nan, np.nan

    phat = k / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2.0 * n)) / denom
    half = (
        z
        * np.sqrt(
            (phat * (1.0 - phat) / n)
            + (z**2 / (4.0 * n**2))
        )
        / denom
    )
    return center - half, center + half


def build_log_bins(values, n_bins):
    """
    Build logarithmic bin edges spanning the finite positive values.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    finite = finite[finite > 0.0]

    if len(finite) == 0:
        raise RuntimeError("No positive finite values available for binning.")

    vmin = finite.min()
    vmax = finite.max()

    log_min = np.floor(np.log10(vmin))
    log_max = np.ceil(np.log10(vmax))

    return np.logspace(log_min, log_max, n_bins + 1)


def summarize_series(x):
    """
    Return q16, median, q84 for a numeric pandas Series.
    """
    if len(x) == 0:
        return np.nan, np.nan, np.nan

    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan, np.nan, np.nan

    q16, q50, q84 = np.quantile(arr, [0.16, 0.50, 0.84])
    return q16, q50, q84


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 90)
    print("34: TRUE tE AND TRUE |piE| VERSUS FINITE-SOURCE STRENGTH")
    print("=" * 90)
    print()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)

    required_columns = [
        "catalog_row",
        "positive_lrt_class",
        "delta_chi2_lrt",
        "rho_over_abs_u0_true",
        "tE_true_days",
        "piE_true_amp",
        "finite_source_dominated_true",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns:\n" + "\n".join(f"- {c}" for c in missing)
        )

    print(f"Input rows = {len(df):,}")

    # ----------------------------------------------------------------------
    # Keep only rows with finite positive R_FS, finite positive tE, finite positive |piE|
    # ----------------------------------------------------------------------
    mask = (
        np.isfinite(df["rho_over_abs_u0_true"].to_numpy(dtype=float))
        & (df["rho_over_abs_u0_true"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(df["tE_true_days"].to_numpy(dtype=float))
        & (df["tE_true_days"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(df["piE_true_amp"].to_numpy(dtype=float))
        & (df["piE_true_amp"].to_numpy(dtype=float) > 0.0)
    )

    work = df.loc[mask].copy()

    print(f"Rows with valid positive inputs = {len(work):,}")

    n_total = len(work)
    n_fs_dominated = int((work["rho_over_abs_u0_true"] > 1.0).sum())
    frac_fs_dominated = n_fs_dominated / n_total if n_total > 0 else np.nan

    print()
    print("Finite-source-dominated regime:")
    print(f"  R_FS > 1 count    = {n_fs_dominated:,}")
    print(f"  R_FS > 1 fraction = {frac_fs_dominated:.6f}")

    print()
    print("positive_lrt_class counts:")
    print(work["positive_lrt_class"].value_counts())

    # ----------------------------------------------------------------------
    # Bin in R_FS
    # ----------------------------------------------------------------------
    bin_edges = build_log_bins(work["rho_over_abs_u0_true"], N_BINS)

    work["rfs_bin"] = pd.cut(
        work["rho_over_abs_u0_true"],
        bins=bin_edges,
        include_lowest=True,
        right=True,
    )

    grouped = work.groupby("rfs_bin", observed=False)

    rows = []
    for interval, g in grouped:
        if pd.isna(interval) or len(g) == 0:
            continue

        left = float(interval.left)
        right = float(interval.right)
        center = np.sqrt(left * right)

        n = len(g)
        n_detected = int((g["positive_lrt_class"] == "parallax_detected").sum())
        n_confused = int((g["positive_lrt_class"] == "confused_fspl_no_parallax").sum())

        tE_q16, tE_med, tE_q84 = summarize_series(g["tE_true_days"])
        pi_q16, pi_med, pi_q84 = summarize_series(g["piE_true_amp"])

        det_frac = n_detected / n if n > 0 else np.nan
        det_lo, det_hi = wilson_interval(n_detected, n)

        rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "bin_center": center,
                "n_total": n,
                "n_detected": n_detected,
                "n_confused": n_confused,
                "detected_fraction": det_frac,
                "detected_fraction_wilson_low": det_lo,
                "detected_fraction_wilson_high": det_hi,
                "tE_true_q16": tE_q16,
                "tE_true_median": tE_med,
                "tE_true_q84": tE_q84,
                "piE_true_q16": pi_q16,
                "piE_true_median": pi_med,
                "piE_true_q84": pi_q84,
            }
        )

    binned = pd.DataFrame(rows).sort_values("bin_center").reset_index(drop=True)

    if len(binned) == 0:
        raise RuntimeError("No non-empty bins were produced.")

    print()
    print("Binned summary:")
    print(
        binned[
            [
                "bin_center",
                "n_total",
                "detected_fraction",
                "tE_true_median",
                "piE_true_median",
            ]
        ].to_string(index=False)
    )

    min_count = int(binned["n_total"].min())
    n_bins_low_count = int((binned["n_total"] < MIN_COUNT_FOR_STATS).sum())

    print()
    print(f"Minimum count across displayed bins = {min_count}")
    print(f"Bins with n < {MIN_COUNT_FOR_STATS} = {n_bins_low_count}")

    # ----------------------------------------------------------------------
    # Save binned table
    # ----------------------------------------------------------------------
    binned.to_csv(TABLE_PATH, index=False)

    # ----------------------------------------------------------------------
    # Save text summary
    # ----------------------------------------------------------------------
    summary_lines = [
        "=" * 90,
        "34: TRUE tE AND TRUE |piE| VERSUS FINITE-SOURCE STRENGTH",
        "=" * 90,
        "",
        f"Input file: {INPUT_PATH}",
        f"Rows used: {n_total:,}",
        "",
        "Finite-source-dominated regime:",
        f"  R_FS > 1 count    = {n_fs_dominated:,}",
        f"  R_FS > 1 fraction = {frac_fs_dominated:.6f}",
        "",
        f"Number of bins = {len(binned)}",
        f"Minimum displayed count = {min_count}",
        f"Bins with n < {MIN_COUNT_FOR_STATS} = {n_bins_low_count}",
        "",
        "Global medians:",
        f"  median tE_true_days   = {work['tE_true_days'].median():.6f}",
        f"  median piE_true_amp   = {work['piE_true_amp'].median():.6f}",
        f"  median R_FS           = {work['rho_over_abs_u0_true'].median():.6e}",
        "",
        "positive_lrt_class counts:",
        str(work["positive_lrt_class"].value_counts()),
        "",
        "Interpretation target:",
        (
            "If tE_true_days and/or piE_true_amp vary systematically with R_FS, "
            "then the apparent finite-source trend in detectability may be partly "
            "or largely a composition effect rather than an independent finite-source effect."
        ),
    ]

    SUMMARY_PATH.write_text("\n".join(summary_lines))

    # ----------------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------------
    x = binned["bin_center"].to_numpy(dtype=float)
    n = binned["n_total"].to_numpy(dtype=float)

    tE_med = binned["tE_true_median"].to_numpy(dtype=float)
    tE_q16 = binned["tE_true_q16"].to_numpy(dtype=float)
    tE_q84 = binned["tE_true_q84"].to_numpy(dtype=float)

    pi_med = binned["piE_true_median"].to_numpy(dtype=float)
    pi_q16 = binned["piE_true_q16"].to_numpy(dtype=float)
    pi_q84 = binned["piE_true_q84"].to_numpy(dtype=float)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.8, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2, 1.2]},
    )

    # ------------------------------------------------------------------
    # Panel 1: counts
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.plot(x, n, marker="o", linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1.0, linestyle="--", linewidth=1.5)
    ax.set_ylabel("Events per bin")
    ax.set_title(
        "Population structure across finite-source strength",
        fontsize=13,
        pad=10,
    )
    ax.grid(True, alpha=0.3)

    text = (
        f"Strictly positive H1 population: N = {n_total:,}\n"
        f"Finite-source dominated (R_FS > 1): {n_fs_dominated:,} "
        f"({100.0 * frac_fs_dominated:.3f}%)\n"
        f"Minimum bin count: {min_count}"
    )

    # ------------------------------------------------------------------
    # Panel 2: tE
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.plot(x, tE_med, marker="o", linewidth=2, label="Median")
    ax.fill_between(
        x,
        tE_q16,
        tE_q84,
        alpha=0.25,
        label="16-84% interval",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1.0, linestyle="--", linewidth=1.5)
    ax.set_ylabel(r"True $t_E$ [days]")
    ax.set_title(
        r"True event timescale versus finite-source strength $R_{\rm FS} = \rho_{\rm true}/|u_{0,\rm true}|$",
        fontsize=13,
        pad=8,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # ------------------------------------------------------------------
    # Panel 3: |piE|
    # ------------------------------------------------------------------
    ax = axes[2]
    ax.plot(x, pi_med, marker="o", linewidth=2, label="Median")
    ax.fill_between(
        x,
        pi_q16,
        pi_q84,
        alpha=0.25,
        label="16-84% interval",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1.0, linestyle="--", linewidth=1.5)
    ax.set_ylabel(r"True $|\pi_E|$")
    ax.set_xlabel(
        r"True finite-source strength $R_{\rm FS} = \rho_{\rm true}/|u_{0,\rm true}|$"
    )
    ax.set_title(
        r"True parallax amplitude versus finite-source strength",
        fontsize=13,
        pad=8,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    label_panels(axes)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    save_pdf_companion(fig, FIGURE_PATH)
    plt.close(fig)

    print()
    print(f"Saved figure : {FIGURE_PATH}")
    print(f"Saved table  : {TABLE_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()


