#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct comparison of 68%-level coverage for:

    1. central Monte-Carlo interval on |pi_E|;
    2. pi_E,N +/- 1 sigma;
    3. pi_E,E +/- 1 sigma;
    4. the 68.27% 2D covariance ellipse.

NO simulation.
NO fitting.

The true Cartesian parallax components are NOT reconstructed here.
They are loaded from the validated output of script 25:

    piE_true_components_and_recovery.parquet

The Monte-Carlo amplitude interval is loaded from script 22:

    piE_amplitude_mc_event_intervals.parquet

The H1 covariance is loaded from the positive truth table.

The purpose is to make explicit the comparison that was previously
distributed across several characterization figures:

    - for weak/confused events, the central interval on |pi_E|
      can strongly undercover;
    - the Cartesian component intervals and 2D covariance ellipse
      remain approximately calibrated or conservative;
    - for detected events, the amplitude interval behaves much better.
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


# ============================================================
# Configuration
# ============================================================

ROOT = Path("analysis/lrt")

COMPONENT_FILE = (
    ROOT
    / "results"
    / "characterization"
    / "25_piE_component_recovery_vs_truth"
    / "piE_true_components_and_recovery.parquet"
)

AMPLITUDE_MC_FILE = (
    ROOT
    / "results"
    / "characterization"
    / "22_piE_amplitude_uncertainty_mc_vs_truth"
    / "piE_amplitude_mc_event_intervals.parquet"
)

POSITIVE_TRUTH_FILE = (
    ROOT
    / "results"
    / "characterization"
    / "18_positive_truth"
    / "h1_positive_truth_characterization.parquet"
)

OUT_RESULT_DIR = (
    ROOT
    / "results"
    / "characterization"
    / "33_piE_coverage_comparison_vs_true_amp"
)

OUT_FIG_DIR = (
    ROOT
    / "figures"
    / "characterization"
    / "33_piE_coverage_comparison_vs_true_amp"
)

OUT_TABLE = (
    OUT_RESULT_DIR
    / "piE_coverage_comparison_vs_true_amp_bins.csv"
)

OUT_SUMMARY = (
    OUT_RESULT_DIR
    / "piE_coverage_comparison_summary.csv"
)

OUT_FIGURE = (
    OUT_FIG_DIR
    / "piE_coverage_comparison_vs_true_amp.png"
)

N_BINS = 12
MIN_COUNT_PER_CURVE = 100

P68 = 0.6826894921370859

# For chi-square with 2 dof:
#
#   CDF(x) = 1 - exp(-x/2)
#
# therefore the 68.27% contour is
#
#   x = -2 ln(1-P68)
#
CHI2_2D_68 = -2.0 * np.log(1.0 - P68)

CLASSES = [
    "confused_fspl_no_parallax",
    "parallax_detected",
]

CLASS_LABELS = {
    "confused_fspl_no_parallax": "Confused with no-parallax FSPL",
    "parallax_detected": "Parallax detected",
}


# ============================================================
# Helpers
# ============================================================

def wilson_interval(k, n, z=1.959963984540054):
    """
    Wilson 95% confidence interval for a binomial proportion.
    """

    if n <= 0:
        return np.nan, np.nan

    p = k / n

    denom = 1.0 + z**2 / n

    center = (
        p + z**2 / (2.0 * n)
    ) / denom

    half = (
        z
        / denom
        * np.sqrt(
            p * (1.0 - p) / n
            + z**2 / (4.0 * n**2)
        )
    )

    return center - half, center + half


def compute_coverage(covered, valid):
    """
    Return coverage and number of valid events.
    """

    covered = np.asarray(covered, dtype=bool)
    valid = np.asarray(valid, dtype=bool)

    n = int(valid.sum())

    if n == 0:
        return np.nan, 0, np.nan, np.nan

    k = int(covered[valid].sum())
    p = k / n

    lo, hi = wilson_interval(k, n)

    return float(p), n, float(lo), float(hi)


def build_common_quantile_bins(values, n_bins):
    """
    Build common bins in log10(|pi_E|), with approximately equal
    population per bin in the complete positive-T sample.

    Using quantile bins avoids extremely sparse bins at the tails while
    preserving a logarithmic physical x-axis.
    """

    x = np.asarray(values, dtype=float)

    good = np.isfinite(x) & (x > 0.0)

    logx = np.log10(x[good])

    q = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    log_edges = np.quantile(
        logx,
        q,
    )

    # Protect against repeated edges.
    log_edges = np.unique(log_edges)

    if len(log_edges) < 3:
        raise RuntimeError(
            "Could not construct enough distinct |pi_E| bin edges."
        )

    edges = 10.0**log_edges

    # Include floating-point edge cases exactly.
    edges[0] = np.nextafter(
        edges[0],
        -np.inf,
    )

    edges[-1] = np.nextafter(
        edges[-1],
        np.inf,
    )

    return edges


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 100)
    print("33: DIRECT COVERAGE COMPARISON")
    print("|piE| central interval vs components vs 2D ellipse")
    print("=" * 100)

    OUT_RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load validated true components from script 25
    # ========================================================

    comp = pd.read_parquet(
        COMPONENT_FILE
    )

    print()
    print("Component-recovery table")
    print("  file =", COMPONENT_FILE)
    print("  rows =", len(comp))

    required_comp = [
        "catalog_row",
        "positive_lrt_class",
        "piE_true_amp",
        "piEN_true",
        "piEE_true",
        "h1_piEN",
        "h1_piEE",
    ]

    missing = [
        c for c in required_comp
        if c not in comp.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing columns in component table: "
            + str(missing)
        )

    comp = comp[
        required_comp
    ].copy()

    # ========================================================
    # Load MC amplitude intervals from script 22
    # ========================================================

    amp = pd.read_parquet(
        AMPLITUDE_MC_FILE
    )

    print()
    print("Amplitude-MC table")
    print("  file =", AMPLITUDE_MC_FILE)
    print("  rows =", len(amp))

    required_amp = [
        "catalog_row",
        "mc_valid_covariance",
        "mc_q16_piE_amp",
        "mc_q84_piE_amp",
    ]

    missing = [
        c for c in required_amp
        if c not in amp.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing columns in amplitude MC table: "
            + str(missing)
        )

    amp = amp[
        required_amp
    ].copy()

    # ========================================================
    # Load H1 covariance from positive truth table
    # ========================================================

    truth = pd.read_parquet(
        POSITIVE_TRUTH_FILE
    )

    print()
    print("Positive-truth table")
    print("  file =", POSITIVE_TRUTH_FILE)
    print("  rows =", len(truth))

    required_truth = [
        "catalog_row",
        "h1_sigma_piEN",
        "h1_sigma_piEE",
        "h1_cov_piEN_piEN",
        "h1_cov_piEN_piEE",
        "h1_cov_piEE_piEE",
    ]

    missing = [
        c for c in required_truth
        if c not in truth.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing covariance columns in truth table: "
            + str(missing)
        )

    truth = truth[
        required_truth
    ].copy()

    # ========================================================
    # Merge
    # ========================================================

    df = comp.merge(
        amp,
        on="catalog_row",
        how="left",
        validate="one_to_one",
    )

    df = df.merge(
        truth,
        on="catalog_row",
        how="left",
        validate="one_to_one",
    )

    print()
    print("Merged rows =", len(df))

    if len(df) != len(comp):
        raise RuntimeError(
            "Merge changed number of component rows."
        )

    # ========================================================
    # 1. Coverage of central MC amplitude interval
    # ========================================================

    piE_true = df[
        "piE_true_amp"
    ].to_numpy(dtype=float)

    q16 = df[
        "mc_q16_piE_amp"
    ].to_numpy(dtype=float)

    q84 = df[
        "mc_q84_piE_amp"
    ].to_numpy(dtype=float)

    mc_valid_flag = df[
        "mc_valid_covariance"
    ].fillna(False).to_numpy(dtype=bool)

    amp_valid = (
        mc_valid_flag
        & np.isfinite(piE_true)
        & np.isfinite(q16)
        & np.isfinite(q84)
        & (q84 >= q16)
    )

    amp_covered = (
        amp_valid
        & (piE_true >= q16)
        & (piE_true <= q84)
    )

    df["amp_interval_valid"] = amp_valid
    df["amp_interval_covered"] = amp_covered

    # ========================================================
    # 2. Component one-sigma coverage
    # ========================================================

    piEN_true = df[
        "piEN_true"
    ].to_numpy(dtype=float)

    piEE_true = df[
        "piEE_true"
    ].to_numpy(dtype=float)

    piEN_fit = df[
        "h1_piEN"
    ].to_numpy(dtype=float)

    piEE_fit = df[
        "h1_piEE"
    ].to_numpy(dtype=float)

    sigma_N = df[
        "h1_sigma_piEN"
    ].to_numpy(dtype=float)

    sigma_E = df[
        "h1_sigma_piEE"
    ].to_numpy(dtype=float)

    N_valid = (
        np.isfinite(piEN_true)
        & np.isfinite(piEN_fit)
        & np.isfinite(sigma_N)
        & (sigma_N >= 0.0)
    )

    E_valid = (
        np.isfinite(piEE_true)
        & np.isfinite(piEE_fit)
        & np.isfinite(sigma_E)
        & (sigma_E >= 0.0)
    )

    N_covered = (
        N_valid
        & (
            np.abs(
                piEN_fit - piEN_true
            )
            <= sigma_N
        )
    )

    E_covered = (
        E_valid
        & (
            np.abs(
                piEE_fit - piEE_true
            )
            <= sigma_E
        )
    )

    df["piEN_interval_valid"] = N_valid
    df["piEN_interval_covered"] = N_covered

    df["piEE_interval_valid"] = E_valid
    df["piEE_interval_covered"] = E_covered

    # ========================================================
    # 3. 2D covariance ellipse coverage
    # ========================================================

    C_NN = df[
        "h1_cov_piEN_piEN"
    ].to_numpy(dtype=float)

    C_NE = df[
        "h1_cov_piEN_piEE"
    ].to_numpy(dtype=float)

    C_EE = df[
        "h1_cov_piEE_piEE"
    ].to_numpy(dtype=float)

    det = (
        C_NN * C_EE
        - C_NE**2
    )

    ellipse_valid = (
        np.isfinite(piEN_true)
        & np.isfinite(piEE_true)
        & np.isfinite(piEN_fit)
        & np.isfinite(piEE_fit)
        & np.isfinite(C_NN)
        & np.isfinite(C_NE)
        & np.isfinite(C_EE)
        & (C_NN > 0.0)
        & (C_EE > 0.0)
        & (det > 0.0)
    )

    dN = (
        piEN_fit
        - piEN_true
    )

    dE = (
        piEE_fit
        - piEE_true
    )

    D2 = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    D2[ellipse_valid] = (
        C_EE[ellipse_valid]
        * dN[ellipse_valid]**2

        - 2.0
        * C_NE[ellipse_valid]
        * dN[ellipse_valid]
        * dE[ellipse_valid]

        + C_NN[ellipse_valid]
        * dE[ellipse_valid]**2
    ) / det[ellipse_valid]

    ellipse_covered = (
        ellipse_valid
        & (D2 <= CHI2_2D_68)
    )

    df["ellipse_valid"] = ellipse_valid
    df["ellipse_D2"] = D2
    df["ellipse_covered"] = ellipse_covered

    # ========================================================
    # Common |pi_E| bins
    # ========================================================

    edges = build_common_quantile_bins(
        piE_true,
        N_BINS,
    )

    df["piE_bin"] = pd.cut(
        df["piE_true_amp"],
        bins=edges,
        include_lowest=True,
        ordered=True,
    )

    # ========================================================
    # Class-level summaries
    # ========================================================

    summary_rows = []

    print()
    print("=" * 100)
    print("GLOBAL COVERAGE BY CLASS")
    print("=" * 100)

    definitions = [
        (
            "amp_mc_q16_q84",
            "amp_interval_covered",
            "amp_interval_valid",
        ),
        (
            "piEN_pm_1sigma",
            "piEN_interval_covered",
            "piEN_interval_valid",
        ),
        (
            "piEE_pm_1sigma",
            "piEE_interval_covered",
            "piEE_interval_valid",
        ),
        (
            "ellipse_2d_68",
            "ellipse_covered",
            "ellipse_valid",
        ),
    ]

    for cls in CLASSES:

        sub = df.loc[
            df["positive_lrt_class"] == cls
        ]

        print()
        print(cls)
        print("-" * 70)

        for label, cov_col, valid_col in definitions:

            coverage, n, lo, hi = compute_coverage(
                sub[cov_col].to_numpy(),
                sub[valid_col].to_numpy(),
            )

            print(
                f"{label:20s} "
                f"coverage={coverage:.6f} "
                f"N={n:,} "
                f"Wilson95=[{lo:.6f}, {hi:.6f}]"
            )

            summary_rows.append(
                {
                    "positive_lrt_class": cls,
                    "method": label,
                    "coverage": coverage,
                    "n_valid": n,
                    "wilson95_low": lo,
                    "wilson95_high": hi,
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    # ========================================================
    # Binned summaries
    # ========================================================

    rows = []

    categories = df[
        "piE_bin"
    ].cat.categories

    for cls in CLASSES:

        sub_class = df.loc[
            df["positive_lrt_class"] == cls
        ]

        for ibin, interval in enumerate(categories):

            sub = sub_class.loc[
                sub_class["piE_bin"] == interval
            ]

            if len(sub) == 0:
                continue

            x_median = float(
                np.nanmedian(
                    sub["piE_true_amp"]
                )
            )

            row = {
                "positive_lrt_class": cls,
                "bin_index": ibin,
                "bin_left": float(interval.left),
                "bin_right": float(interval.right),
                "piE_true_median": x_median,
                "n_events": len(sub),
            }

            for label, cov_col, valid_col in definitions:

                coverage, n, lo, hi = compute_coverage(
                    sub[cov_col].to_numpy(),
                    sub[valid_col].to_numpy(),
                )

                row[f"{label}_coverage"] = coverage
                row[f"{label}_n"] = n
                row[f"{label}_wilson95_low"] = lo
                row[f"{label}_wilson95_high"] = hi

            rows.append(row)

    bins = pd.DataFrame(rows)

    bins.to_csv(
        OUT_TABLE,
        index=False,
    )

    print()
    print("Saved binned table:")
    print(" ", OUT_TABLE)

    # ========================================================
    # Plot
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.5),
        sharey=True,
    )

    plot_specs = [
        (
            "amp_mc_q16_q84",
            r"Central MC interval on $|\pi_E|$",
            "o",
            "-",
        ),
        (
            "piEN_pm_1sigma",
            r"$\pi_{E,N}\pm1\sigma$",
            "s",
            "-",
        ),
        (
            "piEE_pm_1sigma",
            r"$\pi_{E,E}\pm1\sigma$",
            "^",
            "-",
        ),
        (
            "ellipse_2d_68",
            "68% 2D covariance ellipse",
            "D",
            "-",
        ),
    ]

    for ax, cls in zip(
        axes,
        CLASSES,
    ):

        sub = bins.loc[
            bins["positive_lrt_class"] == cls
        ].sort_values(
            "piE_true_median"
        )

        for label, display, marker, linestyle in plot_specs:

            coverage_col = (
                f"{label}_coverage"
            )

            n_col = (
                f"{label}_n"
            )

            lo_col = (
                f"{label}_wilson95_low"
            )

            hi_col = (
                f"{label}_wilson95_high"
            )

            use = (
                np.isfinite(sub["piE_true_median"])
                & np.isfinite(sub[coverage_col])
                & (sub[n_col] >= MIN_COUNT_PER_CURVE)
            )

            x = sub.loc[
                use,
                "piE_true_median"
            ].to_numpy(dtype=float)

            y = sub.loc[
                use,
                coverage_col
            ].to_numpy(dtype=float)

            lo = sub.loc[
                use,
                lo_col
            ].to_numpy(dtype=float)

            hi = sub.loc[
                use,
                hi_col
            ].to_numpy(dtype=float)

            yerr = np.vstack(
                [
                    y - lo,
                    hi - y,
                ]
            )

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.7,
                markersize=5,
                capsize=2,
                label=display,
            )

        ax.axhline(
            P68,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label="Nominal 68.27%",
        )

        ax.set_xscale("log")

        ax.set_ylim(
            0.0,
            1.02,
        )

        ax.set_xlabel(
            r"True $|\pi_E|$"
        )

        n_cls = int(
            (
                df["positive_lrt_class"]
                == cls
            ).sum()
        )

        ax.set_title(
            CLASS_LABELS[cls]
            + "\n"
            + f"N = {n_cls:,}"
        )

        ax.grid(
            alpha=0.25
        )

    axes[0].set_ylabel(
        "Empirical coverage"
    )

    handles, labels = (
        axes[1].get_legend_handles_labels()
    )

    unique = {}

    for h, label in zip(
        handles,
        labels,
    ):
        if label not in unique:
            unique[label] = h

    axes[1].legend(
        unique.values(),
        unique.keys(),
        fontsize=13,
        loc="best",
    )



    label_panels(axes)
    fig.savefig(
        OUT_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    save_pdf_companion(fig, OUT_FIGURE)

    plt.close(fig)

    print()
    print("Saved figure:")
    print(" ", OUT_FIGURE)

    print()
    print("Saved global summary:")
    print(" ", OUT_SUMMARY)

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()


