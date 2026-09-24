#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
From annual-parallax detection to useful parallax measurement.

NO simulation.
NO fitting.
NO new Monte Carlo.

We reuse the event-level Monte-Carlo amplitude intervals generated in
characterization script 22.

Scientific question
-------------------
Among events for which annual parallax is statistically distinguishable,
what fraction also has a useful measurement of the parallax amplitude?

For each event define the observational relative uncertainty

                   (q84 - q16) / 2
    NU_piE =       ----------------
                         q50

where q16, q50 and q84 are the Monte-Carlo quantiles of |pi_E| obtained
from the fitted H1 parallax-component covariance.

This differs deliberately from sigma/piE_true:
piE_true is unavailable observationally and is therefore not used in
the measurement-quality selection.

Primary useful-measurement criterion
------------------------------------
    NU_piE < 0.20

We report:
    - fraction of all parallax-detected events passing the criterion;
    - fraction conditional on a valid MC interval;
    - threshold scan for 0.1, 0.2, 0.3, 0.5 and 1.0;
    - dependence of the useful-measurement fraction on true |pi_E|.

Input
-----
analysis/lrt/results/characterization/
    22_piE_amplitude_uncertainty_mc_vs_truth/
    piE_amplitude_mc_event_intervals.parquet

Outputs
-------
analysis/lrt/figures/characterization/
    46_parallax_measurement_precision/
    parallax_measurement_precision.png
    parallax_measurement_precision.pdf

analysis/lrt/results/characterization/
    46_parallax_measurement_precision/
    precision_threshold_summary.csv
    useful_fraction_vs_true_piE.csv
    parallax_measurement_precision_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT
    / "analysis/lrt/results/characterization"
    / "22_piE_amplitude_uncertainty_mc_vs_truth"
    / "piE_amplitude_mc_event_intervals.parquet"
)

FIGURE_DIR = (
    ROOT
    / "analysis/lrt/figures/characterization"
    / "46_parallax_measurement_precision"
)

RESULT_DIR = (
    ROOT
    / "analysis/lrt/results/characterization"
    / "46_parallax_measurement_precision"
)

FIGURE_BASE = (
    FIGURE_DIR
    / "parallax_measurement_precision"
)

THRESHOLD_TABLE = (
    RESULT_DIR
    / "precision_threshold_summary.csv"
)

BIN_TABLE = (
    RESULT_DIR
    / "useful_fraction_vs_true_piE.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "parallax_measurement_precision_summary.txt"
)


# ============================================================================
# Configuration
# ============================================================================

DETECTED_CLASS = "parallax_detected"

EXPECTED_DETECTED = 229921

PRIMARY_NU_THRESHOLD = 0.20

REPORT_THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.50,
    1.00,
]

N_PIE_BINS = 18

MIN_BIN_COUNT = 50


# ============================================================================
# Style
# ============================================================================

def set_paper_style():

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 19,
            "axes.titlesize": 18,
            "legend.fontsize": 13,

            "xtick.labelsize": 15,
            "ytick.labelsize": 15,

            "axes.linewidth": 0.9,
            "lines.linewidth": 2.0,

            "xtick.direction": "in",
            "ytick.direction": "in",

            "xtick.top": True,
            "ytick.right": True,

            "savefig.dpi": 300,
            "savefig.bbox": "tight",

            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )


# ============================================================================
# Helpers
# ============================================================================

def numeric(df, column):

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def wilson_interval(
    k,
    n,
    z=1.959963984540054,
):

    if n <= 0:

        return (
            np.nan,
            np.nan,
        )

    p = (
        k
        / n
    )

    denominator = (
        1.0
        + z**2 / n
    )

    center = (
        p
        + z**2
        / (
            2.0 * n
        )
    ) / denominator

    half = (
        z
        * np.sqrt(
            p
            * (
                1.0 - p
            )
            / n
            + z**2
            / (
                4.0 * n**2
            )
        )
        / denominator
    )

    return (
        center - half,
        center + half,
    )


def build_log_edges(
    values,
    n_bins,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & (values > 0.0)
    )

    if not np.any(valid):

        raise RuntimeError(
            "No finite positive piE values."
        )

    lo = float(
        np.nanmin(
            values[valid]
        )
    )

    hi = float(
        np.nanmax(
            values[valid]
        )
    )

    edges = np.geomspace(
        lo,
        hi,
        n_bins + 1,
    )

    edges[0] = np.nextafter(
        lo,
        -np.inf,
    )

    edges[-1] = np.nextafter(
        hi,
        np.inf,
    )

    return edges


def save_figure(fig):

    png = FIGURE_BASE.with_suffix(
        ".png"
    )

    pdf = FIGURE_BASE.with_suffix(
        ".pdf"
    )

    fig.savefig(png)
    fig.savefig(pdf)

    print()
    print("Saved:")
    print(" ", png)
    print(" ", pdf)


# ============================================================================
# Main
# ============================================================================

def main():

    set_paper_style()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "46: FROM PARALLAX DETECTION TO USEFUL PARALLAX MEASUREMENT"
    )
    print("=" * 100)

    columns = [
        "catalog_row",
        "positive_lrt_class",
        "piE_true_amp",

        "mc_valid_covariance",

        "mc_q16_piE_amp",
        "mc_q50_piE_amp",
        "mc_q84_piE_amp",
    ]

    df = pd.read_parquet(
        INPUT,
        columns=columns,
    )

    # ========================================================================
    # Restrict to statistically detected annual parallax
    # ========================================================================

    detected = (
        df[
            "positive_lrt_class"
        ].astype(str)
        == DETECTED_CLASS
    )

    work = df.loc[
        detected
    ].copy()

    print()
    print(
        "Parallax-detected events =",
        f"{len(work):,}",
    )

    if len(work) != EXPECTED_DETECTED:

        raise RuntimeError(
            "Detected-population audit failed: "
            f"{len(work):,} != {EXPECTED_DETECTED:,}"
        )

    # ========================================================================
    # Build observational NU_piE
    # ========================================================================

    piE_true = numeric(
        work,
        "piE_true_amp",
    )

    q16 = numeric(
        work,
        "mc_q16_piE_amp",
    )

    q50 = numeric(
        work,
        "mc_q50_piE_amp",
    )

    q84 = numeric(
        work,
        "mc_q84_piE_amp",
    )

    covariance_valid = (
        work[
            "mc_valid_covariance"
        ]
        .fillna(False)
        .to_numpy(dtype=bool)
    )

    sigma68 = (
        0.5
        * (
            q84
            - q16
        )
    )

    valid_interval = (
        covariance_valid
        & np.isfinite(q16)
        & np.isfinite(q50)
        & np.isfinite(q84)
        & (q16 <= q50)
        & (q50 <= q84)
        & (q50 > 0.0)
        & np.isfinite(sigma68)
        & (sigma68 >= 0.0)
    )

    nu_piE = np.full(
        len(work),
        np.nan,
    )

    nu_piE[
        valid_interval
    ] = (
        sigma68[
            valid_interval
        ]
        / q50[
            valid_interval
        ]
    )

    useful = (
        valid_interval
        & np.isfinite(nu_piE)
        & (
            nu_piE
            < PRIMARY_NU_THRESHOLD
        )
    )

    n_total = len(
        work
    )

    n_valid = int(
        valid_interval.sum()
    )

    n_useful = int(
        useful.sum()
    )

    print(
        "Valid amplitude intervals =",
        f"{n_valid:,}",
    )

    print(
        (
            rf"NU_piE < {PRIMARY_NU_THRESHOLD:.2f} ="
        ),
        f"{n_useful:,}",
    )

    print(
        "Useful fraction among all detections =",
        f"{n_useful / n_total:.8f}",
    )

    print(
        "Useful fraction among valid intervals =",
        f"{n_useful / n_valid:.8f}",
    )

    # ========================================================================
    # Threshold summary
    # ========================================================================

    rows = []

    for threshold in REPORT_THRESHOLDS:

        passing = (
            valid_interval
            & np.isfinite(nu_piE)
            & (
                nu_piE
                < threshold
            )
        )

        n_passing = int(
            passing.sum()
        )

        rows.append(
            {
                "nu_threshold":
                    threshold,

                "n_parallax_detected":
                    n_total,

                "n_valid_interval":
                    n_valid,

                "n_passing":
                    n_passing,

                "fraction_of_all_detected":
                    n_passing
                    / n_total,

                "fraction_of_valid_intervals":
                    n_passing
                    / n_valid,
            }
        )

    threshold_table = pd.DataFrame(
        rows
    )

    threshold_table.to_csv(
        THRESHOLD_TABLE,
        index=False,
    )

    # ========================================================================
    # Continuous cumulative curve
    #
    # Denominator = ALL parallax-detected events.
    #
    # Therefore events without a usable covariance do not silently disappear;
    # they simply never satisfy a finite precision threshold.
    # ========================================================================

    valid_nu_values = nu_piE[
        valid_interval
        & np.isfinite(nu_piE)
        & (nu_piE > 0.0)
    ]

    if len(valid_nu_values) == 0:

        raise RuntimeError(
            "No valid NU_piE values."
        )

    threshold_min = max(
        1.0e-3,
        float(
            np.nanquantile(
                valid_nu_values,
                0.001,
            )
        )
        / 2.0,
    )

    threshold_max = max(
        2.0,
        float(
            np.nanquantile(
                valid_nu_values,
                0.995,
            )
        ),
    )

    cdf_thresholds = np.geomspace(
        threshold_min,
        threshold_max,
        300,
    )

    cumulative_fraction = np.array(
        [
            np.sum(
                valid_interval
                & np.isfinite(nu_piE)
                & (
                    nu_piE
                    < threshold
                )
            )
            / n_total
            for threshold in cdf_thresholds
        ],
        dtype=float,
    )

    # ========================================================================
    # Useful-measurement fraction versus true piE
    # ========================================================================

    valid_truth = (
        np.isfinite(piE_true)
        & (piE_true > 0.0)
    )

    edges = build_log_edges(
        piE_true[
            valid_truth
        ],
        N_PIE_BINS,
    )

    bin_rows = []

    for ibin in range(
        N_PIE_BINS
    ):

        left = edges[
            ibin
        ]

        right = edges[
            ibin + 1
        ]

        in_bin = (
            valid_truth
            & (piE_true >= left)
            & (piE_true < right)
        )

        n = int(
            in_bin.sum()
        )

        k = int(
            useful[
                in_bin
            ].sum()
        )

        fraction = (
            k / n
            if n > 0
            else np.nan
        )

        low, high = wilson_interval(
            k,
            n,
        )

        bin_rows.append(
            {
                "piE_left":
                    left,

                "piE_right":
                    right,

                "piE_center":
                    np.sqrt(
                        left
                        * right
                    ),

                "n_detected":
                    n,

                "n_useful":
                    k,

                "useful_fraction":
                    fraction,

                "wilson_low":
                    low,

                "wilson_high":
                    high,
            }
        )

    bin_table = pd.DataFrame(
        bin_rows
    )

    bin_table.to_csv(
        BIN_TABLE,
        index=False,
    )

    # ========================================================================
    # Figure
    # ========================================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            13.0,
            5.4,
        ),
    )

    # ------------------------------------------------------------------------
    # Panel A: how many detected parallaxes are measured precisely?
    # ------------------------------------------------------------------------

    axes[0].plot(
        cdf_thresholds,
        cumulative_fraction,
    )

    axes[0].axvline(
        PRIMARY_NU_THRESHOLD,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label=(
            rf"Useful measurement: "
            rf"${{\rm NU}}_{{\pi_E}}<{PRIMARY_NU_THRESHOLD:.1f}$"
        ),
    )

    overall_useful_fraction = (
        n_useful
        / n_total
    )

    axes[0].axhline(
        overall_useful_fraction,
        linestyle=":",
        linewidth=1.3,
        color="black",
    )

    axes[0].scatter(
        [
            PRIMARY_NU_THRESHOLD
        ],
        [
            overall_useful_fraction
        ],
        zorder=5,
    )

    axes[0].set_xscale(
        "log"
    )

    axes[0].set_ylim(
        0.0,
        1.02,
    )

    axes[0].set_xlabel(
        (
            r"Precision requirement "
            r"${\rm NU}_{\pi_E}"
            r"=\sigma_{68}(|\boldsymbol{\pi}_E|)"
            r"/q_{50}(|\boldsymbol{\pi}_E|)$"
        )
    )

    axes[0].set_ylabel(
        (
            "Fraction of parallax-detected events "
            "meeting requirement"
        )
    )

    axes[0].set_title(
        "Detection does not necessarily imply precise measurement"
    )

    axes[0].grid(
        alpha=0.20,
    )

    axes[0].legend(
        frameon=True,
    )

    axes[0].text(
        0.05,
        0.93,
        (
            rf"${{\rm NU}}_{{\pi_E}}<0.2$: "
            f"{n_useful:,}/{n_total:,}\n"
            f"= {overall_useful_fraction:.3f}"
        ),
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle":
                "round",

            "facecolor":
                "white",

            "alpha":
                0.85,
        },
    )

    # ------------------------------------------------------------------------
    # Panel B: dependence on true piE
    # ------------------------------------------------------------------------

    shown = (
        bin_table[
            "n_detected"
        ].to_numpy()
        >= MIN_BIN_COUNT
    )

    sub = bin_table.loc[
        shown
    ]

    axes[1].plot(
        sub[
            "piE_center"
        ],
        sub[
            "useful_fraction"
        ],
        marker="o",
        markersize=4.5,
    )

    axes[1].fill_between(
        sub[
            "piE_center"
        ],
        sub[
            "wilson_low"
        ],
        sub[
            "wilson_high"
        ],
        alpha=0.20,
    )

    axes[1].axhline(
        overall_useful_fraction,
        linestyle="--",
        linewidth=1.3,
        color="black",
        label="Overall useful-measurement fraction",
    )

    axes[1].set_xscale(
        "log"
    )

    axes[1].set_ylim(
        0.0,
        1.02,
    )

    axes[1].set_xlabel(
        r"True $|\boldsymbol{\pi}_E|$"
    )

    axes[1].set_ylabel(
        (
            rf"$P({{\rm NU}}_{{\pi_E}}"
            rf"<{PRIMARY_NU_THRESHOLD:.1f}"
            rf"\mid \mathrm{{parallax\ detected}})$"
        )
    )

    axes[1].set_title(
        "Fraction with a useful parallax-amplitude measurement"
    )

    axes[1].grid(
        alpha=0.20,
    )

    axes[1].legend(
        frameon=True,
    )

    fig.suptitle(
        "From annual-parallax detection to annual-parallax measurement",
        fontsize=20.2,
    )

    fig.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.94,
        ]
    )

    save_figure(
        fig
    )

    plt.close(
        fig
    )

    # ========================================================================
    # Summary
    # ========================================================================

    lines = [
        "FROM PARALLAX DETECTION TO USEFUL PARALLAX MEASUREMENT",
        "=" * 80,
        "",
        f"N parallax detected = {n_total:,}",
        f"N valid MC amplitude intervals = {n_valid:,}",
        (
            "valid-interval fraction among detected = "
            f"{n_valid / n_total:.8f}"
        ),
        "",
        (
            "Observational precision metric:"
        ),
        (
            "  NU_piE = 0.5*(q84-q16)/q50"
        ),
        "",
        "Threshold scan:",
    ]

    for _, row in threshold_table.iterrows():

        lines.append(
            (
                f"  NU_piE < {row['nu_threshold']:.2f}: "
                f"N={int(row['n_passing']):,}, "
                f"fraction of all detected="
                f"{row['fraction_of_all_detected']:.8f}, "
                f"fraction of valid intervals="
                f"{row['fraction_of_valid_intervals']:.8f}"
            )
        )

    lines.extend(
        [
            "",
            (
                "Primary useful-measurement criterion:"
            ),
            (
                f"  NU_piE < {PRIMARY_NU_THRESHOLD:.2f}"
            ),
            (
                f"  useful events = {n_useful:,}"
            ),
            (
                f"  fraction of all parallax detections = "
                f"{n_useful / n_total:.8f}"
            ),
            (
                f"  fraction conditional on valid MC interval = "
                f"{n_useful / n_valid:.8f}"
            ),
            "",
            "Interpretation:",
            (
                "The LRT answers whether annual parallax is statistically "
                "distinguishable. NU_piE asks the separate downstream "
                "question of whether the detected parallax amplitude is "
                "measured precisely enough for physical inference."
            ),
        ]
    )

    SUMMARY_FILE.write_text(
        "\n".join(lines)
        + "\n"
    )

    print()
    print(
        "\n".join(lines)
    )

    print()
    print("Saved:")
    print(" ", THRESHOLD_TABLE)
    print(" ", BIN_TABLE)
    print(" ", SUMMARY_FILE)


if __name__ == "__main__":
    main()
