#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hidden annual parallax as a bias in the recovered Einstein timescale.

NO simulation.
NO fitting.

Scientific question
-------------------
For H1-generated events in which annual parallax is present but is not
distinguishable from a no-parallax FSPL model, what happens to the
Einstein timescale if the observer adopts the no-parallax fit?

We use the existing "confused_fspl_no_parallax" population and compare

    delta_H0 = log10(tE_H0 / tE_true)

with

    delta_H1 = log10(tE_H1 / tE_true).

If H1 is approximately centered on zero while H0 is systematically
shifted, this is a direct consequence of hidden parallax:

    true parallax
        -> not distinguishable by the LRT
        -> no-parallax model adopted
        -> distorted inferred tE.

This is relevant because observed tE distributions are commonly used
for population and mass-function inference.

Input
-----
analysis/lrt/results/characterization/18_positive_truth/
    h1_positive_truth_characterization.parquet

Outputs
-------
analysis/lrt/figures/characterization/45_hidden_parallax_tE_bias/
    hidden_parallax_tE_bias.png
    hidden_parallax_tE_bias.pdf

analysis/lrt/results/characterization/45_hidden_parallax_tE_bias/
    hidden_parallax_tE_bias_vs_true_tE.csv
    hidden_parallax_tE_bias_summary.txt
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
    / "18_positive_truth"
    / "h1_positive_truth_characterization.parquet"
)

FIGURE_DIR = (
    ROOT
    / "analysis/lrt/figures/characterization"
    / "45_hidden_parallax_tE_bias"
)

RESULT_DIR = (
    ROOT
    / "analysis/lrt/results/characterization"
    / "45_hidden_parallax_tE_bias"
)

FIGURE_BASE = (
    FIGURE_DIR
    / "hidden_parallax_tE_bias"
)

BIN_TABLE = (
    RESULT_DIR
    / "hidden_parallax_tE_bias_vs_true_tE.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "hidden_parallax_tE_bias_summary.txt"
)


# ============================================================================
# Configuration
# ============================================================================

CONFUSED_CLASS = "confused_fspl_no_parallax"

EXPECTED_CONFUSED = 100747

N_TE_BINS = 18

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


def quantiles(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    return tuple(
        np.quantile(
            values,
            [
                0.16,
                0.50,
                0.84,
            ],
        )
    )


def build_log_edges(values, n_bins):

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
            "No finite positive tE values."
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


def fractional_error(fit, truth):

    out = np.full(
        len(fit),
        np.nan,
    )

    valid = (
        np.isfinite(fit)
        & np.isfinite(truth)
        & (truth > 0.0)
    )

    out[valid] = (
        fit[valid]
        / truth[valid]
        - 1.0
    )

    return out


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
    print("45: HIDDEN PARALLAX AS A BIAS IN t_E")
    print("=" * 100)

    columns = [
        "catalog_row",
        "positive_lrt_class",
        "delta_chi2_lrt",

        "tE_true_days",

        "h0_tE",
        "h1_tE",

        "piE_true_amp",
    ]

    df = pd.read_parquet(
        INPUT,
        columns=columns,
    )

    confused = (
        df[
            "positive_lrt_class"
        ].astype(str)
        == CONFUSED_CLASS
    )

    work = df.loc[
        confused
    ].copy()

    print()
    print(
        "Confused events =",
        f"{len(work):,}",
    )

    if len(work) != EXPECTED_CONFUSED:

        raise RuntimeError(
            "Confused-population audit failed: "
            f"{len(work):,} != {EXPECTED_CONFUSED:,}"
        )

    # ========================================================================
    # Core quantities
    # ========================================================================

    tE_true = numeric(
        work,
        "tE_true_days",
    )

    tE_h0 = numeric(
        work,
        "h0_tE",
    )

    tE_h1 = numeric(
        work,
        "h1_tE",
    )

    valid = (
        np.isfinite(tE_true)
        & (tE_true > 0.0)
        & np.isfinite(tE_h0)
        & (tE_h0 > 0.0)
        & np.isfinite(tE_h1)
        & (tE_h1 > 0.0)
    )

    delta_h0 = np.full(
        len(work),
        np.nan,
    )

    delta_h1 = np.full(
        len(work),
        np.nan,
    )

    delta_h0[valid] = np.log10(
        tE_h0[valid]
        / tE_true[valid]
    )

    delta_h1[valid] = np.log10(
        tE_h1[valid]
        / tE_true[valid]
    )

    frac_h0 = fractional_error(
        tE_h0,
        tE_true,
    )

    frac_h1 = fractional_error(
        tE_h1,
        tE_true,
    )

    print(
        "Valid tE recovery rows =",
        f"{int(valid.sum()):,}",
    )

    # ========================================================================
    # Binned dependence on true tE
    # ========================================================================

    edges = build_log_edges(
        tE_true[valid],
        N_TE_BINS,
    )

    rows = []

    for ibin in range(
        N_TE_BINS
    ):

        left = edges[
            ibin
        ]

        right = edges[
            ibin + 1
        ]

        mask = (
            valid
            & (tE_true >= left)
            & (tE_true < right)
        )

        h0_q16, h0_med, h0_q84 = quantiles(
            delta_h0[mask]
        )

        h1_q16, h1_med, h1_q84 = quantiles(
            delta_h1[mask]
        )

        rows.append(
            {
                "tE_left_days":
                    left,

                "tE_right_days":
                    right,

                "tE_center_days":
                    np.sqrt(
                        left
                        * right
                    ),

                "n":
                    int(mask.sum()),

                "h0_bias_q16":
                    h0_q16,

                "h0_bias_median":
                    h0_med,

                "h0_bias_q84":
                    h0_q84,

                "h1_bias_q16":
                    h1_q16,

                "h1_bias_median":
                    h1_med,

                "h1_bias_q84":
                    h1_q84,
            }
        )

    table = pd.DataFrame(
        rows
    )

    table.to_csv(
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
    # Panel A: population distribution
    # ------------------------------------------------------------------------

    combined = np.concatenate(
        [
            delta_h0[valid],
            delta_h1[valid],
        ]
    )

    robust_limit = float(
        np.nanquantile(
            np.abs(combined),
            0.995,
        )
    )

    robust_limit = max(
        robust_limit,
        0.05,
    )

    bins = np.linspace(
        -robust_limit,
        robust_limit,
        101,
    )

    axes[0].hist(
        delta_h0[valid],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        label=r"No-parallax fit ($H_0$)",
    )

    axes[0].hist(
        delta_h1[valid],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        label=r"Parallax fit ($H_1$)",
    )

    axes[0].axvline(
        0.0,
        linestyle="--",
        linewidth=1.4,
        color="black",
    )

    axes[0].set_xlabel(
        r"$\log_{10}(t_{E,\rm fit}/t_{E,\rm true})$"
    )

    axes[0].set_ylabel(
        "Probability density"
    )

    axes[0].set_title(
        "Timescale recovery for hidden-parallax events"
    )

    axes[0].grid(
        alpha=0.20,
    )

    axes[0].legend(
        frameon=True,
    )

    # ------------------------------------------------------------------------
    # Panel B: dependence on true tE
    # ------------------------------------------------------------------------

    shown = (
        table["n"].to_numpy()
        >= MIN_BIN_COUNT
    )

    sub = table.loc[
        shown
    ]

    line_h0, = axes[1].plot(
        sub["tE_center_days"],
        sub["h0_bias_median"],
        marker="o",
        markersize=4.5,
        label=r"No-parallax fit ($H_0$)",
    )

    axes[1].fill_between(
        sub["tE_center_days"],
        sub["h0_bias_q16"],
        sub["h0_bias_q84"],
        color=line_h0.get_color(),
        alpha=0.20,
    )

    line_h1, = axes[1].plot(
        sub["tE_center_days"],
        sub["h1_bias_median"],
        marker="o",
        markersize=4.5,
        label=r"Parallax fit ($H_1$)",
    )

    axes[1].fill_between(
        sub["tE_center_days"],
        sub["h1_bias_q16"],
        sub["h1_bias_q84"],
        color=line_h1.get_color(),
        alpha=0.20,
    )

    axes[1].axhline(
        0.0,
        linestyle="--",
        linewidth=1.4,
        color="black",
    )

    axes[1].set_xscale(
        "log"
    )

    axes[1].set_xlabel(
        r"True $t_E$ [days]"
    )

    axes[1].set_ylabel(
        r"$\log_{10}(t_{E,\rm fit}/t_{E,\rm true})$"
    )

    axes[1].set_title(
        "Bias versus true event timescale"
    )

    axes[1].grid(
        alpha=0.20,
    )

    axes[1].legend(
        frameon=True,
    )

    fig.suptitle(
        "Hidden annual parallax can distort a no-parallax timescale estimate",
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
        "HIDDEN PARALLAX AS A BIAS IN t_E",
        "=" * 80,
        "",
        f"N confused = {len(work):,}",
        f"N valid = {int(valid.sum()):,}",
        "",
    ]

    for label, delta, frac in [
        (
            "H0 no-parallax fit",
            delta_h0,
            frac_h0,
        ),
        (
            "H1 parallax fit",
            delta_h1,
            frac_h1,
        ),
    ]:

        use = (
            valid
            & np.isfinite(delta)
            & np.isfinite(frac)
        )

        q16, med, q84 = quantiles(
            delta[use]
        )

        lines.extend(
            [
                label,
                "-" * 60,
                (
                    "median log10(tE_fit/tE_true) = "
                    f"{med:.8f}"
                ),
                (
                    "q16--q84 = "
                    f"[{q16:.8f}, {q84:.8f}]"
                ),
                (
                    "median fractional error = "
                    f"{np.median(frac[use]):.8f}"
                ),
                (
                    "fraction |fractional error| > 10% = "
                    f"{np.mean(np.abs(frac[use]) > 0.10):.8f}"
                ),
                (
                    "fraction |fractional error| > 20% = "
                    f"{np.mean(np.abs(frac[use]) > 0.20):.8f}"
                ),
                (
                    "fraction |fractional error| > 50% = "
                    f"{np.mean(np.abs(frac[use]) > 0.50):.8f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "Interpretation:",
            (
                "For confused events, H0 represents the timescale that "
                "would be inferred if the hidden annual-parallax signal "
                "were ignored."
            ),
            (
                "The H1 curve provides the corresponding recovery when "
                "the correct parallax model is fitted."
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
    print(" ", BIN_TABLE)
    print(" ", SUMMARY_FILE)


if __name__ == "__main__":
    main()
