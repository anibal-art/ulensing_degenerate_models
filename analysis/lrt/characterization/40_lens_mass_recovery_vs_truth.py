#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Population-level lens-mass recovery for the parallax-only controlled
experiment.

NO simulation.
NO fitting.

Input
-----
analysis/lrt/results/characterization/
    39_lens_mass_parallax_only/
    lens_mass_parallax_only_events.parquet

Figures
-------
1. Lens-mass bias versus true lens mass.

2. Lens-mass bias versus true parallax amplitude.

The recovery statistic is

    log10(Mhat_L / M_L,true)

with theta_E held fixed to truth.

Detected and confused events are shown separately.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT
    / "analysis/lrt/results/characterization"
    / "39_lens_mass_parallax_only"
    / "lens_mass_parallax_only_events.parquet"
)

FIGURE_DIR = (
    ROOT
    / "analysis/lrt/figures/characterization"
    / "40_lens_mass_recovery_vs_truth"
)

RESULT_DIR = (
    ROOT
    / "analysis/lrt/results/characterization"
    / "40_lens_mass_recovery_vs_truth"
)

FIGURE_MASS = (
    FIGURE_DIR
    / "lens_mass_recovery_vs_true_mass.png"
)

FIGURE_MASS_PDF = (
    FIGURE_DIR
    / "lens_mass_recovery_vs_true_mass.pdf"
)

FIGURE_PIE = (
    FIGURE_DIR
    / "lens_mass_recovery_vs_true_piE.png"
)

FIGURE_PIE_PDF = (
    FIGURE_DIR
    / "lens_mass_recovery_vs_true_piE.pdf"
)

BIN_TABLE = (
    RESULT_DIR
    / "lens_mass_recovery_bins.csv"
)


# ============================================================
# Configuration
# ============================================================

N_BINS = 22
MIN_COUNT = 30

CLASSES = [
    "confused_fspl_no_parallax",
    "parallax_detected",
]

CLASS_LABELS = {
    "confused_fspl_no_parallax":
        "Confused with no-parallax FSPL",

    "parallax_detected":
        "Parallax detected",
}


# ============================================================
# Paper style
# ============================================================

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
            "lines.linewidth": 1.8,

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


# ============================================================
# Helpers
# ============================================================

def numeric(df, column):

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def make_log_edges(values, n_bins):

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
            "No positive finite values available for binning."
        )

    vmin = float(
        np.nanmin(values[valid])
    )

    vmax = float(
        np.nanmax(values[valid])
    )

    edges = np.geomspace(
        vmin,
        vmax,
        n_bins + 1,
    )

    # Include exact extrema robustly.
    edges[0] = np.nextafter(
        vmin,
        -np.inf,
    )

    edges[-1] = np.nextafter(
        vmax,
        np.inf,
    )

    return edges


def binned_quantiles(
    x,
    y,
    classes,
    edges,
):

    rows = []

    for cls in CLASSES:

        class_mask = (
            classes
            == cls
        )

        for ibin in range(
            len(edges) - 1
        ):

            left = edges[ibin]
            right = edges[ibin + 1]

            mask = (
                class_mask
                & np.isfinite(x)
                & (x >= left)
                & (x < right)
                & np.isfinite(y)
            )

            values = y[mask]

            n = len(values)

            row = {
                "class": cls,
                "bin_index": ibin,
                "x_left": left,
                "x_right": right,
                "x_center": np.sqrt(
                    left * right
                ),
                "n": n,
                "q16": np.nan,
                "median": np.nan,
                "q84": np.nan,
            }

            if n > 0:

                row["q16"] = np.nanpercentile(
                    values,
                    16,
                )

                row["median"] = np.nanmedian(
                    values
                )

                row["q84"] = np.nanpercentile(
                    values,
                    84,
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def plot_binned_recovery(
    table,
    xlabel,
    title,
    png_path,
    pdf_path,
):

    fig, ax = plt.subplots(
        figsize=(8.0, 5.6)
    )

    for cls in CLASSES:

        sub = table[
            (table["class"] == cls)
            & (table["n"] >= MIN_COUNT)
        ].copy()

        if len(sub) == 0:
            continue

        line, = ax.plot(
            sub["x_center"],
            sub["median"],
            marker="o",
            markersize=4.5,
            label=CLASS_LABELS[cls],
        )

        ax.fill_between(
            sub["x_center"],
            sub["q16"],
            sub["q84"],
            alpha=0.20,
            color=line.get_color(),
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.3,
        color="black",
        alpha=0.8,
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        r"$\log_{10}(\widehat{M}_L/M_{L,\rm true})$"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        alpha=0.20,
    )

    ax.legend(
        frameon=True,
    )

    fig.tight_layout()

    fig.savefig(
        png_path
    )

    fig.savefig(
        pdf_path
    )

    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================

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

    if not INPUT.exists():
        raise FileNotFoundError(
            INPUT
        )

    df = pd.read_parquet(
        INPUT
    )

    required = [
        "positive_lrt_class",
        "lens_mass_true_msun",
        "piE_true_amp",
        "mass_point_valid",
        "delta_log10_mass",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    classes = (
        df["positive_lrt_class"]
        .astype(str)
        .to_numpy()
    )

    valid = (
        df["mass_point_valid"]
        .fillna(False)
        .to_numpy(dtype=bool)
    )

    mass_true = numeric(
        df,
        "lens_mass_true_msun",
    )

    piE_true = numeric(
        df,
        "piE_true_amp",
    )

    delta_mass = numeric(
        df,
        "delta_log10_mass",
    )

    finite = (
        valid
        & np.isfinite(mass_true)
        & (mass_true > 0.0)
        & np.isfinite(piE_true)
        & (piE_true > 0.0)
        & np.isfinite(delta_mass)
    )

    # ========================================================
    # Figure 1: mass recovery vs true lens mass
    # ========================================================

    mass_edges = make_log_edges(
        mass_true[finite],
        N_BINS,
    )

    mass_table = binned_quantiles(
        mass_true,
        delta_mass,
        classes,
        mass_edges,
    )

    mass_table[
        "x_variable"
    ] = "lens_mass_true_msun"

    plot_binned_recovery(
        mass_table,
        xlabel=r"True lens mass $M_{L,\rm true}\ [M_\odot]$",
        title="Parallax-only lens-mass recovery versus true lens mass",
        png_path=FIGURE_MASS,
        pdf_path=FIGURE_MASS_PDF,
    )

    # ========================================================
    # Figure 2: mass recovery vs true piE
    # ========================================================

    piE_edges = make_log_edges(
        piE_true[finite],
        N_BINS,
    )

    piE_table = binned_quantiles(
        piE_true,
        delta_mass,
        classes,
        piE_edges,
    )

    piE_table[
        "x_variable"
    ] = "piE_true_amp"

    plot_binned_recovery(
        piE_table,
        xlabel=r"True $|\boldsymbol{\pi}_E|$",
        title="Parallax-only lens-mass recovery versus true parallax",
        png_path=FIGURE_PIE,
        pdf_path=FIGURE_PIE_PDF,
    )

    # ========================================================
    # Save binned table
    # ========================================================

    combined = pd.concat(
        [
            mass_table,
            piE_table,
        ],
        ignore_index=True,
    )

    combined.to_csv(
        BIN_TABLE,
        index=False,
    )

    # ========================================================
    # Diagnostics
    # ========================================================

    print("=" * 90)
    print("LENS-MASS RECOVERY")
    print("=" * 90)

    print(
        "valid events =",
        int(finite.sum()),
    )

    for cls in CLASSES:

        mask = (
            finite
            & (classes == cls)
        )

        if not np.any(mask):
            continue

        values = delta_mass[
            mask
        ]

        print()
        print(cls)
        print(
            "  N =",
            int(mask.sum()),
        )
        print(
            "  median log10(Mhat/Mtrue) =",
            float(np.nanmedian(values)),
        )
        print(
            "  16--84% =",
            (
                float(
                    np.nanpercentile(
                        values,
                        16,
                    )
                ),
                float(
                    np.nanpercentile(
                        values,
                        84,
                    )
                ),
            ),
        )
        print(
            "  median Mhat/Mtrue =",
            float(
                np.nanmedian(
                    10.0 ** values
                )
            ),
        )

    print()
    print("Saved:")
    print(FIGURE_MASS)
    print(FIGURE_MASS_PDF)
    print(FIGURE_PIE)
    print(FIGURE_PIE_PDF)
    print(BIN_TABLE)


if __name__ == "__main__":
    main()
