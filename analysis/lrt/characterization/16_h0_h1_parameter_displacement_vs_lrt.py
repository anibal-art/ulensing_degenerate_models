#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H0-H1 fitted-parameter displacement versus Delta chi^2_LRT
for the COMPLETE H1-generated population.

NO simulation.
NO fitting.

We compare the common nonlinear parameters of the nested fits:

    H0: t0, u0, tE, rho
    H1: t0, u0, tE, rho, piEN, piEE

Diagnostics
-----------

    dt0_over_tE
        = |t0_H1 - t0_H0| / tE_H1

    du0
        = |u0_H1 - u0_H0|

    dlog10_tE
        = |log10(tE_H1 / tE_H0)|

    dlog10_rho
        = |log10(rho_H1 / rho_H0)|

These quantities measure whether the two nested fits occupy the same physical
region of common-parameter space.

The figure shows the median trend of each metric after division by its global
median.  This normalization is ONLY for plotting the four quantities on the
same axis.  The output CSV stores the physical, unnormalized medians and
16--84 percentile ranges.
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
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "characterization"
    / "11_covariance_geometry"
    / "h1_covariance_geometry.parquet"
)

FIGURE_DIR = (
    ROOT
    / "figures"
    / "characterization"
    / "16_h0_h1_parameter_displacement_vs_lrt"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "characterization"
    / "16_h0_h1_parameter_displacement_vs_lrt"
)

FIGURE = (
    FIGURE_DIR
    / "h0_h1_parameter_displacement_vs_lrt.png"
)

TABLE = (
    RESULT_DIR
    / "h0_h1_parameter_displacement_vs_lrt_bins.csv"
)


# ============================================================
# Configuration
# ============================================================

N_BINS = 45

PRIMARY_THRESHOLD = 13.15982011480088


# ============================================================
# Signed-log coordinate
# ============================================================

def signed_log10_1p(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    return (
        np.sign(x)
        * np.log10(
            1.0 + np.abs(x)
        )
    )


def inverse_signed_log10_1p(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    return (
        np.sign(x)
        * (
            10.0 ** np.abs(x)
            - 1.0
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    if not INPUT.exists():

        raise FileNotFoundError(
            f"Input not found:\n{INPUT}"
        )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "delta_chi2_lrt",

        "h0_t0",
        "h0_u0",
        "h0_tE",
        "h0_rho",

        "h1_t0",
        "h1_u0",
        "h1_tE",
        "h1_rho",
    ]

    df = pd.read_parquet(
        INPUT,
        columns=columns,
    )

    T = pd.to_numeric(
        df["delta_chi2_lrt"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    h0_t0 = pd.to_numeric(
        df["h0_t0"],
        errors="coerce",
    ).to_numpy(float)

    h1_t0 = pd.to_numeric(
        df["h1_t0"],
        errors="coerce",
    ).to_numpy(float)

    h0_u0 = pd.to_numeric(
        df["h0_u0"],
        errors="coerce",
    ).to_numpy(float)

    h1_u0 = pd.to_numeric(
        df["h1_u0"],
        errors="coerce",
    ).to_numpy(float)

    h0_tE = pd.to_numeric(
        df["h0_tE"],
        errors="coerce",
    ).to_numpy(float)

    h1_tE = pd.to_numeric(
        df["h1_tE"],
        errors="coerce",
    ).to_numpy(float)

    h0_rho = pd.to_numeric(
        df["h0_rho"],
        errors="coerce",
    ).to_numpy(float)

    h1_rho = pd.to_numeric(
        df["h1_rho"],
        errors="coerce",
    ).to_numpy(float)

    # ========================================================
    # Physical displacement metrics
    # ========================================================

    dt0_over_tE = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(h0_t0)
        & np.isfinite(h1_t0)
        & np.isfinite(h1_tE)
        & (h1_tE > 0)
    )

    dt0_over_tE[valid] = (
        np.abs(
            h1_t0[valid]
            - h0_t0[valid]
        )
        / h1_tE[valid]
    )

    du0 = np.abs(
        h1_u0 - h0_u0
    )

    dlog10_tE = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(h0_tE)
        & np.isfinite(h1_tE)
        & (h0_tE > 0)
        & (h1_tE > 0)
    )

    dlog10_tE[valid] = np.abs(
        np.log10(
            h1_tE[valid]
            / h0_tE[valid]
        )
    )

    dlog10_rho = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(h0_rho)
        & np.isfinite(h1_rho)
        & (h0_rho > 0)
        & (h1_rho > 0)
    )

    dlog10_rho[valid] = np.abs(
        np.log10(
            h1_rho[valid]
            / h0_rho[valid]
        )
    )

    metrics = {
        "dt0_over_tE":
            dt0_over_tE,

        "du0":
            du0,

        "dlog10_tE":
            dlog10_tE,

        "dlog10_rho":
            dlog10_rho,
    }

    # ========================================================
    # Global medians used only for visual normalization
    # ========================================================

    normalization = {}

    for name, values in metrics.items():

        finite = (
            np.isfinite(values)
            & (values >= 0)
        )

        median = np.median(
            values[finite]
        )

        if (
            not np.isfinite(median)
            or median <= 0
        ):

            median = 1.0

        normalization[name] = median

    # ========================================================
    # LRT bins
    # ========================================================

    finite_T = np.isfinite(T)

    x_transform = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    x_transform[finite_T] = (
        signed_log10_1p(
            T[finite_T]
        )
    )

    finite_x = x_transform[
        np.isfinite(
            x_transform
        )
    ]

    edges = np.linspace(
        np.min(finite_x),
        np.max(finite_x),
        N_BINS + 1,
    )

    rows = []

    for ibin in range(N_BINS):

        left = edges[ibin]
        right = edges[ibin + 1]

        if ibin == N_BINS - 1:

            in_bin = (
                finite_T
                & (x_transform >= left)
                & (x_transform <= right)
            )

        else:

            in_bin = (
                finite_T
                & (x_transform >= left)
                & (x_transform < right)
            )

        n_total = int(
            np.sum(in_bin)
        )

        if n_total == 0:
            continue

        center_x = 0.5 * (
            left + right
        )

        center_T = float(
            inverse_signed_log10_1p(
                center_x
            )
        )

        row = {
            "bin_index":
                ibin,

            "delta_chi2_center":
                center_T,

            "n_total":
                n_total,
        }

        for name, values in metrics.items():

            valid = (
                in_bin
                & np.isfinite(values)
            )

            x = values[valid]

            if len(x):

                q16, median, q84 = np.quantile(
                    x,
                    [
                        0.16,
                        0.50,
                        0.84,
                    ],
                )

            else:

                q16 = np.nan
                median = np.nan
                q84 = np.nan

            row[
                f"{name}_q16"
            ] = q16

            row[
                f"{name}_median"
            ] = median

            row[
                f"{name}_q84"
            ] = q84

            row[
                f"{name}_median_normalized"
            ] = (
                median
                / normalization[name]
            )

        rows.append(row)

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        TABLE,
        index=False,
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.8, 4.4)
    )

    x = signed_log10_1p(
        result[
            "delta_chi2_center"
        ].to_numpy(float)
    )

    labels = {
        "dt0_over_tE":
            r"$|t_{0,H1}-t_{0,H0}|/t_{E,H1}$",

        "du0":
            r"$|u_{0,H1}-u_{0,H0}|$",

        "dlog10_tE":
            r"$|\log_{10}(t_{E,H1}/t_{E,H0})|$",

        "dlog10_rho":
            r"$|\log_{10}(\rho_{H1}/\rho_{H0})|$",
    }

    for name in metrics:

        y = result[
            f"{name}_median_normalized"
        ].to_numpy(float)

        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (y > 0)
        )

        ax.plot(
            x[valid],
            y[valid],
            marker="o",
            markersize=3.0,
            linewidth=1.3,
            label=labels[name],
        )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    threshold_x = float(
        signed_log10_1p(
            PRIMARY_THRESHOLD
        )
    )

    ax.axvline(
        threshold_x,
        linestyle=":",
        linewidth=1.2,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )

    ax.set_yscale(
        "log"
    )

    # --------------------------------------------------------
    # Original Delta-chi2 labels
    # --------------------------------------------------------

    tick_T = np.asarray(
        [
            -10.0,
            -3.0,
            -1.0,
            0.0,
            1.0,
            3.0,
            10.0,
            1e2,
            1e3,
            1e4,
            1e5,
            1e6,
        ],
        dtype=float,
    )

    tick_x = signed_log10_1p(
        tick_T
    )

    xmin = np.min(finite_x)
    xmax = np.max(finite_x)

    keep = (
        (tick_x >= xmin)
        & (tick_x <= xmax)
    )

    tick_labels = []

    for value in tick_T[keep]:

        if abs(value) >= 100:

            tick_labels.append(
                rf"$10^{{{int(np.log10(abs(value)))}}}$"
            )

        else:

            tick_labels.append(
                f"{value:g}"
            )

    ax.set_xticks(
        tick_x[keep],
        tick_labels,
    )

    ax.set_xlabel(
        r"$\Delta\chi^2_{\rm LRT}$"
    )

    ax.set_ylabel(
        "Median displacement / global median"
    )


    ax.grid(
        alpha=0.25
    )

    ax.legend(
        fontsize=13
    )


    fig.savefig(
        FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    save_pdf_companion(fig, FIGURE)

    # ========================================================
    # Compact region summary
    # ========================================================

    regions = {
        "negative":
            T < 0,

        "positive_below_threshold":
            (
                (T >= 0)
                & (
                    T
                    < PRIMARY_THRESHOLD
                )
            ),

        "detected_alpha_1e-3":
            T
            >= PRIMARY_THRESHOLD,
    }

    print(
        "=" * 100
    )

    print(
        "H0-H1 FITTED-PARAMETER DISPLACEMENT"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Global medians used only for plot normalization:"
    )

    for name, value in normalization.items():

        print(
            f"{name:16s} = {value:.8g}"
        )

    print()

    output_rows = []

    for region_name, mask in regions.items():

        row = {
            "region":
                region_name,

            "n":
                int(
                    np.sum(mask)
                ),
        }

        for name, values in metrics.items():

            valid = (
                mask
                & np.isfinite(values)
            )

            row[
                name
            ] = (
                float(
                    np.median(
                        values[valid]
                    )
                )
                if np.any(valid)
                else np.nan
            )

        output_rows.append(
            row
        )

    summary = pd.DataFrame(
        output_rows
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Saved figure:",
        FIGURE,
    )

    print(
        "Saved bins:",
        TABLE,
    )


if __name__ == "__main__":
    main()


