#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Empirical coverage of the full 2D parallax covariance ellipse.

NO simulation.
NO fitting.
NO Monte Carlo.

For each event:

    delta = [
        piEN_fit - piEN_true,
        piEE_fit - piEE_true
    ]

and

    D2 = delta^T C_piE^{-1} delta

where

    C_piE = [[C_NN, C_NE],
             [C_NE, C_EE]].

For a calibrated bivariate Gaussian,

    D2 ~ chi2(df=2).

The 68% joint-confidence threshold is

    chi2_2(0.68) = -2 ln(1 - 0.68).

We measure the empirical 68% ellipse coverage versus true |pi_E|,
separately for confused and detected positive-LRT events.
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


ROOT = Path(__file__).resolve().parents[3]

RECOVERY_INPUT = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "25_piE_component_recovery_vs_truth"
    / "piE_true_components_and_recovery.parquet"
)

COVARIANCE_INPUT = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "18_positive_truth"
    / "h1_positive_truth_characterization.parquet"
)

FIGURE_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "figures"
    / "characterization"
    / "28_piE_2d_covariance_coverage_vs_truth"
)

RESULT_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "28_piE_2d_covariance_coverage_vs_truth"
)

FIGURE = (
    FIGURE_DIR
    / "piE_2d_covariance_coverage_vs_truth.png"
)

TABLE = (
    RESULT_DIR
    / "piE_2d_covariance_coverage_vs_truth_bins.csv"
)


N_BINS = 22
PIE_MIN = 0.01
PIE_MAX = 20.0
MIN_COUNT = 30

NOMINAL_COVERAGE = 0.68

# For chi2 with 2 dof:
# CDF(x) = 1 - exp(-x/2)
D2_THRESHOLD_68 = -2.0 * np.log(
    1.0 - NOMINAL_COVERAGE
)


def numeric(df, column):

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def main():

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    recovery = pd.read_parquet(
        RECOVERY_INPUT,
        columns=[
            "catalog_row",
            "positive_lrt_class",
            "piE_true_amp",
            "piEN_true",
            "piEE_true",
            "h1_piEN",
            "h1_piEE",
        ],
    )

    covariance = pd.read_parquet(
        COVARIANCE_INPUT,
        columns=[
            "catalog_row",
            "h1_cov_piEN_piEN",
            "h1_cov_piEN_piEE",
            "h1_cov_piEE_piEE",
        ],
    )

    if not recovery["catalog_row"].is_unique:
        raise RuntimeError("Recovery catalog_row is not unique.")

    if not covariance["catalog_row"].is_unique:
        raise RuntimeError("Covariance catalog_row is not unique.")

    df = recovery.merge(
        covariance,
        on="catalog_row",
        how="left",
        validate="one_to_one",
    )

    piE_true = numeric(
        df,
        "piE_true_amp",
    )

    piEN_true = numeric(
        df,
        "piEN_true",
    )

    piEE_true = numeric(
        df,
        "piEE_true",
    )

    piEN_fit = numeric(
        df,
        "h1_piEN",
    )

    piEE_fit = numeric(
        df,
        "h1_piEE",
    )

    C_NN = numeric(
        df,
        "h1_cov_piEN_piEN",
    )

    C_NE = numeric(
        df,
        "h1_cov_piEN_piEE",
    )

    C_EE = numeric(
        df,
        "h1_cov_piEE_piEE",
    )

    classes = (
        df["positive_lrt_class"]
        .astype(str)
        .to_numpy()
    )

    # ========================================================
    # Valid positive-definite 2x2 covariance
    # ========================================================

    detC = (
        C_NN * C_EE
        - C_NE**2
    )

    valid = (
        np.isfinite(piE_true)
        & (piE_true > 0)
        & np.isfinite(piEN_true)
        & np.isfinite(piEE_true)
        & np.isfinite(piEN_fit)
        & np.isfinite(piEE_fit)
        & np.isfinite(C_NN)
        & np.isfinite(C_NE)
        & np.isfinite(C_EE)
        & (C_NN > 0)
        & (C_EE > 0)
        & (detC > 0)
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

    # Exact inverse of 2x2 covariance:
    #
    # C^-1 = 1/det(C) [[ C_EE, -C_NE],
    #                  [-C_NE,  C_NN]]
    #
    D2[valid] = (
        C_EE[valid] * dN[valid]**2
        - 2.0
        * C_NE[valid]
        * dN[valid]
        * dE[valid]
        + C_NN[valid]
        * dE[valid]**2
    ) / detC[valid]

    inside68 = (
        D2
        <= D2_THRESHOLD_68
    )

    regions = {
        "confused_fspl_no_parallax":
            classes
            == "confused_fspl_no_parallax",

        "parallax_detected":
            classes
            == "parallax_detected",
    }

    # ========================================================
    # Overall summary
    # ========================================================

    print(
        "=" * 105
    )

    print(
        "2D PARALLAX COVARIANCE COVERAGE"
    )

    print(
        "=" * 105
    )

    print()

    print(
        "68% chi2_2 threshold =",
        D2_THRESHOLD_68,
    )

    print(
        "valid covariance matrices =",
        int(valid.sum()),
    )

    print(
        "invalid =",
        int((~valid).sum()),
    )

    print()

    rows_summary = []

    for class_name, class_mask in regions.items():

        use = (
            valid
            & class_mask
        )

        values = D2[
            use
        ]

        rows_summary.append(
            {
                "class":
                    class_name,

                "n":
                    len(values),

                "D2_median":
                    float(
                        np.median(values)
                    ),

                "D2_q68":
                    float(
                        np.quantile(
                            values,
                            0.68,
                        )
                    ),

                "coverage68":
                    float(
                        np.mean(
                            values
                            <= D2_THRESHOLD_68
                        )
                    ),
            }
        )

    summary = pd.DataFrame(
        rows_summary
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Ideal chi2_2 median =",
        2.0 * np.log(2.0),
    )

    print(
        "Ideal 68% threshold =",
        D2_THRESHOLD_68,
    )

    # ========================================================
    # Bin versus true |pi_E|
    # ========================================================

    edges = np.geomspace(
        PIE_MIN,
        PIE_MAX,
        N_BINS + 1,
    )

    rows = []

    for ibin in range(N_BINS):

        left = edges[ibin]
        right = edges[ibin + 1]

        if ibin == N_BINS - 1:

            in_bin = (
                valid
                & (piE_true >= left)
                & (piE_true <= right)
            )

        else:

            in_bin = (
                valid
                & (piE_true >= left)
                & (piE_true < right)
            )

        center = np.sqrt(
            left * right
        )

        for class_name, class_mask in regions.items():

            use = (
                in_bin
                & class_mask
            )

            values = D2[
                use
            ]

            n = len(values)

            if n:

                coverage = float(
                    np.mean(
                        values
                        <= D2_THRESHOLD_68
                    )
                )

                median_D2 = float(
                    np.median(values)
                )

            else:

                coverage = np.nan
                median_D2 = np.nan

            rows.append(
                {
                    "bin_index":
                        ibin,

                    "class":
                        class_name,

                    "piE_true_left":
                        left,

                    "piE_true_right":
                        right,

                    "piE_true_center":
                        center,

                    "n":
                        n,

                    "coverage68":
                        coverage,

                    "median_D2":
                        median_D2,

                    "shown_in_figure":
                        n >= MIN_COUNT,
                }
            )

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

    plot_definitions = [
        (
            "confused_fspl_no_parallax",
            "Confused with FSPL without parallax",
            "o",
        ),
        (
            "parallax_detected",
            "Parallax detected",
            "s",
        ),
    ]

    for (
        class_name,
        label,
        marker,
    ) in plot_definitions:

        sub = result.loc[
            (
                result["class"]
                == class_name
            )
            & result[
                "shown_in_figure"
            ]
        ]

        x = sub[
            "piE_true_center"
        ].to_numpy(dtype=float)

        y = sub[
            "coverage68"
        ].to_numpy(dtype=float)

        good = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        ax.plot(
            x[good],
            y[good],
            marker=marker,
            markersize=4,
            linewidth=1.5,
            label=label,
        )

    ax.axhline(
        NOMINAL_COVERAGE,
        linestyle=":",
        linewidth=1.2,
        label="Nominal 68% ellipse",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        PIE_MIN,
        PIE_MAX,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        r"True $|\boldsymbol{\pi}_E|$"
    )

    ax.set_ylabel(
        (
            r"$P("
            r"D_{\pi_E}^2"
            r"\leq\chi^2_{2,0.68}"
            r")$"
        )
    )


    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        fontsize=13,
    )


    fig.savefig(
        FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    save_pdf_companion(fig, FIGURE)

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


