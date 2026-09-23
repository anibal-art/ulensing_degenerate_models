#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Empirical 1-sigma coverage of the fitted parallax components.

NO simulation.
NO fitting.
NO Monte Carlo.

For each event:

    z_N = (piEN_fit - piEN_true) / sqrt(C_NN)
    z_E = (piEE_fit - piEE_true) / sqrt(C_EE)

For a correctly calibrated Gaussian marginal uncertainty:

    z ~ N(0,1)

and therefore

    P(|z| <= 1) = 0.682689492...

We measure this coverage versus TRUE |pi_E| separately for:

    confused_fspl_no_parallax
    parallax_detected

and for North and East components.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

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
    / "27_piE_component_one_sigma_coverage"
)

RESULT_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "27_piE_component_one_sigma_coverage"
)

FIGURE = (
    FIGURE_DIR
    / "piE_component_one_sigma_coverage.png"
)

TABLE = (
    RESULT_DIR
    / "piE_component_one_sigma_coverage_bins.csv"
)


# ============================================================
# Configuration
# ============================================================

N_BINS = 22

PIE_MIN = 0.01
PIE_MAX = 20.0

MIN_COUNT = 30

GAUSSIAN_ONE_SIGMA_COVERAGE = 0.6826894921370859


# ============================================================
# Main
# ============================================================

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
            "h1_cov_piEE_piEE",
        ],
    )

    if not recovery["catalog_row"].is_unique:
        raise RuntimeError(
            "Recovery catalog_row is not unique."
        )

    if not covariance["catalog_row"].is_unique:
        raise RuntimeError(
            "Covariance catalog_row is not unique."
        )

    df = recovery.merge(
        covariance,
        on="catalog_row",
        how="left",
        validate="one_to_one",
    )

    if len(df) != len(recovery):
        raise RuntimeError(
            "Merge changed the event count."
        )

    # ========================================================
    # Arrays
    # ========================================================

    piE_true = pd.to_numeric(
        df["piE_true_amp"],
        errors="coerce",
    ).to_numpy(dtype=float)

    piEN_true = pd.to_numeric(
        df["piEN_true"],
        errors="coerce",
    ).to_numpy(dtype=float)

    piEE_true = pd.to_numeric(
        df["piEE_true"],
        errors="coerce",
    ).to_numpy(dtype=float)

    piEN_fit = pd.to_numeric(
        df["h1_piEN"],
        errors="coerce",
    ).to_numpy(dtype=float)

    piEE_fit = pd.to_numeric(
        df["h1_piEE"],
        errors="coerce",
    ).to_numpy(dtype=float)

    C_NN = pd.to_numeric(
        df["h1_cov_piEN_piEN"],
        errors="coerce",
    ).to_numpy(dtype=float)

    C_EE = pd.to_numeric(
        df["h1_cov_piEE_piEE"],
        errors="coerce",
    ).to_numpy(dtype=float)

    classes = (
        df["positive_lrt_class"]
        .astype(str)
        .to_numpy()
    )

    # ========================================================
    # Standardized residuals
    # ========================================================

    valid_N = (
        np.isfinite(piE_true)
        & (piE_true > 0)
        & np.isfinite(piEN_true)
        & np.isfinite(piEN_fit)
        & np.isfinite(C_NN)
        & (C_NN > 0)
    )

    valid_E = (
        np.isfinite(piE_true)
        & (piE_true > 0)
        & np.isfinite(piEE_true)
        & np.isfinite(piEE_fit)
        & np.isfinite(C_EE)
        & (C_EE > 0)
    )

    z_N = np.full(
        len(df),
        np.nan,
    )

    z_E = np.full(
        len(df),
        np.nan,
    )

    z_N[valid_N] = (
        piEN_fit[valid_N]
        - piEN_true[valid_N]
    ) / np.sqrt(
        C_NN[valid_N]
    )

    z_E[valid_E] = (
        piEE_fit[valid_E]
        - piEE_true[valid_E]
    ) / np.sqrt(
        C_EE[valid_E]
    )

    # ========================================================
    # Overall summaries
    # ========================================================

    regions = {
        "confused_fspl_no_parallax":
            classes
            == "confused_fspl_no_parallax",

        "parallax_detected":
            classes
            == "parallax_detected",
    }

    print(
        "=" * 110
    )

    print(
        "PARALLAX COMPONENT STANDARDIZED-RESIDUAL CALIBRATION"
    )

    print(
        "=" * 110
    )

    print()

    summary_rows = []

    for class_name, class_mask in regions.items():

        for component, z in [
            ("piEN", z_N),
            ("piEE", z_E),
        ]:

            use = (
                class_mask
                & np.isfinite(z)
            )

            values = z[use]

            q16, q50, q84 = np.quantile(
                values,
                [0.16, 0.50, 0.84],
            )

            coverage = np.mean(
                np.abs(values) <= 1.0
            )

            median_abs_z = np.median(
                np.abs(values)
            )

            summary_rows.append(
                {
                    "class": class_name,
                    "component": component,
                    "n": len(values),
                    "z_q16": q16,
                    "z_median": q50,
                    "z_q84": q84,
                    "central_width_half":
                        0.5 * (q84 - q16),
                    "median_abs_z":
                        median_abs_z,
                    "coverage_abs_z_le_1":
                        coverage,
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Gaussian references:"
    )

    print(
        "median(|z|) = 0.67448975"
    )

    print(
        "P(|z| <= 1) =",
        GAUSSIAN_ONE_SIGMA_COVERAGE,
    )

    # ========================================================
    # Binned one-sigma coverage
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
                (piE_true >= left)
                & (piE_true <= right)
            )

        else:

            in_bin = (
                (piE_true >= left)
                & (piE_true < right)
            )

        center = np.sqrt(
            left * right
        )

        for class_name, class_mask in regions.items():

            for component, z in [
                ("piEN", z_N),
                ("piEE", z_E),
            ]:

                use = (
                    in_bin
                    & class_mask
                    & np.isfinite(z)
                )

                values = z[use]

                n = len(values)

                if n:

                    coverage = float(
                        np.mean(
                            np.abs(values)
                            <= 1.0
                        )
                    )

                    median_abs_z = float(
                        np.median(
                            np.abs(values)
                        )
                    )

                else:

                    coverage = np.nan
                    median_abs_z = np.nan

                rows.append(
                    {
                        "bin_index":
                            ibin,

                        "class":
                            class_name,

                        "component":
                            component,

                        "piE_true_left":
                            left,

                        "piE_true_right":
                            right,

                        "piE_true_center":
                            center,

                        "n":
                            n,

                        "coverage_abs_z_le_1":
                            coverage,

                        "median_abs_z":
                            median_abs_z,

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
    # Single figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8.0, 5.6)
    )

    plot_definitions = [
        (
            "confused_fspl_no_parallax",
            "piEN",
            r"Confused: $\pi_{E,N}$",
            "-",
            "o",
        ),
        (
            "confused_fspl_no_parallax",
            "piEE",
            r"Confused: $\pi_{E,E}$",
            "--",
            "o",
        ),
        (
            "parallax_detected",
            "piEN",
            r"Detected: $\pi_{E,N}$",
            "-",
            "s",
        ),
        (
            "parallax_detected",
            "piEE",
            r"Detected: $\pi_{E,E}$",
            "--",
            "s",
        ),
    ]

    for (
        class_name,
        component,
        label,
        linestyle,
        marker,
    ) in plot_definitions:

        sub = result.loc[
            (
                result["class"]
                == class_name
            )
            & (
                result["component"]
                == component
            )
            & result[
                "shown_in_figure"
            ]
        ]

        x = sub[
            "piE_true_center"
        ].to_numpy(dtype=float)

        y = sub[
            "coverage_abs_z_le_1"
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
            linestyle=linestyle,
            label=label,
        )

    ax.axhline(
        GAUSSIAN_ONE_SIGMA_COVERAGE,
        linestyle=":",
        linewidth=1.2,
        label=(
            r"Gaussian reference: "
            r"$P(|z|\leq1)=0.6827$"
        ),
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
        r"Empirical $1\sigma$ coverage $P(|z_i|\leq1)$"
    )

    ax.set_title(
        "Calibration of the parallax-component uncertainties"
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE,
        dpi=220,
        bbox_inches="tight",
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
