#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallax uncertainty area versus Delta chi^2_LRT
for the COMPLETE H1-generated population.

NO simulation.
NO fitting.

For the fitted parallax vector

    pi_E = (piEN, piEE)

and its 2x2 covariance block

    C_piE = [[C_NN, C_NE],
             [C_NE, C_EE]],

define

    A_piE = sqrt(det(C_piE))

and the dimensionless relative uncertainty area

    A_piE_rel
        = sqrt(det(C_piE)) / |pi_E_hat|^2.

This measures the area scale of the local parallax uncertainty ellipse
relative to the squared amplitude of the fitted parallax vector.

Unlike det(R), this quantity contains the absolute uncertainty scale.

The figure shows the median and 16--84 percentile interval of

    log10(A_piE_rel)

as a function of Delta chi^2_LRT across the complete H1 population.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    / "14_h1_parallax_area_vs_lrt"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "characterization"
    / "14_h1_parallax_area_vs_lrt"
)

FIGURE = (
    FIGURE_DIR
    / "h1_relative_parallax_area_vs_lrt.png"
)

TABLE = (
    RESULT_DIR
    / "h1_relative_parallax_area_vs_lrt_bins.csv"
)


# ============================================================
# Configuration
# ============================================================

N_BINS = 45

PRIMARY_THRESHOLD = 13.15982011480088


# ============================================================
# Transform used only for x-axis binning
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

    # --------------------------------------------------------
    # Read only what is needed
    # --------------------------------------------------------

    df = pd.read_parquet(
        INPUT,
        columns=[
            "delta_chi2_lrt",
            "h1_piEN",
            "h1_piEE",
            "h1_cov_piEN_piEN",
            "h1_cov_piEN_piEE",
            "h1_cov_piEE_piEE",
        ],
    )

    T = pd.to_numeric(
        df["delta_chi2_lrt"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    piEN = pd.to_numeric(
        df["h1_piEN"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    piEE = pd.to_numeric(
        df["h1_piEE"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    C_NN = pd.to_numeric(
        df["h1_cov_piEN_piEN"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    C_NE = pd.to_numeric(
        df["h1_cov_piEN_piEE"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    C_EE = pd.to_numeric(
        df["h1_cov_piEE_piEE"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    # ========================================================
    # Fitted parallax amplitude
    # ========================================================

    piE = np.hypot(
        piEN,
        piEE,
    )

    # ========================================================
    # Parallax covariance determinant
    # ========================================================

    det_C_piE = (
        C_NN
        * C_EE
        - C_NE**2
    )

    valid_covariance = (
        np.isfinite(
            C_NN
        )
        & np.isfinite(
            C_NE
        )
        & np.isfinite(
            C_EE
        )
        & (C_NN > 0)
        & (C_EE > 0)
        & (det_C_piE > 0)
    )

    # ========================================================
    # Relative parallax uncertainty area
    # ========================================================

    relative_area = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(
            T
        )
        & np.isfinite(
            piE
        )
        & (piE > 0)
        & valid_covariance
    )

    relative_area[
        valid
    ] = (
        np.sqrt(
            det_C_piE[
                valid
            ]
        )
        / piE[
            valid
        ]**2
    )

    log10_relative_area = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    valid_area = (
        np.isfinite(
            relative_area
        )
        & (
            relative_area
            > 0
        )
    )

    log10_relative_area[
        valid_area
    ] = np.log10(
        relative_area[
            valid_area
        ]
    )

    # ========================================================
    # Signed-log LRT coordinate
    # ========================================================

    finite_T = np.isfinite(
        T
    )

    x_transform = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    x_transform[
        finite_T
    ] = signed_log10_1p(
        T[
            finite_T
        ]
    )

    finite_x = x_transform[
        np.isfinite(
            x_transform
        )
    ]

    edges = np.linspace(
        np.min(
            finite_x
        ),
        np.max(
            finite_x
        ),
        N_BINS + 1,
    )

    # ========================================================
    # Bin statistics
    # ========================================================

    rows = []

    for i in range(
        N_BINS
    ):

        left = edges[i]
        right = edges[i + 1]

        if i == N_BINS - 1:

            in_bin = (
                finite_T
                & (
                    x_transform
                    >= left
                )
                & (
                    x_transform
                    <= right
                )
            )

        else:

            in_bin = (
                finite_T
                & (
                    x_transform
                    >= left
                )
                & (
                    x_transform
                    < right
                )
            )

        n_total = int(
            np.sum(
                in_bin
            )
        )

        if n_total == 0:
            continue

        center_transform = (
            0.5
            * (
                left
                + right
            )
        )

        center_T = float(
            inverse_signed_log10_1p(
                center_transform
            )
        )

        good = (
            in_bin
            & valid_area
        )

        values = (
            log10_relative_area[
                good
            ]
        )

        n_valid = len(
            values
        )

        if n_valid:

            q16, median, q84 = np.quantile(
                values,
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

        rows.append(
            {
                "bin_index":
                    i,

                "delta_chi2_center":
                    center_T,

                "n_total":
                    n_total,

                "n_valid_parallax_covariance":
                    n_valid,

                "valid_fraction":
                    (
                        n_valid
                        / n_total
                    ),

                "log10_relative_area_q16":
                    q16,

                "log10_relative_area_median":
                    median,

                "log10_relative_area_q84":
                    q84,
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
        figsize=(7.4, 5.3)
    )

    x = signed_log10_1p(
        result[
            "delta_chi2_center"
        ].to_numpy(
            dtype=float
        )
    )

    median = result[
        "log10_relative_area_median"
    ].to_numpy(
        dtype=float
    )

    q16 = result[
        "log10_relative_area_q16"
    ].to_numpy(
        dtype=float
    )

    q84 = result[
        "log10_relative_area_q84"
    ].to_numpy(
        dtype=float
    )

    valid_plot = (
        np.isfinite(
            x
        )
        & np.isfinite(
            median
        )
        & np.isfinite(
            q16
        )
        & np.isfinite(
            q84
        )
    )

    ax.fill_between(
        x[
            valid_plot
        ],
        q16[
            valid_plot
        ],
        q84[
            valid_plot
        ],
        alpha=0.20,
        label="16--84 percentile",
    )

    ax.plot(
        x[
            valid_plot
        ],
        median[
            valid_plot
        ],
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        label="Median",
    )

    # --------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        label=r"$\Delta\chi^2=0$",
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
        label=r"$\alpha=10^{-3}$ threshold",
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

    xmin = np.min(
        finite_x
    )

    xmax = np.max(
        finite_x
    )

    keep = (
        (tick_x >= xmin)
        & (tick_x <= xmax)
    )

    labels = []

    for value in tick_T[
        keep
    ]:

        if abs(value) >= 100:

            labels.append(
                rf"$10^{{{int(np.log10(abs(value)))}}}$"
            )

        else:

            labels.append(
                f"{value:g}"
            )

    ax.set_xticks(
        tick_x[
            keep
        ],
        labels,
    )

    ax.set_xlabel(
        r"$\Delta\chi^2_{\rm LRT}$"
    )

    ax.set_ylabel(
        (
            r"$\log_{10}"
            r"\left["
            r"\sqrt{\det C_{\pi_E}}"
            r"/|\hat{\boldsymbol{\pi}}_E|^2"
            r"\right]$"
        )
    )

    ax.set_title(
        "Relative parallax uncertainty area vs LRT strength"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE,
        dpi=220,
        bbox_inches="tight",
    )

    # ========================================================
    # Stdout
    # ========================================================

    print(
        "=" * 90
    )

    print(
        "RELATIVE PARALLAX UNCERTAINTY AREA VS LRT"
    )

    print(
        "=" * 90
    )

    print(
        "rows =",
        len(df),
    )

    print(
        "valid parallax covariance =",
        int(
            valid_area.sum()
        ),
    )

    print(
        "invalid =",
        int(
            len(df)
            - valid_area.sum()
        ),
    )

    print()

    print(
        result.to_string(
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
