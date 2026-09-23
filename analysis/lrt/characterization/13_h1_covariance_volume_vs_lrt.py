#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single diagnostic figure:

    H1 correlation-hypervolume per dimension
    versus Delta chi^2_LRT

for the COMPLETE H1-generated population.

NO simulation.
NO fitting.

The plotted quantity is

    V_corr,dim = det(R)^(1/(2d))

with d = 6 for H1.

Interpretation
--------------
V_corr,dim = 1:
    locally uncorrelated parameters.

V_corr,dim << 1:
    strongly compressed / correlated covariance geometry.

Only events with a strictly positive-definite H1 correlation matrix
have a defined determinant and therefore enter the upper-panel
hypervolume statistic.

To make that selection visible rather than hidden, the lower panel
shows the fraction of events in each Delta-chi2 bin with a strictly
positive-definite H1 correlation matrix.

The x coordinate uses

    x_t = sign(T) log10(1 + |T|)

for bin construction, while tick labels are displayed in the original
Delta-chi2 variable.
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

OUTPUT_DIR = (
    ROOT
    / "figures"
    / "characterization"
    / "13_h1_covariance_volume_vs_lrt"
)

OUTPUT = (
    OUTPUT_DIR
    / "h1_covariance_volume_vs_lrt.png"
)

TABLE_OUTPUT_DIR = (
    ROOT
    / "results"
    / "characterization"
    / "13_h1_covariance_volume_vs_lrt"
)

TABLE_OUTPUT = (
    TABLE_OUTPUT_DIR
    / "h1_covariance_volume_vs_lrt_bins.csv"
)


# ============================================================
# Configuration
# ============================================================

N_BINS = 45

PRIMARY_THRESHOLD = 13.15982011480088


# ============================================================
# Transform
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_parquet(
        INPUT,
        columns=[
            "delta_chi2_lrt",
            "h1_corr_volume_per_dimension",
            "h1_corr_positive_definite",
        ],
    )

    T = pd.to_numeric(
        df["delta_chi2_lrt"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    volume = pd.to_numeric(
        df["h1_corr_volume_per_dimension"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    pd_valid = (
        df[
            "h1_corr_positive_definite"
        ]
        .fillna(False)
        .to_numpy(
            dtype=bool
        )
    )

    finite_T = np.isfinite(
        T
    )

    T_transform = np.full(
        len(T),
        np.nan,
        dtype=float,
    )

    T_transform[
        finite_T
    ] = signed_log10_1p(
        T[
            finite_T
        ]
    )

    # --------------------------------------------------------
    # Equal-width bins in signed-log LRT space
    # --------------------------------------------------------

    finite_x = T_transform[
        np.isfinite(
            T_transform
        )
    ]

    edges = np.linspace(
        np.min(finite_x),
        np.max(finite_x),
        N_BINS + 1,
    )

    rows = []

    for i in range(
        N_BINS
    ):

        left = edges[i]
        right = edges[i + 1]

        if i == N_BINS - 1:

            in_bin = (
                finite_T
                & (T_transform >= left)
                & (T_transform <= right)
            )

        else:

            in_bin = (
                finite_T
                & (T_transform >= left)
                & (T_transform < right)
            )

        n_total = int(
            np.sum(
                in_bin
            )
        )

        if n_total == 0:
            continue

        center_transform = 0.5 * (
            left + right
        )

        center_T = float(
            inverse_signed_log10_1p(
                center_transform
            )
        )

        pd_fraction = float(
            np.mean(
                pd_valid[
                    in_bin
                ]
            )
        )

        valid_volume = (
            in_bin
            & pd_valid
            & np.isfinite(
                volume
            )
            & (volume > 0)
        )

        values = volume[
            valid_volume
        ]

        n_volume = len(
            values
        )

        if n_volume:

            q16, q50, q84 = np.quantile(
                values,
                [
                    0.16,
                    0.50,
                    0.84,
                ],
            )

        else:

            q16 = np.nan
            q50 = np.nan
            q84 = np.nan

        rows.append(
            {
                "bin_index":
                    i,

                "x_transform_left":
                    left,

                "x_transform_right":
                    right,

                "x_transform_center":
                    center_transform,

                "delta_chi2_center":
                    center_T,

                "n_total":
                    n_total,

                "n_positive_definite":
                    int(
                        np.sum(
                            in_bin
                            & pd_valid
                        )
                    ),

                "positive_definite_fraction":
                    pd_fraction,

                "n_volume":
                    n_volume,

                "volume_q16":
                    q16,

                "volume_median":
                    q50,

                "volume_q84":
                    q84,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        TABLE_OUTPUT,
        index=False,
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        height_ratios=[
            3.0,
            1.15,
        ],
    )

    ax = axes[0]

    x = result[
        "x_transform_center"
    ].to_numpy(
        dtype=float
    )

    median = result[
        "volume_median"
    ].to_numpy(
        dtype=float
    )

    q16 = result[
        "volume_q16"
    ].to_numpy(
        dtype=float
    )

    q84 = result[
        "volume_q84"
    ].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(median)
        & np.isfinite(q16)
        & np.isfinite(q84)
    )

    ax.fill_between(
        x[
            valid
        ],
        q16[
            valid
        ],
        q84[
            valid
        ],
        alpha=0.20,
        label="16--84 percentile",
    )

    ax.plot(
        x[
            valid
        ],
        median[
            valid
        ],
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        label="Median",
    )

    # Delta chi2 = 0
    ax.axvline(
        signed_log10_1p(
            0.0
        ),
        linestyle="--",
        linewidth=1.0,
    )

    # Primary empirical threshold
    threshold_x = float(
        signed_log10_1p(
            PRIMARY_THRESHOLD
        )
    )

    ax.axvline(
        threshold_x,
        linestyle=":",
        linewidth=1.2,
        label=(
            r"$\alpha=10^{-3}$ threshold"
        ),
    )

    ax.set_ylabel(
        (
            r"$[\det(R_{H1})]^{1/(2d)}$"
            "\n"
            r"($d=6$)"
        )
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    # --------------------------------------------------------
    # Lower panel: strict positive-definite fraction
    # --------------------------------------------------------

    ax2 = axes[1]

    pd_fraction = result[
        "positive_definite_fraction"
    ].to_numpy(
        dtype=float
    )

    ax2.plot(
        x,
        pd_fraction,
        marker="o",
        markersize=3.0,
        linewidth=1.3,
    )

    ax2.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax2.axvline(
        threshold_x,
        linestyle=":",
        linewidth=1.2,
    )

    ax2.set_ylabel(
        "PD fraction"
    )

    ax2.set_ylim(
        0,
        1.03,
    )

    ax2.grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # Original Delta-chi2 labels on transformed x-axis
    # --------------------------------------------------------

    tick_T = np.array(
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

    ax2.set_xticks(
        tick_x[
            keep
        ],
        labels,
    )

    ax2.set_xlabel(
        r"$\Delta\chi^2_{\rm LRT}$"
    )

    fig.suptitle(
        (
            "H1 covariance geometry across the full "
            r"$\Delta\chi^2_{\rm LRT}$ distribution"
        )
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT,
        dpi=220,
        bbox_inches="tight",
    )

    print(
        "=" * 80
    )

    print(
        "H1 COVARIANCE VOLUME VS LRT"
    )

    print(
        "=" * 80
    )

    print(
        "rows =",
        len(df),
    )

    print(
        "bins =",
        len(result),
    )

    print()

    print(
        result[
            [
                "delta_chi2_center",
                "n_total",
                "positive_definite_fraction",
                "volume_median",
                "volume_q16",
                "volume_q84",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Saved figure:",
        OUTPUT,
    )

    print(
        "Saved bins:",
        TABLE_OUTPUT,
    )


if __name__ == "__main__":
    main()
