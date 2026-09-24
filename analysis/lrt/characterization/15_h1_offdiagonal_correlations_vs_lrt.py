#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single heatmap of all H1 off-diagonal correlation coefficients
versus Delta chi^2_LRT.

NO simulation.
NO fitting.

For every pair of H1 nonlinear parameters,

    (t0, u0, tE, rho, piEN, piEE),

we study

    |r_ij| = |C_ij| / sqrt(C_ii C_jj)

across the COMPLETE H1-generated population.

The figure contains one row per off-diagonal parameter pair and one
column per Delta-chi2 bin.

Only events whose complete H1 correlation matrix is strictly positive
definite are used for the correlation medians. Invalid/non-PD covariance
matrices have already been characterized separately and should not be
interpreted as statistical correlation matrices.

The script also prints a compact table comparing:
    negative,
    positive below threshold,
    detected at alpha=1e-3.

This allows us to identify which parameter degeneracies are specifically
enhanced in the Delta-chi2 < 0 population.
"""

from pathlib import Path
from itertools import combinations

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
    / "15_h1_offdiagonal_correlations_vs_lrt"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "characterization"
    / "15_h1_offdiagonal_correlations_vs_lrt"
)

FIGURE = (
    FIGURE_DIR
    / "h1_offdiagonal_correlations_vs_lrt.png"
)

BIN_TABLE = (
    RESULT_DIR
    / "h1_offdiagonal_correlations_vs_lrt_bins.csv"
)

REGION_TABLE = (
    RESULT_DIR
    / "h1_offdiagonal_correlations_by_lrt_region.csv"
)


# ============================================================
# Configuration
# ============================================================

PARAMETERS = [
    "t0",
    "u0",
    "tE",
    "rho",
    "piEN",
    "piEE",
]

PAIRS = list(
    combinations(
        PARAMETERS,
        2,
    )
)

N_BINS = 45

PRIMARY_THRESHOLD = 13.15982011480088


# ============================================================
# LRT coordinate
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
    # Required columns
    # --------------------------------------------------------

    correlation_columns = [
        f"h1_corr_{p1}_{p2}"
        for p1, p2
        in PAIRS
    ]

    columns = [
        "delta_chi2_lrt",
        "h1_corr_positive_definite",
        *correlation_columns,
    ]

    df = pd.read_parquet(
        INPUT,
        columns=columns,
    )

    T = pd.to_numeric(
        df[
            "delta_chi2_lrt"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    positive_definite = (
        df[
            "h1_corr_positive_definite"
        ]
        .fillna(False)
        .to_numpy(
            dtype=bool
        )
    )

    # ========================================================
    # LRT coordinate and bins
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

    heatmap = np.full(
        (
            len(PAIRS),
            N_BINS,
        ),
        np.nan,
        dtype=float,
    )

    bin_rows = []

    # ========================================================
    # Bin-by-bin correlations
    # ========================================================

    for ibin in range(
        N_BINS
    ):

        left = edges[
            ibin
        ]

        right = edges[
            ibin + 1
        ]

        if ibin == N_BINS - 1:

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

        center_x = (
            0.5
            * (
                left
                + right
            )
        )

        center_T = float(
            inverse_signed_log10_1p(
                center_x
            )
        )

        valid_event = (
            in_bin
            & positive_definite
        )

        for ipair, (
            p1,
            p2,
        ) in enumerate(
            PAIRS
        ):

            column = (
                f"h1_corr_{p1}_{p2}"
            )

            values = pd.to_numeric(
                df[
                    column
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            values = np.abs(
                values[
                    valid_event
                ]
            )

            values = values[
                np.isfinite(
                    values
                )
            ]

            if len(
                values
            ):

                median = float(
                    np.median(
                        values
                    )
                )

                q16 = float(
                    np.quantile(
                        values,
                        0.16,
                    )
                )

                q84 = float(
                    np.quantile(
                        values,
                        0.84,
                    )
                )

            else:

                median = np.nan
                q16 = np.nan
                q84 = np.nan

            heatmap[
                ipair,
                ibin,
            ] = median

            bin_rows.append(
                {
                    "bin_index":
                        ibin,

                    "delta_chi2_center":
                        center_T,

                    "parameter_1":
                        p1,

                    "parameter_2":
                        p2,

                    "n_total":
                        int(
                            np.sum(
                                in_bin
                            )
                        ),

                    "n_positive_definite":
                        int(
                            np.sum(
                                valid_event
                            )
                        ),

                    "abs_corr_q16":
                        q16,

                    "abs_corr_median":
                        median,

                    "abs_corr_q84":
                        q84,
                }
            )

    pd.DataFrame(
        bin_rows
    ).to_csv(
        BIN_TABLE,
        index=False,
    )

    # ========================================================
    # Region summaries
    # ========================================================

    negative = (
        T < 0
    )

    positive_below = (
        (T >= 0)
        & (
            T
            < PRIMARY_THRESHOLD
        )
    )

    detected = (
        T
        >= PRIMARY_THRESHOLD
    )

    regions = {
        "negative":
            negative,

        "positive_below_threshold":
            positive_below,

        "detected_alpha_1e-3":
            detected,
    }

    region_rows = []

    for p1, p2 in PAIRS:

        column = (
            f"h1_corr_{p1}_{p2}"
        )

        correlation = np.abs(
            pd.to_numeric(
                df[
                    column
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )
        )

        row = {
            "parameter_1":
                p1,

            "parameter_2":
                p2,
        }

        for region_name, region_mask in (
            regions.items()
        ):

            valid = (
                region_mask
                & positive_definite
                & np.isfinite(
                    correlation
                )
            )

            values = correlation[
                valid
            ]

            row[
                f"{region_name}_n"
            ] = int(
                len(
                    values
                )
            )

            row[
                f"{region_name}_median_abs_corr"
            ] = (
                float(
                    np.median(
                        values
                    )
                )
                if len(
                    values
                )
                else np.nan
            )

            row[
                f"{region_name}_q90_abs_corr"
            ] = (
                float(
                    np.quantile(
                        values,
                        0.90,
                    )
                )
                if len(
                    values
                )
                else np.nan
            )

        row[
            "negative_minus_detected_median"
        ] = (
            row[
                "negative_median_abs_corr"
            ]
            - row[
                "detected_alpha_1e-3_median_abs_corr"
            ]
        )

        row[
            "negative_minus_positive_below_median"
        ] = (
            row[
                "negative_median_abs_corr"
            ]
            - row[
                "positive_below_threshold_median_abs_corr"
            ]
        )

        region_rows.append(
            row
        )

    region_table = pd.DataFrame(
        region_rows
    )

    region_table = region_table.sort_values(
        "negative_minus_detected_median",
        ascending=False,
    )

    region_table.to_csv(
        REGION_TABLE,
        index=False,
    )

    # ========================================================
    # Heatmap
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7.2, 5.2)
    )

    extent = [
        edges[0],
        edges[-1],
        len(PAIRS) - 0.5,
        -0.5,
    ]

    image = ax.imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
    )

    pair_labels = [
        f"{p1} -- {p2}"
        for p1, p2
        in PAIRS
    ]

    ax.set_yticks(
        np.arange(
            len(PAIRS)
        ),
        pair_labels,
    )

    # --------------------------------------------------------
    # Important vertical references
    # --------------------------------------------------------

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.2,
    )

    threshold_x = float(
        signed_log10_1p(
            PRIMARY_THRESHOLD
        )
    )

    ax.axvline(
        threshold_x,
        linestyle=":",
        linewidth=1.4,
    )

    # --------------------------------------------------------
    # Original Delta-chi2 tick labels
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

    xmin = edges[0]
    xmax = edges[-1]

    keep = (
        (tick_x >= xmin)
        & (tick_x <= xmax)
    )

    labels = []

    for value in tick_T[
        keep
    ]:

        if abs(
            value
        ) >= 100:

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


    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        r"Median $|r_{ij}|$"
    )


    fig.savefig(
        FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    save_pdf_companion(fig, FIGURE)

    # ========================================================
    # Stdout
    # ========================================================

    print(
        "=" * 100
    )

    print(
        "H1 OFF-DIAGONAL CORRELATIONS BY LRT REGION"
    )

    print(
        "=" * 100
    )

    print()

    print(
        region_table[
            [
                "parameter_1",
                "parameter_2",
                "negative_median_abs_corr",
                "positive_below_threshold_median_abs_corr",
                "detected_alpha_1e-3_median_abs_corr",
                "negative_minus_detected_median",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Saved figure:",
        FIGURE,
    )

    print(
        "Saved region table:",
        REGION_TABLE,
    )

    print(
        "Saved bin table:",
        BIN_TABLE,
    )


if __name__ == "__main__":
    main()


