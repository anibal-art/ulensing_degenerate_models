#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H1 fitted parameters and marginal uncertainties versus Delta chi^2_LRT.

NO simulation.
NO fitting.

One figure, using the complete H1-generated population.

Top row:
    |u0|
    tE
    rho
    |piE|

Bottom row:
    sigma_u0
    sigma_tE / tE
    sigma_rho / rho
    sigma_piE_parallel / |piE|

For the parallax vector p = (piEN, piEE),

    n = p / |p|

and

    sigma_piE_parallel^2 = n^T C_piE n.

This is the uncertainty projected along the fitted parallax direction.

The purpose is to determine whether the Delta-chi2 < 0 population occupies
a special physical/fit-parameter regime and whether its formal uncertainties
increase continuously as the LRT weakens.
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
    / "17_h1_parameters_uncertainties_vs_lrt"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "characterization"
    / "17_h1_parameters_uncertainties_vs_lrt"
)

FIGURE = (
    FIGURE_DIR
    / "h1_parameters_uncertainties_vs_lrt.png"
)

TABLE = (
    RESULT_DIR
    / "h1_parameters_uncertainties_vs_lrt_bins.csv"
)


# ============================================================
# Configuration
# ============================================================

N_BINS = 45

PRIMARY_THRESHOLD = 13.15982011480088


# ============================================================
# Signed-log LRT coordinate
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
# Numeric helper
# ============================================================

def numeric(df, column):

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(
        dtype=float
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

        "h1_u0",
        "h1_tE",
        "h1_rho",
        "h1_piEN",
        "h1_piEE",

        "h1_sigma_u0",
        "h1_sigma_tE",
        "h1_sigma_rho",

        "h1_cov_piEN_piEN",
        "h1_cov_piEN_piEE",
        "h1_cov_piEE_piEE",
    ]

    df = pd.read_parquet(
        INPUT,
        columns=columns,
    )

    # ========================================================
    # Basic arrays
    # ========================================================

    T = numeric(
        df,
        "delta_chi2_lrt",
    )

    u0 = numeric(
        df,
        "h1_u0",
    )

    tE = numeric(
        df,
        "h1_tE",
    )

    rho = numeric(
        df,
        "h1_rho",
    )

    piEN = numeric(
        df,
        "h1_piEN",
    )

    piEE = numeric(
        df,
        "h1_piEE",
    )

    sigma_u0 = numeric(
        df,
        "h1_sigma_u0",
    )

    sigma_tE = numeric(
        df,
        "h1_sigma_tE",
    )

    sigma_rho = numeric(
        df,
        "h1_sigma_rho",
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

    # ========================================================
    # Parameter estimates
    # ========================================================

    abs_u0 = np.abs(
        u0
    )

    piE = np.hypot(
        piEN,
        piEE,
    )

    # ========================================================
    # Relative / natural uncertainty scales
    # ========================================================

    rel_sigma_tE = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(tE)
        & np.isfinite(sigma_tE)
        & (tE > 0)
        & (sigma_tE >= 0)
    )

    rel_sigma_tE[valid] = (
        sigma_tE[valid]
        / tE[valid]
    )

    rel_sigma_rho = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(rho)
        & np.isfinite(sigma_rho)
        & (rho > 0)
        & (sigma_rho >= 0)
    )

    rel_sigma_rho[valid] = (
        sigma_rho[valid]
        / rho[valid]
    )

    # ========================================================
    # Parallax uncertainty projected along fitted piE
    # ========================================================

    sigma_piE_parallel = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    rel_sigma_piE_parallel = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(piEN)
        & np.isfinite(piEE)
        & np.isfinite(piE)
        & (piE > 0)
        & np.isfinite(C_NN)
        & np.isfinite(C_NE)
        & np.isfinite(C_EE)
    )

    nN = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    nE = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    nN[valid] = (
        piEN[valid]
        / piE[valid]
    )

    nE[valid] = (
        piEE[valid]
        / piE[valid]
    )

    variance_parallel = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    variance_parallel[valid] = (
        nN[valid] ** 2
        * C_NN[valid]

        + 2.0
        * nN[valid]
        * nE[valid]
        * C_NE[valid]

        + nE[valid] ** 2
        * C_EE[valid]
    )

    valid_variance = (
        valid
        & np.isfinite(
            variance_parallel
        )
        & (
            variance_parallel
            >= 0
        )
    )

    sigma_piE_parallel[
        valid_variance
    ] = np.sqrt(
        variance_parallel[
            valid_variance
        ]
    )

    rel_sigma_piE_parallel[
        valid_variance
    ] = (
        sigma_piE_parallel[
            valid_variance
        ]
        / piE[
            valid_variance
        ]
    )

    # ========================================================
    # Quantities to bin
    # ========================================================

    metrics = {
        "abs_u0":
            abs_u0,

        "tE":
            tE,

        "rho":
            rho,

        "piE":
            piE,

        "sigma_u0":
            sigma_u0,

        "rel_sigma_tE":
            rel_sigma_tE,

        "rel_sigma_rho":
            rel_sigma_rho,

        "rel_sigma_piE_parallel":
            rel_sigma_piE_parallel,
    }

    # ========================================================
    # LRT coordinate
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

        row = {
            "bin_index":
                ibin,

            "delta_chi2_center":
                center_T,

            "n_total":
                n_total,
        }

        for name, values in (
            metrics.items()
        ):

            good = (
                in_bin
                & np.isfinite(
                    values
                )
            )

            x = values[
                good
            ]

            # Positive quantities are expected here.
            x = x[
                x >= 0
            ]

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

        rows.append(
            row
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

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.2, 5.0),
        sharex=True,
    )

    x = signed_log10_1p(
        result[
            "delta_chi2_center"
        ].to_numpy(
            dtype=float
        )
    )

    panels = [
        (
            "abs_u0",
            r"$|\hat u_0|$",
            True,
        ),

        (
            "tE",
            r"$\hat t_E$",
            True,
        ),

        (
            "rho",
            r"$\hat\rho$",
            True,
        ),

        (
            "piE",
            r"$|\hat{\boldsymbol{\pi}}_E|$",
            True,
        ),

        (
            "sigma_u0",
            r"$\sigma_{u_0}$",
            True,
        ),

        (
            "rel_sigma_tE",
            r"$\sigma_{t_E}/t_E$",
            True,
        ),

        (
            "rel_sigma_rho",
            r"$\sigma_\rho/\rho$",
            True,
        ),

        (
            "rel_sigma_piE_parallel",
            (
                r"$\sigma_{\pi_E,\parallel}"
                r"/|\hat{\boldsymbol{\pi}}_E|$"
            ),
            True,
        ),
    ]

    threshold_x = float(
        signed_log10_1p(
            PRIMARY_THRESHOLD
        )
    )

    for ax, (
        metric,
        ylabel,
        use_log_y,
    ) in zip(
        axes.flat,
        panels,
    ):

        median = result[
            f"{metric}_median"
        ].to_numpy(
            dtype=float
        )

        q16 = result[
            f"{metric}_q16"
        ].to_numpy(
            dtype=float
        )

        q84 = result[
            f"{metric}_q84"
        ].to_numpy(
            dtype=float
        )

        valid = (
            np.isfinite(x)
            & np.isfinite(median)
            & np.isfinite(q16)
            & np.isfinite(q84)
        )

        if use_log_y:

            valid = (
                valid
                & (median > 0)
                & (q16 > 0)
                & (q84 > 0)
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
            alpha=0.18,
        )

        ax.plot(
            x[
                valid
            ],
            median[
                valid
            ],
            marker="o",
            markersize=2.7,
            linewidth=1.2,
        )

        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=0.9,
        )

        ax.axvline(
            threshold_x,
            linestyle=":",
            linewidth=1.0,
        )

        if use_log_y:

            ax.set_yscale(
                "log"
            )

        ax.set_ylabel(
            ylabel
        )

        ax.grid(
            alpha=0.22
        )

    # ========================================================
    # Original Delta-chi2 labels
    # ========================================================

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

    tick_labels = []

    for value in tick_T[
        keep
    ]:

        if abs(
            value
        ) >= 100:

            tick_labels.append(
                rf"$10^{{{int(np.log10(abs(value)))}}}$"
            )

        else:

            tick_labels.append(
                f"{value:g}"
            )

    for ax in axes[
        1,
        :
    ]:

        ax.set_xticks(
            tick_x[
                keep
            ],
            tick_labels,
        )

        ax.set_xlabel(
            r"$\Delta\chi^2_{\rm LRT}$"
        )



    label_panels(axes)
    fig.savefig(
        FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    save_pdf_companion(fig, FIGURE)

    # ========================================================
    # Region-level summary
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

    summary_rows = []

    for region_name, mask in (
        regions.items()
    ):

        row = {
            "region":
                region_name,

            "n":
                int(
                    np.sum(
                        mask
                    )
                ),
        }

        for name, values in (
            metrics.items()
        ):

            good = (
                mask
                & np.isfinite(
                    values
                )
            )

            x_region = values[
                good
            ]

            row[
                name
            ] = (
                float(
                    np.median(
                        x_region
                    )
                )
                if len(
                    x_region
                )
                else np.nan
            )

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        "=" * 110
    )

    print(
        "H1 PARAMETERS AND UNCERTAINTIES BY LRT REGION"
    )

    print(
        "=" * 110
    )

    print()

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


