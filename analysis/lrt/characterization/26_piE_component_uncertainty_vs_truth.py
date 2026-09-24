#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Formal marginal uncertainties of piEN and piEE versus TRUE |pi_E|
for the STRICTLY POSITIVE H1 LRT population.

NO simulation.
NO fitting.
NO Monte Carlo required.

For a multivariate-normal approximation with covariance C_piE,

    sigma_piEN = sqrt(C_NN)
    sigma_piEE = sqrt(C_EE)

are the exact marginal standard deviations of the two fitted parallax
components.

We characterize

    sigma_piEN / |piE_true|

and

    sigma_piEE / |piE_true|

as functions of TRUE |pi_E|, separately for:

    confused_fspl_no_parallax
    parallax_detected.

Normalizing by the TOTAL true parallax amplitude avoids artificial
divergences when an individual true component passes through zero.
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

ROOT = Path(__file__).resolve().parents[3]

INPUT = (
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
    / "26_piE_component_uncertainty_vs_truth"
)

RESULT_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "26_piE_component_uncertainty_vs_truth"
)

FIGURE = (
    FIGURE_DIR
    / "piE_component_uncertainty_vs_truth.png"
)

TABLE = (
    RESULT_DIR
    / "piE_component_uncertainty_vs_truth_bins.csv"
)


# ============================================================
# Configuration
# ============================================================

PRIMARY_THRESHOLD = 13.15982011480088

N_BINS = 22

PIE_MIN = 0.01
PIE_MAX = 20.0

MIN_COUNT_PER_CLASS = 30


# ============================================================
# Helper
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

    # ========================================================
    # Load frozen fit quantities
    # ============================================================

    df = pd.read_parquet(
        INPUT,
        columns=[
            "catalog_row",
            "delta_chi2_lrt",
            "positive_lrt_class",

            "piE_true_amp",

            "h1_cov_piEN_piEN",
            "h1_cov_piEE_piEE",
        ],
    )

    T = numeric(
        df,
        "delta_chi2_lrt",
    )

    piE_true = numeric(
        df,
        "piE_true_amp",
    )

    C_NN = numeric(
        df,
        "h1_cov_piEN_piEN",
    )

    C_EE = numeric(
        df,
        "h1_cov_piEE_piEE",
    )

    classes = df[
        "positive_lrt_class"
    ].astype(
        str
    ).to_numpy()

    # ========================================================
    # Population audit
    # ============================================================

    if not np.all(
        np.isfinite(T)
        & (T > 0)
    ):

        raise RuntimeError(
            "Input is not the strict positive-LRT population."
        )

    detected = (
        T
        >= PRIMARY_THRESHOLD
    )

    confused = (
        (T > 0)
        & (
            T
            < PRIMARY_THRESHOLD
        )
    )

    expected_class = np.where(
        detected,
        "parallax_detected",
        "confused_fspl_no_parallax",
    )

    if not np.array_equal(
        classes,
        expected_class,
    ):

        raise RuntimeError(
            "positive_lrt_class does not match the frozen threshold."
        )

    # ========================================================
    # Marginal uncertainties
    # ============================================================

    sigma_N = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    sigma_E = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    valid_N = (
        np.isfinite(C_NN)
        & (C_NN > 0)
    )

    valid_E = (
        np.isfinite(C_EE)
        & (C_EE > 0)
    )

    sigma_N[
        valid_N
    ] = np.sqrt(
        C_NN[
            valid_N
        ]
    )

    sigma_E[
        valid_E
    ] = np.sqrt(
        C_EE[
            valid_E
        ]
    )

    valid_truth = (
        np.isfinite(piE_true)
        & (piE_true > 0)
    )

    relative_sigma_N = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    relative_sigma_E = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    valid_rel_N = (
        valid_truth
        & valid_N
    )

    valid_rel_E = (
        valid_truth
        & valid_E
    )

    relative_sigma_N[
        valid_rel_N
    ] = (
        sigma_N[
            valid_rel_N
        ]
        / piE_true[
            valid_rel_N
        ]
    )

    relative_sigma_E[
        valid_rel_E
    ] = (
        sigma_E[
            valid_rel_E
        ]
        / piE_true[
            valid_rel_E
        ]
    )

    # ========================================================
    # Overall summary
    # ========================================================

    print(
        "=" * 105
    )

    print(
        "PARALLAX COMPONENT FORMAL UNCERTAINTIES"
    )

    print(
        "=" * 105
    )

    print()

    regions = {
        "confused_fspl_no_parallax":
            confused,

        "parallax_detected":
            detected,
    }

    summary_rows = []

    for class_name, class_mask in regions.items():

        for component, values in [
            (
                "piEN",
                relative_sigma_N,
            ),
            (
                "piEE",
                relative_sigma_E,
            ),
        ]:

            use = (
                class_mask
                & np.isfinite(
                    values
                )
                & (
                    values > 0
                )
            )

            x = values[
                use
            ]

            summary_rows.append(
                {
                    "class":
                        class_name,

                    "component":
                        component,

                    "n":
                        int(
                            len(x)
                        ),

                    "relative_sigma_q16":
                        float(
                            np.quantile(
                                x,
                                0.16,
                            )
                        ),

                    "relative_sigma_median":
                        float(
                            np.median(
                                x
                            )
                        ),

                    "relative_sigma_q84":
                        float(
                            np.quantile(
                                x,
                                0.84,
                            )
                        ),

                    "fraction_sigma_lt_true_piE":
                        float(
                            np.mean(
                                x < 1.0
                            )
                        ),
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

    # ========================================================
    # Bin by TRUE |pi_E|
    # ========================================================

    edges = np.geomspace(
        PIE_MIN,
        PIE_MAX,
        N_BINS + 1,
    )

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
                valid_truth
                & (
                    piE_true
                    >= left
                )
                & (
                    piE_true
                    <= right
                )
            )

        else:

            in_bin = (
                valid_truth
                & (
                    piE_true
                    >= left
                )
                & (
                    piE_true
                    < right
                )
            )

        center = np.sqrt(
            left
            * right
        )

        for class_name, class_mask in regions.items():

            for component, values in [
                (
                    "piEN",
                    relative_sigma_N,
                ),
                (
                    "piEE",
                    relative_sigma_E,
                ),
            ]:

                use = (
                    in_bin
                    & class_mask
                    & np.isfinite(
                        values
                    )
                    & (
                        values > 0
                    )
                )

                x = values[
                    use
                ]

                n = len(
                    x
                )

                if n:

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
                            int(
                                n
                            ),

                        "relative_sigma_q16":
                            q16,

                        "relative_sigma_median":
                            median,

                        "relative_sigma_q84":
                            q84,

                        "shown_in_figure":
                            (
                                n
                                >= MIN_COUNT_PER_CLASS
                            ),
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
        figsize=(6.8, 4.4)
    )

    plot_definitions = [
        (
            "confused_fspl_no_parallax",
            "piEN",
            r"Confused: $\sigma_{\pi_{E,N}}$",
            "-",
            "o",
        ),
        (
            "confused_fspl_no_parallax",
            "piEE",
            r"Confused: $\sigma_{\pi_{E,E}}$",
            "--",
            "o",
        ),
        (
            "parallax_detected",
            "piEN",
            r"Detected: $\sigma_{\pi_{E,N}}$",
            "-",
            "s",
        ),
        (
            "parallax_detected",
            "piEE",
            r"Detected: $\sigma_{\pi_{E,E}}$",
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
                result[
                    "class"
                ]
                == class_name
            )
            & (
                result[
                    "component"
                ]
                == component
            )
            & result[
                "shown_in_figure"
            ]
        ].copy()

        x = sub[
            "piE_true_center"
        ].to_numpy(
            dtype=float
        )

        median = sub[
            "relative_sigma_median"
        ].to_numpy(
            dtype=float
        )

        q16 = sub[
            "relative_sigma_q16"
        ].to_numpy(
            dtype=float
        )

        q84 = sub[
            "relative_sigma_q84"
        ].to_numpy(
            dtype=float
        )

        good = (
            np.isfinite(x)
            & np.isfinite(median)
            & np.isfinite(q16)
            & np.isfinite(q84)
            & (median > 0)
            & (q16 > 0)
            & (q84 > 0)
        )

        line, = ax.plot(
            x[
                good
            ],
            median[
                good
            ],
            marker=marker,
            markersize=4.0,
            linewidth=1.5,
            linestyle=linestyle,
            label=label,
        )

        ax.fill_between(
            x[
                good
            ],
            q16[
                good
            ],
            q84[
                good
            ],
            alpha=0.10,
            color=line.get_color(),
        )

    ax.axhline(
        1.0,
        linestyle=":",
        linewidth=1.0,
        label=(
            r"$\sigma_{\pi_{E,i}}"
            r"=|\boldsymbol{\pi}_{E,\rm true}|$"
        ),
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlim(
        PIE_MIN,
        PIE_MAX,
    )

    ax.set_xlabel(
        r"True $|\boldsymbol{\pi}_E|$"
    )

    ax.set_ylabel(
        (
            r"Marginal uncertainty "
            r"$\sigma_{\pi_{E,i}}"
            r"/|\boldsymbol{\pi}_{E,\rm true}|$"
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


