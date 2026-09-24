#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte-Carlo propagation of the fitted parallax covariance into |pi_E|
for the STRICTLY POSITIVE H1 LRT population.

NO simulation of light curves.
NO fitting.

For each event we use the frozen H1 fitted parallax vector

    mu = (piEN_hat, piEE_hat)

and its frozen 2x2 covariance block

    C_piE =
        [[C_NN, C_NE],
         [C_NE, C_EE]].

We draw

    (piEN, piEE)_k ~ N(mu, C_piE)

and transform every draw to

    piE_k = sqrt(piEN_k^2 + piEE_k^2).

This automatically propagates:
    - the piEN uncertainty,
    - the piEE uncertainty,
    - their covariance,
    - the nonlinear positive-definite transformation to |pi_E|.

This is NOT an MCMC or a likelihood resampling.
It is Monte-Carlo propagation of the local multivariate-normal
approximation encoded by the frozen fit covariance.

For each event we save:

    q16(|pi_E|)
    q50(|pi_E|)
    q84(|pi_E|)

and define the sigma-like central 68% half-width

    sigma68_MC = (q84 - q16) / 2.

The plotted quantity is

    log10(
        sigma68_MC / |pi_E,true|
    )

as a function of TRUE |pi_E|.

The figure compares:

    confused_fspl_no_parallax
    parallax_detected

within the STRICTLY POSITIVE LRT population.
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
    / "22_piE_amplitude_uncertainty_mc_vs_truth"
)

RESULT_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "22_piE_amplitude_uncertainty_mc_vs_truth"
)

FIGURE = (
    FIGURE_DIR
    / "piE_amplitude_uncertainty_mc_vs_truth.png"
)

EVENT_TABLE = (
    RESULT_DIR
    / "piE_amplitude_mc_event_intervals.parquet"
)

BIN_TABLE = (
    RESULT_DIR
    / "piE_amplitude_uncertainty_mc_vs_truth_bins.csv"
)


# ============================================================
# Frozen analysis configuration
# ============================================================

PRIMARY_THRESHOLD = 13.15982011480088

# Number of covariance samples PER EVENT.
#
# 1024 is enough for stable event-level q16/q50/q84 estimates for
# this population-level characterization while keeping the computation
# tractable for ~330k events.
N_MC = 1024

RNG_SEED = 22092026

# Process events in chunks so that we never allocate a
# (330k x N_MC x 2) array.
CHUNK_SIZE = 1000

N_BINS = 22

PIE_MIN = 0.01
PIE_MAX = 20.0

MIN_COUNT_PER_CLASS = 30


# ============================================================
# Helper
# ============================================================

def numeric(
    df,
    column,
):

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
    # Load frozen production quantities
    # ========================================================

    columns = [
        "catalog_row",
        "delta_chi2_lrt",
        "positive_lrt_class",

        "piE_true_amp",

        "h1_piEN",
        "h1_piEE",

        "h1_cov_piEN_piEN",
        "h1_cov_piEN_piEE",
        "h1_cov_piEE_piEE",
    ]

    df = pd.read_parquet(
        INPUT,
        columns=columns,
    )

    T = numeric(
        df,
        "delta_chi2_lrt",
    )

    piE_true = numeric(
        df,
        "piE_true_amp",
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

    classes = df[
        "positive_lrt_class"
    ].astype(
        str
    ).to_numpy()

    n_events = len(
        df
    )

    # ========================================================
    # Positive-population audit
    # ========================================================

    if not np.all(
        np.isfinite(T)
        & (
            T > 0
        )
    ):

        raise RuntimeError(
            "Input contains events outside the strict "
            "Delta-chi2 > 0 population."
        )

    detected = (
        T
        >= PRIMARY_THRESHOLD
    )

    confused = (
        (T > 0)
        & (
            T < PRIMARY_THRESHOLD
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
            "positive_lrt_class is inconsistent with "
            "delta_chi2_lrt and the frozen threshold."
        )

    # ========================================================
    # Fitted amplitude
    # ========================================================

    piE_fit = np.hypot(
        piEN_fit,
        piEE_fit,
    )

    # ========================================================
    # Validate the 2x2 covariance and construct its Cholesky
    # factor explicitly.
    #
    # For
    #
    #     C = [[a, c],
    #          [c, b]]
    #
    # L can be written as
    #
    #     L11 = sqrt(a)
    #     L21 = c / L11
    #     L22 = sqrt(b - L21^2)
    #
    # such that C = L L^T.
    #
    # Doing this explicitly avoids a batched np.linalg.cholesky()
    # failure caused by a single invalid matrix in a chunk.
    # ========================================================

    finite_covariance = (
        np.isfinite(C_NN)
        & np.isfinite(C_NE)
        & np.isfinite(C_EE)
        & (
            C_NN > 0
        )
        & (
            C_EE > 0
        )
    )

    L11 = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    L21 = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    L22_squared = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    L22 = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    L11[
        finite_covariance
    ] = np.sqrt(
        C_NN[
            finite_covariance
        ]
    )

    L21[
        finite_covariance
    ] = (
        C_NE[
            finite_covariance
        ]
        / L11[
            finite_covariance
        ]
    )

    L22_squared[
        finite_covariance
    ] = (
        C_EE[
            finite_covariance
        ]
        - L21[
            finite_covariance
        ] ** 2
    )

    positive_definite_covariance = (
        finite_covariance
        & np.isfinite(
            L22_squared
        )
        & (
            L22_squared > 0
        )
    )

    L22[
        positive_definite_covariance
    ] = np.sqrt(
        L22_squared[
            positive_definite_covariance
        ]
    )

    valid = (
        positive_definite_covariance
        & np.isfinite(
            piEN_fit
        )
        & np.isfinite(
            piEE_fit
        )
        & np.isfinite(
            piE_fit
        )
        & np.isfinite(
            piE_true
        )
        & (
            piE_true > 0
        )
    )

    valid_indices = np.flatnonzero(
        valid
    )

    # ========================================================
    # Allocate event-level Monte-Carlo summaries
    # ========================================================

    mc_q16 = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    mc_q50 = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    mc_q84 = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    # ========================================================
    # Monte Carlo
    # ========================================================

    rng = np.random.default_rng(
        RNG_SEED
    )

    n_valid = len(
        valid_indices
    )

    n_chunks = (
        n_valid
        + CHUNK_SIZE
        - 1
    ) // CHUNK_SIZE

    print(
        "=" * 110
    )

    print(
        "MONTE-CARLO PARALLAX-AMPLITUDE UNCERTAINTY"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "positive events =",
        n_events,
    )

    print(
        "valid positive-definite parallax covariance =",
        n_valid,
    )

    print(
        "invalid =",
        n_events
        - n_valid,
    )

    print(
        "MC samples per valid event =",
        N_MC,
    )

    print(
        "RNG seed =",
        RNG_SEED,
    )

    print(
        "chunk size =",
        CHUNK_SIZE,
    )

    print(
        "chunks =",
        n_chunks,
    )

    print()

    for ichunk, start in enumerate(
        range(
            0,
            n_valid,
            CHUNK_SIZE,
        ),
        start=1,
    ):

        stop = min(
            start
            + CHUNK_SIZE,
            n_valid,
        )

        idx = valid_indices[
            start:stop
        ]

        m = len(
            idx
        )

        # ----------------------------------------------------
        # Draw independent standard normals.
        #
        # Shape:
        #
        #     (events in chunk, MC samples, 2)
        # ----------------------------------------------------

        z = rng.standard_normal(
            size=(
                m,
                N_MC,
                2,
            )
        )

        z1 = z[
            :,
            :,
            0,
        ]

        z2 = z[
            :,
            :,
            1,
        ]

        # ----------------------------------------------------
        # Correlated draws using the event-specific Cholesky
        #
        # delta_N = L11 z1
        #
        # delta_E = L21 z1 + L22 z2
        # ----------------------------------------------------

        sample_piEN = (
            piEN_fit[
                idx,
                None
            ]
            + L11[
                idx,
                None
            ]
            * z1
        )

        sample_piEE = (
            piEE_fit[
                idx,
                None
            ]
            + L21[
                idx,
                None
            ]
            * z1
            + L22[
                idx,
                None
            ]
            * z2
        )

        # ----------------------------------------------------
        # Nonlinear transformation to amplitude
        # ----------------------------------------------------

        sample_piE = np.hypot(
            sample_piEN,
            sample_piEE,
        )

        quantiles = np.quantile(
            sample_piE,
            [
                0.16,
                0.50,
                0.84,
            ],
            axis=1,
        )

        mc_q16[
            idx
        ] = quantiles[
            0
        ]

        mc_q50[
            idx
        ] = quantiles[
            1
        ]

        mc_q84[
            idx
        ] = quantiles[
            2
        ]

        if (
            ichunk == 1
            or ichunk == n_chunks
            or ichunk % 25 == 0
        ):

            print(
                f"chunk {ichunk:4d}/{n_chunks:4d}"
                f"  events processed = {stop}/{n_valid}"
            )

    # ========================================================
    # Derived MC uncertainty quantities
    # ========================================================

    sigma68_mc = (
        0.5
        * (
            mc_q84
            - mc_q16
        )
    )

    lower_error_mc = (
        mc_q50
        - mc_q16
    )

    upper_error_mc = (
        mc_q84
        - mc_q50
    )

    relative_sigma68_true = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    good_sigma = (
        valid
        & np.isfinite(
            sigma68_mc
        )
        & (
            sigma68_mc > 0
        )
    )

    relative_sigma68_true[
        good_sigma
    ] = (
        sigma68_mc[
            good_sigma
        ]
        / piE_true[
            good_sigma
        ]
    )

    log10_relative_sigma68 = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    good_log = (
        good_sigma
        & np.isfinite(
            relative_sigma68_true
        )
        & (
            relative_sigma68_true > 0
        )
    )

    log10_relative_sigma68[
        good_log
    ] = np.log10(
        relative_sigma68_true[
            good_log
        ]
    )

    # ========================================================
    # Save event-level Monte Carlo results
    #
    # These will be reused later for the coverage test.
    # ========================================================

    event_table = pd.DataFrame(
        {
            "catalog_row":
                df[
                    "catalog_row"
                ].to_numpy(),

            "delta_chi2_lrt":
                T,

            "positive_lrt_class":
                classes,

            "piE_true_amp":
                piE_true,

            "h1_piEN":
                piEN_fit,

            "h1_piEE":
                piEE_fit,

            "piE_fit_amp":
                piE_fit,

            "mc_valid_covariance":
                valid,

            "mc_q16_piE_amp":
                mc_q16,

            "mc_q50_piE_amp":
                mc_q50,

            "mc_q84_piE_amp":
                mc_q84,

            "mc_lower_error_piE_amp":
                lower_error_mc,

            "mc_upper_error_piE_amp":
                upper_error_mc,

            "mc_sigma68_piE_amp":
                sigma68_mc,

            "mc_sigma68_over_true_piE":
                relative_sigma68_true,

            "log10_mc_sigma68_over_true_piE":
                log10_relative_sigma68,
        }
    )

    event_table.to_parquet(
        EVENT_TABLE,
        index=False,
    )

    # ========================================================
    # Class summaries
    # ========================================================

    regions = {
        "confused_fspl_no_parallax":
            confused,

        "parallax_detected":
            detected,
    }

    summary_rows = []

    for class_name, class_mask in (
        regions.items()
    ):

        use = (
            class_mask
            & good_log
        )

        values = relative_sigma68_true[
            use
        ]

        row = {
            "class":
                class_name,

            "n":
                int(
                    use.sum()
                ),

            "relative_sigma68_q16":
                float(
                    np.quantile(
                        values,
                        0.16,
                    )
                ),

            "relative_sigma68_median":
                float(
                    np.median(
                        values
                    )
                ),

            "relative_sigma68_q84":
                float(
                    np.quantile(
                        values,
                        0.84,
                    )
                ),

            "fraction_sigma68_lt_true_piE":
                float(
                    np.mean(
                        values
                        < 1.0
                    )
                ),
        }

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # Bin versus TRUE |pi_E|
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
                good_log
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
                good_log
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

        for class_name, class_mask in (
            regions.items()
        ):

            use = (
                in_bin
                & class_mask
            )

            values = log10_relative_sigma68[
                use
            ]

            n = len(
                values
            )

            if n:

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
                        int(
                            n
                        ),

                    "log10_relative_sigma68_q16":
                        q16,

                    "log10_relative_sigma68_median":
                        median,

                    "log10_relative_sigma68_q84":
                        q84,

                    "shown_in_figure":
                        (
                            n
                            >= MIN_COUNT_PER_CLASS
                        ),
                }
            )

    binned = pd.DataFrame(
        rows
    )

    binned.to_csv(
        BIN_TABLE,
        index=False,
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.8, 4.4)
    )

    for class_name in [
        "confused_fspl_no_parallax",
        "parallax_detected",
    ]:

        sub = binned.loc[
            (
                binned[
                    "class"
                ]
                == class_name
            )
            & binned[
                "shown_in_figure"
            ]
        ].copy()

        x = sub[
            "piE_true_center"
        ].to_numpy(
            dtype=float
        )

        median = sub[
            "log10_relative_sigma68_median"
        ].to_numpy(
            dtype=float
        )

        q16 = sub[
            "log10_relative_sigma68_q16"
        ].to_numpy(
            dtype=float
        )

        q84 = sub[
            "log10_relative_sigma68_q84"
        ].to_numpy(
            dtype=float
        )

        good = (
            np.isfinite(x)
            & np.isfinite(median)
            & np.isfinite(q16)
            & np.isfinite(q84)
        )

        label = (
            "Confused with FSPL without parallax"
            if class_name
            == "confused_fspl_no_parallax"
            else "Parallax detected"
        )

        line, = ax.plot(
            x[
                good
            ],
            median[
                good
            ],
            marker="o",
            markersize=4.0,
            linewidth=1.5,
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
            alpha=0.18,
            color=line.get_color(),
        )

    # --------------------------------------------------------
    # Reference:
    #
    # sigma68_MC = true |pi_E|
    # --------------------------------------------------------

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        label=(
            r"$\sigma_{68,\rm MC}"
            r"=|\pi_{E,\rm true}|$"
        ),
    )

    ax.set_xscale(
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
            r"$\log_{10}"
            r"\left("
            r"\sigma_{68,\rm MC}"
            r"/"
            r"|\boldsymbol{\pi}_{E,\rm true}|"
            r"\right)$"
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

    # ========================================================
    # Final output
    # ========================================================

    print()

    print(
        "Saved figure:",
        FIGURE,
    )

    print(
        "Saved event-level MC intervals:",
        EVENT_TABLE,
    )

    print(
        "Saved bins:",
        BIN_TABLE,
    )


if __name__ == "__main__":
    main()


