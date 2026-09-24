#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lens-mass measurability AFTER annual parallax has been detected.

NO simulation.
NO fitting.

Scientific scope
----------------
This analysis is explicitly conditional on annual-parallax detection:

    positive_lrt_class == "parallax_detected"

The question is then:

    Among events for which annual parallax is distinguishable,
    how many have sufficient finite-source information to support
    a lens-mass estimate through

        theta_E = theta_* / rho

    and

        M_L = theta_* / (kappa rho pi_E) ?

Important distinction
---------------------
The criterion

    sigma_rho / rho_hat < 0.3

is NOT treated as a genuine finite-source detection criterion.

It is only an operational proxy for locally precise rho.

Because we know the simulation truth, we explicitly audit whether
events passing this proxy actually recover rho correctly.

A proper observational finite-source detection should ultimately use
something like an FSPL-vs-PSPL likelihood-ratio / Delta-chi2 criterion.

Mass propagation
----------------
For the proxy-selected sample, rho uncertainty is propagated in log rho,
not in rho.

Define

    y = ln rho.

Using the delta method, the H1 covariance block

    (rho, piEN, piEE)

is transformed into

    (ln rho, piEN, piEE)

through

    Var(ln rho)       = Var(rho) / rho_hat^2
    Cov(ln rho, piEN) = Cov(rho,piEN) / rho_hat
    Cov(ln rho, piEE) = Cov(rho,piEE) / rho_hat.

Monte-Carlo draws are then made in

    (ln rho, piEN, piEE)

and transformed with

    rho = exp(ln rho),

which guarantees rho > 0 without truncating a Gaussian in rho.

Assuming theta_* known exactly,

    M_hat / M_true
        =
    rho_true piE_true
    ----------------
    rho_draw piE_draw.

Outputs
-------
Figure 1:
    finite_source_information_after_parallax_detection.png/pdf

    Panel A:
        fraction of parallax-detected events satisfying
        sigma_rho/rho_hat < threshold.

    Panel B:
        actual rho recovery versus formal relative uncertainty.

Figure 2:
    lens_mass_recovery_selected_logrho.png/pdf

    Panel A:
        mass recovery versus true R_FS for the proxy-selected sample.

    Panel B:
        empirical coverage of the propagated central 68% mass interval.

Event and binned tables are also written.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT
    / "analysis/lrt/results/characterization"
    / "18_positive_truth"
    / "h1_positive_truth_characterization.parquet"
)

FIGURE_DIR = (
    ROOT
    / "analysis/lrt/figures/characterization"
    / "44_mass_measurability_after_parallax"
)

RESULT_DIR = (
    ROOT
    / "analysis/lrt/results/characterization"
    / "44_mass_measurability_after_parallax"
)

FIGURE_INFORMATION = (
    FIGURE_DIR
    / "finite_source_information_after_parallax_detection"
)

FIGURE_MASS = (
    FIGURE_DIR
    / "lens_mass_recovery_selected_logrho"
)

EVENT_TABLE = (
    RESULT_DIR
    / "parallax_detected_mass_measurability_events.parquet"
)

THRESHOLD_TABLE = (
    RESULT_DIR
    / "rho_information_threshold_scan.csv"
)

RHO_RECOVERY_TABLE = (
    RESULT_DIR
    / "rho_recovery_vs_relative_uncertainty_bins.csv"
)

MASS_RECOVERY_TABLE = (
    RESULT_DIR
    / "mass_recovery_selected_vs_true_RFS_bins.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "mass_measurability_after_parallax_summary.txt"
)


# ============================================================================
# Configuration
# ============================================================================

RHO_RELATIVE_UNCERTAINTY_CUT = 0.30

# Truth-side audit only.
#
# 0.30 dex corresponds to approximately a factor of 2.
RHO_ACCURACY_DEX = 0.30

N_MC = 1024

BATCH_SIZE = 1000

RNG_SEED = 20260924

N_BINS = 16

MIN_BIN_COUNT = 20

NOMINAL_COVERAGE = 0.6826894921370859

EXPECTED_PARALLAX_DETECTED = 229921


# ============================================================================
# Style
# ============================================================================

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


# ============================================================================
# Helpers
# ============================================================================

def numeric(
    df,
    column,
):

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def save_figure(
    fig,
    base,
):

    png = base.with_suffix(
        ".png"
    )

    pdf = base.with_suffix(
        ".pdf"
    )

    fig.savefig(
        png
    )

    fig.savefig(
        pdf
    )

    print(
        "Saved:",
        png,
    )

    print(
        "Saved:",
        pdf,
    )


def build_log_edges(
    values,
    n_bins,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & (
            values > 0.0
        )
    )

    if not np.any(
        valid
    ):

        raise RuntimeError(
            "No finite positive values for log binning."
        )

    lo = float(
        np.nanmin(
            values[
                valid
            ]
        )
    )

    hi = float(
        np.nanmax(
            values[
                valid
            ]
        )
    )

    edges = np.geomspace(
        lo,
        hi,
        n_bins + 1,
    )

    edges[0] = np.nextafter(
        lo,
        -np.inf,
    )

    edges[-1] = np.nextafter(
        hi,
        np.inf,
    )

    return edges


def quantiles_or_nan(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    return tuple(
        np.quantile(
            values,
            [
                0.16,
                0.50,
                0.84,
            ],
        )
    )


def wilson_interval(
    k,
    n,
    z=1.959963984540054,
):

    if n <= 0:

        return (
            np.nan,
            np.nan,
        )

    p = (
        k
        / n
    )

    denominator = (
        1.0
        + z**2
        / n
    )

    center = (
        p
        + z**2
        / (
            2.0 * n
        )
    ) / denominator

    half = (
        z
        * np.sqrt(
            p
            * (
                1.0 - p
            )
            / n
            + z**2
            / (
                4.0 * n**2
            )
        )
        / denominator
    )

    return (
        center - half,
        center + half,
    )


# ============================================================================
# Input
# ============================================================================

def load_parallax_detected():

    columns = [
        "catalog_row",
        "delta_chi2_lrt",
        "positive_lrt_class",

        "rho_true",
        "u0_true",
        "rho_over_abs_u0_true",
        "piE_true_amp",

        "h1_rho",
        "h1_piEN",
        "h1_piEE",

        "h1_cov_rho_rho",
        "h1_cov_rho_piEN",
        "h1_cov_rho_piEE",

        "h1_cov_piEN_piEN",
        "h1_cov_piEN_piEE",
        "h1_cov_piEE_piEE",
    ]

    df = pd.read_parquet(
        INPUT,
        columns=columns,
    )

    detected = (
        df[
            "positive_lrt_class"
        ]
        .astype(str)
        == "parallax_detected"
    )

    work = df.loc[
        detected
    ].copy()

    if len(
        work
    ) != EXPECTED_PARALLAX_DETECTED:

        raise RuntimeError(
            "Unexpected number of parallax-detected events: "
            f"{len(work):,} != {EXPECTED_PARALLAX_DETECTED:,}"
        )

    return work


# ============================================================================
# Event-level diagnostics
# ============================================================================

def build_event_diagnostics(
    df,
):

    out = df.copy()

    rho_true = numeric(
        out,
        "rho_true",
    )

    u0_true = numeric(
        out,
        "u0_true",
    )

    piE_true = numeric(
        out,
        "piE_true_amp",
    )

    rho_hat = numeric(
        out,
        "h1_rho",
    )

    piEN_hat = numeric(
        out,
        "h1_piEN",
    )

    piEE_hat = numeric(
        out,
        "h1_piEE",
    )

    C_rr = numeric(
        out,
        "h1_cov_rho_rho",
    )

    piE_hat = np.hypot(
        piEN_hat,
        piEE_hat,
    )

    sigma_rho = np.full(
        len(out),
        np.nan,
    )

    valid_variance = (
        np.isfinite(C_rr)
        & (
            C_rr >= 0.0
        )
    )

    sigma_rho[
        valid_variance
    ] = np.sqrt(
        C_rr[
            valid_variance
        ]
    )

    rho_relative_uncertainty = np.full(
        len(out),
        np.nan,
    )

    valid_relative = (
        np.isfinite(
            sigma_rho
        )
        & np.isfinite(
            rho_hat
        )
        & (
            rho_hat > 0.0
        )
    )

    rho_relative_uncertainty[
        valid_relative
    ] = (
        sigma_rho[
            valid_relative
        ]
        / rho_hat[
            valid_relative
        ]
    )

    delta_log10_rho = np.full(
        len(out),
        np.nan,
    )

    valid_rho_recovery = (
        np.isfinite(
            rho_true
        )
        & (
            rho_true > 0.0
        )
        & np.isfinite(
            rho_hat
        )
        & (
            rho_hat > 0.0
        )
    )

    delta_log10_rho[
        valid_rho_recovery
    ] = np.log10(
        rho_hat[
            valid_rho_recovery
        ]
        / rho_true[
            valid_rho_recovery
        ]
    )

    rfs_true = np.full(
        len(out),
        np.nan,
    )

    valid_rfs = (
        np.isfinite(
            rho_true
        )
        & (
            rho_true > 0.0
        )
        & np.isfinite(
            u0_true
        )
        & (
            np.abs(
                u0_true
            ) > 0.0
        )
    )

    rfs_true[
        valid_rfs
    ] = (
        rho_true[
            valid_rfs
        ]
        / np.abs(
            u0_true[
                valid_rfs
            ]
        )
    )

    # Operational proxy only.
    rho_information_proxy = (
        valid_relative
        & (
            rho_relative_uncertainty
            < RHO_RELATIVE_UNCERTAINTY_CUT
        )
    )

    # Truth-side audit. NEVER use this as an observational selection.
    rho_accurate_truth = (
        np.isfinite(
            delta_log10_rho
        )
        & (
            np.abs(
                delta_log10_rho
            )
            < RHO_ACCURACY_DEX
        )
    )

    mass_ratio_point = np.full(
        len(out),
        np.nan,
    )

    valid_mass_point = (
        valid_rho_recovery
        & np.isfinite(
            piE_true
        )
        & (
            piE_true > 0.0
        )
        & np.isfinite(
            piE_hat
        )
        & (
            piE_hat > 0.0
        )
    )

    mass_ratio_point[
        valid_mass_point
    ] = (
        rho_true[
            valid_mass_point
        ]
        * piE_true[
            valid_mass_point
        ]
        / (
            rho_hat[
                valid_mass_point
            ]
            * piE_hat[
                valid_mass_point
            ]
        )
    )

    out[
        "sigma_rho"
    ] = sigma_rho

    out[
        "sigma_rho_over_rho_hat"
    ] = rho_relative_uncertainty

    out[
        "delta_log10_rho"
    ] = delta_log10_rho

    out[
        "rfs_true"
    ] = rfs_true

    out[
        "rho_information_proxy"
    ] = rho_information_proxy

    out[
        "rho_accurate_truth"
    ] = rho_accurate_truth

    out[
        "piE_hat_amp"
    ] = piE_hat

    out[
        "mass_ratio_point"
    ] = mass_ratio_point

    out[
        "delta_log10_mass_point"
    ] = np.log10(
        mass_ratio_point
    )

    return out


# ============================================================================
# Figure 1:
# finite-source information after parallax detection
# ============================================================================

def make_information_figure(
    events,
):

    relative = events[
        "sigma_rho_over_rho_hat"
    ].to_numpy(dtype=float)

    delta_rho = events[
        "delta_log10_rho"
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(
            relative
        )
        & (
            relative > 0.0
        )
        & np.isfinite(
            delta_rho
        )
    )

    if not np.any(
        valid
    ):

        raise RuntimeError(
            "No events with valid rho uncertainty and recovery."
        )

    # ----------------------------------------------------------------------
    # Threshold scan
    # ----------------------------------------------------------------------

    thresholds = np.logspace(
        -2.0,
        2.5,
        120,
    )

    rows = []

    n_detected = len(
        events
    )

    for threshold in thresholds:

        selected = (
            valid
            & (
                relative
                < threshold
            )
        )

        n = int(
            selected.sum()
        )

        fraction_all_detected = (
            n
            / n_detected
        )

        if n:

            truth_accurate_fraction = float(
                np.mean(
                    np.abs(
                        delta_rho[
                            selected
                        ]
                    )
                    < RHO_ACCURACY_DEX
                )
            )

            median_abs_error = float(
                np.median(
                    np.abs(
                        delta_rho[
                            selected
                        ]
                    )
                )
            )

        else:

            truth_accurate_fraction = np.nan
            median_abs_error = np.nan

        rows.append(
            {
                "relative_uncertainty_threshold":
                    threshold,

                "n_selected":
                    n,

                "fraction_of_parallax_detected":
                    fraction_all_detected,

                "truth_fraction_rho_within_0p3dex":
                    truth_accurate_fraction,

                "median_abs_log10_rho_error":
                    median_abs_error,
            }
        )

    threshold_table = pd.DataFrame(
        rows
    )

    threshold_table.to_csv(
        THRESHOLD_TABLE,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Binned rho recovery versus formal relative uncertainty
    # ----------------------------------------------------------------------

    edges = build_log_edges(
        relative[
            valid
        ],
        N_BINS,
    )

    recovery_rows = []

    for i in range(
        len(edges)
        - 1
    ):

        lo = edges[
            i
        ]

        hi = edges[
            i + 1
        ]

        mask = (
            valid
            & (
                relative
                >= lo
            )
            & (
                relative
                < hi
            )
        )

        q16, median, q84 = quantiles_or_nan(
            delta_rho[
                mask
            ]
        )

        n = int(
            mask.sum()
        )

        if n:

            fraction_accurate = float(
                np.mean(
                    np.abs(
                        delta_rho[
                            mask
                        ]
                    )
                    < RHO_ACCURACY_DEX
                )
            )

        else:

            fraction_accurate = np.nan

        recovery_rows.append(
            {
                "relative_uncertainty_left":
                    lo,

                "relative_uncertainty_right":
                    hi,

                "relative_uncertainty_center":
                    np.sqrt(
                        lo
                        * hi
                    ),

                "n":
                    n,

                "delta_log10_rho_q16":
                    q16,

                "delta_log10_rho_median":
                    median,

                "delta_log10_rho_q84":
                    q84,

                "truth_fraction_rho_within_0p3dex":
                    fraction_accurate,
            }
        )

    recovery_table = pd.DataFrame(
        recovery_rows
    )

    recovery_table.to_csv(
        RHO_RECOVERY_TABLE,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            12.0,
            5.2,
        ),
    )

    # Panel A:
    # fraction selected as a function of formal rho precision threshold

    axes[
        0
    ].plot(
        threshold_table[
            "relative_uncertainty_threshold"
        ],
        threshold_table[
            "fraction_of_parallax_detected"
        ],
        label=(
            "Passing formal "
            r"$\rho$-precision proxy"
        ),
    )

    axes[
        0
    ].plot(
        threshold_table[
            "relative_uncertainty_threshold"
        ],
        threshold_table[
            "truth_fraction_rho_within_0p3dex"
        ],
        label=(
            r"Among selected: "
            r"$|\Delta\log_{10}\rho|<0.3$"
        ),
    )

    axes[
        0
    ].axvline(
        RHO_RELATIVE_UNCERTAINTY_CUT,
        linestyle="--",
        color="black",
        linewidth=1.4,
        label=(
            rf"Adopted proxy: "
            rf"$\sigma_\rho/\hat{{\rho}}"
            rf"<{RHO_RELATIVE_UNCERTAINTY_CUT:.1f}$"
        ),
    )

    axes[
        0
    ].set_xscale(
        "log"
    )

    axes[
        0
    ].set_ylim(
        0.0,
        1.02,
    )

    axes[
        0
    ].set_xlabel(
        (
            r"Formal precision threshold "
            r"$\sigma_\rho/\hat{\rho}$"
        )
    )

    axes[
        0
    ].set_ylabel(
        "Fraction"
    )

    axes[
        0
    ].set_title(
        "How restrictive is a finite-source precision cut?"
    )

    axes[
        0
    ].grid(
        alpha=0.20,
    )

    axes[
        0
    ].legend(
        frameon=True,
    )

    # Panel B:
    # actual truth recovery as a function of formal precision

    shown = (
        recovery_table[
            "n"
        ].to_numpy()
        >= MIN_BIN_COUNT
    )

    sub = recovery_table.loc[
        shown
    ]

    axes[
        1
    ].plot(
        sub[
            "relative_uncertainty_center"
        ],
        sub[
            "delta_log10_rho_median"
        ],
        marker="o",
        markersize=4.5,
    )

    axes[
        1
    ].fill_between(
        sub[
            "relative_uncertainty_center"
        ],
        sub[
            "delta_log10_rho_q16"
        ],
        sub[
            "delta_log10_rho_q84"
        ],
        alpha=0.20,
    )

    axes[
        1
    ].axhline(
        0.0,
        linestyle="--",
        color="black",
        linewidth=1.3,
    )

    axes[
        1
    ].axvline(
        RHO_RELATIVE_UNCERTAINTY_CUT,
        linestyle="--",
        color="black",
        linewidth=1.3,
    )

    axes[
        1
    ].set_xscale(
        "log"
    )

    axes[
        1
    ].set_xlabel(
        r"Formal $\sigma_\rho/\hat{\rho}$"
    )

    axes[
        1
    ].set_ylabel(
        r"$\log_{10}(\hat{\rho}/\rho_{\rm true})$"
    )

    axes[
        1
    ].set_title(
        "Does formal precision imply accurate recovery?"
    )

    axes[
        1
    ].grid(
        alpha=0.20,
    )

    fig.suptitle(
        (
            "Finite-source information among "
            "annual-parallax detections"
        ),
        fontsize=20.2,
    )

    fig.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.95,
        ]
    )

    save_figure(
        fig,
        FIGURE_INFORMATION,
    )

    plt.close(
        fig
    )


# ============================================================================
# Joint log-rho / parallax propagation
# ============================================================================

def propagate_selected_mass(
    events,
):

    selected = (
        events[
            "rho_information_proxy"
        ].to_numpy(dtype=bool)
    )

    work = events.loc[
        selected
    ].copy()

    print()
    print(
        "Proxy-selected events =",
        f"{len(work):,}",
    )

    if len(
        work
    ) == 0:

        raise RuntimeError(
            "No events pass the rho information proxy."
        )

    rho_true = numeric(
        work,
        "rho_true",
    )

    piE_true = numeric(
        work,
        "piE_true_amp",
    )

    rho_hat = numeric(
        work,
        "h1_rho",
    )

    piEN_hat = numeric(
        work,
        "h1_piEN",
    )

    piEE_hat = numeric(
        work,
        "h1_piEE",
    )

    C_rr = numeric(
        work,
        "h1_cov_rho_rho",
    )

    C_rN = numeric(
        work,
        "h1_cov_rho_piEN",
    )

    C_rE = numeric(
        work,
        "h1_cov_rho_piEE",
    )

    C_NN = numeric(
        work,
        "h1_cov_piEN_piEN",
    )

    C_NE = numeric(
        work,
        "h1_cov_piEN_piEE",
    )

    C_EE = numeric(
        work,
        "h1_cov_piEE_piEE",
    )

    n_events = len(
        work
    )

    q16 = np.full(
        n_events,
        np.nan,
    )

    q50 = np.full(
        n_events,
        np.nan,
    )

    q84 = np.full(
        n_events,
        np.nan,
    )

    covariance_valid = np.zeros(
        n_events,
        dtype=bool,
    )

    rng = np.random.default_rng(
        RNG_SEED
    )

    # ----------------------------------------------------------------------
    # Chunked sampling in (ln rho, piEN, piEE)
    # ----------------------------------------------------------------------

    for start in range(
        0,
        n_events,
        BATCH_SIZE,
    ):

        stop = min(
            start
            + BATCH_SIZE,
            n_events,
        )

        sl = slice(
            start,
            stop,
        )

        r = rho_hat[
            sl
        ]

        m = len(
            r
        )

        finite_base = (
            np.isfinite(r)
            & (
                r > 0.0
            )
            & np.isfinite(
                rho_true[
                    sl
                ]
            )
            & (
                rho_true[
                    sl
                ] > 0.0
            )
            & np.isfinite(
                piE_true[
                    sl
                ]
            )
            & (
                piE_true[
                    sl
                ] > 0.0
            )
        )

        C = np.full(
            (
                m,
                3,
                3,
            ),
            np.nan,
        )

        # --------------------------------------------------------------
        # NEW BLOCK:
        # transform covariance from rho to ln(rho)
        # --------------------------------------------------------------

        C[
            :,
            0,
            0,
        ] = (
            C_rr[
                sl
            ]
            / r**2
        )

        C[
            :,
            0,
            1,
        ] = (
            C_rN[
                sl
            ]
            / r
        )

        C[
            :,
            1,
            0,
        ] = C[
            :,
            0,
            1,
        ]

        C[
            :,
            0,
            2,
        ] = (
            C_rE[
                sl
            ]
            / r
        )

        C[
            :,
            2,
            0,
        ] = C[
            :,
            0,
            2,
        ]

        C[
            :,
            1,
            1,
        ] = C_NN[
            sl
        ]

        C[
            :,
            1,
            2,
        ] = C_NE[
            sl
        ]

        C[
            :,
            2,
            1,
        ] = C_NE[
            sl
        ]

        C[
            :,
            2,
            2,
        ] = C_EE[
            sl
        ]

        finite_cov = (
            finite_base
            & np.isfinite(
                C
            ).all(
                axis=(
                    1,
                    2,
                )
            )
        )

        local_valid = np.zeros(
            m,
            dtype=bool,
        )

        if np.any(
            finite_cov
        ):

            candidate_indices = np.flatnonzero(
                finite_cov
            )

            C_candidates = C[
                candidate_indices
            ]

            eigenvalues = np.linalg.eigvalsh(
                C_candidates
            )

            pd_mask = (
                np.isfinite(
                    eigenvalues
                ).all(
                    axis=1
                )
                & (
                    np.min(
                        eigenvalues,
                        axis=1,
                    )
                    > 0.0
                )
            )

            valid_indices = candidate_indices[
                pd_mask
            ]

            local_valid[
                valid_indices
            ] = True

            if len(
                valid_indices
            ):

                C_valid = C[
                    valid_indices
                ]

                L = np.linalg.cholesky(
                    C_valid
                )

                mu = np.column_stack(
                    [
                        np.log(
                            r[
                                valid_indices
                            ]
                        ),

                        piEN_hat[
                            sl
                        ][
                            valid_indices
                        ],

                        piEE_hat[
                            sl
                        ][
                            valid_indices
                        ],
                    ]
                )

                z = rng.standard_normal(
                    size=(
                        len(
                            valid_indices
                        ),
                        N_MC,
                        3,
                    )
                )

                draws = (
                    mu[
                        :,
                        None,
                        :
                    ]
                    + np.einsum(
                        "nij,nkj->nki",
                        L,
                        z,
                    )
                )

                rho_draw = np.exp(
                    draws[
                        :,
                        :,
                        0,
                    ]
                )

                piE_draw = np.hypot(
                    draws[
                        :,
                        :,
                        1,
                    ],
                    draws[
                        :,
                        :,
                        2,
                    ],
                )

                numerator = (
                    rho_true[
                        sl
                    ][
                        valid_indices,
                        None,
                    ]
                    * piE_true[
                        sl
                    ][
                        valid_indices,
                        None,
                    ]
                )

                mass_ratio = (
                    numerator
                    / (
                        rho_draw
                        * piE_draw
                    )
                )

                mass_ratio[
                    ~np.isfinite(
                        mass_ratio
                    )
                    | (
                        mass_ratio
                        <= 0.0
                    )
                ] = np.nan

                with warnings.catch_warnings():

                    warnings.simplefilter(
                        "ignore",
                        RuntimeWarning,
                    )

                    quantiles = np.nanpercentile(
                        mass_ratio,
                        [
                            16,
                            50,
                            84,
                        ],
                        axis=1,
                    )

                q16[
                    start
                    + valid_indices
                ] = quantiles[
                    0
                ]

                q50[
                    start
                    + valid_indices
                ] = quantiles[
                    1
                ]

                q84[
                    start
                    + valid_indices
                ] = quantiles[
                    2
                ]

        covariance_valid[
            start:stop
        ] = local_valid

    work[
        "mass_mc_covariance_valid_logrho"
    ] = covariance_valid

    work[
        "mass_mc_q16_ratio"
    ] = q16

    work[
        "mass_mc_q50_ratio"
    ] = q50

    work[
        "mass_mc_q84_ratio"
    ] = q84

    interval_valid = (
        covariance_valid
        & np.isfinite(
            q16
        )
        & np.isfinite(
            q84
        )
        & (
            q16 > 0.0
        )
        & (
            q84 >= q16
        )
    )

    covered = (
        interval_valid
        & (
            q16 <= 1.0
        )
        & (
            q84 >= 1.0
        )
    )

    work[
        "mass_mc_interval_valid"
    ] = interval_valid

    work[
        "mass_mc_covered"
    ] = covered

    return work


# ============================================================================
# Figure 2:
# mass recovery and coverage for selected sample
# ============================================================================

def make_mass_figure(
    selected,
):

    rfs = selected[
        "rfs_true"
    ].to_numpy(dtype=float)

    point_bias = selected[
        "delta_log10_mass_point"
    ].to_numpy(dtype=float)

    interval_valid = selected[
        "mass_mc_interval_valid"
    ].to_numpy(dtype=bool)

    covered = selected[
        "mass_mc_covered"
    ].to_numpy(dtype=bool)

    valid_x = (
        np.isfinite(
            rfs
        )
        & (
            rfs > 0.0
        )
    )

    edges = build_log_edges(
        rfs[
            valid_x
        ],
        N_BINS,
    )

    rows = []

    for i in range(
        len(edges)
        - 1
    ):

        lo = edges[
            i
        ]

        hi = edges[
            i + 1
        ]

        in_bin = (
            valid_x
            & (
                rfs >= lo
            )
            & (
                rfs < hi
            )
        )

        recovery_mask = (
            in_bin
            & np.isfinite(
                point_bias
            )
        )

        q16, median, q84 = quantiles_or_nan(
            point_bias[
                recovery_mask
            ]
        )

        coverage_mask = (
            in_bin
            & interval_valid
        )

        n_coverage = int(
            coverage_mask.sum()
        )

        n_covered = int(
            covered[
                coverage_mask
            ].sum()
        )

        if n_coverage:

            coverage = (
                n_covered
                / n_coverage
            )

            cov_low, cov_high = wilson_interval(
                n_covered,
                n_coverage,
            )

        else:

            coverage = np.nan
            cov_low = np.nan
            cov_high = np.nan

        rows.append(
            {
                "rfs_left":
                    lo,

                "rfs_right":
                    hi,

                "rfs_center":
                    np.sqrt(
                        lo
                        * hi
                    ),

                "n_recovery":
                    int(
                        recovery_mask.sum()
                    ),

                "mass_bias_q16":
                    q16,

                "mass_bias_median":
                    median,

                "mass_bias_q84":
                    q84,

                "n_coverage":
                    n_coverage,

                "n_covered":
                    n_covered,

                "coverage":
                    coverage,

                "coverage_low":
                    cov_low,

                "coverage_high":
                    cov_high,
            }
        )

    table = pd.DataFrame(
        rows
    )

    table.to_csv(
        MASS_RECOVERY_TABLE,
        index=False,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            8.0,
            8.0,
        ),
        sharex=True,
    )

    shown_recovery = (
        table[
            "n_recovery"
        ].to_numpy()
        >= MIN_BIN_COUNT
    )

    rec = table.loc[
        shown_recovery
    ]

    axes[
        0
    ].plot(
        rec[
            "rfs_center"
        ],
        rec[
            "mass_bias_median"
        ],
        marker="o",
        markersize=4.5,
    )

    axes[
        0
    ].fill_between(
        rec[
            "rfs_center"
        ],
        rec[
            "mass_bias_q16"
        ],
        rec[
            "mass_bias_q84"
        ],
        alpha=0.20,
    )

    axes[
        0
    ].axhline(
        0.0,
        linestyle="--",
        color="black",
        linewidth=1.3,
    )

    axes[
        0
    ].set_ylabel(
        r"$\log_{10}(\hat M_L/M_{L,\rm true})$"
    )

    axes[
        0
    ].set_title(
        (
            "Mass recovery after parallax detection "
            "and finite-source precision selection"
        )
    )

    axes[
        0
    ].grid(
        alpha=0.20,
    )

    shown_coverage = (
        table[
            "n_coverage"
        ].to_numpy()
        >= MIN_BIN_COUNT
    )

    cov = table.loc[
        shown_coverage
    ]

    axes[
        1
    ].plot(
        cov[
            "rfs_center"
        ],
        cov[
            "coverage"
        ],
        marker="o",
        markersize=4.5,
    )

    axes[
        1
    ].fill_between(
        cov[
            "rfs_center"
        ],
        cov[
            "coverage_low"
        ],
        cov[
            "coverage_high"
        ],
        alpha=0.20,
    )

    axes[
        1
    ].axhline(
        NOMINAL_COVERAGE,
        linestyle="--",
        color="black",
        linewidth=1.3,
        label=r"Nominal $68.27\%$",
    )

    axes[
        1
    ].axvline(
        1.0,
        linestyle=":",
        color="0.4",
        linewidth=1.3,
        label=r"$R_{\rm FS}=1$",
    )

    axes[
        1
    ].set_xscale(
        "log"
    )

    axes[
        1
    ].set_ylim(
        0.0,
        1.02,
    )

    axes[
        1
    ].set_xlabel(
        (
            r"True finite-source strength "
            r"$R_{\rm FS,true}="
            r"\rho_{\rm true}/|u_{0,\rm true}|$"
        )
    )

    axes[
        1
    ].set_ylabel(
        "Central 68% mass-interval coverage"
    )

    axes[
        1
    ].set_title(
        r"Mass uncertainty propagated in $\log\rho$"
    )

    axes[
        1
    ].legend(
        frameon=True,
    )

    axes[
        1
    ].grid(
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        FIGURE_MASS,
    )

    plt.close(
        fig
    )


# ============================================================================
# Summary
# ============================================================================

def write_summary(
    events,
    selected,
):

    proxy = events[
        "rho_information_proxy"
    ].to_numpy(dtype=bool)

    accurate = events[
        "rho_accurate_truth"
    ].to_numpy(dtype=bool)

    rfs = events[
        "rfs_true"
    ].to_numpy(dtype=float)

    selected_interval = selected[
        "mass_mc_interval_valid"
    ].to_numpy(dtype=bool)

    selected_covered = selected[
        "mass_mc_covered"
    ].to_numpy(dtype=bool)

    lines = [
        "MASS MEASURABILITY AFTER PARALLAX DETECTION",
        "=" * 80,
        "",
        (
            "This analysis is conditional on "
            "positive_lrt_class == parallax_detected."
        ),
        "",
        f"N parallax detected = {len(events):,}",
        "",
        (
            "Operational rho-information proxy:"
        ),
        (
            f"  sigma_rho/rho_hat < "
            f"{RHO_RELATIVE_UNCERTAINTY_CUT:.2f}"
        ),
        (
            f"N passing proxy = "
            f"{int(proxy.sum()):,}"
        ),
        (
            f"fraction of parallax detections = "
            f"{np.mean(proxy):.6f}"
        ),
        "",
        (
            "Truth-side audit only:"
        ),
        (
            f"  |log10(rho_hat/rho_true)| < "
            f"{RHO_ACCURACY_DEX:.2f} dex"
        ),
    ]

    if np.any(
        proxy
    ):

        lines.extend(
            [
                (
                    "fraction of proxy-selected events "
                    "with accurate rho = "
                    f"{np.mean(accurate[proxy]):.6f}"
                ),
                (
                    "median true R_FS in proxy sample = "
                    f"{np.nanmedian(rfs[proxy]):.6g}"
                ),
                (
                    "fraction proxy sample with true R_FS > 1 = "
                    f"{np.mean(rfs[proxy] > 1.0):.6f}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            (
                "IMPORTANT: sigma_rho/rho_hat is a local-covariance "
                "precision proxy, not a finite-source detection."
            ),
            (
                "If the truth audit shows poor rho recovery even below "
                "the adopted threshold, this proxy must not be used to "
                "claim measurable lens masses."
            ),
            "",
            (
                "Joint mass propagation:"
            ),
            (
                "  Gaussian approximation is made in "
                "(ln rho, piEN, piEE), not in rho."
            ),
            (
                f"N selected with valid propagated mass interval = "
                f"{int(selected_interval.sum()):,}"
            ),
        ]
    )

    if np.any(
        selected_interval
    ):

        lines.append(
            (
                "global central-68% mass coverage = "
                f"{np.mean(selected_covered[selected_interval]):.6f}"
            )
        )

    SUMMARY_FILE.write_text(
        "\n".join(
            lines
        )
        + "\n"
    )

    print()
    print(
        "\n".join(
            lines
        )
    )

    print()
    print(
        "Saved:",
        SUMMARY_FILE,
    )


# ============================================================================
# Main
# ============================================================================

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

    print(
        "=" * 100
    )

    print(
        "44: MASS MEASURABILITY AFTER PARALLAX DETECTION"
    )

    print(
        "=" * 100
    )

    events = load_parallax_detected()

    print()
    print(
        "Parallax-detected events:",
        f"{len(events):,}",
    )

    events = build_event_diagnostics(
        events
    )

    events.to_parquet(
        EVENT_TABLE,
        index=False,
    )

    print(
        "Saved:",
        EVENT_TABLE,
    )

    make_information_figure(
        events
    )

    selected = propagate_selected_mass(
        events
    )

    make_mass_figure(
        selected
    )

    write_summary(
        events,
        selected,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "DONE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
