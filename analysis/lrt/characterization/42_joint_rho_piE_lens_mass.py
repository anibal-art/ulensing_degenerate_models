#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Joint finite-source + parallax propagation into lens mass.

NO light-curve simulation.
NO fitting.

Assumption
----------
The angular source radius theta_* is known exactly.

Then

    theta_E = theta_* / rho

and

    M_L = theta_E / (kappa * pi_E)
        = theta_* / (kappa * rho * pi_E).

Therefore the event-wise mass ratio can be written without explicitly
using theta_* or kappa:

    Mhat_L / M_L,true
        =
    rho_true * piE_true
    ------------------
    rho_hat  * piE_hat

where

    piE_hat = sqrt(piEN_hat^2 + piEE_hat^2).

This is useful because all fixed physical factors cancel exactly.

For uncertainty propagation we use the full H1 covariance block

             rho       piEN       piEE
    rho      Crr       CrN        CrE
    piEN     CrN       CNN        CNE
    piEE     CrE       CNE        CEE

and draw

    (rho, piEN, piEE)
        ~ N(mu, C)

for every event.

Draws with rho <= 0 are discarded, exactly as in the previous
finite-source covariance diagnostic. This remains a local-Gaussian
propagation, not a full posterior.

Four figures
------------
1. Full mass recovery versus true |pi_E|.

2. Mass uncertainty versus true finite-source strength

       R_FS,true = rho_true / |u0_true|.

3. Mass coverage versus true |pi_E| for
       - confused events,
       - detected + rho constrained,
       - detected + rho unconstrained.

   rho constrained is defined by

       sigma_rho / rho_hat < 0.5.

4. Error decomposition versus R_FS,true:

       Delta log10 M
         =
       - Delta log10 rho
       - Delta log10 pi_E.

Input
-----
analysis/lrt/results/characterization/18_positive_truth/
    h1_positive_truth_characterization.parquet

Outputs
-------
analysis/lrt/figures/characterization/
    42_joint_rho_piE_lens_mass/

analysis/lrt/results/characterization/
    42_joint_rho_piE_lens_mass/
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
    / "42_joint_rho_piE_lens_mass"
)

RESULT_DIR = (
    ROOT
    / "analysis/lrt/results/characterization"
    / "42_joint_rho_piE_lens_mass"
)


FIGURE_RECOVERY = (
    FIGURE_DIR
    / "01_joint_mass_recovery_vs_true_piE"
)

FIGURE_UNCERTAINTY = (
    FIGURE_DIR
    / "02_joint_mass_uncertainty_vs_true_RFS"
)

FIGURE_COVERAGE = (
    FIGURE_DIR
    / "03_joint_mass_coverage_vs_true_piE"
)

FIGURE_DECOMPOSITION = (
    FIGURE_DIR
    / "04_joint_mass_error_decomposition_vs_true_RFS"
)


EVENT_TABLE = (
    RESULT_DIR
    / "joint_rho_piE_lens_mass_events.parquet"
)

RECOVERY_TABLE = (
    RESULT_DIR
    / "01_joint_mass_recovery_vs_true_piE_bins.csv"
)

UNCERTAINTY_TABLE = (
    RESULT_DIR
    / "02_joint_mass_uncertainty_vs_true_RFS_bins.csv"
)

COVERAGE_TABLE = (
    RESULT_DIR
    / "03_joint_mass_coverage_vs_true_piE_bins.csv"
)

DECOMPOSITION_TABLE = (
    RESULT_DIR
    / "04_joint_mass_error_decomposition_vs_true_RFS_bins.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "joint_rho_piE_lens_mass_summary.txt"
)


# ============================================================================
# Configuration
# ============================================================================

N_MC = 512

CHUNK_SIZE = 1000

RNG_SEED = 23092026

N_BINS = 20

MIN_BIN_COUNT = 30

RHO_RELATIVE_UNCERTAINTY_THRESHOLD = 0.5

NOMINAL_COVERAGE = 0.6826894921370859

# Require at least this many positive-rho draws before an event-level
# propagated interval is considered usable.
MIN_PHYSICAL_DRAWS = 32


CLASSES = [
    "confused_fspl_no_parallax",
    "parallax_detected",
]

CLASS_LABELS = {
    "confused_fspl_no_parallax":
        "Confused with no-parallax FSPL",

    "parallax_detected":
        "Parallax detected",
}


COVERAGE_GROUPS = [
    "confused",
    "detected_rho_constrained",
    "detected_rho_unconstrained",
]

COVERAGE_LABELS = {
    "confused":
        "Confused",

    "detected_rho_constrained":
        r"Detected, $\sigma_\rho/\hat{\rho}<0.5$",

    "detected_rho_unconstrained":
        r"Detected, $\sigma_\rho/\hat{\rho}\geq0.5$",
}


# ============================================================================
# Plot style
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

def numeric(df, column):

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def require_columns(
    df,
    columns,
):

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(
                f"  {column}"
                for column in missing
            )
            + "\n\nAvailable columns:\n"
            + str(
                list(df.columns)
            )
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

    if not np.any(valid):

        raise RuntimeError(
            "No finite positive values for logarithmic binning."
        )

    vmin = float(
        np.nanmin(
            values[valid]
        )
    )

    vmax = float(
        np.nanmax(
            values[valid]
        )
    )

    edges = np.geomspace(
        vmin,
        vmax,
        n_bins + 1,
    )

    edges[0] = np.nextafter(
        vmin,
        -np.inf,
    )

    edges[-1] = np.nextafter(
        vmax,
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
        np.nanpercentile(
            values,
            [
                16,
                50,
                84,
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
        + z**2 / n
    )

    center = (
        p
        + z**2 / (
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


def save_figure(
    fig,
    base_path,
):

    png = base_path.with_suffix(
        ".png"
    )

    pdf = base_path.with_suffix(
        ".pdf"
    )

    fig.savefig(
        png,
    )

    fig.savefig(
        pdf,
    )

    print(
        "Saved:",
        png,
    )

    print(
        "Saved:",
        pdf,
    )


# ============================================================================
# Vectorized 3x3 Cholesky
# ============================================================================

def cholesky_3x3(
    C_rr,
    C_rN,
    C_rE,
    C_NN,
    C_NE,
    C_EE,
):
    """
    Vectorized Cholesky factorization for

        C =
        [[C_rr, C_rN, C_rE],
         [C_rN, C_NN, C_NE],
         [C_rE, C_NE, C_EE]]

    returning the six non-zero elements of lower-triangular L.

    Invalid / non-positive-definite matrices remain NaN.
    """

    n = len(
        C_rr
    )

    L11 = np.full(
        n,
        np.nan,
    )

    L21 = np.full(
        n,
        np.nan,
    )

    L31 = np.full(
        n,
        np.nan,
    )

    L22 = np.full(
        n,
        np.nan,
    )

    L32 = np.full(
        n,
        np.nan,
    )

    L33 = np.full(
        n,
        np.nan,
    )

    finite = (
        np.isfinite(C_rr)
        & np.isfinite(C_rN)
        & np.isfinite(C_rE)
        & np.isfinite(C_NN)
        & np.isfinite(C_NE)
        & np.isfinite(C_EE)
        & (
            C_rr > 0.0
        )
        & (
            C_NN > 0.0
        )
        & (
            C_EE > 0.0
        )
    )

    L11[
        finite
    ] = np.sqrt(
        C_rr[
            finite
        ]
    )

    L21[
        finite
    ] = (
        C_rN[
            finite
        ]
        / L11[
            finite
        ]
    )

    L31[
        finite
    ] = (
        C_rE[
            finite
        ]
        / L11[
            finite
        ]
    )

    L22_squared = np.full(
        n,
        np.nan,
    )

    L22_squared[
        finite
    ] = (
        C_NN[
            finite
        ]
        - L21[
            finite
        ] ** 2
    )

    valid_second = (
        finite
        & np.isfinite(
            L22_squared
        )
        & (
            L22_squared > 0.0
        )
    )

    L22[
        valid_second
    ] = np.sqrt(
        L22_squared[
            valid_second
        ]
    )

    L32[
        valid_second
    ] = (
        C_NE[
            valid_second
        ]
        - (
            L31[
                valid_second
            ]
            * L21[
                valid_second
            ]
        )
    ) / L22[
        valid_second
    ]

    L33_squared = np.full(
        n,
        np.nan,
    )

    L33_squared[
        valid_second
    ] = (
        C_EE[
            valid_second
        ]
        - L31[
            valid_second
        ] ** 2
        - L32[
            valid_second
        ] ** 2
    )

    valid = (
        valid_second
        & np.isfinite(
            L33_squared
        )
        & (
            L33_squared > 0.0
        )
    )

    L33[
        valid
    ] = np.sqrt(
        L33_squared[
            valid
        ]
    )

    return (
        valid,
        L11,
        L21,
        L31,
        L22,
        L32,
        L33,
    )


# ============================================================================
# Build event-level quantities
# ============================================================================

def build_event_table(
    df,
):

    n_events = len(
        df
    )

    classes = (
        df[
            "positive_lrt_class"
        ]
        .astype(str)
        .to_numpy()
    )

    rho_true = numeric(
        df,
        "rho_true",
    )

    u0_true = numeric(
        df,
        "u0_true",
    )

    piE_true = numeric(
        df,
        "piE_true_amp",
    )

    rho_fit = numeric(
        df,
        "h1_rho",
    )

    piEN_fit = numeric(
        df,
        "h1_piEN",
    )

    piEE_fit = numeric(
        df,
        "h1_piEE",
    )

    C_rr = numeric(
        df,
        "h1_cov_rho_rho",
    )

    C_rN = numeric(
        df,
        "h1_cov_rho_piEN",
    )

    C_rE = numeric(
        df,
        "h1_cov_rho_piEE",
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

    piE_fit = np.hypot(
        piEN_fit,
        piEE_fit,
    )

    abs_u0_true = np.abs(
        u0_true
    )

    rfs_true = np.full(
        n_events,
        np.nan,
    )

    valid_rfs = (
        np.isfinite(rho_true)
        & (
            rho_true > 0.0
        )
        & np.isfinite(abs_u0_true)
        & (
            abs_u0_true > 0.0
        )
    )

    rfs_true[
        valid_rfs
    ] = (
        rho_true[
            valid_rfs
        ]
        / abs_u0_true[
            valid_rfs
        ]
    )

    # ========================================================================
    # Point estimate
    # ========================================================================

    point_valid = (
        np.isfinite(rho_true)
        & (
            rho_true > 0.0
        )
        & np.isfinite(rho_fit)
        & (
            rho_fit > 0.0
        )
        & np.isfinite(piE_true)
        & (
            piE_true > 0.0
        )
        & np.isfinite(piE_fit)
        & (
            piE_fit > 0.0
        )
    )

    delta_log10_rho = np.full(
        n_events,
        np.nan,
    )

    delta_log10_piE = np.full(
        n_events,
        np.nan,
    )

    rho_mass_term = np.full(
        n_events,
        np.nan,
    )

    piE_mass_term = np.full(
        n_events,
        np.nan,
    )

    delta_log10_mass = np.full(
        n_events,
        np.nan,
    )

    mass_ratio_point = np.full(
        n_events,
        np.nan,
    )

    delta_log10_rho[
        point_valid
    ] = np.log10(
        rho_fit[
            point_valid
        ]
        / rho_true[
            point_valid
        ]
    )

    delta_log10_piE[
        point_valid
    ] = np.log10(
        piE_fit[
            point_valid
        ]
        / piE_true[
            point_valid
        ]
    )

    rho_mass_term[
        point_valid
    ] = (
        -delta_log10_rho[
            point_valid
        ]
    )

    piE_mass_term[
        point_valid
    ] = (
        -delta_log10_piE[
            point_valid
        ]
    )

    delta_log10_mass[
        point_valid
    ] = (
        rho_mass_term[
            point_valid
        ]
        + piE_mass_term[
            point_valid
        ]
    )

    mass_ratio_point[
        point_valid
    ] = (
        10.0
        ** delta_log10_mass[
            point_valid
        ]
    )

    # ========================================================================
    # rho-information diagnostic
    # ========================================================================

    sigma_rho = np.full(
        n_events,
        np.nan,
    )

    valid_sigma_rho = (
        np.isfinite(
            C_rr
        )
        & (
            C_rr >= 0.0
        )
    )

    sigma_rho[
        valid_sigma_rho
    ] = np.sqrt(
        C_rr[
            valid_sigma_rho
        ]
    )

    rho_relative_uncertainty = np.full(
        n_events,
        np.nan,
    )

    valid_rho_relative = (
        np.isfinite(
            sigma_rho
        )
        & np.isfinite(
            rho_fit
        )
        & (
            rho_fit > 0.0
        )
    )

    rho_relative_uncertainty[
        valid_rho_relative
    ] = (
        sigma_rho[
            valid_rho_relative
        ]
        / rho_fit[
            valid_rho_relative
        ]
    )

    rho_constrained = (
        valid_rho_relative
        & (
            rho_relative_uncertainty
            < RHO_RELATIVE_UNCERTAINTY_THRESHOLD
        )
    )

    # ========================================================================
    # Full 3x3 covariance
    # ========================================================================

    (
        covariance_valid,
        L11,
        L21,
        L31,
        L22,
        L32,
        L33,
    ) = cholesky_3x3(
        C_rr,
        C_rN,
        C_rE,
        C_NN,
        C_NE,
        C_EE,
    )

    covariance_valid &= (
        point_valid
    )

    valid_indices = np.flatnonzero(
        covariance_valid
    )

    # ========================================================================
    # MC output arrays
    # ========================================================================

    mc_n_physical = np.zeros(
        n_events,
        dtype=np.int32,
    )

    mc_physical_draw_fraction = np.full(
        n_events,
        np.nan,
    )

    mc_q16_mass_ratio = np.full(
        n_events,
        np.nan,
    )

    mc_q50_mass_ratio = np.full(
        n_events,
        np.nan,
    )

    mc_q84_mass_ratio = np.full(
        n_events,
        np.nan,
    )

    mc_sigma_log10_mass = np.full(
        n_events,
        np.nan,
    )

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

    print()
    print(
        "Events =",
        n_events,
    )

    print(
        "Valid 3x3 covariance =",
        n_valid,
    )

    print(
        "MC draws/event =",
        N_MC,
    )

    print(
        "MC chunks =",
        n_chunks,
    )

    # ========================================================================
    # Joint MC propagation
    # ========================================================================

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

        z = rng.standard_normal(
            size=(
                m,
                N_MC,
                3,
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

        z3 = z[
            :,
            :,
            2,
        ]

        sample_rho = (
            rho_fit[
                idx,
                None,
            ]
            + L11[
                idx,
                None,
            ]
            * z1
        )

        sample_piEN = (
            piEN_fit[
                idx,
                None,
            ]
            + L21[
                idx,
                None,
            ]
            * z1
            + L22[
                idx,
                None,
            ]
            * z2
        )

        sample_piEE = (
            piEE_fit[
                idx,
                None,
            ]
            + L31[
                idx,
                None,
            ]
            * z1
            + L32[
                idx,
                None,
            ]
            * z2
            + L33[
                idx,
                None,
            ]
            * z3
        )

        sample_piE = np.hypot(
            sample_piEN,
            sample_piEE,
        )

        physical = (
            np.isfinite(
                sample_rho
            )
            & (
                sample_rho > 0.0
            )
            & np.isfinite(
                sample_piE
            )
            & (
                sample_piE > 0.0
            )
        )

        n_physical = np.sum(
            physical,
            axis=1,
        )

        mc_n_physical[
            idx
        ] = n_physical

        mc_physical_draw_fraction[
            idx
        ] = (
            n_physical
            / N_MC
        )

        numerator = (
            rho_true[
                idx,
                None,
            ]
            * piE_true[
                idx,
                None,
            ]
        )

        mass_ratio_draws = (
            numerator
            / (
                sample_rho
                * sample_piE
            )
        )

        mass_ratio_draws[
            ~physical
        ] = np.nan

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore",
                category=RuntimeWarning,
            )

            q16 = np.nanpercentile(
                mass_ratio_draws,
                16,
                axis=1,
            )

            q50 = np.nanpercentile(
                mass_ratio_draws,
                50,
                axis=1,
            )

            q84 = np.nanpercentile(
                mass_ratio_draws,
                84,
                axis=1,
            )

        enough = (
            n_physical
            >= MIN_PHYSICAL_DRAWS
        )

        current = idx[
            enough
        ]

        mc_q16_mass_ratio[
            current
        ] = q16[
            enough
        ]

        mc_q50_mass_ratio[
            current
        ] = q50[
            enough
        ]

        mc_q84_mass_ratio[
            current
        ] = q84[
            enough
        ]

        valid_width = (
            enough
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
                q84 > 0.0
            )
            & (
                q84 >= q16
            )
        )

        current_width = idx[
            valid_width
        ]

        mc_sigma_log10_mass[
            current_width
        ] = (
            0.5
            * (
                np.log10(
                    q84[
                        valid_width
                    ]
                )
                - np.log10(
                    q16[
                        valid_width
                    ]
                )
            )
        )

        if (
            ichunk == 1
            or ichunk == n_chunks
            or ichunk % 25 == 0
        ):

            print(
                f"MC chunk {ichunk}/{n_chunks}"
            )

    # ========================================================================
    # MC interval coverage
    # ========================================================================

    mc_interval_valid = (
        covariance_valid
        & (
            mc_n_physical
            >= MIN_PHYSICAL_DRAWS
        )
        & np.isfinite(
            mc_q16_mass_ratio
        )
        & np.isfinite(
            mc_q84_mass_ratio
        )
        & (
            mc_q16_mass_ratio > 0.0
        )
        & (
            mc_q84_mass_ratio
            >= mc_q16_mass_ratio
        )
    )

    # Truth is M/M_true = 1.
    mc_mass_covered = (
        mc_interval_valid
        & (
            mc_q16_mass_ratio
            <= 1.0
        )
        & (
            mc_q84_mass_ratio
            >= 1.0
        )
    )

    # ========================================================================
    # Output event table
    # ========================================================================

    out = pd.DataFrame(
        {
            "catalog_row":
                df[
                    "catalog_row"
                ].to_numpy(),

            "positive_lrt_class":
                classes,

            "delta_chi2_lrt":
                numeric(
                    df,
                    "delta_chi2_lrt",
                ),

            "rho_true":
                rho_true,

            "u0_true":
                u0_true,

            "rfs_true":
                rfs_true,

            "piE_true_amp":
                piE_true,

            "h1_rho":
                rho_fit,

            "h1_piEN":
                piEN_fit,

            "h1_piEE":
                piEE_fit,

            "h1_piE_amp":
                piE_fit,

            "sigma_rho":
                sigma_rho,

            "sigma_rho_over_rho_fit":
                rho_relative_uncertainty,

            "rho_constrained":
                rho_constrained,

            "point_valid":
                point_valid,

            "delta_log10_rho":
                delta_log10_rho,

            "delta_log10_piE":
                delta_log10_piE,

            "rho_mass_term":
                rho_mass_term,

            "piE_mass_term":
                piE_mass_term,

            "delta_log10_mass":
                delta_log10_mass,

            "mass_ratio_point":
                mass_ratio_point,

            "mc_covariance_valid":
                covariance_valid,

            "mc_n_physical_draws":
                mc_n_physical,

            "mc_physical_draw_fraction":
                mc_physical_draw_fraction,

            "mc_q16_mass_ratio":
                mc_q16_mass_ratio,

            "mc_q50_mass_ratio":
                mc_q50_mass_ratio,

            "mc_q84_mass_ratio":
                mc_q84_mass_ratio,

            "mc_sigma_log10_mass":
                mc_sigma_log10_mass,

            "mc_interval_valid":
                mc_interval_valid,

            "mc_mass_covered":
                mc_mass_covered,
        }
    )

    return out


# ============================================================================
# Figure 1: full mass recovery versus true piE
# ============================================================================

def make_recovery_figure(
    events,
):

    x = events[
        "piE_true_amp"
    ].to_numpy(dtype=float)

    y = events[
        "delta_log10_mass"
    ].to_numpy(dtype=float)

    classes = (
        events[
            "positive_lrt_class"
        ]
        .astype(str)
        .to_numpy()
    )

    valid = (
        events[
            "point_valid"
        ].to_numpy(dtype=bool)
        & np.isfinite(x)
        & (
            x > 0.0
        )
        & np.isfinite(y)
    )

    edges = build_log_edges(
        x[
            valid
        ],
        N_BINS,
    )

    rows = []

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.6,
        )
    )

    for cls in CLASSES:

        for ibin in range(
            len(edges)
            - 1
        ):

            left = edges[
                ibin
            ]

            right = edges[
                ibin
                + 1
            ]

            mask = (
                valid
                & (
                    classes
                    == cls
                )
                & (
                    x >= left
                )
                & (
                    x < right
                )
            )

            q16, med, q84 = (
                quantiles_or_nan(
                    y[
                        mask
                    ]
                )
            )

            rows.append(
                {
                    "class":
                        cls,

                    "bin_index":
                        ibin,

                    "x_left":
                        left,

                    "x_right":
                        right,

                    "x_center":
                        np.sqrt(
                            left
                            * right
                        ),

                    "n":
                        int(
                            mask.sum()
                        ),

                    "q16":
                        q16,

                    "median":
                        med,

                    "q84":
                        q84,
                }
            )

    table = pd.DataFrame(
        rows
    )

    for cls in CLASSES:

        sub = table[
            (
                table[
                    "class"
                ]
                == cls
            )
            & (
                table[
                    "n"
                ]
                >= MIN_BIN_COUNT
            )
        ]

        line, = ax.plot(
            sub[
                "x_center"
            ],
            sub[
                "median"
            ],
            marker="o",
            markersize=4.5,
            label=CLASS_LABELS[
                cls
            ],
        )

        ax.fill_between(
            sub[
                "x_center"
            ],
            sub[
                "q16"
            ],
            sub[
                "q84"
            ],
            color=line.get_color(),
            alpha=0.20,
        )

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.4,
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        r"True $|\boldsymbol{\pi}_E|$"
    )

    ax.set_ylabel(
        r"$\log_{10}(\widehat{M}_L/M_{L,\rm true})$"
    )

    ax.set_title(
        "Lens-mass recovery with parallax and finite-source uncertainty"
    )

    ax.grid(
        alpha=0.20,
    )

    ax.legend(
        frameon=True,
    )

    fig.tight_layout()

    save_figure(
        fig,
        FIGURE_RECOVERY,
    )

    plt.close(
        fig
    )

    table.to_csv(
        RECOVERY_TABLE,
        index=False,
    )


# ============================================================================
# Figure 2: mass uncertainty versus finite-source strength
# ============================================================================

def make_uncertainty_figure(
    events,
):

    x = events[
        "rfs_true"
    ].to_numpy(dtype=float)

    y = events[
        "mc_sigma_log10_mass"
    ].to_numpy(dtype=float)

    classes = (
        events[
            "positive_lrt_class"
        ]
        .astype(str)
        .to_numpy()
    )

    valid = (
        events[
            "mc_interval_valid"
        ].to_numpy(dtype=bool)
        & np.isfinite(x)
        & (
            x > 0.0
        )
        & np.isfinite(y)
        & (
            y >= 0.0
        )
    )

    edges = build_log_edges(
        x[
            valid
        ],
        N_BINS,
    )

    rows = []

    for cls in CLASSES:

        for ibin in range(
            len(edges)
            - 1
        ):

            left = edges[
                ibin
            ]

            right = edges[
                ibin
                + 1
            ]

            mask = (
                valid
                & (
                    classes
                    == cls
                )
                & (
                    x >= left
                )
                & (
                    x < right
                )
            )

            q16, med, q84 = (
                quantiles_or_nan(
                    y[
                        mask
                    ]
                )
            )

            rows.append(
                {
                    "class":
                        cls,

                    "bin_index":
                        ibin,

                    "x_left":
                        left,

                    "x_right":
                        right,

                    "x_center":
                        np.sqrt(
                            left
                            * right
                        ),

                    "n":
                        int(
                            mask.sum()
                        ),

                    "q16":
                        q16,

                    "median":
                        med,

                    "q84":
                        q84,
                }
            )

    table = pd.DataFrame(
        rows
    )

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.6,
        )
    )

    for cls in CLASSES:

        sub = table[
            (
                table[
                    "class"
                ]
                == cls
            )
            & (
                table[
                    "n"
                ]
                >= MIN_BIN_COUNT
            )
        ]

        line, = ax.plot(
            sub[
                "x_center"
            ],
            sub[
                "median"
            ],
            marker="o",
            markersize=4.5,
            label=CLASS_LABELS[
                cls
            ],
        )

        ax.fill_between(
            sub[
                "x_center"
            ],
            sub[
                "q16"
            ],
            sub[
                "q84"
            ],
            color=line.get_color(),
            alpha=0.20,
        )

    ax.axvline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=r"$R_{\rm FS}=1$",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        (
            r"True finite-source strength "
            r"$R_{\rm FS}=\rho_{\rm true}/|u_{0,\rm true}|$"
        )
    )

    ax.set_ylabel(
        (
            r"Central-$68\%$ mass half-width "
            r"$\sigma_{\log_{10} M}$ [dex]"
        )
    )

    ax.set_title(
        "Lens-mass uncertainty versus finite-source strength"
    )

    ax.grid(
        alpha=0.20,
    )

    ax.legend(
        frameon=True,
    )

    fig.tight_layout()

    save_figure(
        fig,
        FIGURE_UNCERTAINTY,
    )

    plt.close(
        fig
    )

    table.to_csv(
        UNCERTAINTY_TABLE,
        index=False,
    )


# ============================================================================
# Figure 3: coverage versus true piE
# ============================================================================

def make_coverage_figure(
    events,
):

    x = events[
        "piE_true_amp"
    ].to_numpy(dtype=float)

    classes = (
        events[
            "positive_lrt_class"
        ]
        .astype(str)
        .to_numpy()
    )

    rho_constrained = (
        events[
            "rho_constrained"
        ].to_numpy(dtype=bool)
    )

    valid = (
        events[
            "mc_interval_valid"
        ].to_numpy(dtype=bool)
        & np.isfinite(x)
        & (
            x > 0.0
        )
    )

    covered = (
        events[
            "mc_mass_covered"
        ].to_numpy(dtype=bool)
    )

    group = np.full(
        len(events),
        "",
        dtype=object,
    )

    confused = (
        classes
        == "confused_fspl_no_parallax"
    )

    detected = (
        classes
        == "parallax_detected"
    )

    group[
        confused
    ] = "confused"

    group[
        detected
        & rho_constrained
    ] = (
        "detected_rho_constrained"
    )

    group[
        detected
        & ~rho_constrained
    ] = (
        "detected_rho_unconstrained"
    )

    edges = build_log_edges(
        x[
            valid
        ],
        N_BINS,
    )

    rows = []

    for group_name in COVERAGE_GROUPS:

        for ibin in range(
            len(edges)
            - 1
        ):

            left = edges[
                ibin
            ]

            right = edges[
                ibin
                + 1
            ]

            mask = (
                valid
                & (
                    group
                    == group_name
                )
                & (
                    x >= left
                )
                & (
                    x < right
                )
            )

            n = int(
                mask.sum()
            )

            k = int(
                covered[
                    mask
                ].sum()
            )

            coverage = (
                k / n
                if n > 0
                else np.nan
            )

            lo, hi = wilson_interval(
                k,
                n,
            )

            rows.append(
                {
                    "group":
                        group_name,

                    "bin_index":
                        ibin,

                    "x_left":
                        left,

                    "x_right":
                        right,

                    "x_center":
                        np.sqrt(
                            left
                            * right
                        ),

                    "n":
                        n,

                    "n_covered":
                        k,

                    "coverage":
                        coverage,

                    "wilson_low":
                        lo,

                    "wilson_high":
                        hi,
                }
            )

    table = pd.DataFrame(
        rows
    )

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.6,
        )
    )

    for group_name in COVERAGE_GROUPS:

        sub = table[
            (
                table[
                    "group"
                ]
                == group_name
            )
            & (
                table[
                    "n"
                ]
                >= MIN_BIN_COUNT
            )
        ]

        line, = ax.plot(
            sub[
                "x_center"
            ],
            sub[
                "coverage"
            ],
            marker="o",
            markersize=4.5,
            label=COVERAGE_LABELS[
                group_name
            ],
        )

        ax.fill_between(
            sub[
                "x_center"
            ],
            sub[
                "wilson_low"
            ],
            sub[
                "wilson_high"
            ],
            color=line.get_color(),
            alpha=0.18,
        )

    ax.axhline(
        NOMINAL_COVERAGE,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=r"Nominal $68.27\%$",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_xlabel(
        r"True $|\boldsymbol{\pi}_E|$"
    )

    ax.set_ylabel(
        "Lens-mass interval coverage"
    )

    ax.set_title(
        (
            r"Coverage of the joint $\rho$--"
            r"$\boldsymbol{\pi}_E$ lens-mass interval"
        )
    )

    ax.grid(
        alpha=0.20,
    )

    ax.legend(
        frameon=True,
    )

    fig.tight_layout()

    save_figure(
        fig,
        FIGURE_COVERAGE,
    )

    plt.close(
        fig
    )

    table.to_csv(
        COVERAGE_TABLE,
        index=False,
    )


# ============================================================================
# Figure 4: error decomposition
# ============================================================================

def make_decomposition_figure(
    events,
):

    x = events[
        "rfs_true"
    ].to_numpy(dtype=float)

    classes = (
        events[
            "positive_lrt_class"
        ]
        .astype(str)
        .to_numpy()
    )

    total = events[
        "delta_log10_mass"
    ].to_numpy(dtype=float)

    rho_term = events[
        "rho_mass_term"
    ].to_numpy(dtype=float)

    piE_term = events[
        "piE_mass_term"
    ].to_numpy(dtype=float)

    valid = (
        events[
            "point_valid"
        ].to_numpy(dtype=bool)
        & np.isfinite(x)
        & (
            x > 0.0
        )
        & np.isfinite(total)
        & np.isfinite(rho_term)
        & np.isfinite(piE_term)
    )

    edges = build_log_edges(
        x[
            valid
        ],
        N_BINS,
    )

    rows = []

    for cls in CLASSES:

        for ibin in range(
            len(edges)
            - 1
        ):

            left = edges[
                ibin
            ]

            right = edges[
                ibin
                + 1
            ]

            mask = (
                valid
                & (
                    classes
                    == cls
                )
                & (
                    x >= left
                )
                & (
                    x < right
                )
            )

            total_q16, total_med, total_q84 = (
                quantiles_or_nan(
                    total[
                        mask
                    ]
                )
            )

            _, rho_med, _ = (
                quantiles_or_nan(
                    rho_term[
                        mask
                    ]
                )
            )

            _, piE_med, _ = (
                quantiles_or_nan(
                    piE_term[
                        mask
                    ]
                )
            )

            rows.append(
                {
                    "class":
                        cls,

                    "bin_index":
                        ibin,

                    "x_left":
                        left,

                    "x_right":
                        right,

                    "x_center":
                        np.sqrt(
                            left
                            * right
                        ),

                    "n":
                        int(
                            mask.sum()
                        ),

                    "total_q16":
                        total_q16,

                    "total_median":
                        total_med,

                    "total_q84":
                        total_q84,

                    "rho_term_median":
                        rho_med,

                    "piE_term_median":
                        piE_med,
                }
            )

    table = pd.DataFrame(
        rows
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            8.0,
            8.2,
        ),
        sharex=True,
    )

    for ax, cls in zip(
        axes,
        CLASSES,
    ):

        sub = table[
            (
                table[
                    "class"
                ]
                == cls
            )
            & (
                table[
                    "n"
                ]
                >= MIN_BIN_COUNT
            )
        ]

        total_line, = ax.plot(
            sub[
                "x_center"
            ],
            sub[
                "total_median"
            ],
            marker="o",
            markersize=4.5,
            label=(
                r"Total "
                r"$\Delta\log_{10}M$"
            ),
        )

        ax.fill_between(
            sub[
                "x_center"
            ],
            sub[
                "total_q16"
            ],
            sub[
                "total_q84"
            ],
            color=total_line.get_color(),
            alpha=0.15,
        )

        ax.plot(
            sub[
                "x_center"
            ],
            sub[
                "rho_term_median"
            ],
            marker="s",
            markersize=4,
            label=(
                r"Finite source: "
                r"$-\Delta\log_{10}\rho$"
            ),
        )

        ax.plot(
            sub[
                "x_center"
            ],
            sub[
                "piE_term_median"
            ],
            marker="^",
            markersize=4,
            label=(
                r"Parallax: "
                r"$-\Delta\log_{10}|\boldsymbol{\pi}_E|$"
            ),
        )

        ax.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.2,
        )

        ax.axvline(
            1.0,
            color="0.4",
            linestyle=":",
            linewidth=1.2,
        )

        ax.set_xscale(
            "log"
        )

        ax.set_ylabel(
            "Bias [dex]"
        )

        ax.set_title(
            CLASS_LABELS[
                cls
            ]
        )

        ax.grid(
            alpha=0.20,
        )

    axes[
        -1
    ].set_xlabel(
        (
            r"True finite-source strength "
            r"$R_{\rm FS}=\rho_{\rm true}/|u_{0,\rm true}|$"
        )
    )

    axes[
        0
    ].legend(
        frameon=True,
        loc="best",
    )

    fig.suptitle(
        "Decomposition of lens-mass recovery error",
        fontsize=20.2,
    )

    fig.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.97,
        ]
    )

    save_figure(
        fig,
        FIGURE_DECOMPOSITION,
    )

    plt.close(
        fig
    )

    table.to_csv(
        DECOMPOSITION_TABLE,
        index=False,
    )


# ============================================================================
# Summary
# ============================================================================

def write_summary(
    events,
):

    classes = (
        events[
            "positive_lrt_class"
        ]
        .astype(str)
        .to_numpy()
    )

    rho_constrained = (
        events[
            "rho_constrained"
        ].to_numpy(dtype=bool)
    )

    interval_valid = (
        events[
            "mc_interval_valid"
        ].to_numpy(dtype=bool)
    )

    covered = (
        events[
            "mc_mass_covered"
        ].to_numpy(dtype=bool)
    )

    sigma_mass = events[
        "mc_sigma_log10_mass"
    ].to_numpy(dtype=float)

    delta_mass = events[
        "delta_log10_mass"
    ].to_numpy(dtype=float)

    lines = []

    lines.append(
        "JOINT rho + piE LENS-MASS PROPAGATION"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"events = {len(events):,}"
    )

    lines.append(
        (
            "MC covariance valid = "
            f"{int(events['mc_covariance_valid'].sum()):,}"
        )
    )

    lines.append(
        (
            "MC interval valid = "
            f"{int(interval_valid.sum()):,}"
        )
    )

    lines.append("")
    lines.append(
        "Assumption: theta_* known exactly."
    )

    lines.append(
        (
            "Mass ratio = "
            "(rho_true*piE_true)/(rho_fit*piE_fit)."
        )
    )

    for cls in CLASSES:

        class_mask = (
            classes
            == cls
        )

        lines.append("")
        lines.append(
            CLASS_LABELS[
                cls
            ]
        )

        lines.append(
            "-" * 60
        )

        lines.append(
            f"N = {int(class_mask.sum()):,}"
        )

        if np.any(
            class_mask
            & np.isfinite(
                delta_mass
            )
        ):

            mask = (
                class_mask
                & np.isfinite(
                    delta_mass
                )
            )

            lines.append(
                (
                    "median log10(Mhat/Mtrue) = "
                    f"{np.nanmedian(delta_mass[mask]):.6f}"
                )
            )

        constrained_fraction = (
            np.mean(
                rho_constrained[
                    class_mask
                ]
            )
            if np.any(
                class_mask
            )
            else np.nan
        )

        lines.append(
            (
                "rho-constrained fraction = "
                f"{constrained_fraction:.6f}"
            )
        )

        mask_interval = (
            class_mask
            & interval_valid
        )

        if np.any(
            mask_interval
        ):

            lines.append(
                (
                    "mass interval coverage = "
                    f"{np.mean(covered[mask_interval]):.6f}"
                )
            )

            lines.append(
                (
                    "median sigma_log10_M = "
                    f"{np.nanmedian(sigma_mass[mask_interval]):.6f} dex"
                )
            )

    detected = (
        classes
        == "parallax_detected"
    )

    for name, mask in [
        (
            "Detected + rho constrained",
            detected
            & rho_constrained,
        ),
        (
            "Detected + rho unconstrained",
            detected
            & ~rho_constrained,
        ),
    ]:

        lines.append("")
        lines.append(
            name
        )

        lines.append(
            "-" * 60
        )

        lines.append(
            f"N = {int(mask.sum()):,}"
        )

        valid_mask = (
            mask
            & interval_valid
        )

        lines.append(
            (
                "N with valid mass interval = "
                f"{int(valid_mask.sum()):,}"
            )
        )

        if np.any(
            valid_mask
        ):

            lines.append(
                (
                    "coverage = "
                    f"{np.mean(covered[valid_mask]):.6f}"
                )
            )

            lines.append(
                (
                    "median sigma_log10_M = "
                    f"{np.nanmedian(sigma_mass[valid_mask]):.6f} dex"
                )
            )

    summary = "\n".join(
        lines
    )

    SUMMARY_FILE.write_text(
        summary
        + "\n"
    )

    print()
    print(summary)

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

    if not INPUT.exists():

        raise FileNotFoundError(
            INPUT
        )

    required_columns = [
        "catalog_row",
        "delta_chi2_lrt",
        "positive_lrt_class",

        "rho_true",
        "u0_true",
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
        columns=required_columns,
    )

    require_columns(
        df,
        required_columns,
    )

    if not df[
        "catalog_row"
    ].is_unique:

        raise RuntimeError(
            "catalog_row is not unique."
        )

    print(
        "=" * 100
    )

    print(
        "42: JOINT FINITE-SOURCE + PARALLAX LENS-MASS PROPAGATION"
    )

    print(
        "=" * 100
    )

    print(
        "Input:",
        INPUT,
    )

    print(
        "Rows:",
        len(df),
    )

    events = build_event_table(
        df
    )

    events.to_parquet(
        EVENT_TABLE,
        index=False,
    )

    print()
    print(
        "Saved:",
        EVENT_TABLE,
    )

    make_recovery_figure(
        events
    )

    make_uncertainty_figure(
        events
    )

    make_coverage_figure(
        events
    )

    make_decomposition_figure(
        events
    )

    write_summary(
        events
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
