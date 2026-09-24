#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recoverability and uncertainty of the finite-source-strength parameter

    R_FS = rho / |u0|.

NO simulation.
NO fitting.

This script uses the frozen H1 best-fit parameters and the local H1
covariance in physical (u0, rho) coordinates.

For each event we draw

    (u0, rho) ~ N(
        (u0_hat, rho_hat),
        C_(u0,rho)
    )

and propagate each draw through

    R_FS = rho / |u0|.

Because rho is physically positive, draws with rho <= 0 are excluded
from the propagated R_FS distribution. The retained physical-draw
fraction is recorded explicitly as a diagnostic of the local Gaussian
approximation.

Main diagnostics
----------------
1. Point-estimate recovery:
       log10(R_FS,fit / R_FS,true)

2. Monte-Carlo uncertainty:
       sigma_log10_RFS =
           0.5 * [log10(q84) - log10(q16)]

3. Probability of the finite-source-dominated regime:
       P(R_FS > 1)

The main figure shows recovery and uncertainty as functions of the
true R_FS.

Input
-----
analysis/lrt/results/characterization/18_positive_truth/
    h1_positive_truth_characterization.parquet

Outputs
-------
Figure:
analysis/lrt/figures/characterization/
    35_finite_source_strength_uncertainty/
    finite_source_strength_uncertainty.png

Event-level Monte-Carlo table:
analysis/lrt/results/characterization/
    35_finite_source_strength_uncertainty/
    finite_source_strength_mc_events.parquet

Binned summary:
analysis/lrt/results/characterization/
    35_finite_source_strength_uncertainty/
    finite_source_strength_uncertainty_bins.csv

Classification summary:
analysis/lrt/results/characterization/
    35_finite_source_strength_uncertainty/
    finite_source_strength_classification_summary.csv

Text summary:
analysis/lrt/results/characterization/
    35_finite_source_strength_uncertainty/
    finite_source_strength_uncertainty_summary.txt
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


# ============================================================================
# Configuration
# ============================================================================

INPUT_PATH = Path(
    "analysis/lrt/results/characterization/18_positive_truth/"
    "h1_positive_truth_characterization.parquet"
)

FIGURE_DIR = Path(
    "analysis/lrt/figures/characterization/"
    "35_finite_source_strength_uncertainty"
)

RESULTS_DIR = Path(
    "analysis/lrt/results/characterization/"
    "35_finite_source_strength_uncertainty"
)

FIGURE_PATH = (
    FIGURE_DIR
    / "finite_source_strength_uncertainty.png"
)

EVENT_TABLE_PATH = (
    RESULTS_DIR
    / "finite_source_strength_mc_events.parquet"
)

BIN_TABLE_PATH = (
    RESULTS_DIR
    / "finite_source_strength_uncertainty_bins.csv"
)

CLASSIFICATION_TABLE_PATH = (
    RESULTS_DIR
    / "finite_source_strength_classification_summary.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "finite_source_strength_uncertainty_summary.txt"
)


# Monte-Carlo configuration.
#
# 512 draws/event is enough for this population-level diagnostic while
# remaining tractable for ~330k events. Increase later if we need
# event-level precision.
N_DRAWS = 512

# Process events in batches so we never allocate the full
# N_events x N_draws arrays at once.
BATCH_SIZE = 2000

RNG_SEED = 20260923

N_BINS = 18

# Do not display a population-statistics point unless the bin contains
# at least this many valid events.
MIN_BIN_COUNT = 20

# Minimum number of physically valid MC draws required for an event.
MIN_VALID_DRAWS = 100

# Numerical protection against division by zero.
U0_ABS_EPS = 1.0e-12

# Classification based on the posterior-like propagated probability
# from the local Gaussian covariance approximation.
P_HIGH = 0.84
P_LOW = 0.16


# ============================================================================
# Helpers
# ============================================================================

def quantiles_or_nan(values, quantiles=(0.16, 0.50, 0.84)):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return tuple(np.nan for _ in quantiles)

    return tuple(
        np.quantile(arr, quantiles)
    )


def build_log_bins(values, n_bins):
    """
    Reproduce the simple log-bin philosophy used in script 34.
    """
    arr = np.asarray(values, dtype=float)

    arr = arr[
        np.isfinite(arr)
        & (arr > 0.0)
    ]

    if len(arr) == 0:
        raise RuntimeError(
            "No positive finite R_FS values available for binning."
        )

    log_min = np.floor(
        np.log10(arr.min())
    )

    log_max = np.ceil(
        np.log10(arr.max())
    )

    return np.logspace(
        log_min,
        log_max,
        n_bins + 1,
    )


def classify_probability(p_gt1):
    if not np.isfinite(p_gt1):
        return "invalid"

    if p_gt1 > P_HIGH:
        return "robust_RFS_gt_1"

    if p_gt1 < P_LOW:
        return "robust_RFS_lt_1"

    return "ambiguous"


# ============================================================================
# Monte-Carlo propagation
# ============================================================================

def propagate_rfs_mc(df):
    """
    Propagate the local H1 covariance in (u0, rho) to

        R_FS = rho / |u0|.

    The 2x2 Gaussian is generated analytically using a conditional
    Cholesky representation, avoiding a per-event scipy call.

    Returns an event-level DataFrame.
    """

    n_events = len(df)

    rng = np.random.default_rng(
        RNG_SEED
    )

    u0_fit = df[
        "h1_u0"
    ].to_numpy(dtype=float)

    rho_fit = df[
        "h1_rho"
    ].to_numpy(dtype=float)

    Cuu = df[
        "h1_cov_u0_u0"
    ].to_numpy(dtype=float)

    Cur = df[
        "h1_cov_u0_rho"
    ].to_numpy(dtype=float)

    Crr = df[
        "h1_cov_rho_rho"
    ].to_numpy(dtype=float)

    rfs_true = df[
        "rho_over_abs_u0_true"
    ].to_numpy(dtype=float)

    # Point estimate.
    rfs_fit = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    valid_point = (
        np.isfinite(u0_fit)
        & np.isfinite(rho_fit)
        & (rho_fit > 0.0)
        & (np.abs(u0_fit) > U0_ABS_EPS)
    )

    rfs_fit[valid_point] = (
        rho_fit[valid_point]
        / np.abs(u0_fit[valid_point])
    )

    # Check positive definiteness of the 2x2 covariance.
    det = (
        Cuu * Crr
        - Cur * Cur
    )

    covariance_valid = (
        np.isfinite(Cuu)
        & np.isfinite(Cur)
        & np.isfinite(Crr)
        & (Cuu > 0.0)
        & (Crr > 0.0)
        & (det > 0.0)
        & np.isfinite(u0_fit)
        & np.isfinite(rho_fit)
    )

    q16 = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    q50 = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    q84 = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    p_gt1 = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    physical_draw_fraction = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    n_valid_draws = np.zeros(
        n_events,
        dtype=int,
    )

    print()
    print("Monte-Carlo propagation")
    print(f"  events       = {n_events:,}")
    print(f"  draws/event  = {N_DRAWS}")
    print(f"  batch size   = {BATCH_SIZE}")
    print(
        "  valid 2x2 covariance =",
        f"{covariance_valid.sum():,}",
        f"({covariance_valid.mean():.6f})",
    )

    valid_indices = np.flatnonzero(
        covariance_valid
    )

    n_valid_events = len(
        valid_indices
    )

    for start in range(
        0,
        n_valid_events,
        BATCH_SIZE,
    ):
        stop = min(
            start + BATCH_SIZE,
            n_valid_events,
        )

        idx = valid_indices[
            start:stop
        ]

        m = len(idx)

        # ------------------------------------------------------------
        # Analytic Cholesky representation
        #
        # u = mu_u + a z1
        #
        # rho = mu_r + b z1 + c z2
        #
        # where
        #
        # a = sqrt(Cuu)
        # b = Cur / a
        # c^2 = Crr - b^2
        # ------------------------------------------------------------

        a = np.sqrt(
            Cuu[idx]
        )

        b = (
            Cur[idx]
            / a
        )

        c2 = (
            Crr[idx]
            - b * b
        )

        # Positive-definiteness ensures c2 > 0 mathematically.
        # Clip tiny floating-point negatives only.
        c2 = np.maximum(
            c2,
            0.0,
        )

        c = np.sqrt(
            c2
        )

        z1 = rng.standard_normal(
            size=(m, N_DRAWS)
        )

        z2 = rng.standard_normal(
            size=(m, N_DRAWS)
        )

        draw_u0 = (
            u0_fit[idx, None]
            + a[:, None] * z1
        )

        draw_rho = (
            rho_fit[idx, None]
            + b[:, None] * z1
            + c[:, None] * z2
        )

        physical = (
            np.isfinite(draw_u0)
            & np.isfinite(draw_rho)
            & (draw_rho > 0.0)
            & (
                np.abs(draw_u0)
                > U0_ABS_EPS
            )
        )

        n_phys = physical.sum(
            axis=1
        )

        n_valid_draws[idx] = n_phys

        physical_draw_fraction[idx] = (
            n_phys
            / float(N_DRAWS)
        )

        # Compute R_FS only for physical draws.
        draw_rfs = np.full(
            shape=(m, N_DRAWS),
            fill_value=np.nan,
            dtype=float,
        )

        draw_rfs[physical] = (
            draw_rho[physical]
            / np.abs(
                draw_u0[physical]
            )
        )

        enough = (
            n_phys
            >= MIN_VALID_DRAWS
        )

        local_enough = np.flatnonzero(
            enough
        )

        for j in local_enough:
            vals = draw_rfs[
                j,
                physical[j],
            ]

            if len(vals) < MIN_VALID_DRAWS:
                continue

            q16_j, q50_j, q84_j = np.quantile(
                vals,
                [0.16, 0.50, 0.84],
            )

            global_i = idx[j]

            q16[global_i] = q16_j
            q50[global_i] = q50_j
            q84[global_i] = q84_j

            p_gt1[global_i] = np.mean(
                vals > 1.0
            )

        if (
            start == 0
            or stop == n_valid_events
            or stop % 20000 < BATCH_SIZE
        ):
            print(
                f"  processed "
                f"{stop:,}/{n_valid_events:,} "
                f"valid-covariance events"
            )

    # ------------------------------------------------------------------
    # Derived uncertainty metrics
    # ------------------------------------------------------------------

    mc_valid = (
        np.isfinite(q16)
        & np.isfinite(q50)
        & np.isfinite(q84)
        & (q16 > 0.0)
        & (q50 > 0.0)
        & (q84 > 0.0)
        & (q84 >= q16)
    )

    sigma_log10_rfs = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    sigma_log10_rfs[mc_valid] = (
        0.5
        * (
            np.log10(q84[mc_valid])
            - np.log10(q16[mc_valid])
        )
    )

    log10_fit_over_true = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    valid_recovery = (
        np.isfinite(rfs_fit)
        & (rfs_fit > 0.0)
        & np.isfinite(rfs_true)
        & (rfs_true > 0.0)
    )

    log10_fit_over_true[valid_recovery] = (
        np.log10(
            rfs_fit[valid_recovery]
            / rfs_true[valid_recovery]
        )
    )

    log10_mc_median_over_true = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    valid_mc_recovery = (
        mc_valid
        & np.isfinite(rfs_true)
        & (rfs_true > 0.0)
    )

    log10_mc_median_over_true[
        valid_mc_recovery
    ] = (
        np.log10(
            q50[valid_mc_recovery]
            / rfs_true[valid_mc_recovery]
        )
    )

    classification = np.array(
        [
            classify_probability(p)
            for p in p_gt1
        ],
        dtype=object,
    )

    out = pd.DataFrame(
        {
            "catalog_row":
                df["catalog_row"].to_numpy(),

            "positive_lrt_class":
                df["positive_lrt_class"].to_numpy(),

            "delta_chi2_lrt":
                df["delta_chi2_lrt"].to_numpy(),

            "rfs_true":
                rfs_true,

            "u0_true":
                df["u0_true"].to_numpy(dtype=float),

            "rho_true":
                df["rho_true"].to_numpy(dtype=float),

            "h1_u0":
                u0_fit,

            "h1_rho":
                rho_fit,

            "rfs_fit":
                rfs_fit,

            "log10_rfs_fit_over_true":
                log10_fit_over_true,

            "mc_covariance_valid":
                covariance_valid,

            "mc_n_valid_draws":
                n_valid_draws,

            "mc_physical_draw_fraction":
                physical_draw_fraction,

            "mc_q16_rfs":
                q16,

            "mc_q50_rfs":
                q50,

            "mc_q84_rfs":
                q84,

            "mc_sigma_log10_rfs":
                sigma_log10_rfs,

            "log10_mc_median_over_true":
                log10_mc_median_over_true,

            "mc_probability_rfs_gt_1":
                p_gt1,

            "mc_rfs_regime":
                classification,
        }
    )

    return out


# ============================================================================
# Binned summaries
# ============================================================================

def build_binned_summary(events):
    edges = build_log_bins(
        events["rfs_true"],
        N_BINS,
    )

    events = events.copy()

    events["rfs_true_bin"] = pd.cut(
        events["rfs_true"],
        bins=edges,
        include_lowest=True,
        right=True,
    )

    rows = []

    grouped = events.groupby(
        "rfs_true_bin",
        observed=False,
    )

    for interval, g in grouped:
        if pd.isna(interval):
            continue

        g = g.loc[
            np.isfinite(
                g["rfs_true"]
            )
            & (
                g["rfs_true"] > 0.0
            )
        ]

        if len(g) == 0:
            continue

        # ------------------------------------------------------------
        # Point-estimate recovery distribution
        # ------------------------------------------------------------

        recovery = g[
            "log10_rfs_fit_over_true"
        ].to_numpy(dtype=float)

        recovery = recovery[
            np.isfinite(recovery)
        ]

        rec_q16, rec_med, rec_q84 = (
            quantiles_or_nan(
                recovery
            )
        )

        # ------------------------------------------------------------
        # MC uncertainty distribution
        # ------------------------------------------------------------

        uncertainty = g[
            "mc_sigma_log10_rfs"
        ].to_numpy(dtype=float)

        uncertainty = uncertainty[
            np.isfinite(uncertainty)
        ]

        unc_q16, unc_med, unc_q84 = (
            quantiles_or_nan(
                uncertainty
            )
        )

        # ------------------------------------------------------------
        # Physical Gaussian-draw fraction
        # ------------------------------------------------------------

        phys = g[
            "mc_physical_draw_fraction"
        ].to_numpy(dtype=float)

        phys = phys[
            np.isfinite(phys)
        ]

        phys_q16, phys_med, phys_q84 = (
            quantiles_or_nan(
                phys
            )
        )

        # ------------------------------------------------------------
        # P(R_FS > 1)
        # ------------------------------------------------------------

        prob = g[
            "mc_probability_rfs_gt_1"
        ].to_numpy(dtype=float)

        prob = prob[
            np.isfinite(prob)
        ]

        p_q16, p_med, p_q84 = (
            quantiles_or_nan(
                prob
            )
        )

        rows.append(
            {
                "bin_left":
                    float(interval.left),

                "bin_right":
                    float(interval.right),

                "rfs_true_median":
                    float(
                        np.nanmedian(
                            g["rfs_true"]
                        )
                    ),

                "n_events":
                    len(g),

                "n_recovery_valid":
                    len(recovery),

                "recovery_log10_ratio_q16":
                    rec_q16,

                "recovery_log10_ratio_median":
                    rec_med,

                "recovery_log10_ratio_q84":
                    rec_q84,

                "n_uncertainty_valid":
                    len(uncertainty),

                "sigma_log10_rfs_q16":
                    unc_q16,

                "sigma_log10_rfs_median":
                    unc_med,

                "sigma_log10_rfs_q84":
                    unc_q84,

                "physical_draw_fraction_q16":
                    phys_q16,

                "physical_draw_fraction_median":
                    phys_med,

                "physical_draw_fraction_q84":
                    phys_q84,

                "prob_rfs_gt1_q16":
                    p_q16,

                "prob_rfs_gt1_median":
                    p_med,

                "prob_rfs_gt1_q84":
                    p_q84,
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "rfs_true_median"
    ).reset_index(
        drop=True
    )


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 100)
    print("35: FINITE-SOURCE-STRENGTH RECOVERY AND UNCERTAINTY")
    print("=" * 100)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_parquet(
        INPUT_PATH
    )

    print()
    print("Input:")
    print(" ", INPUT_PATH)
    print("rows =", f"{len(df):,}")

    required = [
        "catalog_row",
        "delta_chi2_lrt",
        "positive_lrt_class",
        "u0_true",
        "rho_true",
        "rho_over_abs_u0_true",
        "h1_u0",
        "h1_rho",
        "h1_cov_u0_u0",
        "h1_cov_u0_rho",
        "h1_cov_rho_rho",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {c}"
                for c in missing
            )
        )

    # ----------------------------------------------------------------------
    # Sanity check truth ratio
    # ----------------------------------------------------------------------

    truth_ratio_rebuilt = (
        df["rho_true"].to_numpy(dtype=float)
        / np.abs(
            df["u0_true"].to_numpy(dtype=float)
        )
    )

    truth_ratio_stored = df[
        "rho_over_abs_u0_true"
    ].to_numpy(dtype=float)

    check = (
        np.isfinite(truth_ratio_rebuilt)
        & np.isfinite(truth_ratio_stored)
        & (truth_ratio_stored > 0.0)
    )

    rel = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    rel[check] = (
        np.abs(
            truth_ratio_rebuilt[check]
            - truth_ratio_stored[check]
        )
        / truth_ratio_stored[check]
    )

    print()
    print("Truth R_FS reconstruction check:")
    print(
        "  max relative difference =",
        np.nanmax(rel),
    )

    # ----------------------------------------------------------------------
    # MC propagation
    # ----------------------------------------------------------------------

    events = propagate_rfs_mc(
        df
    )

    events.to_parquet(
        EVENT_TABLE_PATH,
        index=False,
    )

    print()
    print(
        "Saved event-level MC table:",
        EVENT_TABLE_PATH,
    )

    # ----------------------------------------------------------------------
    # Global diagnostics
    # ----------------------------------------------------------------------

    n = len(events)

    n_cov_valid = int(
        events[
            "mc_covariance_valid"
        ].sum()
    )

    mc_valid = (
        np.isfinite(
            events["mc_q50_rfs"]
        )
    )

    n_mc_valid = int(
        mc_valid.sum()
    )

    true_gt1 = (
        events["rfs_true"]
        > 1.0
    )

    print()
    print("=" * 100)
    print("GLOBAL DIAGNOSTICS")
    print("=" * 100)

    print(
        f"all positive-T events      = {n:,}"
    )

    print(
        f"valid (u0,rho) covariance = "
        f"{n_cov_valid:,} "
        f"({n_cov_valid / n:.6f})"
    )

    print(
        f"valid propagated MC        = "
        f"{n_mc_valid:,} "
        f"({n_mc_valid / n:.6f})"
    )

    print(
        f"true R_FS > 1              = "
        f"{true_gt1.sum():,}"
    )

    # ----------------------------------------------------------------------
    # Probability-based classification
    # ----------------------------------------------------------------------

    class_summary_rows = []

    for truth_label, truth_mask in [
        (
            "true_RFS_lt_1",
            events["rfs_true"] < 1.0,
        ),
        (
            "true_RFS_gt_1",
            events["rfs_true"] > 1.0,
        ),
    ]:
        sub = events.loc[
            truth_mask
        ]

        counts = (
            sub["mc_rfs_regime"]
            .value_counts()
        )

        total = len(sub)

        print()
        print(truth_label)
        print("-" * 70)
        print("N =", f"{total:,}")

        for category in [
            "robust_RFS_lt_1",
            "ambiguous",
            "robust_RFS_gt_1",
            "invalid",
        ]:
            count = int(
                counts.get(
                    category,
                    0,
                )
            )

            frac = (
                count / total
                if total > 0
                else np.nan
            )

            print(
                f"  {category:20s} "
                f"{count:8,d} "
                f"({frac:.6f})"
            )

            class_summary_rows.append(
                {
                    "truth_regime":
                        truth_label,

                    "mc_classification":
                        category,

                    "count":
                        count,

                    "fraction":
                        frac,
                }
            )

    classification_summary = pd.DataFrame(
        class_summary_rows
    )

    classification_summary.to_csv(
        CLASSIFICATION_TABLE_PATH,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Specifically inspect the 375 true source-transit events
    # ----------------------------------------------------------------------

    fs = events.loc[
        true_gt1
    ].copy()

    print()
    print("=" * 100)
    print("TRUE R_FS > 1 SUBSET")
    print("=" * 100)

    print(
        "N =",
        f"{len(fs):,}",
    )

    print()
    print("positive_lrt_class:")
    print(
        fs[
            "positive_lrt_class"
        ].value_counts()
    )

    print()
    print("P(R_FS > 1) classification:")
    print(
        fs[
            "mc_rfs_regime"
        ].value_counts(
            dropna=False
        )
    )

    valid_fs_unc = fs[
        "mc_sigma_log10_rfs"
    ]

    valid_fs_unc = valid_fs_unc[
        np.isfinite(
            valid_fs_unc
        )
    ]

    if len(valid_fs_unc) > 0:
        print()
        print(
            "MC uncertainty half-width in dex "
            "for true R_FS > 1:"
        )

        print(
            "  median =",
            float(
                valid_fs_unc.median()
            ),
        )

        print(
            "  q16,q84 =",
            tuple(
                np.quantile(
                    valid_fs_unc,
                    [0.16, 0.84],
                )
            ),
        )

    # ----------------------------------------------------------------------
    # Binned summary
    # ----------------------------------------------------------------------

    bins = build_binned_summary(
        events
    )

    bins.to_csv(
        BIN_TABLE_PATH,
        index=False,
    )

    print()
    print("Binned summary:")
    print(
        bins[
            [
                "rfs_true_median",
                "n_events",
                "recovery_log10_ratio_median",
                "sigma_log10_rfs_median",
                "physical_draw_fraction_median",
                "prob_rfs_gt1_median",
            ]
        ].to_string(
            index=False
        )
    )

    # ----------------------------------------------------------------------
    # Main figure
    # ----------------------------------------------------------------------

    use_recovery = (
        np.isfinite(
            bins[
                "recovery_log10_ratio_median"
            ]
        )
        & (
            bins[
                "n_recovery_valid"
            ]
            >= MIN_BIN_COUNT
        )
    )

    use_uncertainty = (
        np.isfinite(
            bins[
                "sigma_log10_rfs_median"
            ]
        )
        & (
            bins[
                "n_uncertainty_valid"
            ]
            >= MIN_BIN_COUNT
        )
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.8, 5.8),
        sharex=True,
    )

    # ------------------------------------------------------------------
    # Panel 1: recovery
    # ------------------------------------------------------------------

    ax = axes[0]

    sub = bins.loc[
        use_recovery
    ]

    x = sub[
        "rfs_true_median"
    ].to_numpy(dtype=float)

    med = sub[
        "recovery_log10_ratio_median"
    ].to_numpy(dtype=float)

    q16 = sub[
        "recovery_log10_ratio_q16"
    ].to_numpy(dtype=float)

    q84 = sub[
        "recovery_log10_ratio_q84"
    ].to_numpy(dtype=float)

    ax.plot(
        x,
        med,
        marker="o",
        linewidth=2,
        label="Median",
    )

    ax.fill_between(
        x,
        q16,
        q84,
        alpha=0.25,
        label="16–84% population spread",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.4,
    )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1.4,
    )

    ax.set_xscale(
        "log"
    )

    ax.set_ylabel(
        r"$\log_{10}(\widehat R_{\rm FS}/R_{\rm FS,true})$"
    )

    ax.set_title(
        "Recovery of finite-source strength"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        loc="best"
    )

    # ------------------------------------------------------------------
    # Panel 2: MC uncertainty
    # ------------------------------------------------------------------

    ax = axes[1]

    sub = bins.loc[
        use_uncertainty
    ]

    x = sub[
        "rfs_true_median"
    ].to_numpy(dtype=float)

    med = sub[
        "sigma_log10_rfs_median"
    ].to_numpy(dtype=float)

    q16 = sub[
        "sigma_log10_rfs_q16"
    ].to_numpy(dtype=float)

    q84 = sub[
        "sigma_log10_rfs_q84"
    ].to_numpy(dtype=float)

    ax.plot(
        x,
        med,
        marker="o",
        linewidth=2,
        label="Median uncertainty",
    )

    ax.fill_between(
        x,
        q16,
        q84,
        alpha=0.25,
        label="16–84% across events",
    )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1.4,
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        r"True finite-source strength "
        r"$R_{\rm FS}=\rho_{\rm true}/|u_{0,\rm true}|$"
    )

    ax.set_ylabel(
        r"MC half-width in $\log_{10}R_{\rm FS}$ [dex]"
    )

    ax.set_title(
        "Propagated uncertainty from the H1 $(u_0,\\rho)$ covariance"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        loc="best"
    )



    label_panels(axes)
    fig.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    save_pdf_companion(fig, FIGURE_PATH)

    plt.close(
        fig
    )

    # ----------------------------------------------------------------------
    # Text summary
    # ----------------------------------------------------------------------

    physical_fraction = events[
        "mc_physical_draw_fraction"
    ]

    physical_fraction = physical_fraction[
        np.isfinite(
            physical_fraction
        )
    ]

    uncertainty = events[
        "mc_sigma_log10_rfs"
    ]

    uncertainty = uncertainty[
        np.isfinite(
            uncertainty
        )
    ]

    summary_lines = [
        "=" * 100,
        "35: FINITE-SOURCE-STRENGTH RECOVERY AND UNCERTAINTY",
        "=" * 100,
        "",
        f"Input rows: {n:,}",
        f"MC draws/event: {N_DRAWS}",
        f"Valid covariance events: {n_cov_valid:,}",
        f"Valid propagated MC events: {n_mc_valid:,}",
        "",
        f"True R_FS > 1 events: {int(true_gt1.sum()):,}",
        "",
        "Global physical-draw fraction:",
        (
            f"  median = {physical_fraction.median():.6f}"
            if len(physical_fraction)
            else "  no valid values"
        ),
        "",
        "Global MC half-width in log10(R_FS):",
        (
            f"  median = {uncertainty.median():.6f} dex"
            if len(uncertainty)
            else "  no valid values"
        ),
        "",
        "Interpretation:",
        (
            "Large MC half-widths or low physical-draw fractions near "
            "R_FS ~ 1 indicate that finite-source strength is weakly "
            "identified by the local H1 covariance and should not be "
            "treated as a sharply measured explanatory variable."
        ),
        "",
        (
            "The P(R_FS > 1) classification is a diagnostic based on "
            "the local Gaussian covariance approximation, not a full "
            "posterior probability."
        ),
    ]

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        )
    )

    print()
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)

    print(
        "Figure:",
        FIGURE_PATH,
    )

    print(
        "Event table:",
        EVENT_TABLE_PATH,
    )

    print(
        "Bin table:",
        BIN_TABLE_PATH,
    )

    print(
        "Classification:",
        CLASSIFICATION_TABLE_PATH,
    )

    print(
        "Summary:",
        SUMMARY_PATH,
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()


