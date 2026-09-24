#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finite-source / parallax interaction diagnostic.

NO simulation.
NO fitting.

Scientific question
-------------------
Can the no-parallax model H0 partly absorb a true parallax signal by
moving the finite-source parameter rho?

We study three complementary quantities as functions of the true
finite-source strength

    R_FS,true = rho_true / |u0_true|.

1. Excess H0 rho error

       E_rho =
           |log10(rho_H0 / rho_true)|
           -
           |log10(rho_H1 / rho_true)|

   E_rho > 0:
       H0 is farther from the true rho than H1.

   E_rho < 0:
       H1 is farther from the true rho than H0.

2. Standardized H0-H1 rho displacement

       D_rho =
           |rho_H1 - rho_H0|
           / sqrt(sigma_rho,H0^2 + sigma_rho,H1^2)

   This measures whether the change in rho between the two hypotheses
   is large compared with their local formal uncertainties.

3. Local rho-parallax covariance coupling in H1

       C_rho_piE =
           max(
               |corr(rho, piEN)|,
               |corr(rho, piEE)|
           )

   Large values indicate a strong local covariance coupling between
   finite-source size and parallax.

The positive-LRT H1 population is split into

    confused_fspl_no_parallax
    parallax_detected

to test whether rho-parallax coupling differs between events that can
and cannot be distinguished from H0.

Input
-----
analysis/lrt/results/characterization/18_positive_truth/
    h1_positive_truth_characterization.parquet

Outputs
-------
Figure:
analysis/lrt/figures/characterization/
    36_rho_parallax_degeneracy/
    rho_parallax_degeneracy_vs_finite_source_strength.png

Binned results:
analysis/lrt/results/characterization/
    36_rho_parallax_degeneracy/
    rho_parallax_degeneracy_bins.csv

Class summary:
analysis/lrt/results/characterization/
    36_rho_parallax_degeneracy/
    rho_parallax_degeneracy_summary.csv

Text audit:
analysis/lrt/results/characterization/
    36_rho_parallax_degeneracy/
    rho_parallax_degeneracy_audit.txt
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
    "36_rho_parallax_degeneracy"
)

RESULTS_DIR = Path(
    "analysis/lrt/results/characterization/"
    "36_rho_parallax_degeneracy"
)

FIGURE_PATH = (
    FIGURE_DIR
    / "rho_parallax_degeneracy_vs_finite_source_strength.png"
)

BIN_TABLE_PATH = (
    RESULTS_DIR
    / "rho_parallax_degeneracy_bins.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "rho_parallax_degeneracy_summary.csv"
)

AUDIT_PATH = (
    RESULTS_DIR
    / "rho_parallax_degeneracy_audit.txt"
)


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


# Keep the same broad physical R_FS binning philosophy as scripts 34/35.
N_BINS = 18

# Do not plot class/bin combinations with extremely low statistics.
MIN_BIN_COUNT = 20

# Numerical tolerance for identifying rho lower-bound occupancy.
BOUND_RTOL = 1.0e-6
BOUND_ATOL = 1.0e-12


# ============================================================================
# Helpers
# ============================================================================

def quantile_summary(values):
    """
    Return n, q16, median, q84 for finite values.
    """

    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[
        np.isfinite(arr)
    ]

    if len(arr) == 0:
        return 0, np.nan, np.nan, np.nan

    q16, q50, q84 = np.quantile(
        arr,
        [0.16, 0.50, 0.84],
    )

    return (
        len(arr),
        float(q16),
        float(q50),
        float(q84),
    )


def build_log_bins(values, n_bins):
    """
    Logarithmic bins across the complete positive R_FS range.
    """

    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[
        np.isfinite(arr)
        & (arr > 0.0)
    ]

    if len(arr) == 0:
        raise RuntimeError(
            "No positive finite R_FS values available."
        )

    log_min = np.floor(
        np.log10(
            arr.min()
        )
    )

    log_max = np.ceil(
        np.log10(
            arr.max()
        )
    )

    return np.logspace(
        log_min,
        log_max,
        n_bins + 1,
    )


def at_lower_bound(
    value,
    lower,
):
    """
    Identify parameters effectively at the lower production bound.
    """

    value = np.asarray(
        value,
        dtype=float,
    )

    lower = np.asarray(
        lower,
        dtype=float,
    )

    return (
        np.isfinite(value)
        & np.isfinite(lower)
        & np.isclose(
            value,
            lower,
            rtol=BOUND_RTOL,
            atol=BOUND_ATOL,
        )
    )


# ============================================================================
# Main
# ============================================================================

def main():

    print("=" * 100)
    print("36: RHO-PARALLAX DEGENERACY AND H0-H1 RHO DISPLACEMENT")
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

    # ----------------------------------------------------------------------
    # Required inputs
    # ----------------------------------------------------------------------

    required = [
        "catalog_row",
        "positive_lrt_class",
        "delta_chi2_lrt",

        "rho_true",
        "u0_true",
        "rho_over_abs_u0_true",

        "h0_rho",
        "h1_rho",

        "h0_sigma_rho",
        "h1_sigma_rho",

        "h0_bound_rho_min",
        "h1_bound_rho_min",

        "h1_corr_rho_piEN",
        "h1_corr_rho_piEE",

        "h0_cov_finite",
        "h1_cov_finite",

        "h1_cov_positive_definite",
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
    # Extract arrays
    # ----------------------------------------------------------------------

    rho_true = df[
        "rho_true"
    ].to_numpy(dtype=float)

    rho_h0 = df[
        "h0_rho"
    ].to_numpy(dtype=float)

    rho_h1 = df[
        "h1_rho"
    ].to_numpy(dtype=float)

    sigma_h0 = df[
        "h0_sigma_rho"
    ].to_numpy(dtype=float)

    sigma_h1 = df[
        "h1_sigma_rho"
    ].to_numpy(dtype=float)

    rfs_true = df[
        "rho_over_abs_u0_true"
    ].to_numpy(dtype=float)

    corr_N = df[
        "h1_corr_rho_piEN"
    ].to_numpy(dtype=float)

    corr_E = df[
        "h1_corr_rho_piEE"
    ].to_numpy(dtype=float)

    # ----------------------------------------------------------------------
    # 1. H0 excess absolute log-rho error
    # ----------------------------------------------------------------------

    valid_logrho = (
        np.isfinite(rho_true)
        & np.isfinite(rho_h0)
        & np.isfinite(rho_h1)
        & (rho_true > 0.0)
        & (rho_h0 > 0.0)
        & (rho_h1 > 0.0)
    )

    logerr_h0 = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    logerr_h1 = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    excess_h0_logrho_error = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    logerr_h0[valid_logrho] = np.abs(
        np.log10(
            rho_h0[valid_logrho]
            / rho_true[valid_logrho]
        )
    )

    logerr_h1[valid_logrho] = np.abs(
        np.log10(
            rho_h1[valid_logrho]
            / rho_true[valid_logrho]
        )
    )

    excess_h0_logrho_error[
        valid_logrho
    ] = (
        logerr_h0[valid_logrho]
        - logerr_h1[valid_logrho]
    )

    # Positive:
    # H0 is farther from truth than H1.
    #
    # Negative:
    # H1 is farther from truth.

    df[
        "rho_log10_abs_error_h0"
    ] = logerr_h0

    df[
        "rho_log10_abs_error_h1"
    ] = logerr_h1

    df[
        "rho_excess_h0_log10_abs_error"
    ] = excess_h0_logrho_error

    # ----------------------------------------------------------------------
    # Directional H0/H1 shift, retained in output for interpretation.
    # ----------------------------------------------------------------------

    log10_h0_over_h1 = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    valid_ratio = (
        np.isfinite(rho_h0)
        & np.isfinite(rho_h1)
        & (rho_h0 > 0.0)
        & (rho_h1 > 0.0)
    )

    log10_h0_over_h1[
        valid_ratio
    ] = np.log10(
        rho_h0[valid_ratio]
        / rho_h1[valid_ratio]
    )

    df[
        "log10_rho_h0_over_h1"
    ] = log10_h0_over_h1

    # ----------------------------------------------------------------------
    # 2. Standardized H0-H1 displacement
    # ----------------------------------------------------------------------

    denom = np.sqrt(
        sigma_h0**2
        + sigma_h1**2
    )

    valid_separation = (
        np.isfinite(rho_h0)
        & np.isfinite(rho_h1)
        & np.isfinite(sigma_h0)
        & np.isfinite(sigma_h1)
        & (sigma_h0 >= 0.0)
        & (sigma_h1 >= 0.0)
        & np.isfinite(denom)
        & (denom > 0.0)
    )

    D_rho = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    D_rho[
        valid_separation
    ] = (
        np.abs(
            rho_h1[valid_separation]
            - rho_h0[valid_separation]
        )
        / denom[valid_separation]
    )

    df[
        "rho_h0_h1_standardized_displacement"
    ] = D_rho

    # ----------------------------------------------------------------------
    # Cross-check against the already-generated characterization column,
    # if present.
    # ----------------------------------------------------------------------

    separation_crosscheck = np.nan

    if "separation_rho_h0_h1" in df.columns:

        old = df[
            "separation_rho_h0_h1"
        ].to_numpy(dtype=float)

        common = (
            np.isfinite(old)
            & np.isfinite(D_rho)
        )

        if common.any():
            separation_crosscheck = np.nanmax(
                np.abs(
                    old[common]
                    - D_rho[common]
                )
            )

    # ----------------------------------------------------------------------
    # 3. Local H1 rho-parallax covariance coupling
    # ----------------------------------------------------------------------

    valid_corr = (
        df[
            "h1_cov_finite"
        ].fillna(False).to_numpy(dtype=bool)
        & df[
            "h1_cov_positive_definite"
        ].fillna(False).to_numpy(dtype=bool)
        & np.isfinite(corr_N)
        & np.isfinite(corr_E)
    )

    rho_piE_corr_max = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    rho_piE_corr_max[
        valid_corr
    ] = np.maximum(
        np.abs(
            corr_N[valid_corr]
        ),
        np.abs(
            corr_E[valid_corr]
        ),
    )

    df[
        "h1_rho_piE_max_abs_corr"
    ] = rho_piE_corr_max

    # ----------------------------------------------------------------------
    # rho bound occupancy
    # ----------------------------------------------------------------------

    h0_lower = df[
        "h0_bound_rho_min"
    ].to_numpy(dtype=float)

    h1_lower = df[
        "h1_bound_rho_min"
    ].to_numpy(dtype=float)

    h0_rho_lower_bound = at_lower_bound(
        rho_h0,
        h0_lower,
    )

    h1_rho_lower_bound = at_lower_bound(
        rho_h1,
        h1_lower,
    )

    df[
        "h0_rho_at_lower_bound"
    ] = h0_rho_lower_bound

    df[
        "h1_rho_at_lower_bound"
    ] = h1_rho_lower_bound

    # ----------------------------------------------------------------------
    # Global audit
    # ----------------------------------------------------------------------

    print()
    print("=" * 100)
    print("GLOBAL AUDIT")
    print("=" * 100)

    print(
        "valid rho log-error comparison =",
        f"{valid_logrho.sum():,}",
    )

    print(
        "valid standardized displacement =",
        f"{valid_separation.sum():,}",
    )

    print(
        "valid H1 rho-piE correlations =",
        f"{valid_corr.sum():,}",
    )

    print()
    print(
        "H0 rho lower-bound occupancy =",
        f"{h0_rho_lower_bound.sum():,}",
        f"({h0_rho_lower_bound.mean():.6f})",
    )

    print(
        "H1 rho lower-bound occupancy =",
        f"{h1_rho_lower_bound.sum():,}",
        f"({h1_rho_lower_bound.mean():.6f})",
    )

    if np.isfinite(
        separation_crosscheck
    ):
        print()
        print(
            "max |recomputed D_rho - stored separation_rho_h0_h1| =",
            separation_crosscheck,
        )

    # ----------------------------------------------------------------------
    # Audit by detected/confused class
    # ----------------------------------------------------------------------

    summary_rows = []

    print()
    print("=" * 100)
    print("SUMMARY BY LRT CLASS")
    print("=" * 100)

    for cls in CLASSES:

        sub = df.loc[
            df["positive_lrt_class"] == cls
        ]

        print()
        print(cls)
        print("-" * 80)
        print("N =", f"{len(sub):,}")

        metrics = [
            (
                "excess_H0_logrho_error",
                "rho_excess_h0_log10_abs_error",
            ),
            (
                "standardized_H0_H1_rho_displacement",
                "rho_h0_h1_standardized_displacement",
            ),
            (
                "max_abs_corr_rho_piE",
                "h1_rho_piE_max_abs_corr",
            ),
            (
                "log10_rho_H0_over_H1",
                "log10_rho_h0_over_h1",
            ),
        ]

        for label, column in metrics:

            n_valid, q16, median, q84 = (
                quantile_summary(
                    sub[column]
                )
            )

            print(
                f"{label:40s} "
                f"N={n_valid:8,d} "
                f"q16={q16: .6f} "
                f"median={median: .6f} "
                f"q84={q84: .6f}"
            )

            summary_rows.append(
                {
                    "positive_lrt_class":
                        cls,

                    "metric":
                        label,

                    "n_valid":
                        n_valid,

                    "q16":
                        q16,

                    "median":
                        median,

                    "q84":
                        q84,
                }
            )

        n_cls = len(sub)

        h0_bound_count = int(
            sub[
                "h0_rho_at_lower_bound"
            ].sum()
        )

        h1_bound_count = int(
            sub[
                "h1_rho_at_lower_bound"
            ].sum()
        )

        print(
            f"{'H0 rho at lower bound':40s} "
            f"{h0_bound_count:8,d} "
            f"({h0_bound_count/n_cls:.6f})"
        )

        print(
            f"{'H1 rho at lower bound':40s} "
            f"{h1_bound_count:8,d} "
            f"({h1_bound_count/n_cls:.6f})"
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Specific true R_FS > 1 population
    # ----------------------------------------------------------------------

    fs_mask = (
        np.isfinite(rfs_true)
        & (rfs_true > 1.0)
    )

    fs = df.loc[
        fs_mask
    ]

    print()
    print("=" * 100)
    print("TRUE R_FS > 1 SUBSET")
    print("=" * 100)
    print("N =", f"{len(fs):,}")

    print()
    print("LRT classes:")
    print(
        fs[
            "positive_lrt_class"
        ].value_counts()
    )

    for cls in CLASSES:

        sub = fs.loc[
            fs[
                "positive_lrt_class"
            ] == cls
        ]

        if len(sub) == 0:
            continue

        print()
        print(
            f"R_FS > 1, {cls}"
        )
        print("-" * 80)

        for column in [
            "rho_excess_h0_log10_abs_error",
            "rho_h0_h1_standardized_displacement",
            "h1_rho_piE_max_abs_corr",
            "log10_rho_h0_over_h1",
        ]:
            n_valid, q16, med, q84 = (
                quantile_summary(
                    sub[column]
                )
            )

            print(
                f"{column:42s} "
                f"N={n_valid:4d} "
                f"q16={q16: .5f} "
                f"med={med: .5f} "
                f"q84={q84: .5f}"
            )

    # ----------------------------------------------------------------------
    # Binning by true R_FS
    # ----------------------------------------------------------------------

    finite_rfs = (
        np.isfinite(
            df[
                "rho_over_abs_u0_true"
            ]
        )
        & (
            df[
                "rho_over_abs_u0_true"
            ]
            > 0.0
        )
    )

    work = df.loc[
        finite_rfs
    ].copy()

    edges = build_log_bins(
        work[
            "rho_over_abs_u0_true"
        ],
        N_BINS,
    )

    work[
        "rfs_true_bin"
    ] = pd.cut(
        work[
            "rho_over_abs_u0_true"
        ],
        bins=edges,
        include_lowest=True,
        right=True,
    )

    rows = []

    for cls in CLASSES:

        class_df = work.loc[
            work[
                "positive_lrt_class"
            ] == cls
        ]

        grouped = class_df.groupby(
            "rfs_true_bin",
            observed=False,
        )

        for interval, g in grouped:

            if pd.isna(
                interval
            ):
                continue

            if len(g) == 0:
                continue

            row = {
                "positive_lrt_class":
                    cls,

                "bin_left":
                    float(
                        interval.left
                    ),

                "bin_right":
                    float(
                        interval.right
                    ),

                "rfs_true_median":
                    float(
                        np.nanmedian(
                            g[
                                "rho_over_abs_u0_true"
                            ]
                        )
                    ),

                "n_events":
                    len(g),
            }

            metrics = {
                "excess_h0_logrho_error":
                    "rho_excess_h0_log10_abs_error",

                "rho_standardized_displacement":
                    "rho_h0_h1_standardized_displacement",

                "rho_piE_max_abs_corr":
                    "h1_rho_piE_max_abs_corr",

                "log10_rho_h0_over_h1":
                    "log10_rho_h0_over_h1",
            }

            for short_name, column in metrics.items():

                n_valid, q16, med, q84 = (
                    quantile_summary(
                        g[column]
                    )
                )

                row[
                    f"{short_name}_n"
                ] = n_valid

                row[
                    f"{short_name}_q16"
                ] = q16

                row[
                    f"{short_name}_median"
                ] = med

                row[
                    f"{short_name}_q84"
                ] = q84

            row[
                "h0_rho_lower_bound_fraction"
            ] = float(
                g[
                    "h0_rho_at_lower_bound"
                ].mean()
            )

            row[
                "h1_rho_lower_bound_fraction"
            ] = float(
                g[
                    "h1_rho_at_lower_bound"
                ].mean()
            )

            rows.append(
                row
            )

    bins = pd.DataFrame(
        rows
    )

    bins = bins.sort_values(
        [
            "positive_lrt_class",
            "rfs_true_median",
        ]
    ).reset_index(
        drop=True
    )

    bins.to_csv(
        BIN_TABLE_PATH,
        index=False,
    )

    print()
    print("Saved binned table:")
    print(" ", BIN_TABLE_PATH)

    # ----------------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.8, 7.8),
        sharex=True,
    )

    plot_specs = [
        (
            axes[0],
            "excess_h0_logrho_error",
            (
                r"$|\log_{10}(\rho_{H0}/\rho_{\rm true})|"
                r"-|\log_{10}(\rho_{H1}/\rho_{\rm true})|$"
            ),
            "Does H0 distort $\\rho$ more than H1?",
        ),

        (
            axes[1],
            "rho_standardized_displacement",
            (
                r"$|\rho_{H1}-\rho_{H0}|/"
                r"\sqrt{\sigma_{\rho,H0}^2+\sigma_{\rho,H1}^2}$"
            ),
            "H0-H1 displacement of $\\rho$",
        ),

        (
            axes[2],
            "rho_piE_max_abs_corr",
            (
                r"$\max\{|\mathrm{corr}(\rho,\pi_{E,N})|,"
                r"|\mathrm{corr}(\rho,\pi_{E,E})|\}$"
            ),
            "Local $\\rho$-parallax covariance coupling in H1",
        ),
    ]

    for ax, metric, ylabel, title in plot_specs:

        for cls in CLASSES:

            sub = bins.loc[
                bins[
                    "positive_lrt_class"
                ] == cls
            ].copy()

            n_col = (
                f"{metric}_n"
            )

            med_col = (
                f"{metric}_median"
            )

            q16_col = (
                f"{metric}_q16"
            )

            q84_col = (
                f"{metric}_q84"
            )

            valid = (
                np.isfinite(
                    sub[
                        "rfs_true_median"
                    ]
                )
                & np.isfinite(
                    sub[
                        med_col
                    ]
                )
                & (
                    sub[
                        n_col
                    ]
                    >= MIN_BIN_COUNT
                )
            )

            sub = sub.loc[
                valid
            ]

            if len(sub) == 0:
                continue

            x = sub[
                "rfs_true_median"
            ].to_numpy(dtype=float)

            med = sub[
                med_col
            ].to_numpy(dtype=float)

            q16 = sub[
                q16_col
            ].to_numpy(dtype=float)

            q84 = sub[
                q84_col
            ].to_numpy(dtype=float)

            ax.plot(
                x,
                med,
                marker="o",
                linewidth=2,
                label=CLASS_LABELS[cls],
            )

            ax.fill_between(
                x,
                q16,
                q84,
                alpha=0.15,
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
            ylabel
        )

        ax.set_title(
            title
        )

        ax.grid(
            alpha=0.3
        )

    # Zero line has a direct interpretation only in first panel.
    axes[0].axhline(
        0.0,
        linestyle="--",
        linewidth=1.3,
    )

    # D_rho is non-negative.
    axes[1].set_ylim(
        bottom=0.0
    )

    # Correlation is bounded.
    axes[2].set_ylim(
        0.0,
        1.02,
    )

    axes[2].set_xlabel(
        r"True finite-source strength "
        r"$R_{\rm FS}=\rho_{\rm true}/|u_{0,\rm true}|$"
    )

    axes[0].legend(
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
    # Text audit
    # ----------------------------------------------------------------------

    audit_lines = [
        "=" * 100,
        "36: RHO-PARALLAX DEGENERACY AND H0-H1 RHO DISPLACEMENT",
        "=" * 100,
        "",
        f"Input rows: {len(df):,}",
        "",
        f"Valid rho log-error comparisons: {int(valid_logrho.sum()):,}",
        f"Valid standardized displacements: {int(valid_separation.sum()):,}",
        f"Valid H1 rho-piE correlations: {int(valid_corr.sum()):,}",
        "",
        (
            "H0 rho lower-bound occupancy: "
            f"{int(h0_rho_lower_bound.sum()):,} "
            f"({h0_rho_lower_bound.mean():.6f})"
        ),
        (
            "H1 rho lower-bound occupancy: "
            f"{int(h1_rho_lower_bound.sum()):,} "
            f"({h1_rho_lower_bound.mean():.6f})"
        ),
        "",
        (
            "max |recomputed D_rho - stored separation_rho_h0_h1| = "
            f"{separation_crosscheck}"
        ),
        "",
        "Metric interpretation:",
        "",
        (
            "rho_excess_h0_log10_abs_error > 0 means H0 is farther "
            "from the true rho than H1."
        ),
        "",
        (
            "rho_h0_h1_standardized_displacement measures how large "
            "the H0-H1 rho shift is relative to the local formal "
            "rho uncertainties."
        ),
        "",
        (
            "h1_rho_piE_max_abs_corr measures the strongest local "
            "covariance coupling between rho and either parallax "
            "component."
        ),
        "",
        (
            "A finite-source absorption interpretation would be "
            "supported if confused events systematically show both "
            "larger H0 rho distortion/displacement and stronger "
            "rho-parallax coupling than detected events in the same "
            "R_FS regime."
        ),
    ]

    AUDIT_PATH.write_text(
        "\n".join(
            audit_lines
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
        "Binned table:",
        BIN_TABLE_PATH,
    )

    print(
        "Summary:",
        SUMMARY_PATH,
    )

    print(
        "Audit:",
        AUDIT_PATH,
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()


