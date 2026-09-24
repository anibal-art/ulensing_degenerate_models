#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finite-source information among annual-parallax detections.

NO simulation.
NO fitting.

Scientific question
-------------------
Among events for which annual parallax is already distinguishable,
does a small formal uncertainty on rho actually imply that rho has
been measured accurately?

We restrict the analysis to

    positive_lrt_class == "parallax_detected"

and study the local-covariance quantity

    sigma_rho / rho_hat.

IMPORTANT
---------
This quantity is NOT treated as a finite-source detection criterion.
It is only a diagnostic/proxy.

Because this is a simulation, we can audit the proxy against truth via

    Delta log10 rho = log10(rho_hat / rho_true).

The diagnostic asks:

1. What fraction of parallax-detected events passes
       sigma_rho / rho_hat < threshold ?

2. Among events passing that cut, what fraction actually satisfies
       |log10(rho_hat/rho_true)| < 0.3 dex ?

3. How does actual rho recovery depend on the reported formal precision?

Input
-----
analysis/lrt/results/characterization/18_positive_truth/
    h1_positive_truth_characterization.parquet

Outputs
-------
analysis/lrt/figures/characterization/
    44_finite_source_information_after_parallax_detection/
    finite_source_information_after_parallax_detection.png
    finite_source_information_after_parallax_detection.pdf

analysis/lrt/results/characterization/
    44_finite_source_information_after_parallax_detection/
    threshold_scan.csv
    rho_recovery_vs_formal_precision.csv
    summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# Configuration
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
    / "44_finite_source_information_after_parallax_detection"
)

RESULT_DIR = (
    ROOT
    / "analysis/lrt/results/characterization"
    / "44_finite_source_information_after_parallax_detection"
)

FIGURE_BASE = (
    FIGURE_DIR
    / "finite_source_information_after_parallax_detection"
)

THRESHOLD_TABLE = (
    RESULT_DIR
    / "threshold_scan.csv"
)

RECOVERY_TABLE = (
    RESULT_DIR
    / "rho_recovery_vs_formal_precision.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "summary.txt"
)


# Frozen population audit
EXPECTED_PARALLAX_DETECTED = 229921

# Diagnostic precision cut only.
ADOPTED_PROXY = 0.30

# Truth-side definition used only to audit the proxy.
# 0.3 dex is approximately a factor of 2.
ACCURATE_RHO_DEX = 0.30

# Binning in formal sigma_rho / rho_hat.
N_BINS = 16
MIN_BIN_COUNT = 20


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
            "lines.linewidth": 2.0,

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


def build_log_edges(values, n_bins):

    x = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & (x > 0.0)
    )

    if not np.any(valid):

        raise RuntimeError(
            "No positive finite values available for logarithmic binning."
        )

    lo = float(
        np.nanmin(
            x[valid]
        )
    )

    hi = float(
        np.nanmax(
            x[valid]
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


def quantiles_or_nan(values):

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    return tuple(
        np.quantile(
            x,
            [
                0.16,
                0.50,
                0.84,
            ],
        )
    )


def save_figure(fig, base):

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

    print()
    print("Saved figure:")
    print(" ", png)
    print(" ", pdf)


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

    print("=" * 100)
    print(
        "44: FINITE-SOURCE INFORMATION AMONG ANNUAL-PARALLAX DETECTIONS"
    )
    print("=" * 100)

    if not INPUT.exists():

        raise FileNotFoundError(
            INPUT
        )

    required = [
        "catalog_row",
        "positive_lrt_class",
        "delta_chi2_lrt",

        "rho_true",

        "h1_rho",
        "h1_cov_rho_rho",
    ]

    df = pd.read_parquet(
        INPUT,
        columns=required,
    )

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(
                f"  {c}"
                for c in missing
            )
        )

    print()
    print(
        "Input rows =",
        f"{len(df):,}",
    )

    # ========================================================================
    # Restrict to actual annual-parallax detections
    # ========================================================================

    detected = (
        df[
            "positive_lrt_class"
        ].astype(str)
        == "parallax_detected"
    )

    work = df.loc[
        detected
    ].copy()

    if len(work) != EXPECTED_PARALLAX_DETECTED:

        raise RuntimeError(
            "Parallax-detected population audit failed: "
            f"{len(work):,} != {EXPECTED_PARALLAX_DETECTED:,}"
        )

    print(
        "Parallax-detected events =",
        f"{len(work):,}",
    )

    # ========================================================================
    # Build rho diagnostics
    # ========================================================================

    rho_true = numeric(
        work,
        "rho_true",
    )

    rho_hat = numeric(
        work,
        "h1_rho",
    )

    var_rho = numeric(
        work,
        "h1_cov_rho_rho",
    )

    sigma_rho = np.full(
        len(work),
        np.nan,
    )

    valid_var = (
        np.isfinite(var_rho)
        & (var_rho >= 0.0)
    )

    sigma_rho[
        valid_var
    ] = np.sqrt(
        var_rho[
            valid_var
        ]
    )

    relative_uncertainty = np.full(
        len(work),
        np.nan,
    )

    valid_relative = (
        np.isfinite(
            sigma_rho
        )
        & np.isfinite(
            rho_hat
        )
        & (rho_hat > 0.0)
    )

    relative_uncertainty[
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
        len(work),
        np.nan,
    )

    valid_recovery = (
        np.isfinite(
            rho_true
        )
        & (rho_true > 0.0)
        & np.isfinite(
            rho_hat
        )
        & (rho_hat > 0.0)
    )

    delta_log10_rho[
        valid_recovery
    ] = np.log10(
        rho_hat[
            valid_recovery
        ]
        / rho_true[
            valid_recovery
        ]
    )

    accurate = (
        np.isfinite(
            delta_log10_rho
        )
        & (
            np.abs(
                delta_log10_rho
            )
            < ACCURATE_RHO_DEX
        )
    )

    valid = (
        np.isfinite(
            relative_uncertainty
        )
        & (relative_uncertainty > 0.0)
        & np.isfinite(
            delta_log10_rho
        )
    )

    print(
        "Events with valid formal rho uncertainty + truth recovery =",
        f"{int(valid.sum()):,}",
    )

    # ========================================================================
    # Threshold scan
    # ========================================================================

    thresholds = np.logspace(
        -2.0,
        3.0,
        150,
    )

    threshold_rows = []

    n_total = len(
        work
    )

    for threshold in thresholds:

        selected = (
            valid
            & (
                relative_uncertainty
                < threshold
            )
        )

        n_selected = int(
            selected.sum()
        )

        fraction_selected = (
            n_selected
            / n_total
        )

        if n_selected > 0:

            accurate_fraction = float(
                np.mean(
                    accurate[
                        selected
                    ]
                )
            )

            median_abs_error = float(
                np.median(
                    np.abs(
                        delta_log10_rho[
                            selected
                        ]
                    )
                )
            )

        else:

            accurate_fraction = np.nan
            median_abs_error = np.nan

        threshold_rows.append(
            {
                "precision_threshold":
                    threshold,

                "n_selected":
                    n_selected,

                "fraction_of_parallax_detected":
                    fraction_selected,

                "fraction_accurate_among_selected":
                    accurate_fraction,

                "median_abs_log10_rho_error":
                    median_abs_error,
            }
        )

    threshold_table = pd.DataFrame(
        threshold_rows
    )

    threshold_table.to_csv(
        THRESHOLD_TABLE,
        index=False,
    )

    # ========================================================================
    # Recovery versus reported formal precision
    # ========================================================================

    edges = build_log_edges(
        relative_uncertainty[
            valid
        ],
        N_BINS,
    )

    recovery_rows = []

    for ibin in range(
        len(edges)
        - 1
    ):

        left = edges[
            ibin
        ]

        right = edges[
            ibin + 1
        ]

        mask = (
            valid
            & (
                relative_uncertainty
                >= left
            )
            & (
                relative_uncertainty
                < right
            )
        )

        q16, median, q84 = quantiles_or_nan(
            delta_log10_rho[
                mask
            ]
        )

        n = int(
            mask.sum()
        )

        if n > 0:

            accurate_fraction = float(
                np.mean(
                    accurate[
                        mask
                    ]
                )
            )

        else:

            accurate_fraction = np.nan

        recovery_rows.append(
            {
                "precision_left":
                    left,

                "precision_right":
                    right,

                "precision_center":
                    np.sqrt(
                        left
                        * right
                    ),

                "n":
                    n,

                "rho_error_q16":
                    q16,

                "rho_error_median":
                    median,

                "rho_error_q84":
                    q84,

                "fraction_accurate":
                    accurate_fraction,
            }
        )

    recovery_table = pd.DataFrame(
        recovery_rows
    )

    recovery_table.to_csv(
        RECOVERY_TABLE,
        index=False,
    )

    # ========================================================================
    # Adopted cut summary
    # ========================================================================

    proxy_selected = (
        valid
        & (
            relative_uncertainty
            < ADOPTED_PROXY
        )
    )

    n_proxy = int(
        proxy_selected.sum()
    )

    proxy_fraction = (
        n_proxy
        / n_total
    )

    if n_proxy:

        proxy_accuracy = float(
            np.mean(
                accurate[
                    proxy_selected
                ]
            )
        )

        proxy_median_rho_error = float(
            np.median(
                delta_log10_rho[
                    proxy_selected
                ]
            )
        )

        proxy_q16, _, proxy_q84 = quantiles_or_nan(
            delta_log10_rho[
                proxy_selected
            ]
        )

    else:

        proxy_accuracy = np.nan
        proxy_median_rho_error = np.nan
        proxy_q16 = np.nan
        proxy_q84 = np.nan

    # ========================================================================
    # Figure
    # ========================================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            13.0,
            5.6,
        ),
    )

    # ------------------------------------------------------------------------
    # Panel A
    # ------------------------------------------------------------------------

    ax = axes[
        0
    ]

    ax.plot(
        threshold_table[
            "precision_threshold"
        ],
        threshold_table[
            "fraction_of_parallax_detected"
        ],
        label=(
            r"Passing formal "
            r"$\rho$-precision proxy"
        ),
    )

    ax.plot(
        threshold_table[
            "precision_threshold"
        ],
        threshold_table[
            "fraction_accurate_among_selected"
        ],
        label=(
            r"Among selected: "
            r"$|\Delta\log_{10}\rho|<0.3$"
        ),
    )

    ax.axvline(
        ADOPTED_PROXY,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=(
            rf"Diagnostic proxy: "
            rf"$\sigma_\rho/\hat{{\rho}}<{ADOPTED_PROXY}$"
        ),
    )

    ax.set_xscale(
        "log"
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_xlabel(
        (
            r"Formal precision threshold "
            r"$\sigma_\rho/\hat{\rho}$"
        )
    )

    ax.set_ylabel(
        "Fraction"
    )

    ax.set_title(
        "How restrictive is a finite-source precision cut?"
    )

    ax.grid(
        alpha=0.20,
    )

    ax.legend(
        frameon=True,
        loc="best",
    )

    # ------------------------------------------------------------------------
    # Panel B
    # ------------------------------------------------------------------------

    ax = axes[
        1
    ]

    shown = (
        recovery_table[
            "n"
        ].to_numpy()
        >= MIN_BIN_COUNT
    )

    sub = recovery_table.loc[
        shown
    ]

    ax.plot(
        sub[
            "precision_center"
        ],
        sub[
            "rho_error_median"
        ],
        marker="o",
        markersize=4.5,
    )

    ax.fill_between(
        sub[
            "precision_center"
        ],
        sub[
            "rho_error_q16"
        ],
        sub[
            "rho_error_q84"
        ],
        alpha=0.20,
    )

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.4,
    )

    ax.axvline(
        ADOPTED_PROXY,
        color="black",
        linestyle="--",
        linewidth=1.4,
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        r"Formal $\sigma_\rho/\hat{\rho}$"
    )

    ax.set_ylabel(
        r"$\log_{10}(\hat{\rho}/\rho_{\rm true})$"
    )

    ax.set_title(
        "Does formal precision imply accurate recovery?"
    )

    ax.grid(
        alpha=0.20,
    )

    fig.suptitle(
        (
            "Finite-source information among "
            "annual-parallax detections"
        ),
        fontsize=21.6,
    )

    fig.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.94,
        ]
    )

    save_figure(
        fig,
        FIGURE_BASE,
    )

    plt.close(
        fig
    )

    # ========================================================================
    # Summary
    # ========================================================================

    summary_lines = [
        "FINITE-SOURCE INFORMATION AMONG ANNUAL-PARALLAX DETECTIONS",
        "=" * 80,
        "",
        f"N parallax detected = {n_total:,}",
        (
            "N with valid formal rho uncertainty + truth recovery = "
            f"{int(valid.sum()):,}"
        ),
        "",
        "Diagnostic proxy only:",
        (
            f"  sigma_rho / rho_hat < "
            f"{ADOPTED_PROXY:.2f}"
        ),
        (
            f"  N passing = "
            f"{n_proxy:,}"
        ),
        (
            f"  fraction of parallax detections = "
            f"{proxy_fraction:.8f}"
        ),
        "",
        "Truth-side audit:",
        (
            f"  accurate rho defined as "
            f"|log10(rho_hat/rho_true)| < "
            f"{ACCURATE_RHO_DEX:.2f} dex"
        ),
        (
            f"  fraction accurate among proxy-selected = "
            f"{proxy_accuracy:.8f}"
        ),
        (
            f"  median log10(rho_hat/rho_true) = "
            f"{proxy_median_rho_error:.8f}"
        ),
        (
            f"  q16--q84 log10(rho_hat/rho_true) = "
            f"[{proxy_q16:.8f}, {proxy_q84:.8f}]"
        ),
        "",
        "Interpretation:",
        (
            "sigma_rho/rho_hat is only a local-covariance precision "
            "diagnostic. It must not be interpreted as a finite-source "
            "detection unless it also corresponds to accurate rho recovery."
        ),
        "",
        (
            "A genuine observational finite-source selection should "
            "ultimately be defined using evidence for FSPL over PSPL, "
            "for example an FSPL-vs-PSPL likelihood-ratio or Delta-chi2 test."
        ),
    ]

    SUMMARY_FILE.write_text(
        "\n".join(
            summary_lines
        )
        + "\n"
    )

    print()
    print(
        "\n".join(
            summary_lines
        )
    )

    print()
    print("Saved tables:")
    print(" ", THRESHOLD_TABLE)
    print(" ", RECOVERY_TABLE)
    print(" ", SUMMARY_FILE)

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
