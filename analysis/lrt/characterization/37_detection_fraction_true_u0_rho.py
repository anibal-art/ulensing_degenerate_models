#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallax detection fraction in the true (|u0|, rho) plane.

NO simulation.
NO fitting.

Scientific question
-------------------
The previous finite-source analysis used

    R_FS = rho_true / |u0_true|.

An apparent increase in parallax detection fraction was seen for
R_FS > 1. However, R_FS mixes two physical quantities:

    - source-size parameter rho;
    - impact parameter |u0|.

This script separates them directly by measuring the parallax
detection fraction in the two-dimensional plane

    (|u0_true|, rho_true).

The strictly positive-LRT H1 population is used, consistently with
the previous characterization:

    detected:
        positive_lrt_class == "parallax_detected"

    confused:
        positive_lrt_class == "confused_fspl_no_parallax"

The line

    rho_true = |u0_true|

corresponds to

    R_FS = 1.

Points above this line are finite-source dominated / source-transit
events according to the adopted geometric criterion.

Outputs
-------
Figure:
analysis/lrt/figures/characterization/
    37_detection_fraction_true_u0_rho/
    detection_fraction_true_u0_rho.png

Cell table:
analysis/lrt/results/characterization/
    37_detection_fraction_true_u0_rho/
    detection_fraction_true_u0_rho_cells.csv

Region summary:
analysis/lrt/results/characterization/
    37_detection_fraction_true_u0_rho/
    detection_fraction_true_u0_rho_regions.csv

Text audit:
analysis/lrt/results/characterization/
    37_detection_fraction_true_u0_rho/
    detection_fraction_true_u0_rho_audit.txt
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
from matplotlib.colors import LogNorm


# ============================================================================
# Configuration
# ============================================================================

INPUT_PATH = Path(
    "analysis/lrt/results/characterization/18_positive_truth/"
    "h1_positive_truth_characterization.parquet"
)

FIGURE_DIR = Path(
    "analysis/lrt/figures/characterization/"
    "37_detection_fraction_true_u0_rho"
)

RESULTS_DIR = Path(
    "analysis/lrt/results/characterization/"
    "37_detection_fraction_true_u0_rho"
)

FIGURE_PATH = (
    FIGURE_DIR
    / "detection_fraction_true_u0_rho.png"
)

CELL_TABLE_PATH = (
    RESULTS_DIR
    / "detection_fraction_true_u0_rho_cells.csv"
)

REGION_TABLE_PATH = (
    RESULTS_DIR
    / "detection_fraction_true_u0_rho_regions.csv"
)

AUDIT_PATH = (
    RESULTS_DIR
    / "detection_fraction_true_u0_rho_audit.txt"
)


DETECTED_CLASS = "parallax_detected"
CONFUSED_CLASS = "confused_fspl_no_parallax"

# Enough resolution to separate rho and u0 without making the
# source-transit region impossibly sparse.
N_U0_BINS = 16
N_RHO_BINS = 16

# Same philosophy as the previous detection map:
# do not interpret a cell with fewer than this many events.
MIN_CELL_COUNT = 30


# ============================================================================
# Helpers
# ============================================================================

def wilson_interval(k, n, z=1.959963984540054):
    """
    Wilson 95% confidence interval for a binomial fraction.
    """

    if n <= 0:
        return np.nan, np.nan

    p = k / n

    denom = 1.0 + z**2 / n

    center = (
        p
        + z**2 / (2.0 * n)
    ) / denom

    half = (
        z
        / denom
        * np.sqrt(
            p * (1.0 - p) / n
            + z**2 / (4.0 * n**2)
        )
    )

    return (
        center - half,
        center + half,
    )


def build_log_edges(values, n_bins):
    """
    Logarithmic edges spanning complete positive finite range.
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
            "No positive finite values available for logarithmic binning."
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


def summarize_region(
    df,
    label,
):
    """
    Summarize one physical region.
    """

    n = len(df)

    if n == 0:
        return {
            "region": label,
            "n_total": 0,
            "n_detected": 0,
            "n_confused": 0,
            "detection_fraction": np.nan,
            "wilson95_low": np.nan,
            "wilson95_high": np.nan,
            "median_abs_u0_true": np.nan,
            "median_rho_true": np.nan,
            "median_rfs_true": np.nan,
            "median_tE_true_days": np.nan,
            "median_piE_true_amp": np.nan,
        }

    n_detected = int(
        (
            df["positive_lrt_class"]
            == DETECTED_CLASS
        ).sum()
    )

    n_confused = int(
        (
            df["positive_lrt_class"]
            == CONFUSED_CLASS
        ).sum()
    )

    frac = (
        n_detected / n
    )

    lo, hi = wilson_interval(
        n_detected,
        n,
    )

    return {
        "region":
            label,

        "n_total":
            n,

        "n_detected":
            n_detected,

        "n_confused":
            n_confused,

        "detection_fraction":
            frac,

        "wilson95_low":
            lo,

        "wilson95_high":
            hi,

        "median_abs_u0_true":
            float(
                np.nanmedian(
                    df["abs_u0_true"]
                )
            ),

        "median_rho_true":
            float(
                np.nanmedian(
                    df["rho_true"]
                )
            ),

        "median_rfs_true":
            float(
                np.nanmedian(
                    df[
                        "rho_over_abs_u0_true"
                    ]
                )
            ),

        "median_tE_true_days":
            float(
                np.nanmedian(
                    df["tE_true_days"]
                )
            ),

        "median_piE_true_amp":
            float(
                np.nanmedian(
                    df["piE_true_amp"]
                )
            ),
    }


# ============================================================================
# Main
# ============================================================================

def main():

    print("=" * 100)
    print("37: PARALLAX DETECTION FRACTION IN TRUE (|u0|, rho) PLANE")
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
    # Required columns
    # ----------------------------------------------------------------------

    required = [
        "catalog_row",
        "positive_lrt_class",
        "delta_chi2_lrt",

        "u0_true",
        "rho_true",
        "rho_over_abs_u0_true",

        "tE_true_days",
        "piE_true_amp",
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
    # Positive-LRT characterization should contain exactly these classes.
    # Keep the filtering explicit.
    # ----------------------------------------------------------------------

    class_mask = df[
        "positive_lrt_class"
    ].isin(
        [
            DETECTED_CLASS,
            CONFUSED_CLASS,
        ]
    )

    work = df.loc[
        class_mask
    ].copy()

    work[
        "abs_u0_true"
    ] = np.abs(
        work[
            "u0_true"
        ].to_numpy(dtype=float)
    )

    valid = (
        np.isfinite(
            work["abs_u0_true"]
        )
        & (
            work["abs_u0_true"]
            > 0.0
        )
        & np.isfinite(
            work["rho_true"]
        )
        & (
            work["rho_true"]
            > 0.0
        )
        & np.isfinite(
            work[
                "rho_over_abs_u0_true"
            ]
        )
        & (
            work[
                "rho_over_abs_u0_true"
            ]
            > 0.0
        )
    )

    work = work.loc[
        valid
    ].copy()

    print()
    print(
        "Rows with valid positive |u0| and rho =",
        f"{len(work):,}",
    )

    print()
    print("LRT classes:")
    print(
        work[
            "positive_lrt_class"
        ].value_counts()
    )

    # ----------------------------------------------------------------------
    # Verify R_FS reconstruction
    # ----------------------------------------------------------------------

    rfs_rebuilt = (
        work["rho_true"].to_numpy(dtype=float)
        / work["abs_u0_true"].to_numpy(dtype=float)
    )

    rfs_stored = work[
        "rho_over_abs_u0_true"
    ].to_numpy(dtype=float)

    relative_difference = np.abs(
        rfs_rebuilt
        - rfs_stored
    ) / rfs_stored

    max_rfs_relative_difference = float(
        np.nanmax(
            relative_difference
        )
    )

    print()
    print(
        "max relative difference in reconstructed R_FS =",
        max_rfs_relative_difference,
    )

    # ----------------------------------------------------------------------
    # Global detection fraction
    # ----------------------------------------------------------------------

    detected = (
        work[
            "positive_lrt_class"
        ]
        == DETECTED_CLASS
    ).to_numpy(dtype=bool)

    overall_detection_fraction = float(
        detected.mean()
    )

    print()
    print(
        "Overall positive-T detection fraction =",
        overall_detection_fraction,
    )

    # ----------------------------------------------------------------------
    # Region summaries: R_FS < 1 versus > 1
    # ----------------------------------------------------------------------

    below = work.loc[
        work[
            "rho_over_abs_u0_true"
        ] < 1.0
    ].copy()

    above = work.loc[
        work[
            "rho_over_abs_u0_true"
        ] > 1.0
    ].copy()

    region_rows = [
        summarize_region(
            work,
            "all_positive_T",
        ),

        summarize_region(
            below,
            "RFS_lt_1",
        ),

        summarize_region(
            above,
            "RFS_gt_1",
        ),
    ]

    regions = pd.DataFrame(
        region_rows
    )

    regions.to_csv(
        REGION_TABLE_PATH,
        index=False,
    )

    print()
    print("=" * 100)
    print("REGION SUMMARY")
    print("=" * 100)

    print(
        regions.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------------------
    # Additional detected/confused summaries specifically for R_FS > 1
    # ----------------------------------------------------------------------

    print()
    print("=" * 100)
    print("TRUE R_FS > 1: DETECTED VERSUS CONFUSED")
    print("=" * 100)

    for cls in [
        DETECTED_CLASS,
        CONFUSED_CLASS,
    ]:

        sub = above.loc[
            above[
                "positive_lrt_class"
            ] == cls
        ]

        print()
        print(cls)
        print("-" * 70)
        print("N =", f"{len(sub):,}")

        if len(sub) == 0:
            continue

        for column in [
            "abs_u0_true",
            "rho_true",
            "rho_over_abs_u0_true",
            "tE_true_days",
            "piE_true_amp",
        ]:

            arr = sub[
                column
            ].to_numpy(dtype=float)

            arr = arr[
                np.isfinite(arr)
            ]

            if len(arr) == 0:
                continue

            q16, med, q84 = np.quantile(
                arr,
                [0.16, 0.50, 0.84],
            )

            print(
                f"{column:28s} "
                f"q16={q16:.6g} "
                f"median={med:.6g} "
                f"q84={q84:.6g}"
            )

    # ----------------------------------------------------------------------
    # Build logarithmic 2D grid
    # ----------------------------------------------------------------------

    u0_edges = build_log_edges(
        work["abs_u0_true"],
        N_U0_BINS,
    )

    rho_edges = build_log_edges(
        work["rho_true"],
        N_RHO_BINS,
    )

    u0 = work[
        "abs_u0_true"
    ].to_numpy(dtype=float)

    rho = work[
        "rho_true"
    ].to_numpy(dtype=float)

    # histogram2d returns shape:
    #
    #   (N_U0_BINS, N_RHO_BINS)
    #
    # We transpose below for pcolormesh:
    #
    #   rows -> rho
    #   columns -> |u0|
    count_all, _, _ = np.histogram2d(
        u0,
        rho,
        bins=[
            u0_edges,
            rho_edges,
        ],
    )

    count_detected, _, _ = np.histogram2d(
        u0[detected],
        rho[detected],
        bins=[
            u0_edges,
            rho_edges,
        ],
    )

    count_confused, _, _ = np.histogram2d(
        u0[~detected],
        rho[~detected],
        bins=[
            u0_edges,
            rho_edges,
        ],
    )

    # Transpose to [rho_bin, u0_bin].
    count_all = count_all.T
    count_detected = count_detected.T
    count_confused = count_confused.T

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        detection_fraction = (
            count_detected
            / count_all
        )

    detection_fraction[
        count_all == 0
    ] = np.nan

    # ----------------------------------------------------------------------
    # Save one row per 2D cell
    # ----------------------------------------------------------------------

    cell_rows = []

    for irho in range(
        len(rho_edges) - 1
    ):

        for iu0 in range(
            len(u0_edges) - 1
        ):

            n = int(
                count_all[
                    irho,
                    iu0,
                ]
            )

            k = int(
                count_detected[
                    irho,
                    iu0,
                ]
            )

            nc = int(
                count_confused[
                    irho,
                    iu0,
                ]
            )

            if n > 0:
                frac = k / n
                lo, hi = wilson_interval(
                    k,
                    n,
                )
            else:
                frac = np.nan
                lo = np.nan
                hi = np.nan

            u0_left = float(
                u0_edges[iu0]
            )

            u0_right = float(
                u0_edges[iu0 + 1]
            )

            rho_low = float(
                rho_edges[irho]
            )

            rho_high = float(
                rho_edges[irho + 1]
            )

            u0_center = np.sqrt(
                u0_left
                * u0_right
            )

            rho_center = np.sqrt(
                rho_low
                * rho_high
            )

            cell_rows.append(
                {
                    "u0_left":
                        u0_left,

                    "u0_right":
                        u0_right,

                    "u0_center":
                        u0_center,

                    "rho_low":
                        rho_low,

                    "rho_high":
                        rho_high,

                    "rho_center":
                        rho_center,

                    "cell_center_RFS":
                        rho_center
                        / u0_center,

                    "n_total":
                        n,

                    "n_detected":
                        k,

                    "n_confused":
                        nc,

                    "detection_fraction":
                        frac,

                    "wilson95_low":
                        lo,

                    "wilson95_high":
                        hi,

                    "shown_in_detection_map":
                        n >= MIN_CELL_COUNT,
                }
            )

    cells = pd.DataFrame(
        cell_rows
    )

    cells.to_csv(
        CELL_TABLE_PATH,
        index=False,
    )

    n_occupied = int(
        (
            cells["n_total"]
            > 0
        ).sum()
    )

    n_shown = int(
        cells[
            "shown_in_detection_map"
        ].sum()
    )

    shown_counts = cells.loc[
        cells[
            "shown_in_detection_map"
        ],
        "n_total",
    ]

    if len(shown_counts) > 0:
        min_shown_count = int(
            shown_counts.min()
        )
    else:
        min_shown_count = 0

    print()
    print("=" * 100)
    print("2D GRID AUDIT")
    print("=" * 100)

    print(
        "grid =",
        f"{N_U0_BINS} x {N_RHO_BINS}",
    )

    print(
        "occupied cells =",
        n_occupied,
    )

    print(
        f"cells with N >= {MIN_CELL_COUNT} =",
        n_shown,
    )

    print(
        "minimum count among displayed cells =",
        min_shown_count,
    )

    # ----------------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.6),
        sharex=True,
        sharey=True,
    )

    # ------------------------------------------------------------------
    # Left: detection fraction
    # ------------------------------------------------------------------

    detection_for_plot = np.ma.masked_where(
        (count_all < MIN_CELL_COUNT)
        | ~np.isfinite(
            detection_fraction
        ),
        detection_fraction,
    )

    mesh = axes[0].pcolormesh(
        u0_edges,
        rho_edges,
        detection_for_plot,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    cbar = fig.colorbar(
        mesh,
        ax=axes[0],
        pad=0.02,
    )

    cbar.set_label(
        "Parallax detection fraction"
    )

    axes[0].set_title(
        "Detection fraction"
    )

    # ------------------------------------------------------------------
    # Right: population count
    # ------------------------------------------------------------------

    count_for_plot = np.ma.masked_where(
        count_all <= 0,
        count_all,
    )

    positive_counts = count_all[
        count_all > 0
    ]

    if len(positive_counts) == 0:
        raise RuntimeError(
            "No occupied cells available for count map."
        )

    count_mesh = axes[1].pcolormesh(
        u0_edges,
        rho_edges,
        count_for_plot,
        shading="auto",
        cmap="magma",
        norm=LogNorm(
            vmin=max(
                1.0,
                float(
                    positive_counts.min()
                ),
            ),
            vmax=float(
                positive_counts.max()
            ),
        ),
    )

    cbar_count = fig.colorbar(
        count_mesh,
        ax=axes[1],
        pad=0.02,
    )

    cbar_count.set_label(
        "Events per cell"
    )

    axes[1].set_title(
        "Population density"
    )

    # ------------------------------------------------------------------
    # R_FS = 1 diagonal
    # ------------------------------------------------------------------

    x_min = max(
        u0_edges[0],
        rho_edges[0],
    )

    x_max = min(
        u0_edges[-1],
        rho_edges[-1],
    )

    if x_min < x_max:
        line_x = np.logspace(
            np.log10(x_min),
            np.log10(x_max),
            300,
        )

        # ========================================================
        # NEW BLOCK: high-contrast R_FS = 1 boundary
        # ========================================================
        #
        # Draw a broad white underlay plus a narrower black dashed
        # line.  This keeps the boundary visible over both dark
        # heatmap cells and white / masked regions.

        for ax in axes:

            ax.plot(
                line_x,
                line_x,
                linestyle="-",
                linewidth=3.4,
                color="white",
                alpha=0.95,
                zorder=5,
            )

            ax.plot(
                line_x,
                line_x,
                linestyle="--",
                linewidth=1.6,
                color="black",
                zorder=6,
                label=r"$R_{\rm FS}=1$",
            )

    # ------------------------------------------------------------------
    # Common formatting
    # ------------------------------------------------------------------

    for ax in axes:

        ax.set_xscale(
            "log"
        )

        ax.set_yscale(
            "log"
        )

        ax.set_xlabel(
            r"True $|u_0|$"
        )

        ax.grid(
            alpha=0.18,
        )


    axes[0].set_ylabel(
        r"True $\rho$"
    )

    axes[0].legend(
        loc="lower right",
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
        "37: PARALLAX DETECTION FRACTION IN TRUE (|u0|, rho) PLANE",
        "=" * 100,
        "",
        f"Input rows: {len(df):,}",
        f"Rows used: {len(work):,}",
        "",
        (
            "Positive-T detection fraction: "
            f"{overall_detection_fraction:.6f}"
        ),
        "",
        (
            "max relative difference in reconstructed R_FS: "
            f"{max_rfs_relative_difference:.6e}"
        ),
        "",
        f"Grid: {N_U0_BINS} x {N_RHO_BINS}",
        f"Occupied cells: {n_occupied}",
        (
            f"Cells with N >= {MIN_CELL_COUNT}: "
            f"{n_shown}"
        ),
        (
            "Minimum count among displayed cells: "
            f"{min_shown_count}"
        ),
        "",
        "Interpretation guide:",
        "",
        (
            "A predominantly vertical gradient at fixed |u0| would "
            "indicate a rho-driven association."
        ),
        "",
        (
            "A predominantly horizontal gradient at fixed rho would "
            "indicate a |u0|-driven association."
        ),
        "",
        (
            "Structure organized approximately parallel to the "
            "rho=|u0| boundary would support R_FS itself as a useful "
            "organizing variable."
        ),
        "",
        (
            "This is still a descriptive conditional map, not a causal "
            "adjustment for tE, |piE|, sampling, or signal-to-noise."
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
        "Cell table:",
        CELL_TABLE_PATH,
    )

    print(
        "Region summary:",
        REGION_TABLE_PATH,
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


