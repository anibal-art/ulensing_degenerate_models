#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Annual-parallax distinguishability as a function of true lens mass.

NO simulation.
NO fitting.

Scientific question
-------------------
Does the empirical LRT selection for annual parallax induce a
selection in true lens mass?

For the full final H1-generated population, define

    detected:
        Delta chi2_LRT >= c_alpha

    hidden:
        Delta chi2_LRT < c_alpha

with the primary empirical threshold

    alpha = 1e-3
    c_alpha = 13.15982011480088.

The figure contains two panels:

1. Normalized true lens-mass distributions for detected and hidden
   parallax events.

2. Detection fraction

       P(parallax detected | M_L,true)

   in logarithmic true-mass bins, with Wilson binomial intervals.

The first panel is descriptive and still contains the intrinsic
population mass distribution.

The second panel isolates the mass-dependent LRT selection function
within the final H1 population.

Inputs
------
analysis/lrt/data/h1_lrt_results_20260920.parquet

Parallax_LSST/data_sedighe/LSSTMONTS.dat
Parallax_LSST/data_sedighe/columns

Outputs
-------
analysis/lrt/figures/characterization/
    43_parallax_detectability_vs_lens_mass/
    parallax_detectability_vs_true_lens_mass.png
    parallax_detectability_vs_true_lens_mass.pdf

analysis/lrt/results/characterization/
    43_parallax_detectability_vs_lens_mass/
    parallax_detectability_vs_true_lens_mass_bins.csv
    parallax_detectability_vs_true_lens_mass_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parents[3]

H1_INPUT = (
    ROOT
    / "analysis/lrt/data"
    / "h1_lrt_results_20260920.parquet"
)

CATALOG_DIR = (
    ROOT
    / "Parallax_LSST/data_sedighe"
)

CATALOG_COLUMNS = (
    CATALOG_DIR
    / "columns"
)

CATALOG_DATA = (
    CATALOG_DIR
    / "LSSTMONTS.dat"
)

FIGURE_DIR = (
    ROOT
    / "analysis/lrt/figures/characterization"
    / "43_parallax_detectability_vs_lens_mass"
)

RESULT_DIR = (
    ROOT
    / "analysis/lrt/results/characterization"
    / "43_parallax_detectability_vs_lens_mass"
)

FIGURE_BASE = (
    FIGURE_DIR
    / "parallax_detectability_vs_true_lens_mass"
)

BIN_TABLE = (
    RESULT_DIR
    / "parallax_detectability_vs_true_lens_mass_bins.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "parallax_detectability_vs_true_lens_mass_summary.txt"
)


# ============================================================================
# Frozen primary LRT calibration
# ============================================================================

ALPHA = 1.0e-3

C_ALPHA = 13.15982011480088

EXPECTED_H1_N = 333653
EXPECTED_DETECTED_N = 229921

N_BINS = 24

MIN_BIN_COUNT = 30


# ============================================================================
# Catalog definition
# ============================================================================

MASS_COLUMN_CANDIDATES = [
    "Lens_mass (solar mass)",
    "lens_mass_msun",
]


# ============================================================================
# Plot style
# ============================================================================

def set_paper_style():

    plt.rcParams.update(
        {
            "font.size": 16,

            "axes.labelsize": 19,
            "axes.titlesize": 18,

            "xtick.labelsize": 15,
            "ytick.labelsize": 15,

            "legend.fontsize": 13,

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

def find_column(
    columns,
    candidates,
):

    for candidate in candidates:

        if candidate in columns:

            return candidate

    raise RuntimeError(
        "Could not identify lens-mass column.\n"
        f"Tried: {candidates}\n"
        f"Available columns:\n{columns}"
    )


def load_true_lens_mass(
    required_rows,
):
    """
    Load the true lens mass directly from LSSTMONTS.dat.

    catalog_row is the exact zero-based physical row number in the
    original catalog.
    """

    if not CATALOG_COLUMNS.exists():

        raise FileNotFoundError(
            CATALOG_COLUMNS
        )

    if not CATALOG_DATA.exists():

        raise FileNotFoundError(
            CATALOG_DATA
        )

    columns = [
        line.strip()
        for line in CATALOG_COLUMNS.read_text().splitlines()
        if line.strip()
    ]

    mass_column = find_column(
        columns,
        MASS_COLUMN_CANDIDATES,
    )

    print(
        "Catalog lens-mass column:",
        mass_column,
    )

    # Read only the required physical quantity.
    catalog_mass = pd.read_csv(
        CATALOG_DATA,
        sep=r"\s+",
        header=None,
        names=columns,
        usecols=[
            mass_column,
        ],
    )

    catalog_mass = catalog_mass.rename(
        columns={
            mass_column:
                "lens_mass_true_msun"
        }
    )

    catalog_mass.insert(
        0,
        "catalog_row",
        np.arange(
            len(catalog_mass),
            dtype=np.int64,
        ),
    )

    required_rows = np.asarray(
        required_rows,
        dtype=np.int64,
    )

    if required_rows.size == 0:

        raise RuntimeError(
            "No catalog rows requested."
        )

    if np.min(required_rows) < 0:

        raise RuntimeError(
            "Negative catalog_row found."
        )

    if np.max(required_rows) >= len(
        catalog_mass
    ):

        raise RuntimeError(
            "catalog_row exceeds catalog length."
        )

    unique_rows = np.unique(
        required_rows
    )

    physical = (
        catalog_mass
        .iloc[
            unique_rows
        ]
        .copy()
    )

    if not np.array_equal(
        physical[
            "catalog_row"
        ].to_numpy(dtype=np.int64),
        unique_rows,
    ):

        raise RuntimeError(
            "catalog_row identity audit failed."
        )

    return physical


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

    half_width = (
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
        center - half_width,
        center + half_width,
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
            "No finite positive lens masses."
        )

    m_min = float(
        np.nanmin(
            values[
                valid
            ]
        )
    )

    m_max = float(
        np.nanmax(
            values[
                valid
            ]
        )
    )

    edges = np.geomspace(
        m_min,
        m_max,
        n_bins + 1,
    )

    edges[
        0
    ] = np.nextafter(
        m_min,
        -np.inf,
    )

    edges[
        -1
    ] = np.nextafter(
        m_max,
        np.inf,
    )

    return edges


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

    if not H1_INPUT.exists():

        raise FileNotFoundError(
            H1_INPUT
        )

    # ========================================================================
    # Load full final H1 population
    # ========================================================================

    h1 = pd.read_parquet(
        H1_INPUT,
        columns=[
            "catalog_row",
            "delta_chi2_lrt",
        ],
    )

    if not h1[
        "catalog_row"
    ].is_unique:

        raise RuntimeError(
            "catalog_row is not unique in H1 dataset."
        )

    if len(
        h1
    ) != EXPECTED_H1_N:

        raise RuntimeError(
            "Unexpected H1 sample size: "
            f"{len(h1):,} != {EXPECTED_H1_N:,}"
        )

    # ========================================================================
    # Join true lens mass
    # ========================================================================

    physical = load_true_lens_mass(
        h1[
            "catalog_row"
        ].to_numpy(dtype=np.int64)
    )

    df = h1.merge(
        physical,
        on="catalog_row",
        how="left",
        validate="one_to_one",
    )

    if len(
        df
    ) != len(
        h1
    ):

        raise RuntimeError(
            "Physical join changed the H1 sample size."
        )

    if df[
        "lens_mass_true_msun"
    ].isna().any():

        raise RuntimeError(
            "Missing true lens masses after catalog join."
        )

    # ========================================================================
    # Primary LRT classification
    # ========================================================================

    T = pd.to_numeric(
        df[
            "delta_chi2_lrt"
        ],
        errors="coerce",
    ).to_numpy(dtype=float)

    mass = pd.to_numeric(
        df[
            "lens_mass_true_msun"
        ],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(T)
        & np.isfinite(mass)
        & (
            mass > 0.0
        )
    )

    detected = (
        valid
        & (
            T >= C_ALPHA
        )
    )

    hidden = (
        valid
        & (
            T < C_ALPHA
        )
    )

    n_detected = int(
        detected.sum()
    )

    n_hidden = int(
        hidden.sum()
    )

    if n_detected != EXPECTED_DETECTED_N:

        raise RuntimeError(
            "Primary detection-count audit failed: "
            f"{n_detected:,} != {EXPECTED_DETECTED_N:,}"
        )

    if (
        n_detected
        + n_hidden
        != int(
            valid.sum()
        )
    ):

        raise RuntimeError(
            "Detected + hidden does not recover valid H1 population."
        )

    # ========================================================================
    # Common logarithmic true-mass bins
    # ========================================================================

    edges = build_log_edges(
        mass[
            valid
        ],
        N_BINS,
    )

    centers = np.sqrt(
        edges[
            :-1
        ]
        * edges[
            1:
        ]
    )

    total_counts, _ = np.histogram(
        mass[
            valid
        ],
        bins=edges,
    )

    detected_counts, _ = np.histogram(
        mass[
            detected
        ],
        bins=edges,
    )

    hidden_counts, _ = np.histogram(
        mass[
            hidden
        ],
        bins=edges,
    )

    # ========================================================================
    # Panel 1: normalized mass distributions
    # ========================================================================
    #
    # Equal logarithmic bins are used. Therefore the quantity plotted
    # here is the fraction of each class in each log-mass bin rather
    # than a linear-mass density.
    # ========================================================================

    detected_distribution = (
        detected_counts
        / detected_counts.sum()
    )

    hidden_distribution = (
        hidden_counts
        / hidden_counts.sum()
    )

    # ========================================================================
    # Panel 2: detection probability conditional on true mass
    # ========================================================================

    detection_fraction = np.full(
        N_BINS,
        np.nan,
        dtype=float,
    )

    wilson_low = np.full(
        N_BINS,
        np.nan,
        dtype=float,
    )

    wilson_high = np.full(
        N_BINS,
        np.nan,
        dtype=float,
    )

    for i in range(
        N_BINS
    ):

        n = int(
            total_counts[
                i
            ]
        )

        k = int(
            detected_counts[
                i
            ]
        )

        if n <= 0:

            continue

        detection_fraction[
            i
        ] = (
            k / n
        )

        (
            wilson_low[
                i
            ],
            wilson_high[
                i
            ],
        ) = wilson_interval(
            k,
            n,
        )

    # ========================================================================
    # Save numerical table
    # ========================================================================

    table = pd.DataFrame(
        {
            "mass_left_msun":
                edges[
                    :-1
                ],

            "mass_right_msun":
                edges[
                    1:
                ],

            "mass_center_msun":
                centers,

            "n_total":
                total_counts,

            "n_detected":
                detected_counts,

            "n_hidden":
                hidden_counts,

            "detected_distribution_fraction":
                detected_distribution,

            "hidden_distribution_fraction":
                hidden_distribution,

            "detection_fraction":
                detection_fraction,

            "wilson_low":
                wilson_low,

            "wilson_high":
                wilson_high,
        }
    )

    table.to_csv(
        BIN_TABLE,
        index=False,
    )

    # ========================================================================
    # Figure
    # ========================================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            8.0,
            8.0,
        ),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1.0,
                1.15,
            ],
        },
    )

    # ------------------------------------------------------------------------
    # Top: normalized distributions
    # ------------------------------------------------------------------------

    axes[
        0
    ].step(
        centers,
        detected_distribution,
        where="mid",
        linewidth=2.0,
        label=(
            "Parallax detected "
            f"($N={n_detected:,}$)"
        ),
    )

    axes[
        0
    ].step(
        centers,
        hidden_distribution,
        where="mid",
        linewidth=2.0,
        label=(
            "Parallax hidden "
            f"($N={n_hidden:,}$)"
        ),
    )

    axes[
        0
    ].set_ylabel(
        "Fraction per log-mass bin"
    )

    axes[
        0
    ].set_title(
        "True lens-mass distributions after the annual-parallax LRT"
    )

    axes[
        0
    ].legend(
        frameon=True,
    )

    axes[
        0
    ].grid(
        alpha=0.20,
    )

    # ------------------------------------------------------------------------
    # Bottom: detection fraction
    # ------------------------------------------------------------------------

    shown = (
        total_counts
        >= MIN_BIN_COUNT
    )

    axes[
        1
    ].plot(
        centers[
            shown
        ],
        detection_fraction[
            shown
        ],
        marker="o",
        markersize=5,
        linewidth=2.0,
    )

    axes[
        1
    ].fill_between(
        centers[
            shown
        ],
        wilson_low[
            shown
        ],
        wilson_high[
            shown
        ],
        alpha=0.20,
    )

    axes[
        1
    ].axhline(
        n_detected
        / int(
            valid.sum()
        ),
        linestyle="--",
        linewidth=1.4,
        color="black",
        label=(
            "Population-average "
            r"$P_{\rm det}$"
        ),
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
        r"True lens mass $M_{L,\rm true}\ [M_\odot]$"
    )

    axes[
        1
    ].set_ylabel(
        (
            r"$P(\mathrm{parallax\ detected}"
            r"\mid M_{L,\rm true})$"
        )
    )

    axes[
        1
    ].set_title(
        "Annual-parallax distinguishability versus true lens mass"
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

    png = FIGURE_BASE.with_suffix(
        ".png"
    )

    pdf = FIGURE_BASE.with_suffix(
        ".pdf"
    )

    fig.savefig(
        png
    )

    fig.savefig(
        pdf
    )

    plt.close(
        fig
    )

    # ========================================================================
    # Summary
    # ========================================================================

    overall_power = (
        n_detected
        / int(
            valid.sum()
        )
    )

    detected_mass = mass[
        detected
    ]

    hidden_mass = mass[
        hidden
    ]

    lines = [
        "PARALLAX DISTINGUISHABILITY VS TRUE LENS MASS",
        "=" * 80,
        "",
        f"alpha = {ALPHA:g}",
        f"c_alpha = {C_ALPHA:.12f}",
        "",
        f"H1 valid events = {int(valid.sum()):,}",
        f"detected = {n_detected:,}",
        f"hidden = {n_hidden:,}",
        f"overall detection fraction = {overall_power:.6f}",
        "",
        (
            "detected median true mass [Msun] = "
            f"{np.nanmedian(detected_mass):.6g}"
        ),
        (
            "detected 16--84% true mass [Msun] = "
            f"[{np.nanpercentile(detected_mass, 16):.6g}, "
            f"{np.nanpercentile(detected_mass, 84):.6g}]"
        ),
        "",
        (
            "hidden median true mass [Msun] = "
            f"{np.nanmedian(hidden_mass):.6g}"
        ),
        (
            "hidden 16--84% true mass [Msun] = "
            f"[{np.nanpercentile(hidden_mass, 16):.6g}, "
            f"{np.nanpercentile(hidden_mass, 84):.6g}]"
        ),
        "",
        (
            "Interpretation: the normalized distributions show where the "
            "detected and hidden populations lie in true mass, whereas "
            "P(detected | M_true) isolates the mass-dependent selection "
            "induced by the annual-parallax LRT."
        ),
    ]

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
        png,
    )

    print(
        "Saved:",
        pdf,
    )

    print(
        "Saved:",
        BIN_TABLE,
    )

    print(
        "Saved:",
        SUMMARY_FILE,
    )


if __name__ == "__main__":
    main()
