#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test of the constant-acceleration scaling for annual-parallax detectability.

NO simulation.
NO fitting.

Physical motivation
-------------------
For events short compared with one year, Earth's orbital motion can be
approximated locally by a constant acceleration.

The parallax perturbation then scales approximately as

    delta u_pi ~ pi_E (Omega_earth t_E)^2,

up to a geometrical factor depending on event orientation, ecliptic
position and season.

Ignoring the constant factor Omega_earth^2, define

    X_pi = |pi_E,true| (t_E,true / 1 yr)^2.

Using

    theta_E = sqrt(kappa M pi_rel),
    t_E     = theta_E / mu_rel,
    pi_E    = pi_rel / theta_E,

one obtains

    pi_E t_E^2
        = sqrt(kappa) M^(1/2) pi_rel^(3/2) / mu_rel^2.

Therefore X_pi is a physically motivated candidate variable organizing
annual-parallax detectability in the short-event regime.

This script performs three tests.

TEST 1
------
Overlay X_pi = constant curves on the existing
(t_E,true, |pi_E,true|) detection map.

In log-log space,

    log |pi_E| = -2 log t_E + constant,

so constant-X_pi curves have slope -2.

TEST 2
------
Plot

    P(parallax detected | X_pi, t_E bin)

for several true-timescale bins.

If the scaling captures the dominant short-event physics, the curves
for short t_E should approximately collapse onto one another.

Their systematic separation at long t_E identifies where the local
constant-acceleration approximation / observing-window approximation
breaks down.

TEST 3
------
Since, approximately,

    Delta chi2 ~ (delta u_pi)^2,

the perturbative expectation is

    Delta chi2 ~ X_pi^2,

or

    log Delta chi2 ~ 2 log X_pi + constant.

We bin Delta chi2 in X_pi for each t_E range and estimate the slope.
The expected slope ~2 is only an approximate population-level scaling:
u0, source flux, cadence, geometry and season are not controlled here.

Population
----------
The input characterization table contains only the strict-positive
H1 population:

    Delta chi2_LRT > 0.

Therefore detection fractions in this script are conditional on that
strict-positive population, exactly as in the existing 2D detection map.

Input
-----
analysis/lrt/results/characterization/18_positive_truth/
    h1_positive_truth_characterization.parquet

Outputs
-------
analysis/lrt/figures/characterization/
    47_parallax_acceleration_scaling/

    01_detection_map_with_Xpi_lines.png/pdf
    02_detection_collapse_vs_Xpi.png/pdf
    03_delta_chi2_scaling_vs_Xpi.png/pdf

analysis/lrt/results/characterization/
    47_parallax_acceleration_scaling/

    01_detection_map_table.csv
    02_detection_collapse_table.csv
    03_delta_chi2_scaling_table.csv
    03_delta_chi2_slope_fits.csv
    parallax_acceleration_scaling_summary.txt
"""

from pathlib import Path

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
    / "47_parallax_acceleration_scaling"
)

RESULT_DIR = (
    ROOT
    / "analysis/lrt/results/characterization"
    / "47_parallax_acceleration_scaling"
)

FIGURE_MAP = (
    FIGURE_DIR
    / "01_detection_map_with_Xpi_lines"
)

FIGURE_COLLAPSE = (
    FIGURE_DIR
    / "02_detection_collapse_vs_Xpi"
)

FIGURE_T = (
    FIGURE_DIR
    / "03_delta_chi2_scaling_vs_Xpi"
)

MAP_TABLE = (
    RESULT_DIR
    / "01_detection_map_table.csv"
)

COLLAPSE_TABLE = (
    RESULT_DIR
    / "02_detection_collapse_table.csv"
)

T_TABLE = (
    RESULT_DIR
    / "03_delta_chi2_scaling_table.csv"
)

SLOPE_TABLE = (
    RESULT_DIR
    / "03_delta_chi2_slope_fits.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "parallax_acceleration_scaling_summary.txt"
)


# ============================================================================
# Frozen LRT configuration
# ============================================================================

PRIMARY_THRESHOLD = 13.15982011480088

EXPECTED_POSITIVE_N = 330668
EXPECTED_DETECTED_N = 229921
EXPECTED_CONFUSED_N = 100747

YEAR_DAYS = 365.25


# ============================================================================
# Binning configuration
# ============================================================================

# Existing 2D-map resolution.
N_TE_MAP_BINS = 9
N_PIE_MAP_BINS = 9
MIN_MAP_CELL_N = 30

# True-timescale ranges used in the collapse/scaling tests.
#
# The separation around 100--200 d is deliberate:
# this is approximately where we want to test whether the
# constant-acceleration approximation begins to fail.
TE_BIN_EDGES = np.array(
    [
        0.0,
        10.0,
        30.0,
        60.0,
        100.0,
        200.0,
        365.25,
        np.inf,
    ],
    dtype=float,
)

TE_BIN_LABELS = [
    r"$t_E<10$ d",
    r"$10\leq t_E<30$ d",
    r"$30\leq t_E<60$ d",
    r"$60\leq t_E<100$ d",
    r"$100\leq t_E<200$ d",
    r"$200\leq t_E<365$ d",
    r"$t_E\geq365$ d",
]

N_X_BINS = 22
MIN_X_BIN_N = 30

# For the approximate slope test:
#
# Avoid using bins whose median T is extremely close to the truncation
# at T=0, and avoid very extreme high-T bins where a simple perturbative
# scaling is less informative.
T_FIT_MIN = 1.0
T_FIT_MAX = 1.0e4

MIN_SLOPE_POINTS = 4


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
            "lines.linewidth": 1.9,

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


def save_figure(fig, base):

    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")

    fig.savefig(png)
    fig.savefig(pdf)

    print("Saved:")
    print(" ", png)
    print(" ", pdf)


def build_log_edges(values, n_bins):

    values = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(values)
        & (values > 0.0)
    )

    if not np.any(valid):

        raise RuntimeError(
            "No finite positive values available for logarithmic binning."
        )

    lo = float(
        np.nanmin(
            values[valid]
        )
    )

    hi = float(
        np.nanmax(
            values[valid]
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

    p = k / n

    z2 = z**2

    denominator = (
        1.0
        + z2 / n
    )

    center = (
        p
        + z2 / (2.0 * n)
    ) / denominator

    half = (
        z
        / denominator
        * np.sqrt(
            p * (1.0 - p) / n
            + z2 / (4.0 * n**2)
        )
    )

    return (
        center - half,
        center + half,
    )


def quantiles(values):

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


def assign_te_bins(tE):

    idx = np.full(
        len(tE),
        -1,
        dtype=int,
    )

    for i in range(
        len(TE_BIN_EDGES)
        - 1
    ):

        left = TE_BIN_EDGES[i]
        right = TE_BIN_EDGES[i + 1]

        if np.isinf(right):

            mask = (
                np.isfinite(tE)
                & (tE >= left)
            )

        else:

            mask = (
                np.isfinite(tE)
                & (tE >= left)
                & (tE < right)
            )

        idx[mask] = i

    return idx


# ============================================================================
# Load and audit
# ============================================================================

def load_population():

    columns = [
        "catalog_row",
        "delta_chi2_lrt",
        "positive_lrt_class",
        "tE_true_days",
        "piE_true_amp",
    ]

    df = pd.read_parquet(
        INPUT,
        columns=columns,
    )

    if len(df) != EXPECTED_POSITIVE_N:

        raise RuntimeError(
            "Positive-population audit failed: "
            f"{len(df):,} != {EXPECTED_POSITIVE_N:,}"
        )

    T = numeric(
        df,
        "delta_chi2_lrt",
    )

    tE = numeric(
        df,
        "tE_true_days",
    )

    piE = numeric(
        df,
        "piE_true_amp",
    )

    classes = (
        df[
            "positive_lrt_class"
        ]
        .astype(str)
        .to_numpy()
    )

    if not np.all(
        np.isfinite(T)
        & (T > 0.0)
    ):

        raise RuntimeError(
            "Input is not the strict-positive Delta-chi2 population."
        )

    detected = (
        T
        >= PRIMARY_THRESHOLD
    )

    confused = (
        (T > 0.0)
        & (
            T
            < PRIMARY_THRESHOLD
        )
    )

    if int(detected.sum()) != EXPECTED_DETECTED_N:

        raise RuntimeError(
            "Detected-count audit failed: "
            f"{int(detected.sum()):,} != {EXPECTED_DETECTED_N:,}"
        )

    if int(confused.sum()) != EXPECTED_CONFUSED_N:

        raise RuntimeError(
            "Confused-count audit failed: "
            f"{int(confused.sum()):,} != {EXPECTED_CONFUSED_N:,}"
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
            "positive_lrt_class does not match the frozen LRT threshold."
        )

    valid = (
        np.isfinite(tE)
        & (tE > 0.0)
        & np.isfinite(piE)
        & (piE > 0.0)
    )

    df = df.loc[
        valid
    ].copy()

    T = T[valid]
    tE = tE[valid]
    piE = piE[valid]
    detected = detected[valid]

    X = (
        piE
        * (
            tE
            / YEAR_DAYS
        ) ** 2
    )

    te_bin_idx = assign_te_bins(
        tE
    )

    return (
        df,
        T,
        tE,
        piE,
        X,
        detected,
        te_bin_idx,
    )


# ============================================================================
# TEST 1
# Detection map with constant-X_pi lines
# ============================================================================

def make_detection_map(
    tE,
    piE,
    X,
    detected,
):

    te_edges = build_log_edges(
        tE,
        N_TE_MAP_BINS,
    )

    pie_edges = build_log_edges(
        piE,
        N_PIE_MAP_BINS,
    )

    te_idx = pd.cut(
        tE,
        bins=te_edges,
        include_lowest=True,
        labels=False,
    )

    pie_idx = pd.cut(
        piE,
        bins=pie_edges,
        include_lowest=True,
        labels=False,
    )

    probability = np.full(
        (
            N_PIE_MAP_BINS,
            N_TE_MAP_BINS,
        ),
        np.nan,
    )

    counts = np.zeros(
        (
            N_PIE_MAP_BINS,
            N_TE_MAP_BINS,
        ),
        dtype=int,
    )

    rows = []

    for j in range(
        N_PIE_MAP_BINS
    ):

        for i in range(
            N_TE_MAP_BINS
        ):

            mask = (
                np.asarray(te_idx == i)
                & np.asarray(pie_idx == j)
            )

            n = int(
                mask.sum()
            )

            k = int(
                detected[
                    mask
                ].sum()
            )

            p = (
                k / n
                if n > 0
                else np.nan
            )

            counts[
                j,
                i,
            ] = n

            if n >= MIN_MAP_CELL_N:

                probability[
                    j,
                    i,
                ] = p

            rows.append(
                {
                    "te_bin":
                        i,

                    "piE_bin":
                        j,

                    "te_left_days":
                        te_edges[i],

                    "te_right_days":
                        te_edges[i + 1],

                    "piE_left":
                        pie_edges[j],

                    "piE_right":
                        pie_edges[j + 1],

                    "n":
                        n,

                    "n_detected":
                        k,

                    "detection_fraction":
                        p,

                    "shown":
                        n >= MIN_MAP_CELL_N,
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        MAP_TABLE,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Dynamic constant-X levels spanning the central X distribution
    # ------------------------------------------------------------------------

    X_valid = X[
        np.isfinite(X)
        & (X > 0.0)
    ]

    x_lo = float(
        np.quantile(
            X_valid,
            0.10,
        )
    )

    x_hi = float(
        np.quantile(
            X_valid,
            0.90,
        )
    )

    X_levels = np.geomspace(
        x_lo,
        x_hi,
        5,
    )

    # ------------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            6.2,
        )
    )

    mesh = ax.pcolormesh(
        te_edges,
        pie_edges,
        probability,
        shading="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )

    colorbar = fig.colorbar(
        mesh,
        ax=ax,
    )

    colorbar.set_label(
        (
            r"$P(\mathrm{parallax\ detected}"
            r"\mid T>0)$"
        )
    )

    t_plot = np.geomspace(
        te_edges[0],
        te_edges[-1],
        1000,
    )

    for level in X_levels:

        piE_line = (
            level
            * (
                YEAR_DAYS
                / t_plot
            ) ** 2
        )

        visible = (
            (piE_line >= pie_edges[0])
            & (piE_line <= pie_edges[-1])
        )

        if np.any(visible):

            ax.plot(
                t_plot[visible],
                piE_line[visible],
                linestyle="--",
                linewidth=1.25,
                alpha=0.85,
                label=(
                    rf"$X_\pi={level:.2g}$"
                ),
            )

    ax.axvline(
        100.0,
        linestyle=":",
        linewidth=1.2,
        color="black",
        alpha=0.7,
    )

    ax.axvline(
        200.0,
        linestyle=":",
        linewidth=1.2,
        color="black",
        alpha=0.7,
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        r"True $t_E$ [days]"
    )

    ax.set_ylabel(
        r"True $|\boldsymbol{\pi}_E|$"
    )

    ax.set_title(
        (
            "Annual-parallax detectability with "
            r"$X_\pi=|\pi_E|(t_E/{\rm yr})^2$ contours"
        )
    )

    ax.legend(
        fontsize=13,
        frameon=True,
        loc="best",
    )

    fig.tight_layout()

    save_figure(
        fig,
        FIGURE_MAP,
    )

    plt.close(
        fig
    )

    return X_levels


# ============================================================================
# TEST 2
# Collapse of P_det versus X_pi
# ============================================================================

def make_collapse_figure(
    X,
    detected,
    te_bin_idx,
):

    x_edges = build_log_edges(
        X,
        N_X_BINS,
    )

    rows = []

    fig, ax = plt.subplots(
        figsize=(
            8.2,
            5.8,
        )
    )

    for ite, label in enumerate(
        TE_BIN_LABELS
    ):

        for ix in range(
            N_X_BINS
        ):

            left = x_edges[
                ix
            ]

            right = x_edges[
                ix + 1
            ]

            mask = (
                (te_bin_idx == ite)
                & np.isfinite(X)
                & (X >= left)
                & (X < right)
            )

            n = int(
                mask.sum()
            )

            k = int(
                detected[
                    mask
                ].sum()
            )

            fraction = (
                k / n
                if n > 0
                else np.nan
            )

            low, high = wilson_interval(
                k,
                n,
            )

            rows.append(
                {
                    "te_bin":
                        ite,

                    "te_label":
                        label,

                    "X_left":
                        left,

                    "X_right":
                        right,

                    "X_center":
                        np.sqrt(
                            left
                            * right
                        ),

                    "n":
                        n,

                    "n_detected":
                        k,

                    "detection_fraction":
                        fraction,

                    "wilson_low":
                        low,

                    "wilson_high":
                        high,
                }
            )

    table = pd.DataFrame(
        rows
    )

    table.to_csv(
        COLLAPSE_TABLE,
        index=False,
    )

    for ite, label in enumerate(
        TE_BIN_LABELS
    ):

        sub = table[
            (table["te_bin"] == ite)
            & (table["n"] >= MIN_X_BIN_N)
        ]

        if len(sub) == 0:

            continue

        line, = ax.plot(
            sub[
                "X_center"
            ],
            sub[
                "detection_fraction"
            ],
            marker="o",
            markersize=4.0,
            label=label,
        )

        ax.fill_between(
            sub[
                "X_center"
            ],
            sub[
                "wilson_low"
            ],
            sub[
                "wilson_high"
            ],
            color=line.get_color(),
            alpha=0.10,
        )

    ax.axhline(
        EXPECTED_DETECTED_N
        / EXPECTED_POSITIVE_N,
        linestyle="--",
        linewidth=1.3,
        color="black",
        label="Positive-population average",
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
            r"$X_\pi="
            r"|\boldsymbol{\pi}_{E,\rm true}|"
            r"(t_{E,\rm true}/1\,{\rm yr})^2$"
        )
    )

    ax.set_ylabel(
        (
            r"$P(\mathrm{parallax\ detected}"
            r"\mid X_\pi,t_E,T>0)$"
        )
    )

    ax.set_title(
        "Do short-event detection curves collapse onto one scaling?"
    )

    ax.grid(
        alpha=0.20,
    )

    ax.legend(
        frameon=True,
        ncol=2,
        fontsize=13,
    )

    fig.tight_layout()

    save_figure(
        fig,
        FIGURE_COLLAPSE,
    )

    plt.close(
        fig
    )


# ============================================================================
# TEST 3
# Delta-chi2 scaling versus X_pi
# ============================================================================

def make_delta_chi2_scaling(
    X,
    T,
    te_bin_idx,
):

    x_edges = build_log_edges(
        X,
        N_X_BINS,
    )

    rows = []

    for ite, label in enumerate(
        TE_BIN_LABELS
    ):

        for ix in range(
            N_X_BINS
        ):

            left = x_edges[
                ix
            ]

            right = x_edges[
                ix + 1
            ]

            mask = (
                (te_bin_idx == ite)
                & np.isfinite(X)
                & (X >= left)
                & (X < right)
                & np.isfinite(T)
                & (T > 0.0)
            )

            q16, median, q84 = quantiles(
                T[
                    mask
                ]
            )

            rows.append(
                {
                    "te_bin":
                        ite,

                    "te_label":
                        label,

                    "X_left":
                        left,

                    "X_right":
                        right,

                    "X_center":
                        np.sqrt(
                            left
                            * right
                        ),

                    "n":
                        int(
                            mask.sum()
                        ),

                    "T_q16":
                        q16,

                    "T_median":
                        median,

                    "T_q84":
                        q84,
                }
            )

    table = pd.DataFrame(
        rows
    )

    table.to_csv(
        T_TABLE,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Slope fits to binned medians
    # ------------------------------------------------------------------------

    slope_rows = []

    for ite, label in enumerate(
        TE_BIN_LABELS
    ):

        sub = table[
            (table["te_bin"] == ite)
            & (table["n"] >= MIN_X_BIN_N)
            & np.isfinite(
                table["T_median"]
            )
            & (table["T_median"] >= T_FIT_MIN)
            & (table["T_median"] <= T_FIT_MAX)
            & (table["X_center"] > 0.0)
        ].copy()

        if len(sub) < MIN_SLOPE_POINTS:

            slope_rows.append(
                {
                    "te_bin":
                        ite,

                    "te_label":
                        label,

                    "n_fit_points":
                        len(sub),

                    "slope":
                        np.nan,

                    "intercept":
                        np.nan,

                    "r2":
                        np.nan,
                }
            )

            continue

        log_x = np.log10(
            sub[
                "X_center"
            ].to_numpy(dtype=float)
        )

        log_t = np.log10(
            sub[
                "T_median"
            ].to_numpy(dtype=float)
        )

        slope, intercept = np.polyfit(
            log_x,
            log_t,
            1,
        )

        prediction = (
            slope * log_x
            + intercept
        )

        ss_res = float(
            np.sum(
                (
                    log_t
                    - prediction
                ) ** 2
            )
        )

        ss_tot = float(
            np.sum(
                (
                    log_t
                    - np.mean(log_t)
                ) ** 2
            )
        )

        r2 = (
            1.0
            - ss_res / ss_tot
            if ss_tot > 0.0
            else np.nan
        )

        slope_rows.append(
            {
                "te_bin":
                    ite,

                "te_label":
                    label,

                "n_fit_points":
                    len(sub),

                "slope":
                    slope,

                "intercept":
                    intercept,

                "r2":
                    r2,
            }
        )

    slope_table = pd.DataFrame(
        slope_rows
    )

    slope_table.to_csv(
        SLOPE_TABLE,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            8.4,
            6.0,
        )
    )

    for ite, label in enumerate(
        TE_BIN_LABELS
    ):

        sub = table[
            (table["te_bin"] == ite)
            & (table["n"] >= MIN_X_BIN_N)
            & np.isfinite(
                table["T_median"]
            )
            & (table["T_median"] > 0.0)
        ]

        if len(sub) == 0:

            continue

        slope_row = slope_table[
            slope_table["te_bin"]
            == ite
        ]

        slope = (
            float(
                slope_row[
                    "slope"
                ].iloc[0]
            )
            if len(slope_row)
            and np.isfinite(
                slope_row[
                    "slope"
                ].iloc[0]
            )
            else np.nan
        )

        if np.isfinite(slope):

            plot_label = (
                f"{label}, "
                rf"$m={slope:.2f}$"
            )

        else:

            plot_label = label

        line, = ax.plot(
            sub[
                "X_center"
            ],
            sub[
                "T_median"
            ],
            marker="o",
            markersize=4.0,
            label=plot_label,
        )

        ax.fill_between(
            sub[
                "X_center"
            ],
            sub[
                "T_q16"
            ],
            sub[
                "T_q84"
            ],
            color=line.get_color(),
            alpha=0.08,
        )

        if np.isfinite(slope):

            fit_sub = sub[
                (sub["T_median"] >= T_FIT_MIN)
                & (sub["T_median"] <= T_FIT_MAX)
            ]

            if len(fit_sub) >= MIN_SLOPE_POINTS:

                intercept = float(
                    slope_row[
                        "intercept"
                    ].iloc[0]
                )

                xx = np.geomspace(
                    fit_sub[
                        "X_center"
                    ].min(),
                    fit_sub[
                        "X_center"
                    ].max(),
                    100,
                )

                yy = (
                    10.0 ** intercept
                    * xx ** slope
                )

                ax.plot(
                    xx,
                    yy,
                    linestyle="--",
                    linewidth=1.0,
                    color=line.get_color(),
                    alpha=0.8,
                )

    ax.axhline(
        PRIMARY_THRESHOLD,
        linestyle="--",
        linewidth=1.4,
        color="black",
        label=(
            rf"Detection threshold "
            rf"$c_\alpha={PRIMARY_THRESHOLD:.2f}$"
        ),
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        (
            r"$X_\pi="
            r"|\boldsymbol{\pi}_{E,\rm true}|"
            r"(t_{E,\rm true}/1\,{\rm yr})^2$"
        )
    )

    ax.set_ylabel(
        r"Median $\Delta\chi^2_{\rm LRT}$"
    )

    ax.set_title(
        (
            r"Does $\Delta\chi^2_{\rm LRT}$ scale "
            r"approximately as $X_\pi^2$?"
        )
    )

    ax.grid(
        alpha=0.20,
    )

    ax.legend(
        frameon=True,
        ncol=2,
        fontsize=13,
    )

    fig.tight_layout()

    save_figure(
        fig,
        FIGURE_T,
    )

    plt.close(
        fig
    )

    return slope_table


# ============================================================================
# Summary
# ============================================================================

def write_summary(
    T,
    tE,
    piE,
    X,
    detected,
    te_bin_idx,
    X_levels,
    slope_table,
):

    lines = [
        "PARALLAX CONSTANT-ACCELERATION SCALING",
        "=" * 90,
        "",
        "Definition:",
        (
            "  X_pi = |piE_true| * "
            "(tE_true / 365.25 d)^2"
        ),
        "",
        "Physical expectation for short events:",
        "  delta u_pi ~ X_pi",
        "  Delta chi2_LRT ~ X_pi^2",
        "  expected log-log slope ~ 2",
        "",
        "Population:",
        (
            "  strict-positive H1 characterization sample "
            "(Delta chi2_LRT > 0)"
        ),
        f"  N = {len(T):,}",
        (
            f"  detected = "
            f"{int(detected.sum()):,}"
        ),
        (
            f"  confused = "
            f"{int((~detected).sum()):,}"
        ),
        (
            "  conditional detection fraction = "
            f"{np.mean(detected):.8f}"
        ),
        "",
        "X_pi summary:",
        (
            f"  median = "
            f"{np.median(X):.8g}"
        ),
        (
            f"  q16--q84 = "
            f"[{np.quantile(X, 0.16):.8g}, "
            f"{np.quantile(X, 0.84):.8g}]"
        ),
        "",
        "Constant-X_pi lines used in map:",
    ]

    for level in X_levels:

        lines.append(
            f"  {level:.8g}"
        )

    lines.extend(
        [
            "",
            "Approximate log Delta-chi2 vs log X_pi slope fits:",
            (
                f"  fit only binned medians with "
                f"{T_FIT_MIN:g} <= median(T) <= {T_FIT_MAX:g}"
            ),
        ]
    )

    for _, row in slope_table.iterrows():

        if np.isfinite(
            row[
                "slope"
            ]
        ):

            lines.append(
                (
                    f"  {row['te_label']}: "
                    f"slope={row['slope']:.4f}, "
                    f"R2={row['r2']:.4f}, "
                    f"Nbin={int(row['n_fit_points'])}"
                )
            )

        else:

            lines.append(
                (
                    f"  {row['te_label']}: "
                    "insufficient bins for slope fit"
                )
            )

    lines.extend(
        [
            "",
            "Interpretation guide:",
            (
                "  1. Parallelism between the 2D detection transition "
                "and constant-X_pi lines supports the scaling."
            ),
            (
                "  2. Collapse of the short-tE P_det(X_pi) curves "
                "supports X_pi as the dominant short-event variable."
            ),
            (
                "  3. Systematic separation of the curves around "
                "100--200 d identifies the empirical breakdown scale."
            ),
            (
                "  4. Short-tE slopes near 2 provide a more direct "
                "Delta-chi2 test of the perturbative scaling."
            ),
            "",
            "Caveat:",
            (
                "  X_pi is not expected to define an exact boundary: "
                "u0, source brightness, event geometry, season and "
                "cadence also affect detectability."
            ),
        ]
    )

    SUMMARY_FILE.write_text(
        "\n".join(lines)
        + "\n"
    )

    print()
    print(
        "\n".join(lines)
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

    print("=" * 100)
    print("47: PARALLAX CONSTANT-ACCELERATION SCALING")
    print("=" * 100)

    (
        df,
        T,
        tE,
        piE,
        X,
        detected,
        te_bin_idx,
    ) = load_population()

    print()
    print(
        "Strict-positive H1 population =",
        f"{len(df):,}",
    )

    print(
        "Detected =",
        f"{int(detected.sum()):,}",
    )

    print(
        "Confused =",
        f"{int((~detected).sum()):,}",
    )

    print(
        "Conditional detection fraction =",
        f"{np.mean(detected):.8f}",
    )

    print()
    print(
        "X_pi median =",
        np.median(X),
    )

    print(
        "X_pi q16--q84 =",
        np.quantile(
            X,
            [
                0.16,
                0.84,
            ],
        ),
    )

    print()
    print(
        "TEST 1: detection map + constant-X_pi contours"
    )

    X_levels = make_detection_map(
        tE=tE,
        piE=piE,
        X=X,
        detected=detected,
    )

    print()
    print(
        "TEST 2: curve collapse in X_pi"
    )

    make_collapse_figure(
        X=X,
        detected=detected,
        te_bin_idx=te_bin_idx,
    )

    print()
    print(
        "TEST 3: Delta-chi2 scaling with X_pi"
    )

    slope_table = make_delta_chi2_scaling(
        X=X,
        T=T,
        te_bin_idx=te_bin_idx,
    )

    write_summary(
        T=T,
        tE=tE,
        piE=piE,
        X=X,
        detected=detected,
        te_bin_idx=te_bin_idx,
        X_levels=X_levels,
        slope_table=slope_table,
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
