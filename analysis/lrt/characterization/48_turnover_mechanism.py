#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Physical origin of the large-X_pi turnover in annual-parallax detectability.

This script uses the compact canonical cell-level table produced from the
complete H1 population (N=333653). No simulation or fitting is performed.

The decomposition is

    T = D_profile - A_H0 + I_H1,

with

    D_fixed
        raw H0-vs-H1 contrast at the true physical parameters and
        generating fluxes;

    D_profile
        contrast after profiling telescope fluxes at the true physical
        parameters;

    D_profile - A_H0
        contrast remaining after the H0 model adapts its shared physical
        parameters;

    T
        final operational likelihood-ratio statistic.

The figures focus on the long-event regime where the empirical turnover
appears. Bins with fewer than MIN_N events are omitted from the plotted
curves but remain in the underlying CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Paths
# ============================================================================

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]

INPUT = (
    REPO
    / "Parallax_LSST"
    / "lsstmonts_catalog_sedighe"
    / "validation"
    / "production_profiling"
    / "results"
    / "h1_turnover_chi2_layers"
    / "full_population_v1"
    / "canonical_analysis"
    / "turnover_layers_by_tE_Xpi.csv"
)

OUT = (
    REPO
    / "analysis"
    / "lrt"
    / "figures"
    / "characterization"
    / "48_turnover_mechanism"
)

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Configuration
# ============================================================================

T_CRIT = 13.15982011480088
MIN_N = 30

LONG_TE_BINS = [3, 4, 5, 6]

TE_LABELS = {
    3: r"$60\leq t_E<100$ d",
    4: r"$100\leq t_E<200$ d",
    5: r"$200\leq t_E<365$ d",
    6: r"$t_E\geq365$ d",
}

LAYERS = [
    (
        "D_fixed_median",
        r"raw signal: $D_{\rm fixed}$",
        "o",
        "-",
    ),
    (
        "D_profile_median",
        r"after flux profiling: $D_{\rm profile}$",
        "s",
        "-",
    ),
    (
        "post_morph_median",
        r"after H0 morphology fit: $D_{\rm profile}-A_{H0}$",
        "^",
        "-",
    ),
    (
        "T_fit_median",
        r"final LRT: $T$",
        "D",
        "--",
    ),
]


# ============================================================================
# Load canonical cell summary
# ============================================================================

df = pd.read_csv(INPUT)

required = {
    "sample",
    "tE_bin_index",
    "n",
    "X_pi_median",
    "detection_fraction",
    "D_fixed_median",
    "D_profile_median",
    "post_morph_median",
    "T_fit_median",
    "f_flux_median",
    "f_morph_median",
    "f_retained_median",
    "h0_tE_frac_bias_median",
}

missing = sorted(required - set(df.columns))

if missing:
    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(missing)
    )

df = df[
    df["sample"] == "all"
].copy()

df = df[
    df["tE_bin_index"].isin(LONG_TE_BINS)
].copy()

df = df[
    df["n"] >= MIN_N
].copy()

df = df.sort_values(
    ["tE_bin_index", "X_pi_median"]
)


# ============================================================================
# Figure 1:
# where does the signal disappear?
# ============================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12.0, 9.0),
    sharex=False,
    sharey=True,
)

axes = axes.ravel()

for ax, te_bin in zip(axes, LONG_TE_BINS):

    d = df[
        df["tE_bin_index"] == te_bin
    ].copy()

    x = d["X_pi_median"].to_numpy(float)

    for column, label, marker, linestyle in LAYERS:

        y = d[column].to_numpy(float)

        good = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x > 0)
            & (y > 0)
        )

        ax.plot(
            x[good],
            y[good],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=4.5,
            label=label,
        )

    ax.axhline(
        T_CRIT,
        linestyle=":",
        linewidth=1.2,
        label=r"$T_{\rm crit}$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(
        TE_LABELS[te_bin]
    )

    ax.grid(
        True,
        alpha=0.22,
    )

    ax.set_xlabel(
        r"$X_\pi=|\pi_E|(t_E/365.25\,{\rm d})^2$"
    )


axes[0].set_ylabel(
    r"Median $\Delta\chi^2$-like quantity"
)

axes[2].set_ylabel(
    r"Median $\Delta\chi^2$-like quantity"
)

handles, labels = axes[0].get_legend_handles_labels()

fig.suptitle(
    "Where the annual-parallax signal is absorbed",
    fontsize=15,
    y=0.995,
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.958),
    ncol=3,
    frameon=False,
    fontsize=9,
)

fig.text(
    0.5,
    0.012,
    rf"Canonical H1 population; plotted cells have $N\geq{MIN_N}$.",
    ha="center",
    fontsize=9,
)

fig.tight_layout(
    rect=[0, 0.035, 1, 0.885]
)

fig.savefig(
    OUT / "01_turnover_signal_layers.png",
    dpi=250,
)

fig.savefig(
    OUT / "01_turnover_signal_layers.pdf",
)

plt.close(fig)


# ============================================================================
# Figure 2:
# operational turnover + morphological adaptation
# ============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.0, 4.9),
)


# ----------------------------------------------------------------------------
# Left: detection probability
# ----------------------------------------------------------------------------

ax = axes[0]

for te_bin in LONG_TE_BINS:

    d = df[
        df["tE_bin_index"] == te_bin
    ]

    ax.plot(
        d["X_pi_median"],
        d["detection_fraction"],
        marker="o",
        linewidth=1.5,
        markersize=4.5,
        label=TE_LABELS[te_bin],
    )

ax.set_xscale("log")

ax.set_ylim(
    -0.03,
    1.03,
)

ax.set_xlabel(
    r"$X_\pi$"
)

ax.set_ylabel(
    r"$P(T>T_{\rm crit})$"
)

ax.set_title(
    "Operational parallax detectability"
)

ax.grid(
    True,
    alpha=0.22,
)

ax.legend(
    fontsize=8.5,
)


# ----------------------------------------------------------------------------
# Right: H0 timescale adaptation
# ----------------------------------------------------------------------------

ax = axes[1]

for te_bin in LONG_TE_BINS:

    d = df[
        df["tE_bin_index"] == te_bin
    ]

    ax.plot(
        d["X_pi_median"],
        d["h0_tE_frac_bias_median"],
        marker="o",
        linewidth=1.5,
        markersize=4.5,
        label=TE_LABELS[te_bin],
    )

ax.axhline(
    0.0,
    linewidth=1.0,
)

ax.set_xscale("log")

ax.set_xlabel(
    r"$X_\pi$"
)

ax.set_ylabel(
    r"Median $(t_{E,H0}-t_{E,\rm true})/t_{E,\rm true}$"
)

ax.set_title(
    r"H0 timescale adaptation"
)

ax.grid(
    True,
    alpha=0.22,
)


fig.suptitle(
    "The large-$X_\\pi$ turnover tracks H0 model adaptation",
    fontsize=14,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)

fig.savefig(
    OUT / "02_detection_and_tE_adaptation.png",
    dpi=250,
)

fig.savefig(
    OUT / "02_detection_and_tE_adaptation.pdf",
)

plt.close(fig)


# ============================================================================
# Compact numerical check for the terminal
# ============================================================================

print("=" * 88)
print("TURNOVER MECHANISM FIGURES")
print("=" * 88)
print(f"input  = {INPUT}")
print(f"output = {OUT}")
print(f"minimum plotted cell count = {MIN_N}")

print()
print("Key high-X_pi cells:")

targets = [
    (4, 0.50, 1.00),
    (5, 0.50, 1.00),
    (5, 2.00, np.inf),
    (6, 1.00, 2.00),
    (6, 2.00, np.inf),
]

source = pd.read_csv(INPUT)
source = source[
    source["sample"] == "all"
]

for te_bin, lo, hi in targets:

    d = source[
        source["tE_bin_index"] == te_bin
    ].copy()

    if np.isinf(hi):
        q = d[
            d["X_pi_median"] >= lo
        ]
    else:
        q = d[
            (d["X_pi_median"] >= lo)
            & (d["X_pi_median"] < hi)
        ]

    if len(q) == 0:
        continue

    row = q.iloc[0]

    print(
        f"{TE_LABELS[te_bin]:26s} "
        f"Xpi={row['X_pi_median']:.3g} "
        f"N={int(row['n']):5d} "
        f"Pdet={row['detection_fraction']:.3f} "
        f"Dfixed={row['D_fixed_median']:.3g} "
        f"Dprofile={row['D_profile_median']:.3g} "
        f"postMorph={row['post_morph_median']:.3g} "
        f"T={row['T_fit_median']:.3g} "
        f"bias_tE={row['h0_tE_frac_bias_median']:.3f}"
    )

print()
print("DONE")
