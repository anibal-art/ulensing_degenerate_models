#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent

ROOT = (
    HERE
    / "results"
    / "h1_turnover_chi2_layers"
    / "full_population_v1"
)

INPUT = (
    ROOT
    / "h1_chi2_layers_full_canonical_333653.parquet"
)

OUT = (
    ROOT
    / "canonical_analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

THRESHOLD = 13.15982011480088

TE_EDGES = np.array([
    0.0,
    10.0,
    30.0,
    60.0,
    100.0,
    200.0,
    365.0,
    np.inf,
])

TE_LABELS = [
    r"$t_E<10$ d",
    r"$10\leq t_E<30$ d",
    r"$30\leq t_E<60$ d",
    r"$60\leq t_E<100$ d",
    r"$100\leq t_E<200$ d",
    r"$200\leq t_E<365$ d",
    r"$t_E\geq365$ d",
]

# Deliberately resolve the empirical turnover region
# around X_pi ~ 0.1--0.3.
XPI_EDGES = np.array([
    0.0,
    0.001,
    0.003,
    0.01,
    0.03,
    0.06,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    1.00,
    2.00,
    np.inf,
])

XPI_LABELS = [
    "<0.001",
    "0.001-0.003",
    "0.003-0.01",
    "0.01-0.03",
    "0.03-0.06",
    "0.06-0.10",
    "0.10-0.15",
    "0.15-0.20",
    "0.20-0.30",
    "0.30-0.50",
    "0.50-1.00",
    "1.00-2.00",
    ">=2.00",
]


# ============================================================
# Helpers
# ============================================================

def q16(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return np.quantile(x, 0.16)


def q50(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return np.quantile(x, 0.50)


def q84(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return np.quantile(x, 0.84)


def summarize_cells(df, sample_name):

    rows = []

    for it, te_label in enumerate(TE_LABELS):

        dte = df[
            df["tE_bin_index"] == it
        ]

        for ix, xpi_label in enumerate(XPI_LABELS):

            d = dte[
                dte["Xpi_bin_index"] == ix
            ]

            if len(d) == 0:
                continue

            row = {
                "sample": sample_name,
                "tE_bin_index": it,
                "tE_bin": te_label,
                "Xpi_bin_index": ix,
                "Xpi_bin": xpi_label,
                "Xpi_left": XPI_EDGES[ix],
                "Xpi_right": XPI_EDGES[ix + 1],
                "n": len(d),
                "X_pi_median": q50(d["X_pi"]),
                "tE_true_median": q50(d["tE_true"]),
                "detection_fraction": np.mean(
                    d["T_fit"].to_numpy(float) > THRESHOLD
                ),
            }

            quantities = [
                "D_fixed",
                "D_profile",
                "post_morph",
                "T_fit",
                "flux_absorption",
                "A_h0_physical",
                "I_h1_physical",
                "f_flux",
                "f_morph",
                "f_retained",
                "h0_tE_frac_bias",
                "h1_tE_frac_bias",
            ]

            for c in quantities:

                x = pd.to_numeric(
                    d[c],
                    errors="coerce",
                )

                row[f"{c}_q16"] = q16(x)
                row[f"{c}_median"] = q50(x)
                row[f"{c}_q84"] = q84(x)
                row[f"{c}_nfinite"] = int(
                    np.isfinite(x).sum()
                )

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Load canonical population
# ============================================================

df = pd.read_parquet(INPUT)

if len(df) != 333653:
    raise RuntimeError(
        f"Expected canonical N=333653, got {len(df)}"
    )

if df["catalog_row"].nunique() != 333653:
    raise RuntimeError(
        "catalog_row is not unique."
    )


# ============================================================
# Derived physical decomposition
# ============================================================

for c in [
    "tE_true",
    "X_pi",
    "D_fixed",
    "D_profile",
    "A_h0_physical",
    "I_h1_physical",
    "T_fit",
    "h0_tE_frac_bias",
    "h1_tE_frac_bias",
]:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )


# Signal left after H0 physical/morphological adaptation.
df["post_morph"] = (
    df["D_profile"]
    - df["A_h0_physical"]
)


# Event-level absorption fractions.
#
# IMPORTANT:
# ratios are computed event by event,
# then medians are taken inside bins.
#
# We do NOT use ratios of population medians.

df["f_flux"] = np.nan
m = (
    np.isfinite(df["D_fixed"])
    & (df["D_fixed"] > 0)
)
df.loc[m, "f_flux"] = (
    1.0
    - df.loc[m, "D_profile"]
    / df.loc[m, "D_fixed"]
)


df["f_morph"] = np.nan
m = (
    np.isfinite(df["D_profile"])
    & (df["D_profile"] > 0)
)
df.loc[m, "f_morph"] = (
    df.loc[m, "A_h0_physical"]
    / df.loc[m, "D_profile"]
)


df["f_retained"] = np.nan
m = (
    np.isfinite(df["D_fixed"])
    & (df["D_fixed"] > 0)
)
df.loc[m, "f_retained"] = (
    df.loc[m, "T_fit"]
    / df.loc[m, "D_fixed"]
)


# ============================================================
# Bin assignments
# ============================================================

df["tE_bin_index"] = (
    np.digitize(
        df["tE_true"],
        TE_EDGES[1:-1],
        right=False,
    )
)


df["Xpi_bin_index"] = (
    np.digitize(
        df["X_pi"],
        XPI_EDGES[1:-1],
        right=False,
    )
)


# ============================================================
# Samples
# ============================================================

samples = {
    "all": df,
    "positive": df[
        df["T_fit"] > 0
    ],
    "detected": df[
        df["T_fit"] > THRESHOLD
    ],
}


print("=" * 100)
print("CANONICAL TURNOVER LAYER ANALYSIS")
print("=" * 100)

for name, x in samples.items():
    print(
        f"{name:10s} N={len(x):7d}"
    )


# ============================================================
# Cell tables
# ============================================================

tables = []

for name, x in samples.items():

    s = summarize_cells(
        x,
        name,
    )

    tables.append(s)


summary = pd.concat(
    tables,
    ignore_index=True,
)

summary_path = (
    OUT
    / "turnover_layers_by_tE_Xpi.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Exact sanity checks
# ============================================================

closure_direct = (
    df["T_fit"]
    -
    (
        df["D_profile"]
        - df["A_h0_physical"]
        + df["I_h1_physical"]
    )
)

print()
print("IDENTITY")
print(
    "max |T - (Dprofile-A0+I1)| =",
    np.nanmax(
        np.abs(closure_direct)
    )
)


# ============================================================
# Figure 1:
# four physical layers vs X_pi
# ============================================================

layers = [
    ("D_fixed", r"$D_{\rm fixed}$"),
    ("D_profile", r"$D_{\rm profile}$"),
    ("post_morph", r"$D_{\rm profile}-A_{H0}$"),
    ("T_fit", r"$T_{\rm fit}$"),
]


for it, te_label in enumerate(TE_LABELS):

    s = summary[
        (summary["sample"] == "all")
        & (summary["tE_bin_index"] == it)
    ].sort_values(
        "Xpi_bin_index"
    )

    if len(s) == 0:
        continue

    fig, ax = plt.subplots(
        figsize=(8.2, 5.8)
    )

    for col, label in layers:

        x = s["X_pi_median"].to_numpy(float)
        y = s[f"{col}_median"].to_numpy(float)

        good = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x > 0)
            & (y > 0)
        )

        ax.plot(
            x[good],
            y[good],
            marker="o",
            linewidth=1.4,
            label=label,
        )

    ax.axhline(
        THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label=r"$T_{\rm crit}$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        r"$X_\pi=|\pi_E|(t_E/365.25\,{\rm d})^2$"
    )

    ax.set_ylabel(
        r"Median $\Delta\chi^2$-like quantity"
    )

    ax.set_title(
        te_label
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / f"layers_vs_Xpi_tEbin_{it}.png",
        dpi=220,
    )

    plt.close(fig)


# ============================================================
# Figure 2:
# event-level absorption fractions
# ============================================================

for it, te_label in enumerate(TE_LABELS):

    s = summary[
        (summary["sample"] == "all")
        & (summary["tE_bin_index"] == it)
    ].sort_values(
        "Xpi_bin_index"
    )

    if len(s) == 0:
        continue

    fig, ax = plt.subplots(
        figsize=(8.2, 5.8)
    )

    for col, label in [
        (
            "f_flux",
            r"$1-D_{\rm profile}/D_{\rm fixed}$",
        ),
        (
            "f_morph",
            r"$A_{H0}/D_{\rm profile}$",
        ),
        (
            "f_retained",
            r"$T/D_{\rm fixed}$",
        ),
    ]:

        x = s["X_pi_median"].to_numpy(float)
        y = s[f"{col}_median"].to_numpy(float)

        good = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x > 0)
        )

        ax.plot(
            x[good],
            y[good],
            marker="o",
            linewidth=1.4,
            label=label,
        )

    ax.axhline(
        0.0,
        linewidth=0.8,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        r"$X_\pi$"
    )

    ax.set_ylabel(
        "Median event-level fraction"
    )

    ax.set_title(
        te_label
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / f"absorption_fractions_vs_Xpi_tEbin_{it}.png",
        dpi=220,
    )

    plt.close(fig)


# ============================================================
# Figure 3:
# H0 tE adaptation
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.0, 6.2)
)

sall = summary[
    summary["sample"] == "all"
]

for it, te_label in enumerate(TE_LABELS):

    s = sall[
        sall["tE_bin_index"] == it
    ].sort_values(
        "Xpi_bin_index"
    )

    x = s["X_pi_median"].to_numpy(float)
    y = s[
        "h0_tE_frac_bias_median"
    ].to_numpy(float)

    good = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
    )

    ax.plot(
        x[good],
        y[good],
        marker="o",
        linewidth=1.2,
        label=te_label,
    )

ax.axhline(
    0.0,
    linewidth=0.8,
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
    alpha=0.25,
)

ax.legend(
    fontsize=8,
    ncol=2,
)

fig.tight_layout()

fig.savefig(
    OUT
    / "h0_tE_bias_vs_Xpi.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# Figure 4:
# operational detection fraction
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.0, 6.2)
)

for it, te_label in enumerate(TE_LABELS):

    s = sall[
        sall["tE_bin_index"] == it
    ].sort_values(
        "Xpi_bin_index"
    )

    x = s["X_pi_median"].to_numpy(float)
    y = s[
        "detection_fraction"
    ].to_numpy(float)

    good = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
    )

    ax.plot(
        x[good],
        y[good],
        marker="o",
        linewidth=1.2,
        label=te_label,
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
    "Operational annual-parallax detection fraction"
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend(
    fontsize=8,
    ncol=2,
)

fig.tight_layout()

fig.savefig(
    OUT
    / "detection_fraction_vs_Xpi.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# Compact high-tE diagnostic table
# ============================================================

focus = sall[
    sall["tE_bin_index"] >= 3
].copy()

focus_cols = [
    "tE_bin",
    "Xpi_bin",
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
]

focus = focus[
    focus_cols
]

focus.to_csv(
    OUT
    / "high_tE_turnover_diagnostics.csv",
    index=False,
)


# ============================================================
# Terminal report
# ============================================================

print()
print("=" * 100)
print("HIGH-tE TURNOVER DIAGNOSTICS")
print("=" * 100)

print(
    focus.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print()
print("saved summary =", summary_path)
print("output dir    =", OUT)
