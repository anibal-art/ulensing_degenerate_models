#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "validation/production_profiling/results/"
    "h1_turnover_chi2_layers/"
    "turnover_chi2_layers.csv"
)

OUTDIR = Path(
    "validation/production_profiling/results/"
    "h1_turnover_chi2_layers/"
    "analysis"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


TE_LABELS = {
    0: "60-100",
    1: "100-200",
    2: "200-365",
    3: ">=365",
}

XPI_LABELS = {
    0: "0.01-0.03",
    1: "0.03-0.08",
    2: "0.08-0.15",
    3: "0.15-0.30",
    4: "0.30-1.00",
    5: ">=1.00",
}


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT)

df = df[
    df["status"] == "success"
].copy()

print("=" * 100)
print("TURNOVER DECOMPOSITION ANALYSIS")
print("=" * 100)
print("N =", len(df))


# ============================================================
# Derived fractions
# ============================================================

df["flux_retention"] = np.nan
df["flux_absorption_fraction"] = np.nan

m = (
    np.isfinite(df["D_fixed"])
    & np.isfinite(df["D_profile"])
    & (df["D_fixed"] > 0.0)
)

df.loc[
    m,
    "flux_retention"
] = (
    df.loc[m, "D_profile"]
    / df.loc[m, "D_fixed"]
)

df.loc[
    m,
    "flux_absorption_fraction"
] = (
    1.0
    - df.loc[m, "flux_retention"]
)


df["morph_absorption_fraction"] = np.nan

m = (
    np.isfinite(df["D_profile"])
    & np.isfinite(df["A_h0_physical"])
    & (df["D_profile"] > 0.0)
)

df.loc[
    m,
    "morph_absorption_fraction"
] = (
    df.loc[m, "A_h0_physical"]
    / df.loc[m, "D_profile"]
)


df["total_retention_fraction"] = np.nan

m = (
    np.isfinite(df["D_fixed"])
    & np.isfinite(df["T_fit"])
    & (df["D_fixed"] > 0.0)
)

df.loc[
    m,
    "total_retention_fraction"
] = (
    df.loc[m, "T_fit"]
    / df.loc[m, "D_fixed"]
)


df["post_morph_signal"] = (
    df["D_profile"]
    - df["A_h0_physical"]
)

# Should differ from T only by I_H1.
df["post_morph_plus_h1"] = (
    df["post_morph_signal"]
    + df["I_h1_physical"]
)


# ============================================================
# Parse profiled fluxes
# ============================================================

def flux_diagnostics(value):

    try:
        d = json.loads(value)
    except Exception:
        return {
            "negative_blend": np.nan,
            "min_blend": np.nan,
            "min_blend_over_ftotal": np.nan,
            "max_fs_over_ftotal": np.nan,
        }

    blends = []
    blend_fraction = []
    fs_over_total = []

    for tel, flux in d.items():

        fs = float(
            flux["fsource"]
        )

        ft = float(
            flux["ftotal"]
        )

        fb = ft - fs

        blends.append(fb)

        if ft != 0:
            blend_fraction.append(
                fb / ft
            )

            fs_over_total.append(
                fs / ft
            )

    return {
        "negative_blend":
            bool(
                np.any(
                    np.asarray(blends) < 0
                )
            ),

        "min_blend":
            float(
                np.min(blends)
            ),

        "min_blend_over_ftotal":
            float(
                np.min(blend_fraction)
            )
            if blend_fraction
            else np.nan,

        "max_fs_over_ftotal":
            float(
                np.max(fs_over_total)
            )
            if fs_over_total
            else np.nan,
    }


for prefix, column in [
    (
        "h0",
        "h0_profile_fluxes_json",
    ),
    (
        "h1",
        "h1_profile_fluxes_json",
    ),
]:

    diag = pd.DataFrame(
        [
            flux_diagnostics(v)
            for v in df[column]
        ],
        index=df.index,
    )

    for c in diag.columns:

        df[
            f"{prefix}_{c}"
        ] = diag[c]


# ============================================================
# Human-readable bin labels
# ============================================================

df["tE_bin"] = (
    df["tE_bin_index"]
    .astype(int)
    .map(TE_LABELS)
)

df["Xpi_bin"] = (
    df["Xpi_bin_index"]
    .astype(int)
    .map(XPI_LABELS)
)


# ============================================================
# Event-level output
# ============================================================

event_path = (
    OUTDIR
    / "turnover_decomposition_events.csv"
)

df.to_csv(
    event_path,
    index=False,
)


# ============================================================
# Cell summaries
# ============================================================

def q16(x):
    return np.nanpercentile(
        np.asarray(x, float),
        16,
    )


def q84(x):
    return np.nanpercentile(
        np.asarray(x, float),
        84,
    )


summary = (
    df
    .groupby(
        [
            "tE_bin_index",
            "Xpi_bin_index",
            "tE_bin",
            "Xpi_bin",
        ],
        observed=True,
    )
    .agg(
        N=(
            "catalog_row",
            "size",
        ),

        median_tE=(
            "tE_true",
            "median",
        ),

        median_Xpi=(
            "X_pi",
            "median",
        ),

        median_D_fixed=(
            "D_fixed",
            "median",
        ),

        median_D_profile=(
            "D_profile",
            "median",
        ),

        median_flux_absorption=(
            "flux_absorption",
            "median",
        ),

        median_A_h0=(
            "A_h0_physical",
            "median",
        ),

        median_I_h1=(
            "I_h1_physical",
            "median",
        ),

        median_T=(
            "T_fit",
            "median",
        ),

        median_flux_retention=(
            "flux_retention",
            "median",
        ),

        q16_flux_retention=(
            "flux_retention",
            q16,
        ),

        q84_flux_retention=(
            "flux_retention",
            q84,
        ),

        median_flux_abs_frac=(
            "flux_absorption_fraction",
            "median",
        ),

        median_morph_abs_frac=(
            "morph_absorption_fraction",
            "median",
        ),

        q16_morph_abs_frac=(
            "morph_absorption_fraction",
            q16,
        ),

        q84_morph_abs_frac=(
            "morph_absorption_fraction",
            q84,
        ),

        median_total_retention=(
            "total_retention_fraction",
            "median",
        ),

        median_h0_tE_bias=(
            "h0_tE_frac_bias",
            "median",
        ),

        median_abs_h0_tE_bias=(
            "h0_tE_frac_bias",
            lambda x:
                np.nanmedian(
                    np.abs(
                        np.asarray(
                            x,
                            float,
                        )
                    )
                ),
        ),

        frac_h0_negative_blend=(
            "h0_negative_blend",
            "mean",
        ),

        frac_h1_negative_blend=(
            "h1_negative_blend",
            "mean",
        ),

        median_h0_max_fs_over_ftotal=(
            "h0_max_fs_over_ftotal",
            "median",
        ),
    )
    .reset_index()
)


summary_path = (
    OUTDIR
    / "turnover_decomposition_by_cell.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Print compact science table
# ============================================================

cols = [
    "tE_bin",
    "Xpi_bin",
    "N",

    "median_D_fixed",
    "median_D_profile",
    "median_T",

    "median_flux_abs_frac",
    "median_morph_abs_frac",
    "median_total_retention",

    "median_h0_tE_bias",
    "frac_h0_negative_blend",
]

print()
print("=" * 100)
print("CELL SUMMARY")
print("=" * 100)

print(
    summary[
        cols
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4g}",
    )
)


# ============================================================
# Global diagnostics
# ============================================================

print()
print("=" * 100)
print("GLOBAL DIAGNOSTICS")
print("=" * 100)

print(
    "H0 any negative blend =",
    df[
        "h0_negative_blend"
    ].mean(),
)

print(
    "H1 any negative blend =",
    df[
        "h1_negative_blend"
    ].mean(),
)

print(
    "median flux retention Dprofile/Dfixed =",
    df[
        "flux_retention"
    ].median(),
)

print(
    "median flux absorption fraction =",
    df[
        "flux_absorption_fraction"
    ].median(),
)

print(
    "median morphology absorption fraction =",
    df[
        "morph_absorption_fraction"
    ].median(),
)

print(
    "median total retention T/Dfixed =",
    df[
        "total_retention_fraction"
    ].median(),
)

print(
    "median |H0 tE fractional bias| =",
    np.nanmedian(
        np.abs(
            df[
                "h0_tE_frac_bias"
            ]
        )
    ),
)

print()
print("event table =", event_path)
print("cell table  =", summary_path)
