#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot-only version of the adjusted-rho diagnostic.

IMPORTANT
---------
This script DOES NOT:
- refit classifiers;
- repeat cross-validation;
- recompute OOF predictions;
- modify the statistical results.

It only reads the previously saved outputs from
38_adjusted_rho_effect_on_parallax_detection.py and regenerates
the presentation figure.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# Paths
# ============================================================================

RESULTS_DIR = Path(
    "analysis/lrt/results/characterization/"
    "38_adjusted_rho_effect_on_parallax_detection"
)

FIGURE_DIR = Path(
    "analysis/lrt/figures/characterization/"
    "38_adjusted_rho_effect_on_parallax_detection"
)

CV_PATH = RESULTS_DIR / "cv_metrics.csv"

OBSERVED_PATH = (
    RESULTS_DIR
    / "observed_rho_detection_bins.csv"
)

ADJUSTED_PATH = (
    RESULTS_DIR
    / "adjusted_rho_detection_curve.csv"
)

PNG_PATH = (
    FIGURE_DIR
    / "adjusted_rho_effect_on_parallax_detection.png"
)

PDF_PATH = (
    FIGURE_DIR
    / "adjusted_rho_effect_on_parallax_detection.pdf"
)


# ============================================================================
# Presentation style
# ============================================================================

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.titlesize": 18,
    }
)


# ============================================================================
# Load frozen derived results
# ============================================================================

for path in [
    CV_PATH,
    OBSERVED_PATH,
    ADJUSTED_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing saved result: {path}\n"
            "Run 38_adjusted_rho_effect_on_parallax_detection.py "
            "once to compute the statistical results."
        )

cv = pd.read_csv(CV_PATH)
obs = pd.read_csv(OBSERVED_PATH)
adj = pd.read_csv(ADJUSTED_PATH)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# Derived quantities from SAVED tables only
# ============================================================================

n_total = int(
    obs["n_total"].sum()
)

n_detected = int(
    obs["n_detected"].sum()
)

detected_fraction = (
    n_detected
    / n_total
)

rho_grid = adj[
    "rho_true"
].to_numpy(float)

adjusted_mean = adj[
    "adjusted_probability_mean"
].to_numpy(float)

adjusted_min = adj[
    "adjusted_probability_fold_min"
].to_numpy(float)

adjusted_max = adj[
    "adjusted_probability_fold_max"
].to_numpy(float)

fold_cols = [
    c
    for c in adj.columns
    if c.startswith("fold_")
]

fold_cols = sorted(
    fold_cols,
    key=lambda x: int(
        x.split("_")[1]
    ),
)

fold_gain = cv[
    "relative_logloss_improvement_pct"
].to_numpy(float)

mean_gain = float(
    np.mean(fold_gain)
)

std_gain = float(
    np.std(
        fold_gain,
        ddof=1,
    )
)


# ============================================================================
# Figure
# ============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.8, 5.4),
    gridspec_kw={
        "width_ratios": [
            1.25,
            1.0,
        ]
    },
    constrained_layout=True,
)


# ----------------------------------------------------------------------------
# Panel A: rho dependence after adjustment
# ----------------------------------------------------------------------------

ax = axes[0]

obs_x = obs[
    "rho_true_median"
].to_numpy(float)

obs_y = obs[
    "detection_fraction"
].to_numpy(float)

obs_low = obs[
    "wilson95_low"
].to_numpy(float)

obs_high = obs[
    "wilson95_high"
].to_numpy(float)

obs_yerr = np.vstack(
    [
        obs_y - obs_low,
        obs_high - obs_y,
    ]
)

ax.fill_between(
    rho_grid,
    adjusted_min,
    adjusted_max,
    alpha=0.18,
    label="Range across CV folds",
)

ax.plot(
    rho_grid,
    adjusted_mean,
    linewidth=2.5,
    label=(
        r"Adjusted for "
        r"$t_E$, $|\pi_E|$, $|u_0|$"
    ),
)

ax.axhline(
    detected_fraction,
    linestyle="--",
    linewidth=1.5,
    label=(
        "Overall detection fraction "
        f"({detected_fraction:.3f})"
    ),
)

ax.errorbar(
    obs_x,
    obs_y,
    yerr=obs_yerr,
    marker="o",
    linestyle="-",
    linewidth=1.5,
    markersize=5,
    capsize=3,
    label="Observed fraction",
)

ax.set_xscale(
    "log"
)

ax.set_ylim(
    0.0,
    1.0,
)

ax.set_xlabel(
    r"True source size $\rho_{\rm true}$"
)

ax.set_ylabel(
    "Parallax detection probability"
)

ax.set_title(
    r"Detection probability vs. $\rho$"
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    loc="best",
)


# ----------------------------------------------------------------------------
# Panel B: incremental information from rho
# ----------------------------------------------------------------------------

ax = axes[1]

fold_x = np.arange(
    1,
    len(fold_gain) + 1,
)

mean_x = (
    len(fold_gain)
    + 1.2
)

ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1.5,
)

ax.plot(
    fold_x,
    fold_gain,
    marker="o",
    linewidth=1.8,
    markersize=6,
    label="CV folds",
)

ax.errorbar(
    [mean_x],
    [mean_gain],
    yerr=[[std_gain], [std_gain]],
    marker="D",
    markersize=7,
    capsize=5,
    linestyle="none",
    label=r"Mean $\pm$ SD",
)

ax.set_xticks(
    list(fold_x)
    + [mean_x]
)

ax.set_xticklabels(
    [
        str(i)
        for i in fold_x
    ]
    + ["mean"]
)

ax.set_xlabel(
    "Cross-validation fold"
)

ax.set_ylabel(
    "Relative log-loss improvement [%]"
)

ax.set_title(
    r"Incremental information from $\rho$"
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    loc="best",
)


# ============================================================================
# Save
# ============================================================================

fig.savefig(
    PNG_PATH,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF_PATH,
    bbox_inches="tight",
)

plt.close(fig)

print("Plot-only regeneration complete.")
print("Loaded:")
print(" ", CV_PATH)
print(" ", OBSERVED_PATH)
print(" ", ADJUSTED_PATH)
print()
print("Saved:")
print(" ", PNG_PATH)
print(" ", PDF_PATH)
print()
print(
    "No classifier fitting or cross-validation was performed."
)
