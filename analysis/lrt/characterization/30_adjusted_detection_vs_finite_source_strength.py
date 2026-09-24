#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adjusted parallax-detection probability versus TRUE finite-source strength.

NO simulation.
NO fitting of light curves.

We model the binary empirical-LRT detection outcome with a flexible
logistic regression using:

    log10(tE_true)
    log10(piE_true)
    log10(rho_true / |u0_true|)

Each continuous variable is represented with a cubic spline.

The goal is NOT to predict individual events.  The regression is used as
a standardization tool to ask:

    Does finite-source strength retain an association with parallax
    detectability after controlling for the two major truth-level
    determinants already identified, tE and |pi_E|?

For a grid of finite-source strengths R_FS, we replace R_FS for every
event while keeping that event's observed tE and piE, compute its
predicted detection probability, and average over the population:

    P_adj(R_FS)
      = mean_j P(det | tE_j, piE_j, R_FS).

This is a marginal standardized probability, not the raw conditional
fraction plotted in script 29.
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

from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LogisticRegression


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "18_positive_truth"
    / "h1_positive_truth_characterization.parquet"
)

RAW_TABLE = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "29_detection_fraction_vs_finite_source_strength"
    / "detection_fraction_vs_finite_source_strength_bins.csv"
)

FIGURE_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "figures"
    / "characterization"
    / "30_adjusted_detection_vs_finite_source_strength"
)

RESULT_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "30_adjusted_detection_vs_finite_source_strength"
)

FIGURE = (
    FIGURE_DIR
    / "adjusted_detection_vs_finite_source_strength.png"
)

TABLE = (
    RESULT_DIR
    / "adjusted_detection_vs_finite_source_strength.csv"
)


# ============================================================
# Configuration
# ============================================================

PRIMARY_THRESHOLD = 13.15982011480088

# Number of internal knots for each cubic spline.
N_KNOTS = 7

# Evaluation grid.
N_GRID = 100

# To keep standardization fast while still representing the
# full population, use a deterministic subsample.
STANDARDIZATION_SAMPLE = 50000

RNG_SEED = 23092026


# ============================================================
# Helper
# ============================================================

def numeric(df, column):

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


# ============================================================
# Main
# ============================================================

def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input not found:\n{INPUT}"
        )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load positive-LRT population
    # ========================================================

    df = pd.read_parquet(
        INPUT,
        columns=[
            "catalog_row",
            "delta_chi2_lrt",
            "tE_true_days",
            "piE_true_amp",
            "rho_over_abs_u0_true",
        ],
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

    R = numeric(
        df,
        "rho_over_abs_u0_true",
    )

    detected = (
        T >= PRIMARY_THRESHOLD
    ).astype(int)

    valid = (
        np.isfinite(T)
        & (T > 0)
        & np.isfinite(tE)
        & (tE > 0)
        & np.isfinite(piE)
        & (piE > 0)
        & np.isfinite(R)
        & (R > 0)
    )

    tE = tE[valid]
    piE = piE[valid]
    R = R[valid]
    detected = detected[valid]

    log_tE = np.log10(tE)
    log_piE = np.log10(piE)
    log_R = np.log10(R)

    print(
        "=" * 110
    )

    print(
        "ADJUSTED PARALLAX DETECTABILITY VS FINITE-SOURCE STRENGTH"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "valid events =",
        len(R),
    )

    print(
        "raw detection fraction =",
        float(
            detected.mean()
        ),
    )

    # ========================================================
    # Flexible spline representation
    # ========================================================

    X_raw = np.column_stack(
        [
            log_tE,
            log_piE,
            log_R,
        ]
    )

    spline = SplineTransformer(
        degree=3,
        n_knots=N_KNOTS,
        knots="quantile",
        include_bias=False,
    )

    X = spline.fit_transform(
        X_raw
    )

    # Very weak regularization only for numerical stability.
    model = LogisticRegression(
        penalty="l2",
        C=1.0e6,
        solver="lbfgs",
        max_iter=2000,
    )

    model.fit(
        X,
        detected,
    )

    print(
        "logistic iterations =",
        int(
            model.n_iter_[0]
        ),
    )

    # ========================================================
    # Deterministic population sample for standardization
    # ========================================================

    rng = np.random.default_rng(
        RNG_SEED
    )

    if len(R) > STANDARDIZATION_SAMPLE:

        idx = rng.choice(
            len(R),
            size=STANDARDIZATION_SAMPLE,
            replace=False,
        )

    else:

        idx = np.arange(
            len(R)
        )

    base_log_tE = log_tE[
        idx
    ]

    base_log_piE = log_piE[
        idx
    ]

    print(
        "standardization sample =",
        len(idx),
    )

    # ========================================================
    # Evaluation range
    #
    # Avoid extrapolating beyond the actual positive population.
    # ========================================================

    R_min = float(
        np.quantile(
            R,
            0.001,
        )
    )

    R_max = float(
        np.quantile(
            R,
            0.999,
        )
    )

    R_grid = np.geomspace(
        R_min,
        R_max,
        N_GRID,
    )

    adjusted_probability = []

    # ========================================================
    # Standardized predictions
    # ========================================================

    for R_value in R_grid:

        log_R_value = np.log10(
            R_value
        )

        X_eval_raw = np.column_stack(
            [
                base_log_tE,
                base_log_piE,
                np.full(
                    len(idx),
                    log_R_value,
                ),
            ]
        )

        X_eval = spline.transform(
            X_eval_raw
        )

        p = model.predict_proba(
            X_eval
        )[:, 1]

        adjusted_probability.append(
            float(
                np.mean(p)
            )
        )

    adjusted_probability = np.asarray(
        adjusted_probability,
        dtype=float,
    )

    result = pd.DataFrame(
        {
            "R_FS":
                R_grid,

            "adjusted_detection_probability":
                adjusted_probability,
        }
    )

    result.to_csv(
        TABLE,
        index=False,
    )

    # ========================================================
    # Raw binned relation from script 29
    # ========================================================

    raw_available = RAW_TABLE.exists()

    if raw_available:

        raw = pd.read_csv(
            RAW_TABLE
        )

        raw = raw.loc[
            raw[
                "shown_in_figure"
            ].astype(bool)
        ].copy()

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.8, 4.4)
    )

    if raw_available:

        raw_x = pd.to_numeric(
            raw[
                "R_FS_median"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        raw_y = pd.to_numeric(
            raw[
                "detection_fraction"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        good_raw = (
            np.isfinite(raw_x)
            & np.isfinite(raw_y)
            & (raw_x > 0)
        )

        ax.plot(
            raw_x[
                good_raw
            ],
            raw_y[
                good_raw
            ],
            marker="o",
            markersize=4,
            linewidth=1.0,
            alpha=0.55,
            label="Raw detection fraction",
        )

    ax.plot(
        R_grid,
        adjusted_probability,
        linewidth=2.0,
        label=(
            r"Adjusted for true $t_E$ and $|\pi_E|$"
        ),
    )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label=(
            r"$\rho_{\rm true}=|u_{0,\rm true}|$"
        ),
    )

    ax.axhline(
        float(
            detected.mean()
        ),
        linestyle=":",
        linewidth=1.0,
        label="Overall detection fraction",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        (
            r"True finite-source strength "
            r"$\rho_{\rm true}/|u_{0,\rm true}|$"
        )
    )

    ax.set_ylabel(
        "Parallax detection probability"
    )


    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        fontsize=13,
    )


    fig.savefig(
        FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    save_pdf_companion(fig, FIGURE)

    print()

    print(
        "Adjusted probability range =",
        float(
            np.min(
                adjusted_probability
            )
        ),
        "to",
        float(
            np.max(
                adjusted_probability
            )
        ),
    )

    print(
        "Adjusted P at R_FS nearest 1 =",
        float(
            adjusted_probability[
                np.argmin(
                    np.abs(
                        np.log10(
                            R_grid
                        )
                    )
                )
            ]
        ),
    )

    print()

    print(
        "Saved figure:",
        FIGURE,
    )

    print(
        "Saved adjusted curve:",
        TABLE,
    )


if __name__ == "__main__":
    main()


