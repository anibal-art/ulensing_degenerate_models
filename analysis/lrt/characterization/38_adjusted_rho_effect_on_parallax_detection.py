#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adjusted association between true source size rho and annual-parallax
detectability.

NO simulation.
NO fitting of microlensing light curves.

Scientific question
-------------------
After accounting for the dominant event properties

    tE_true,
    |piE_true|,
    |u0_true|,

does the true source-size parameter rho still provide independent
predictive information about whether annual parallax is detected?

Population
----------
Strictly positive-LRT H1 characterization population:

    detected:
        positive_lrt_class == "parallax_detected"

    confused:
        positive_lrt_class == "confused_fspl_no_parallax"

Statistical strategy
--------------------
We compare two flexible classifiers using stratified K-fold
cross-validation.

Base model:
    log10(tE)
    log10(|piE|)
    log10(|u0|)

Full model:
    log10(tE)
    log10(|piE|)
    log10(|u0|)
    log10(rho)

Estimator:
    sklearn HistGradientBoostingClassifier

This model is deliberately nonlinear and interaction-capable.
The analysis is descriptive/associational, not causal.

Panel A
-------
Observed detection fraction versus rho_true, compared with a
cross-fitted standardized prediction from the full model.

For each cross-validation fold and each rho grid value:

    - rho is fixed to that value for every event in the held-out fold;
    - tE, |piE| and |u0| retain their observed values;
    - predicted detection probabilities are averaged.

This is a marginal-standardization / partial-dependence style
diagnostic.

The rho grid is restricted to the central population support to reduce
extreme extrapolation.

Panel B
-------
Out-of-sample improvement from adding rho:

    Delta log loss =
        logloss(base) - logloss(full)

Positive values mean that rho improves out-of-sample prediction.

The figure also reports the relevant model hyperparameters and
cross-validated metrics.

Outputs
-------
Figure:
analysis/lrt/figures/characterization/
    38_adjusted_rho_effect_on_parallax_detection/
    adjusted_rho_effect_on_parallax_detection.png

Cross-validation metrics:
analysis/lrt/results/characterization/
    38_adjusted_rho_effect_on_parallax_detection/
    cv_metrics.csv

Observed rho bins:
analysis/lrt/results/characterization/
    38_adjusted_rho_effect_on_parallax_detection/
    observed_rho_detection_bins.csv

Adjusted standardized curve:
analysis/lrt/results/characterization/
    38_adjusted_rho_effect_on_parallax_detection/
    adjusted_rho_detection_curve.csv

Cross-fitted predictions:
analysis/lrt/results/characterization/
    38_adjusted_rho_effect_on_parallax_detection/
    cross_fitted_predictions.parquet

Audit:
analysis/lrt/results/characterization/
    38_adjusted_rho_effect_on_parallax_detection/
    model_audit.txt
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


try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss, roc_auc_score
except ImportError as exc:
    raise ImportError(
        "This diagnostic requires scikit-learn. "
        "Run it in the analysis environment containing sklearn."
    ) from exc


# ============================================================================
# Configuration
# ============================================================================

INPUT_PATH = Path(
    "analysis/lrt/results/characterization/18_positive_truth/"
    "h1_positive_truth_characterization.parquet"
)

FIGURE_DIR = Path(
    "analysis/lrt/figures/characterization/"
    "38_adjusted_rho_effect_on_parallax_detection"
)

RESULTS_DIR = Path(
    "analysis/lrt/results/characterization/"
    "38_adjusted_rho_effect_on_parallax_detection"
)

FIGURE_PATH = (
    FIGURE_DIR
    / "adjusted_rho_effect_on_parallax_detection.png"
)

CV_TABLE_PATH = (
    RESULTS_DIR
    / "cv_metrics.csv"
)

OBSERVED_BIN_PATH = (
    RESULTS_DIR
    / "observed_rho_detection_bins.csv"
)

ADJUSTED_CURVE_PATH = (
    RESULTS_DIR
    / "adjusted_rho_detection_curve.csv"
)

PREDICTION_PATH = (
    RESULTS_DIR
    / "cross_fitted_predictions.parquet"
)

AUDIT_PATH = (
    RESULTS_DIR
    / "model_audit.txt"
)


DETECTED_CLASS = "parallax_detected"
CONFUSED_CLASS = "confused_fspl_no_parallax"


# ============================================================================
# Cross-validation configuration
# ============================================================================

N_FOLDS = 5
CV_RANDOM_STATE = 20260923


# ============================================================================
# Flexible classifier configuration
# ============================================================================

LEARNING_RATE = 0.08
MAX_ITER = 250
MAX_LEAF_NODES = 31
MIN_SAMPLES_LEAF = 200
L2_REGULARIZATION = 1.0
MAX_BINS = 255

EARLY_STOPPING = True
VALIDATION_FRACTION = 0.10
N_ITER_NO_CHANGE = 15
TOL = 1.0e-7


# ============================================================================
# Figure / standardized curve configuration
# ============================================================================

N_OBSERVED_BINS = 15
N_RHO_GRID = 40

# Restrict the standardized curve to the central observed support.
RHO_GRID_PERCENTILE_LOW = 2.0
RHO_GRID_PERCENTILE_HIGH = 98.0


# ============================================================================
# Helpers
# ============================================================================

def wilson_interval(k, n, z=1.959963984540054):
    """Wilson 95% confidence interval for a binomial fraction."""

    if n <= 0:
        return np.nan, np.nan

    p = k / n
    denom = 1.0 + z**2 / n

    center = (
        p + z**2 / (2.0 * n)
    ) / denom

    half = (
        z
        / denom
        * np.sqrt(
            p * (1.0 - p) / n
            + z**2 / (4.0 * n**2)
        )
    )

    return center - half, center + half


def make_classifier(random_state):
    """
    Construct one fixed flexible classifier.

    Hyperparameters are intentionally held fixed across folds and across
    the base/full comparison.
    """

    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=LEARNING_RATE,
        max_iter=MAX_ITER,
        max_leaf_nodes=MAX_LEAF_NODES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        l2_regularization=L2_REGULARIZATION,
        max_bins=MAX_BINS,
        early_stopping=EARLY_STOPPING,
        validation_fraction=VALIDATION_FRACTION,
        n_iter_no_change=N_ITER_NO_CHANGE,
        tol=TOL,
        random_state=random_state,
    )


def binary_log_loss_per_event(y, p):
    """
    Per-event binary log loss with numerical clipping.
    """

    p = np.clip(
        np.asarray(p, dtype=float),
        1.0e-12,
        1.0 - 1.0e-12,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    return -(
        y * np.log(p)
        + (1.0 - y) * np.log(1.0 - p)
    )


# ============================================================================
# Main
# ============================================================================

def main():

    print("=" * 105)
    print("38: ADJUSTED EFFECT OF TRUE rho ON PARALLAX DETECTABILITY")
    print("=" * 105)

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
        "tE_true_days",
        "piE_true_amp",
        "u0_true",
        "rho_true",
    ]

    missing = [
        c for c in required
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
    # Explicit characterization population
    # ----------------------------------------------------------------------

    work = df.loc[
        df["positive_lrt_class"].isin(
            [
                DETECTED_CLASS,
                CONFUSED_CLASS,
            ]
        )
    ].copy()

    work["abs_u0_true"] = np.abs(
        work["u0_true"].to_numpy(dtype=float)
    )

    valid = (
        np.isfinite(work["tE_true_days"])
        & (work["tE_true_days"] > 0.0)
        & np.isfinite(work["piE_true_amp"])
        & (work["piE_true_amp"] > 0.0)
        & np.isfinite(work["abs_u0_true"])
        & (work["abs_u0_true"] > 0.0)
        & np.isfinite(work["rho_true"])
        & (work["rho_true"] > 0.0)
    )

    work = work.loc[
        valid
    ].copy()

    work = work.reset_index(
        drop=True
    )

    # ----------------------------------------------------------------------
    # Response
    # ----------------------------------------------------------------------

    y = (
        work["positive_lrt_class"]
        == DETECTED_CLASS
    ).astype(int).to_numpy()

    n_events = len(work)
    detected_fraction = float(
        y.mean()
    )

    print()
    print("Analysis population:")
    print("  N =", f"{n_events:,}")
    print(
        "  detected =",
        f"{int(y.sum()):,}",
    )
    print(
        "  confused =",
        f"{int((1-y).sum()):,}",
    )
    print(
        "  detected fraction =",
        f"{detected_fraction:.6f}",
    )

    # ----------------------------------------------------------------------
    # Log-transformed physical predictors
    # ----------------------------------------------------------------------

    log_tE = np.log10(
        work["tE_true_days"].to_numpy(dtype=float)
    )

    log_piE = np.log10(
        work["piE_true_amp"].to_numpy(dtype=float)
    )

    log_u0 = np.log10(
        work["abs_u0_true"].to_numpy(dtype=float)
    )

    log_rho = np.log10(
        work["rho_true"].to_numpy(dtype=float)
    )

    # Base model:
    # tE, piE, u0
    X_base = np.column_stack(
        [
            log_tE,
            log_piE,
            log_u0,
        ]
    )

    # Full model:
    # same + rho
    X_full = np.column_stack(
        [
            log_tE,
            log_piE,
            log_u0,
            log_rho,
        ]
    )

    base_feature_names = [
        "log10(tE_true_days)",
        "log10(piE_true_amp)",
        "log10(abs_u0_true)",
    ]

    full_feature_names = (
        base_feature_names
        + ["log10(rho_true)"]
    )

    # ----------------------------------------------------------------------
    # Rho grid for marginal standardization
    # ----------------------------------------------------------------------

    rho_low = float(
        np.percentile(
            work["rho_true"],
            RHO_GRID_PERCENTILE_LOW,
        )
    )

    rho_high = float(
        np.percentile(
            work["rho_true"],
            RHO_GRID_PERCENTILE_HIGH,
        )
    )

    rho_grid = np.logspace(
        np.log10(rho_low),
        np.log10(rho_high),
        N_RHO_GRID,
    )

    log_rho_grid = np.log10(
        rho_grid
    )

    print()
    print("Standardized rho curve:")
    print(
        f"  rho percentile support = "
        f"{RHO_GRID_PERCENTILE_LOW:.1f}"
        f"–{RHO_GRID_PERCENTILE_HIGH:.1f}%"
    )
    print(
        f"  rho range = "
        f"{rho_low:.6g} – {rho_high:.6g}"
    )
    print(
        f"  grid points = {N_RHO_GRID}"
    )

    # ----------------------------------------------------------------------
    # Cross-validation
    # ----------------------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    p_base_oof = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    p_full_oof = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    fold_id = np.full(
        n_events,
        -1,
        dtype=int,
    )

    fold_curves = []
    cv_rows = []

    print()
    print("=" * 105)
    print("CROSS-VALIDATION")
    print("=" * 105)

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(
            X_full,
            y,
        ),
        start=1,
    ):

        print()
        print(
            f"Fold {fold}/{N_FOLDS}"
        )
        print(
            "  train =",
            f"{len(train_idx):,}",
        )
        print(
            "  test  =",
            f"{len(test_idx):,}",
        )

        # --------------------------------------------------------------
        # Base model
        # --------------------------------------------------------------

        base_model = make_classifier(
            random_state=(
                CV_RANDOM_STATE
                + 100 * fold
            )
        )

        base_model.fit(
            X_base[train_idx],
            y[train_idx],
        )

        p_base = base_model.predict_proba(
            X_base[test_idx]
        )[:, 1]

        # --------------------------------------------------------------
        # Full model
        # --------------------------------------------------------------

        full_model = make_classifier(
            random_state=(
                CV_RANDOM_STATE
                + 100 * fold
                + 1
            )
        )

        full_model.fit(
            X_full[train_idx],
            y[train_idx],
        )

        p_full = full_model.predict_proba(
            X_full[test_idx]
        )[:, 1]

        # --------------------------------------------------------------
        # Store OOF predictions
        # --------------------------------------------------------------

        p_base_oof[
            test_idx
        ] = p_base

        p_full_oof[
            test_idx
        ] = p_full

        fold_id[
            test_idx
        ] = fold

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

        loss_base = log_loss(
            y[test_idx],
            p_base,
            labels=[0, 1],
        )

        loss_full = log_loss(
            y[test_idx],
            p_full,
            labels=[0, 1],
        )

        delta_loss = (
            loss_base
            - loss_full
        )

        relative_loss_improvement_pct = (
            100.0
            * delta_loss
            / loss_base
        )

        auc_base = roc_auc_score(
            y[test_idx],
            p_base,
        )

        auc_full = roc_auc_score(
            y[test_idx],
            p_full,
        )

        delta_auc = (
            auc_full
            - auc_base
        )

        # --------------------------------------------------------------
        # Standardized rho curve on the held-out fold
        #
        # Keep tE, piE and |u0| at their observed values.
        # Replace only rho.
        # --------------------------------------------------------------

        X_eval = X_full[
            test_idx
        ].copy()

        fold_curve = np.empty(
            N_RHO_GRID,
            dtype=float,
        )

        for j, log_rho_value in enumerate(
            log_rho_grid
        ):

            X_eval[:, 3] = (
                log_rho_value
            )

            probs = (
                full_model.predict_proba(
                    X_eval
                )[:, 1]
            )

            fold_curve[j] = float(
                np.mean(
                    probs
                )
            )

        fold_curves.append(
            fold_curve
        )

        n_iter_base = int(
            getattr(
                base_model,
                "n_iter_",
                MAX_ITER,
            )
        )

        n_iter_full = int(
            getattr(
                full_model,
                "n_iter_",
                MAX_ITER,
            )
        )

        cv_rows.append(
            {
                "fold":
                    fold,

                "n_train":
                    len(train_idx),

                "n_test":
                    len(test_idx),

                "test_detected_fraction":
                    float(
                        y[test_idx].mean()
                    ),

                "logloss_base":
                    loss_base,

                "logloss_full":
                    loss_full,

                "delta_logloss_base_minus_full":
                    delta_loss,

                "relative_logloss_improvement_pct":
                    relative_loss_improvement_pct,

                "auc_base":
                    auc_base,

                "auc_full":
                    auc_full,

                "delta_auc_full_minus_base":
                    delta_auc,

                "n_iter_base":
                    n_iter_base,

                "n_iter_full":
                    n_iter_full,
            }
        )

        print(
            f"  log loss: "
            f"base={loss_base:.8f}, "
            f"full={loss_full:.8f}, "
            f"Delta={delta_loss:+.8e}"
        )

        print(
            f"  relative log-loss improvement = "
            f"{relative_loss_improvement_pct:+.6f}%"
        )

        print(
            f"  AUC: "
            f"base={auc_base:.8f}, "
            f"full={auc_full:.8f}, "
            f"Delta={delta_auc:+.8e}"
        )

        print(
            f"  iterations: "
            f"base={n_iter_base}, "
            f"full={n_iter_full}"
        )

    cv_df = pd.DataFrame(
        cv_rows
    )

    cv_df.to_csv(
        CV_TABLE_PATH,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Validate OOF predictions
    # ----------------------------------------------------------------------

    if (
        np.any(~np.isfinite(p_base_oof))
        or np.any(~np.isfinite(p_full_oof))
        or np.any(fold_id < 1)
    ):
        raise RuntimeError(
            "Cross-fitted predictions are incomplete."
        )

    # ----------------------------------------------------------------------
    # Overall cross-fitted metrics
    # ----------------------------------------------------------------------

    oof_logloss_base = log_loss(
        y,
        p_base_oof,
        labels=[0, 1],
    )

    oof_logloss_full = log_loss(
        y,
        p_full_oof,
        labels=[0, 1],
    )

    oof_delta_logloss = (
        oof_logloss_base
        - oof_logloss_full
    )

    oof_relative_improvement_pct = (
        100.0
        * oof_delta_logloss
        / oof_logloss_base
    )

    oof_auc_base = roc_auc_score(
        y,
        p_base_oof,
    )

    oof_auc_full = roc_auc_score(
        y,
        p_full_oof,
    )

    oof_delta_auc = (
        oof_auc_full
        - oof_auc_base
    )

    # ----------------------------------------------------------------------
    # Per-event paired loss difference
    # ----------------------------------------------------------------------

    event_loss_base = (
        binary_log_loss_per_event(
            y,
            p_base_oof,
        )
    )

    event_loss_full = (
        binary_log_loss_per_event(
            y,
            p_full_oof,
        )
    )

    event_delta_loss = (
        event_loss_base
        - event_loss_full
    )

    # ----------------------------------------------------------------------
    # Save OOF predictions
    # ----------------------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "catalog_row":
                work["catalog_row"].to_numpy(),

            "positive_lrt_class":
                work["positive_lrt_class"].to_numpy(),

            "detected":
                y.astype(bool),

            "cv_fold":
                fold_id,

            "rho_true":
                work["rho_true"].to_numpy(dtype=float),

            "tE_true_days":
                work["tE_true_days"].to_numpy(dtype=float),

            "piE_true_amp":
                work["piE_true_amp"].to_numpy(dtype=float),

            "abs_u0_true":
                work["abs_u0_true"].to_numpy(dtype=float),

            "p_detect_base_oof":
                p_base_oof,

            "p_detect_full_oof":
                p_full_oof,

            "logloss_event_base":
                event_loss_base,

            "logloss_event_full":
                event_loss_full,

            "delta_logloss_event":
                event_delta_loss,
        }
    )

    prediction_df.to_parquet(
        PREDICTION_PATH,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Observed detection fraction versus rho
    #
    # Equal-population bins in log rho make the direct curve stable
    # across the dynamic range.
    # ----------------------------------------------------------------------

    log_rho_series = pd.Series(
        log_rho
    )

    quantile_edges = np.quantile(
        log_rho,
        np.linspace(
            0.0,
            1.0,
            N_OBSERVED_BINS + 1,
        ),
    )

    quantile_edges = np.unique(
        quantile_edges
    )

    if len(quantile_edges) < 3:
        raise RuntimeError(
            "Could not construct enough distinct rho bins."
        )

    rho_bin = pd.cut(
        log_rho_series,
        bins=quantile_edges,
        include_lowest=True,
        ordered=True,
    )

    observed_rows = []

    for interval in rho_bin.cat.categories:

        mask = (
            rho_bin == interval
        ).to_numpy()

        n_bin = int(
            mask.sum()
        )

        if n_bin == 0:
            continue

        k_bin = int(
            y[mask].sum()
        )

        fraction = (
            k_bin / n_bin
        )

        lo, hi = wilson_interval(
            k_bin,
            n_bin,
        )

        rho_median = float(
            np.median(
                work.loc[
                    mask,
                    "rho_true",
                ]
            )
        )

        observed_rows.append(
            {
                "log10_rho_left":
                    float(interval.left),

                "log10_rho_right":
                    float(interval.right),

                "rho_true_median":
                    rho_median,

                "n_total":
                    n_bin,

                "n_detected":
                    k_bin,

                "detection_fraction":
                    fraction,

                "wilson95_low":
                    lo,

                "wilson95_high":
                    hi,
            }
        )

    observed_df = pd.DataFrame(
        observed_rows
    )

    observed_df.to_csv(
        OBSERVED_BIN_PATH,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Combine standardized fold curves
    # ----------------------------------------------------------------------

    fold_curves = np.asarray(
        fold_curves,
        dtype=float,
    )

    adjusted_mean = np.mean(
        fold_curves,
        axis=0,
    )

    adjusted_min = np.min(
        fold_curves,
        axis=0,
    )

    adjusted_max = np.max(
        fold_curves,
        axis=0,
    )

    adjusted_q16 = np.quantile(
        fold_curves,
        0.16,
        axis=0,
    )

    adjusted_q84 = np.quantile(
        fold_curves,
        0.84,
        axis=0,
    )

    adjusted_rows = []

    for j in range(
        N_RHO_GRID
    ):

        row = {
            "rho_true":
                rho_grid[j],

            "adjusted_probability_mean":
                adjusted_mean[j],

            "adjusted_probability_fold_min":
                adjusted_min[j],

            "adjusted_probability_fold_max":
                adjusted_max[j],

            "adjusted_probability_fold_q16":
                adjusted_q16[j],

            "adjusted_probability_fold_q84":
                adjusted_q84[j],
        }

        for fold in range(
            N_FOLDS
        ):
            row[
                f"fold_{fold+1}"
            ] = fold_curves[
                fold,
                j,
            ]

        adjusted_rows.append(
            row
        )

    adjusted_df = pd.DataFrame(
        adjusted_rows
    )

    adjusted_df.to_csv(
        ADJUSTED_CURVE_PATH,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Model summary values
    # ----------------------------------------------------------------------

    mean_delta_logloss = float(
        cv_df[
            "delta_logloss_base_minus_full"
        ].mean()
    )

    std_delta_logloss = float(
        cv_df[
            "delta_logloss_base_minus_full"
        ].std(
            ddof=1
        )
    )

    mean_relative_gain = float(
        cv_df[
            "relative_logloss_improvement_pct"
        ].mean()
    )

    std_relative_gain = float(
        cv_df[
            "relative_logloss_improvement_pct"
        ].std(
            ddof=1
        )
    )

    mean_delta_auc = float(
        cv_df[
            "delta_auc_full_minus_base"
        ].mean()
    )

    iter_base_median = float(
        cv_df[
            "n_iter_base"
        ].median()
    )

    iter_full_median = float(
        cv_df[
            "n_iter_full"
        ].median()
    )

    print()
    print("=" * 105)
    print("OVERALL CROSS-FITTED RESULTS")
    print("=" * 105)

    print(
        f"OOF log loss base = "
        f"{oof_logloss_base:.9f}"
    )

    print(
        f"OOF log loss full = "
        f"{oof_logloss_full:.9f}"
    )

    print(
        f"OOF Delta log loss = "
        f"{oof_delta_logloss:+.9e}"
    )

    print(
        f"OOF relative improvement = "
        f"{oof_relative_improvement_pct:+.6f}%"
    )

    print()

    print(
        f"OOF AUC base = "
        f"{oof_auc_base:.9f}"
    )

    print(
        f"OOF AUC full = "
        f"{oof_auc_full:.9f}"
    )

    print(
        f"OOF Delta AUC = "
        f"{oof_delta_auc:+.9e}"
    )

    print()

    print(
        "Fold mean Delta log loss =",
        f"{mean_delta_logloss:+.9e}",
    )

    print(
        "Fold SD Delta log loss   =",
        f"{std_delta_logloss:.9e}",
    )

    print(
        "Fold mean relative gain  =",
        f"{mean_relative_gain:+.6f}%",
    )

    print(
        "Fold SD relative gain    =",
        f"{std_relative_gain:.6f}%",
    )

    # ----------------------------------------------------------------------
    # Figure
    # ----------------------------------------------------------------------

    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.45, 1.0],
        height_ratios=[1.30, 1.0],
    )
    ax_left = fig.add_subplot(gs[:, 0])
    ax_gain = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis("off")
    axes = np.asarray([ax_left, ax_gain], dtype=object)

    # ------------------------------------------------------------------
    # LEFT: observed and adjusted detection probability
    # ------------------------------------------------------------------

    ax = axes[0]

    obs_x = observed_df[
        "rho_true_median"
    ].to_numpy(dtype=float)

    obs_y = observed_df[
        "detection_fraction"
    ].to_numpy(dtype=float)

    obs_low = observed_df[
        "wilson95_low"
    ].to_numpy(dtype=float)

    obs_high = observed_df[
        "wilson95_high"
    ].to_numpy(dtype=float)

    obs_yerr = np.vstack(
        [
            obs_y - obs_low,
            obs_high - obs_y,
        ]
    )

    ax.errorbar(
        obs_x,
        obs_y,
        yerr=obs_yerr,
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=5,
        capsize=2,
        label="Observed fraction",
    )

    # Individual fold-standardized curves.
    for fold in range(
        N_FOLDS
    ):
        ax.plot(
            rho_grid,
            fold_curves[
                fold
            ],
            linewidth=0.9,
            alpha=0.25,
        )

    ax.fill_between(
        rho_grid,
        adjusted_min,
        adjusted_max,
        alpha=0.15,
        label="Range across CV folds",
    )

    ax.plot(
        rho_grid,
        adjusted_mean,
        linewidth=2.5,
        label=(
            r"Adjusted for "
            r"$t_E$, $|\pi_E|$, and $|u_0|$"
        ),
    )

    ax.axhline(
        detected_fraction,
        linestyle="--",
        linewidth=1.2,
        label=(
            "Overall detected fraction "
            f"({detected_fraction:.3f})"
        ),
    )

    ax.set_xscale(
        "log"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        r"True source-size parameter $\rho_{\rm true}$"
    )

    ax.set_ylabel(
        "Parallax detection probability"
    )

    ax.set_title(
        "Observed and covariate-adjusted detection probability"
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        fontsize=13,
        loc="best",
    )


    # ------------------------------------------------------------------
    # RIGHT: incremental predictive information from rho
    # ------------------------------------------------------------------

    ax = axes[1]

    fold_x = np.arange(
        1,
        N_FOLDS + 1,
    )

    fold_gain = cv_df[
        "relative_logloss_improvement_pct"
    ].to_numpy(dtype=float)

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2,
    )

    ax.plot(
        fold_x,
        fold_gain,
        marker="o",
        linewidth=1.5,
        label="CV folds",
    )

    # Add mean point one step to the right.
    mean_x = (
        N_FOLDS
        + 1.2
    )

    ax.errorbar(
        [mean_x],
        [mean_relative_gain],
        yerr=[
            [
                std_relative_gain
            ],
            [
                std_relative_gain
            ],
        ],
        marker="D",
        markersize=7,
        capsize=4,
        linestyle="none",
        label="Fold mean ± SD",
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
        "Relative log-loss improvement from adding "
        r"$\rho$ [%]"
    )

    ax.set_title(
        r"Incremental out-of-sample information from $\rho$"
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        fontsize=13,
        loc="best",
    )

    # ------------------------------------------------------------------
    # Model information printed directly in the figure
    # ------------------------------------------------------------------

    model_text = (
        (
        "Model and cross-validation\n"
        f"$N={n_events:,}$; detected fraction = {detected_fraction:.4f}\n"
        f"{N_FOLDS}-fold stratified CV; HistGradientBoosting\n"
        f"learning rate = {LEARNING_RATE}; leaves = {MAX_LEAF_NODES}; "
        f"min leaf = {MIN_SAMPLES_LEAF}\n"
        f"$L_2$ = {L2_REGULARIZATION}; max iter = {MAX_ITER}; "
        f"median iter = {iter_base_median:.0f}/{iter_full_median:.0f}\n"
        r"Base: $\log t_E,\ \log|\pi_E|,\ \log|u_0|$"
        "\n"
        r"Full: Base $+\ \log\rho$"
        "\n\n"
        "Cross-fitted performance\n"
        f"log loss: {oof_logloss_base:.6f} $\\rightarrow$ {oof_logloss_full:.6f}\n"
        f"relative gain = {oof_relative_improvement_pct:+.4f}%\n"
        f"AUC: {oof_auc_base:.6f} $\\rightarrow$ {oof_auc_full:.6f}\n"
        f"$\\Delta$AUC = {oof_delta_auc:+.3e}"
    )    )




    ax_info.text(
        0.0,
        1.0,
        model_text,
        transform=ax_info.transAxes,
        ha="left",
        va="top",
        fontsize=13,
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
    # Audit text
    # ----------------------------------------------------------------------

    audit_lines = [
        "=" * 105,
        "38: ADJUSTED TRUE-rho EFFECT ON PARALLAX DETECTABILITY",
        "=" * 105,
        "",
        f"Input file: {INPUT_PATH}",
        f"N events: {n_events:,}",
        f"Detected fraction: {detected_fraction:.9f}",
        "",
        "Population:",
        "  positive-LRT H1 characterization sample only",
        f"  detected class: {DETECTED_CLASS}",
        f"  confused class: {CONFUSED_CLASS}",
        "",
        "Base predictors:",
        *[
            f"  {x}"
            for x in base_feature_names
        ],
        "",
        "Full predictors:",
        *[
            f"  {x}"
            for x in full_feature_names
        ],
        "",
        "Estimator:",
        "  HistGradientBoostingClassifier",
        f"  learning_rate={LEARNING_RATE}",
        f"  max_iter={MAX_ITER}",
        f"  max_leaf_nodes={MAX_LEAF_NODES}",
        f"  min_samples_leaf={MIN_SAMPLES_LEAF}",
        f"  l2_regularization={L2_REGULARIZATION}",
        f"  max_bins={MAX_BINS}",
        f"  early_stopping={EARLY_STOPPING}",
        f"  validation_fraction={VALIDATION_FRACTION}",
        f"  n_iter_no_change={N_ITER_NO_CHANGE}",
        f"  tol={TOL}",
        "",
        "Cross-validation:",
        f"  folds={N_FOLDS}",
        f"  random_state={CV_RANDOM_STATE}",
        "",
        "Standardized rho curve:",
        (
            f"  support percentile = "
            f"{RHO_GRID_PERCENTILE_LOW:.1f}"
            f"–{RHO_GRID_PERCENTILE_HIGH:.1f}"
        ),
        f"  rho_min={rho_low:.12g}",
        f"  rho_max={rho_high:.12g}",
        f"  n_grid={N_RHO_GRID}",
        "",
        "Overall cross-fitted metrics:",
        f"  logloss_base={oof_logloss_base:.12g}",
        f"  logloss_full={oof_logloss_full:.12g}",
        f"  delta_logloss={oof_delta_logloss:.12g}",
        (
            f"  relative_logloss_improvement_pct="
            f"{oof_relative_improvement_pct:.12g}"
        ),
        f"  auc_base={oof_auc_base:.12g}",
        f"  auc_full={oof_auc_full:.12g}",
        f"  delta_auc={oof_delta_auc:.12g}",
        "",
        "Interpretation:",
        (
            "The adjusted curve is a descriptive marginal-standardization "
            "diagnostic. It should not be interpreted as a causal effect "
            "of rho."
        ),
        "",
        (
            "If the adjusted detection probability is nearly flat and "
            "adding rho yields negligible out-of-sample log-loss gain, "
            "then source size provides little additional information after "
            "accounting for tE, |piE| and |u0|."
        ),
        "",
        (
            "If a substantial adjusted rho dependence remains and rho "
            "improves out-of-sample prediction, then finite-source size "
            "retains an independent association with parallax detectability."
        ),
    ]

    AUDIT_PATH.write_text(
        "\n".join(
            audit_lines
        )
    )

    print()
    print("=" * 105)
    print("OUTPUTS")
    print("=" * 105)

    print(
        "Figure:",
        FIGURE_PATH,
    )

    print(
        "CV metrics:",
        CV_TABLE_PATH,
    )

    print(
        "Observed bins:",
        OBSERVED_BIN_PATH,
    )

    print(
        "Adjusted curve:",
        ADJUSTED_CURVE_PATH,
    )

    print(
        "OOF predictions:",
        PREDICTION_PATH,
    )

    print(
        "Audit:",
        AUDIT_PATH,
    )

    print()
    print("=" * 105)
    print("DONE")
    print("=" * 105)


if __name__ == "__main__":
    main()


