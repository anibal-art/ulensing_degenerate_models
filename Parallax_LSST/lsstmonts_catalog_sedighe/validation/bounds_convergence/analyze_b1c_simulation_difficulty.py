#!/usr/bin/env python3

"""
Experiment B1c -- simulation-aware event difficulty (DIAGNOSTIC /
CANDIDATE GENERATION).

STATUS: DIAGNOSTIC / CANDIDATE GENERATION, development-set analysis
(extreme100, N=100). Not a production policy. See
ADAPTIVE_H0_ANALYSIS.md, Experiment B1c.

SCOPE NOTE -- why truth is allowed here
    This project's goal is a Monte Carlo simulation pipeline for
    parallax-detectability science, not a fitter deployed on real
    survey data. In that setting, TRUE simulation parameters and
    generated-light-curve properties are genuinely available, in
    memory, before any H0 fit ever runs -- they are not "leakage" in
    the sense that matters here. What must never be used to decide
    computational effort is the OUTCOME we are trying to estimate:
    chi2_oracle_52, delta_vs_oracle, or which strategy a posteriori
    won -- those require the exhaustive analysis this experiment is
    trying to avoid running for every future event, and are used here
    strictly as OFFLINE evaluation labels (see LEAKAGE BOUNDARY below).

PURPOSE
    Test whether truth/simulation and generated-light-curve
    properties, ALL available before any H0 fit runs, predict (a)
    whether a small fixed H0 base set (from Experiment B1) already
    reaches the base-domain oracle within 0.1, (b) which family
    (coordinate mode) of additional strategy would rescue the event
    if not, and (c) whether such "pre-H0" information adds anything
    beyond what Experiment B1's own fixed-cost fit diagnostics already
    show once the small base set has actually been run.

LEAKAGE BOUNDARY (the temporal argument, traced from code -- not
assumed)
    Traced in this session directly from
    validation/bounds_audit/run_bounds_audit_refit_core.py:
      - load_case() (lines ~1649-1745) loads, per event: the H5
        light-curve file via load_h5_lightcurves() (lines 1540-1606,
        replicated read-only below) and the truth parquet
        (true_rr_manual_*.parquet).
      - THEN, still before any H0 bounds/starts/optimizer setup, the
        DATA-DRIVEN t0 BOUND block (lines ~2091-2168) computes
        data_t_min/data_t_max/t_obs purely from meta["curves"] -- this
        is proof, in the fitter's own code, that Tobs (and by
        extension Ndata, per-band counts, sampling/cadence, and
        photometric-precision quantities derived from the same
        arrays) are available BEFORE H0 starts, not after.
      - meta["curves"] is passed UNMODIFIED (no further filtering) to
        fit_lc.fit_rubin_roman(...) as the actual H0 fit input (lines
        ~2246-2264) -- so Ndata_H0_input, defined identically here,
        is exactly what H0 sees, not an approximation of it.
    Everything computed by THIS script from the H5 + truth parquet +
    extreme100_refit_manifest.csv is therefore legitimately "pre-H0"
    for a future simulation-aware policy. chi2_oracle_52 and anything
    derived from the Experiment-0/B1 exhaustive analysis is NOT --
    those are read here ONLY to build offline evaluation labels
    (Phase B onward), never as a feature (Phase A never touches them).

Ndata_H0_input -- resolved definition (do not use n_data_true)
    Ndata_H0_input := sum over bands of len(meta["curves"][band]),
    i.e. exactly load_h5_lightcurves()'s own filter
    (photometry_keep & isfinite(t,mag,err_mag) & err_mag>0), summed
    across the 7 bands the core fitter itself iterates
    (W149,u,g,r,i,z,y). Verified in this session against 4 events:
    this quantity equals the truth parquet's `fit_n_points_total` (and
    `phot_n_total`) in every case checked; it does NOT always equal
    `n_data_true`, which is computed at an earlier/external simulation
    stage outside this repository and can diverge (e.g. catalog_row
    35101: Ndata_H0_input=159, n_data_true=129). This script computes
    Ndata_H0_input from the H5 (primary source) and cross-checks it
    against `fit_n_points_total` from the parquet, flagging any
    mismatch explicitly rather than trusting either source blindly.
    `n_data_true` is not used as a sampling feature anywhere here.

INPUTS
    Local, not repository-tracked (same situation as the Gate-1 raw
    fit directories used by aggregate_gate1_oracle_diagnostics.py; the
    root is configurable, see ARTIFACTS_ROOT below):
      <root>/<catalog_row>/models/<field>/<realization_key>/Event_*.h5
      <root>/<catalog_row>/results/<field>/<realization_key>/true/true_rr_manual_*.parquet
    Repository-tracked:
      validation/bounds_convergence/data/extreme100_refit_manifest.csv
          (canonical truth u0/tE/rho/piEN/piEE -- the same "truth"
          anchor the fitter itself uses)
      validation/bounds_convergence/results/b1_predictors_per_event.csv
          (Experiment B1's per-event safe/unsafe + delta labels, and
          its own fixed-cost fit diagnostics, for candidate_rank==1)
      validation/bounds_convergence/results/b1_fixed_budget_candidate_sets.csv
          (which (mode, strategy_id) belong to the k=4 candidate_rank==1
          set, for reference/documentation only)
      validation/bounds_convergence/results/b1_rescue_matrix.csv
          (k=4, candidate_rank==1 residual-event x remaining-strategy
          rescue outcomes, for the rescue-family labels)

OUTPUTS (validation/bounds_convergence/results/)
    b1c_event_features.csv       -- Phase A: one row per event.
    b1c_feature_vs_difficulty.csv -- Phase B/C: univariate association
        of every feature against unsafe_k4/k5/k6, tagged with
        scientific_risk_variable so Phase C's mandatory stratification
        is directly extractable from this one table.
    b1c_difficulty_models_cv.csv -- Phase D/F: CV results for the
        logistic-regression / shallow-tree models, across the A/B/C
        feature-scenario comparison.
    b1c_rescue_family_associations.csv -- Phase E.
    (b1c_combined_feature_models.csv is NOT written separately: Phase
    F's three scenarios (A/B/C) are rows within
    b1c_difficulty_models_cv.csv, tagged by `feature_scenario` -- a
    separate file would duplicate the same table with no new
    information, so it is not created, per instruction not to create
    outputs that do not add information.)

WHAT QUESTION THIS ANSWERS
    See PURPOSE. This does not select or freeze any production
    policy; Phase D/F models are explicitly diagnostic-only given
    N=100.

ORACLE USAGE
    chi2_oracle_52 / delta_vs_oracle / safe-unsafe are read ONLY from
    Experiment B1's already-computed, already-integrity-checked
    b1_predictors_per_event.csv and b1_rescue_matrix.csv, to build
    OFFLINE labels (Phase A's feature columns never include them).
    This remains extreme100 development-set analysis; Gate 3 is
    required before any generalization claim.

NO FITS, NO pyLIMA, NO CHE: this script only reads already-generated
local artifacts and already-committed B1 results.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RESULTS_DIR = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results"
)
DATA_DIR = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/data"
)

MANIFEST_PATH = DATA_DIR / "extreme100_refit_manifest.csv"
B1_PREDICTORS_PATH = RESULTS_DIR / "b1_predictors_per_event.csv"
B1_CANDIDATES_PATH = RESULTS_DIR / "b1_fixed_budget_candidate_sets.csv"
B1_RESCUE_MATRIX_PATH = RESULTS_DIR / "b1_rescue_matrix.csv"

OUT_EVENT_FEATURES = RESULTS_DIR / "b1c_event_features.csv"
OUT_FEATURE_VS_DIFFICULTY = (
    RESULTS_DIR / "b1c_feature_vs_difficulty.csv"
)
OUT_MODELS_CV = RESULTS_DIR / "b1c_difficulty_models_cv.csv"
OUT_RESCUE_ASSOC = RESULTS_DIR / "b1c_rescue_family_associations.csv"

DEFAULT_ARTIFACTS_ROOT = Path(
    "~/Downloads/hidden_parallax/hidden_parallax_refit_test/"
    "artifacts/extreme100"
).expanduser()

BANDS = ["W149", "u", "g", "r", "i", "z", "y"]

TOL = 0.1
RESCUE_REFERENCE_K = 4
RESCUE_REFERENCE_CANDIDATE_RANK = 1
MODES = ["physical", "log_te", "log_rho", "log_te_rho"]

CORE_TRUTH_FEATURES = [
    "log10_tE_true",
    "log10_rho_true",
    "abs_u0_true",
    "abs_u0_over_rho",
    "piE_true",
]

CORE_SAMPLING_FEATURES = [
    "Tobs_over_tE",
    "Ndata_H0_input",
    "n_filters_H0",
    "frac_within_0p5_tE",
    "frac_within_1_tE",
    "frac_within_2_tE",
    "pre_post_imbalance",
    "median_cadence_across_bands",
    "max_gap_within_1tE_across_bands",
]

CORE_PHOTOMETRY_FEATURES = [
    "median_err_mag",
    "median_abs_flux_over_err_flux",
]

CORE_BRIGHTNESS_FEATURES = [
    "brightest_source_mag",
    "median_source_mag_visible",
    "min_source_fraction_visible",
    "median_source_fraction_visible",
]

CORE_FEATURES = (
    CORE_TRUTH_FEATURES
    + CORE_SAMPLING_FEATURES
    + CORE_PHOTOMETRY_FEATURES
    + CORE_BRIGHTNESS_FEATURES
)

EXPLORATORY_FEATURES = [
    "lens_mass_msun",
    "D_L",
    "D_S",
    "mu_rel",
    "thetaE_mas",
    "source_radius_rsun_catalog",
    "l_deg",
    "b_deg",
    "piEN_true",
    "piEE_true",
]

SCIENTIFIC_RISK_FEATURES = {
    "log10_tE_true": "tE_true",
    "piE_true": "piE_true",
    "log10_rho_true": "rho_true",
    "abs_u0_true": "abs_u0_true",
    "median_source_mag_visible": "brightness",
    "median_source_fraction_visible": "blending",
}


# ============================================================
# PHASE A -- event feature table
# ============================================================


def _find_one(patterns: list[Path]) -> Path:
    for p in patterns:
        if p.exists():
            return p
    raise RuntimeError(f"None of these paths exist: {patterns}")


def _glob_one(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly 1 {label} match for {pattern!r} "
            f"under {root}, found {len(matches)}: {matches}"
        )
    return matches[0]


def load_h5_lightcurves(path: Path) -> dict[str, np.ndarray]:
    """
    Read-only replica of run_bounds_audit_refit_core.py:
    load_h5_lightcurves (lines 1540-1606). Same filter, same band
    list, same column order (time, mag, err_mag). Not imported from
    core to keep this script fully decoupled from any production code
    path; kept in exact correspondence, see module docstring.
    """

    curves = {band: np.empty((0, 3), dtype=float) for band in BANDS}

    with h5py.File(path, "r") as f:
        for band in BANDS:
            if band not in f:
                continue

            g = f[band]

            t = np.asarray(g["time"], dtype=float)
            m = np.asarray(g["mag"], dtype=float)
            e = np.asarray(g["err_mag"], dtype=float)

            if "photometry_keep" in g:
                keep = np.asarray(g["photometry_keep"], dtype=bool)
            else:
                keep = np.ones(len(t), dtype=bool)

            good = (
                keep
                & np.isfinite(t)
                & np.isfinite(m)
                & np.isfinite(e)
                & (e > 0)
            )

            curves[band] = np.column_stack(
                [t[good], m[good], e[good]]
            )

    return curves


def load_h5_flux_precision(
    path: Path,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Additional per-band arrays not carried by load_h5_lightcurves
    (flux, err_flux), read with the identical keep-mask logic, for
    the photometric-precision features. Kept as a separate reader
    (rather than folded into load_h5_lightcurves) to keep that
    function an exact, minimal replica of the core fitter's own
    reader.
    """

    out = {}

    with h5py.File(path, "r") as f:
        for band in BANDS:
            if band not in f:
                continue

            g = f[band]

            t = np.asarray(g["time"], dtype=float)
            m = np.asarray(g["mag"], dtype=float)
            e = np.asarray(g["err_mag"], dtype=float)

            if "photometry_keep" in g:
                keep = np.asarray(g["photometry_keep"], dtype=bool)
            else:
                keep = np.ones(len(t), dtype=bool)

            good = (
                keep
                & np.isfinite(t)
                & np.isfinite(m)
                & np.isfinite(e)
                & (e > 0)
            )

            if "flux" in g and "err_flux" in g:
                flux = np.asarray(g["flux"], dtype=float)[good]
                err_flux = np.asarray(g["err_flux"], dtype=float)[
                    good
                ]
            else:
                flux = np.array([])
                err_flux = np.array([])

            out[band] = {
                "err_mag": e[good],
                "flux": flux,
                "err_flux": err_flux,
            }

    return out


def build_event_row(
    catalog_row: int,
    root: Path,
    truth_row: pd.Series,
) -> dict:

    event_dir = root / str(catalog_row)

    h5_path = _glob_one(
        event_dir, "models/*/*/Event_*.h5", "H5 lightcurve"
    )
    parquet_path = _glob_one(
        event_dir,
        "results/*/*/true/true_rr_manual_*.parquet",
        "truth parquet",
    )

    truth_parquet = pd.read_parquet(parquet_path).iloc[0]

    curves = load_h5_lightcurves(h5_path)
    flux_prec = load_h5_flux_precision(h5_path)

    t0_true = float(truth_parquet["t0"])
    tE_true = float(truth_row["true_tE"])
    rho_true = float(truth_row["true_rho"])
    u0_true = float(truth_row["true_u0"])
    piEN_true = float(truth_row["true_piEN"])
    piEE_true = float(truth_row["true_piEE"])

    # --- pooled times across all bands, matching the core fitter's
    # own DATA-DRIVEN t0 BOUND block exactly (pool first, then
    # min/max) ---
    all_times = np.concatenate(
        [curves[b][:, 0] for b in BANDS if len(curves[b])]
    )

    if len(all_times) == 0:
        raise RuntimeError(
            f"catalog_row={catalog_row}: no photometry found in "
            f"{h5_path}."
        )

    Tobs_global = float(all_times.max() - all_times.min())

    ndata_per_band = {b: len(curves[b]) for b in BANDS}
    Ndata_H0_input = int(sum(ndata_per_band.values()))
    n_filters_H0 = int(sum(1 for n in ndata_per_band.values() if n > 0))

    visible_bands = [b for b in BANDS if ndata_per_band[b] > 0]

    n_before_t0 = int((all_times < t0_true).sum())
    n_after_t0 = int((all_times >= t0_true).sum())

    def _frac_within(mult: float) -> float:
        window = mult * tE_true
        n = int((np.abs(all_times - t0_true) <= window).sum())
        return n / Ndata_H0_input if Ndata_H0_input else float("nan")

    frac_within_0p5_tE = _frac_within(0.5)
    frac_within_1_tE = _frac_within(1.0)
    frac_within_2_tE = _frac_within(2.0)

    pre_post_imbalance = (
        (n_before_t0 - n_after_t0) / Ndata_H0_input
        if Ndata_H0_input
        else float("nan")
    )

    # --- per-band cadence / gap-within-1tE, aggregated across bands
    # (median of per-band values) -- never mixing interleaved-band
    # timestamps ---
    per_band_cadence = []
    per_band_max_gap = []

    for b in visible_bands:
        t_b = np.sort(curves[b][:, 0])

        if len(t_b) >= 2:
            per_band_cadence.append(float(np.median(np.diff(t_b))))

        window_mask = np.abs(t_b - t0_true) <= tE_true
        t_b_win = t_b[window_mask]
        if len(t_b_win) >= 2:
            per_band_max_gap.append(float(np.max(np.diff(t_b_win))))

    median_cadence_across_bands = (
        float(np.median(per_band_cadence))
        if per_band_cadence
        else float("nan")
    )
    max_gap_within_1tE_across_bands = (
        float(np.max(per_band_max_gap))
        if per_band_max_gap
        else float("nan")
    )

    # --- photometric precision, per band then median-of-medians ---
    per_band_median_err_mag = []
    per_band_median_flux_prec = []

    for b in visible_bands:
        e = flux_prec[b]["err_mag"]
        if len(e):
            per_band_median_err_mag.append(float(np.median(e)))

        flux = flux_prec[b]["flux"]
        err_flux = flux_prec[b]["err_flux"]
        if len(flux) and len(err_flux):
            ratio = np.abs(flux) / err_flux
            ratio = ratio[np.isfinite(ratio)]
            if len(ratio):
                per_band_median_flux_prec.append(
                    float(np.median(ratio))
                )

    median_err_mag = (
        float(np.median(per_band_median_err_mag))
        if per_band_median_err_mag
        else float("nan")
    )
    median_abs_flux_over_err_flux = (
        float(np.median(per_band_median_flux_prec))
        if per_band_median_flux_prec
        else float("nan")
    )

    # --- brightness / blending, restricted to visible bands only ---
    source_mag = {
        b: truth_parquet.get(f"source_mag_{b}", np.nan)
        for b in ["u", "g", "r", "i", "z", "y"]
    }
    source_fraction = {
        b: truth_parquet.get(f"source_fraction_{b}", np.nan)
        for b in ["u", "g", "r", "i", "z", "y"]
    }

    visible_lsst_bands = [
        b for b in ["u", "g", "r", "i", "z", "y"] if b in visible_bands
    ]

    mags_visible = [
        source_mag[b]
        for b in visible_lsst_bands
        if pd.notna(source_mag[b])
    ]
    fracs_visible = [
        source_fraction[b]
        for b in visible_lsst_bands
        if pd.notna(source_fraction[b])
    ]

    brightest_source_mag = (
        float(np.min(mags_visible)) if mags_visible else float("nan")
    )
    median_source_mag_visible = (
        float(np.median(mags_visible)) if mags_visible else float("nan")
    )
    min_source_fraction_visible = (
        float(np.min(fracs_visible)) if fracs_visible else float("nan")
    )
    median_source_fraction_visible = (
        float(np.median(fracs_visible))
        if fracs_visible
        else float("nan")
    )

    # --- consistency check: Ndata_H0_input vs fit_n_points_total ---
    fit_n_points_total = truth_parquet.get(
        "fit_n_points_total", np.nan
    )
    ndata_consistency_ok = (
        bool(int(fit_n_points_total) == Ndata_H0_input)
        if pd.notna(fit_n_points_total)
        else False
    )
    ndata_mismatch_abs = (
        abs(int(fit_n_points_total) - Ndata_H0_input)
        if pd.notna(fit_n_points_total)
        else float("nan")
    )

    row = {
        "catalog_row": int(catalog_row),
        # --- raw audit columns ---
        "t0_true": t0_true,
        "tE_true": tE_true,
        "rho_true": rho_true,
        "u0_true": u0_true,
        "piEN_true": piEN_true,
        "piEE_true": piEE_true,
        "Tobs_global": Tobs_global,
        "fit_n_points_total_parquet": (
            float(fit_n_points_total)
            if pd.notna(fit_n_points_total)
            else float("nan")
        ),
        "ndata_h0_input_matches_fit_n_points_total": (
            ndata_consistency_ok
        ),
        "ndata_mismatch_abs": ndata_mismatch_abs,
        # --- core: truth / geometry ---
        "log10_tE_true": float(np.log10(tE_true)),
        "log10_rho_true": float(np.log10(rho_true)),
        "abs_u0_true": float(abs(u0_true)),
        "abs_u0_over_rho": float(abs(u0_true) / rho_true),
        "piE_true": float(
            np.hypot(piEN_true, piEE_true)
        ),
        # --- core: sampling ---
        "Tobs_over_tE": Tobs_global / tE_true,
        "Ndata_H0_input": Ndata_H0_input,
        "n_filters_H0": n_filters_H0,
        "frac_within_0p5_tE": frac_within_0p5_tE,
        "frac_within_1_tE": frac_within_1_tE,
        "frac_within_2_tE": frac_within_2_tE,
        "pre_post_imbalance": pre_post_imbalance,
        "median_cadence_across_bands": median_cadence_across_bands,
        "max_gap_within_1tE_across_bands": (
            max_gap_within_1tE_across_bands
        ),
        # --- core: photometry ---
        "median_err_mag": median_err_mag,
        "median_abs_flux_over_err_flux": (
            median_abs_flux_over_err_flux
        ),
        # --- core: brightness / blending ---
        "brightest_source_mag": brightest_source_mag,
        "median_source_mag_visible": median_source_mag_visible,
        "min_source_fraction_visible": min_source_fraction_visible,
        "median_source_fraction_visible": (
            median_source_fraction_visible
        ),
        # --- exploratory ---
        "lens_mass_msun": float(
            truth_parquet.get("lens_mass_msun", np.nan)
        ),
        "D_L": float(truth_parquet.get("D_L", np.nan)),
        "D_S": float(truth_parquet.get("D_S", np.nan)),
        "mu_rel": float(
            truth_parquet.get("mu_rel_for_pipeline_masyr", np.nan)
        ),
        "thetaE_mas": float(truth_parquet.get("thetaE_mas", np.nan)),
        "source_radius_rsun_catalog": float(
            truth_parquet.get("source_radius_rsun_catalog", np.nan)
        ),
        "l_deg": float(truth_parquet.get("l_deg", np.nan)),
        "b_deg": float(truth_parquet.get("b_deg", np.nan)),
    }

    for b in BANDS:
        row[f"Ndata_{b}"] = ndata_per_band[b]

    return row


def build_event_features(root: Path) -> pd.DataFrame:

    manifest = pd.read_csv(MANIFEST_PATH)

    rows = []
    for _, r in manifest.iterrows():
        rows.append(
            build_event_row(int(r["catalog_row"]), root, r)
        )

    return pd.DataFrame.from_records(rows)


def audit_event_features(df: pd.DataFrame) -> None:

    print("=" * 70)
    print("PHASE A AUDIT")
    print("=" * 70)

    errors = []

    if len(df) != 100:
        errors.append(f"expected 100 rows, found {len(df)}")

    if df["catalog_row"].duplicated().any():
        errors.append("duplicate catalog_row values")

    if df["catalog_row"].nunique() != len(df):
        errors.append("catalog_row is not unique")

    n_mismatch = int(
        (~df["ndata_h0_input_matches_fit_n_points_total"]).sum()
    )
    print(
        f"n_rows={len(df)}, n_distinct_catalog_row="
        f"{df['catalog_row'].nunique()}"
    )
    print(
        "Ndata_H0_input vs fit_n_points_total: "
        f"{len(df) - n_mismatch}/{len(df)} match; "
        f"{n_mismatch} mismatch"
    )
    if n_mismatch:
        mism = df[~df["ndata_h0_input_matches_fit_n_points_total"]]
        print(
            "  mismatching catalog_row(s): "
            f"{mism['catalog_row'].tolist()}, "
            f"abs diffs: {mism['ndata_mismatch_abs'].tolist()}"
        )

    print()
    print("Missingness by feature (core + exploratory):")
    for feat in CORE_FEATURES + EXPLORATORY_FEATURES:
        n_na = int(df[feat].isna().sum())
        n_nonfinite = int(
            (~np.isfinite(df[feat].astype(float))).sum()
        )
        if n_na or n_nonfinite:
            print(f"  {feat}: n_na={n_na}, n_nonfinite={n_nonfinite}")

    if errors:
        raise RuntimeError(
            "Phase A structural integrity violation(s): "
            + "; ".join(errors)
        )

    print()
    print("Phase A audit: PASSED (no structural violations)")
    print()


# ============================================================
# PHASE-A/B/C LABELS (offline only)
# ============================================================


def load_labels() -> pd.DataFrame:

    pred = pd.read_csv(B1_PREDICTORS_PATH)
    pred1 = pred[pred["candidate_rank"] == 1]

    pieces = []
    for k in [2, 3, 4, 5, 6]:
        sub = pred1[pred1["k"] == k][
            ["catalog_row", "safe", "delta_chi2_vs_base_domain_oracle"]
        ].rename(
            columns={
                "safe": f"safe_k{k}",
                "delta_chi2_vs_base_domain_oracle": (
                    f"delta_vs_oracle_k{k}"
                ),
            }
        )
        pieces.append(sub.set_index("catalog_row"))

    labels = pd.concat(pieces, axis=1).reset_index()

    for k in [4, 5, 6]:
        labels[f"unsafe_k{k}"] = ~labels[f"safe_k{k}"]

    def _min_budget(row):
        for k in [2, 3, 4, 5, 6]:
            if bool(row[f"safe_k{k}"]):
                return k
        return None

    labels["min_candidate_budget_reaching_tol"] = labels.apply(
        _min_budget, axis=1
    )

    keep_cols = (
        ["catalog_row"]
        + [f"unsafe_k{k}" for k in [4, 5, 6]]
        + [f"delta_vs_oracle_k{k}" for k in [4, 5, 6]]
        + ["min_candidate_budget_reaching_tol"]
    )

    return labels[keep_cols]


def load_rescue_labels() -> pd.DataFrame:

    rescue = pd.read_csv(B1_RESCUE_MATRIX_PATH)
    rescue = rescue[
        (rescue["k"] == RESCUE_REFERENCE_K)
        & (rescue["candidate_rank"] == RESCUE_REFERENCE_CANDIDATE_RANK)
    ].copy()

    rescue["mode"] = rescue["remaining_strategy"].str.split(
        "/", n=1
    ).str[0]

    records = []
    for catalog_row, g in rescue.groupby("catalog_row"):
        rec = {"catalog_row": int(catalog_row)}
        for mode in MODES:
            gm = g[g["mode"] == mode]
            n_rescuing = int(gm["rescued"].sum())
            rec[f"rescued_by_{mode}"] = bool(n_rescuing > 0)
            rec[f"n_rescuing_strategies_{mode}"] = n_rescuing
        rec["n_rescuing_modes"] = int(
            sum(
                rec[f"rescued_by_{mode}"] for mode in MODES
            )
        )
        records.append(rec)

    return pd.DataFrame.from_records(records)


# ============================================================
# PHASE B/C -- univariate descriptive + scientific-risk tagging
# ============================================================


def _auc_and_groups(
    values: pd.Series, unsafe: pd.Series
) -> dict:

    mask = values.notna()
    v = values[mask]
    u = unsafe[mask]

    safe_vals = v[~u]
    unsafe_vals = v[u]

    n_safe = len(safe_vals)
    n_unsafe = len(unsafe_vals)

    if n_safe == 0 or n_unsafe == 0:
        auc = float("nan")
    else:
        stat, _ = mannwhitneyu(
            unsafe_vals, safe_vals, alternative="two-sided"
        )
        auc = float(stat) / (n_safe * n_unsafe)

    return {
        "n_obs": int(mask.sum()),
        "n_safe": n_safe,
        "n_unsafe": n_unsafe,
        "median_safe": (
            float(safe_vals.median()) if n_safe else float("nan")
        ),
        "median_unsafe": (
            float(unsafe_vals.median()) if n_unsafe else float("nan")
        ),
        "q25_safe": (
            float(safe_vals.quantile(0.25)) if n_safe else float("nan")
        ),
        "q75_safe": (
            float(safe_vals.quantile(0.75)) if n_safe else float("nan")
        ),
        "q25_unsafe": (
            float(unsafe_vals.quantile(0.25))
            if n_unsafe
            else float("nan")
        ),
        "q75_unsafe": (
            float(unsafe_vals.quantile(0.75))
            if n_unsafe
            else float("nan")
        ),
        "auc_unsafe_gt_safe": auc,
    }


def descriptive_univariate(
    features: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:

    df = features.merge(labels, on="catalog_row", how="inner")

    records = []

    all_features = CORE_FEATURES + EXPLORATORY_FEATURES

    for feat in all_features:

        feature_group = (
            "core" if feat in CORE_FEATURES else "exploratory"
        )
        is_risk = feat in SCIENTIFIC_RISK_FEATURES
        risk_variable = SCIENTIFIC_RISK_FEATURES.get(feat, "")

        for k in [4, 5, 6]:

            unsafe = df[f"unsafe_k{k}"]
            stats = _auc_and_groups(df[feat], unsafe)

            spearman_rho, spearman_p = spearmanr(
                df[feat], df[f"delta_vs_oracle_k{k}"],
                nan_policy="omit",
            )

            records.append(
                {
                    "feature": feat,
                    "feature_group": feature_group,
                    "scientific_risk_variable": is_risk,
                    "risk_category": risk_variable,
                    "k": k,
                    **stats,
                    "spearman_rho_vs_delta": float(spearman_rho),
                    "spearman_p_vs_delta": float(spearman_p),
                }
            )

    return pd.DataFrame.from_records(records)


# ============================================================
# PHASE D/F -- exploratory models
# ============================================================


B1_FIT_DIAGNOSTIC_FEATURES = [
    "delta_spread",
    "abs_log_ratio_tE",
    "abs_log_ratio_rho",
    "diff_t0_norm_by_tE",
    "n_modes_agreeing",
]


def load_b1_fit_diagnostics(k: int) -> pd.DataFrame:

    pred = pd.read_csv(
        RESULTS_DIR / "b1_predictors_per_event.csv"
    )
    sub = pred[
        (pred["k"] == k) & (pred["candidate_rank"] == 1)
    ][["catalog_row"] + B1_FIT_DIAGNOSTIC_FEATURES]

    return sub


def _cv_scenario(
    X: pd.DataFrame, y: np.ndarray, scenario_name: str, k_label: int
) -> list[dict]:

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    records = []

    if n_pos < 3 or n_neg < 3:
        records.append(
            {
                "feature_scenario": scenario_name,
                "k": k_label,
                "model": "N/A",
                "n_pos": n_pos,
                "n_neg": n_neg,
                "status": "INCONCLUSIVE (too few positives/negatives)",
            }
        )
        return records

    n_splits = min(5, n_pos, n_neg)
    n_repeats = 10

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=20260914
    )

    X_arr = X.to_numpy(dtype=float)
    n_nan_total = int(np.isnan(X_arr).sum())

    prevalence = n_pos / (n_pos + n_neg)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "shallow_decision_tree": DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=20260914,
        ),
    }

    for model_name, model in models.items():

        roc_aucs = []
        pr_aucs = []

        for train_idx, test_idx in cv.split(X_arr, y):

            X_train, X_test = (
                X_arr[train_idx].copy(),
                X_arr[test_idx].copy(),
            )
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_test)) < 2:
                continue

            # DETERMINISM / NO-LEAKAGE: median imputation is fit on
            # the TRAINING fold only and applied to both splits --
            # never on the full dataset before CV. In this dataset
            # n_nan_total==0 for every scenario tested (see the audit
            # printout below), so this branch is inert here, but the
            # code must not depend on that being true.
            if n_nan_total:
                train_median = np.nanmedian(X_train, axis=0)
                train_nan = np.where(np.isnan(X_train))
                X_train[train_nan] = np.take(
                    train_median, train_nan[1]
                )
                test_nan = np.where(np.isnan(X_test))
                X_test[test_nan] = np.take(train_median, test_nan[1])

            if model_name == "logistic_regression":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)

            # AUDIT (point 1): positive class is unsafe==True (y=1);
            # verify predict_proba's column 1 is P(class==1) via
            # model.classes_ rather than assuming column order.
            positive_col = int(
                np.where(model.classes_ == 1)[0][0]
            )
            proba = model.predict_proba(X_test)[:, positive_col]

            roc_aucs.append(roc_auc_score(y_test, proba))
            pr_aucs.append(
                average_precision_score(y_test, proba)
            )

        roc_auc_mean = float(np.mean(roc_aucs))
        pr_auc_mean = float(np.mean(pr_aucs))

        print(
            f"  [CV audit] k={k_label} model={model_name} "
            f"scenario={scenario_name}: "
            f"prevalence(unsafe)={prevalence:.3f} "
            f"roc_auc={roc_auc_mean:.3f} "
            f"pr_auc={pr_auc_mean:.3f} "
            f"pr_auc_baseline(=prevalence)={prevalence:.3f} "
            f"n_nan_total={n_nan_total}"
        )

        records.append(
            {
                "feature_scenario": scenario_name,
                "k": k_label,
                "model": model_name,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "prevalence_unsafe": prevalence,
                "n_splits": n_splits,
                "n_repeats": n_repeats,
                "n_fold_evals": len(roc_aucs),
                "roc_auc_mean": roc_auc_mean,
                "roc_auc_std": float(np.std(roc_aucs)),
                "pr_auc_mean": pr_auc_mean,
                "pr_auc_std": float(np.std(pr_aucs)),
                "pr_auc_baseline_prevalence": prevalence,
                "status": "DIAGNOSTIC (N=100, not validated)",
            }
        )

    return records


def exploratory_models(
    features: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:

    all_records = []

    for k in [4, 5, 6]:

        fit_diag = load_b1_fit_diagnostics(k)

        merged = (
            features.merge(labels, on="catalog_row", how="inner")
            .merge(fit_diag, on="catalog_row", how="inner")
        )

        y = merged[f"unsafe_k{k}"].to_numpy(dtype=int)

        scenario_A = merged[CORE_FEATURES]
        scenario_B = merged[B1_FIT_DIAGNOSTIC_FEATURES]
        scenario_C = merged[CORE_FEATURES + B1_FIT_DIAGNOSTIC_FEATURES]

        all_records += _cv_scenario(
            scenario_A, y, "A_core_simulation_pre_H0_only", k
        )
        all_records += _cv_scenario(
            scenario_B, y, "B_b1_fit_diagnostics_only", k
        )
        all_records += _cv_scenario(
            scenario_C, y, "C_combined", k
        )

    return pd.DataFrame.from_records(all_records)


# ============================================================
# PHASE E -- rescue-family associations
# ============================================================


def rescue_family_associations(
    features: pd.DataFrame, rescue_labels: pd.DataFrame
) -> pd.DataFrame:

    df = features.merge(rescue_labels, on="catalog_row", how="inner")

    print(
        f"Phase E: {len(df)} residual events under k={RESCUE_REFERENCE_K} "
        f"candidate_rank={RESCUE_REFERENCE_CANDIDATE_RANK} "
        "(events already safe under that candidate are not in "
        "b1_rescue_matrix.csv and are excluded here by construction)."
    )

    records = []

    for feat in CORE_FEATURES:
        for mode in MODES:

            rescued = df[f"rescued_by_{mode}"]
            n_pos = int(rescued.sum())
            n_neg = int((~rescued).sum())

            if n_pos < 3 or n_neg < 3:
                records.append(
                    {
                        "feature": feat,
                        "mode": mode,
                        "n_rescued": n_pos,
                        "n_not_rescued": n_neg,
                        "auc_rescued_gt_not": float("nan"),
                        "status": (
                            "INCONCLUSIVE (too few residual events "
                            "in one class)"
                        ),
                    }
                )
                continue

            stats = _auc_and_groups(df[feat], rescued)

            records.append(
                {
                    "feature": feat,
                    "mode": mode,
                    "n_rescued": n_pos,
                    "n_not_rescued": n_neg,
                    "median_rescued": stats["median_unsafe"],
                    "median_not_rescued": stats["median_safe"],
                    "auc_rescued_gt_not": stats["auc_unsafe_gt_safe"],
                    "status": "DIAGNOSTIC",
                }
            )

    return pd.DataFrame.from_records(records)


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Experiment B1c: simulation-aware event difficulty. "
            "Runs no fits; reads local H5/parquet artifacts and "
            "already-committed B1 results."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Override the local extreme100 artifacts root "
            f"(default: {DEFAULT_ARTIFACTS_ROOT}, or the "
            "B1C_ARTIFACTS_ROOT environment variable)."
        ),
    )
    args = parser.parse_args()

    root = Path(
        args.root
        or os.environ.get("B1C_ARTIFACTS_ROOT")
        or DEFAULT_ARTIFACTS_ROOT
    ).expanduser()

    print(f"Artifacts root: {root}")
    print()

    # ---- Phase A ----
    features = build_event_features(root)
    audit_event_features(features)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_EVENT_FEATURES, index=False)
    print(f"wrote {OUT_EVENT_FEATURES} ({len(features)} rows)")
    print()

    # ---- labels ----
    labels = load_labels()
    rescue_labels = load_rescue_labels()

    print("Label summary:")
    for k in [4, 5, 6]:
        n_unsafe = int(labels[f"unsafe_k{k}"].sum())
        print(f"  unsafe_k{k}: {n_unsafe}/{len(labels)}")
    print(
        "  min_candidate_budget_reaching_tol counts:\n"
        f"{labels['min_candidate_budget_reaching_tol'].value_counts(dropna=False).sort_index()}"
    )
    print()

    # ---- Phase B/C ----
    feat_vs_diff = descriptive_univariate(features, labels)
    feat_vs_diff.to_csv(OUT_FEATURE_VS_DIFFICULTY, index=False)
    print(
        f"wrote {OUT_FEATURE_VS_DIFFICULTY} ({len(feat_vs_diff)} rows)"
    )
    print()

    # ---- Phase D/F ----
    models_cv = exploratory_models(features, labels)
    models_cv.to_csv(OUT_MODELS_CV, index=False)
    print(f"wrote {OUT_MODELS_CV} ({len(models_cv)} rows)")
    print()

    # ---- Phase E ----
    rescue_assoc = rescue_family_associations(
        features, rescue_labels
    )
    rescue_assoc.to_csv(OUT_RESCUE_ASSOC, index=False)
    print(f"wrote {OUT_RESCUE_ASSOC} ({len(rescue_assoc)} rows)")


if __name__ == "__main__":
    main()
