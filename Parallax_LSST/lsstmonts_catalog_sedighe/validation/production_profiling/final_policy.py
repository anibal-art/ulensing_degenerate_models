#!/usr/bin/env python3
"""
FINAL frozen production policy (v2, corrected per the 4-blocker
methodological review, 2026-09-15).

  nominal:   1 H0 TRF + 1 H1 TRF per event (fit_two_policy.run_two_fit_policy),
             truth start for the matching model, deterministic truth-blind
             morphology start for the other model.

  safeguard 1 (catastrophic false convergence): if a fit's reduced chi2
             (chi2/dof, using only its own photometry point count and
             parameter count -- no reference/other model needed) exceeds
             CHI2_DOF_SANITY_THRESHOLD, run exactly ONE same-point
             continuation (same model/objective/production_candidate
             bounds, all parameters free, stricter tolerances, no new
             physical seed) -- this repaired row 451245 to within ~1e-4
             of the robust reference without any new physical seed. Kept
             only if it does not worsen chi2. The reduced-chi2 threshold
             (not optimizer_optimality -- NUMERICAL_SAFEGUARDS.md found
             optimality fires on 12-14/14 of ALL fits while usually
             changing chi2 negligibly, not exceptional) is empirically
             clean: 0 false positives across every valid fit inspected
             in this project (74 fits total across 3 independent
             samples), true positive on every catastrophic case found.

  safeguard 2 (nested-model consistency): if chi2_H1 > chi2_H0 +
             EPSILON_NUMERIC (checked AFTER safeguard 1, using whatever
             chi2 that leaves), run exactly one extra H1 TRF initialized
             at the final H0 solution embedded with piEN=piEE=0, all six
             H1 parameters free; final H1 = min(current H1, rescue H1).

  morphology_not_measurable: if morphology_seed_from_curves returns None,
             the affected fit is SKIPPED (not run, not substituted with
             any fallback value). The event is recorded with
             estimator_failure="morphology_not_measurable" and excluded
             from any Delta_chi2_LRT / classification statistic (there is
             no chi2 to compare) but IS counted in the total processed
             population, contributing to an explicit failure-rate metric.
             This rule is identical for H0-calibration and H1-production
             events -- same code path, no special-casing either way.

Both safeguards are conditional on an explicit, per-fit diagnostic; there
is still no unconditional multistart anywhere in this policy.
"""
import time

from fit_two_policy import run_two_fit_policy
from numerical_safeguards import run_nested_h1_rescue, run_same_point_continuation, EPSILON_NUMERIC

# See module docstring safeguard 1. Empirically: reduced chi2 across all
# 74 non-catastrophic nominal fits inspected across this project's three
# independent validation samples (9+5 event initial batch, 25-event
# basin-gap sample, 60-event H0-calibration sample) ranges roughly
# 0.75-4.5; row 451245's own catastrophic H0 fit had reduced chi2 = 778.0
# before continuation. Threshold=50 sits with >10x headroom above the
# highest valid value seen and >15x below the one observed failure.
CHI2_DOF_SANITY_THRESHOLD = 50.0


def n_photometry_points(curves):
    return sum(len(curves[b]) for b in ["u", "g", "r", "i", "z", "y"] if len(curves[b]))


def reduced_chi2(fit_record, n_points, n_params):
    dof = max(1, n_points - n_params)
    return fit_record["chi2"] / dof


def chi2_dof_sanity_flag(fit_record, n_points, n_params):
    r = reduced_chi2(fit_record, n_points, n_params)
    return r > CHI2_DOF_SANITY_THRESHOLD, r


def run_final_policy(meta, core, fit_lc):
    """
    Returns the nominal result plus final_h0/final_h1 (after both
    safeguards, if triggered), n_total_trf, and sanity_flags (computed
    on the FINAL h0/h1, i.e. after any continuation) -- never silently
    altering or dropping a result; every automatic step is recorded.
    """
    nominal = run_two_fit_policy(meta, core, fit_lc)
    h0, h1 = nominal["h0"], nominal["h1"]
    n_points = n_photometry_points(meta["curves"])

    # ---- safeguard 1: catastrophic-false-convergence continuation ----
    continuation_ran_h0 = continuation_ran_h1 = False
    continuation_record_h0 = continuation_record_h1 = None

    if h0 is not None:
        flagged, _ = chi2_dof_sanity_flag(h0, n_points, 4)
        if flagged:
            continuation_ran_h0 = True
            t0c, cpu0 = time.time(), time.process_time()
            cont = run_same_point_continuation(core, fit_lc, meta, "H0", h0, label_suffix="chi2dof_continuation")
            continuation_record_h0 = cont
            if cont["fit_record"]["chi2"] <= h0["chi2"]:
                h0 = cont["fit_record"]

    if h1 is not None:
        flagged, _ = chi2_dof_sanity_flag(h1, n_points, 6)
        if flagged:
            continuation_ran_h1 = True
            cont = run_same_point_continuation(core, fit_lc, meta, "H1", h1, label_suffix="chi2dof_continuation")
            continuation_record_h1 = cont
            if cont["fit_record"]["chi2"] <= h1["chi2"]:
                h1 = cont["fit_record"]

    # ---- safeguard 2: nested-model-consistency rescue ----
    rescue_ran = False
    rescue_record = None
    if h0 is not None and h1 is not None and (h1["chi2"] > h0["chi2"] + EPSILON_NUMERIC):
        t0c, cpu0 = time.time(), time.process_time()
        rescue_record = run_nested_h1_rescue(core, fit_lc, meta, h0)
        rescue_record["wall_s"] = time.time() - t0c
        rescue_record["cpu_s"] = time.process_time() - cpu0
        rescue_ran = True
        if rescue_record["chi2"] < h1["chi2"]:
            h1 = rescue_record

    # ---- final sanity flags, computed on the settled h0/h1 ----
    sanity_flags = {}
    if h0 is not None:
        flagged, red = chi2_dof_sanity_flag(h0, n_points, 4)
        sanity_flags["h0"] = {"flagged": flagged, "reduced_chi2": red}
    if h1 is not None:
        flagged, red = chi2_dof_sanity_flag(h1, n_points, 6)
        sanity_flags["h1"] = {"flagged": flagged, "reduced_chi2": red}

    n_total_trf = nominal["n_trf"] + int(continuation_ran_h0) + int(continuation_ran_h1) + int(rescue_ran)

    estimator_failure = None
    if h0 is None or h1 is None:
        estimator_failure = "morphology_not_measurable"

    result = {
        "catalog_row": meta["row"], "generating_model": meta["generating_model"],
        "morphology_seed": nominal["morphology_seed"],
        "morphology_seed_domain_violations": nominal["morphology_seed_domain_violations"],
        "final_h0": h0, "final_h1": h1,
        "continuation_ran_h0": continuation_ran_h0, "continuation_ran_h1": continuation_ran_h1,
        "continuation_record_h0": continuation_record_h0, "continuation_record_h1": continuation_record_h1,
        "rescue_ran": rescue_ran, "rescue_record": rescue_record,
        "n_total_trf": n_total_trf, "sanity_flags": sanity_flags,
        "estimator_failure": estimator_failure,
    }
    if h0 is not None and h1 is not None:
        result["chi2_h0"] = h0["chi2"]
        result["chi2_h1"] = h1["chi2"]
        result["delta_chi2_lrt"] = h0["chi2"] - h1["chi2"]

    return result
