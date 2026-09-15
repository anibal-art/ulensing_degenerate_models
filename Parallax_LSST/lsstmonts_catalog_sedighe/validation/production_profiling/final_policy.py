#!/usr/bin/env python3
"""
FINAL frozen production policy, adopted after the two-fit baseline
(TWO_FIT_ARCHITECTURE.md) and the numerical-safeguards characterization
(NUMERICAL_SAFEGUARDS.md):

  nominal:   1 H0 TRF + 1 H1 TRF per event (fit_two_policy.run_two_fit_policy),
             truth start for the matching model, deterministic truth-blind
             morphology start for the other model.

  safeguard: nested-model-consistency rescue ONLY. If
             chi2_H1 > chi2_H0 + EPSILON_NUMERIC, run exactly one extra H1
             TRF initialized at the final H0 solution embedded with
             piEN=piEE=0, all six H1 parameters free; final H1 =
             min(nominal H1, rescue H1). This is adopted because it is
             narrowly targeted (fired on 4/14 = 28.6% of the validation
             sample, exactly the pathological cases) and fully resolves
             every negative-LRT case observed.

  NOT adopted: the generic optimizer_optimality > 0.05 same-point
  continuation (NUMERICAL_SAFEGUARDS.md found it fires on 14/14 and
  12/14 of all nominal fits while usually changing chi2 negligibly --
  not a genuinely exceptional condition under production_candidate's
  wide bounds). No automatic extra fit is triggered for catastrophic
  false convergence (e.g. row 451245); instead a POST-HOC, reference-
  free FAILURE FLAG is computed from each fit's own chi2 relative to
  its degrees of freedom (see chi2_dof_sanity_flag below) -- large
  enough to never fire on any of the 28 nominal fits inspected in this
  project's validation work, small enough to have caught row 451245
  by roughly 150x. Flagged events are logged for batch-level review,
  not silently re-fit.
"""
import time

from fit_two_policy import run_two_fit_policy
from numerical_safeguards import run_nested_h1_rescue, EPSILON_NUMERIC

# Reduced-chi2 sanity bound: a fit whose chi2/dof exceeds this is flagged
# as a catastrophic-failure candidate for review, purely from its own
# chi2 and photometry point count -- no reference/other-model comparison
# needed. Empirically: reduced chi2 across all 27 non-catastrophic nominal
# fits in this project's validation samples (9 H1-generated + 5
# H0-generated events, H0+H1 each) ranges 0.85-4.46; row 451245's own
# catastrophic H0 fit has reduced chi2 = 778.0. Threshold=50 sits with
# >11x headroom above the highest valid value and >15x below the one
# observed failure -- fires on 0/27 valid fits and 1/1 catastrophic case.
CHI2_DOF_SANITY_THRESHOLD = 50.0


def n_photometry_points(curves):
    return sum(len(curves[b]) for b in ["u", "g", "r", "i", "z", "y"] if len(curves[b]))


def chi2_dof_sanity_flag(fit_record, n_points, n_params):
    dof = max(1, n_points - n_params)
    reduced_chi2 = fit_record["chi2"] / dof
    return reduced_chi2 > CHI2_DOF_SANITY_THRESHOLD, reduced_chi2


def run_final_policy(meta, core, fit_lc):
    """
    Returns the nominal result plus final_h0/final_h1 (after the
    nested rescue, if triggered), n_total_trf, and a sanity_flags
    dict (never silently altering the fit -- only logged).
    """
    nominal = run_two_fit_policy(meta, core, fit_lc)
    h0, h1 = nominal["h0"], nominal["h1"]

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

    n_points = n_photometry_points(meta["curves"])
    sanity_flags = {}
    if h0 is not None:
        flagged, reduced = chi2_dof_sanity_flag(h0, n_points, 4)
        sanity_flags["h0"] = {"flagged": flagged, "reduced_chi2": reduced}
    if h1 is not None:
        flagged, reduced = chi2_dof_sanity_flag(h1, n_points, 6)
        sanity_flags["h1"] = {"flagged": flagged, "reduced_chi2": reduced}

    n_total_trf = nominal["n_trf"] + (1 if rescue_ran else 0)

    result = {
        "catalog_row": meta["row"], "generating_model": meta["generating_model"],
        "morphology_seed": nominal["morphology_seed"],
        "morphology_seed_domain_violations": nominal["morphology_seed_domain_violations"],
        "final_h0": h0, "final_h1": h1,
        "rescue_ran": rescue_ran, "rescue_record": rescue_record,
        "n_total_trf": n_total_trf, "sanity_flags": sanity_flags,
    }
    if h0 is not None and h1 is not None:
        result["chi2_h0"] = h0["chi2"]
        result["chi2_h1"] = h1["chi2"]
        result["delta_chi2_lrt"] = h0["chi2"] - h1["chi2"]

    return result
