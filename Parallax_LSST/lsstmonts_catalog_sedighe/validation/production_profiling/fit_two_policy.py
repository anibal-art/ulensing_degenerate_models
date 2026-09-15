#!/usr/bin/env python3
"""
Simplified production architecture: exactly 1 H0 TRF + 1 H1 TRF per
simulated event. No old_H0 bridge, no old_final, no oracle multistart,
no morphology multistart FAMILIES (a single deterministic morphology
seed is reused as-is), no Fisher/scouting/hybrid machinery, no
conditional rescue.

Matrix (see PRODUCTION_CANDIDATE_ARCHITECTURE.md's replacement,
TWO_FIT_ARCHITECTURE.md, for the full writeup):

  H0-generated event:
    H0 fit: start = truth (t0,u0,tE,rho).            1 TRF.
    H1 fit: start = morphology_seed(curves) for
            (t0,u0,tE,rho), piEN=piEE=0 (nested null). 1 TRF.

  H1-generated event:
    H1 fit: start = truth (t0,u0,tE,rho,piEN,piEE).   1 TRF.
    H0 fit: start = morphology_seed(curves) for
            (t0,u0,tE,rho).                            1 TRF.

Truth is initialization only for the model that matches the generating
model; it is NEVER used (not even partially) to seed the other model.
Bounds are run_bounds_audit_refit_core's "production_candidate"
profile throughout -- truth-independent, fixed absolute domain for
u0/tE/rho/piEN/piEE, data-driven (photometry-range) t0 -- so the same
domain is used whether the start came from truth or from morphology,
and it is never re-centered on the start.
"""
import time

from morphology_seed import morphology_seed_from_curves
from run_one_fit_full import run_one_fit_full

PRODUCTION_CANDIDATE_BOUNDS = {
    "u0": [-10.0, 10.0], "tE": [0.1, 500000.0], "rho": [1.0e-7, 10.0],
    "piEN": [-40.0, 40.0], "piEE": [-40.0, 40.0],
}


def check_domain_containment(seed):
    """
    The production domain must contain the morphology-derived start
    BY CONSTRUCTION. If it doesn't, that is an estimator/domain
    inconsistency to be reported, never silently clipped.
    """
    violations = {}
    for par in ["u0", "tE", "rho"]:
        lo, hi = PRODUCTION_CANDIDATE_BOUNDS[par]
        val = seed[par]
        if not (lo <= val <= hi):
            violations[par] = {"value": val, "lo": lo, "hi": hi}
    return violations


def run_two_fit_policy(meta, core, fit_lc):
    """
    meta: as returned by load_new_event.load_new_case (has
    meta["generating_model"] in {"H0","H1"}, meta["truth"],
    meta["curves"]).
    core: the imported run_bounds_audit_refit_core module (with
    BOUNDS_PROFILE == "production_candidate" already set by the
    caller before import).
    fit_lc: the imported fit_lc module (must be imported AFTER core,
    per core.py's own sys.path side effect).

    Returns a dict with the 2 fit records, the morphology seed used,
    any domain-containment violation, and total optimizer-call count
    (must always be exactly 2). Each fit record is the FULL
    fit.fit_results dict (via run_one_fit_full), so optimizer_nfev/
    njev/status/message/optimality/active_mask are always retained.
    """
    assert core.BOUNDS_PROFILE == "production_candidate", (
        "the simplified architecture requires the production_candidate "
        f"bounds profile; got {core.BOUNDS_PROFILE!r}"
    )

    gen = meta["generating_model"]
    truth = meta["truth"]

    seed = morphology_seed_from_curves(meta["curves"])
    domain_violations = check_domain_containment(seed) if seed is not None else None

    result = {
        "catalog_row": meta["row"], "generating_model": gen,
        "morphology_seed": seed, "morphology_seed_domain_violations": domain_violations,
        "morphology_seed_is_none": seed is None,
    }

    def timed_fit(hypothesis, initial, label):
        t0c, cpu0 = time.time(), time.process_time()
        r = run_one_fit_full(core, fit_lc, meta, hypothesis, initial, label)
        return {**r, "wall_s": time.time() - t0c, "cpu_s": time.process_time() - cpu0}

    if gen == "H0":
        h0_initial = {"t0": truth["t0"], "u0": truth["u0"], "tE": truth["tE"], "rho": truth["rho"]}
        result["h0"] = timed_fit("H0", h0_initial, "truth_start")

        if seed is None:
            result["h1"] = None
            result["estimator_failure"] = "morphology_not_measurable"
        else:
            h1_initial = {"t0": seed["t0"], "u0": seed["u0"], "tE": seed["tE"], "rho": seed["rho"],
                           "piEN": 0.0, "piEE": 0.0}
            result["h1"] = timed_fit("H1", h1_initial, "morphology_seed_piE0")

    else:  # H1-generated
        h1_initial = {"t0": truth["t0"], "u0": truth["u0"], "tE": truth["tE"], "rho": truth["rho"],
                       "piEN": truth["piEN"], "piEE": truth["piEE"]}
        result["h1"] = timed_fit("H1", h1_initial, "truth_start")

        if seed is None:
            result["h0"] = None
            result["estimator_failure"] = "morphology_not_measurable"
        else:
            h0_initial = {"t0": seed["t0"], "u0": seed["u0"], "tE": seed["tE"], "rho": seed["rho"]}
            result["h0"] = timed_fit("H0", h0_initial, "morphology_seed")

    n_trf = int(result["h0"] is not None) + int(result["h1"] is not None)
    result["n_trf"] = n_trf

    if result["h0"] is not None and result["h1"] is not None:
        result["chi2_h0"] = result["h0"]["chi2"]
        result["chi2_h1"] = result["h1"]["chi2"]
        result["delta_chi2_lrt"] = result["chi2_h0"] - result["chi2_h1"]

    return result
