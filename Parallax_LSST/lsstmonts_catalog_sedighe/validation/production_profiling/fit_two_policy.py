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
    Run exactly one H0 fit and one H1 fit.

    Initialization policy
    ---------------------

    H0-generated event = event simulated WITHOUT annual parallax.

        H0 fit:
            Start from the true generating parameters
            (t0, u0, tE, rho).

        H1 fit:
            Start from the SAME true generating parameters
            (t0, u0, tE, rho), augmented with

                piEN = 0
                piEE = 0

            Since H0 is nested inside H1 at piE = 0, this initial
            point is exactly the generating no-parallax model
            represented inside the H1 parameter space.

        No morphology seed is used for H0-generated events.

    H1-generated event = event simulated WITH annual parallax.

        H1 fit:
            Start from the true generating parameters
            (t0, u0, tE, rho, piEN, piEE).

        H0 fit:
            Start from morphology_seed_from_curves(curves).

            This is intentional: when fitting a parallax-generated
            event with a no-parallax model, the best H0 values of
            t0, u0, tE, and rho need not equal the true H1 generating
            parameters. In particular, annual-parallax asymmetry can
            be partially absorbed by a different no-parallax
            morphology and timescale.

    No rescue, continuation, or multistart is performed here.

    When both fits are available, n_trf must be exactly 2.

    Each fit record is the FULL fit.fit_results dictionary returned
    by run_one_fit_full, so optimizer diagnostics are retained.
    """
    assert core.BOUNDS_PROFILE == "production_candidate", (
        "the simplified architecture requires the production_candidate "
        f"bounds profile; got {core.BOUNDS_PROFILE!r}"
    )

    gen = meta["generating_model"]
    truth = meta["truth"]

    # NEW BLOCK:
    # The morphology seed is needed ONLY for an H1-generated event
    # fitted with the competing no-parallax H0 model.
    #
    # For an H0-generated event, both H0 and H1 start from the known
    # generating parameters, with piE=(0,0) added for H1.
    if gen == "H1":
        seed = morphology_seed_from_curves(meta["curves"])
        domain_violations = (
            check_domain_containment(seed)
            if seed is not None
            else None
        )
    else:
        seed = None
        domain_violations = None

    result = {
        "catalog_row": meta["row"],
        "generating_model": gen,
        "morphology_seed": seed,
        "morphology_seed_domain_violations": domain_violations,
        "morphology_seed_is_none": seed is None,
    }

    def timed_fit(hypothesis, initial, label):
        t0c, cpu0 = time.time(), time.process_time()

        r = run_one_fit_full(
            core,
            fit_lc,
            meta,
            hypothesis,
            initial,
            label,
        )

        return {
            **r,
            "wall_s": time.time() - t0c,
            "cpu_s": time.process_time() - cpu0,
        }

    if gen == "H0":
        # NEW BLOCK:
        # Event generated WITHOUT parallax.
        #
        # H0 is initialized at the true H0 generating parameters.
        h0_initial = {
            "t0": truth["t0"],
            "u0": truth["u0"],
            "tE": truth["tE"],
            "rho": truth["rho"],
        }

        result["h0"] = timed_fit(
            "H0",
            h0_initial,
            "truth_start",
        )

        # NEW BLOCK:
        # Fit the SAME no-parallax-generated event with H1.
        #
        # Use the true H0 generating parameters and embed them
        # exactly inside H1 by setting both parallax components to 0.
        #
        # NO morphology seed is used in this branch.
        h1_initial = {
            "t0": truth["t0"],
            "u0": truth["u0"],
            "tE": truth["tE"],
            "rho": truth["rho"],
            "piEN": 0.0,
            "piEE": 0.0,
        }

        result["h1"] = timed_fit(
            "H1",
            h1_initial,
            "truth_start_piE0",
        )

    else:  # H1-generated
        # Event generated WITH parallax.
        #
        # H1 is initialized at the complete true generating vector.
        h1_initial = {
            "t0": truth["t0"],
            "u0": truth["u0"],
            "tE": truth["tE"],
            "rho": truth["rho"],
            "piEN": truth["piEN"],
            "piEE": truth["piEE"],
        }

        result["h1"] = timed_fit(
            "H1",
            h1_initial,
            "truth_start",
        )

        # The competing H0 fit still uses the morphology seed.
        #
        # This is the branch where morphology is scientifically useful:
        # the best no-parallax approximation to an asymmetric
        # parallax-generated light curve need not share the true H1
        # values of t0, u0, tE, or rho.
        if seed is None:
            result["h0"] = None
            result["estimator_failure"] = "morphology_not_measurable"
        else:
            h0_initial = {
                "t0": seed["t0"],
                "u0": seed["u0"],
                "tE": seed["tE"],
                "rho": seed["rho"],
            }

            result["h0"] = timed_fit(
                "H0",
                h0_initial,
                "morphology_seed",
            )

    n_trf = (
        int(result["h0"] is not None)
        + int(result["h1"] is not None)
    )
    result["n_trf"] = n_trf

    if result["h0"] is not None and result["h1"] is not None:
        result["chi2_h0"] = result["h0"]["chi2"]
        result["chi2_h1"] = result["h1"]["chi2"]
        result["delta_chi2_lrt"] = (
            result["chi2_h0"]
            - result["chi2_h1"]
        )

    return result
