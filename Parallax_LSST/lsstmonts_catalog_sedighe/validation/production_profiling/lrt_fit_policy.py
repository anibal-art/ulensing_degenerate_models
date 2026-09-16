#!/usr/bin/env python3
"""
LRT fitting policy for the simulation study.

H1-generated:
    1 x H1 truth start
    1 x H0 morphology start
    1 x H0 projected from settled H1
    final H0 = min(two H0 fits)

H0-generated:
    1 x H0 truth start
    1 x H1 nested from settled H0 with piEN=piEE=0

All nonlinear parameters remain free.

Each nominal fit may receive one catastrophic same-point continuation.
After H0/H1 settlement, nesting is rechecked and one nested-H1 rescue
is allowed if required.
"""

import time

from morphology_seed import morphology_seed_from_curves
from run_one_fit_full import run_one_fit_full
from final_policy import (
    n_photometry_points,
    chi2_dof_sanity_flag,
)
from numerical_safeguards import (
    EPSILON_NUMERIC,
    run_nested_h1_rescue,
    run_same_point_continuation,
)


POLICY_NAME = "lrt_simulation_policy_v1"

PRODUCTION_CANDIDATE_BOUNDS = {
    "u0": (-10.0, 10.0),
    "tE": (0.1, 500000.0),
    "rho": (1.0e-7, 10.0),
    "piEN": (-40.0, 40.0),
    "piEE": (-40.0, 40.0),
}


def check_shared_domain(seed):
    """
    Check fixed production bounds for shared nonlinear parameters.
    t0 is omitted because its production domain is data-driven.
    """
    violations = {}

    for par in ("u0", "tE", "rho"):
        lo, hi = PRODUCTION_CANDIDATE_BOUNDS[par]
        value = float(seed[par])

        if not lo <= value <= hi:
            violations[par] = {
                "value": value,
                "lo": lo,
                "hi": hi,
            }

    return violations


def _timed_fit(
    core,
    fit_lc,
    meta,
    hypothesis,
    initial,
    label,
):
    wall0 = time.time()
    cpu0 = time.process_time()

    record = run_one_fit_full(
        core,
        fit_lc,
        meta,
        hypothesis,
        initial,
        label,
    )

    record = dict(record)
    record["wall_s"] = time.time() - wall0
    record["cpu_s"] = time.process_time() - cpu0

    return record


def _run_and_settle(
    core,
    fit_lc,
    meta,
    hypothesis,
    initial,
    label,
    n_points,
):
    """
    One nominal TRF plus, only when reduced chi2 > 50,
    one same-point continuation.
    """
    n_params = 4 if hypothesis == "H0" else 6

    nominal = _timed_fit(
        core,
        fit_lc,
        meta,
        hypothesis,
        initial,
        label,
    )

    flagged, reduced_nominal = chi2_dof_sanity_flag(
        nominal,
        n_points,
        n_params,
    )

    continuation_ran = False
    continuation_record = None
    settled = nominal

    if flagged:
        continuation_ran = True

        continuation_record = run_same_point_continuation(
            core,
            fit_lc,
            meta,
            hypothesis,
            nominal,
            label_suffix=f"{label}_chi2dof_continuation",
        )

        candidate = continuation_record["fit_record"]

        if float(candidate["chi2"]) <= float(settled["chi2"]):
            settled = candidate

    flagged_final, reduced_final = chi2_dof_sanity_flag(
        settled,
        n_points,
        n_params,
    )

    return {
        "initial": dict(initial),
        "label": label,
        "nominal": nominal,
        "settled": settled,
        "continuation_ran": continuation_ran,
        "continuation_record": continuation_record,
        "reduced_chi2_nominal": reduced_nominal,
        "reduced_chi2_final": reduced_final,
        "sanity_flag_final": flagged_final,
    }


def _shared_from_fit(fit_record):
    return {
        "t0": float(fit_record["t0"]),
        "u0": float(fit_record["u0"]),
        "tE": float(fit_record["tE"]),
        "rho": float(fit_record["rho"]),
    }


def _select_min_fit(candidates):
    valid = [
        (label, fit_record)
        for label, fit_record in candidates
        if fit_record is not None
    ]

    if not valid:
        return None, None

    return min(
        valid,
        key=lambda item: float(item[1]["chi2"]),
    )


def _run_h1_generated(
    meta,
    core,
    fit_lc,
    truth,
    n_points,
    result,
):
    """
    H1-generated event:

      1. H1 truth start
      2. H0 morphology start
      3. H0 projected from settled H1

    Returns
    -------
    final_h0, final_h1
    """

    # ----------------------------------------------------------
    # 1) H1: truth start
    # ----------------------------------------------------------
    h1_initial = {
        "t0": float(truth["t0"]),
        "u0": float(truth["u0"]),
        "tE": float(truth["tE"]),
        "rho": float(truth["rho"]),
        "piEN": float(truth["piEN"]),
        "piEE": float(truth["piEE"]),
    }

    h1_truth = _run_and_settle(
        core,
        fit_lc,
        meta,
        "H1",
        h1_initial,
        "H1_truth_start",
        n_points,
    )

    result["nominal_fits"]["H1_truth"] = h1_truth

    final_h1 = h1_truth["settled"]

    # ----------------------------------------------------------
    # 2) H0: morphology start
    # ----------------------------------------------------------
    morphology_seed = morphology_seed_from_curves(
        meta["curves"]
    )

    result["morphology_seed"] = morphology_seed

    h0_morphology = None

    if morphology_seed is None:
        result["morphology_fit_skipped_reason"] = (
            "morphology_not_measurable"
        )

    else:
        violations = check_shared_domain(
            morphology_seed
        )

        result[
            "morphology_seed_domain_violations"
        ] = violations

        if violations:
            result["morphology_fit_skipped_reason"] = (
                "morphology_seed_out_of_bounds"
            )

        else:
            h0_morph_initial = {
                "t0": float(morphology_seed["t0"]),
                "u0": float(morphology_seed["u0"]),
                "tE": float(morphology_seed["tE"]),
                "rho": float(morphology_seed["rho"]),
            }

            h0_morphology = _run_and_settle(
                core,
                fit_lc,
                meta,
                "H0",
                h0_morph_initial,
                "H0_morphology_start",
                n_points,
            )

            result["nominal_fits"][
                "H0_morphology"
            ] = h0_morphology

    # ----------------------------------------------------------
    # 3) H0: projection from settled H1
    # ----------------------------------------------------------
    h0_projected_initial = _shared_from_fit(
        final_h1
    )

    h0_projected = _run_and_settle(
        core,
        fit_lc,
        meta,
        "H0",
        h0_projected_initial,
        "H0_H1_projected_start",
        n_points,
    )

    result["nominal_fits"][
        "H0_H1_projected"
    ] = h0_projected

    # ----------------------------------------------------------
    # H0 selection
    # ----------------------------------------------------------
    h0_winner, final_h0 = _select_min_fit(
        [
            (
                "H0_morphology",
                (
                    None
                    if h0_morphology is None
                    else h0_morphology["settled"]
                ),
            ),
            (
                "H0_H1_projected",
                h0_projected["settled"],
            ),
        ]
    )

    result["h0_winner"] = h0_winner

    return final_h0, final_h1


def _run_h0_generated(
    meta,
    core,
    fit_lc,
    truth,
    n_points,
    result,
):
    """
    H0-generated event:

      1. H0 truth start
      2. H1 nested from settled H0 with piEN=piEE=0

    Returns
    -------
    final_h0, final_h1
    """

    # ----------------------------------------------------------
    # 1) H0: truth start
    # ----------------------------------------------------------
    h0_initial = {
        "t0": float(truth["t0"]),
        "u0": float(truth["u0"]),
        "tE": float(truth["tE"]),
        "rho": float(truth["rho"]),
    }

    h0_truth = _run_and_settle(
        core,
        fit_lc,
        meta,
        "H0",
        h0_initial,
        "H0_truth_start",
        n_points,
    )

    result["nominal_fits"]["H0_truth"] = h0_truth

    final_h0 = h0_truth["settled"]
    result["h0_winner"] = "H0_truth"

    # ----------------------------------------------------------
    # 2) H1: exact nested start from settled H0
    # ----------------------------------------------------------
    h1_initial = {
        **_shared_from_fit(final_h0),
        "piEN": 0.0,
        "piEE": 0.0,
    }

    h1_nested = _run_and_settle(
        core,
        fit_lc,
        meta,
        "H1",
        h1_initial,
        "H1_H0_nested_piE0_start",
        n_points,
    )

    result["nominal_fits"]["H1_H0_nested"] = h1_nested

    final_h1 = h1_nested["settled"]

    return final_h0, final_h1


def run_lrt_fit_policy(
    meta,
    core,
    fit_lc,
):
    """
    Execute the frozen LRT simulation fitting policy.

    Expected nominal costs:
        H1-generated -> 3 TRF
        H0-generated -> 2 TRF

    Additional TRFs are only conditional numerical safeguards.
    """

    assert core.BOUNDS_PROFILE == "production_candidate", (
        "lrt_simulation_policy_v1 requires "
        f"BOUNDS_PROFILE='production_candidate'; "
        f"got {core.BOUNDS_PROFILE!r}"
    )

    generating_model = str(
        meta["generating_model"]
    )

    if generating_model not in {"H0", "H1"}:
        raise ValueError(
            "Unsupported generating_model="
            f"{generating_model!r}"
        )

    truth = meta["truth"]

    n_points = n_photometry_points(
        meta["curves"]
    )

    result = {
        "policy_name": POLICY_NAME,
        "catalog_row": meta["row"],
        "generating_model": generating_model,
        "n_photometry_points": n_points,
        "nominal_fits": {},
        "morphology_seed": None,
        "morphology_seed_domain_violations": None,
        "morphology_fit_skipped_reason": None,
        "h0_winner": None,
        "nested_h1_rescue_ran": False,
        "nested_h1_rescue_record": None,
        "estimator_failure": None,
    }

    # ----------------------------------------------------------
    # Run generating-model-specific nominal policy
    # ----------------------------------------------------------
    if generating_model == "H1":

        final_h0, final_h1 = _run_h1_generated(
            meta,
            core,
            fit_lc,
            truth,
            n_points,
            result,
        )

    else:

        final_h0, final_h1 = _run_h0_generated(
            meta,
            core,
            fit_lc,
            truth,
            n_points,
            result,
        )

    # ----------------------------------------------------------
    # Final nested-model consistency check
    # ----------------------------------------------------------
    if (
        final_h0 is not None
        and final_h1 is not None
        and float(final_h1["chi2"])
        > float(final_h0["chi2"])
        + EPSILON_NUMERIC
    ):

        rescue = run_nested_h1_rescue(
            core,
            fit_lc,
            meta,
            final_h0,
        )

        result["nested_h1_rescue_ran"] = True
        result["nested_h1_rescue_record"] = rescue

        if (
            float(rescue["chi2"])
            < float(final_h1["chi2"])
        ):
            final_h1 = rescue

    # ----------------------------------------------------------
    # Optimizer-call accounting
    # ----------------------------------------------------------
    n_nominal_trf = len(
        result["nominal_fits"]
    )

    n_continuation_trf = sum(
        int(record["continuation_ran"])
        for record
        in result["nominal_fits"].values()
    )

    n_rescue_trf = int(
        result["nested_h1_rescue_ran"]
    )

    n_total_trf = (
        n_nominal_trf
        + n_continuation_trf
        + n_rescue_trf
    )

    result["n_nominal_trf"] = n_nominal_trf
    result["n_continuation_trf"] = (
        n_continuation_trf
    )
    result["n_rescue_trf"] = n_rescue_trf
    result["n_total_trf"] = n_total_trf

    result["final_h0"] = final_h0
    result["final_h1"] = final_h1

    # ----------------------------------------------------------
    # Final scientific statistic
    # ----------------------------------------------------------
    if final_h0 is None or final_h1 is None:

        result["estimator_failure"] = (
            "missing_final_fit"
        )

        return result

    result["chi2_h0"] = float(
        final_h0["chi2"]
    )

    result["chi2_h1"] = float(
        final_h1["chi2"]
    )

    result["delta_chi2_lrt"] = (
        result["chi2_h0"]
        - result["chi2_h1"]
    )

    result["final_nesting_ok"] = (
        result["chi2_h1"]
        <= result["chi2_h0"]
        + EPSILON_NUMERIC
    )

    # Nominal-count invariant.
    expected_nominal = (
        3
        if generating_model == "H1"
        else 2
    )

    if (
        result["morphology_fit_skipped_reason"]
        is None
        and n_nominal_trf != expected_nominal
    ):
        raise RuntimeError(
            "Unexpected nominal TRF count: "
            f"gen={generating_model}, "
            f"expected={expected_nominal}, "
            f"got={n_nominal_trf}"
        )

    return result
