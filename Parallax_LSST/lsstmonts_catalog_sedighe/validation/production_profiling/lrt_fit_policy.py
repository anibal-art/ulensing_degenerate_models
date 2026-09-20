#!/usr/bin/env python3
"""
Final two-fit LRT policy for the simulation study.

H1-generated:
    1 x H1 truth start
        - physical coordinates
    1 x H0 shared-truth start
        - log_te_rho coordinates

H0-generated:
    1 x H0 truth start
        - physical coordinates
    1 x H1 exact nested-null truth start
        - shared truth plus piEN=piEE=0
        - physical coordinates

Exactly one H0 TRF and one H1 TRF are run per selected event.

Truth is used only to initialize the nonlinear parameters.
All fitted parameters remain free.

There is no morphology start, projected-H0 fit, multistart,
same-point continuation, or nested-H1 rescue in this policy.

The LRT statistic is stored without clipping:

    delta_chi2_lrt = chi2_H0 - chi2_H1

A negative value is retained as the raw optimizer outcome.

Peak-coverage quantities are diagnostic only. They do not modify
detectability, event selection, fitting, or the LRT statistic.
"""

import os
import time
from contextlib import contextmanager

import numpy as np

from run_one_fit_full import run_one_fit_full
from final_policy import (
    n_photometry_points,
    chi2_dof_sanity_flag,
)
from numerical_safeguards import (
    EPSILON_NUMERIC,
)


POLICY_NAME = "lrt_two_fit_policy_v2"


# ============================================================
# NEW BLOCK: final-policy diagnostics
# ============================================================

# Diagnostic only.
#
# This threshold does NOT reject an event and does NOT modify
# the pre-fit detectability criterion.  It only marks events
# whose central light curve is poorly sampled.
POOR_PEAK_COVERAGE_THRESHOLD_TE = 0.5


# ============================================================
# NEW BLOCK: peak-coverage diagnostics
# ============================================================

def _all_photometry_times(meta):
    """
    Return all timestamps present in the event light curves.

    This function is used ONLY for diagnostic sampling metrics.
    It does not participate in the pre-fit detectability gate.
    """

    chunks = []

    for band in [
        "W149",
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
    ]:
        curve = meta["curves"].get(
            band,
            None,
        )

        if curve is None:
            continue

        curve = np.asarray(
            curve,
            dtype=float,
        )

        if (
            curve.ndim != 2
            or curve.shape[0] == 0
        ):
            continue

        times = np.asarray(
            curve[:, 0],
            dtype=float,
        )

        times = times[
            np.isfinite(times)
        ]

        if len(times):
            chunks.append(
                times
            )

    if not chunks:
        return np.asarray(
            [],
            dtype=float,
        )

    return np.concatenate(
        chunks
    )


def compute_peak_coverage(meta):
    """
    Compute truth-referenced central-sampling diagnostics.

    Definitions
    -----------

        tau_i =
            (t_i - t0_true) / tE_true

        nearest_peak_distance_tE =
            min_i |tau_i|

        poor_peak_coverage =
            nearest_peak_distance_tE > 0.5

    IMPORTANT
    ---------

    poor_peak_coverage is only a FLAG.

    It does NOT:
        - reject the event;
        - change the detectability criterion;
        - suppress H0 or H1 fitting;
        - modify Delta chi2.

    The additional counts are retained so the chosen 0.5 threshold
    can later be audited without reloading every H5 light curve.
    """

    truth = meta[
        "truth"
    ]

    t0_true = float(
        truth["t0"]
    )

    tE_true = float(
        truth["tE"]
    )

    if (
        not np.isfinite(tE_true)
        or tE_true <= 0.0
    ):
        raise ValueError(
            "Invalid true tE for peak-coverage "
            f"diagnostic: {tE_true!r}"
        )

    times = _all_photometry_times(
        meta
    )

    if len(times) == 0:
        return {
            "nearest_peak_distance_tE":
                np.nan,

            "n_within_0p25_tE":
                0,

            "n_within_0p5_tE":
                0,

            "n_within_1_tE":
                0,

            "n_within_2_tE":
                0,

            "n_left_within_1_tE":
                0,

            "n_right_within_1_tE":
                0,

            "poor_peak_coverage":
                True,

            "poor_peak_coverage_threshold_tE":
                POOR_PEAK_COVERAGE_THRESHOLD_TE,
        }

    tau = (
        times
        - t0_true
    ) / tE_true

    abs_tau = np.abs(
        tau
    )

    nearest = float(
        np.min(
            abs_tau
        )
    )

    return {
        "nearest_peak_distance_tE":
            nearest,

        "n_within_0p25_tE":
            int(
                np.sum(
                    abs_tau <= 0.25
                )
            ),

        "n_within_0p5_tE":
            int(
                np.sum(
                    abs_tau <= 0.5
                )
            ),

        "n_within_1_tE":
            int(
                np.sum(
                    abs_tau <= 1.0
                )
            ),

        "n_within_2_tE":
            int(
                np.sum(
                    abs_tau <= 2.0
                )
            ),

        "n_left_within_1_tE":
            int(
                np.sum(
                    (tau >= -1.0)
                    & (tau < 0.0)
                )
            ),

        "n_right_within_1_tE":
            int(
                np.sum(
                    (tau > 0.0)
                    & (tau <= 1.0)
                )
            ),

        "poor_peak_coverage":
            bool(
                nearest
                > POOR_PEAK_COVERAGE_THRESHOLD_TE
            ),

        "poor_peak_coverage_threshold_tE":
            POOR_PEAK_COVERAGE_THRESHOLD_TE,
    }


# ============================================================
# NEW BLOCK: temporary TRF-coordinate switch
# ============================================================

@contextmanager
def temporary_trf_coordinates(
    core,
    coordinate_mode,
):
    """
    Temporarily select the coordinates seen by scipy TRF.

    The fitting core installs its coordinate implementation at
    import time.  The normal production state is physical.

    For the final policy we need exactly one exception:

        H1-generated event fitted with H0
            -> log_te_rho

    Therefore this context manager:

        1. saves the currently installed TRFfit.fit method;
        2. installs the already validated log_te_rho implementation;
        3. runs one fit;
        4. restores the original method exactly.

    The fitting core source file itself is not modified.
    """

    coordinate_mode = str(
        coordinate_mode
    ).strip().lower()

    if coordinate_mode not in {
        "physical",
        "log_te_rho",
    }:
        raise ValueError(
            "Unsupported production coordinate mode: "
            f"{coordinate_mode!r}"
        )

    if coordinate_mode == "physical":
        yield
        return

    if not hasattr(
        core,
        "_install_trf_coordinate_runtime_patch",
    ):
        raise RuntimeError(
            "run_bounds_audit_refit_core does not expose "
            "_install_trf_coordinate_runtime_patch"
        )

    from pyLIMA.fits import TRF_fit as trf_module

    original_fit_method = (
        trf_module.TRFfit.fit
    )

    original_coordinate_env = (
        os.environ.get(
            "HIDDEN_PARALLAX_TRF_COORDS"
        )
    )

    original_xscale_env = (
        os.environ.get(
            "HIDDEN_PARALLAX_TRF_X_SCALE"
        )
    )

    try:
        os.environ[
            "HIDDEN_PARALLAX_TRF_COORDS"
        ] = coordinate_mode

        # The validated logarithmic-coordinate implementation
        # requires pyLIMA's original coordinate scaling.
        os.environ[
            "HIDDEN_PARALLAX_TRF_X_SCALE"
        ] = "pylima"

        core._install_trf_coordinate_runtime_patch()

        yield

    finally:
        # Restore the exact pre-existing physical implementation.
        trf_module.TRFfit.fit = (
            original_fit_method
        )

        if original_coordinate_env is None:
            os.environ.pop(
                "HIDDEN_PARALLAX_TRF_COORDS",
                None,
            )
        else:
            os.environ[
                "HIDDEN_PARALLAX_TRF_COORDS"
            ] = original_coordinate_env

        if original_xscale_env is None:
            os.environ.pop(
                "HIDDEN_PARALLAX_TRF_X_SCALE",
                None,
            )
        else:
            os.environ[
                "HIDDEN_PARALLAX_TRF_X_SCALE"
            ] = original_xscale_env


def _timed_fit(
    core,
    fit_lc,
    meta,
    hypothesis,
    initial,
    label,
    coordinate_mode="physical",
):
    """
    Run exactly one nonlinear TRF fit.

    coordinate_mode defaults to "physical", so all existing callers
    preserve their current behavior.

    The final policy will explicitly request "log_te_rho" only for
    H1-generated events fitted with the competing H0 model.
    """

    wall0 = time.time()
    cpu0 = time.process_time()

    # NEW BLOCK:
    # Coordinate selection is local to this one TRF call.
    with temporary_trf_coordinates(
        core,
        coordinate_mode,
    ):
        record = run_one_fit_full(
            core,
            fit_lc,
            meta,
            hypothesis,
            initial,
            label,
        )

    record = dict(record)

    # Save the actual coordinate policy with every fit so the
    # production output is self-describing.
    record["coordinate_mode"] = str(
        coordinate_mode
    )

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
    coordinate_mode="physical",
):
    """
    Run exactly one nonlinear TRF fit.

    No continuation, rescue, or secondary optimization is allowed
    here.  The chi2/dof sanity calculation is retained strictly as
    a diagnostic of the returned fit.
    """
    n_params = 4 if hypothesis == "H0" else 6

    nominal = _timed_fit(
        core,
        fit_lc,
        meta,
        hypothesis,
        initial,
        label,
        coordinate_mode=coordinate_mode,
    )

    sanity_flag, reduced_chi2 = chi2_dof_sanity_flag(
        nominal,
        n_points,
        n_params,
    )

    return {
        "initial": dict(initial),
        "label": label,
        "nominal": nominal,
        "settled": nominal,
        "reduced_chi2_nominal": reduced_chi2,
        "reduced_chi2_final": reduced_chi2,
        "sanity_flag_final": sanity_flag,
    }


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
         - physical coordinates
         - exactly one TRF

      2. H0 shared-truth start
         - initial (t0, u0, tE, rho) taken from the H1 truth
         - log_te_rho coordinates
         - exactly one TRF

    Truth is used only as the initial nonlinear parameter vector.
    All fitted parameters remain free.

    No morphology start, projected-H1 start, multistart, or
    continuation is used in this branch.

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
        coordinate_mode="physical",
    )

    result["nominal_fits"]["H1_truth"] = h1_truth

    final_h1 = h1_truth["settled"]

    # ----------------------------------------------------------
    # 2) H0: shared H1-truth start in log(tE), log(rho)
    # ----------------------------------------------------------
    #
    # NEW BLOCK:
    # The competing no-parallax model starts from the shared
    # generating parameters of the H1 event:
    #
    #     (t0, u0, tE, rho)
    #
    # These values are INITIALIZATION ONLY.  All four H0
    # nonlinear parameters remain free during the TRF fit.
    #
    # The validated internal parameterization for this particular
    # model mismatch is:
    #
    #     (t0, u0, log10(tE), log10(rho))
    #
    # Bounds and returned best-fit parameters remain in physical
    # units because the runtime coordinate patch performs the
    # transformation internally.
    h0_initial = {
        "t0": float(truth["t0"]),
        "u0": float(truth["u0"]),
        "tE": float(truth["tE"]),
        "rho": float(truth["rho"]),
    }

    h0_truth_shared = _run_and_settle(
        core,
        fit_lc,
        meta,
        "H0",
        h0_initial,
        "H0_truth_shared_log_te_rho",
        n_points,
        coordinate_mode="log_te_rho",
    )

    result["nominal_fits"][
        "H0_truth_shared"
    ] = h0_truth_shared

    final_h0 = h0_truth_shared[
        "settled"
    ]

    result["h0_winner"] = (
        "H0_truth_shared"
    )

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
         - initial (t0, u0, tE, rho) = generating truth
         - physical coordinates
         - exactly one TRF

      2. H1 exact nested-null truth start
         - same true shared parameters (t0, u0, tE, rho)
         - piEN = 0
         - piEE = 0
         - physical coordinates
         - exactly one TRF

    Truth is used only as the initial nonlinear parameter vector.
    All fitted parameters remain free.

    The H1 start is constructed directly from the H0 generating
    truth, not from the settled H0 fit.

    No continuation or nested rescue is part of this branch.

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
        coordinate_mode="physical",
    )

    result["nominal_fits"]["H0_truth"] = h0_truth

    final_h0 = h0_truth["settled"]
    result["h0_winner"] = "H0_truth"

    # ----------------------------------------------------------
    # 2) H1: exact nested-null start from H0 generating truth
    # ----------------------------------------------------------
    #
    # NEW BLOCK:
    # H1 starts from the same shared nonlinear parameters used
    # to generate the H0 event, augmented by piE = (0, 0).
    #
    # This point is exactly the no-parallax generating model
    # represented inside the nested H1 parameter space.
    #
    # The settled H0 fit is NOT used to initialize H1.
    h1_initial = {
        "t0": float(truth["t0"]),
        "u0": float(truth["u0"]),
        "tE": float(truth["tE"]),
        "rho": float(truth["rho"]),
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
        coordinate_mode="physical",
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
    Execute the frozen final two-fit LRT policy.

    For every selected event:
        - exactly one H0 TRF
        - exactly one H1 TRF
        - exactly two total TRFs

    H1-generated:
        H1 truth start in physical coordinates
        H0 shared-truth start in log_te_rho coordinates

    H0-generated:
        H0 truth start in physical coordinates
        H1 shared-truth plus piE=(0, 0) in physical coordinates

    No continuation, rescue, morphology start, projected-H0 fit,
    or multistart is allowed.
    """

    assert core.BOUNDS_PROFILE == "production_candidate", (
        "lrt_two_fit_policy_v2 requires "
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

    # NEW BLOCK:
    # Truth-referenced sampling diagnostics.
    #
    # These quantities are recorded only for later sensitivity
    # analyses. They do NOT alter detectability or fitting.
    coverage = compute_peak_coverage(
        meta
    )

    result = {
        "policy_name": POLICY_NAME,
        "catalog_row": meta["row"],
        "generating_model": generating_model,
        "n_photometry_points": n_points,
        "nominal_fits": {},
        "h0_winner": None,
        "estimator_failure": None,

        # NEW BLOCK:
        # Persist central-coverage diagnostics in every event record.
        **coverage,
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
    # Final nesting diagnostic
    # ----------------------------------------------------------
    #
    # NEW BLOCK:
    # No nested-H1 rescue is performed in the final two-fit policy.
    #
    # If the independently fitted H1 lands at a slightly larger
    # chi2 than H0, the raw result is retained.  This preserves the
    # actual optimizer outcome and avoids introducing an additional
    # TRF that would change the calibrated null distribution.
    #
    # final_nesting_ok is computed below as a diagnostic only.

    # ----------------------------------------------------------
    # Final two-fit optimizer accounting
    # ----------------------------------------------------------
    #
    # NEW BLOCK:
    # The frozen production policy has one H0 TRF and one H1 TRF.
    #
    # Continuations and rescues are not part of this policy.
    # Their optimizer-call counts are therefore fixed to zero.
    # The explicit invariants below protect the two-fit contract.
    n_nominal_trf = len(
        result["nominal_fits"]
    )

    # Continuation is not part of the frozen production policy.
    n_continuation_trf = 0

    n_rescue_trf = 0

    n_total_trf = (
        n_nominal_trf
        + n_continuation_trf
        + n_rescue_trf
    )

    result["n_nominal_trf"] = (
        n_nominal_trf
    )

    result["n_continuation_trf"] = (
        n_continuation_trf
    )

    result["n_rescue_trf"] = (
        n_rescue_trf
    )

    result["n_total_trf"] = (
        n_total_trf
    )

    if n_nominal_trf != 2:
        raise RuntimeError(
            "Final two-fit policy violated: "
            f"expected 2 nominal TRFs, "
            f"got {n_nominal_trf}"
        )

    if n_continuation_trf != 0:
        raise RuntimeError(
            "Final two-fit policy violated: "
            "a same-point continuation was executed."
        )

    if n_rescue_trf != 0:
        raise RuntimeError(
            "Final two-fit policy violated: "
            "a rescue TRF was executed."
        )

    if n_total_trf != 2:
        raise RuntimeError(
            "Final two-fit policy violated: "
            f"expected exactly 2 total TRFs, "
            f"got {n_total_trf}"
        )

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

    return result
