"""
Two minimal, CONDITIONAL numerical safeguards on top of the frozen
2-fit policy (fit_two_policy.run_two_fit_policy) -- not a return to
unconditional multistart. Both are opt-in per fit/event, triggered
only by an explicit diagnostic condition on the nominal result.

1. Same-point continuation: if a nominal fit's optimizer_optimality is
   non-finite or > OPTIMALITY_THRESHOLD (0.05, matching the
   polish_optimality_threshold convention already used historically in
   this project's old_final producer), run exactly ONE continuation
   fit from the SAME final nonlinear vector -- same model, same
   objective, same production_candidate bounds, all parameters free,
   stricter tolerances, no new physical seed.

2. Nested H1 rescue: if chi2_H1 > chi2_H0 + EPSILON_NUMERIC after step
   1, run exactly ONE additional H1 fit initialized at the (possibly
   continuation-repaired) H0 solution embedded with piEN=piEE=0, all
   six H1 parameters free. Final H1 = min(current H1, rescue H1).
"""
import time
import numpy as np

from run_one_fit_full import run_one_fit_full

OPTIMALITY_THRESHOLD = 0.05
EPSILON_NUMERIC = 1.0e-6
CONTINUATION_OPTIMIZER_OPTIONS = {
    "xtol": 1e-12, "ftol": 1e-12, "gtol": 1e-10, "max_nfev": 50000, "x_scale": "jac",
}


def _timed_run_one_fit(core, fit_lc, meta, hypothesis, initial, label):
    t0c, cpu0 = time.time(), time.process_time()
    r = run_one_fit_full(core, fit_lc, meta, hypothesis, initial, label)
    return {**r, "wall_s": time.time() - t0c, "cpu_s": time.process_time() - cpu0}


def needs_continuation(fit_record):
    opt = fit_record.get("optimizer_optimality")
    return (opt is None) or (not np.isfinite(opt)) or (opt > OPTIMALITY_THRESHOLD)


def run_same_point_continuation(core, fit_lc, meta, hypothesis, nominal_fit, label_suffix="continuation"):
    """Exactly one continuation TRF from the nominal fit's own final vector."""
    initial = {"t0": nominal_fit["t0"], "u0": nominal_fit["u0"],
               "tE": nominal_fit["tE"], "rho": nominal_fit["rho"]}
    if hypothesis == "H1":
        initial["piEN"] = nominal_fit["piEN"]
        initial["piEE"] = nominal_fit["piEE"]

    original_options = core.OPTIMIZER_OPTIONS
    core.OPTIMIZER_OPTIONS = CONTINUATION_OPTIMIZER_OPTIONS
    try:
        cont = _timed_run_one_fit(core, fit_lc, meta, hypothesis, initial,
                                   f"{nominal_fit['label']}_{label_suffix}")
    finally:
        core.OPTIMIZER_OPTIONS = original_options

    displacement = {
        "t0": cont["t0"] - nominal_fit["t0"], "u0": cont["u0"] - nominal_fit["u0"],
        "tE": cont["tE"] - nominal_fit["tE"], "rho": cont["rho"] - nominal_fit["rho"],
    }
    if hypothesis == "H1":
        displacement["piEN"] = cont["piEN"] - nominal_fit["piEN"]
        displacement["piEE"] = cont["piEE"] - nominal_fit["piEE"]

    return {
        "ran": True,
        "chi2_before": nominal_fit["chi2"], "chi2_after": cont["chi2"],
        "optimality_before": nominal_fit.get("optimizer_optimality"),
        "optimality_after": cont.get("optimizer_optimality"),
        "status_before": nominal_fit.get("status"), "status_after": cont.get("status"),
        "optimizer_status_before": nominal_fit.get("optimizer_status"),
        "optimizer_status_after": cont.get("optimizer_status"),
        "optimizer_message_after": cont.get("optimizer_message"),
        "active_mask_after": cont.get("optimizer_active_mask"),
        "displacement": displacement,
        "fit_record": cont,
    }


def run_nested_h1_rescue(core, fit_lc, meta, h0_fit):
    """Exactly one H1 TRF, initialized at the H0 solution embedded with piE=0."""
    initial = {"t0": h0_fit["t0"], "u0": h0_fit["u0"], "tE": h0_fit["tE"], "rho": h0_fit["rho"],
               "piEN": 0.0, "piEE": 0.0}
    r = _timed_run_one_fit(core, fit_lc, meta, "H1", initial, "nested_h0_rescue")
    return r


def apply_safeguards(core, fit_lc, meta, nominal_result):
    """
    nominal_result: the dict returned by fit_two_policy.run_two_fit_policy.
    Returns a dict describing what safeguards ran and the final H0/H1
    fit records and Delta_chi2_LRT, plus a full audit trail.
    """
    audit = {"catalog_row": meta["row"], "generating_model": meta["generating_model"],
              "continuation_h0": None, "continuation_h1": None, "nested_rescue": None}

    h0 = nominal_result["h0"]
    h1 = nominal_result["h1"]
    n_extra_trf = 0

    if h0 is not None and needs_continuation(h0):
        cont = run_same_point_continuation(core, fit_lc, meta, "H0", h0)
        audit["continuation_h0"] = cont
        n_extra_trf += 1
        if np.isfinite(cont["chi2_after"]) and cont["chi2_after"] <= h0["chi2"]:
            h0 = cont["fit_record"]

    if h1 is not None and needs_continuation(h1):
        cont = run_same_point_continuation(core, fit_lc, meta, "H1", h1)
        audit["continuation_h1"] = cont
        n_extra_trf += 1
        if np.isfinite(cont["chi2_after"]) and cont["chi2_after"] <= h1["chi2"]:
            h1 = cont["fit_record"]

    nested_ran = False
    if h0 is not None and h1 is not None and (h1["chi2"] > h0["chi2"] + EPSILON_NUMERIC):
        rescue = run_nested_h1_rescue(core, fit_lc, meta, h0)
        n_extra_trf += 1
        nested_ran = True
        audit["nested_rescue"] = {
            "ran": True, "chi2_nominal_h1": h1["chi2"], "chi2_rescue_h1": rescue["chi2"],
            "status_rescue": rescue.get("status"), "optimality_rescue": rescue.get("optimizer_optimality"),
            "fit_record": rescue,
        }
        if np.isfinite(rescue["chi2"]) and rescue["chi2"] < h1["chi2"]:
            h1 = rescue

    audit["final_h0"] = h0
    audit["final_h1"] = h1
    audit["n_nominal_trf"] = nominal_result["n_trf"]
    audit["n_extra_trf"] = n_extra_trf
    audit["n_total_trf"] = nominal_result["n_trf"] + n_extra_trf
    audit["continuation_ran_h0"] = audit["continuation_h0"] is not None
    audit["continuation_ran_h1"] = audit["continuation_h1"] is not None
    audit["nested_rescue_ran"] = nested_ran

    if h0 is not None and h1 is not None:
        audit["chi2_h0_final"] = h0["chi2"]
        audit["chi2_h1_final"] = h1["chi2"]
        audit["delta_chi2_lrt_final"] = h0["chi2"] - h1["chi2"]
        audit["still_significantly_negative"] = (h1["chi2"] - h0["chi2"]) > EPSILON_NUMERIC and \
            (h0["chi2"] - h1["chi2"]) < -1.0  # more than a token negative amount

    return audit
