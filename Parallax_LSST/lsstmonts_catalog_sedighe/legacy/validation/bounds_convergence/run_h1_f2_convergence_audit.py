#!/usr/bin/env python3
"""
H1-F2: convergence integrity audit for the 3 H1-F survivors.

Re-runs the EXACT same controlled4-winner-polish fit (same start, same
bounds, same bounded_flux_profile, same tightened tolerances), but via
fit_lc.fit_rubin_roman() directly instead of core.run_one_fit()'s
trimmed wrapper, to recover the full scipy OptimizeResult (status,
message, njev) that run_one_fit's return dict does not carry. Then
evaluates, at the returned final point: exact chi2 (cross-check),
numerical gradient J^T r in the optimizer's own physical coordinates,
and a bound-aware projected gradient norm (zeroing components whose
parameter sits on an active bound and whose gradient points outward --
the correct first-order stationarity condition at a bound). Finally,
for any event whose optimality is far from gtol, attempts ONE more
continuation polish from the exact same final vector to see whether
the solution can still descend materially.

No new physical starts. No bounds/seed/morphology changes.
"""
import sys
import os
import copy
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bounds_audit"))
sys.path.insert(0, HERE)

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row", "71181",
    "--bounds-profile", "production_candidate",
    "--fit-scope", "h1",
    "--dry-run",
]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pathlib import Path  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402
from fisher_engine import numerical_jacobian  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"
assert core.FIT_SCOPE == "h1"

SURVIVORS = [87786, 50179, 86451]
STARTS = ["truth", "H0_NESTED_piE_0", "truth_half_piE", "truth_mirror_u0_piEN"]

ROOT4 = Path("~/Downloads/hidden_parallax/production_validation/gate2_controlled4_extreme100").expanduser()
ref = pd.read_csv(os.path.join(HERE, "results", "gate2_start_ablation_per_event.csv")).set_index("catalog_row")

POLISH_OPTIMIZER_OPTIONS = {
    "xtol": 1.0e-12,
    "ftol": 1.0e-12,
    "gtol": 1.0e-8,
    "max_nfev": 50000,
    "x_scale": "jac",
}


def data_driven_t0_bounds(meta):
    all_times = []
    for band in ["u", "g", "r", "i", "z", "y"]:
        lc = meta["curves"][band]
        if len(lc):
            all_times.extend(np.asarray(lc[:, 0], dtype=float).tolist())
    all_times = np.asarray(all_times, dtype=float)
    return float(np.min(all_times)), float(np.max(all_times))


def build_bounds(meta):
    t_min, t_max = data_driven_t0_bounds(meta)
    bounds = copy.deepcopy(core.H1_BOUNDS)
    bounds = core.apply_bounds_profile(bounds, h1=True, profile=core.BOUNDS_PROFILE)
    bounds["t0"] = {
        "type": "center_width",
        "center": 0.5 * (t_min + t_max),
        "half_width": 0.5 * (t_max - t_min),
    }
    return bounds


def run_full_fit(meta, initial, optimizer_options, label):
    bounds = build_bounds(meta)
    fit, e, model = fit_lc.fit_rubin_roman(
        Source=meta["Source"], event_params=meta["true_params"],
        path_save=f"/tmp/h1f2_discard_{label}", path_ephemerides=str(core.EPHEMERIDES),
        model="FSPL", algo="TRF", Origin=None, rango=meta["rango"],
        wfirst_lc=meta["curves"]["W149"], lsst_u=meta["curves"]["u"],
        lsst_g=meta["curves"]["g"], lsst_r=meta["curves"]["r"],
        lsst_i=meta["curves"]["i"], lsst_z=meta["curves"]["z"],
        lsst_y=meta["curves"]["y"], fit_model="FSPL", fit_parallax=True,
        fit_defaults=None, fit_bounds=bounds, random_state=20260909 + int(meta["row"]),
        initial_guess=initial, optimizer_options=optimizer_options,
        event_ra=meta["event_ra"], event_dec=meta["event_dec"],
    )
    return fit.fit_results, bounds


def build_forward_fit_object(meta, bounds):
    """A fresh TRFfit-like object bound to this event, for a zero-step
    objective_function() gradient evaluation at an arbitrary point."""
    lsst_lcs = {b: meta["curves"][b] for b in ["u", "g", "r", "i", "z", "y"]}
    event_ra_use, event_dec_use = fit_lc.resolve_event_coordinates(
        event_ra=meta["event_ra"], event_dec=meta["event_dec"],
    )
    e = fit_lc.create_fit_event(
        meta["Source"], str(core.EPHEMERIDES), meta["curves"]["W149"], lsst_lcs,
        ra=event_ra_use, dec=event_dec_use, roman_name="Roman",
    )
    fit_params = fit_lc.initial_params_for_fit_model(
        meta["true_params"], "FSPL", fit_parallax=True, fit_defaults=None,
    )
    pyLIMAmodel = fit_lc.build_fit_pyLIMA_model(
        e, "FSPL", fit_params, Origin=None, fit_parallax=True,
    )
    fit_obj, _ = fit_lc.build_fitter(pyLIMAmodel, "TRF")
    fit_lc.apply_fit_bounds(
        fit_obj, fit_params, "FSPL", meta["rango"], fit_parallax=True, fit_bounds=bounds,
    )
    param_order = fit_lc.fit_parameter_order("FSPL", fit_parallax=True)
    return fit_obj, param_order


STATUS_MEANING = {
    -1: "improper_input",
    0: "max_nfev_reached",
    1: "gtol_satisfied",
    2: "ftol_satisfied",
    3: "xtol_satisfied",
    4: "ftol_and_xtol_satisfied",
}

records = []
for row in SURVIVORS:
    print("=" * 100)
    print("row", row)

    d4 = pd.read_csv(list(ROOT4.rglob(f"extreme100/{row}/all_refits.csv"))[0])
    d4 = d4[(d4["hypothesis"] == "H1") & (d4["status"] == "success")]
    best_label, best_chi2, best_row = None, np.inf, None
    for label in STARTS:
        x = d4[d4["label"] == label].iloc[0]
        if x["chi2"] < best_chi2:
            best_chi2, best_label, best_row = x["chi2"], label, x

    meta = core.load_case({"sample": "extreme100", "catalog_row": int(row), "matched_tail_catalog_row": -1})

    polish_initial = {
        "t0": float(best_row["t0"]), "u0": float(best_row["u0"]),
        "tE": float(best_row["tE"]), "rho": float(best_row["rho"]),
        "piEN": float(best_row["piEN"]), "piEE": float(best_row["piEE"]),
    }

    fr, bounds = run_full_fit(meta, polish_initial, POLISH_OPTIMIZER_OPTIONS, f"{row}_polish")

    status = fr.get("optimizer_status")
    message = fr.get("optimizer_message")
    nfev = fr.get("optimizer_nfev")
    njev = fr.get("optimizer_njev")
    optimality = fr.get("optimizer_optimality")
    active_mask = fr.get("optimizer_active_mask")
    chi2 = float(fr["chi2"])
    best = np.asarray(fr["best_model"], dtype=float)
    param_order = fr.get("initial_guess_parameter_order")

    final_vec = {name: float(best[k]) for k, name in enumerate(param_order)}

    # gradient at the final point, same objective, zero-step (fresh
    # fit object, no optimizer state carried over)
    fit_obj, po = build_forward_fit_object(meta, bounds)
    assert po == param_order
    x_final = np.array([final_vec[n] for n in po])
    scale = [final_vec["tE"], 1.0, final_vec["tE"], final_vec["rho"], 1.0, 1.0]
    J, r0 = numerical_jacobian(fit_obj.objective_function, x_final, 1e-6, scale)
    chi2_check = float(np.sum(r0 ** 2))
    grad = J.T @ r0  # d(0.5 * sum r^2)/dx = J^T r ; grad of chi2 = 2 * J^T r

    # bound-aware projected gradient: zero out components at an active
    # bound whose gradient points further into the bound (i.e. the
    # unconstrained direction of decrease is infeasible there).
    lb = np.array([
        bounds["t0"]["center"] - bounds["t0"]["half_width"],
        bounds["u0"][0] if isinstance(bounds["u0"], (list, tuple)) else np.nan,
        None, None, None, None,
    ], dtype=object)
    if active_mask is None:
        active = np.zeros(len(x_final))
    elif isinstance(active_mask, str):
        active = np.array(eval(active_mask))
    else:
        active = np.atleast_1d(np.asarray(active_mask))
    proj_grad = grad.copy()
    for i, a in enumerate(active):
        if a != 0:
            # active at lower(-1) or upper(+1) bound: decreasing chi2
            # further would require moving outside the bound iff
            # grad component has the sign that pushes further out.
            if (a == -1 and grad[i] > 0) or (a == 1 and grad[i] < 0):
                proj_grad[i] = 0.0
    proj_grad_norm = float(np.max(np.abs(proj_grad)))
    raw_grad_norm = float(np.max(np.abs(grad)))

    rec = {
        "catalog_row": int(row),
        "controlled4_winner_label": best_label,
        "chi2_polish": chi2,
        "chi2_check_forward": chi2_check,
        "optimizer_status": status,
        "status_meaning": STATUS_MEANING.get(status, "unknown"),
        "optimizer_message": message,
        "nfev": nfev, "njev": njev,
        "optimality_scipy": optimality,
        "active_mask": active_mask,
        "raw_grad_linf_norm": raw_grad_norm,
        "proj_grad_linf_norm_bound_aware": proj_grad_norm,
        **{f"final_{k}": v for k, v in final_vec.items()},
    }
    records.append(rec)
    print(f"status={status} ({STATUS_MEANING.get(status)}) message={message!r}")
    print(f"nfev={nfev} njev={njev} chi2={chi2:.6f} chi2_check={chi2_check:.6f}")
    print(f"optimality(scipy)={optimality} active_mask={active_mask}")
    print(f"raw grad linf={raw_grad_norm:.6g}  bound-aware projected grad linf={proj_grad_norm:.6g}")
    print(f"final vector: {final_vec}")

    # continuation polish from the EXACT same final vector, if the
    # bound-aware gradient is not clearly near zero (diagnosis only)
    if proj_grad_norm > 1e-3:
        cont_fr, _ = run_full_fit(meta, final_vec, POLISH_OPTIMIZER_OPTIONS, f"{row}_continuation")
        cont_chi2 = float(cont_fr["chi2"])
        print(f"CONTINUATION from exact final vector: chi2 {chi2:.6f} -> {cont_chi2:.6f} "
              f"(further improvement={chi2-cont_chi2:.6g}) status={cont_fr.get('optimizer_status')} "
              f"optimality={cont_fr.get('optimizer_optimality')}")
        rec["continuation_chi2"] = cont_chi2
        rec["continuation_improvement"] = chi2 - cont_chi2
        rec["continuation_status"] = cont_fr.get("optimizer_status")
        rec["continuation_optimality"] = cont_fr.get("optimizer_optimality")
    else:
        rec["continuation_chi2"] = None
        rec["continuation_improvement"] = None
        print("gradient already near zero (bound-aware) -- no continuation run")

df = pd.DataFrame(records)
out_csv = os.path.join(HERE, "results", "h1_f2_convergence_audit.csv")
df.to_csv(out_csv, index=False)
print()
print("DONE ->", out_csv)
