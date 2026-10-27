#!/usr/bin/env python3
"""
Stage A: fixed-start objective scouting for H0.

For each of the 100 extreme100 events, evaluate the CURRENT profiled
objective (bounded_flux_profile, production_candidate bounds domain,
zero optimizer steps) at the literal INITIAL vector of each of the 13
distinct physical starting points used by the 52-strategy oracle
(7 old_H0-anchored rho rules + 6 truth-anchored rho rules). Mode
(physical/log_te/log_rho/log_te_rho) does not affect the initial
vector -- it only changes the internal TRF coordinate parametrization
during optimization -- so chi2_at_start is identical across the 4
modes for the same (anchor, rho_rule); the 52-strategy ranking will
show that explicitly as tied groups of 4.
"""
import os as _os_path_setup
HERE = _os_path_setup.path.dirname(_os_path_setup.path.abspath(__file__))

import sys
import os
import copy
import time

CORE_DIR = (
    "/home/anibal-pc/ulensing_degenerate_models/"
    "Parallax_LSST/lsstmonts_catalog_sedighe/validation/bounds_audit"
)
sys.path.insert(0, CORE_DIR)
SCRATCH = _os_path_setup.path.join(HERE, "results")
sys.path.insert(0, HERE)

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"

sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row", "71181",
    "--bounds-profile", "production_candidate",
    "--fit-scope", "h0",
    "--dry-run",
]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"
assert core.FIT_SCOPE == "h0"


def data_driven_t0_bounds(meta):
    all_times = []
    for band in ["u", "g", "r", "i", "z", "y"]:
        lc = meta["curves"][band]
        if len(lc):
            all_times.extend(np.asarray(lc[:, 0], dtype=float).tolist())
    all_times = np.asarray(all_times, dtype=float)
    return float(np.min(all_times)), float(np.max(all_times))


def build_h0_forward_fit(meta):
    t_min, t_max = data_driven_t0_bounds(meta)
    bounds = copy.deepcopy(core.H0_BOUNDS)
    bounds = core.apply_bounds_profile(bounds, h1=False, profile=core.BOUNDS_PROFILE)
    bounds["t0"] = {
        "type": "center_width",
        "center": 0.5 * (t_min + t_max),
        "half_width": 0.5 * (t_max - t_min),
    }
    lsst_lcs = {b: meta["curves"][b] for b in ["u", "g", "r", "i", "z", "y"]}
    event_ra_use, event_dec_use = fit_lc.resolve_event_coordinates(
        event_ra=meta["event_ra"], event_dec=meta["event_dec"],
    )
    e = fit_lc.create_fit_event(
        meta["Source"], str(core.EPHEMERIDES), meta["curves"]["W149"], lsst_lcs,
        ra=event_ra_use, dec=event_dec_use, roman_name="Roman",
    )
    fit_params = fit_lc.initial_params_for_fit_model(
        meta["true_params"], "FSPL", fit_parallax=False, fit_defaults=None,
    )
    pyLIMAmodel = fit_lc.build_fit_pyLIMA_model(
        e, "FSPL", fit_params, Origin=None, fit_parallax=False,
    )
    fit, pool_processes = fit_lc.build_fitter(pyLIMAmodel, "TRF")
    fit_lc.apply_fit_bounds(
        fit, fit_params, "FSPL", meta["rango"], fit_parallax=False, fit_bounds=bounds,
    )
    param_order = fit_lc.fit_parameter_order("FSPL", fit_parallax=False)
    return fit, param_order


RHO_GRID = [1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0]

truth_feat = pd.read_csv(
    "/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/b1c_event_features.csv"
).set_index("catalog_row")

old_h0_vec = pd.read_csv(f"{SCRATCH}/old_h0_vector_current_objective.csv").set_index("catalog_row")

rows_100 = sorted(truth_feat.index.tolist())
assert len(rows_100) == 100
print("n events:", len(rows_100))

records = []
t_start = time.time()

for i, row in enumerate(rows_100, 1):
    record = {"sample": "extreme100", "catalog_row": int(row), "matched_tail_catalog_row": -1}
    meta = core.load_case(record)
    fit_obj, param_order = build_h0_forward_fit(meta)
    assert param_order == ["t0", "u0", "tE", "rho"], param_order

    t0_truth = float(truth_feat.loc[row, "t0_true"])
    u0_truth = float(truth_feat.loc[row, "u0_true"])
    tE_truth = float(truth_feat.loc[row, "tE_true"])
    rho_truth = float(truth_feat.loc[row, "rho_true"])

    t0_old = float(old_h0_vec.loc[row, "old_h0_t0"])
    u0_old = float(old_h0_vec.loc[row, "old_h0_u0"])
    tE_old = float(old_h0_vec.loc[row, "old_h0_tE"])
    rho_old_own = float(old_h0_vec.loc[row, "old_h0_rho"])
    chi2_old_asis = float(old_h0_vec.loc[row, "chi2_old_H0_vector_current_objective"])

    def eval_chi2(t0v, u0v, tEv, rhov):
        r = fit_obj.objective_function(np.array([t0v, u0v, tEv, rhov]))
        return float(np.sum(np.asarray(r, dtype=float) ** 2))

    starts = []
    # old_H0 anchor, 7 rho rules
    starts.append(("old_H0", "rho_from_old_H0", chi2_old_asis))
    for rv in RHO_GRID:
        starts.append(("old_H0", f"{rv:.10g}", eval_chi2(t0_old, u0_old, tE_old, rv)))
    starts.append(("old_H0", "truth_rho", eval_chi2(t0_old, u0_old, tE_old, rho_truth)))

    # truth anchor, 6 rho rules
    starts.append(("truth", "truth_rho", eval_chi2(t0_truth, u0_truth, tE_truth, rho_truth)))
    for rv in RHO_GRID:
        starts.append(("truth", f"{rv:.10g}", eval_chi2(t0_truth, u0_truth, tE_truth, rv)))

    assert len(starts) == 13

    for anchor, rho_tag, chi2v in starts:
        records.append({
            "catalog_row": int(row),
            "anchor": anchor,
            "rho_tag": rho_tag,
            "chi2_at_start": chi2v,
        })

    if i % 10 == 0 or i == len(rows_100):
        print(f"[{i}/{len(rows_100)}] row={row} elapsed={time.time()-t_start:.0f}s")

df_out = pd.DataFrame(records)
OUT_CSV = os.path.join(SCRATCH, "h0_scout_stageA_starts.csv")
df_out.to_csv(OUT_CSV, index=False)
print("DONE ->", OUT_CSV, "n_rows=", len(df_out))
print("total wall time:", time.time() - t_start)
