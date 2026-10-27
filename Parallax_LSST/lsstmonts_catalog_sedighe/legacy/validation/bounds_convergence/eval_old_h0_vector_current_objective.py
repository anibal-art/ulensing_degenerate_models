#!/usr/bin/env python3
"""
Forward-evaluate the raw old_H0[:4] vector (t0,u0,tE,rho) with the
CURRENT objective (bounded_flux_profile patch, production bounds
profile), zero optimizer steps -- for all 100 extreme100 events.

This is NOT a new fit: old_H0 already exists (it's the legacy bridge
vector our fitter already loads via load_case/meta['old_h0']). We are
only re-scoring that existing point under the current objective, to
get a chi2 that is directly comparable to everything else in the 52-
oracle table (same flux treatment, same bounds domain for u0/tE/rho).
"""
import sys
import os
import copy
import time

CORE_DIR = (
    "/home/anibal-pc/ulensing_degenerate_models/"
    "Parallax_LSST/lsstmonts_catalog_sedighe/validation/bounds_audit"
)
sys.path.insert(0, CORE_DIR)

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


# ------------------------------------------------------------------
# 100 extreme100 catalog rows, from already-computed oracle52 data.
# ------------------------------------------------------------------
oracle = pd.read_csv(
    "/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/"
    "lsstmonts_catalog_sedighe/validation/bounds_convergence/results/"
    "gate1_oracle_diagnostics.csv"
)
rows_100 = sorted(oracle[oracle["source"] == "gate1_oracle_52"]["catalog_row"].unique().tolist())
print("n events:", len(rows_100))

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(HERE, "results")
OUT_CSV = os.path.join(SCRATCH, "old_h0_vector_current_objective.csv")

records = []
t_start = time.time()

for i, row in enumerate(rows_100, 1):
    record = {
        "sample": "extreme100",
        "catalog_row": int(row),
        "matched_tail_catalog_row": -1,
    }
    meta = core.load_case(record)

    old_h0_vec = np.asarray(meta["old_h0"], dtype=float)[:4]

    fit_obj, param_order = build_h0_forward_fit(meta)
    assert param_order == ["t0", "u0", "tE", "rho"], param_order

    x0 = old_h0_vec
    t0c = time.time()
    residuals = fit_obj.objective_function(x0)
    chi2 = float(np.sum(np.asarray(residuals, dtype=float) ** 2))
    dt = time.time() - t0c

    records.append({
        "catalog_row": int(row),
        "old_h0_t0": x0[0], "old_h0_u0": x0[1], "old_h0_tE": x0[2], "old_h0_rho": x0[3],
        "chi2_old_H0_vector_current_objective": chi2,
        "wall_time_s": dt,
    })

    if i % 10 == 0 or i == len(rows_100):
        print(f"[{i}/{len(rows_100)}] row={row} chi2={chi2:.4f} elapsed={time.time()-t_start:.0f}s")

pd.DataFrame(records).to_csv(OUT_CSV, index=False)
print("DONE ->", OUT_CSV)
