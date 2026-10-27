#!/usr/bin/env python3
"""
CORRECTED decisive test for current_H0_grid_reseed (Part B, v2).

Fix vs v1: grid points are now evaluated with the true fixed-point
objective (fit.objective_function(x0)) -- zero nonlinear optimizer
steps, exact same flux-profiling code TRF uses (bounded_flux_profile
patch), no scipy.optimize.least_squares call at all for the grid.
Only the final winner goes through a real TRF polish (core.run_one_fit,
unchanged from v1).
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
import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"
assert core.FIT_SCOPE == "h1"

REAL_OPTIMIZER_OPTIONS = copy.deepcopy(core.OPTIMIZER_OPTIONS)

PIEN_HALF_WIDTH = core.H1_BOUNDS["piEN"]["half_width"]
PIEE_HALF_WIDTH = core.H1_BOUNDS["piEE"]["half_width"]
FRACTIONS = [-0.5, 0.0, 0.5]

grid_n = [f * PIEN_HALF_WIDTH for f in FRACTIONS]
grid_e = [f * PIEE_HALF_WIDTH for f in FRACTIONS]

print(f"[grid] piEN_half_width={PIEN_HALF_WIDTH} piEE_half_width={PIEE_HALF_WIDTH}")
print(f"[grid] grid_n={grid_n} grid_e={grid_e}")

H0_MANIFEST = (
    "/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/"
    "lsstmonts_catalog_sedighe/validation/bounds_convergence/data/"
    "gate2_h0_anchor_manifest_extreme100.csv"
)
h0 = pd.read_csv(H0_MANIFEST).set_index("catalog_row")

FAIL_EVENTS = [85380, 87786, 79700, 50179, 86451]

SCRATCH = _os_path_setup.path.join(HERE, "results")
OUT_CSV = os.path.join(SCRATCH, "h1_grid_reseed_results_v2.csv")


def data_driven_t0_bounds(meta):
    all_times = []
    for band in ["u", "g", "r", "i", "z", "y"]:
        lc = meta["curves"][band]
        if len(lc):
            all_times.extend(np.asarray(lc[:, 0], dtype=float).tolist())
    all_times = np.asarray(all_times, dtype=float)
    return float(np.min(all_times)), float(np.max(all_times))


def build_h1_forward_fit(meta):
    """
    Replicates fit_lc.fit_rubin_roman()'s SETUP exactly, up to (but not
    including) run_fit(). Returns (fit, param_order) so the caller can
    call fit.objective_function(x0) directly -- zero optimizer steps.
    """
    t_min, t_max = data_driven_t0_bounds(meta)
    bounds = copy.deepcopy(core.H1_BOUNDS)
    bounds = core.apply_bounds_profile(bounds, h1=True, profile=core.BOUNDS_PROFILE)
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
        meta["true_params"], "FSPL", fit_parallax=True, fit_defaults=None,
    )

    pyLIMAmodel = fit_lc.build_fit_pyLIMA_model(
        e, "FSPL", fit_params, Origin=None, fit_parallax=True,
    )

    fit, pool_processes = fit_lc.build_fitter(pyLIMAmodel, "TRF")

    fit_lc.apply_fit_bounds(
        fit, fit_params, "FSPL", meta["rango"], fit_parallax=True, fit_bounds=bounds,
    )

    param_order = fit_lc.fit_parameter_order("FSPL", fit_parallax=True)

    return fit, param_order


records = []

for row in FAIL_EVENTS:
    t_row_start = time.time()
    print("=" * 80)
    print("row", row)

    record = {
        "sample": "extreme100",
        "catalog_row": row,
        "matched_tail_catalog_row": -1,
    }
    meta = core.load_case(record)

    h0row = h0.loc[row]
    t0_h0 = float(h0row["t0"])
    u0_h0 = float(h0row["u0"])
    tE_h0 = float(h0row["tE"])
    rho_h0 = float(h0row["rho"])

    fit_obj, param_order = build_h1_forward_fit(meta)
    print("  param_order =", param_order)

    grid_results = []
    for gi, pn in enumerate(grid_n):
        for gj, pe in enumerate(grid_e):
            params = {
                "t0": t0_h0, "u0": u0_h0, "tE": tE_h0, "rho": rho_h0,
                "piEN": pn, "piEE": pe,
            }
            x0 = np.array([params[k] for k in param_order], dtype=float)

            t0c = time.time()
            residuals = fit_obj.objective_function(x0)
            chi2 = float(np.sum(np.asarray(residuals, dtype=float) ** 2))
            dt = time.time() - t0c

            grid_results.append({
                "piEN": pn, "piEE": pe, "chi2": chi2, "wall_time_s": dt,
            })
            print(f"  [grid] pn={pn:+.5f} pe={pe:+.5f} chi2={chi2:.6f} dt={dt:.4f}s")

    best = min(grid_results, key=lambda r: r["chi2"])
    print(f"  [grid] WINNER piEN={best['piEN']:+.5f} piEE={best['piEE']:+.5f} chi2={best['chi2']}")

    polish_initial = {
        "t0": t0_h0, "u0": u0_h0, "tE": tE_h0, "rho": rho_h0,
        "piEN": best["piEN"], "piEE": best["piEE"],
    }
    core.OPTIMIZER_OPTIONS = REAL_OPTIMIZER_OPTIONS
    t0c = time.time()
    polish = core.run_one_fit(meta, "H1", polish_initial, "current_H0_grid_reseed_v2")
    dt_polish = time.time() - t0c
    print(f"  [polish] chi2={polish.get('chi2')} status={polish.get('status')} dt={dt_polish:.2f}s")

    final_chi2 = min(best["chi2"], polish.get("chi2", np.inf))
    final_source = "grid_winner" if best["chi2"] <= polish.get("chi2", np.inf) else "trf_polish"

    print(f"  [final 5th-candidate] chi2={final_chi2} source={final_source}")
    print(f"  [row total] {time.time()-t_row_start:.1f}s")

    records.append({
        "catalog_row": row,
        "h0_t0": t0_h0, "h0_u0": u0_h0, "h0_tE": tE_h0, "h0_rho": rho_h0,
        "n_grid_evals": len(grid_results),
        "grid_wall_time_total_s": sum(r["wall_time_s"] for r in grid_results),
        "grid_winner_piEN": best["piEN"],
        "grid_winner_piEE": best["piEE"],
        "grid_winner_chi2": best["chi2"],
        "polish_status": polish.get("status"),
        "polish_chi2": polish.get("chi2"),
        "polish_t0": polish.get("t0"),
        "polish_u0": polish.get("u0"),
        "polish_tE": polish.get("tE"),
        "polish_rho": polish.get("rho"),
        "polish_piEN": polish.get("piEN"),
        "polish_piEE": polish.get("piEE"),
        "polish_wall_time_s": dt_polish,
        "final_chi2": final_chi2,
        "final_source": final_source,
    })

    pd.DataFrame(records).to_csv(OUT_CSV, index=False)

print("DONE ->", OUT_CSV)
