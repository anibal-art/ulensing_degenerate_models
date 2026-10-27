#!/usr/bin/env python3
"""
Stage B: short bounded TRF scouts, exactly gate1_final18's 18 strategies,
run under their OWN coordinate mode (mode is baked in at import time by
core.py's TRF_COORDS runtime patch, hence one subprocess per mode).

Budgets frozen: max_nfev in {3, 5, 10}. Everything else (bounds profile,
bounded flux profiling, xtol/ftol/gtol, x_scale, Jacobian) identical to
the production optimizer_options.
"""
import os as _os_path_setup
HERE = _os_path_setup.path.dirname(_os_path_setup.path.abspath(__file__))

import sys
import os
import copy
import time
import argparse

CORE_DIR = (
    "/home/anibal-pc/ulensing_degenerate_models/"
    "Parallax_LSST/lsstmonts_catalog_sedighe/validation/bounds_audit"
)
SCRATCH = _os_path_setup.path.join(HERE, "results")
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["physical", "log_te", "log_rho", "log_te_rho"])
ap.add_argument("--dummy-import-row", type=int, default=71181)
args = ap.parse_args()

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = args.mode
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"

sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row", str(args.dummy_import_row),
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

GATE1_FINAL18 = {
    "physical": [("old_H0", 1.0), ("truth", 0.1), ("truth", 1.0)],
    "log_te": [("old_H0", "truth_rho"), ("old_H0", 0.01), ("old_H0", 0.1),
               ("old_H0", 1.0), ("truth", 0.01), ("truth", 1.0)],
    "log_rho": [("old_H0", "truth_rho"), ("old_H0", 0.01), ("old_H0", 1.0),
                ("truth", "truth_rho"), ("truth", 1e-6), ("truth", 1e-4),
                ("truth", 0.1), ("truth", 1.0)],
    "log_te_rho": [("truth", 1e-6)],
}
STRATS = GATE1_FINAL18[args.mode]
BUDGETS = [3, 5, 10]

truth_feat = pd.read_csv(
    "/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/b1c_event_features.csv"
).set_index("catalog_row")
old_h0_vec = pd.read_csv(f"{SCRATCH}/old_h0_vector_current_objective.csv").set_index("catalog_row")

rows_100 = sorted(truth_feat.index.tolist())
assert len(rows_100) == 100


def data_driven_t0_bounds(meta):
    all_times = []
    for band in ["u", "g", "r", "i", "z", "y"]:
        lc = meta["curves"][band]
        if len(lc):
            all_times.extend(np.asarray(lc[:, 0], dtype=float).tolist())
    all_times = np.asarray(all_times, dtype=float)
    return float(np.min(all_times)), float(np.max(all_times))


def rho_value(rho_rule, rho_truth, rho_old_own):
    if rho_rule == "truth_rho":
        return rho_truth
    return float(rho_rule)


records = []
t_start = time.time()
n_total = len(rows_100) * len(STRATS) * len(BUDGETS)
n_done = 0

for row in rows_100:
    record = {"sample": "extreme100", "catalog_row": int(row), "matched_tail_catalog_row": -1}
    meta = core.load_case(record)

    t_min, t_max = data_driven_t0_bounds(meta)
    bounds = copy.deepcopy(core.H0_BOUNDS)
    bounds = core.apply_bounds_profile(bounds, h1=False, profile=core.BOUNDS_PROFILE)
    bounds["t0"] = {
        "type": "center_width",
        "center": 0.5 * (t_min + t_max),
        "half_width": 0.5 * (t_max - t_min),
    }

    t0_truth = float(truth_feat.loc[row, "t0_true"])
    u0_truth = float(truth_feat.loc[row, "u0_true"])
    tE_truth = float(truth_feat.loc[row, "tE_true"])
    rho_truth = float(truth_feat.loc[row, "rho_true"])

    t0_old = float(old_h0_vec.loc[row, "old_h0_t0"])
    u0_old = float(old_h0_vec.loc[row, "old_h0_u0"])
    tE_old = float(old_h0_vec.loc[row, "old_h0_tE"])
    rho_old_own = float(old_h0_vec.loc[row, "old_h0_rho"])
    chi2_old_asis = float(old_h0_vec.loc[row, "chi2_old_H0_vector_current_objective"])

    for anchor, rho_rule in STRATS:
        if anchor == "old_H0":
            t0v, u0v, tEv = t0_old, u0_old, tE_old
        else:
            t0v, u0v, tEv = t0_truth, u0_truth, tE_truth

        rhov = rho_value(rho_rule, rho_truth, rho_old_own)

        if anchor == "old_H0" and rho_rule == "rho_from_old_H0":
            chi2_start = chi2_old_asis
        else:
            chi2_start = None  # computed as part of the scout's first eval anyway

        initial = {"t0": t0v, "u0": u0v, "tE": tEv, "rho": rhov}

        for budget in BUDGETS:
            opt_opts = copy.deepcopy(core.OPTIMIZER_OPTIONS)
            opt_opts["max_nfev"] = budget

            t0c = time.time()
            fit, e, model = fit_lc.fit_rubin_roman(
                Source=meta["Source"], event_params=meta["true_params"],
                path_save="/tmp/scout_discard", path_ephemerides=str(core.EPHEMERIDES),
                model="FSPL", algo="TRF", Origin=None, rango=meta["rango"],
                wfirst_lc=meta["curves"]["W149"], lsst_u=meta["curves"]["u"],
                lsst_g=meta["curves"]["g"], lsst_r=meta["curves"]["r"],
                lsst_i=meta["curves"]["i"], lsst_z=meta["curves"]["z"],
                lsst_y=meta["curves"]["y"], fit_model="FSPL", fit_parallax=False,
                fit_defaults=None, fit_bounds=bounds, random_state=20260909 + int(row),
                initial_guess=initial, optimizer_options=opt_opts,
                event_ra=meta["event_ra"], event_dec=meta["event_dec"],
            )
            dt = time.time() - t0c
            fr = fit.fit_results
            best = np.asarray(fr["best_model"], dtype=float)

            records.append({
                "catalog_row": int(row),
                "mode": args.mode,
                "anchor": anchor,
                "rho_tag": ("truth_rho" if rho_rule == "truth_rho" else f"{float(rho_rule):.10g}"),
                "budget": budget,
                "chi2_scout": float(fr["chi2"]),
                "t0_scout": float(best[0]), "u0_scout": float(best[1]),
                "tE_scout": float(best[2]), "rho_scout": float(best[3]),
                "nfev": fr.get("optimizer_nfev"),
                "njev": fr.get("optimizer_njev"),
                "optimizer_status": fr.get("optimizer_status"),
                "optimizer_success": fr.get("optimizer_success"),
                "wall_time_s": dt,
            })
            n_done += 1
            del fit, e, model

    if n_done % 300 < len(STRATS) * len(BUDGETS):
        print(f"[{args.mode}] row={row} n_done={n_done}/{n_total} elapsed={time.time()-t_start:.0f}s", flush=True)

df_out = pd.DataFrame(records)
OUT_CSV = os.path.join(SCRATCH, f"h0_scout_stageB_{args.mode}.csv")
df_out.to_csv(OUT_CSV, index=False)
print(f"DONE [{args.mode}] ->", OUT_CSV, "n_rows=", len(df_out), "total wall=", time.time() - t_start)
