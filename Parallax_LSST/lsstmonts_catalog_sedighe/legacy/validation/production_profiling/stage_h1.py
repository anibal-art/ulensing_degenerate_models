#!/usr/bin/env python3
"""
Stage 3 (physical mode, per event): H1.

Part A -- 'old_final' legacy producer: 10 unconditional real TRF
(3x3 piE-fraction grid [-0.5,0,0.5]^2 + 1 center_full point),
embedded at the old_H0 bridge vector (t0,u0,tE,rho), old narrow FAST
H1 bounds (piEN half_width=1.9973, piEE half_width=2.201096, matching
configs/validation/lrt_benchmark/FAST.json), standard optimizer
options. 0-1 conditional polish TRF (tighter tolerances) if the
coarse winner's optimality is non-finite or > 0.05.

Part B -- 'controlled5' H1 policy: 5 real TRF, production_candidate
bounds: truth, H0_NESTED_piE_0 (uses the aggregated H0 final passed
in via --h0-final-json), truth_half_piE, truth_mirror_u0_piEN,
old_final_reseed (uses Part A's own winning vector -- computed here,
not read from any legacy dataset column).
"""
import sys
import os
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
BA = os.path.join(HERE, "..", "bounds_audit")
sys.path.insert(0, BA)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--h5", required=True)
ap.add_argument("--bridge-json", required=True)
ap.add_argument("--h0-final-json", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = ["run_bounds_audit_refit_core.py", "--catalog-row", "71181",
            "--bounds-profile", "production_candidate", "--fit-scope", "h1", "--dry-run"]

import numpy as np  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402
from load_new_event import load_new_case  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"

meta = load_new_case(args.h5, core)

with open(args.bridge_json) as f:
    bridge_data = json.load(f)
with open(args.h0_final_json) as f:
    h0_final = json.load(f)

bridge_vector = bridge_data["bridge_vector"]

# ---- Part A: old_final legacy producer ----
OLD_FINAL_H1_BOUNDS = {
    "t0": {"type": "center_width", "half_width": 30.0},
    "u0": {"type": "center_width", "half_width": 1.0},
    "tE": {"type": "relative", "frac": 1.0, "lower": 0.1, "upper": 20000.0, "min_width": 0.1},
    "rho": {"type": "relative", "frac": 1.0, "lower": 1.0e-7, "upper": 10.0, "min_width": 1.0e-7},
    "piEN": {"type": "center_width", "center": 0.0, "half_width": 1.9973},
    "piEE": {"type": "center_width", "center": 0.0, "half_width": 2.2010959999999997},
}
OLD_OPTIMIZER_OPTIONS = {"xtol": 1e-8, "ftol": 1e-8, "gtol": 1e-8, "max_nfev": 50000, "x_scale": "jac"}
POLISH_OPTIMIZER_OPTIONS = {"xtol": 1e-10, "ftol": 1e-10, "gtol": 1e-8, "max_nfev": 50000, "x_scale": "jac"}
POLISH_OPTIMALITY_THRESHOLD = 0.05
WIDTH_N = 1.9973
WIDTH_E = 2.2010959999999997
FRACTIONS = [-0.5, 0.0, 0.5]


def run_h1_fit(meta, initial, fit_bounds, optimizer_options, label, t0_center):
    # t0/u0 bounds in OLD_FINAL_H1_BOUNDS have no explicit 'center', so
    # resolve_bound_spec falls back to the 'center' kwarg it is called
    # with internally -- which fit_rubin_roman derives from initial_guess.
    fit, e, model = fit_lc.fit_rubin_roman(
        Source=meta["Source"], event_params=meta["true_params"], path_save=f"/tmp/prof_discard_h1_{label}",
        path_ephemerides=str(core.EPHEMERIDES), model="FSPL", algo="TRF", Origin=None, rango=meta["rango"],
        wfirst_lc=meta["curves"]["W149"], lsst_u=meta["curves"]["u"], lsst_g=meta["curves"]["g"],
        lsst_r=meta["curves"]["r"], lsst_i=meta["curves"]["i"], lsst_z=meta["curves"]["z"],
        lsst_y=meta["curves"]["y"], fit_model="FSPL", fit_parallax=True, fit_defaults=None,
        fit_bounds=fit_bounds, random_state=20260909 + int(meta["row"]), initial_guess=initial,
        optimizer_options=optimizer_options, event_ra=meta["event_ra"], event_dec=meta["event_dec"],
    )
    return fit.fit_results


grid_values_n = [f * WIDTH_N for f in FRACTIONS]
grid_values_e = [f * WIDTH_E for f in FRACTIONS]

candidates = [("center_full", 0.0, 0.0)]
for vn in grid_values_n:
    for ve in grid_values_e:
        candidates.append((f"grid_{vn:+.4f}_{ve:+.4f}", vn, ve))

grid_records = []
for label, pien, piee in candidates:
    initial = {"t0": bridge_vector["t0"], "u0": bridge_vector["u0"], "tE": bridge_vector["tE"],
               "rho": bridge_vector["rho"], "piEN": pien, "piEE": piee}
    t0c, cpu0 = time.time(), time.process_time()
    fr = run_h1_fit(meta, initial, OLD_FINAL_H1_BOUNDS, OLD_OPTIMIZER_OPTIONS, label, bridge_vector["t0"])
    wall_s, cpu_s = time.time() - t0c, time.process_time() - cpu0
    best = np.asarray(fr["best_model"], dtype=float)
    grid_records.append({
        "label": label, "chi2": float(fr["chi2"]), "wall_s": wall_s, "cpu_s": cpu_s,
        "optimality": fr.get("optimizer_optimality"), "status": fr.get("optimizer_status"),
        "t0": float(best[0]), "u0": float(best[1]), "tE": float(best[2]), "rho": float(best[3]),
        "piEN": float(best[4]), "piEE": float(best[5]),
    })

finite = [r for r in grid_records if np.isfinite(r["chi2"])]
coarse_winner = min(finite, key=lambda r: r["chi2"]) if finite else grid_records[0]

polish_ran = False
polish_record = None
opt = coarse_winner.get("optimality")
needs_polish = (opt is None) or (not np.isfinite(opt)) or (opt > POLISH_OPTIMALITY_THRESHOLD)

if needs_polish:
    polish_ran = True
    initial = {"t0": coarse_winner["t0"], "u0": coarse_winner["u0"], "tE": coarse_winner["tE"],
               "rho": coarse_winner["rho"], "piEN": coarse_winner["piEN"], "piEE": coarse_winner["piEE"]}
    t0c, cpu0 = time.time(), time.process_time()
    fr = run_h1_fit(meta, initial, OLD_FINAL_H1_BOUNDS, POLISH_OPTIMIZER_OPTIONS, "polish", bridge_vector["t0"])
    wall_s, cpu_s = time.time() - t0c, time.process_time() - cpu0
    best = np.asarray(fr["best_model"], dtype=float)
    polish_record = {
        "label": "polish", "chi2": float(fr["chi2"]), "wall_s": wall_s, "cpu_s": cpu_s,
        "optimality": fr.get("optimizer_optimality"), "status": fr.get("optimizer_status"),
        "t0": float(best[0]), "u0": float(best[1]), "tE": float(best[2]), "rho": float(best[3]),
        "piEN": float(best[4]), "piEE": float(best[5]),
    }

old_final_vector = polish_record if (polish_ran and np.isfinite(polish_record["chi2"])
                                      and polish_record["chi2"] <= coarse_winner["chi2"]) else coarse_winner

old_final_wall_s = sum(r["wall_s"] for r in grid_records) + (polish_record["wall_s"] if polish_record else 0.0)
old_final_cpu_s = sum(r["cpu_s"] for r in grid_records) + (polish_record["cpu_s"] if polish_record else 0.0)
old_final_n_trf = len(grid_records) + (1 if polish_ran else 0)

# ---- Part B: controlled5 H1 policy (production_candidate bounds) ----
truth = meta["truth"]

truth_initial = {"t0": truth["t0"], "u0": truth["u0"], "tE": truth["tE"], "rho": truth["rho"],
                  "piEN": truth["piEN"], "piEE": truth["piEE"]}
nested_initial = {"t0": h0_final["t0"], "u0": h0_final["u0"], "tE": h0_final["tE"], "rho": h0_final["rho"],
                   "piEN": 0.0, "piEE": 0.0}
half_pie_initial = {"t0": truth["t0"], "u0": truth["u0"], "tE": truth["tE"], "rho": truth["rho"],
                     "piEN": 0.5 * truth["piEN"], "piEE": 0.5 * truth["piEE"]}
mirror_initial = {"t0": truth["t0"], "u0": -truth["u0"], "tE": truth["tE"], "rho": truth["rho"],
                   "piEN": -truth["piEN"], "piEE": truth["piEE"]}
old_final_reseed_initial = {"t0": old_final_vector["t0"], "u0": old_final_vector["u0"],
                             "tE": old_final_vector["tE"], "rho": old_final_vector["rho"],
                             "piEN": old_final_vector["piEN"], "piEE": old_final_vector["piEE"]}

controlled5_starts = [
    ("truth", truth_initial),
    ("H0_NESTED_piE_0", nested_initial),
    ("truth_half_piE", half_pie_initial),
    ("truth_mirror_u0_piEN", mirror_initial),
    ("old_final_reseed", old_final_reseed_initial),
]

controlled5_records = []
for label, initial in controlled5_starts:
    t0c, cpu0 = time.time(), time.process_time()
    r = core.run_one_fit(meta, "H1", initial, label)
    wall_s, cpu_s = time.time() - t0c, time.process_time() - cpu0
    controlled5_records.append({**r, "wall_s": wall_s, "cpu_s": cpu_s})

finite_c5 = [r for r in controlled5_records if np.isfinite(r.get("chi2", np.nan))]
h1_final = min(finite_c5, key=lambda r: r["chi2"]) if finite_c5 else None

out = {
    "catalog_row": meta["row"],
    "old_final_grid": grid_records,
    "old_final_polish_ran": polish_ran,
    "old_final_polish": polish_record,
    "old_final_vector": old_final_vector,
    "old_final_n_trf": old_final_n_trf,
    "old_final_wall_s": old_final_wall_s,
    "old_final_cpu_s": old_final_cpu_s,
    "controlled5": controlled5_records,
    "h1_final_chi2": h1_final["chi2"] if h1_final else None,
    "h1_final_label": h1_final["label"] if h1_final else None,
}
with open(args.out, "w") as f:
    json.dump(out, f, indent=2)

print(f"row={meta['row']} old_final_n_trf={old_final_n_trf} old_final_wall={old_final_wall_s:.2f}s "
      f"controlled5_wall={sum(r['wall_s'] for r in controlled5_records):.2f}s "
      f"h1_final={out['h1_final_label']} chi2={out['h1_final_chi2']} -> {args.out}")
