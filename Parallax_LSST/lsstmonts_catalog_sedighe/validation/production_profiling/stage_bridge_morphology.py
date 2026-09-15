#!/usr/bin/env python3
"""
Stage 1 (physical mode, per event): old_H0 bridge (1 real TRF, old
narrow FAST bounds, start=truth) + morphology extraction/lookup
(no TRF). Both are mode-agnostic prerequisites consumed by the H0
downstream stage(s) and, for the bridge, the H1 legacy producer.

Uses the CURRENT frozen fitter's bounded_flux_profile objective for
the bridge fit (see ADAPTIVE_H0_ANALYSIS.md for why: we are measuring
cost of the algorithmic recipe with this session's already-verified
fitter, not re-running 2026-era legacy software).
"""
import sys
import os
import copy
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
BC = os.path.join(HERE, "..", "bounds_convergence")
BA = os.path.join(HERE, "..", "bounds_audit")
sys.path.insert(0, BA)
sys.path.insert(0, BC)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--h5", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = ["run_bounds_audit_refit_core.py", "--catalog-row", "71181",
            "--bounds-profile", "production_candidate", "--fit-scope", "h0", "--dry-run"]

import numpy as np  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402
from load_new_event import load_new_case  # noqa: E402
from morphology_extraction import extract_event_morphology  # noqa: E402

t_stage = time.time()
meta = load_new_case(args.h5, core)

# ---- old_H0 bridge: 1 TRF, old narrow FAST bounds, start=truth ----
OLD_H0_BOUNDS = {
    "t0": {"type": "center_width", "center": meta["truth"]["t0"], "half_width": 30.0},
    "u0": [-1.0, 1.0],
    "tE": [0.1, 20000.0],
    "rho": [1.0e-7, 10.0],
}
OLD_OPTIMIZER_OPTIONS = {"xtol": 1e-8, "ftol": 1e-8, "gtol": 1e-8, "max_nfev": 50000, "x_scale": "jac"}


def data_driven_t0_bounds(meta):
    all_t = []
    for b in ["u", "g", "r", "i", "z", "y"]:
        lc = meta["curves"][b]
        if len(lc):
            all_t.extend(np.asarray(lc[:, 0], dtype=float).tolist())
    return float(min(all_t)), float(max(all_t))


def run_fit(meta, initial, fit_bounds, fit_parallax, optimizer_options, label):
    fit, e, model = fit_lc.fit_rubin_roman(
        Source=meta["Source"], event_params=meta["true_params"], path_save=f"/tmp/prof_discard_{label}",
        path_ephemerides=str(core.EPHEMERIDES), model="FSPL", algo="TRF", Origin=None, rango=meta["rango"],
        wfirst_lc=meta["curves"]["W149"], lsst_u=meta["curves"]["u"], lsst_g=meta["curves"]["g"],
        lsst_r=meta["curves"]["r"], lsst_i=meta["curves"]["i"], lsst_z=meta["curves"]["z"],
        lsst_y=meta["curves"]["y"], fit_model="FSPL", fit_parallax=fit_parallax, fit_defaults=None,
        fit_bounds=fit_bounds, random_state=20260909 + int(meta["row"]), initial_guess=initial,
        optimizer_options=optimizer_options, event_ra=meta["event_ra"], event_dec=meta["event_dec"],
    )
    return fit.fit_results


t0c = time.time()
cpu0 = time.process_time()
bridge_initial = {"t0": meta["truth"]["t0"], "u0": meta["truth"]["u0"],
                   "tE": meta["truth"]["tE"], "rho": meta["truth"]["rho"]}
bridge_fr = run_fit(meta, bridge_initial, OLD_H0_BOUNDS, False, OLD_OPTIMIZER_OPTIONS, "bridge")
bridge_wall = time.time() - t0c
bridge_cpu = time.process_time() - cpu0
bridge_best = np.asarray(bridge_fr["best_model"], dtype=float)
bridge_vector = {"t0": float(bridge_best[0]), "u0": float(bridge_best[1]),
                  "tE": float(bridge_best[2]), "rho": float(bridge_best[3])}

# ---- morphology extraction + FSPL lookup inversion (no TRF) ----
t0c = time.time()
cpu0 = time.process_time()
morph_raw = extract_event_morphology(meta["curves"])

morph_candidate = None
if morph_raw.get("measurable"):
    import pandas as pd

    def eff_width(entry):
        if entry is None:
            return None
        if not entry["left_censored"] and not entry["right_censored"] and entry["value"] is not None:
            return float(entry["value"])
        if not entry["left_censored"] and entry["left_width"] is not None and entry["right_censored"]:
            return 2.0 * float(entry["left_width"])
        if not entry["right_censored"] and entry["right_width"] is not None and entry["left_censored"]:
            return 2.0 * float(entry["right_width"])
        return None

    e25, e50, e75 = eff_width(morph_raw["T25"]), eff_width(morph_raw["T50"]), eff_width(morph_raw["T75"])
    if e25 and e50 and e75 and e50 > 0:
        lookup = pd.read_csv(os.path.join(BC, "results", "fspl_dimensionless_lookup.csv"))
        obs25, obs75 = e25 / e50, e75 / e50
        mismatch = np.sqrt((lookup["T25_over_T50"].values - obs25) ** 2 +
                            (lookup["T75_over_T50"].values - obs75) ** 2)
        best_idx = int(np.argmin(mismatch))
        u0m = float(lookup["u0"].values[best_idx])
        rhom = float(lookup["rho"].values[best_idx])
        tEm = e50 / float(lookup["T50_over_tE"].values[best_idx])
        morph_candidate = {"t0": float(morph_raw["t_peak_morph"]), "u0": u0m, "tE": tEm, "rho": rhom,
                            "mismatch": float(mismatch[best_idx])}

morph_wall = time.time() - t0c
morph_cpu = time.process_time() - cpu0

out = {
    "catalog_row": meta["row"],
    "bridge_vector": bridge_vector,
    "bridge_chi2": float(bridge_fr["chi2"]),
    "bridge_wall_s": bridge_wall,
    "bridge_cpu_s": bridge_cpu,
    "bridge_nfev": bridge_fr.get("optimizer_nfev"),
    "bridge_status": bridge_fr.get("optimizer_status"),
    "morphology_candidate": morph_candidate,
    "morphology_measurable": bool(morph_raw.get("measurable")),
    "morphology_wall_s": morph_wall,
    "morphology_cpu_s": morph_cpu,
    "truth": meta["truth"],
    "stage_total_wall_s": time.time() - t_stage,
}

with open(args.out, "w") as f:
    json.dump(out, f, indent=2)

print(f"row={meta['row']} bridge_chi2={out['bridge_chi2']:.4f} morph_ok={morph_candidate is not None} "
      f"bridge_wall={bridge_wall:.2f}s morph_wall={morph_wall:.3f}s -> {args.out}")
