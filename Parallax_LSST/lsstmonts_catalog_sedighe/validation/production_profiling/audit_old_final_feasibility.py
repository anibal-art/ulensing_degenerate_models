#!/usr/bin/env python3
"""
Production-feasibility audit (NOT a fitter retune).

Question: does the frozen old_final legacy-producer recipe even reach
its first TRF call on a representative sample of genuinely new events,
or does validate_initial_guess_inside_bounds reject the H0-bridge-
embedded initial guess before any H1 optimization happens?

Per event, cost is exactly:
  - 1 bridge TRF (reused from an existing results/pipeline_stage_json/
    {row}_bridge.json if present -- all 5 serial-baseline events and
    929641's earlier smoke test already have one on disk; the other
    20 events get exactly one fresh bridge TRF here).
  - ZERO H1 TRF calls. Bounds are resolved with fit_lc.resolve_bound_spec
    (the frozen fitter's own function -- not a reimplementation) and
    checked against the bridge vector directly; validate_initial_guess_
    inside_bounds's raise-on-first-failure behavior is replicated
    diagnostically by checking every parameter independently instead
    of stopping at the first one, so partial (e.g. rho-only vs both)
    failures are visible.

No bounds/starts/old_final/bridge are modified. If a start would be
invalid in production, it is marked invalid here exactly the same way.
"""
import sys
import os
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
BA = os.path.join(HERE, "..", "bounds_audit")
BC = os.path.join(HERE, "..", "bounds_convergence")
sys.path.insert(0, BA)
sys.path.insert(0, BC)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--rows-csv", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = ["run_bounds_audit_refit_core.py", "--catalog-row", "71181",
            "--bounds-profile", "production_candidate", "--fit-scope", "h0", "--dry-run"]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402
from load_new_event import load_new_case  # noqa: E402

BRIDGE_JSON_DIR = os.path.join(HERE, "results", "pipeline_stage_json")
os.makedirs(BRIDGE_JSON_DIR, exist_ok=True)

OLD_H0_BOUNDS = {
    "t0": {"type": "center_width", "half_width": 30.0},
    "u0": [-1.0, 1.0],
    "tE": [0.1, 20000.0],
    "rho": [1.0e-7, 10.0],
}
OLD_OPTIMIZER_OPTIONS = {"xtol": 1e-8, "ftol": 1e-8, "gtol": 1e-8, "max_nfev": 50000, "x_scale": "jac"}

# Exact old_final H1 bound specs (configs/validation/lrt_benchmark/FAST.json).
OLD_FINAL_H1_BOUND_SPECS = {
    "t0": {"type": "center_width", "half_width": 30.0},
    "u0": {"type": "center_width", "half_width": 1.0},
    "tE": {"type": "relative", "frac": 1.0, "lower": 0.1, "upper": 20000.0, "min_width": 0.1},
    "rho": {"type": "relative", "frac": 1.0, "lower": 1.0e-7, "upper": 10.0, "min_width": 1.0e-7},
    "piEN": {"type": "center_width", "center": 0.0, "half_width": 1.9973},
    "piEE": {"type": "center_width", "center": 0.0, "half_width": 2.2010959999999997},
}


def bridge_fit(meta):
    """Exactly the frozen old_H0 bridge: 1 TRF, start=truth, old narrow bounds."""
    initial = {"t0": meta["truth"]["t0"], "u0": meta["truth"]["u0"],
               "tE": meta["truth"]["tE"], "rho": meta["truth"]["rho"]}
    fit, e, model = fit_lc.fit_rubin_roman(
        Source=meta["Source"], event_params=meta["true_params"], path_save="/tmp/prof_discard_audit_bridge",
        path_ephemerides=str(core.EPHEMERIDES), model="FSPL", algo="TRF", Origin=None, rango=meta["rango"],
        wfirst_lc=meta["curves"]["W149"], lsst_u=meta["curves"]["u"], lsst_g=meta["curves"]["g"],
        lsst_r=meta["curves"]["r"], lsst_i=meta["curves"]["i"], lsst_z=meta["curves"]["z"],
        lsst_y=meta["curves"]["y"], fit_model="FSPL", fit_parallax=False, fit_defaults=None,
        fit_bounds=OLD_H0_BOUNDS, random_state=20260909 + int(meta["row"]), initial_guess=initial,
        optimizer_options=OLD_OPTIMIZER_OPTIONS, event_ra=meta["event_ra"], event_dec=meta["event_dec"],
    )
    res = fit.fit_results
    best = np.asarray(res["best_model"], dtype=float)
    return {"t0": float(best[0]), "u0": float(best[1]), "tE": float(best[2]), "rho": float(best[3])}, \
        float(res["chi2"])


def audit_event(row, h5_path):
    meta = load_new_case(h5_path, core)
    truth = meta["truth"]

    existing_path = os.path.join(BRIDGE_JSON_DIR, f"{row}_bridge.json")
    bridge_wall_s = 0.0
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            data = json.load(f)
        bridge_vector = data["bridge_vector"]
        bridge_chi2 = data["bridge_chi2"]
        bridge_reused = True
    else:
        t0c = time.time()
        bridge_vector, bridge_chi2 = bridge_fit(meta)
        bridge_wall_s = time.time() - t0c
        bridge_reused = False

    # Resolve the exact old_final H1 bounds using the frozen fitter's OWN
    # function, centered on truth (fit_params == event truth in this
    # pipeline, per fit_lc.apply_custom_bounds -> _center_for_parameter).
    bounds = {}
    for par in ["t0", "u0", "tE", "rho", "piEN", "piEE"]:
        center = truth[par]
        bounds[par] = fit_lc.resolve_bound_spec(
            OLD_FINAL_H1_BOUND_SPECS[par], center=center, parameter_name=par
        )

    checks = {}
    offending = []
    for par in ["t0", "u0", "tE", "rho"]:
        lo, hi = bounds[par]
        val = bridge_vector[par]
        inside = (lo - 1e-12) <= val <= (hi + 1e-12)
        checks[par] = {
            "value": val, "lo": lo, "hi": hi, "inside": bool(inside),
            "distance_outside": (0.0 if inside else (lo - val if val < lo else val - hi)),
        }
        if not inside:
            offending.append(par)

    passed = len(offending) == 0

    rho_only = offending == ["rho"]
    te_only = offending == ["tE"]
    both = set(offending) >= {"tE", "rho"}
    other_invalid = any(p in offending for p in ["t0", "u0"])

    result = {
        "catalog_row": row,
        "truth_tE": truth["tE"], "truth_rho": truth["rho"],
        "bridge_tE": bridge_vector["tE"], "bridge_rho": bridge_vector["rho"],
        "bridge_chi2": bridge_chi2,
        "ratio_rho_bridge_over_truth": bridge_vector["rho"] / truth["rho"],
        "ratio_tE_bridge_over_truth": bridge_vector["tE"] / truth["tE"],
        "bound_tE_lo": bounds["tE"][0], "bound_tE_hi": bounds["tE"][1],
        "bound_rho_lo": bounds["rho"][0], "bound_rho_hi": bounds["rho"][1],
        "tE_inside": checks["tE"]["inside"], "rho_inside": checks["rho"]["inside"],
        "t0_inside": checks["t0"]["inside"], "u0_inside": checks["u0"]["inside"],
        "tE_distance_outside": checks["tE"]["distance_outside"],
        "rho_distance_outside": checks["rho"]["distance_outside"],
        "offending_params": offending,
        "failure_class": ("PASS" if passed else
                           "rho_only" if rho_only else
                           "tE_only" if te_only else
                           "both_tE_rho" if both else
                           "other" if other_invalid else "unclassified"),
        "all_old_final_starts_valid": passed,
        "PASS": passed,
        "bridge_reused": bridge_reused,
        "bridge_wall_s": bridge_wall_s,
    }
    return result


if __name__ == "__main__":
    manifest = pd.read_csv(args.rows_csv)

    rows = []
    for _, r in manifest.iterrows():
        row = int(r["catalog_row"])
        h5_path = r["h5_path"]
        res = audit_event(row, h5_path)
        rows.append(res)
        print(f"row={row} PASS={res['PASS']} class={res['failure_class']} "
              f"rho_ratio={res['ratio_rho_bridge_over_truth']:.4g} "
              f"tE_ratio={res['ratio_tE_bridge_over_truth']:.4g} "
              f"(bridge_reused={res['bridge_reused']})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    n = len(df)
    n_pass = int(df["PASS"].sum())
    n_fail = n - n_pass

    print()
    print("=" * 100)
    print(f"N={n} audited")
    print(f"executable (PASS): {n_pass}/{n} = {n_pass/n:.1%}")
    print(f"would crash before any old_final TRF (FAIL): {n_fail}/{n} = {n_fail/n:.1%}")
    print()
    print("failure_class breakdown:")
    print(df["failure_class"].value_counts())
    print()
    print("rho ratio (bridge/truth) distribution, PASS vs FAIL:")
    for grp, sub in df.groupby("PASS"):
        print(f"  PASS={grp}: n={len(sub)} "
              f"median={sub['ratio_rho_bridge_over_truth'].median():.4g} "
              f"min={sub['ratio_rho_bridge_over_truth'].min():.4g} "
              f"max={sub['ratio_rho_bridge_over_truth'].max():.4g}")
    print(f"-> {args.out}")
