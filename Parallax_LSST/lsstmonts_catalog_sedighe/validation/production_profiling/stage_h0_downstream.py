#!/usr/bin/env python3
"""
Stage 2 (per mode, per event): the H0 downstream strategies assigned
to this coordinate mode in the frozen 16-cost architecture (commit
7dadcbd, tag h0-morphology-freeze-2026-09-15), production_candidate
bounds, full convergence. Consumes the bridge vector (for old_H0-
anchored strategies) and the morphology candidate (for the morphology
strategy in this mode, if any) from stage_bridge_morphology.py's
output -- neither is recomputed.
"""
import sys
import os
import copy
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
BA = os.path.join(HERE, "..", "bounds_audit")
sys.path.insert(0, BA)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["physical", "log_te", "log_rho", "log_te_rho"])
ap.add_argument("--h5", required=True)
ap.add_argument("--bridge-json", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = args.mode
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = ["run_bounds_audit_refit_core.py", "--catalog-row", "71181",
            "--bounds-profile", "production_candidate", "--fit-scope", "h0", "--dry-run"]

import run_bounds_audit_refit_core as core  # noqa: E402
from load_new_event import load_new_case  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"

# exact 16-cost architecture (commit 7dadcbd), downstream strategies by mode
ARCHITECTURE = {
    "physical": {"truth_old_H0": [("old_H0", 1.0)], "truth": [("truth", 0.1)], "morph": False},
    "log_te": {"truth_old_H0": [("old_H0", 1.0), ("old_H0", "truth_rho")],
               "truth": [("truth", 0.01), ("truth", 1.0)], "morph": False},
    "log_rho": {"truth_old_H0": [("old_H0", 0.01), ("old_H0", 1.0), ("old_H0", "truth_rho")],
                "truth": [("truth", 0.0001), ("truth", 0.1), ("truth", 1.0), ("truth", "truth_rho")],
                "morph": True},
    "log_te_rho": {"truth_old_H0": [], "truth": [], "morph": True},
}

with open(args.bridge_json) as f:
    bridge_data = json.load(f)

meta = load_new_case(args.h5, core)

records = []
spec = ARCHITECTURE[args.mode]

for anchor, rho_rule in spec["truth_old_H0"]:
    src = bridge_data["bridge_vector"]
    rho = meta["truth"]["rho"] if rho_rule == "truth_rho" else float(rho_rule)
    initial = {"t0": src["t0"], "u0": src["u0"], "tE": src["tE"], "rho": rho}
    label = f"{args.mode}/old_H0/{rho_rule}"
    t0c, cpu0 = time.time(), time.process_time()
    r = core.run_one_fit(meta, "H0", initial, label)
    records.append({"label": label, "chi2": r.get("chi2"), "status": r.get("status"),
                     "wall_s": time.time() - t0c, "cpu_s": time.process_time() - cpu0,
                     "t0": r.get("t0"), "u0": r.get("u0"), "tE": r.get("tE"), "rho": r.get("rho")})

for anchor, rho_rule in spec["truth"]:
    rho = meta["truth"]["rho"] if rho_rule == "truth_rho" else float(rho_rule)
    initial = {"t0": meta["truth"]["t0"], "u0": meta["truth"]["u0"], "tE": meta["truth"]["tE"], "rho": rho}
    label = f"{args.mode}/truth/{rho_rule}"
    t0c, cpu0 = time.time(), time.process_time()
    r = core.run_one_fit(meta, "H0", initial, label)
    records.append({"label": label, "chi2": r.get("chi2"), "status": r.get("status"),
                     "wall_s": time.time() - t0c, "cpu_s": time.process_time() - cpu0,
                     "t0": r.get("t0"), "u0": r.get("u0"), "tE": r.get("tE"), "rho": r.get("rho")})

if spec["morph"] and bridge_data.get("morphology_candidate"):
    mc = bridge_data["morphology_candidate"]
    initial = {"t0": mc["t0"], "u0": mc["u0"], "tE": mc["tE"], "rho": mc["rho"]}
    label = f"morphology/{args.mode}"
    t0c, cpu0 = time.time(), time.process_time()
    r = core.run_one_fit(meta, "H0", initial, label)
    records.append({"label": label, "chi2": r.get("chi2"), "status": r.get("status"),
                     "wall_s": time.time() - t0c, "cpu_s": time.process_time() - cpu0,
                     "t0": r.get("t0"), "u0": r.get("u0"), "tE": r.get("tE"), "rho": r.get("rho")})

out = {"catalog_row": meta["row"], "mode": args.mode, "fits": records}
with open(args.out, "w") as f:
    json.dump(out, f, indent=2)

print(f"row={meta['row']} mode={args.mode} n_fits={len(records)} "
      f"total_wall={sum(x['wall_s'] for x in records):.2f}s -> {args.out}")
