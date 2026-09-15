#!/usr/bin/env python3
"""
M7: morphology-start TRF, FULL convergence (no budget cap), on the 12
old_H0-essential events only, one mode per process (mode is fixed at
core.py import time). production_candidate bounds, current
bounded_flux_profiling, default OPTIMIZER_OPTIONS (same as every other
full TRF fit in this log) -- no new parametrization, no reduced
budget.
"""
import sys
import os
import copy
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bounds_audit"))
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

assert core.BOUNDS_PROFILE == "production_candidate"
assert core.FIT_SCOPE == "h0"

CRITICAL_12 = [62786, 83189, 567800, 579320, 82728, 574423,
               557860, 558189, 567365, 35927, 563210, 79501]

cand = pd.read_csv(os.path.join(HERE, "results", "morphology_M3_M4_candidates.csv"))
c1 = cand[(cand["invertible"] == True) & (cand["candidate_rank"] == 1)
          & (cand["catalog_row"].isin(CRITICAL_12))].set_index("catalog_row")

records = []
for row in CRITICAL_12:
    record = {"sample": "extreme100", "catalog_row": int(row), "matched_tail_catalog_row": -1}
    meta = core.load_case(record)

    initial = {
        "t0": float(c1.loc[row, "t0_morph"]),
        "u0": float(c1.loc[row, "u0_morph"]),
        "tE": float(c1.loc[row, "tE_morph"]),
        "rho": float(c1.loc[row, "rho_morph"]),
    }

    t0c = time.time()
    r = core.run_one_fit(meta, "H0", initial, f"morphology_rank1_{args.mode}")
    dt = time.time() - t0c

    r["mode"] = args.mode
    r["catalog_row"] = int(row)
    r["wall_time_s"] = dt
    r["init_t0"] = initial["t0"]
    r["init_u0"] = initial["u0"]
    r["init_tE"] = initial["tE"]
    r["init_rho"] = initial["rho"]
    records.append(r)
    print(f"[{args.mode}] row={row} chi2={r.get('chi2')} status={r.get('status')} dt={dt:.2f}s", flush=True)

df = pd.DataFrame(records)
out_csv = os.path.join(HERE, "results", f"h0_morphology_trf_{args.mode}.csv")
df.to_csv(out_csv, index=False)
print(f"DONE [{args.mode}] -> {out_csv}")
