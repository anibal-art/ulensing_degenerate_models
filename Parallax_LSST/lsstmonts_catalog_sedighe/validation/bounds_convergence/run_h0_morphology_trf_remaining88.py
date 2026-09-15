#!/usr/bin/env python3
"""
Extension of M7 to the remaining 88 extreme100 events (the 12 critical
events were already done in run_h0_morphology_trf.py / M7). Exactly
the same frozen procedure: rank-1 morphology candidate, full
convergence (no budget cap), production_candidate bounds, current
bounded_flux_profile, one process per coordinate mode (mode fixed at
core.py import time). No new morphology definition of any kind.
"""
import sys
import os
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

truth_feat = pd.read_csv(os.path.join(HERE, "results", "b1c_event_features.csv"))
all_100 = sorted(truth_feat["catalog_row"].unique().tolist())
assert len(all_100) == 100
REMAINING_88 = [r for r in all_100 if r not in CRITICAL_12]
assert len(REMAINING_88) == 88

cand = pd.read_csv(os.path.join(HERE, "results", "morphology_M3_M4_candidates.csv"))
c1 = cand[(cand["invertible"] == True) & (cand["candidate_rank"] == 1)
          & (cand["catalog_row"].isin(REMAINING_88))].set_index("catalog_row")
assert len(c1) == 88, f"expected all 88 to be invertible, got {len(c1)}"

records = []
t_start = time.time()
for row in REMAINING_88:
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
out_csv = os.path.join(HERE, "results", f"h0_morphology_trf_remaining88_{args.mode}.csv")
df.to_csv(out_csv, index=False)
print(f"DONE [{args.mode}] -> {out_csv} n={len(df)} total_wall={time.time()-t_start:.1f}s")
