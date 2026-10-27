#!/usr/bin/env python3
"""
Run the simplified 2-fit architecture (fit_two_policy.run_two_fit_policy)
over a batch of events (one process, one coordinate mode -- physical --
production_candidate bounds throughout). Used both for the small
independent validation comparison and for the runtime profile.
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

import pandas as pd  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402
from load_new_event import load_new_case  # noqa: E402
from fit_two_policy import run_two_fit_policy  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"

manifest = pd.read_csv(args.rows_csv)

results = []
batch_t0 = time.time()
for _, row_entry in manifest.iterrows():
    h5_path = row_entry["h5_path"]
    meta = load_new_case(h5_path, core)

    event_wall0, event_cpu0 = time.time(), time.process_time()
    r = run_two_fit_policy(meta, core, fit_lc)
    r["event_wall_s"] = time.time() - event_wall0
    r["event_cpu_s"] = time.process_time() - event_cpu0

    results.append(r)
    print(f"row={meta['row']} gen={meta['generating_model']} n_trf={r['n_trf']} "
          f"chi2_h0={r.get('chi2_h0')} chi2_h1={r.get('chi2_h1')} "
          f"delta_chi2_lrt={r.get('delta_chi2_lrt')} "
          f"wall={r['event_wall_s']:.2f}s cpu={r['event_cpu_s']:.2f}s", flush=True)

batch_wall_s = time.time() - batch_t0

with open(args.out, "w") as f:
    json.dump({"batch_wall_s": batch_wall_s, "n_events": len(results), "events": results},
               f, indent=2, default=str)

print()
print("=" * 100)
print(f"TWO-FIT BATCH: {len(results)} events, batch_wall_s={batch_wall_s:.1f}s -> {args.out}")
