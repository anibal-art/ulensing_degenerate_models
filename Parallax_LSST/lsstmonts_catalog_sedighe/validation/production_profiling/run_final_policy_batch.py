#!/usr/bin/env python3
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
BA = os.path.join(HERE, "..", "bounds_audit")
BC = os.path.join(HERE, "..", "bounds_convergence")
sys.path.insert(0, BA); sys.path.insert(0, BC); sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--rows-csv", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = ["run_bounds_audit_refit_core.py", "--catalog-row", "71181",
            "--bounds-profile", "production_candidate", "--fit-scope", "h0", "--dry-run"]

import pandas as pd
import run_bounds_audit_refit_core as core
import fit_lc
from load_new_event import load_new_case
from final_policy import run_final_policy

assert core.BOUNDS_PROFILE == "production_candidate"
manifest = pd.read_csv(args.rows_csv)

results = []
batch_t0 = time.time()
for _, row_entry in manifest.iterrows():
    h5_path = row_entry["h5_path"]
    meta = load_new_case(h5_path, core)
    ew0, ec0 = time.time(), time.process_time()
    r = run_final_policy(meta, core, fit_lc)
    r["event_wall_s"] = time.time() - ew0
    r["event_cpu_s"] = time.process_time() - ec0
    results.append(r)
    print(f"row={meta['row']} gen={meta['generating_model']} n_trf={r['n_total_trf']} "
          f"rescue={r['rescue_ran']} chi2_h0={r.get('chi2_h0')} chi2_h1={r.get('chi2_h1')} "
          f"delta_lrt={r.get('delta_chi2_lrt')} sanity_flags={r['sanity_flags']} "
          f"wall={r['event_wall_s']:.2f}s", flush=True)

batch_wall_s = time.time() - batch_t0
with open(args.out, "w") as f:
    json.dump({"batch_wall_s": batch_wall_s, "n_events": len(results), "events": results},
               f, indent=2, default=str)
print(f"DONE: {len(results)} events, batch_wall_s={batch_wall_s:.1f}s -> {args.out}")
