#!/usr/bin/env python3
"""
NEW independent H1-generated basin-validation sample. Zero overlap with
development, extreme100, the 25-event profiling sample, or any
H0-generated sample. Deterministic seed distinct from every other
sample built in this project. Real catalog piE (truth_parallax=True,
generating_model="H1", same as the profiling sample).
"""
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=838383)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out-manifest", default="results/h1_basin_validation_sample_frozen.csv")
ap.add_argument("--out-log", default="results/h1_basin_validation_candidate_log.csv")
ap.add_argument("--out-dir", default="runtime_local/h1_basin_validation_h5")
args = ap.parse_args()

N_TOTAL_ROWS = 966000
dev_excluded = set(json.load(open("/tmp/dev_excluded_rows.json")))
h1_profiling = set(pd.read_csv(os.path.join(HERE, "results", "profiling_sample_frozen_100.csv"))["catalog_row"].tolist())
h0_smoke = set(pd.read_csv(os.path.join(HERE, "results", "h0_generated_sample_frozen.csv"))["catalog_row"].tolist())
h0_calib = set(pd.read_csv(os.path.join(HERE, "results", "h0_calibration_sample_frozen.csv"))["catalog_row"].tolist())
excluded = dev_excluded | h1_profiling | h0_smoke | h0_calib

rng = np.random.default_rng(args.seed)
full_order = rng.permutation(N_TOTAL_ROWS)
candidate_order = [int(c) for c in full_order if c not in excluded]

from standalone_materialize import materialize_event

out_dir = os.path.join(HERE, args.out_dir)
os.makedirs(out_dir, exist_ok=True)

log_rows = []
frozen = []
t_start = time.time()

for candidate_row in candidate_order:
    if len(frozen) >= args.n:
        break
    timing = {}
    t0 = time.time()
    r = materialize_event(candidate_row, out_dir, timing=timing, generating_model="H1")
    dt = time.time() - t0
    passed = r["status"] == "materialized"
    log_rows.append({"candidate_row": candidate_row, "status": r["status"], "passed": passed, "wall_time_s": dt})
    if passed:
        frozen.append({"catalog_row": candidate_row, "global_i": r["global_i"], "h5_path": r["h5_path"]})
    if len(log_rows) % 10 == 0 or passed:
        print(f"[{len(log_rows)}] row={candidate_row} status={r['status']} n_frozen={len(frozen)}/{args.n} "
              f"dt={dt:.2f}s total_elapsed={time.time()-t_start:.0f}s", flush=True)
    # periodic incremental save so a kill doesn't lose progress
    if len(log_rows) % 20 == 0:
        pd.DataFrame(log_rows).to_csv(os.path.join(HERE, args.out_log), index=False)
        pd.DataFrame(frozen).to_csv(os.path.join(HERE, args.out_manifest), index=False)

pd.DataFrame(log_rows).to_csv(os.path.join(HERE, args.out_log), index=False)
pd.DataFrame(frozen).to_csv(os.path.join(HERE, args.out_manifest), index=False)
print()
print("=" * 100)
print(f"FROZEN: {len(frozen)} H1-BASIN-VALIDATION events, seed={args.seed}, "
      f"candidates evaluated={len(log_rows)}, pass rate={len(frozen)/len(log_rows):.2%}")
print("total wall time:", time.time() - t_start, "s")
