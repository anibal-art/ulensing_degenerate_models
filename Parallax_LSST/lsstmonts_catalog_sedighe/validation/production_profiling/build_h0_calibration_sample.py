#!/usr/bin/env python3
"""
Build the dedicated H0-generated calibration sample -- NOT the 1M
Sedighe population, NOT the profiling sample, NOT a development set.
Deterministic seed distinct from every other sample used in this
project. truth_parallax=False (genuine no-parallax simulation).
"""
import sys
import os
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=707070)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--out-manifest", default="results/h0_calibration_sample_frozen.csv")
ap.add_argument("--out-log", default="results/h0_calibration_sample_candidate_log.csv")
ap.add_argument("--out-dir", default="runtime_local/h0_calibration_h5")
args = ap.parse_args()

N_TOTAL_ROWS = 966000

dev_excluded = set(json.load(open("/tmp/dev_excluded_rows.json")))
h1_gen_sample = set(pd.read_csv(os.path.join(HERE, "results", "profiling_sample_frozen_100.csv"))
                     ["catalog_row"].tolist())
h0_smoke_sample = set(pd.read_csv(os.path.join(HERE, "results", "h0_generated_sample_frozen.csv"))
                       ["catalog_row"].tolist())
excluded = dev_excluded | h1_gen_sample | h0_smoke_sample

rng = np.random.default_rng(args.seed)
full_order = rng.permutation(N_TOTAL_ROWS)
candidate_order = [int(c) for c in full_order if c not in excluded]

from standalone_materialize import materialize_event  # noqa: E402

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
    r = materialize_event(candidate_row, out_dir, timing=timing, generating_model="H0")
    dt = time.time() - t0
    passed = r["status"] == "materialized"
    log_rows.append({"candidate_row": candidate_row, "status": r["status"], "passed": passed,
                      "wall_time_s": dt})
    if passed:
        frozen.append({"catalog_row": candidate_row, "global_i": r["global_i"], "h5_path": r["h5_path"]})
    if len(log_rows) % 10 == 0 or passed:
        print(f"[{len(log_rows)}] row={candidate_row} status={r['status']} "
              f"n_frozen={len(frozen)}/{args.n} dt={dt:.2f}s total_elapsed={time.time()-t_start:.0f}s",
              flush=True)

pd.DataFrame(log_rows).to_csv(os.path.join(HERE, args.out_log), index=False)
frozen_df = pd.DataFrame(frozen)
frozen_df.to_csv(os.path.join(HERE, args.out_manifest), index=False)

print()
print("=" * 100)
print(f"FROZEN: {len(frozen)} H0-CALIBRATION events, seed={args.seed}, "
      f"candidates evaluated={len(log_rows)}, pass rate={len(frozen)/len(log_rows):.2%}")
print("total wall time:", time.time() - t_start, "s")
