#!/usr/bin/env python3
"""
Build a small frozen sample of H0-GENERATED events (truth_parallax=
False -- a genuine no-parallax simulation, parallax=["None",0.0] in
the simulator, not a truth override). Separate deterministic seed
from the H1-generated profiling sample (424242) so the two samples
never overlap.

Needed for the H0-generated side of the new 2-fit architecture's
4-combination smoke test and small validation comparison. Small on
purpose (N=5): this is a validation/smoke sample, not a production or
calibration sample.
"""
import sys
import os
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

H0_GEN_SEED = 636363
TARGET_N = 5
N_TOTAL_ROWS = 966000

dev_excluded = set(json.load(open("/tmp/dev_excluded_rows.json")))
h1_gen_sample = set(pd.read_csv(os.path.join(HERE, "results", "profiling_sample_frozen_100.csv"))
                     ["catalog_row"].tolist())
excluded = dev_excluded | h1_gen_sample

rng = np.random.default_rng(H0_GEN_SEED)
full_order = rng.permutation(N_TOTAL_ROWS)
candidate_order = [int(c) for c in full_order if c not in excluded]

from standalone_materialize import materialize_event  # noqa: E402

out_dir = os.path.join(HERE, "runtime_local", "h0_generated_h5")
os.makedirs(out_dir, exist_ok=True)

log_rows = []
frozen = []
t_start = time.time()

for candidate_row in candidate_order:
    if len(frozen) >= TARGET_N:
        break

    timing = {}
    t0 = time.time()
    r = materialize_event(candidate_row, out_dir, timing=timing, generating_model="H0")
    dt = time.time() - t0

    passed = r["status"] == "materialized"
    log_rows.append({"candidate_row": candidate_row, "status": r["status"], "passed": passed,
                      "wall_time_s": dt, **{f"stage_{k}": v for k, v in timing.items()}})
    if passed:
        frozen.append({"catalog_row": candidate_row, "global_i": r["global_i"], "h5_path": r["h5_path"]})

    print(f"[{len(log_rows)}] row={candidate_row} status={r['status']} "
          f"n_frozen={len(frozen)}/{TARGET_N} dt={dt:.2f}s total_elapsed={time.time()-t_start:.0f}s",
          flush=True)

log_df = pd.DataFrame(log_rows)
log_df.to_csv(os.path.join(HERE, "results", "h0_generated_sample_candidate_log.csv"), index=False)

frozen_df = pd.DataFrame(frozen)
frozen_df.to_csv(os.path.join(HERE, "results", "h0_generated_sample_frozen.csv"), index=False)

print()
print("=" * 100)
print(f"FROZEN: {len(frozen)} H0-generated events, seed={H0_GEN_SEED}, "
      f"candidates evaluated={len(log_rows)}, pass rate={len(frozen)/len(log_rows):.2%}")
print("frozen catalog_row list:", [f["catalog_row"] for f in frozen])
print("total wall time:", time.time() - t_start, "s")
