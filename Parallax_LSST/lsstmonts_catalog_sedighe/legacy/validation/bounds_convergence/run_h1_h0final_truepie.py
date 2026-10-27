#!/usr/bin/env python3
"""
H1-A: H0_FINAL_TRUE_PIE candidate -- 5 decisive events only.

Initial vector: (t0,u0,tE,rho) = the final robust H0 solution already
available under the current architecture (gate2_h0_anchor_manifest,
the same reference controlled5's own H0_NESTED_piE_0 start uses),
piEN,piEE = truth. This is initialization only -- all 6 H1 nonlinear
parameters are free during the fit. Same production_candidate H1
bounds, current bounded_flux_profile, same (unbudgeted) optimizer
options as every other H1 fit in this log, physical coordinate mode
(matching controlled4/controlled5's own precedent -- H1 was never
run multi-mode in this project).
"""
import sys
import os
import copy
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bounds_audit"))
sys.path.insert(0, HERE)

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row", "71181",
    "--bounds-profile", "production_candidate",
    "--fit-scope", "h1",
    "--dry-run",
]

import pandas as pd  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"
assert core.FIT_SCOPE == "h1"

FIVE_EVENTS = [85380, 87786, 79700, 50179, 86451]

h0_final = pd.read_csv(os.path.join(HERE, "data", "gate2_h0_anchor_manifest_extreme100.csv")).set_index("catalog_row")
truth = pd.read_csv(os.path.join(HERE, "results", "b1c_event_features.csv")).set_index("catalog_row")

records = []
for row in FIVE_EVENTS:
    meta = core.load_case({"sample": "extreme100", "catalog_row": int(row), "matched_tail_catalog_row": -1})

    initial = {
        "t0": float(h0_final.loc[row, "t0"]),
        "u0": float(h0_final.loc[row, "u0"]),
        "tE": float(h0_final.loc[row, "tE"]),
        "rho": float(h0_final.loc[row, "rho"]),
        "piEN": float(truth.loc[row, "piEN_true"]),
        "piEE": float(truth.loc[row, "piEE_true"]),
    }

    t0c = time.time()
    r = core.run_one_fit(meta, "H1", initial, "H0_FINAL_TRUE_PIE")
    dt = time.time() - t0c

    r["catalog_row"] = int(row)
    r["wall_time_s"] = dt
    for k, v in initial.items():
        r[f"init_{k}"] = v
    records.append(r)
    print(f"row={row} chi2={r.get('chi2')} status={r.get('status')} "
          f"final=(t0={r.get('t0')},u0={r.get('u0')},tE={r.get('tE')},rho={r.get('rho')},"
          f"piEN={r.get('piEN')},piEE={r.get('piEE')}) "
          f"active_mask={r.get('optimizer_active_mask')} dt={dt:.2f}s", flush=True)

df = pd.DataFrame(records)
out_csv = os.path.join(HERE, "results", "h1_H0_FINAL_TRUE_PIE_5events.csv")
df.to_csv(out_csv, index=False)
print("DONE ->", out_csv)
