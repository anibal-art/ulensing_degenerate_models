#!/usr/bin/env python3
"""
Rerun a frozen sample under: 2 nominal fits (fit_two_policy) + the two
CONDITIONAL numerical safeguards (numerical_safeguards.apply_safeguards)
-- same-point continuation on poor optimality, nested-H1 rescue on
chi2_H1 > chi2_H0 + epsilon. Not unconditional multistart: each
safeguard fires only when its explicit diagnostic condition is met.
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
from numerical_safeguards import apply_safeguards  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"

manifest = pd.read_csv(args.rows_csv)

results = []
batch_t0 = time.time()
for _, row_entry in manifest.iterrows():
    h5_path = row_entry["h5_path"]
    meta = load_new_case(h5_path, core)

    event_wall0, event_cpu0 = time.time(), time.process_time()
    nominal = run_two_fit_policy(meta, core, fit_lc)
    audit = apply_safeguards(core, fit_lc, meta, nominal)
    audit["event_wall_s"] = time.time() - event_wall0
    audit["event_cpu_s"] = time.process_time() - event_cpu0
    audit["morphology_seed"] = nominal["morphology_seed"]

    results.append(audit)
    print(f"row={meta['row']} gen={meta['generating_model']} "
          f"n_total_trf={audit['n_total_trf']} "
          f"(nominal={audit['n_nominal_trf']}+extra={audit['n_extra_trf']}: "
          f"cont_h0={audit['continuation_ran_h0']} cont_h1={audit['continuation_ran_h1']} "
          f"rescue={audit['nested_rescue_ran']}) "
          f"chi2_h0={audit.get('chi2_h0_final')} chi2_h1={audit.get('chi2_h1_final')} "
          f"delta_lrt={audit.get('delta_chi2_lrt_final')} "
          f"wall={audit['event_wall_s']:.2f}s", flush=True)

batch_wall_s = time.time() - batch_t0

with open(args.out, "w") as f:
    json.dump({"batch_wall_s": batch_wall_s, "n_events": len(results), "events": results},
               f, indent=2, default=str)

print()
print("=" * 100)
print(f"TWO-FIT + SAFEGUARDS BATCH: {len(results)} events, batch_wall_s={batch_wall_s:.1f}s -> {args.out}")
