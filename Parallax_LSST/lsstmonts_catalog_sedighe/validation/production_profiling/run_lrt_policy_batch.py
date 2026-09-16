#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BA = os.path.join(HERE, "..", "bounds_audit")
BC = os.path.join(HERE, "..", "bounds_convergence")

sys.path.insert(0, BA)
sys.path.insert(0, BC)
sys.path.insert(0, HERE)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--rows-csv",
    required=True,
)

parser.add_argument(
    "--out",
    required=True,
)

parser.add_argument(
    "--limit",
    type=int,
    default=None,
)

args = parser.parse_args()

# ------------------------------------------------------------
# Frozen production choices
# ------------------------------------------------------------
os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_TRF_X_SCALE"] = "pylima"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"

# Import the fitting core under the wide production-candidate bounds.
sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row",
    "71181",
    "--bounds-profile",
    "production_candidate",
    "--fit-scope",
    "h0",
    "--dry-run",
]

import pandas as pd
import run_bounds_audit_refit_core as core
import fit_lc

from load_new_event import load_new_case
from lrt_fit_policy import (
    POLICY_NAME,
    run_lrt_fit_policy,
)

assert core.BOUNDS_PROFILE == "production_candidate"

manifest = pd.read_csv(
    args.rows_csv
)

if args.limit is not None:
    if args.limit <= 0:
        raise ValueError(
            "--limit must be positive"
        )

    manifest = manifest.head(
        args.limit
    ).copy()

results = []

batch_wall0 = time.time()
batch_cpu0 = time.process_time()

for _, row_entry in manifest.iterrows():

    h5_path = row_entry["h5_path"]

    meta = load_new_case(
        h5_path,
        core,
    )

    event_wall0 = time.time()
    event_cpu0 = time.process_time()

    result = run_lrt_fit_policy(
        meta,
        core,
        fit_lc,
    )

    result["event_wall_s"] = (
        time.time() - event_wall0
    )

    result["event_cpu_s"] = (
        time.process_time() - event_cpu0
    )

    results.append(
        result
    )

    print(
        f"row={meta['row']} "
        f"gen={meta['generating_model']} "
        f"nominal={result['n_nominal_trf']} "
        f"continuations={result['n_continuation_trf']} "
        f"rescue={result['n_rescue_trf']} "
        f"total={result['n_total_trf']} "
        f"h0_winner={result.get('h0_winner')} "
        f"chi2_h0={result.get('chi2_h0')} "
        f"chi2_h1={result.get('chi2_h1')} "
        f"D={result.get('delta_chi2_lrt')} "
        f"nesting_ok={result.get('final_nesting_ok')} "
        f"wall={result['event_wall_s']:.2f}s",
        flush=True,
    )

payload = {
    "policy_name": POLICY_NAME,
    "bounds_profile": core.BOUNDS_PROFILE,
    "coordinates": os.environ[
        "HIDDEN_PARALLAX_TRF_COORDS"
    ],
    "x_scale": os.environ[
        "HIDDEN_PARALLAX_TRF_X_SCALE"
    ],
    "t0_margin_factor": os.environ[
        "HIDDEN_PARALLAX_T0_MARGIN_FACTOR"
    ],
    "batch_wall_s": (
        time.time() - batch_wall0
    ),
    "batch_cpu_s": (
        time.process_time() - batch_cpu0
    ),
    "n_events": len(results),
    "events": results,
}

with open(
    args.out,
    "w",
) as handle:
    json.dump(
        payload,
        handle,
        indent=2,
        default=str,
    )

print(
    f"DONE "
    f"policy={POLICY_NAME} "
    f"events={len(results)} "
    f"batch_wall_s={payload['batch_wall_s']:.1f}s "
    f"-> {args.out}"
)
