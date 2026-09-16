#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
import traceback

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
# Frozen production-candidate numerical choices
# ------------------------------------------------------------
os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_TRF_X_SCALE"] = "jac"
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

out_path = os.path.abspath(args.out)
out_dir = os.path.dirname(out_path)

if out_dir:
    os.makedirs(out_dir, exist_ok=True)

results = []

batch_wall0 = time.time()
batch_cpu0 = time.process_time()


def build_payload():
    n_success = sum(
        r.get("status") == "success"
        for r in results
    )

    n_failed = sum(
        r.get("status") == "failure"
        for r in results
    )

    return {
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
        "n_requested_events": int(len(manifest)),
        "n_processed_events": int(len(results)),
        "n_success": int(n_success),
        "n_failed": int(n_failed),
        "events": results,
    }


def write_checkpoint():
    payload = build_payload()

    tmp_path = out_path + ".tmp"

    with open(
        tmp_path,
        "w",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            default=str,
        )

    os.replace(
        tmp_path,
        out_path,
    )

    return payload


for _, row_entry in manifest.iterrows():

    h5_path = row_entry["h5_path"]

    fallback_row = row_entry.get(
        "catalog_row",
        None,
    )

    event_wall0 = time.time()
    event_cpu0 = time.process_time()

    try:
        meta = load_new_case(
            h5_path,
            core,
        )

        result = run_lrt_fit_policy(
            meta,
            core,
            fit_lc,
        )

        result["status"] = "success"
        result["h5_path"] = h5_path
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
            f"status=success "
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

    except Exception as exc:

        failure = {
            "status": "failure",
            "catalog_row": (
                None
                if pd.isna(fallback_row)
                else fallback_row
            ),
            "h5_path": h5_path,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "event_wall_s": (
                time.time() - event_wall0
            ),
            "event_cpu_s": (
                time.process_time() - event_cpu0
            ),
        }

        results.append(
            failure
        )

        print(
            f"row={fallback_row} "
            f"status=failure "
            f"exception={type(exc).__name__}: {exc}",
            flush=True,
        )

    # Persist every completed event so that a later interruption
    # cannot erase already-finished work.
    write_checkpoint()


payload = write_checkpoint()

print(
    f"DONE "
    f"policy={POLICY_NAME} "
    f"processed={payload['n_processed_events']} "
    f"success={payload['n_success']} "
    f"failed={payload['n_failed']} "
    f"batch_wall_s={payload['batch_wall_s']:.1f}s "
    f"-> {out_path}"
)
