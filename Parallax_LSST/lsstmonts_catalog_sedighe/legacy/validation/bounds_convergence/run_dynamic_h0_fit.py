#!/usr/bin/env python3
"""
Standalone single-fit runner for the H0 self-seeding decisive test.

Imports run_bounds_audit_refit_core.py (the FROZEN scientific fitter,
unmodified) and calls its own run_one_fit() directly for one custom
(mode, label, initial) H0 strategy on one extreme100 event.

Coordinate mode (HIDDEN_PARALLAX_TRF_COORDS) is baked in at import time
by the core module, so this script must be invoked once per (event,
strategy) as a fresh subprocess with the correct env var already set.
"""
import sys
import os
import json
import argparse

CORE_DIR = (
    "/home/anibal-pc/ulensing_degenerate_models/"
    "Parallax_LSST/lsstmonts_catalog_sedighe/validation/bounds_audit"
)
sys.path.insert(0, CORE_DIR)

ap = argparse.ArgumentParser()
ap.add_argument("--catalog-row", type=int, required=True)
ap.add_argument("--label", type=str, required=True)
ap.add_argument("--t0", type=float, required=True)
ap.add_argument("--u0", type=float, required=True)
ap.add_argument("--tE", type=float, required=True)
ap.add_argument("--rho", type=float, required=True)
ap.add_argument("--dummy-import-row", type=int, default=35101)
args = ap.parse_args()

# Fake argv so the core module's own module-level argparse call
# succeeds harmlessly at import time (dry-run, no real fit executed
# for the dummy row).
sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row", str(args.dummy_import_row),
    "--bounds-profile", "te500000",
    "--fit-scope", "h0",
    "--dry-run",
]

import run_bounds_audit_refit_core as core  # noqa: E402

assert core.BOUNDS_PROFILE == "te500000"
assert core.FIT_SCOPE == "h0"

record = {
    "sample": "extreme100",
    "catalog_row": args.catalog_row,
    "matched_tail_catalog_row": -1,
}

meta = core.load_case(record)

initial = {
    "t0": args.t0,
    "u0": args.u0,
    "tE": args.tE,
    "rho": args.rho,
}

result = core.run_one_fit(meta, "H0", initial, args.label)

# Only JSON-serializable fields.
out = {
    k: (
        v.tolist() if hasattr(v, "tolist") else v
    )
    for k, v in result.items()
}
out["trf_coords"] = os.environ.get("HIDDEN_PARALLAX_TRF_COORDS", "physical")
out["catalog_row"] = args.catalog_row

print("RESULT_JSON " + json.dumps(out))
