#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
PP = HERE.parent
BA = PP.parent / "bounds_audit"

sys.path.insert(0, str(BA))
sys.path.insert(0, str(PP))


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()


# ============================================================
# EXACT H0 NUMERICAL STATE USED BY TWO_FIT_V2
# ============================================================

# H1-generated -> H0 shared-truth fit was performed in
# log_te_rho coordinates.
os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "log_te_rho"

# Required by the validated coordinate implementation.
# The effective optimizer options remain x_scale="jac".
os.environ["HIDDEN_PARALLAX_TRF_X_SCALE"] = "pylima"

os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


core_manifest = (
    BA / "data" / "refit_manifest.csv"
).resolve()

if not core_manifest.is_file():
    raise FileNotFoundError(core_manifest)


original_argv = list(sys.argv)

sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row",
    "71181",
    "--manifest",
    str(core_manifest),
    "--bounds-profile",
    "production_candidate",
    "--fit-scope",
    "h0",
    "--dry-run",
]

try:
    import run_bounds_audit_refit_core as core
finally:
    sys.argv = original_argv

from load_new_event import load_new_case


if core.BOUNDS_PROFILE != "production_candidate":
    raise RuntimeError(core.BOUNDS_PROFILE)

if core.OPTIMIZER_OPTIONS.get("x_scale") != "jac":
    raise RuntimeError(
        "effective optimizer x_scale is not jac: "
        f"{core.OPTIMIZER_OPTIONS.get('x_scale')!r}"
    )


df = pd.read_csv(args.manifest)

records = []

for _, row in df.iterrows():

    catalog_row = int(row["catalog_row"])

    meta = load_new_case(
        row["h5_path"],
        core,
    )

    truth = meta["truth"]

    initial = {
        "t0": float(truth["t0"]),
        "u0": float(truth["u0"]),
        "tE": float(truth["tE"]),
        "rho": float(truth["rho"]),
    }

    result = core.run_one_fit(
        meta,
        "H0",
        initial,
        "production_shared_truth_log_te_rho",
    )

    chi2_rerun = float(result["chi2"])
    chi2_stored = float(row["chi2_h0_fit"])

    delta = (
        chi2_rerun
        - chi2_stored
    )

    rec = {
        "catalog_row": catalog_row,
        "chi2_h0_stored": chi2_stored,
        "chi2_h0_rerun": chi2_rerun,
        "rerun_minus_stored": delta,
        "abs_difference": abs(delta),

        "stored_T": float(row["T_fit"]),

        "rerun_t0": result.get("t0"),
        "rerun_u0": result.get("u0"),
        "rerun_tE": result.get("tE"),
        "rerun_rho": result.get("rho"),

        "status": result.get("status"),
    }

    records.append(rec)

    print(
        f"row={catalog_row} "
        f"stored={chi2_stored:.12f} "
        f"rerun={chi2_rerun:.12f} "
        f"delta={delta:+.6e}",
        flush=True,
    )


out = pd.DataFrame(records)

out.to_csv(
    args.out,
    index=False,
)

print()
print(out.to_string(index=False))
print()

print(
    "max_abs_difference =",
    out["abs_difference"].max(),
)
