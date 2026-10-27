#!/usr/bin/env python3
"""
Independent H1 basin-reference search for H0-generated events.

Purpose
-------
Validate the production H0-generated branch:

    H0 truth start
      ->
    H1 nested start from settled H0 with piEN=piEE=0

against a deliberately broader, truth-blind H1 multistart search.

This script is validation machinery ONLY. It does not modify the
production LRT policy.

Reference starts
----------------
Shared H1 parameters (t0, u0, tE, rho) are initialized from the settled
production-policy H0 solution.

The H1 reference explores:
    - both signs of u0
    - piE = 0
    - |piE| = 0.03, 0.3, 3, 30
    - four cardinal piE directions

Total:
    2 * (1 + 4 * 4) = 34 H1 TRF starts/event.

All six H1 nonlinear parameters remain free in every fit.
"""

# ============================================================
# NEW BLOCK: standard imports
# ============================================================

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# NEW BLOCK: CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--rows-csv",
    required=True,
    help="Frozen H0-generated validation manifest.",
)

parser.add_argument(
    "--candidate-json",
    required=True,
    help="Output of run_lrt_policy_batch.py on the same events.",
)

parser.add_argument(
    "--out",
    required=True,
    help="Output JSON for the independent H1 reference.",
)

parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Optional number of manifest rows to process.",
)

args = parser.parse_args()


# ============================================================
# NEW BLOCK: repository paths
# ============================================================

ROOT = Path.cwd().resolve()

PP = (
    ROOT
    / "validation"
    / "production_profiling"
)

BA = (
    ROOT
    / "validation"
    / "bounds_audit"
)

BC = (
    ROOT
    / "validation"
    / "bounds_convergence"
)

for path in (PP, BA, BC):
    if not path.exists():
        raise RuntimeError(
            f"Expected repository path does not exist: {path}"
        )

sys.path.insert(0, str(BA))
sys.path.insert(0, str(BC))
sys.path.insert(0, str(PP))


# ============================================================
# NEW BLOCK: frozen numerical policy environment
# ============================================================

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_TRF_X_SCALE"] = "jac"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"


# run_bounds_audit_refit_core parses argv at import time.
# Supply a harmless valid import-time configuration.
sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row",
    "71181",
    "--manifest",
    str(
        BA
        / "data"
        / "refit_manifest.csv"
    ),
    "--bounds-profile",
    "production_candidate",
    "--fit-scope",
    "h1",
    "--dry-run",
]


# ============================================================
# NEW BLOCK: project fitting imports
# ============================================================

import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402

from load_new_event import load_new_case  # noqa: E402
from run_one_fit_full import run_one_fit_full  # noqa: E402


if core.BOUNDS_PROFILE != "production_candidate":
    raise RuntimeError(
        "Reference is not using production_candidate bounds."
    )


# ============================================================
# NEW BLOCK: deterministic reference grid
# ============================================================

RADII = (
    0.03,
    0.3,
    3.0,
    30.0,
)

ANGLES_DEG = (
    0.0,
    90.0,
    180.0,
    270.0,
)


def build_reference_starts(final_h0):
    """
    Construct deterministic, truth-blind H1 starts.

    Shared parameters come exclusively from the settled H0 fit
    produced by the candidate production policy.
    """

    t0 = float(final_h0["t0"])
    u0 = float(final_h0["u0"])
    tE = float(final_h0["tE"])
    rho = float(final_h0["rho"])

    starts = []

    u0_options = (
        ("u0_same", u0),
        ("u0_mirror", -u0),
    )

    for sign_label, seeded_u0 in u0_options:

        # Exact nested H1 point.
        starts.append(
            (
                f"{sign_label}_piE_0",
                {
                    "t0": t0,
                    "u0": seeded_u0,
                    "tE": tE,
                    "rho": rho,
                    "piEN": 0.0,
                    "piEE": 0.0,
                },
            )
        )

        # Non-zero deterministic parallax starts.
        for radius in RADII:

            for angle_deg in ANGLES_DEG:

                theta = math.radians(
                    angle_deg
                )

                piEN = (
                    radius
                    * math.cos(theta)
                )

                piEE = (
                    radius
                    * math.sin(theta)
                )

                label = (
                    f"{sign_label}"
                    f"_r{radius:g}"
                    f"_a{angle_deg:g}"
                )

                starts.append(
                    (
                        label,
                        {
                            "t0": t0,
                            "u0": seeded_u0,
                            "tE": tE,
                            "rho": rho,
                            "piEN": piEN,
                            "piEE": piEE,
                        },
                    )
                )

    if len(starts) != 34:
        raise RuntimeError(
            f"Expected 34 starts, got {len(starts)}"
        )

    return starts


# ============================================================
# NEW BLOCK: compact reference fit record
# ============================================================

def compact_record(
    label,
    initial,
    fit_record,
    wall_s,
):
    return {
        "label": label,

        "initial": {
            key: float(value)
            for key, value
            in initial.items()
        },

        "chi2": float(
            fit_record["chi2"]
        ),

        "status": fit_record.get(
            "status"
        ),

        "optimizer_success": fit_record.get(
            "optimizer_success"
        ),

        "optimizer_status": fit_record.get(
            "optimizer_status"
        ),

        "t0": float(
            fit_record["t0"]
        ),

        "u0": float(
            fit_record["u0"]
        ),

        "tE": float(
            fit_record["tE"]
        ),

        "rho": float(
            fit_record["rho"]
        ),

        "piEN": float(
            fit_record["piEN"]
        ),

        "piEE": float(
            fit_record["piEE"]
        ),

        "wall_s": float(
            wall_s
        ),
    }


# ============================================================
# NEW BLOCK: atomic output helper
# ============================================================

def write_payload_atomic(
    out_path,
    payload,
):
    out_path = Path(out_path)

    tmp_path = Path(
        str(out_path) + ".tmp"
    )

    tmp_path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        )
        + "\n"
    )

    os.replace(
        tmp_path,
        out_path,
    )


# ============================================================
# NEW BLOCK: load frozen inputs
# ============================================================

manifest = pd.read_csv(
    args.rows_csv
)

required_manifest_columns = {
    "catalog_row",
    "h5_path",
}

missing = (
    required_manifest_columns
    - set(manifest.columns)
)

if missing:
    raise RuntimeError(
        f"Manifest missing columns: "
        f"{sorted(missing)}"
    )

if args.limit is not None:
    manifest = (
        manifest
        .head(args.limit)
        .copy()
    )


candidate_payload = json.loads(
    Path(
        args.candidate_json
    ).read_text()
)

candidate_events = (
    candidate_payload["events"]
)

candidate_by_row = {
    int(event["catalog_row"]):
    event
    for event
    in candidate_events
    if event.get("status") == "success"
}


# ============================================================
# NEW BLOCK: validation run
# ============================================================

results = []

batch_wall0 = time.time()

for event_index, entry in manifest.iterrows():

    row = int(
        entry["catalog_row"]
    )

    h5_path = str(
        entry["h5_path"]
    )

    candidate = (
        candidate_by_row.get(row)
    )

    if candidate is None:
        raise RuntimeError(
            f"row {row}: no successful "
            f"candidate-policy result"
        )

    if (
        candidate["generating_model"]
        != "H0"
    ):
        raise RuntimeError(
            f"row {row}: expected "
            f"generating_model=H0, got "
            f"{candidate['generating_model']}"
        )

    final_h0 = (
        candidate["final_h0"]
    )

    # Load exactly the already-materialized H5 event.
    meta = load_new_case(
        h5_path,
        core,
    )

    starts = build_reference_starts(
        final_h0
    )

    fits = []

    event_wall0 = time.time()

    print()
    print("=" * 90)
    print(
        f"REFERENCE EVENT "
        f"{len(results)+1}/{len(manifest)} "
        f"row={row}"
    )
    print("=" * 90)

    for fit_index, (
        label,
        initial,
    ) in enumerate(
        starts,
        start=1,
    ):

        fit_wall0 = time.time()

        try:

            record = run_one_fit_full(
                core,
                fit_lc,
                meta,
                "H1",
                initial,
                label,
            )

            compact = compact_record(
                label,
                initial,
                record,
                time.time() - fit_wall0,
            )

        except Exception as exc:

            compact = {
                "label": label,

                "initial": {
                    key: float(value)
                    for key, value
                    in initial.items()
                },

                "chi2": None,

                "status": "exception",

                "exception_type": (
                    type(exc).__name__
                ),

                "exception_message": (
                    str(exc)
                ),

                "wall_s": float(
                    time.time()
                    - fit_wall0
                ),
            }

        fits.append(
            compact
        )

        print(
            f"row={row} "
            f"fit={fit_index:02d}/"
            f"{len(starts)} "
            f"label={label:24s} "
            f"chi2={compact.get('chi2')} "
            f"status={compact.get('status')}",
            flush=True,
        )

    finite = [
        fit
        for fit in fits
        if (
            fit.get("chi2")
            is not None
            and np.isfinite(
                fit["chi2"]
            )
        )
    ]

    if not finite:
        raise RuntimeError(
            f"row {row}: no finite "
            f"H1 reference fits"
        )

    best = min(
        finite,
        key=lambda fit: fit["chi2"],
    )

    chi2_h0 = float(
        candidate["chi2_h0"]
    )

    chi2_h1_policy = float(
        candidate["chi2_h1"]
    )

    chi2_h1_reference = float(
        best["chi2"]
    )

    delta_policy = (
        chi2_h0
        - chi2_h1_policy
    )

    delta_reference = (
        chi2_h0
        - chi2_h1_reference
    )

    h1_basin_gap = (
        chi2_h1_policy
        - chi2_h1_reference
    )

    event_result = {
        "catalog_row": row,

        "h5_path": h5_path,

        "n_reference_starts": (
            len(starts)
        ),

        "n_reference_finite": (
            len(finite)
        ),

        "chi2_h0_policy": (
            chi2_h0
        ),

        "chi2_h1_policy": (
            chi2_h1_policy
        ),

        "chi2_h1_reference": (
            chi2_h1_reference
        ),

        "h1_basin_gap": (
            h1_basin_gap
        ),

        "delta_lrt_policy": (
            delta_policy
        ),

        "delta_lrt_reference": (
            delta_reference
        ),

        "delta_lrt_difference": (
            delta_reference
            - delta_policy
        ),

        "reference_winner": (
            best["label"]
        ),

        "reference_winner_record": (
            best
        ),

        "fits": fits,

        "event_wall_s": float(
            time.time()
            - event_wall0
        ),
    }

    results.append(
        event_result
    )

    payload = {
        "reference_name":
            "h0generated_h1_multistart_v1",

        "reference_purpose":
            "validation_only",

        "radii": list(
            RADII
        ),

        "angles_deg": list(
            ANGLES_DEG
        ),

        "n_requested_events": int(
            len(manifest)
        ),

        "n_processed_events": int(
            len(results)
        ),

        "events": results,

        "batch_wall_s": float(
            time.time()
            - batch_wall0
        ),
    }

    # Save after every event so a lost SSH session
    # does not lose completed reference work.
    write_payload_atomic(
        args.out,
        payload,
    )

    print()
    print(
        f"SUMMARY row={row}"
    )
    print(
        f"  chi2 H1 policy = "
        f"{chi2_h1_policy:.12g}"
    )
    print(
        f"  chi2 H1 ref    = "
        f"{chi2_h1_reference:.12g}"
    )
    print(
        f"  H1 basin gap   = "
        f"{h1_basin_gap:.12g}"
    )
    print(
        f"  D policy       = "
        f"{delta_policy:.12g}"
    )
    print(
        f"  D reference    = "
        f"{delta_reference:.12g}"
    )
    print(
        f"  winner         = "
        f"{best['label']}"
    )
    print(
        f"  finite starts  = "
        f"{len(finite)}/{len(starts)}"
    )
    print(
        f"  event wall     = "
        f"{event_result['event_wall_s']:.2f} s"
    )


print()
print("=" * 90)
print(
    f"DONE: "
    f"{len(results)} events, "
    f"wall="
    f"{time.time()-batch_wall0:.1f}s"
)
print(
    f"OUTPUT: {args.out}"
)
print("=" * 90)
