#!/usr/bin/env python3
"""
Final H1-production chunk runner for the frozen two-fit LRT policy.

This file deliberately does NOT call the historical multi-fit pipeline.

Planned event flow:

    catalog row
        -> standalone H1 materialization
        -> pre-fit detectability
        -> if selected: load H5
        -> lrt_two_fit_policy_v2
        -> exactly 2 TRF fits
        -> checkpoint

This first implementation block contains only the production bootstrap.
No catalog row is simulated or fitted yet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ============================================================
# NEW BLOCK: repository paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

PROFILING_DIR = (
    PROJECT_DIR
    / "validation"
    / "production_profiling"
)

BOUNDS_AUDIT_DIR = (
    PROJECT_DIR
    / "validation"
    / "bounds_audit"
)

DEFAULT_CONFIG_PATH = (
    PROJECT_DIR
    / "configs"
    / "production"
    / "LRT_TWO_FIT_V2.json"
).resolve()

CORE_MANIFEST_PATH = (
    BOUNDS_AUDIT_DIR
    / "data"
    / "refit_manifest.csv"
).resolve()


# ============================================================
# NEW BLOCK: CLI
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Run one catalog-row chunk through the final "
        "two-fit H1 LRT production policy."
    )
)

parser.add_argument(
    "--config",
    default=str(DEFAULT_CONFIG_PATH),
)

parser.add_argument(
    "--row-start",
    type=int,
    required=True,
)

parser.add_argument(
    "--row-stop",
    type=int,
    required=True,
)

parser.add_argument(
    "--out-dir",
    required=True,
)

args = parser.parse_args()


if args.row_start < 0:
    raise ValueError(
        "--row-start must be non-negative."
    )

if args.row_stop <= args.row_start:
    raise ValueError(
        "--row-stop must be larger than --row-start."
    )

if args.row_stop > 966000:
    raise ValueError(
        "--row-stop cannot exceed 966000."
    )


config_path = Path(
    os.path.expandvars(
        os.path.expanduser(
            args.config
        )
    )
).resolve()

if not config_path.is_file():
    raise FileNotFoundError(
        f"Production config not found: {config_path}"
    )

out_dir = Path(
    os.path.expandvars(
        os.path.expanduser(
            args.out_dir
        )
    )
).resolve()

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# NEW BLOCK: make project validation modules importable
# ============================================================

sys.path.insert(
    0,
    str(PROFILING_DIR),
)

sys.path.insert(
    0,
    str(BOUNDS_AUDIT_DIR),
)


# ============================================================
# NEW BLOCK: simulation/materialization bootstrap
# ============================================================
#
# standalone_materialize chooses its driver configuration when the
# module is imported. Therefore this environment variable MUST be set
# before importing it.
# ============================================================

os.environ[
    "LRT_MATERIALIZE_CONFIG"
] = str(config_path)

from standalone_materialize import (  # noqa: E402
    materialize_event,
)


# ============================================================
# NEW BLOCK: frozen fitting-core numerical configuration
# ============================================================
#
# These values MUST be present before run_bounds_audit_refit_core is
# imported because that module installs its TRF runtime patches at
# import time.
# ============================================================

os.environ[
    "HIDDEN_PARALLAX_TRF_COORDS"
] = "physical"

os.environ[
    "HIDDEN_PARALLAX_TRF_X_SCALE"
] = "jac"

os.environ[
    "HIDDEN_PARALLAX_T0_MARGIN_FACTOR"
] = "0"


if not CORE_MANIFEST_PATH.is_file():
    raise FileNotFoundError(
        "Bounds-audit bootstrap manifest not found: "
        f"{CORE_MANIFEST_PATH}"
    )


# run_bounds_audit_refit_core parses its CLI at import time.
#
# Use the same harmless dry-run bootstrap row already exercised by
# run_lrt_policy_batch.py. The actual production event is supplied
# later through load_new_case(); no fit is executed by this import.
original_argv = list(
    sys.argv
)

sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row",
    "71181",
    "--manifest",
    str(CORE_MANIFEST_PATH),
    "--bounds-profile",
    "production_candidate",
    "--fit-scope",
    "h0",
    "--dry-run",
]

try:
    import run_bounds_audit_refit_core as core  # noqa: E402
finally:
    sys.argv = original_argv


# The core itself imports the canonical Roman/Rubin fit_lc module.
import fit_lc  # noqa: E402

from load_new_event import load_new_case  # noqa: E402
from lrt_fit_policy import (  # noqa: E402
    POLICY_NAME,
    run_lrt_fit_policy,
)

# NEW BLOCK: strict production JSON output.
from lrt_json_io import atomic_write_json  # noqa: E402


# ============================================================
# NEW BLOCK: production bootstrap invariants
# ============================================================

EXPECTED_POLICY_NAME = (
    "lrt_two_fit_policy_v2"
)

EXPECTED_BOUNDS = {
    "u0": [-10.0, 10.0],
    "tE": [0.1, 500000.0],
    "rho": [1.0e-7, 10.0],
    "piEN": [-40.0, 40.0],
    "piEE": [-40.0, 40.0],
}


if POLICY_NAME != EXPECTED_POLICY_NAME:
    raise RuntimeError(
        "Unexpected LRT policy: "
        f"{POLICY_NAME!r}"
    )

if core.BOUNDS_PROFILE != "production_candidate":
    raise RuntimeError(
        "Unexpected bounds profile: "
        f"{core.BOUNDS_PROFILE!r}"
    )

actual_bounds = core.BOUNDS_PROFILES[
    "production_candidate"
]

if actual_bounds != EXPECTED_BOUNDS:
    raise RuntimeError(
        "production_candidate bounds changed: "
        f"{actual_bounds!r}"
    )


production_config = json.loads(
    config_path.read_text()
)

expected_optimizer_options = (
    production_config[
        "fit"
    ][
        "optimizer_options"
    ]
)

if core.OPTIMIZER_OPTIONS != expected_optimizer_options:
    raise RuntimeError(
        "Fitting-core optimizer options do not match "
        "the final production config.\n"
        f"core={core.OPTIMIZER_OPTIONS!r}\n"
        f"config={expected_optimizer_options!r}"
    )


if os.environ[
    "HIDDEN_PARALLAX_TRF_COORDS"
] != "physical":
    raise RuntimeError(
        "Base TRF coordinates are not physical."
    )

if os.environ[
    "HIDDEN_PARALLAX_TRF_X_SCALE"
] != "jac":
    raise RuntimeError(
        "Base TRF x_scale is not jac."
    )

if os.environ[
    "HIDDEN_PARALLAX_T0_MARGIN_FACTOR"
] != "0":
    raise RuntimeError(
        "Unexpected t0 margin factor."
    )


# ============================================================
# NEW BLOCK: bootstrap audit output
# ============================================================

print()
print("=" * 80)
print("FINAL TWO-FIT PRODUCTION BOOTSTRAP")
print("=" * 80)

print(
    "materialization config =",
    config_path,
)

print(
    "row interval           =",
    f"[{args.row_start}, {args.row_stop})",
)

print(
    "output directory       =",
    out_dir,
)

print(
    "policy                 =",
    POLICY_NAME,
)

print(
    "bounds profile         =",
    core.BOUNDS_PROFILE,
)

print(
    "base coordinates       =",
    os.environ[
        "HIDDEN_PARALLAX_TRF_COORDS"
    ],
)

print(
    "base x_scale           =",
    os.environ[
        "HIDDEN_PARALLAX_TRF_X_SCALE"
    ],
)

print(
    "optimizer options      =",
    core.OPTIMIZER_OPTIONS,
)

print(
    "fit_lc                 =",
    Path(fit_lc.__file__).resolve(),
)

print(
    "materialize_event      =",
    materialize_event.__module__,
)

print(
    "load_new_case          =",
    load_new_case.__module__,
)

print(
    "run_lrt_fit_policy     =",
    run_lrt_fit_policy.__module__,
)

print()
print("BOOTSTRAP PASS")


# ============================================================
# NEW BLOCK: H1 materialization only
# ============================================================
#
# At this stage the runner only:
#
#   catalog row
#       -> simulate H1 event
#       -> apply the configured pre-fit detectability selection
#       -> save H5 only when selected
#       -> record the materialization status
#
# No H0 or H1 fit is executed in this block.
# ============================================================

import time  # noqa: E402

import pandas as pd  # noqa: E402


h5_dir = (
    out_dir
    / "h5"
)

h5_dir.mkdir(
    parents=True,
    exist_ok=True,
)

materialization_records = []

materialization_start = time.time()


for catalog_row in range(
    args.row_start,
    args.row_stop,
):
    timing = {}

    row_start_time = time.time()

    result = materialize_event(
        catalog_row,
        str(h5_dir),
        timing=timing,
        generating_model="H1",
    )

    wall_time_s = (
        time.time()
        - row_start_time
    )

    status = str(
        result["status"]
    )

    if status not in {
        "materialized",
        "not_detectable",
    }:
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            f"unexpected materialization status "
            f"{status!r}"
        )

    record = {
        "catalog_row": int(
            catalog_row
        ),
        "generating_model": "H1",
        "status": status,
        "selected_for_fitting": (
            status == "materialized"
        ),
        "wall_time_s": float(
            wall_time_s
        ),
        "global_i": None,
        "h5_path": None,
    }

    if status == "materialized":
        h5_path = Path(
            result["h5_path"]
        ).resolve()

        if not h5_path.is_file():
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                "materialize_event reported success "
                f"but H5 is missing: {h5_path}"
            )

        if int(
            result["global_i"]
        ) != catalog_row:
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                "global_i mismatch: "
                f"{result['global_i']}"
            )

        record[
            "global_i"
        ] = int(
            result["global_i"]
        )

        record[
            "h5_path"
        ] = str(
            h5_path
        )

        # NEW BLOCK: validate the materialized H1 event through
        # the exact loader used by the frozen fitting policy.
        #
        # No fit is executed here.
        meta = load_new_case(
            str(h5_path),
            core,
        )

        if int(meta["row"]) != catalog_row:
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                "load_new_case row mismatch: "
                f"{meta['row']}"
            )

        if int(meta["Source"]) != catalog_row:
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                "load_new_case Source mismatch: "
                f"{meta['Source']}"
            )

        if meta["generating_model"] != "H1":
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                "materialized production event was not "
                "recognized as H1: "
                f"{meta['generating_model']!r}"
            )

        required_truth = {
            "t0",
            "u0",
            "tE",
            "rho",
            "piEN",
            "piEE",
        }

        missing_truth = (
            required_truth
            - set(meta["truth"])
        )

        if missing_truth:
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                "H1 truth is incomplete after loading: "
                f"{sorted(missing_truth)}"
            )

        print(
            "[load] "
            f"row={catalog_row} "
            f"generating_model="
            f"{meta['generating_model']} "
            "truth=H1_complete",
            flush=True,
        )

        # NEW BLOCK: execute the frozen final two-fit policy.
        #
        # Exactly:
        #   1 x H1 TRF from the H1 truth start, physical coordinates
        #   1 x H0 TRF from the shared truth start, log_te_rho
        #
        # No continuation, rescue, morphology seed, or multistart.
        fit_wall0 = time.time()
        fit_cpu0 = time.process_time()

        fit_result = run_lrt_fit_policy(
            meta,
            core,
            fit_lc,
        )

        fit_wall_s = (
            time.time()
            - fit_wall0
        )

        fit_cpu_s = (
            time.process_time()
            - fit_cpu0
        )

        # Redundant production-level enforcement of the frozen
        # optimizer-call contract.
        expected_counts = {
            "n_nominal_trf": 2,
            "n_continuation_trf": 0,
            "n_rescue_trf": 0,
            "n_total_trf": 2,
        }

        for key, expected in expected_counts.items():
            actual = fit_result.get(
                key,
                None,
            )

            if actual != expected:
                raise RuntimeError(
                    f"catalog_row={catalog_row}: "
                    f"{key}={actual!r}; "
                    f"expected {expected}"
                )

        final_h0 = fit_result.get(
            "final_h0"
        )

        final_h1 = fit_result.get(
            "final_h1"
        )

        if final_h0 is None or final_h1 is None:
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                "final two-fit policy returned a missing final fit"
            )

        if (
            final_h0.get("coordinate_mode")
            != "log_te_rho"
        ):
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                "H1-generated -> H0 fit did not use "
                "log_te_rho coordinates: "
                f"{final_h0.get('coordinate_mode')!r}"
            )

        if (
            final_h1.get("coordinate_mode")
            != "physical"
        ):
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                "H1-generated -> H1 fit did not use "
                "physical coordinates: "
                f"{final_h1.get('coordinate_mode')!r}"
            )

        # The temporary H0 coordinate switch must leave the global
        # production runtime exactly in its baseline state.
        if (
            os.environ.get(
                "HIDDEN_PARALLAX_TRF_COORDS"
            )
            != "physical"
        ):
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                "TRF coordinate state was not restored to physical"
            )

        if (
            os.environ.get(
                "HIDDEN_PARALLAX_TRF_X_SCALE"
            )
            != "jac"
        ):
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                "TRF x_scale state was not restored to jac"
            )

        # Save the complete scientific result for this event.
        event_record = dict(
            fit_result
        )

        event_record["status"] = (
            "success"
        )

        event_record["h5_path"] = str(
            h5_path
        )

        event_record["truth"] = dict(
            meta["truth"]
        )

        event_record["event_fit_wall_s"] = float(
            fit_wall_s
        )

        event_record["event_fit_cpu_s"] = float(
            fit_cpu_s
        )

        event_result_path = (
            out_dir
            / "events"
            / f"Event_{catalog_row}.json"
        )

        atomic_write_json(
            event_record,
            event_result_path,
        )

        print(
            "[fit] "
            f"row={catalog_row} "
            f"n_trf={fit_result['n_total_trf']} "
            f"chi2_h0={fit_result['chi2_h0']:.6f} "
            f"chi2_h1={fit_result['chi2_h1']:.6f} "
            f"D={fit_result['delta_chi2_lrt']:.6f} "
            f"nesting_ok="
            f"{fit_result['final_nesting_ok']} "
            f"dt={fit_wall_s:.2f}s",
            flush=True,
        )

        print(
            "[save] "
            f"row={catalog_row} "
            f"path={event_result_path}",
            flush=True,
        )

    materialization_records.append(
        record
    )

    print(
        "[materialize] "
        f"row={catalog_row} "
        f"status={status} "
        f"selected="
        f"{record['selected_for_fitting']} "
        f"dt={wall_time_s:.2f}s",
        flush=True,
    )


materialization_status_path = (
    out_dir
    / "materialization_status.csv"
)

materialization_tmp_path = (
    out_dir
    / "materialization_status.csv.tmp"
)

pd.DataFrame(
    materialization_records,
    columns=[
        "catalog_row",
        "generating_model",
        "status",
        "selected_for_fitting",
        "wall_time_s",
        "global_i",
        "h5_path",
    ],
).to_csv(
    materialization_tmp_path,
    index=False,
)

os.replace(
    materialization_tmp_path,
    materialization_status_path,
)


n_materialized = sum(
    row["status"] == "materialized"
    for row in materialization_records
)

n_not_detectable = sum(
    row["status"] == "not_detectable"
    for row in materialization_records
)


print()
print("=" * 80)
print("H1 MATERIALIZATION SUMMARY")
print("=" * 80)

print(
    "rows processed         =",
    len(materialization_records),
)

print(
    "materialized           =",
    n_materialized,
)

print(
    "not detectable         =",
    n_not_detectable,
)

print(
    "status table           =",
    materialization_status_path,
)

print(
    "H5 directory           =",
    h5_dir,
)

print(
    "wall time [s]          =",
    time.time() - materialization_start,
)

print()
print(
    "H1 MATERIALIZATION: PASS"
)
