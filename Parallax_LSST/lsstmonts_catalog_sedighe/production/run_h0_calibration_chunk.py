#!/usr/bin/env python3

"""
Parallel H0-calibration chunk runner.

The scientific definition of the H0 calibration sample is inherited
from validation/production_profiling/build_h0_calibration_sample.py:

  - deterministic RNG seed;
  - identical development/profiling/smoke exclusions;
  - candidate order = filtered RNG permutation of the 966000 catalog rows.

Parallelization is performed over candidate_rank, never directly over
catalog_row.

Scientific fitting is NOT reimplemented here. Selected H0 events use:

  standalone_materialize.materialize_event(..., generating_model="H0")
  load_new_event.load_new_case(...)
  lrt_fit_policy.run_lrt_fit_policy(...)

The frozen H0-generated policy therefore remains:

  1 x H0 TRF, truth start, physical coordinates
  1 x H1 TRF, truth + piE=(0,0), physical coordinates

No continuation, rescue, DE, or multistart.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


N_TOTAL_ROWS = 966000


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--rank-start",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--rank-stop",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--n-candidates",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=707070,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    return parser.parse_args()


args = parse_args()

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent

PP = (
    PROJECT_DIR
    / "validation"
    / "production_profiling"
)

BA = (
    PROJECT_DIR
    / "validation"
    / "bounds_audit"
)

BC = (
    PROJECT_DIR
    / "validation"
    / "bounds_convergence"
)

sys.path.insert(
    0,
    str(PP),
)

sys.path.insert(
    0,
    str(BA),
)

sys.path.insert(
    0,
    str(BC),
)


# ============================================================
# Frozen production numerical environment
# ============================================================

CONFIG_PATH = Path(
    os.path.expanduser(
        os.path.expandvars(
            args.config
        )
    )
).resolve()

if not CONFIG_PATH.is_file():
    raise FileNotFoundError(
        f"Config does not exist: {CONFIG_PATH}"
    )

os.environ[
    "LRT_MATERIALIZE_CONFIG"
] = str(CONFIG_PATH)

os.environ[
    "HIDDEN_PARALLAX_TRF_COORDS"
] = "physical"

os.environ[
    "HIDDEN_PARALLAX_TRF_X_SCALE"
] = "jac"

os.environ[
    "HIDDEN_PARALLAX_T0_MARGIN_FACTOR"
] = "0"


# ============================================================
# Import existing scientific implementation
# ============================================================

from standalone_materialize import materialize_event  # noqa: E402


_original_argv = list(
    sys.argv
)

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
    "h0",
    "--dry-run",
]

import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402

sys.argv = _original_argv

from load_new_event import load_new_case  # noqa: E402
from lrt_fit_policy import (  # noqa: E402
    POLICY_NAME,
    run_lrt_fit_policy,
)


if core.BOUNDS_PROFILE != "production_candidate":
    raise RuntimeError(
        "H0 production requires "
        "BOUNDS_PROFILE='production_candidate'."
    )


# ============================================================
# Candidate order: exact semantics of sequential builder
# ============================================================

def build_candidate_order(seed):
    dev_excluded_path = (
        PP
        / "data"
        / "dev_excluded_rows.json"
    )

    with open(
        dev_excluded_path
    ) as handle:
        dev_excluded = {
            int(row)
            for row in json.load(
                handle
            )
        }

    h1_sample = set(
        pd.read_csv(
            PP
            / "results"
            / "profiling_sample_frozen_100.csv"
        )["catalog_row"]
        .astype(int)
        .tolist()
    )

    h0_smoke = set(
        pd.read_csv(
            PP
            / "results"
            / "h0_generated_sample_frozen.csv"
        )["catalog_row"]
        .astype(int)
        .tolist()
    )

    excluded = (
        dev_excluded
        | h1_sample
        | h0_smoke
    )

    rng = np.random.default_rng(
        int(seed)
    )

    full_order = rng.permutation(
        N_TOTAL_ROWS
    )

    candidate_order = [
        int(c)
        for c in full_order
        if int(c) not in excluded
    ]

    return candidate_order


candidate_order = build_candidate_order(
    args.seed
)

if args.n_candidates <= 0:
    raise ValueError(
        "--n-candidates must be positive."
    )

if args.n_candidates > len(
    candidate_order
):
    raise ValueError(
        "--n-candidates exceeds available "
        "candidate order."
    )

if args.rank_start < 0:
    raise ValueError(
        "--rank-start must be >= 0."
    )

if args.rank_stop <= args.rank_start:
    raise ValueError(
        "--rank-stop must be larger than "
        "--rank-start."
    )

if args.rank_stop > args.n_candidates:
    raise ValueError(
        "--rank-stop exceeds --n-candidates."
    )


# ============================================================
# Chunk outputs
# ============================================================

OUT_DIR = Path(
    args.out_dir
).resolve()

H5_DIR = (
    OUT_DIR
    / "h5"
)

EVENT_DIR = (
    OUT_DIR
    / "events"
)

LEDGER_PATH = (
    OUT_DIR
    / "candidate_status.csv"
)

LEDGER_TMP = Path(
    str(LEDGER_PATH)
    + ".tmp"
)

SUMMARY_PATH = (
    OUT_DIR
    / "summary.json"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

H5_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EVENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CHECKPOINT_COLUMNS = [
    "candidate_rank",
    "candidate_row",
    "materialization_status",
    "passed",
    "wall_time_s",
    "global_i",
    "h5_path",
    "event_json_path",
]


def json_default(value):
    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    return str(value)


def atomic_write_json(
    payload,
    path,
):
    path = Path(
        path
    )

    tmp = Path(
        str(path)
        + ".tmp"
    )

    with open(
        tmp,
        "w",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            default=json_default,
        )

    os.replace(
        tmp,
        path,
    )


records = []


def write_checkpoint():
    pd.DataFrame(
        records,
        columns=CHECKPOINT_COLUMNS,
    ).to_csv(
        LEDGER_TMP,
        index=False,
    )

    os.replace(
        LEDGER_TMP,
        LEDGER_PATH,
    )


expected_pairs = [
    (
        rank,
        int(
            candidate_order[
                rank
            ]
        ),
    )
    for rank in range(
        args.rank_start,
        args.rank_stop,
    )
]


# ============================================================
# Validate/resume existing ordered prefix
# ============================================================

if LEDGER_PATH.is_file():
    existing = pd.read_csv(
        LEDGER_PATH
    )

    records = existing.to_dict(
        orient="records"
    )

    actual_pairs = [
        (
            int(
                row[
                    "candidate_rank"
                ]
            ),
            int(
                row[
                    "candidate_row"
                ]
            ),
        )
        for row in records
    ]

    expected_prefix = expected_pairs[
        :len(actual_pairs)
    ]

    if actual_pairs != expected_prefix:
        raise RuntimeError(
            "Existing H0 chunk ledger is not "
            "an exact prefix of the deterministic "
            "candidate order."
        )

    if len(actual_pairs) != len(
        set(actual_pairs)
    ):
        raise RuntimeError(
            "Duplicate candidate entries in "
            "existing H0 ledger."
        )

    print(
        "[resume] "
        f"rank_start={args.rank_start} "
        f"rank_stop={args.rank_stop} "
        f"completed={len(records)} "
        f"remaining="
        f"{len(expected_pairs)-len(records)}",
        flush=True,
    )


# ============================================================
# Frozen H0 two-fit wrapper
# ============================================================

def validate_success_json(
    catalog_row,
    h5_path,
    event_json_path,
):
    try:
        payload = json.loads(
            Path(
                event_json_path
            ).read_text()
        )
    except Exception as exc:
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "existing event JSON cannot be read."
        ) from exc

    if int(
        payload.get(
            "catalog_row",
            -1,
        )
    ) != int(
        catalog_row
    ):
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "event JSON catalog_row mismatch."
        )

    if payload.get(
        "generating_model"
    ) != "H0":
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "event JSON is not H0-generated."
        )

    if payload.get(
        "policy_name"
    ) != POLICY_NAME:
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "event JSON policy mismatch."
        )

    if payload.get(
        "status"
    ) != "success":
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "event JSON is not successful."
        )

    expected_counts = {
        "n_nominal_trf": 2,
        "n_continuation_trf": 0,
        "n_rescue_trf": 0,
        "n_total_trf": 2,
    }

    for key, expected in (
        expected_counts.items()
    ):
        if payload.get(
            key
        ) != expected:
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                f"{key}={payload.get(key)!r}; "
                f"expected {expected}."
            )

    final_h0 = payload.get(
        "final_h0"
    )

    final_h1 = payload.get(
        "final_h1"
    )

    if (
        not isinstance(
            final_h0,
            dict,
        )
        or not isinstance(
            final_h1,
            dict,
        )
    ):
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "missing final H0/H1 fit."
        )

    if (
        final_h0.get(
            "coordinate_mode"
        )
        != "physical"
    ):
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "H0 final fit is not physical."
        )

    if (
        final_h1.get(
            "coordinate_mode"
        )
        != "physical"
    ):
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "H1 final fit is not physical."
        )

    payload_h5 = Path(
        str(
            payload.get(
                "h5_path"
            )
        )
    ).resolve()

    if payload_h5 != Path(
        h5_path
    ).resolve():
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "event JSON H5 path mismatch."
        )

    return payload


def fit_materialized_h0_event(
    catalog_row,
    h5_path,
):
    meta = load_new_case(
        str(
            h5_path
        ),
        core,
    )

    if int(
        meta["row"]
    ) != int(
        catalog_row
    ):
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "loaded row mismatch."
        )

    if int(
        meta["Source"]
    ) != int(
        catalog_row
    ):
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "loaded Source mismatch."
        )

    if meta[
        "generating_model"
    ] != "H0":
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "materialized event was not "
            "recognized as H0."
        )

    required_truth = {
        "t0",
        "u0",
        "tE",
        "rho",
    }

    missing = (
        required_truth
        - set(
            meta["truth"]
        )
    )

    if missing:
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "H0 truth is incomplete: "
            f"{sorted(missing)}"
        )

    if (
        "piEN" in meta["truth"]
        or "piEE" in meta["truth"]
    ):
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "H0 truth unexpectedly contains piE."
        )

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

    expected_counts = {
        "n_nominal_trf": 2,
        "n_continuation_trf": 0,
        "n_rescue_trf": 0,
        "n_total_trf": 2,
    }

    for key, expected in (
        expected_counts.items()
    ):
        actual = fit_result.get(
            key
        )

        if actual != expected:
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                f"{key}={actual!r}; "
                f"expected {expected}."
            )

    final_h0 = fit_result.get(
        "final_h0"
    )

    final_h1 = fit_result.get(
        "final_h1"
    )

    if (
        final_h0 is None
        or final_h1 is None
    ):
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "missing final H0/H1 fit."
        )

    if (
        final_h0.get(
            "coordinate_mode"
        )
        != "physical"
    ):
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "H0 fit did not use physical "
            "coordinates."
        )

    if (
        final_h1.get(
            "coordinate_mode"
        )
        != "physical"
    ):
        raise RuntimeError(
            f"catalog_row={catalog_row}: "
            "H1 fit did not use physical "
            "coordinates."
        )

    if (
        os.environ.get(
            "HIDDEN_PARALLAX_TRF_COORDS"
        )
        != "physical"
    ):
        raise RuntimeError(
            "TRF coordinate environment was "
            "not restored to physical."
        )

    event_record = dict(
        fit_result
    )

    event_record[
        "status"
    ] = "success"

    event_record[
        "h5_path"
    ] = str(
        Path(
            h5_path
        ).resolve()
    )

    event_record[
        "truth"
    ] = dict(
        meta["truth"]
    )

    event_record[
        "event_fit_wall_s"
    ] = float(
        fit_wall_s
    )

    event_record[
        "event_fit_cpu_s"
    ] = float(
        fit_cpu_s
    )

    event_json_path = (
        EVENT_DIR
        / f"Event_{catalog_row}.json"
    )

    atomic_write_json(
        event_record,
        event_json_path,
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

    return (
        event_json_path,
        fit_result,
    )


# ============================================================
# Process remaining candidate-rank prefix
# ============================================================

chunk_wall0 = time.time()

start_offset = len(
    records
)

try:
    for (
        candidate_rank,
        candidate_row,
    ) in expected_pairs[
        start_offset:
    ]:

        row_wall0 = time.time()

        h5_path = (
            H5_DIR
            / f"Event_{candidate_row}.h5"
        ).resolve()

        event_json_path = (
            EVENT_DIR
            / f"Event_{candidate_row}.json"
        ).resolve()

        h5_exists = h5_path.is_file()
        json_exists = event_json_path.is_file()

        if (
            json_exists
            and not h5_exists
        ):
            raise RuntimeError(
                f"candidate_rank={candidate_rank}, "
                f"catalog_row={candidate_row}: "
                "scientific JSON exists but H5 "
                "is missing."
            )

        if (
            h5_exists
            and json_exists
        ):
            validate_success_json(
                candidate_row,
                h5_path,
                event_json_path,
            )

            record = {
                "candidate_rank": int(
                    candidate_rank
                ),
                "candidate_row": int(
                    candidate_row
                ),
                "materialization_status":
                    "materialized",
                "passed": True,
                "wall_time_s": float(
                    time.time()
                    - row_wall0
                ),
                "global_i": int(
                    candidate_row
                ),
                "h5_path": str(
                    h5_path
                ),
                "event_json_path": str(
                    event_json_path
                ),
            }

            records.append(
                record
            )

            write_checkpoint()

            print(
                "[resume-json] "
                f"rank={candidate_rank} "
                f"row={candidate_row}",
                flush=True,
            )

            continue

        if h5_exists:
            print(
                "[resume-h5] "
                f"rank={candidate_rank} "
                f"row={candidate_row}",
                flush=True,
            )

            fit_materialized_h0_event(
                candidate_row,
                h5_path,
            )

            record = {
                "candidate_rank": int(
                    candidate_rank
                ),
                "candidate_row": int(
                    candidate_row
                ),
                "materialization_status":
                    "materialized",
                "passed": True,
                "wall_time_s": float(
                    time.time()
                    - row_wall0
                ),
                "global_i": int(
                    candidate_row
                ),
                "h5_path": str(
                    h5_path
                ),
                "event_json_path": str(
                    event_json_path
                ),
            }

            records.append(
                record
            )

            write_checkpoint()

            continue

        timing = {}

        result = materialize_event(
            candidate_row,
            str(
                H5_DIR
            ),
            timing=timing,
            generating_model="H0",
        )

        status = str(
            result["status"]
        )

        if status not in {
            "materialized",
            "not_detectable",
            "invalid_catalog_row",
        }:
            raise RuntimeError(
                f"candidate_rank={candidate_rank}, "
                f"catalog_row={candidate_row}: "
                f"unexpected materialization "
                f"status={status!r}"
            )

        passed = (
            status
            == "materialized"
        )

        if passed:
            actual_h5 = Path(
                result["h5_path"]
            ).resolve()

            if actual_h5 != h5_path:
                raise RuntimeError(
                    f"catalog_row={candidate_row}: "
                    "materializer returned unexpected "
                    f"H5 path {actual_h5}; "
                    f"expected {h5_path}."
                )

            if int(
                result["global_i"]
            ) != int(
                candidate_row
            ):
                raise RuntimeError(
                    f"catalog_row={candidate_row}: "
                    "global_i mismatch."
                )

            fit_materialized_h0_event(
                candidate_row,
                h5_path,
            )

            global_i = int(
                result["global_i"]
            )

            h5_record = str(
                h5_path
            )

            json_record = str(
                event_json_path
            )

        else:
            global_i = None
            h5_record = None
            json_record = None

        record = {
            "candidate_rank": int(
                candidate_rank
            ),
            "candidate_row": int(
                candidate_row
            ),
            "materialization_status":
                status,
            "passed": bool(
                passed
            ),
            "wall_time_s": float(
                time.time()
                - row_wall0
            ),
            "global_i": global_i,
            "h5_path": h5_record,
            "event_json_path":
                json_record,
        }

        records.append(
            record
        )

        write_checkpoint()

        print(
            "[candidate] "
            f"rank={candidate_rank} "
            f"row={candidate_row} "
            f"status={status} "
            f"passed={passed}",
            flush=True,
        )

finally:
    write_checkpoint()


if len(
    records
) != len(
    expected_pairs
):
    raise RuntimeError(
        "Chunk did not complete its full "
        "candidate-rank interval."
    )


n_materialized = sum(
    str(
        row[
            "materialization_status"
        ]
    )
    == "materialized"
    for row in records
)

n_not_detectable = sum(
    str(
        row[
            "materialization_status"
        ]
    )
    == "not_detectable"
    for row in records
)

n_invalid = sum(
    str(
        row[
            "materialization_status"
        ]
    )
    == "invalid_catalog_row"
    for row in records
)


summary = {
    "status": "done",
    "policy_name": POLICY_NAME,
    "seed": int(
        args.seed
    ),
    "n_candidates_global": int(
        args.n_candidates
    ),
    "rank_start": int(
        args.rank_start
    ),
    "rank_stop": int(
        args.rank_stop
    ),
    "n_processed": int(
        len(
            records
        )
    ),
    "n_materialized": int(
        n_materialized
    ),
    "n_not_detectable": int(
        n_not_detectable
    ),
    "n_invalid_catalog_row": int(
        n_invalid
    ),
    "chunk_wall_s": float(
        time.time()
        - chunk_wall0
    ),
}

atomic_write_json(
    summary,
    SUMMARY_PATH,
)

print(
    "DONE "
    f"ranks=[{args.rank_start},"
    f"{args.rank_stop}) "
    f"processed={len(records)} "
    f"materialized={n_materialized} "
    f"not_detectable={n_not_detectable} "
    f"invalid={n_invalid}",
    flush=True,
)
