#!/usr/bin/env python3
"""
Build the dedicated H0-generated calibration sample -- NOT the 1M
Sedighe population, NOT the profiling sample, NOT a development set.
Deterministic seed distinct from every other sample used in this
project. truth_parallax=False (genuine no-parallax simulation).
"""
import sys
import os
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# NEW BLOCK: final production materialization config.
DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(
        HERE,
        "..",
        "..",
        "configs",
        "production",
        "LRT_TWO_FIT_V2.json",
    )
)

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=707070)
ap.add_argument("--n", type=int, default=10000)
ap.add_argument(
    "--config",
    default=DEFAULT_CONFIG_PATH,
    help=(
        "Simulation/materialization config. "
        "Defaults to the final two-fit production config."
    ),
)
ap.add_argument(
    "--out-manifest",
    default="results/h0_calibration_v2_sample_frozen.csv",
)
ap.add_argument(
    "--out-log",
    default="results/h0_calibration_v2_candidate_log.csv",
)
ap.add_argument(
    "--out-dir",
    default="runtime_local/h0_calibration_v2_h5",
)
ap.add_argument(
    "--checkpoint-every",
    type=int,
    default=25,
    help=(
        "Persist candidate log and selected-event manifest "
        "every N evaluated candidates."
    ),
)
args = ap.parse_args()

# standalone_materialize selects its config at import time.
# Set it explicitly before importing that module.
MATERIALIZE_CONFIG_PATH = os.path.abspath(
    os.path.expanduser(
        os.path.expandvars(args.config)
    )
)

if not os.path.isfile(MATERIALIZE_CONFIG_PATH):
    raise FileNotFoundError(
        "Materialization config does not exist: "
        f"{MATERIALIZE_CONFIG_PATH}"
    )

os.environ[
    "LRT_MATERIALIZE_CONFIG"
] = MATERIALIZE_CONFIG_PATH

N_TOTAL_ROWS = 966000

# NEW BLOCK: reproducible development-set exclusion.
#
# This list is versioned with the repository so calibration does
# not depend on host-local /tmp state.
DEV_EXCLUDED_PATH = os.path.join(
    HERE,
    "data",
    "dev_excluded_rows.json",
)

with open(DEV_EXCLUDED_PATH) as handle:
    dev_excluded = set(
        int(row)
        for row in json.load(handle)
    )
h1_gen_sample = set(pd.read_csv(os.path.join(HERE, "results", "profiling_sample_frozen_100.csv"))
                     ["catalog_row"].tolist())
h0_smoke_sample = set(pd.read_csv(os.path.join(HERE, "results", "h0_generated_sample_frozen.csv"))
                       ["catalog_row"].tolist())
excluded = dev_excluded | h1_gen_sample | h0_smoke_sample

rng = np.random.default_rng(args.seed)
full_order = rng.permutation(N_TOTAL_ROWS)
candidate_order = [int(c) for c in full_order if c not in excluded]

from standalone_materialize import materialize_event  # noqa: E402

# NEW BLOCK: resumable production outputs.

def resolve_output_path(value):
    value = os.path.expanduser(
        os.path.expandvars(value)
    )

    if os.path.isabs(value):
        return os.path.abspath(value)

    return os.path.abspath(
        os.path.join(
            HERE,
            value,
        )
    )


def atomic_write_csv(rows, path, columns):
    path = os.path.abspath(path)
    parent = os.path.dirname(path)

    if parent:
        os.makedirs(
            parent,
            exist_ok=True,
        )

    tmp_path = path + ".tmp"

    pd.DataFrame(
        rows,
        columns=columns,
    ).to_csv(
        tmp_path,
        index=False,
    )

    os.replace(
        tmp_path,
        path,
    )


out_dir = resolve_output_path(
    args.out_dir
)

manifest_path = resolve_output_path(
    args.out_manifest
)

log_path = resolve_output_path(
    args.out_log
)

os.makedirs(
    out_dir,
    exist_ok=True,
)

if args.n <= 0:
    raise ValueError(
        "--n must be positive."
    )

if args.checkpoint_every <= 0:
    raise ValueError(
        "--checkpoint-every must be positive."
    )


# NEW BLOCK: resume only from a complete, internally consistent
# checkpoint pair.
manifest_exists = os.path.isfile(
    manifest_path
)

log_exists = os.path.isfile(
    log_path
)

if manifest_exists != log_exists:
    raise RuntimeError(
        "Incomplete H0 calibration checkpoint: manifest and "
        "candidate log must either both exist or both be absent. "
        f"manifest={manifest_exists}, log={log_exists}"
    )

if manifest_exists:
    log_df = pd.read_csv(
        log_path
    )

    frozen_df = pd.read_csv(
        manifest_path
    )

    log_rows = log_df.to_dict(
        orient="records"
    )

    frozen = frozen_df.to_dict(
        orient="records"
    )

    processed_candidates = [
        int(row["candidate_row"])
        for row in log_rows
    ]

    expected_prefix = candidate_order[
        :len(processed_candidates)
    ]

    if processed_candidates != expected_prefix:
        raise RuntimeError(
            "Existing candidate log is not an exact prefix of "
            "the deterministic candidate order. Refusing to "
            "mix calibration campaigns."
        )

    if len(set(processed_candidates)) != len(
        processed_candidates
    ):
        raise RuntimeError(
            "Duplicate candidate rows found in existing log."
        )

    frozen_catalog_rows = [
        int(row["catalog_row"])
        for row in frozen
    ]

    if len(set(frozen_catalog_rows)) != len(
        frozen_catalog_rows
    ):
        raise RuntimeError(
            "Duplicate catalog rows found in existing manifest."
        )

    passed_catalog_rows = {
        int(row["candidate_row"])
        for row in log_rows
        if bool(row["passed"])
    }

    if set(frozen_catalog_rows) != passed_catalog_rows:
        raise RuntimeError(
            "Existing manifest does not match the passed "
            "candidate rows in the candidate log."
        )

    if len(frozen) > args.n:
        raise RuntimeError(
            "Existing checkpoint already contains more selected "
            f"events ({len(frozen)}) than requested --n={args.n}."
        )

    print(
        "RESUME: "
        f"evaluated={len(log_rows)} "
        f"selected={len(frozen)}/{args.n}",
        flush=True,
    )

else:
    log_rows = []
    frozen = []


def write_checkpoint():
    atomic_write_csv(
        log_rows,
        log_path,
        columns=[
            "candidate_row",
            "status",
            "passed",
            "wall_time_s",
        ],
    )

    atomic_write_csv(
        frozen,
        manifest_path,
        columns=[
            "catalog_row",
            "global_i",
            "h5_path",
        ],
    )


t_start = time.time()

start_index = len(
    log_rows
)

try:
    for candidate_row in candidate_order[
        start_index:
    ]:

        if len(frozen) >= args.n:
            break

        timing = {}

        t0 = time.time()

        r = materialize_event(
            candidate_row,
            out_dir,
            timing=timing,
            generating_model="H0",
        )

        dt = time.time() - t0

        passed = (
            r["status"] == "materialized"
        )

        log_rows.append(
            {
                "candidate_row": candidate_row,
                "status": r["status"],
                "passed": passed,
                "wall_time_s": dt,
            }
        )

        if passed:
            frozen.append(
                {
                    "catalog_row": candidate_row,
                    "global_i": r["global_i"],
                    "h5_path": r["h5_path"],
                }
            )

        if (
            len(log_rows) % args.checkpoint_every == 0
            or len(frozen) >= args.n
        ):
            write_checkpoint()

        if (
            len(log_rows) % 10 == 0
            or passed
        ):
            print(
                f"[{len(log_rows)}] "
                f"row={candidate_row} "
                f"status={r['status']} "
                f"n_frozen={len(frozen)}/{args.n} "
                f"dt={dt:.2f}s "
                f"total_elapsed="
                f"{time.time()-t_start:.0f}s",
                flush=True,
            )

finally:
    # Persist all completed candidates even after Ctrl-C or an
    # exception raised between periodic checkpoints.
    write_checkpoint()


if len(frozen) != args.n:
    raise RuntimeError(
        "Candidate order exhausted before reaching the requested "
        f"H0 calibration size: {len(frozen)}/{args.n}."
    )


pass_rate = (
    len(frozen) / len(log_rows)
    if log_rows
    else float("nan")
)

print()
print("=" * 100)
print(
    f"FROZEN: {len(frozen)} H0-CALIBRATION events, "
    f"seed={args.seed}, "
    f"candidates evaluated={len(log_rows)}, "
    f"pass rate={pass_rate:.2%}"
)
print(
    "manifest:",
    manifest_path,
)
print(
    "candidate log:",
    log_path,
)
print(
    "H5 directory:",
    out_dir,
)
print(
    "total wall time:",
    time.time() - t_start,
    "s",
)
