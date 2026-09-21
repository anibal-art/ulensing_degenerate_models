#!/usr/bin/env python3

"""
Finalize a parallel H0-calibration campaign.

The final frozen sample is defined exactly as in the sequential builder:
the first N_TARGET materialized events in deterministic candidate order.

Parallel execution may change wall-clock order, but must never change
scientific sample membership.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


N_TOTAL_ROWS = 966000


parser = argparse.ArgumentParser()

parser.add_argument(
    "--run-tag",
    required=True,
)

parser.add_argument(
    "--config",
    required=True,
)

parser.add_argument(
    "--n-candidates",
    type=int,
    default=40000,
)

parser.add_argument(
    "--target-n",
    type=int,
    default=10000,
)

parser.add_argument(
    "--seed",
    type=int,
    default=707070,
)

args = parser.parse_args()


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent

PP = (
    PROJECT_DIR
    / "validation"
    / "production_profiling"
)


def build_candidate_order(seed):
    with open(
        PP
        / "data"
        / "dev_excluded_rows.json"
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
        seed
    )

    return [
        int(c)
        for c in rng.permutation(
            N_TOTAL_ROWS
        )
        if int(c) not in excluded
    ]


config_path = Path(
    args.config
).resolve()

config = json.loads(
    config_path.read_text()
)

output_root = Path(
    os.path.expandvars(
        config["paths"]["output_root"]
    )
)

run_name = config[
    "run_name"
]

run_base = (
    output_root
    / "runs"
    / f"{run_name}_H0_CALIBRATION"
    / args.run_tag
)

if not run_base.is_dir():
    raise FileNotFoundError(
        f"Run directory not found: {run_base}"
    )


ledger_paths = sorted(
    run_base.glob(
        "ranks_*/candidate_status.csv"
    )
)

if not ledger_paths:
    raise RuntimeError(
        "No H0 candidate ledgers found."
    )


frames = []

for path in ledger_paths:
    df = pd.read_csv(
        path
    )

    df[
        "_ledger_path"
    ] = str(
        path
    )

    frames.append(
        df
    )


all_rows = pd.concat(
    frames,
    ignore_index=True,
)


if all_rows[
    "candidate_rank"
].duplicated().any():
    dup = all_rows[
        all_rows[
            "candidate_rank"
        ].duplicated(
            keep=False
        )
    ]

    raise RuntimeError(
        "Duplicate candidate ranks found:\n"
        + dup[
            [
                "candidate_rank",
                "candidate_row",
                "_ledger_path",
            ]
        ].to_string(
            index=False
        )
    )


if all_rows[
    "candidate_row"
].duplicated().any():
    raise RuntimeError(
        "Duplicate candidate catalog rows "
        "found across shards."
    )


all_rows[
    "candidate_rank"
] = all_rows[
    "candidate_rank"
].astype(int)

all_rows[
    "candidate_row"
] = all_rows[
    "candidate_row"
].astype(int)


all_rows = all_rows.sort_values(
    "candidate_rank"
).reset_index(
    drop=True
)


expected_ranks = np.arange(
    args.n_candidates,
    dtype=np.int64,
)

actual_ranks = all_rows[
    "candidate_rank"
].to_numpy(
    dtype=np.int64
)


if not np.array_equal(
    actual_ranks,
    expected_ranks,
):
    expected = set(
        expected_ranks.tolist()
    )

    actual = set(
        actual_ranks.tolist()
    )

    missing = sorted(
        expected
        - actual
    )

    extra = sorted(
        actual
        - expected
    )

    raise RuntimeError(
        "Candidate-rank coverage is incomplete. "
        f"missing={missing[:20]}, "
        f"extra={extra[:20]}"
    )


candidate_order = build_candidate_order(
    args.seed
)


expected_rows = np.asarray(
    candidate_order[
        :args.n_candidates
    ],
    dtype=np.int64,
)

actual_rows = all_rows[
    "candidate_row"
].to_numpy(
    dtype=np.int64
)


if not np.array_equal(
    actual_rows,
    expected_rows,
):
    mismatch = np.flatnonzero(
        actual_rows
        != expected_rows
    )

    i = int(
        mismatch[0]
    )

    raise RuntimeError(
        "Candidate-order mismatch at rank "
        f"{i}: expected "
        f"{expected_rows[i]}, "
        f"got {actual_rows[i]}"
    )


selected = all_rows[
    all_rows[
        "materialization_status"
    ]
    == "materialized"
].copy()


if len(
    selected
) < args.target_n:
    raise RuntimeError(
        "Not enough selected H0 events. "
        f"selected={len(selected)}, "
        f"target={args.target_n}"
    )


frozen = selected.head(
    args.target_n
).copy()


last_selected_rank = int(
    frozen[
        "candidate_rank"
    ].iloc[-1]
)


sequential_prefix = all_rows[
    all_rows[
        "candidate_rank"
    ]
    <= last_selected_rank
].copy()


# ============================================================
# Validate every frozen scientific result
# ============================================================

policy_names = set()

for row in frozen.itertuples(
    index=False
):
    h5_path = Path(
        str(
            row.h5_path
        )
    )

    event_json_path = Path(
        str(
            row.event_json_path
        )
    )

    if not h5_path.is_file():
        raise RuntimeError(
            "Frozen event missing H5: "
            f"{h5_path}"
        )

    if not event_json_path.is_file():
        raise RuntimeError(
            "Frozen event missing scientific "
            f"JSON: {event_json_path}"
        )

    payload = json.loads(
        event_json_path.read_text()
    )

    if payload.get(
        "status"
    ) != "success":
        raise RuntimeError(
            "Frozen event does not have "
            f"status=success: {event_json_path}"
        )

    if payload.get(
        "generating_model"
    ) != "H0":
        raise RuntimeError(
            "Frozen event is not H0-generated: "
            f"{event_json_path}"
        )

    if int(
        payload.get(
            "catalog_row",
            -1,
        )
    ) != int(
        row.candidate_row
    ):
        raise RuntimeError(
            "Frozen event catalog-row mismatch: "
            f"{event_json_path}"
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
                f"{event_json_path}: "
                f"{key} != {expected}"
            )

    if (
        payload[
            "final_h0"
        ].get(
            "coordinate_mode"
        )
        != "physical"
    ):
        raise RuntimeError(
            "Frozen H0 fit is not physical."
        )

    if (
        payload[
            "final_h1"
        ].get(
            "coordinate_mode"
        )
        != "physical"
    ):
        raise RuntimeError(
            "Frozen H1 fit is not physical."
        )

    policy_names.add(
        payload.get(
            "policy_name"
        )
    )


if len(
    policy_names
) != 1:
    raise RuntimeError(
        "Frozen events contain multiple "
        f"policy names: {policy_names}"
    )


final_dir = (
    run_base
    / "final"
)

final_dir.mkdir(
    parents=True,
    exist_ok=True,
)


manifest_path = (
    final_dir
    / "h0_calibration_v2_sample_frozen.csv"
)

candidate_log_path = (
    final_dir
    / "h0_calibration_v2_candidate_log.csv"
)

summary_path = (
    final_dir
    / "h0_calibration_v2_summary.json"
)


manifest = pd.DataFrame(
    {
        "catalog_row": frozen[
            "candidate_row"
        ].astype(int),
        "global_i": frozen[
            "global_i"
        ].astype(int),
        "h5_path": frozen[
            "h5_path"
        ],
    }
)


candidate_log = pd.DataFrame(
    {
        "candidate_row":
            sequential_prefix[
                "candidate_row"
            ].astype(int),

        "status":
            sequential_prefix[
                "materialization_status"
            ],

        "passed":
            sequential_prefix[
                "materialization_status"
            ]
            .eq(
                "materialized"
            ),

        "wall_time_s":
            sequential_prefix[
                "wall_time_s"
            ],
    }
)


manifest.to_csv(
    manifest_path,
    index=False,
)

candidate_log.to_csv(
    candidate_log_path,
    index=False,
)


summary = {
    "status": "frozen",
    "run_tag": args.run_tag,
    "seed": int(
        args.seed
    ),
    "n_candidates_produced": int(
        args.n_candidates
    ),
    "target_n": int(
        args.target_n
    ),
    "n_materialized_in_40k": int(
        len(
            selected
        )
    ),
    "sequential_candidates_evaluated": int(
        len(
            sequential_prefix
        )
    ),
    "last_selected_candidate_rank": int(
        last_selected_rank
    ),
    "sequential_pass_rate": float(
        args.target_n
        / len(
            sequential_prefix
        )
    ),
    "policy_name": next(
        iter(
            policy_names
        )
    ),
    "config_path": str(
        config_path
    ),
    "manifest_path": str(
        manifest_path
    ),
    "candidate_log_path": str(
        candidate_log_path
    ),
}


tmp_summary = Path(
    str(
        summary_path
    )
    + ".tmp"
)

tmp_summary.write_text(
    json.dumps(
        summary,
        indent=2,
    )
)

os.replace(
    tmp_summary,
    summary_path,
)


print(
    "=" * 80
)

print(
    "H0 CALIBRATION FROZEN"
)

print(
    "=" * 80
)

print(
    "run_tag                    =",
    args.run_tag,
)

print(
    "candidate ranks produced   =",
    args.n_candidates,
)

print(
    "materialized in production =",
    len(
        selected
    ),
)

print(
    "target frozen              =",
    args.target_n,
)

print(
    "sequential prefix length   =",
    len(
        sequential_prefix
    ),
)

print(
    "last selected rank         =",
    last_selected_rank,
)

print(
    "sequential pass rate       =",
    f"{summary['sequential_pass_rate']:.2%}",
)

print(
    "manifest                   =",
    manifest_path,
)

print(
    "candidate log              =",
    candidate_log_path,
)

print(
    "summary                    =",
    summary_path,
)
