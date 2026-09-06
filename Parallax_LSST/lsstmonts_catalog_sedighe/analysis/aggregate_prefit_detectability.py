#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument(
    "--study-dir",
    required=True,
)

args = parser.parse_args()

study_dir = Path(
    args.study_dir
).resolve()

manifest_file = (
    study_dir
    / "manifest.env"
)

if not manifest_file.exists():
    raise FileNotFoundError(
        manifest_file
    )


# ----------------------------------------------------------------------
# Parse simple shell manifest.
# ----------------------------------------------------------------------

manifest = {}

for raw in manifest_file.read_text().splitlines():

    raw = raw.strip()

    if not raw or "=" not in raw:
        continue

    key, value = raw.split(
        "=",
        1,
    )

    manifest[
        key.strip()
    ] = value.strip().strip("'").strip('"')


run_name = manifest[
    "RUN_NAME"
]

run_tag = manifest[
    "RUN_TAG"
]

output_root = Path(
    manifest[
        "OUTPUT_ROOT"
    ]
)

n_rows_requested = int(
    manifest[
        "N_ROWS"
    ]
)

n_chunks_expected = int(
    manifest[
        "N_CHUNKS"
    ]
)


run_base = (
    output_root
    / "runs"
    / run_name
)


# IMPORTANT:
# Search only inside this dedicated run_name.
run_dirs = sorted(
    run_base.glob(
        f"{run_tag}_chunk_*"
    )
)


print("=" * 100)
print("PREFIT DETECTABILITY AGGREGATION")
print("=" * 100)
print("study_dir        =", study_dir)
print("run_base         =", run_base)
print("requested rows   =", n_rows_requested)
print("expected chunks  =", n_chunks_expected)
print("found run dirs   =", len(run_dirs))
print("=" * 100)


frames = []
missing = []


for run_dir in run_dirs:

    summary = (
        run_dir
        / "logs"
        / "run_summary.parquet"
    )

    if not summary.exists():
        missing.append(
            str(run_dir)
        )
        continue

    df = pd.read_parquet(
        summary
    )

    df[
        "_source_run_dir"
    ] = str(
        run_dir
    )

    frames.append(
        df
    )


if not frames:
    raise RuntimeError(
        "No run_summary.parquet files found."
    )


all_df = pd.concat(
    frames,
    ignore_index=True,
    sort=False,
)


if "catalog_row" not in all_df.columns:
    raise RuntimeError(
        "catalog_row missing from summaries."
    )


all_df = (
    all_df
    .sort_values(
        "catalog_row"
    )
    .reset_index(
        drop=True
    )
)


# ----------------------------------------------------------------------
# Duplicate QC.
# ----------------------------------------------------------------------

duplicates = (
    all_df[
        "catalog_row"
    ]
    .duplicated(
        keep=False
    )
)

n_duplicates = int(
    duplicates.sum()
)


# ----------------------------------------------------------------------
# Detectability classification.
# ----------------------------------------------------------------------

if (
    "detectability_pass"
    not in all_df.columns
):
    raise RuntimeError(
        "detectability_pass missing."
    )


detectable_mask = (
    all_df[
        "detectability_pass"
    ]
    .fillna(False)
    .astype(bool)
)

detectable = (
    all_df[
        detectable_mask
    ]
    .copy()
)


failures = (
    all_df[
        ~detectable_mask
    ]
    .copy()
)


# ----------------------------------------------------------------------
# Outputs.
# ----------------------------------------------------------------------

all_path = (
    study_dir
    / "all_detectability.parquet"
)

detectable_path = (
    study_dir
    / "detectable_events.parquet"
)

rows_path = (
    study_dir
    / "detectable_catalog_rows.txt"
)

failure_path = (
    study_dir
    / "failure_reasons.csv"
)

quantiles_path = (
    study_dir
    / "metric_quantiles.csv"
)

summary_json = (
    study_dir
    / "summary.json"
)


all_df.to_parquet(
    all_path,
    index=False,
)

detectable.to_parquet(
    detectable_path,
    index=False,
)


with rows_path.open(
    "w"
) as f:

    for row in detectable[
        "catalog_row"
    ].astype(int):

        f.write(
            f"{row}\n"
        )


# ----------------------------------------------------------------------
# Failure reasons.
# ----------------------------------------------------------------------

reason_counts = {}

if (
    "detectability_reasons"
    in failures.columns
):

    for value in failures[
        "detectability_reasons"
    ].fillna(""):

        for reason in str(
            value
        ).split(";"):

            reason = (
                reason.strip()
            )

            if not reason:
                continue

            reason_counts[
                reason
            ] = (
                reason_counts.get(
                    reason,
                    0,
                )
                + 1
            )


failure_df = pd.DataFrame(
    [
        {
            "reason": key,
            "count": value,
        }
        for key, value
        in reason_counts.items()
    ]
)

if not failure_df.empty:

    failure_df = (
        failure_df
        .sort_values(
            "count",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

failure_df.to_csv(
    failure_path,
    index=False,
)


# ----------------------------------------------------------------------
# Metric distributions.
# ----------------------------------------------------------------------

metrics = [
    "detectability_n_obs",
    "detectability_n_bands",
    "detectability_n_peak",
    "detectability_n_left_peak",
    "detectability_n_right_peak",
    "detectability_nearest_dt_over_tE",
    "detectability_delta_chi2_const_asimov",
    "detectability_delta_chi2_per_point",
    "detectability_n_nsigma",
    "detectability_max_snr",
]


quantile_rows = []

for population_name, frame in [
    ("all", all_df),
    ("pass", detectable),
    ("fail", failures),
]:

    for metric in metrics:

        if metric not in frame.columns:
            continue

        values = pd.to_numeric(
            frame[
                metric
            ],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            continue

        quantile_rows.append(
            {
                "population": population_name,
                "metric": metric,
                "N": len(values),
                "min": values.min(),
                "q05": values.quantile(0.05),
                "q25": values.quantile(0.25),
                "median": values.median(),
                "q75": values.quantile(0.75),
                "q95": values.quantile(0.95),
                "max": values.max(),
                "mean": values.mean(),
            }
        )


quantiles = pd.DataFrame(
    quantile_rows
)

quantiles.to_csv(
    quantiles_path,
    index=False,
)


# ----------------------------------------------------------------------
# High-level summary.
# ----------------------------------------------------------------------

n_audited = len(
    all_df
)

n_pass = len(
    detectable
)

n_fail = len(
    failures
)

pass_fraction = (
    n_pass
    / n_audited
    if n_audited
    else np.nan
)


status_counts = (
    all_df[
        "status"
    ]
    .value_counts(
        dropna=False
    )
    .to_dict()
    if "status"
    in all_df.columns
    else {}
)


summary = {
    "run_name": run_name,
    "run_tag": run_tag,

    "requested_raw_rows": (
        n_rows_requested
    ),

    "expected_chunks": (
        n_chunks_expected
    ),

    "found_run_dirs": (
        len(
            run_dirs
        )
    ),

    "missing_summaries": (
        len(
            missing
        )
    ),

    "n_audited_valid_events": (
        n_audited
    ),

    "n_detectable": (
        n_pass
    ),

    "n_not_detectable": (
        n_fail
    ),

    "detectable_fraction": (
        pass_fraction
    ),

    "n_duplicate_catalog_rows": (
        n_duplicates
    ),

    "status_counts": {
        str(k): int(v)
        for k, v
        in status_counts.items()
    },

    "failure_reason_counts": {
        str(k): int(v)
        for k, v
        in reason_counts.items()
    },
}


summary_json.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)


# ----------------------------------------------------------------------
# Console report.
# ----------------------------------------------------------------------

print()
print("=" * 100)
print("RESULT")
print("=" * 100)

print(
    f"Audited valid events : {n_audited}"
)

print(
    f"Detectable           : {n_pass}"
)

print(
    f"Not detectable       : {n_fail}"
)

print(
    f"Detectable fraction  : {pass_fraction:.6f}"
)

print(
    f"Duplicate rows       : {n_duplicates}"
)

print(
    f"Missing summaries    : {len(missing)}"
)


print()
print("Status counts:")

for key, value in status_counts.items():
    print(
        f"  {key}: {value}"
    )


print()
print("Failure reasons:")

for key, value in sorted(
    reason_counts.items(),
    key=lambda x: -x[1],
):

    print(
        f"  {key:40s} {value}"
    )


print()
print("Selected metric medians:")

for metric in [
    "detectability_n_peak",
    "detectability_nearest_dt_over_tE",
    "detectability_delta_chi2_per_point",
    "detectability_n_nsigma",
    "detectability_max_snr",
]:

    if metric not in all_df.columns:
        continue

    for label, frame in [
        ("all ", all_df),
        ("pass", detectable),
        ("fail", failures),
    ]:

        values = pd.to_numeric(
            frame[
                metric
            ],
            errors="coerce",
        ).dropna()

        if len(values):

            print(
                f"  {metric:45s} "
                f"{label}: median={values.median():.6g}"
            )


print()
print("Saved:")
print(" ", all_path)
print(" ", detectable_path)
print(" ", rows_path)
print(" ", failure_path)
print(" ", quantiles_path)
print(" ", summary_json)
