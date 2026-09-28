#!/usr/bin/env python3

"""
Compact the completed H0-generated calibration campaign into one Parquet.

Scientific content
------------------
One row is written for every H0-generated event that passed the approved
pre-fit detectability selection and therefore received the final two-fit LRT:

    H0: FSPL, no parallax
    H1: FSPL + parallax

The Parquet preserves:
    - candidate_rank and catalog_row
    - final H0/H1 chi2 and Delta chi2 LRT
    - final H0/H1 fitted parameters
    - physical covariance matrices
    - optimizer diagnostics
    - peak/sampling diagnostics
    - TRF accounting

It intentionally does NOT copy:
    - H5 light curves
    - full fit_object arrays
    - residual arrays
    - Jacobians
    - RA/Dec
    - truth/catalog information recoverable locally through catalog_row

The complete candidate_status.csv population is audited, so the metadata
records how many of the campaign candidates were:
    materialized
    not_detectable
    invalid
    etc.

No TARGET_N truncation is applied.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# Paths / defaults
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

H1_EXPORTER_PATH = (
    SCRIPT_DIR
    / "export_h1_results_parquet.py"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/"
    "hidden_parallax"
)

DEFAULT_RUN_TAG = (
    "lrt_two_fit_v2_h0_calibration_300k_20260921"
)

DEFAULT_RUN_DIR = (
    DEFAULT_OUTPUT_ROOT
    / "runs"
    / "LSSTMONTS_LRT_TWO_FIT_V2_H0_CALIBRATION"
    / DEFAULT_RUN_TAG
)

DEFAULT_CONFIG = (
    DEFAULT_OUTPUT_ROOT
    / "production_configs"
    / DEFAULT_RUN_TAG
    / "config.json"
)

DEFAULT_OUTPUT = (
    DEFAULT_OUTPUT_ROOT
    / "exports"
    / "h0_lrt_results_20260921.parquet"
)


# ============================================================
# Import shared H1 exporter utilities
# ============================================================

def import_h1_exporter():
    if not H1_EXPORTER_PATH.is_file():
        raise FileNotFoundError(
            f"Missing shared exporter:\n{H1_EXPORTER_PATH}"
        )

    name = "h1_exporter_shared"

    spec = importlib.util.spec_from_file_location(
        name,
        H1_EXPORTER_PATH,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module

    spec.loader.exec_module(
        module
    )

    return module


H1 = import_h1_exporter()


# ============================================================
# Ledger helpers
# ============================================================

def find_column(
    columns,
    candidates,
    label,
):
    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise RuntimeError(
        f"Could not identify {label} column.\n"
        f"Available columns: {list(columns)}"
    )


def load_candidate_ledgers(
    run_dir: Path,
):
    paths = sorted(
        run_dir.rglob(
            "candidate_status.csv"
        )
    )

    if not paths:
        raise RuntimeError(
            f"No candidate_status.csv found under:\n{run_dir}"
        )

    frames = []

    for path in paths:
        df = pd.read_csv(
            path
        )

        rank_col = find_column(
            df.columns,
            [
                "candidate_rank",
                "rank",
                "global_rank",
                "candidate_index",
            ],
            "candidate rank",
        )

        row_col = find_column(
            df.columns,
            [
                "candidate_row",
                "catalog_row",
                "row",
            ],
            "catalog row",
        )

        status_col = find_column(
            df.columns,
            [
                "status",
                "candidate_status",
                "materialization_status",
            ],
            "status",
        )

        out = pd.DataFrame(
            {
                "candidate_rank":
                    pd.to_numeric(
                        df[rank_col],
                        errors="raise",
                    ).astype(np.int64),

                "catalog_row":
                    pd.to_numeric(
                        df[row_col],
                        errors="raise",
                    ).astype(np.int64),

                "candidate_status":
                    df[status_col].astype(str),
            }
        )

        out["ledger_file"] = str(
            path
        )

        frames.append(
            out
        )

    ledger = pd.concat(
        frames,
        ignore_index=True,
    )

    return ledger, paths


# ============================================================
# H0 scientific JSON extraction
# ============================================================

def extract_one_h0(item):
    path_string, candidate_rank = item

    path = Path(
        path_string
    )

    data = H1.load_json(
        path
    )

    catalog_row = H1.as_int(
        data.get(
            "catalog_row"
        )
    )

    if catalog_row is None:
        raise ValueError(
            f"{path}: missing catalog_row"
        )

    if data.get(
        "generating_model"
    ) != "H0":
        raise ValueError(
            f"{path}: generating_model="
            f"{data.get('generating_model')!r}, expected H0"
        )

    if data.get(
        "status"
    ) != "success":
        raise ValueError(
            f"{path}: scientific status="
            f"{data.get('status')!r}"
        )

    final_h0 = data.get(
        "final_h0"
    )

    final_h1 = data.get(
        "final_h1"
    )

    if not isinstance(
        final_h0,
        dict,
    ):
        raise ValueError(
            f"{path}: missing final_h0"
        )

    if not isinstance(
        final_h1,
        dict,
    ):
        raise ValueError(
            f"{path}: missing final_h1"
        )

    row = {
        "candidate_rank":
            int(candidate_rank),

        "candidate_status":
            "materialized",

        "catalog_row":
            catalog_row,

        "production_source":
            DEFAULT_RUN_TAG,

        "policy_name":
            H1.as_text(
                data.get(
                    "policy_name"
                )
            ),

        "generating_model":
            "H0",

        "status":
            "success",

        "prefit_detectability_pass":
            True,

        "chi2_h0":
            H1.as_float(
                data.get(
                    "chi2_h0"
                )
            ),

        "chi2_h1":
            H1.as_float(
                data.get(
                    "chi2_h1"
                )
            ),

        "delta_chi2_lrt":
            H1.as_float(
                data.get(
                    "delta_chi2_lrt"
                )
            ),

        "final_nesting_ok":
            H1.as_bool(
                data.get(
                    "final_nesting_ok"
                )
            ),

        "h0_winner":
            H1.as_text(
                data.get(
                    "h0_winner"
                )
            ),

        "estimator_failure":
            H1.as_text(
                data.get(
                    "estimator_failure"
                )
            ),

        "poor_peak_coverage":
            H1.as_bool(
                data.get(
                    "poor_peak_coverage"
                )
            ),

        "event_fit_wall_s":
            H1.as_float(
                data.get(
                    "event_fit_wall_s"
                )
            ),

        "event_fit_cpu_s":
            H1.as_float(
                data.get(
                    "event_fit_cpu_s"
                )
            ),
    }

    for field in H1.SAMPLING_INT_FIELDS:
        row[field] = H1.as_int(
            data.get(
                field
            )
        )

    for field in H1.SAMPLING_FLOAT_FIELDS:
        row[field] = H1.as_float(
            data.get(
                field
            )
        )

    for field in H1.TRF_COUNT_FIELDS:
        row[field] = H1.as_int(
            data.get(
                field
            )
        )

    n_data = row[
        "n_photometry_points"
    ]

    row[
        "reduced_chi2_h0"
    ] = H1.reduced_chi2(
        row["chi2_h0"],
        n_data,
        len(H1.H0_PARAMETERS),
    )

    row[
        "reduced_chi2_h1"
    ] = H1.reduced_chi2(
        row["chi2_h1"],
        n_data,
        len(H1.H1_PARAMETERS),
    )

    H1.extract_fit(
        row=row,
        prefix="h0",
        fit=final_h0,
        parameters=H1.H0_PARAMETERS,
    )

    H1.extract_fit(
        row=row,
        prefix="h1",
        fit=final_h1,
        parameters=H1.H1_PARAMETERS,
    )

    return row


# ============================================================
# Schema
# ============================================================

def build_h0_schema(
    metadata,
):
    base = H1.build_schema(
        metadata
    )

    fields = [
        pa.field(
            "candidate_rank",
            pa.int64(),
            nullable=False,
        ),
        pa.field(
            "candidate_status",
            pa.string(),
            nullable=False,
        ),
    ]

    fields.extend(
        list(base)
    )

    return pa.schema(
        fields,
        metadata=base.metadata,
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--rank-start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--rank-stop",
        type=int,
        default=300000,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    run_dir = (
        args.run_dir
        .expanduser()
        .resolve()
    )

    config_path = (
        args.config
        .expanduser()
        .resolve()
    )

    output_path = (
        args.output
        .expanduser()
        .resolve()
    )

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory not found:\n{run_dir}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Frozen config not found:\n{config_path}"
        )

    config = H1.load_json(
        config_path
    )

    prefit_config = (
        config
        .get("selection", {})
        .get("prefit_detectability", {})
    )

    # ========================================================
    # Candidate ledger audit
    # ========================================================

    ledger, ledger_paths = (
        load_candidate_ledgers(
            run_dir
        )
    )

    print(
        "============================================================"
    )
    print(
        "H0 calibration candidate audit"
    )
    print(
        "============================================================"
    )

    print(
        "candidate_status files =",
        len(
            ledger_paths
        ),
    )

    print(
        "candidate rows         =",
        len(
            ledger
        ),
    )

    duplicate_ranks = int(
        ledger[
            "candidate_rank"
        ].duplicated().sum()
    )

    duplicate_rows = int(
        ledger[
            "catalog_row"
        ].duplicated().sum()
    )

    print(
        "duplicate ranks        =",
        duplicate_ranks,
    )

    print(
        "duplicate catalog rows =",
        duplicate_rows,
    )

    if duplicate_ranks:
        raise RuntimeError(
            "Duplicate candidate ranks found."
        )

    if duplicate_rows:
        raise RuntimeError(
            "Duplicate catalog rows found."
        )

    expected_ranks = np.arange(
        args.rank_start,
        args.rank_stop,
        dtype=np.int64,
    )

    actual_ranks = np.sort(
        ledger[
            "candidate_rank"
        ].to_numpy(
            dtype=np.int64
        )
    )

    if not np.array_equal(
        expected_ranks,
        actual_ranks,
    ):
        missing = np.setdiff1d(
            expected_ranks,
            actual_ranks,
        )

        extra = np.setdiff1d(
            actual_ranks,
            expected_ranks,
        )

        raise RuntimeError(
            "Candidate-rank coverage mismatch.\n"
            f"Missing first: {missing[:20]}\n"
            f"Extra first: {extra[:20]}"
        )

    status_counts = (
        ledger[
            "candidate_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    print()
    print(
        "candidate status counts:"
    )

    for key, value in status_counts.items():
        print(
            f"  {key:20s} {value}"
        )

    materialized = ledger.loc[
        ledger[
            "candidate_status"
        ] == "materialized"
    ].copy()

    print()
    print(
        "materialized candidates =",
        len(
            materialized
        ),
    )

    # ========================================================
    # Discover scientific JSONs
    # ========================================================

    json_paths = sorted(
        p
        for p in run_dir.rglob(
            "Event_*.json"
        )
        if p.parent.name == "events"
    )

    print(
        "scientific JSONs        =",
        len(
            json_paths
        ),
    )

    if len(
        json_paths
    ) != len(
        materialized
    ):
        raise RuntimeError(
            "Scientific JSON count does not match "
            "materialized candidate count:\n"
            f"JSON={len(json_paths)}, "
            f"materialized={len(materialized)}"
        )

    rank_by_row = dict(
        zip(
            materialized[
                "catalog_row"
            ].astype(int),
            materialized[
                "candidate_rank"
            ].astype(int),
        )
    )

    items = []

    seen_rows = set()

    for path in json_paths:
        try:
            catalog_row = int(
                path.stem.split(
                    "_"
                )[-1]
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot infer catalog_row from {path}"
            ) from exc

        if catalog_row not in rank_by_row:
            raise RuntimeError(
                "JSON catalog_row absent from "
                "materialized candidate ledger: "
                f"{catalog_row}"
            )

        if catalog_row in seen_rows:
            raise RuntimeError(
                f"Duplicate Event JSON for row={catalog_row}"
            )

        seen_rows.add(
            catalog_row
        )

        items.append(
            (
                str(path),
                rank_by_row[
                    catalog_row
                ],
            )
        )

    items.sort(
        key=lambda x: x[1]
    )

    missing_json_rows = (
        set(
            rank_by_row
        )
        - seen_rows
    )

    if missing_json_rows:
        raise RuntimeError(
            "Missing JSONs for materialized rows. "
            f"First: {sorted(missing_json_rows)[:20]}"
        )

    # ========================================================
    # Output metadata
    # ========================================================

    metadata = {
        "description":
            (
                "Compact H0-generated LRT calibration "
                "results"
            ),

        "catalog_join_key":
            "catalog_row",

        "candidate_order_key":
            "candidate_rank",

        "generating_model":
            "H0",

        "candidate_rank_start":
            args.rank_start,

        "candidate_rank_stop":
            args.rank_stop,

        "n_candidates":
            len(
                ledger
            ),

        "n_materialized":
            len(
                materialized
            ),

        "candidate_status_counts":
            json.dumps(
                status_counts,
                sort_keys=True,
            ),

        "truth_columns_included":
            "false",

        "covariance_coordinates":
            "physical",

        "h0_covariance_order":
            json.dumps(
                H1.H0_PARAMETERS
            ),

        "h1_covariance_order":
            json.dumps(
                H1.H1_PARAMETERS
            ),

        "prefit_detectability_config":
            json.dumps(
                prefit_config,
                sort_keys=True,
            ),

        "target_n_truncation":
            "none",

        "sample_definition":
            (
                "all materialized events in candidate "
                "ranks [0,300000)"
            ),
    }

    schema = build_h0_schema(
        metadata
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = Path(
        str(
            output_path
        )
        + ".tmp"
    )

    metadata_path = (
        output_path.with_suffix(
            ".metadata.json"
        )
    )

    if temp_path.exists():
        temp_path.unlink()

    writer = pq.ParquetWriter(
        temp_path,
        schema=schema,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "Exporting H0 scientific results"
    )
    print(
        "============================================================"
    )
    print(
        "output =",
        output_path,
    )
    print(
        "workers =",
        args.workers,
    )
    print(
        "events =",
        len(
            items
        ),
    )

    n_written = 0

    executor = None

    try:
        if args.workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=args.workers
            )

        for start in range(
            0,
            len(items),
            args.batch_size,
        ):
            batch = items[
                start:
                start + args.batch_size
            ]

            if executor is None:
                rows = [
                    extract_one_h0(
                        item
                    )
                    for item in batch
                ]
            else:
                rows = list(
                    executor.map(
                        extract_one_h0,
                        batch,
                        chunksize=32,
                    )
                )

            table = pa.Table.from_pylist(
                rows,
                schema=schema,
            )

            writer.write_table(
                table
            )

            n_written += len(
                rows
            )

            print(
                f"written = "
                f"{n_written}/{len(items)} "
                f"({100.0*n_written/len(items):.1f}%)",
                flush=True,
            )

    finally:
        writer.close()

        if executor is not None:
            executor.shutdown(
                wait=True
            )

    if n_written != len(
        materialized
    ):
        raise RuntimeError(
            f"Final row count mismatch: "
            f"{n_written} != {len(materialized)}"
        )

    os.replace(
        temp_path,
        output_path,
    )

    size_bytes = (
        output_path
        .stat()
        .st_size
    )

    summary = {
        "output":
            str(
                output_path
            ),

        "run_dir":
            str(
                run_dir
            ),

        "frozen_config":
            str(
                config_path
            ),

        "candidate_rank_start":
            args.rank_start,

        "candidate_rank_stop":
            args.rank_stop,

        "n_candidates":
            len(
                ledger
            ),

        "n_materialized":
            len(
                materialized
            ),

        "candidate_status_counts":
            status_counts,

        "n_rows_parquet":
            n_written,

        "file_size_bytes":
            size_bytes,

        "file_size_mib":
            size_bytes
            / 1024**2,

        "target_n_truncation":
            False,

        "covariance": {
            "coordinates":
                "physical",

            "h0_order":
                H1.H0_PARAMETERS,

            "h1_order":
                H1.H1_PARAMETERS,

            "stored_elements":
                "upper_triangle",
        },

        "prefit_detectability":
            prefit_config,
    }

    metadata_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "============================================================"
    )
    print(
        "EXPORT COMPLETE"
    )
    print(
        "============================================================"
    )
    print(
        "candidate rows    =",
        len(
            ledger
        ),
    )
    print(
        "materialized H0   =",
        len(
            materialized
        ),
    )
    print(
        "Parquet rows      =",
        n_written,
    )
    print(
        "Parquet           =",
        output_path,
    )
    print(
        "metadata          =",
        metadata_path,
    )
    print(
        "size [MiB]        =",
        f"{size_bytes / 1024**2:.2f}",
    )


if __name__ == "__main__":
    main()
