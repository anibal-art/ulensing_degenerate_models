#!/usr/bin/env python3

"""
Compact the final H1-generated LRT production into one scientific Parquet.

The export intentionally does NOT copy information that can be recovered
locally from the original LSSTMONTS catalogue through catalog_row.

Kept:
    - catalog_row and production provenance
    - final H0 / H1 parameters
    - final H0 / H1 chi2 and LRT Delta chi2
    - physical covariance matrices (upper triangle)
    - optimizer diagnostics needed to audit pathological fits
    - sampling / peak-coverage diagnostics
    - approved pre-fit detectability PASS flag
    - TRF accounting

Not kept:
    - truth parameters
    - RA / Dec
    - catalogue magnitudes / blending
    - H5 paths
    - residual vectors
    - Jacobians
    - initial guesses
    - nominal_fits
    - full fit_object structures

The exact frozen pre-fit detectability configuration is stored in:
    1. Parquet schema metadata
    2. a companion *.metadata.json file

Important:
    The production JSON does not contain the exact per-event pre-fit
    N>=nsigma, max SNR, or Asimov Delta-chi2 diagnostics. They are
    therefore NOT fabricated here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# NEW BLOCK: defaults
# ============================================================

DEFAULT_OUTPUT_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/"
    "hidden_parallax"
)

RUN_BASE_NAME = "LSSTMONTS_LRT_TWO_FIT_V2"

DEFAULT_SOURCE_TAGS = {
    "main": "lrt_two_fit_v2_h1_full_milliways_20260920",
    "repair_63798_65000":
        "lrt_two_fit_v2_h1_repair_63798_65000_20260921",
    "repair_244132_245000":
        "lrt_two_fit_v2_h1_repair_244132_245000_20260921",
}

DEFAULT_CONFIG_TAG = (
    "lrt_two_fit_v2_h1_full_milliways_20260920"
)

H0_PARAMETERS = [
    "t0",
    "u0",
    "tE",
    "rho",
]

H1_PARAMETERS = [
    "t0",
    "u0",
    "tE",
    "rho",
    "piEN",
    "piEE",
]

SAMPLING_INT_FIELDS = [
    "n_photometry_points",
    "n_within_0p25_tE",
    "n_within_0p5_tE",
    "n_within_1_tE",
    "n_within_2_tE",
    "n_left_within_1_tE",
    "n_right_within_1_tE",
]

SAMPLING_FLOAT_FIELDS = [
    "nearest_peak_distance_tE",
    "poor_peak_coverage_threshold_tE",
]

TRF_COUNT_FIELDS = [
    "n_nominal_trf",
    "n_continuation_trf",
    "n_rescue_trf",
    "n_total_trf",
]


# ============================================================
# NEW BLOCK: JSON loading
# ============================================================

try:
    import orjson

    def load_json(path: Path):
        return orjson.loads(
            path.read_bytes()
        )

except ImportError:

    def load_json(path: Path):
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)


# ============================================================
# NEW BLOCK: scalar helpers
# ============================================================

def as_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value):
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_bool(value):
    if value is None:
        return None

    return bool(value)


def as_text(value):
    if value is None:
        return None

    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def reduced_chi2(
    chi2,
    n_data,
    n_parameters,
):
    chi2 = as_float(chi2)
    n_data = as_int(n_data)

    if (
        chi2 is None
        or n_data is None
        or n_data <= n_parameters
    ):
        return None

    return (
        chi2
        / float(
            n_data - n_parameters
        )
    )


# ============================================================
# NEW BLOCK: fit-bound extraction
# ============================================================

def get_bound_pair(
    fit,
    parameter,
):
    bounds = fit.get(
        "fit_bounds",
        {},
    )

    value = bounds.get(
        parameter,
        None,
    )

    if value is None:
        return None, None

    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
    ):
        return (
            as_float(value[0]),
            as_float(value[1]),
        )

    if isinstance(value, dict):
        bound_type = value.get(
            "type",
            None,
        )

        if bound_type == "center_width":
            center = as_float(
                value.get("center")
            )

            half_width = as_float(
                value.get("half_width")
            )

            if (
                center is not None
                and half_width is not None
            ):
                return (
                    center - half_width,
                    center + half_width,
                )

        if (
            "lower" in value
            and "upper" in value
        ):
            return (
                as_float(value["lower"]),
                as_float(value["upper"]),
            )

    return None, None


# ============================================================
# NEW BLOCK: covariance extraction
# ============================================================

def validate_parameter_order(
    fit,
    expected,
    label,
):
    order = fit.get(
        "initial_guess_parameter_order",
        None,
    )

    if order is None:
        return

    order = list(order)

    if order != list(expected):
        raise ValueError(
            f"{label}: unexpected covariance/fit parameter "
            f"order. expected={expected}, got={order}"
        )


def flatten_covariance(
    row,
    prefix,
    fit,
    parameters,
):
    covariance = np.asarray(
        fit.get(
            "covariance_matrix",
            None,
        ),
        dtype=float,
    )

    expected_shape = (
        len(parameters),
        len(parameters),
    )

    if covariance.shape != expected_shape:
        raise ValueError(
            f"{prefix}: covariance shape "
            f"{covariance.shape} != {expected_shape}"
        )

    row[
        f"{prefix}_covariance_finite"
    ] = bool(
        np.all(
            np.isfinite(covariance)
        )
    )

    row[
        f"{prefix}_covariance_diag_nonnegative"
    ] = bool(
        np.all(
            np.diag(covariance) >= 0.0
        )
    )

    for i, p1 in enumerate(parameters):
        for j in range(i, len(parameters)):
            p2 = parameters[j]

            row[
                f"{prefix}_cov_{p1}_{p2}"
            ] = float(
                covariance[i, j]
            )


# ============================================================
# NEW BLOCK: fit extraction
# ============================================================

def extract_fit(
    row,
    prefix,
    fit,
    parameters,
):
    validate_parameter_order(
        fit,
        parameters,
        prefix,
    )

    row[
        f"{prefix}_status"
    ] = as_text(
        fit.get("status")
    )

    row[
        f"{prefix}_coordinate_mode"
    ] = as_text(
        fit.get("coordinate_mode")
    )

    row[
        f"{prefix}_fit_time"
    ] = as_float(
        fit.get("fit_time")
    )

    for parameter in parameters:
        row[
            f"{prefix}_{parameter}"
        ] = as_float(
            fit.get(parameter)
        )

        lower, upper = get_bound_pair(
            fit,
            parameter,
        )

        row[
            f"{prefix}_bound_{parameter}_min"
        ] = lower

        row[
            f"{prefix}_bound_{parameter}_max"
        ] = upper

    row[
        f"{prefix}_optimizer_success"
    ] = as_bool(
        fit.get(
            "optimizer_success"
        )
    )

    row[
        f"{prefix}_optimizer_status"
    ] = as_int(
        fit.get(
            "optimizer_status"
        )
    )

    row[
        f"{prefix}_optimizer_optimality"
    ] = as_float(
        fit.get(
            "optimizer_optimality"
        )
    )

    row[
        f"{prefix}_optimizer_nfev"
    ] = as_int(
        fit.get(
            "optimizer_nfev"
        )
    )

    row[
        f"{prefix}_optimizer_njev"
    ] = as_int(
        fit.get(
            "optimizer_njev"
        )
    )

    row[
        f"{prefix}_optimizer_n_active_bounds"
    ] = as_int(
        fit.get(
            "optimizer_n_active_bounds"
        )
    )

    row[
        f"{prefix}_optimizer_active_mask"
    ] = as_text(
        fit.get(
            "optimizer_active_mask"
        )
    )

    flatten_covariance(
        row=row,
        prefix=prefix,
        fit=fit,
        parameters=parameters,
    )


# ============================================================
# NEW BLOCK: one-event extraction
# ============================================================

def extract_one(item):
    source_label, path_string = item

    path = Path(
        path_string
    )

    data = load_json(
        path
    )

    catalog_row = as_int(
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
    ) != "H1":
        raise ValueError(
            f"{path}: generating_model is not H1"
        )

    if data.get(
        "status"
    ) != "success":
        raise ValueError(
            f"{path}: scientific status is "
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
        "catalog_row":
            catalog_row,

        "production_source":
            source_label,

        "policy_name":
            as_text(
                data.get(
                    "policy_name"
                )
            ),

        "generating_model":
            "H1",

        "status":
            "success",

        # Every Event_*.json in this final production exists only
        # after the approved pre-fit detectability gate passed.
        "prefit_detectability_pass":
            True,

        "chi2_h0":
            as_float(
                data.get(
                    "chi2_h0"
                )
            ),

        "chi2_h1":
            as_float(
                data.get(
                    "chi2_h1"
                )
            ),

        "delta_chi2_lrt":
            as_float(
                data.get(
                    "delta_chi2_lrt"
                )
            ),

        "final_nesting_ok":
            as_bool(
                data.get(
                    "final_nesting_ok"
                )
            ),

        "h0_winner":
            as_text(
                data.get(
                    "h0_winner"
                )
            ),

        "estimator_failure":
            as_text(
                data.get(
                    "estimator_failure"
                )
            ),

        "poor_peak_coverage":
            as_bool(
                data.get(
                    "poor_peak_coverage"
                )
            ),

        "event_fit_wall_s":
            as_float(
                data.get(
                    "event_fit_wall_s"
                )
            ),

        "event_fit_cpu_s":
            as_float(
                data.get(
                    "event_fit_cpu_s"
                )
            ),
    }

    for field in SAMPLING_INT_FIELDS:
        row[field] = as_int(
            data.get(field)
        )

    for field in SAMPLING_FLOAT_FIELDS:
        row[field] = as_float(
            data.get(field)
        )

    for field in TRF_COUNT_FIELDS:
        row[field] = as_int(
            data.get(field)
        )

    n_photometry_points = row[
        "n_photometry_points"
    ]

    row[
        "reduced_chi2_h0"
    ] = reduced_chi2(
        row["chi2_h0"],
        n_photometry_points,
        len(H0_PARAMETERS),
    )

    row[
        "reduced_chi2_h1"
    ] = reduced_chi2(
        row["chi2_h1"],
        n_photometry_points,
        len(H1_PARAMETERS),
    )

    extract_fit(
        row=row,
        prefix="h0",
        fit=final_h0,
        parameters=H0_PARAMETERS,
    )

    extract_fit(
        row=row,
        prefix="h1",
        fit=final_h1,
        parameters=H1_PARAMETERS,
    )

    return row


# ============================================================
# NEW BLOCK: Parquet schema
# ============================================================

def build_schema(
    parquet_metadata,
):
    fields = [
        pa.field(
            "catalog_row",
            pa.int64(),
            nullable=False,
        ),
        pa.field(
            "production_source",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "policy_name",
            pa.string(),
        ),
        pa.field(
            "generating_model",
            pa.string(),
        ),
        pa.field(
            "status",
            pa.string(),
        ),
        pa.field(
            "prefit_detectability_pass",
            pa.bool_(),
        ),
        pa.field(
            "chi2_h0",
            pa.float64(),
        ),
        pa.field(
            "chi2_h1",
            pa.float64(),
        ),
        pa.field(
            "delta_chi2_lrt",
            pa.float64(),
        ),
        pa.field(
            "reduced_chi2_h0",
            pa.float64(),
        ),
        pa.field(
            "reduced_chi2_h1",
            pa.float64(),
        ),
        pa.field(
            "final_nesting_ok",
            pa.bool_(),
        ),
        pa.field(
            "h0_winner",
            pa.string(),
        ),
        pa.field(
            "estimator_failure",
            pa.string(),
        ),
    ]

    for field in SAMPLING_INT_FIELDS:
        fields.append(
            pa.field(
                field,
                pa.int64(),
            )
        )

    for field in SAMPLING_FLOAT_FIELDS:
        fields.append(
            pa.field(
                field,
                pa.float64(),
            )
        )

    fields.append(
        pa.field(
            "poor_peak_coverage",
            pa.bool_(),
        )
    )

    for field in TRF_COUNT_FIELDS:
        fields.append(
            pa.field(
                field,
                pa.int64(),
            )
        )

    fields.extend(
        [
            pa.field(
                "event_fit_wall_s",
                pa.float64(),
            ),
            pa.field(
                "event_fit_cpu_s",
                pa.float64(),
            ),
        ]
    )

    for prefix, parameters in [
        ("h0", H0_PARAMETERS),
        ("h1", H1_PARAMETERS),
    ]:
        fields.extend(
            [
                pa.field(
                    f"{prefix}_status",
                    pa.string(),
                ),
                pa.field(
                    f"{prefix}_coordinate_mode",
                    pa.string(),
                ),
                pa.field(
                    f"{prefix}_fit_time",
                    pa.float64(),
                ),
            ]
        )

        for parameter in parameters:
            fields.append(
                pa.field(
                    f"{prefix}_{parameter}",
                    pa.float64(),
                )
            )

            fields.append(
                pa.field(
                    f"{prefix}_bound_{parameter}_min",
                    pa.float64(),
                )
            )

            fields.append(
                pa.field(
                    f"{prefix}_bound_{parameter}_max",
                    pa.float64(),
                )
            )

        fields.extend(
            [
                pa.field(
                    f"{prefix}_optimizer_success",
                    pa.bool_(),
                ),
                pa.field(
                    f"{prefix}_optimizer_status",
                    pa.int64(),
                ),
                pa.field(
                    f"{prefix}_optimizer_optimality",
                    pa.float64(),
                ),
                pa.field(
                    f"{prefix}_optimizer_nfev",
                    pa.int64(),
                ),
                pa.field(
                    f"{prefix}_optimizer_njev",
                    pa.int64(),
                ),
                pa.field(
                    f"{prefix}_optimizer_n_active_bounds",
                    pa.int64(),
                ),
                pa.field(
                    f"{prefix}_optimizer_active_mask",
                    pa.string(),
                ),
                pa.field(
                    f"{prefix}_covariance_finite",
                    pa.bool_(),
                ),
                pa.field(
                    f"{prefix}_covariance_diag_nonnegative",
                    pa.bool_(),
                ),
            ]
        )

        for i, p1 in enumerate(parameters):
            for j in range(
                i,
                len(parameters),
            ):
                p2 = parameters[j]

                fields.append(
                    pa.field(
                        f"{prefix}_cov_{p1}_{p2}",
                        pa.float64(),
                    )
                )

    metadata_bytes = {
        str(key).encode():
            str(value).encode()
        for key, value
        in parquet_metadata.items()
    }

    return pa.schema(
        fields,
        metadata=metadata_bytes,
    )


# ============================================================
# NEW BLOCK: source discovery
# ============================================================

EVENT_RE = re.compile(
    r"Event_(\d+)\.json$"
)


def event_number(path):
    match = EVENT_RE.search(
        path.name
    )

    if match is None:
        raise ValueError(
            f"Unexpected event filename: {path}"
        )

    return int(
        match.group(1)
    )


def discover_event_files(
    source_dir,
):
    files = [
        p
        for p in source_dir.rglob(
            "Event_*.json"
        )
        if p.parent.name == "events"
    ]

    files.sort(
        key=event_number
    )

    return files


# ============================================================
# NEW BLOCK: CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help=(
            "Override default production sources. "
            "May be repeated."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Frozen H1 production config. "
            "Used only to preserve the approved "
            "detectability definition."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional smoke-test limit.",
    )

    parser.add_argument(
        "--expect-events",
        type=int,
        default=None,
        help=(
            "Fail if final number of unique "
            "scientific events differs."
        ),
    )

    return parser.parse_args()


def resolve_sources(
    args,
):
    if args.source:
        sources = []

        for spec in args.source:
            if "=" not in spec:
                raise ValueError(
                    "--source must be LABEL=PATH"
                )

            label, path = spec.split(
                "=",
                1,
            )

            sources.append(
                (
                    label.strip(),
                    Path(path).expanduser(),
                )
            )

        return sources

    run_root = (
        args.output_root
        / "runs"
        / RUN_BASE_NAME
    )

    return [
        (
            label,
            run_root / tag,
        )
        for label, tag
        in DEFAULT_SOURCE_TAGS.items()
    ]


# ============================================================
# NEW BLOCK: main export
# ============================================================

def main():
    args = parse_args()

    if args.workers < 1:
        raise ValueError(
            "--workers must be >= 1"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    sources = resolve_sources(
        args
    )

    for label, source_dir in sources:
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Missing source {label}: "
                f"{source_dir}"
            )

    if args.config is None:
        config_path = (
            args.output_root
            / "production_configs"
            / DEFAULT_CONFIG_TAG
            / "config.json"
        )
    else:
        config_path = args.config

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen config: "
            f"{config_path}"
        )

    config = load_json(
        config_path
    )

    prefit_config = (
        config
        .get("selection", {})
        .get("prefit_detectability", {})
    )

    if args.output is None:
        output_path = (
            args.output_root
            / "exports"
            / "h1_lrt_results_20260920.parquet"
        )
    else:
        output_path = args.output

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        str(output_path)
        + ".tmp"
    )

    metadata_path = (
        output_path.with_suffix(
            ".metadata.json"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    parquet_metadata = {
        "description":
            "Compact final H1-generated two-fit LRT results",

        "catalog_join_key":
            "catalog_row",

        "truth_columns_included":
            "false",

        "covariance_coordinates":
            "physical",

        "h0_covariance_order":
            json.dumps(
                H0_PARAMETERS
            ),

        "h1_covariance_order":
            json.dumps(
                H1_PARAMETERS
            ),

        "prefit_detectability_config":
            json.dumps(
                prefit_config,
                sort_keys=True,
            ),

        "prefit_missing_per_event_metrics":
            (
                "n_nsigma,max_snr,"
                "delta_chi2_asimov,"
                "delta_chi2_per_point,"
                "n_bands"
            ),

        "sampling_note":
            (
                "n_within_* and left/right counts are "
                "truth-referenced sampling diagnostics "
                "stored by the final LRT policy"
            ),
    }

    schema = build_schema(
        parquet_metadata
    )

    print(
        "============================================================"
    )
    print(
        "H1 compact scientific export"
    )
    print(
        "============================================================"
    )

    print(
        "Frozen config =",
        config_path,
    )

    print(
        "Output        =",
        output_path,
    )

    print(
        "Workers       =",
        args.workers,
    )

    print()
    print(
        "Approved pre-fit detectability config:"
    )
    print(
        json.dumps(
            prefit_config,
            indent=2,
            sort_keys=True,
        )
    )

    source_files = {}

    total_discovered = 0

    remaining = (
        args.max_events
    )

    for label, source_dir in sources:
        files = discover_event_files(
            source_dir
        )

        if remaining is not None:
            take = min(
                len(files),
                remaining,
            )

            files = files[:take]

            remaining -= take

        source_files[label] = files

        total_discovered += len(
            files
        )

        print(
            f"{label:28s} "
            f"{len(files):9d} JSON"
        )

        if (
            remaining is not None
            and remaining <= 0
        ):
            break

    print(
        "TOTAL discovered =",
        total_discovered,
    )

    if total_discovered == 0:
        raise RuntimeError(
            "No Event_*.json files found."
        )

    seen_catalog_rows = set()

    source_written = Counter()

    n_written = 0

    writer = pq.ParquetWriter(
        temporary_path,
        schema=schema,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
    )

    executor = None

    try:
        if args.workers > 1:
            executor = (
                ProcessPoolExecutor(
                    max_workers=args.workers
                )
            )

        for label, _ in sources:
            files = source_files.get(
                label,
                [],
            )

            if not files:
                continue

            for start in range(
                0,
                len(files),
                args.batch_size,
            ):
                batch_paths = files[
                    start:
                    start + args.batch_size
                ]

                items = [
                    (
                        label,
                        str(path),
                    )
                    for path in batch_paths
                ]

                if executor is None:
                    rows = [
                        extract_one(item)
                        for item in items
                    ]
                else:
                    rows = list(
                        executor.map(
                            extract_one,
                            items,
                            chunksize=32,
                        )
                    )

                for row in rows:
                    catalog_row = row[
                        "catalog_row"
                    ]

                    if (
                        catalog_row
                        in seen_catalog_rows
                    ):
                        raise RuntimeError(
                            "Duplicate catalog_row across "
                            "H1 production sources: "
                            f"{catalog_row}"
                        )

                    seen_catalog_rows.add(
                        catalog_row
                    )

                    source_written[
                        label
                    ] += 1

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

                if (
                    n_written % 10000
                    < len(rows)
                ):
                    print(
                        f"written = {n_written}"
                    )

    finally:
        writer.close()

        if executor is not None:
            executor.shutdown(
                wait=True
            )

    if (
        args.expect_events is not None
        and n_written
        != args.expect_events
    ):
        raise RuntimeError(
            "Unexpected final event count: "
            f"{n_written} != "
            f"{args.expect_events}"
        )

    os.replace(
        temporary_path,
        output_path,
    )

    file_size_bytes = (
        output_path.stat().st_size
    )

    summary = {
        "output":
            str(output_path),

        "n_rows":
            n_written,

        "n_unique_catalog_rows":
            len(
                seen_catalog_rows
            ),

        "catalog_row_min":
            (
                min(seen_catalog_rows)
                if seen_catalog_rows
                else None
            ),

        "catalog_row_max":
            (
                max(seen_catalog_rows)
                if seen_catalog_rows
                else None
            ),

        "source_counts":
            dict(
                source_written
            ),

        "file_size_bytes":
            file_size_bytes,

        "file_size_mib":
            file_size_bytes
            / 1024**2,

        "frozen_config":
            str(config_path),

        "prefit_detectability":
            prefit_config,

        "covariance": {
            "coordinates":
                "physical",

            "h0_order":
                H0_PARAMETERS,

            "h1_order":
                H1_PARAMETERS,

            "stored_elements":
                "upper_triangle",
        },

        "sampling_columns": (
            SAMPLING_INT_FIELDS
            + SAMPLING_FLOAT_FIELDS
            + [
                "poor_peak_coverage"
            ]
        ),

        "not_exported_from_catalog": [
            "truth",
            "RA/Dec",
            "magnitudes",
            "catalog detection flags",
            "blending",
        ],

        "prefit_per_event_metrics_not_available_in_scientific_json": [
            "Nbands",
            "N>=nsigma",
            "max_snr",
            "delta_chi2_asimov",
            "delta_chi2_per_point",
        ],
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
        "rows             =",
        n_written,
    )

    print(
        "unique rows      =",
        len(
            seen_catalog_rows
        ),
    )

    print(
        "source counts    =",
        dict(
            source_written
        ),
    )

    print(
        "Parquet          =",
        output_path,
    )

    print(
        "metadata         =",
        metadata_path,
    )

    print(
        "size [MiB]       =",
        f"{file_size_bytes / 1024**2:.2f}",
    )


if __name__ == "__main__":
    main()
