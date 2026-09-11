#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def first_existing(row, names):
    for name in names:
        if name in row.index:
            value = row[name]

            if pd.notna(value):
                return value

    return np.nan


def run_dir_from_log(log_file):

    if pd.isna(log_file):
        return None

    text = str(log_file)

    marker = "/logs/"

    if marker not in text:
        return None

    return text.split(
        marker,
        1,
    )[0]


def chunk_range_from_run_dir(run_dir):

    if not run_dir:
        return None, None

    m = re.search(
        r"_rows_(\d+)_(\d+)_w\d+$",
        str(run_dir),
    )

    if m is None:
        return None, None

    return (
        int(m.group(1)),
        int(m.group(2)),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--selection",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--science",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    selection = pd.read_csv(
        args.selection
    )

    science = pd.read_parquet(
        args.science
    )

    required_selection = {
        "catalog_row",
        "selection_reason_bounds",
        "u0",
        "tE",
        "rho",
        "piE",
        "piEN",
        "piEE",
    }

    missing = (
        required_selection
        - set(selection.columns)
    )

    if missing:
        raise RuntimeError(
            "Selection missing columns: "
            + repr(sorted(missing))
        )

    if len(selection) != 100:
        raise RuntimeError(
            f"Expected 100 selected events, found {len(selection)}"
        )

    if selection["catalog_row"].duplicated().any():
        raise RuntimeError(
            "Selection contains duplicate catalog_row values."
        )

    # Keep only the H1 scientific realization when these fields exist.
    if "truth_case" in science.columns:
        science = science[
            science["truth_case"].astype(str)
            == "H1"
        ].copy()

    if "realization" in science.columns:
        science = science[
            pd.to_numeric(
                science["realization"],
                errors="coerce",
            ).fillna(0).astype(int)
            == 0
        ].copy()

    selected_rows = set(
        selection["catalog_row"]
        .astype(int)
    )

    science = science[
        science["catalog_row"]
        .astype(int)
        .isin(selected_rows)
    ].copy()

    counts = (
        science["catalog_row"]
        .astype(int)
        .value_counts()
    )

    bad = counts[
        counts != 1
    ]

    missing_rows = sorted(
        selected_rows
        - set(
            science["catalog_row"]
            .astype(int)
        )
    )

    if missing_rows:
        raise RuntimeError(
            "Selected rows absent from science parquet: "
            + repr(missing_rows)
        )

    if len(bad):
        raise RuntimeError(
            "Science parquet has non-unique selected catalog rows:\n"
            + bad.to_string()
        )

    science = science.copy()
    science["catalog_row"] = (
        science["catalog_row"]
        .astype(int)
    )

    science = science.set_index(
        "catalog_row",
        drop=False,
    )

    rows = []

    for _, sel in selection.iterrows():

        catalog_row = int(
            sel["catalog_row"]
        )

        src = science.loc[
            catalog_row
        ]

        log_file = first_existing(
            src,
            [
                "log_file",
            ],
        )

        run_dir = first_existing(
            src,
            [
                "source_run_dir",
                "run_dir",
            ],
        )

        if pd.isna(run_dir):
            run_dir = run_dir_from_log(
                log_file
            )

        if not run_dir:
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                "cannot determine source run directory"
            )

        row_start = first_existing(
            src,
            [
                "source_row_start",
                "source_row_start_fitcov",
            ],
        )

        row_stop = first_existing(
            src,
            [
                "source_row_stop",
                "source_row_stop_fitcov",
            ],
        )

        parsed_start, parsed_stop = (
            chunk_range_from_run_dir(
                run_dir
            )
        )

        if pd.isna(row_start):
            row_start = parsed_start

        if pd.isna(row_stop):
            row_stop = parsed_stop

        if pd.isna(row_start) or pd.isna(row_stop):
            raise RuntimeError(
                f"catalog_row={catalog_row}: "
                "cannot determine source chunk range"
            )

        row_start = int(row_start)
        row_stop = int(row_stop)

        archive = (
            Path(str(run_dir))
            / "artifacts"
            / f"artifacts_rows_{row_start}_{row_stop}.tar"
        )

        h0_file = first_existing(
            src,
            ["H0_fit_file"],
        )

        h1_file = first_existing(
            src,
            ["H1_fit_file"],
        )

        multifit = first_existing(
            src,
            ["multi_fit_summary_path"],
        )

        rows.append(
            {
                # Required by the audit runner.
                "sample":
                    "extreme100",

                "catalog_row":
                    catalog_row,

                # Compatibility field only.
                "matched_tail_catalog_row":
                    catalog_row,

                # Why this event entered the validation set.
                "selection_reason_bounds":
                    str(
                        sel[
                            "selection_reason_bounds"
                        ]
                    ),

                # Frozen truth summaries for audit/readability.
                "true_u0":
                    float(sel["u0"]),

                "true_tE":
                    float(sel["tE"]),

                "true_rho":
                    float(sel["rho"]),

                "true_piE":
                    float(sel["piE"]),

                "true_piEN":
                    float(sel["piEN"]),

                "true_piEE":
                    float(sel["piEE"]),

                # Production provenance.
                "source_run_tag":
                    first_existing(
                        src,
                        ["source_run_tag"],
                    ),

                "source_row_start":
                    row_start,

                "source_row_stop":
                    row_stop,

                "source_run_dir":
                    str(run_dir),

                "source_archive":
                    str(archive),

                "H0_fit_file":
                    h0_file,

                "H1_fit_file":
                    h1_file,

                "multi_fit_summary_path":
                    multifit,

                "log_file":
                    log_file,
            }
        )

    out = pd.DataFrame(
        rows
    ).sort_values(
        "catalog_row"
    ).reset_index(
        drop=True
    )

    if len(out) != 100:
        raise RuntimeError(
            f"Final manifest has {len(out)} rows, expected 100"
        )

    if out["catalog_row"].duplicated().any():
        raise RuntimeError(
            "Final manifest contains duplicate catalog_row values."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        args.output,
        index=False,
    )

    print("=" * 80)
    print("EXTREME-100 EXPLICIT MANIFEST")
    print("=" * 80)
    print("saved =", args.output)
    print("N =", len(out))
    print(
        "N unique catalog_row =",
        out["catalog_row"].nunique(),
    )
    print(
        "N source runs =",
        out["source_run_dir"].nunique(),
    )
    print(
        "N source archives =",
        out["source_archive"].nunique(),
    )

    print()
    print("selection reasons:")
    print(
        out["selection_reason_bounds"]
        .value_counts()
        .to_string()
    )

    print()
    print("truth ranges:")

    for c in [
        "true_u0",
        "true_tE",
        "true_rho",
        "true_piE",
        "true_piEN",
        "true_piEE",
    ]:
        x = pd.to_numeric(
            out[c],
            errors="coerce",
        )

        print(
            f"{c:12s}"
            f" min={x.min():.9g}"
            f" median={x.median():.9g}"
            f" max={x.max():.9g}"
        )

    print()
    print("first rows:")
    print(
        out[
            [
                "catalog_row",
                "selection_reason_bounds",
                "source_row_start",
                "source_row_stop",
                "source_run_tag",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
