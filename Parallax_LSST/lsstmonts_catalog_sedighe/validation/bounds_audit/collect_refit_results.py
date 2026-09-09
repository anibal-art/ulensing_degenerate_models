#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def build_parser():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--work-root",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--output",
        type=Path,
        default=(
            HERE
            / "results"
            / "refit_34_summary.csv"
        ),
    )

    p.add_argument(
        "--summary-output",
        type=Path,
        default=(
            HERE
            / "results"
            / "summary.txt"
        ),
    )

    p.add_argument(
        "--expected-events",
        type=int,
        default=34,
    )

    return p


def main():

    args = build_parser().parse_args()

    work_root = (
        args.work_root
        .expanduser()
        .resolve()
    )

    refits = (
        work_root
        / "refits"
    )

    files = sorted(
        refits.rglob(
            "summary.json"
        )
    )

    rows = []

    for path in files:

        try:

            with open(path) as f:
                row = json.load(f)

            row[
                "summary_path"
            ] = str(path)

            rows.append(
                row
            )

        except Exception as exc:

            print(
                "WARNING:",
                path,
                exc,
            )

    if not rows:
        raise RuntimeError(
            f"No summary.json files found under {refits}"
        )

    df = pd.DataFrame(
        rows
    )

    numeric = [
        "catalog_row",
        "matched_tail_catalog_row",
        "old_h0_chi2",
        "new_h0_chi2",
        "improve_h0",
        "old_h1_chi2",
        "new_h1_chi2",
        "improve_h1",
        "old_lrt",
        "new_lrt",
        "delta_lrt",
        "n_h0_starts",
        "n_h1_anchors",
        "n_h1_starts",
    ]

    for col in numeric:

        if col in df:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    df[
        "lrt_ratio_new_over_old"
    ] = (
        df["new_lrt"]
        / df["old_lrt"]
    )

    df[
        "fractional_lrt_reduction"
    ] = (
        1.0
        - df[
            "lrt_ratio_new_over_old"
        ]
    )

    df = df.sort_values(
        [
            "sample",
            "catalog_row",
        ]
    ).reset_index(
        drop=True
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        args.output,
        index=False,
    )

    n_tail = int(
        (
            df["sample"]
            == "tail"
        ).sum()
    )

    n_control = int(
        (
            df["sample"]
            == "control"
        ).sum()
    )

    lines = []

    lines.append(
        f"N events = {len(df)}"
    )

    lines.append(
        f"tails = {n_tail}"
    )

    lines.append(
        f"controls = {n_control}"
    )

    lines.append("")

    lines.append(
        "median I0 overall = "
        f"{df['improve_h0'].median():.6f}"
    )

    lines.append(
        "median I1 overall = "
        f"{df['improve_h1'].median():.6f}"
    )

    lines.append(
        "median new/old LRT = "
        f"{df['lrt_ratio_new_over_old'].median():.6f}"
    )

    lines.append(
        "median fractional LRT reduction = "
        f"{df['fractional_lrt_reduction'].median():.6f}"
    )

    lines.append(
        "min new LRT = "
        f"{df['new_lrt'].min():.6f}"
    )

    for threshold in [
        9.21,
        1e3,
        1e4,
    ]:

        n = int(
            (
                df["new_lrt"]
                > threshold
            ).sum()
        )

        lines.append(
            "N new LRT > "
            f"{threshold:g} = "
            f"{n}/{len(df)}"
        )

    lines.append("")

    for sample in [
        "tail",
        "control",
    ]:

        g = df[
            df["sample"]
            == sample
        ]

        lines.append(
            f"{sample}: "
            f"median I0={g['improve_h0'].median():.6f}, "
            f"median I1={g['improve_h1'].median():.6f}, "
            "median fractional LRT reduction="
            f"{g['fractional_lrt_reduction'].median():.6f}"
        )

    summary = "\n".join(
        lines
    )

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary_output.write_text(
        summary + "\n"
    )

    print(
        summary
    )

    print()
    print(
        "CSV =",
        args.output,
    )

    if (
        args.expected_events is not None
        and len(df) != args.expected_events
    ):

        raise RuntimeError(
            f"Expected {args.expected_events} events, "
            f"found {len(df)}."
        )


if __name__ == "__main__":
    main()
