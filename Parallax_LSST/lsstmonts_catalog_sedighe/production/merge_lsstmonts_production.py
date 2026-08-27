#!/usr/bin/env python3
"""
Merge multi_fit parquet files from a LSSTMONTS production run-tag.
Also computes quality-control flags from saved H0/H1/true_generator columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_quality_flags(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()

    required = [
        "H0_n_data",
        "H1_n_data",
        "true_generator_n_data",
        "H0_chi2",
        "H1_chi2",
        "true_generator_chi2",
        "LRT",
        "delta_chi2_H0_minus_H1",
        "p_value_LRT",
    ]

    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Missing required columns in multi_fit parquet: {missing}")

    out["same_n_data_H0_H1"] = out["H0_n_data"].astype(int) == out["H1_n_data"].astype(int)
    out["same_n_data_H1_true"] = out["H1_n_data"].astype(int) == out["true_generator_n_data"].astype(int)
    out["same_n_data_all"] = out["same_n_data_H0_H1"] & out["same_n_data_H1_true"]

    out["LRT_minus_delta_chi2"] = out["LRT"].astype(float) - out["delta_chi2_H0_minus_H1"].astype(float)
    out["LRT_consistent"] = np.abs(out["LRT_minus_delta_chi2"]) < 1e-8
    out["LRT_negative"] = out["LRT"].astype(float) < -1e-8
    out["p_value_finite"] = np.isfinite(out["p_value_LRT"].astype(float))

    out["H1_minus_true_chi2"] = out["H1_chi2"].astype(float) - out["true_generator_chi2"].astype(float)
    out["H1_worse_than_true"] = out["H1_minus_true_chi2"] > 0.0
    out["H1_worse_than_true_gt5"] = out["H1_minus_true_chi2"] > 5.0
    out["H1_worse_than_true_gt10"] = out["H1_minus_true_chi2"] > 10.0
    out["H1_worse_than_H0"] = out["H1_chi2"].astype(float) > out["H0_chi2"].astype(float) + 1e-8

    def classify(row: pd.Series) -> str:
        reasons: list[str] = []

        if not bool(row["same_n_data_all"]):
            reasons.append("different_n_data")
        if not bool(row["LRT_consistent"]):
            reasons.append("lrt_inconsistent")
        if bool(row["LRT_negative"]):
            reasons.append("negative_lrt")
        if not bool(row["p_value_finite"]):
            reasons.append("nonfinite_pvalue")
        if bool(row["H1_worse_than_H0"]):
            reasons.append("H1_worse_than_H0")
        if bool(row["H1_worse_than_true_gt10"]):
            reasons.append("H1_bad_local_minimum")

        return "ok" if not reasons else "|".join(reasons)

    out["quality_flag"] = out.apply(classify, axis=1)
    out["use_for_detection_statistics"] = out["quality_flag"].eq("ok")

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        default=(
            "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/runs/"
            "LSSTMONTS_xi_baseline_v5p3p5_hiddenParallax_truthFSPLparallax_multiFit_LRT_smallConservativeBounds"
        ),
        help="Directory containing chunk run directories.",
    )
    parser.add_argument(
        "--run-tag",
        required=True,
        help="Production run tag used in run-name suffix, e.g. prod_multifit_20260826T030000Z.",
    )
    parser.add_argument(
        "--out-dir",
        default="/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/merged",
    )
    parser.add_argument(
        "--glob",
        default="**/multi_fit/*.parquet",
        help="Glob pattern under each matching chunk directory.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_dirs = sorted(
        p for p in run_root.glob(f"{args.run_tag}_rows_*_w*") if p.is_dir()
    )

    if not chunk_dirs:
        raise SystemExit(f"No chunk directories found for run_tag={args.run_tag} under {run_root}")

    print(f"run_root    = {run_root}")
    print(f"run_tag     = {args.run_tag}")
    print(f"chunk dirs  = {len(chunk_dirs)}")

    files: list[Path] = []
    for d in chunk_dirs:
        files.extend(sorted(d.glob(args.glob)))

    files = sorted(files)
    print(f"multi_fit files = {len(files)}")

    if not files:
        raise SystemExit("No multi_fit parquet files found.")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["multi_fit_file"] = str(f)
        df["chunk_dir"] = str(next((p for p in f.parents if p.parent == run_root), ""))
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = build_quality_flags(out)

    # Duplicate detection.
    duplicate_cols = [c for c in ["global_i", "simulation_seed"] if c in out.columns]
    if duplicate_cols:
        out["duplicate_key"] = out.duplicated(duplicate_cols, keep=False)
        n_dup = int(out["duplicate_key"].sum())
    else:
        n_dup = -1

    out_parquet = out_dir / f"multifit_{args.run_tag}.parquet"
    out_csv = out_dir / f"multifit_{args.run_tag}.csv"
    summary_json = out_dir / f"multifit_{args.run_tag}_summary.json"

    out.to_parquet(out_parquet, index=False)
    out.to_csv(out_csv, index=False)

    summary = {
        "run_tag": args.run_tag,
        "run_root": str(run_root),
        "n_chunk_dirs": len(chunk_dirs),
        "n_multi_fit_files": len(files),
        "n_rows": int(len(out)),
        "n_duplicates": n_dup,
        "quality_counts": out["quality_flag"].value_counts(dropna=False).to_dict(),
        "p_value_summary": out["p_value_LRT"].describe().to_dict() if "p_value_LRT" in out.columns else {},
        "output_parquet": str(out_parquet),
        "output_csv": str(out_csv),
    }

    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print("\nquality counts:")
    print(out["quality_flag"].value_counts(dropna=False))

    if "duplicate_key" in out.columns:
        print(f"\nduplicates by {duplicate_cols}: {n_dup}")

    print("\np-value summary:")
    print(out["p_value_LRT"].describe())

    print("\nsaved:")
    print(out_parquet)
    print(out_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
