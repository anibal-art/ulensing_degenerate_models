#!/usr/bin/env python
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RUN_SEARCH_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/runs"
)

DEFAULT_ANALYSIS_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/partial_analysis"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", required=True)
    p.add_argument("--run-search-root", default=str(DEFAULT_RUN_SEARCH_ROOT))
    p.add_argument("--analysis-root", default=str(DEFAULT_ANALYSIS_ROOT))
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--n-lightcurves-per-category", type=int, default=5)
    p.add_argument("--h1-true-bad-threshold", type=float, default=10.0)
    p.add_argument("--lrt-tol", type=float, default=1.0e-8)
    return p.parse_args()


def infer_chunk_dir(path, run_tag):
    path = Path(path)
    for parent in path.parents:
        if run_tag in parent.name:
            return parent
    return None


def infer_event_dir_from_multifit(path):
    path = Path(path)
    if path.parent.name == "multi_fit":
        return path.parent.parent
    return path.parent


def infer_event_id(path):
    event_dir = infer_event_dir_from_multifit(path)
    m = re.search(r"event_(\d+)", event_dir.name)
    if m:
        return int(m.group(1))
    return np.nan


def infer_chunk_bounds(chunk_name):
    m = re.search(r"rows_(\d+)_(\d+)", chunk_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return np.nan, np.nan


def read_multifit_file(path, run_tag):
    path = Path(path)
    df = pd.read_parquet(path)

    df["multi_fit_file"] = str(path)
    df["event_dir"] = str(infer_event_dir_from_multifit(path))

    chunk_dir = infer_chunk_dir(path, run_tag)
    if chunk_dir is not None:
        df["chunk_dir"] = str(chunk_dir)
        df["chunk_label"] = chunk_dir.name
        start, stop = infer_chunk_bounds(chunk_dir.name)
        df["chunk_row_start"] = start
        df["chunk_row_stop"] = stop
    else:
        df["chunk_dir"] = ""
        df["chunk_label"] = ""
        df["chunk_row_start"] = np.nan
        df["chunk_row_stop"] = np.nan

    if "global_i" not in df.columns:
        df["global_i"] = infer_event_id(path)

    return df


def load_all_multifits(run_search_root, run_tag, max_files=None):
    run_search_root = Path(run_search_root)

    files = sorted(
        run_search_root.glob(f"**/*{run_tag}*/**/multi_fit/*.parquet")
    )

    if max_files is not None:
        files = files[:max_files]

    dfs = []
    bad_files = []

    for k, f in enumerate(files, start=1):
        try:
            dfs.append(read_multifit_file(f, run_tag))
        except Exception as e:
            bad_files.append((str(f), repr(e)))

        if k % 1000 == 0:
            print(f"[read] {k}/{len(files)} files")

    if len(dfs) == 0:
        raise RuntimeError("No readable multi_fit parquet files found.")

    out = pd.concat(dfs, ignore_index=True)

    return out, files, bad_files


def ensure_numeric(df, col, default=np.nan):
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[col]


def add_quality_flags(df, h1_true_bad_threshold=10.0, lrt_tol=1.0e-8):
    df = df.copy()

    for c in [
        "H0_n_data",
        "H1_n_data",
        "true_generator_n_data",
        "H0_chi2",
        "H1_chi2",
        "true_generator_chi2",
        "LRT",
        "delta_chi2_H0_minus_H1",
        "p_value_LRT",
    ]:
        ensure_numeric(df, c)

    if df["LRT"].isna().all():
        df["LRT"] = df["H0_chi2"] - df["H1_chi2"]

    if df["delta_chi2_H0_minus_H1"].isna().all():
        df["delta_chi2_H0_minus_H1"] = df["H0_chi2"] - df["H1_chi2"]

    missing_p = ~np.isfinite(df["p_value_LRT"].to_numpy(dtype=float))
    if missing_p.any():
        lrt = df["LRT"].to_numpy(dtype=float)
        pval = np.exp(-0.5 * np.maximum(lrt, 0.0))
        df.loc[missing_p, "p_value_LRT"] = pval[missing_p]

    df["same_n_data_H0_H1"] = (
        df["H0_n_data"].astype("Int64") == df["H1_n_data"].astype("Int64")
    )

    df["same_n_data_H1_true"] = (
        df["H1_n_data"].astype("Int64")
        == df["true_generator_n_data"].astype("Int64")
    )

    df["same_n_data_all"] = (
        df["same_n_data_H0_H1"].fillna(False)
        & df["same_n_data_H1_true"].fillna(False)
    )

    df["LRT_minus_delta_chi2"] = (
        df["LRT"] - df["delta_chi2_H0_minus_H1"]
    )

    df["LRT_consistent"] = (
        np.isfinite(df["LRT_minus_delta_chi2"])
        & (np.abs(df["LRT_minus_delta_chi2"]) < lrt_tol)
    )

    df["LRT_negative"] = df["LRT"] < -lrt_tol

    df["p_value_finite"] = np.isfinite(df["p_value_LRT"])

    df["H1_minus_true_chi2"] = (
        df["H1_chi2"] - df["true_generator_chi2"]
    )

    df["H1_worse_than_true"] = df["H1_minus_true_chi2"] > 0.0

    df["H1_worse_than_true_gt5"] = df["H1_minus_true_chi2"] > 5.0

    df["H1_worse_than_true_gt10"] = (
        df["H1_minus_true_chi2"] > h1_true_bad_threshold
    )

    df["H1_worse_than_H0"] = (
        df["H1_chi2"] > df["H0_chi2"] + lrt_tol
    )

    if "H0_n_params" in df.columns:
        ensure_numeric(df, "H0_n_params")
        df["H0_dof_from_nparams"] = df["H0_n_data"] - df["H0_n_params"]
        df["H0_chi2_red_from_nparams"] = df["H0_chi2"] / df["H0_dof_from_nparams"]

    if "H1_n_params" in df.columns:
        ensure_numeric(df, "H1_n_params")
        df["H1_dof_from_nparams"] = df["H1_n_data"] - df["H1_n_params"]
        df["H1_chi2_red_from_nparams"] = df["H1_chi2"] / df["H1_dof_from_nparams"]

    for alpha in [0.05, 0.01, 1.0e-3, 1.0e-4, 1.0e-6]:
        label = str(alpha).replace(".", "p").replace("-", "m")
        df[f"detected_p_lt_{label}"] = df["p_value_LRT"] < alpha

    def classify(row):
        reasons = []

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

        if len(reasons) == 0:
            return "ok"

        return "|".join(reasons)

    df["quality_flag"] = df.apply(classify, axis=1)
    df["use_for_detection_statistics"] = df["quality_flag"].eq("ok")

    return df


def finite_values(df, col):
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def savefig(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_hist_lrt(df, outdir):
    x = finite_values(df, "LRT")
    if len(x) == 0:
        return

    plt.figure(figsize=(7, 5))
    plt.hist(x, bins=80)
    plt.xlabel(r"$\Delta \chi^2 = \chi^2_{H0} - \chi^2_{H1}$")
    plt.ylabel("Number of events")
    plt.title("LRT distribution")
    plt.grid(True, alpha=0.3)
    savefig(outdir / "hist_LRT_linear.png")

    xp = x[x > 0]
    if len(xp) > 0:
        plt.figure(figsize=(7, 5))
        plt.hist(np.log10(xp), bins=80)
        plt.xlabel(r"$\log_{10}(\Delta \chi^2)$")
        plt.ylabel("Number of events")
        plt.title("Positive LRT distribution")
        plt.grid(True, alpha=0.3)
        savefig(outdir / "hist_log10_LRT_positive.png")


def plot_hist_pvalue(df, outdir):
    x = finite_values(df, "p_value_LRT")
    if len(x) == 0:
        return

    x = np.clip(x, 1.0e-300, 1.0)

    plt.figure(figsize=(7, 5))
    plt.hist(np.log10(x), bins=80)
    plt.xlabel(r"$\log_{10}(p_{\rm LRT})$")
    plt.ylabel("Number of events")
    plt.title("LRT p-value distribution")
    plt.grid(True, alpha=0.3)
    savefig(outdir / "hist_log10_pvalue.png")

    plt.figure(figsize=(7, 5))
    plt.hist(x, bins=80)
    plt.xlabel(r"$p_{\rm LRT}$")
    plt.ylabel("Number of events")
    plt.title("LRT p-value distribution")
    plt.grid(True, alpha=0.3)
    savefig(outdir / "hist_pvalue_linear.png")


def plot_h1_minus_true(df, outdir):
    x = finite_values(df, "H1_minus_true_chi2")
    if len(x) == 0:
        return

    lo, hi = np.nanpercentile(x, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.nanmin(x), np.nanmax(x)

    plt.figure(figsize=(7, 5))
    plt.hist(x, bins=80)
    plt.xlabel(r"$\chi^2_{H1} - \chi^2_{\rm true}$")
    plt.ylabel("Number of events")
    plt.title("H1 optimizer quality")
    plt.grid(True, alpha=0.3)
    savefig(outdir / "hist_H1_minus_true_chi2_full.png")

    plt.figure(figsize=(7, 5))
    plt.hist(x[(x >= lo) & (x <= hi)], bins=80)
    plt.xlabel(r"$\chi^2_{H1} - \chi^2_{\rm true}$")
    plt.ylabel("Number of events")
    plt.title("H1 optimizer quality, central 98%")
    plt.grid(True, alpha=0.3)
    savefig(outdir / "hist_H1_minus_true_chi2_central98.png")


def plot_chi2_scatter(df, outdir):
    x = finite_values(df, "true_generator_chi2")
    y = finite_values(df, "H1_chi2")

    tmp = df[["true_generator_chi2", "H1_chi2"]].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    if len(tmp) == 0:
        return

    x = tmp["true_generator_chi2"].to_numpy(dtype=float)
    y = tmp["H1_chi2"].to_numpy(dtype=float)

    lo = np.nanpercentile(np.r_[x, y], 1)
    hi = np.nanpercentile(np.r_[x, y], 99)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=7, alpha=0.35)
    plt.plot([lo, hi], [lo, hi], lw=1)
    plt.xlabel(r"$\chi^2_{\rm true}$")
    plt.ylabel(r"$\chi^2_{H1}$")
    plt.title("H1 fit vs true-generator evaluation")
    plt.grid(True, alpha=0.3)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    savefig(outdir / "scatter_H1_chi2_vs_true_chi2.png")


def plot_lrt_vs_h1_minus_true(df, outdir):
    tmp = df[["LRT", "H1_minus_true_chi2", "p_value_LRT"]].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    if len(tmp) == 0:
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(
        tmp["LRT"].to_numpy(dtype=float),
        tmp["H1_minus_true_chi2"].to_numpy(dtype=float),
        s=7,
        alpha=0.35,
    )
    plt.axhline(10.0, lw=1, ls="--")
    plt.xlabel(r"$\Delta \chi^2 = \chi^2_{H0} - \chi^2_{H1}$")
    plt.ylabel(r"$\chi^2_{H1} - \chi^2_{\rm true}$")
    plt.title("LRT vs H1 optimizer quality")
    plt.grid(True, alpha=0.3)
    savefig(outdir / "scatter_LRT_vs_H1_minus_true_chi2.png")


def plot_quality_counts(df, outdir):
    vc = df["quality_flag"].value_counts(dropna=False)

    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(len(vc)), vc.to_numpy())
    plt.xticks(np.arange(len(vc)), vc.index.astype(str), rotation=45, ha="right")
    plt.ylabel("Number of events")
    plt.title("Quality flags")
    plt.grid(True, axis="y", alpha=0.3)
    savefig(outdir / "bar_quality_flags.png")


def plot_detection_fractions(df, outdir):
    alphas = [0.05, 0.01, 1.0e-3, 1.0e-4, 1.0e-6]

    rows = []
    good = df["use_for_detection_statistics"].fillna(False)

    for alpha in alphas:
        mask = good & (df["p_value_LRT"] < alpha)
        rows.append(
            {
                "alpha": alpha,
                "n_good": int(good.sum()),
                "n_detected": int(mask.sum()),
                "fraction": float(mask.sum() / max(good.sum(), 1)),
            }
        )

    tab = pd.DataFrame(rows)

    plt.figure(figsize=(7, 5))
    plt.plot(
        np.arange(len(tab)),
        tab["fraction"].to_numpy(dtype=float),
        marker="o",
    )
    plt.xticks(
        np.arange(len(tab)),
        [f"{a:g}" for a in tab["alpha"]],
        rotation=0,
    )
    plt.xlabel(r"$p_{\rm LRT}$ threshold")
    plt.ylabel("Detection fraction")
    plt.title("Parallax detection fraction")
    plt.grid(True, alpha=0.3)
    savefig(outdir / "detection_fraction_by_pvalue_threshold.png")

    tab.to_csv(outdir / "detection_fraction_by_pvalue_threshold.csv", index=False)


def plot_detection_fraction_vs_n_data(df, outdir):
    good = df["use_for_detection_statistics"].fillna(False)
    tmp = df.loc[good, ["H1_n_data", "p_value_LRT"]].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    if len(tmp) < 20:
        return

    tmp["detected_p_lt_0p01"] = tmp["p_value_LRT"] < 0.01

    try:
        tmp["n_data_bin"] = pd.qcut(
            tmp["H1_n_data"],
            q=min(8, max(2, len(tmp) // 20)),
            duplicates="drop",
        )
    except Exception:
        return

    g = tmp.groupby("n_data_bin", observed=True)
    stat = g.agg(
        n=("p_value_LRT", "size"),
        n_data_mid=("H1_n_data", "median"),
        det_frac=("detected_p_lt_0p01", "mean"),
    ).reset_index()

    plt.figure(figsize=(7, 5))
    plt.plot(
        stat["n_data_mid"].to_numpy(dtype=float),
        stat["det_frac"].to_numpy(dtype=float),
        marker="o",
    )
    plt.xlabel("Number of fitted data points")
    plt.ylabel(r"Fraction with $p_{\rm LRT} < 0.01$")
    plt.title("Detection fraction vs number of points")
    plt.grid(True, alpha=0.3)
    savefig(outdir / "detection_fraction_vs_n_data.png")

    stat.to_csv(outdir / "detection_fraction_vs_n_data.csv", index=False)


def plot_lrt_vs_n_data(df, outdir):
    tmp = df[["H1_n_data", "LRT", "use_for_detection_statistics"]].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    if len(tmp) == 0:
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(
        tmp["H1_n_data"].to_numpy(dtype=float),
        tmp["LRT"].to_numpy(dtype=float),
        s=7,
        alpha=0.35,
    )
    plt.xlabel("Number of fitted data points")
    plt.ylabel(r"$\Delta \chi^2$")
    plt.title("LRT vs number of points")
    plt.grid(True, alpha=0.3)
    savefig(outdir / "scatter_LRT_vs_n_data.png")


def plot_processed_by_global_i(df, outdir):
    if "global_i" not in df.columns:
        return

    tmp = df[["global_i"]].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) == 0:
        return

    x = np.sort(tmp["global_i"].to_numpy(dtype=float))
    y = np.arange(1, len(x) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel("global_i")
    plt.ylabel("Processed events")
    plt.title("Processed events by catalog index")
    plt.grid(True, alpha=0.3)
    savefig(outdir / "processed_events_by_global_i.png")


def select_lightcurve_events(df, outdir, n_per_category=5):
    selections = []

    good = df["use_for_detection_statistics"].fillna(False)

    categories = [
        (
            "strong_detection",
            good & (df["p_value_LRT"] < 1.0e-6),
            ["LRT"],
            [False],
        ),
        (
            "moderate_detection",
            good & (df["p_value_LRT"] >= 1.0e-6) & (df["p_value_LRT"] < 0.01),
            ["p_value_LRT"],
            [True],
        ),
        (
            "marginal_detection",
            good & (df["p_value_LRT"] >= 0.01) & (df["p_value_LRT"] < 0.1),
            ["p_value_LRT"],
            [True],
        ),
        (
            "non_detection",
            good & (df["p_value_LRT"] > 0.5),
            ["p_value_LRT"],
            [False],
        ),
        (
            "bad_H1_local_minimum",
            df["H1_worse_than_true_gt10"].fillna(False),
            ["H1_minus_true_chi2"],
            [False],
        ),
    ]

    for name, mask, sort_cols, ascending in categories:
        sub = df.loc[mask].copy()

        if len(sub) == 0:
            continue

        sub = sub.sort_values(sort_cols, ascending=ascending)
        sub = sub.head(n_per_category).copy()
        sub["lightcurve_category"] = name
        selections.append(sub)

    if len(selections) == 0:
        selected = pd.DataFrame()
    else:
        selected = pd.concat(selections, ignore_index=True)
        selected = selected.drop_duplicates(subset=["multi_fit_file"])

    selected_path = outdir / "selected_lightcurve_events.csv"
    selected.to_csv(selected_path, index=False)

    return selected, selected_path


def make_plots(df, outdir):
    plot_hist_lrt(df, outdir)
    plot_hist_pvalue(df, outdir)
    plot_h1_minus_true(df, outdir)
    plot_chi2_scatter(df, outdir)
    plot_lrt_vs_h1_minus_true(df, outdir)
    plot_quality_counts(df, outdir)
    plot_detection_fractions(df, outdir)
    plot_detection_fraction_vs_n_data(df, outdir)
    plot_lrt_vs_n_data(df, outdir)
    plot_processed_by_global_i(df, outdir)


def json_clean(x):
    if isinstance(x, dict):
        return {str(k): json_clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_clean(v) for v in x]
    if isinstance(x, tuple):
        return [json_clean(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        if np.isfinite(x):
            return float(x)
        return None
    if isinstance(x, np.ndarray):
        return json_clean(x.tolist())
    if pd.isna(x) if not isinstance(x, (str, bytes, dict, list, tuple)) else False:
        return None
    return x


def main():
    args = parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.analysis_root) / f"{args.run_tag}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("run_tag:", args.run_tag)
    print("run_search_root:", args.run_search_root)
    print("outdir:", outdir)

    df, files, bad_files = load_all_multifits(
        args.run_search_root,
        args.run_tag,
        max_files=args.max_files,
    )

    df = add_quality_flags(
        df,
        h1_true_bad_threshold=args.h1_true_bad_threshold,
        lrt_tol=args.lrt_tol,
    )

    merged_csv = outdir / "multi_fit_partial_with_quality_flags.csv"
    merged_parquet = outdir / "multi_fit_partial_with_quality_flags.parquet"

    df.to_csv(merged_csv, index=False)
    df.to_parquet(merged_parquet, index=False)

    make_plots(df, outdir)

    selected, selected_path = select_lightcurve_events(
        df,
        outdir,
        n_per_category=args.n_lightcurves_per_category,
    )

    summary = {
        "run_tag": args.run_tag,
        "analysis_time_utc": timestamp,
        "n_multifit_files_found": len(files),
        "n_multifit_files_read": int(len(df)),
        "n_bad_files": len(bad_files),
        "n_events": int(len(df)),
        "n_quality_ok": int(df["quality_flag"].eq("ok").sum()),
        "quality_counts": df["quality_flag"].value_counts(dropna=False).to_dict(),
        "n_same_n_data_all": int(df["same_n_data_all"].sum()),
        "n_lrt_consistent": int(df["LRT_consistent"].sum()),
        "n_H1_bad_local_minimum": int(df["H1_worse_than_true_gt10"].sum()),
        "n_detected_p_lt_0p05": int((df["p_value_LRT"] < 0.05).sum()),
        "n_detected_p_lt_0p01": int((df["p_value_LRT"] < 0.01).sum()),
        "n_detected_p_lt_1em3": int((df["p_value_LRT"] < 1.0e-3).sum()),
        "n_detected_p_lt_1em6": int((df["p_value_LRT"] < 1.0e-6).sum()),
        "lrt_summary": df["LRT"].describe().to_dict(),
        "p_value_summary": df["p_value_LRT"].describe().to_dict(),
        "H1_minus_true_chi2_summary": df["H1_minus_true_chi2"].describe().to_dict(),
        "merged_csv": str(merged_csv),
        "merged_parquet": str(merged_parquet),
        "selected_lightcurve_events_csv": str(selected_path),
        "plots_dir": str(outdir),
    }

    with open(outdir / "summary.json", "w") as f:
        json.dump(json_clean(summary), f, indent=2)

    if len(bad_files) > 0:
        bad_path = outdir / "bad_multifit_files.txt"
        with open(bad_path, "w") as f:
            for path, err in bad_files:
                f.write(f"{path}\t{err}\n")

    print()
    print("saved:")
    print(merged_csv)
    print(merged_parquet)
    print(outdir / "summary.json")
    print(selected_path)
    print()
    print("quality counts:")
    print(df["quality_flag"].value_counts(dropna=False))
    print()
    print("selected lightcurve events:")
    if len(selected) > 0:
        cols = [
            "lightcurve_category",
            "global_i",
            "simulation_seed",
            "LRT",
            "p_value_LRT",
            "H0_chi2",
            "H1_chi2",
            "true_generator_chi2",
            "H1_minus_true_chi2",
            "event_dir",
            "multi_fit_file",
        ]
        cols = [c for c in cols if c in selected.columns]
        print(selected[cols].to_string(index=False))
    else:
        print("none")


if __name__ == "__main__":
    main()
