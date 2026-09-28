#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import (
    chi2,
    kstest,
    cramervonmises,
    norm,
)


RUN_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/"
    "runs/LSSTMONTS_LRT_population_FAST"
)

DONE_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/"
    "slurm_done"
)

ANALYSIS_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/"
    "analysis"
)

T_COL = "lrt_delta_chi2_H0_minus_H1"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--run-tag", required=True)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--run-root", type=Path, default=RUN_ROOT)
    p.add_argument("--done-root", type=Path, default=DONE_ROOT)
    p.add_argument("--analysis-root", type=Path, default=ANALYSIS_ROOT)
    p.add_argument("--n-bins", type=int, default=4)

    return p.parse_args()


def load_done_chunks(args):
    done_dir = args.done_root / args.run_tag

    frames = []
    chunks = []

    for marker in done_dir.glob("rows_*.DONE"):

        m = re.fullmatch(
            r"rows_(\d+)_(\d+)\.DONE",
            marker.name,
        )

        if not m:
            continue

        start, stop = map(int, m.groups())

        summary = (
            args.run_root
            / f"{args.run_tag}_rows_{start}_{stop}_w{args.workers}"
            / "logs"
            / "run_summary.parquet"
        )

        if not summary.is_file():
            print("WARNING missing summary:", summary)
            continue

        df = pd.read_parquet(summary)

        df["_chunk_start"] = start
        df["_chunk_stop"] = stop

        frames.append(df)
        chunks.append((start, stop))

    if not frames:
        raise RuntimeError("No completed chunks found.")

    order = np.argsort(
        [start for start, _ in chunks]
    )

    frames = [frames[i] for i in order]
    chunks = [chunks[i] for i in order]

    return pd.concat(
        frames,
        ignore_index=True,
    ), chunks


def wilson_interval(k, n, confidence=0.95):

    if n == 0:
        return np.nan, np.nan

    z = norm.ppf(
        0.5 + confidence / 2
    )

    p = k / n

    denominator = 1 + z**2 / n

    center = (
        p + z**2 / (2 * n)
    ) / denominator

    half = (
        z
        * np.sqrt(
            p * (1 - p) / n
            + z**2 / (4 * n**2)
        )
        / denominator
    )

    return center - half, center + half


def false_positive_calibration(T):

    rows = []

    for alpha in [
        0.10,
        0.05,
        0.01,
        0.001,
    ]:

        threshold = chi2.ppf(
            1 - alpha,
            df=2,
        )

        n_exceed = int(
            np.sum(T > threshold)
        )

        lo, hi = wilson_interval(
            n_exceed,
            len(T),
        )

        rows.append({
            "alpha_nominal": alpha,
            "Tcrit_chi2_df2": threshold,
            "n_h0": len(T),
            "n_exceed": n_exceed,
            "alpha_empirical": n_exceed / len(T),
            "ci95_low": lo,
            "ci95_high": hi,
            "nominal_inside_ci95": (
                lo <= alpha <= hi
            ),
        })

    return pd.DataFrame(rows)


def find_column(df, candidates):

    columns_lower = {
        c.lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[
                candidate.lower()
            ]

    return None


def stratified_calibration(df, n_bins):

    variables = []

    tE_col = find_column(
        df,
        [
            "tE_catalog_days",
            "tE",
            "tE_true",
            "true_tE",
            "tE_days",
        ],
    )

    ndata_col = find_column(
        df,
        [
            "n_data_catalog",
            "Ndata",
            "n_data",
            "detectability_nobs",
            "n_obs",
            "Nobs",
        ],
    )

    blending_col = find_column(
        df,
        [
            "blending_factor",
            "blend_fraction",
            "f_blending",
            "source_fraction",
            "fs_over_fs_fb",
            "blending",
        ],
    )

    rho_col = find_column(
        df,
        [
            "rho_catalog",
            "rho",
            "rho_true",
            "true_rho",
        ],
    )

    u0_col = find_column(
        df,
        [
            "u0",
            "u0_catalog",
            "u0_true",
            "true_u0",
        ],
    )

    if tE_col:
        variables.append(
            (
                "tE",
                pd.to_numeric(
                    df[tE_col],
                    errors="coerce",
                ),
            )
        )

    if ndata_col:
        variables.append(
            (
                "Ndata",
                pd.to_numeric(
                    df[ndata_col],
                    errors="coerce",
                ),
            )
        )

    if blending_col:
        variables.append(
            (
                "blending",
                pd.to_numeric(
                    df[blending_col],
                    errors="coerce",
                ),
            )
        )

    if rho_col and u0_col:

        rho = pd.to_numeric(
            df[rho_col],
            errors="coerce",
        )

        u0 = pd.to_numeric(
            df[u0_col],
            errors="coerce",
        )

        ratio = (
            rho
            / u0.abs().replace(
                0,
                np.nan,
            )
        )

        variables.append(
            (
                "rho_over_abs_u0",
                ratio,
            )
        )

    output = []

    for name, values in variables:

        work = pd.DataFrame({
            "x": values,
            "T": pd.to_numeric(
                df[T_COL],
                errors="coerce",
            ),
        }).dropna()

        work = work[
            np.isfinite(work["x"])
            & np.isfinite(work["T"])
        ]

        if work["x"].nunique() < 2:
            continue

        try:

            work["bin"] = pd.qcut(
                work["x"],
                q=n_bins,
                duplicates="drop",
            ).astype(str)

        except ValueError:
            continue

        for bin_name, group in work.groupby(
            "bin",
            observed=True,
        ):

            for alpha in [
                0.05,
                0.01,
            ]:

                threshold = chi2.ppf(
                    1 - alpha,
                    df=2,
                )

                k = int(
                    np.sum(
                        group["T"]
                        > threshold
                    )
                )

                lo, hi = wilson_interval(
                    k,
                    len(group),
                )

                output.append({
                    "variable": name,
                    "bin": bin_name,
                    "n": len(group),
                    "x_min": group["x"].min(),
                    "x_max": group["x"].max(),
                    "alpha_nominal": alpha,
                    "alpha_empirical": (
                        k / len(group)
                    ),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "nominal_inside_ci95": (
                        lo <= alpha <= hi
                    ),
                })

    return pd.DataFrame(output)


def main():

    args = parse_args()

    df, chunks = load_done_chunks(args)

    output_dir = (
        args.analysis_root
        / args.run_tag
        / "empirical_h0"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    h0 = df[
        (df["truth_case"] == "H0")
        & (df["status"] == "ok")
    ].copy()

    h0[T_COL] = pd.to_numeric(
        h0[T_COL],
        errors="coerce",
    )

    h0 = h0[
        np.isfinite(
            h0[T_COL]
        )
    ].copy()

    T = h0[T_COL].to_numpy(
        dtype=float
    )

    print("=" * 72)
    print("H0 DIAGNOSTICS")
    print("=" * 72)
    print("DONE chunks =", len(chunks))
    print("H0 finite T =", len(T))
    print(
        "negative T =",
        int(np.sum(T < 0)),
    )

    # ========================================================
    # False positive calibration
    # ========================================================

    calibration = (
        false_positive_calibration(T)
    )

    calibration.to_csv(
        output_dir
        / "h0_false_positive_calibration.csv",
        index=False,
    )

    print()
    print("FALSE POSITIVE CALIBRATION")
    print(
        calibration.to_string(
            index=False
        )
    )

    # ========================================================
    # Goodness of fit
    # ========================================================

    ks = kstest(
        T,
        "chi2",
        args=(2,),
    )

    cvm = cramervonmises(
        T,
        "chi2",
        args=(2,),
    )

    goodness = pd.DataFrame([
        {
            "n": len(T),
            "ks_statistic": ks.statistic,
            "ks_pvalue": ks.pvalue,
            "cvm_statistic": cvm.statistic,
            "cvm_pvalue": cvm.pvalue,
        }
    ])

    goodness.to_csv(
        output_dir
        / "h0_goodness_of_fit.csv",
        index=False,
    )

    print()
    print("GOODNESS OF FIT")
    print(
        goodness.to_string(
            index=False
        )
    )

    # ========================================================
    # QQ plot
    # ========================================================

    T_sorted = np.sort(T)

    probabilities = (
        np.arange(
            1,
            len(T_sorted) + 1,
        )
        - 0.5
    ) / len(T_sorted)

    theoretical = chi2.ppf(
        probabilities,
        df=2,
    )

    limit = max(
        np.quantile(
            T_sorted,
            0.999,
        ),
        np.quantile(
            theoretical,
            0.999,
        ),
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.scatter(
        theoretical,
        T_sorted,
        s=8,
        alpha=0.5,
    )

    ax.plot(
        [0, limit],
        [0, limit],
        linewidth=1.5,
    )

    ax.set_xlim(
        0,
        limit,
    )

    ax.set_ylim(
        0,
        limit,
    )

    ax.set_xlabel(
        r"Theoretical quantiles: $\chi^2_2$"
    )

    ax.set_ylabel(
        "Empirical H0 quantiles"
    )

    ax.grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        output_dir / "h0_qq.png",
        dpi=200,
    )

    fig.savefig(
        output_dir / "h0_qq.pdf",
    )

    plt.close(fig)

    # ========================================================
    # Convergence
    # ========================================================

    ordered = h0.sort_values(
        [
            "_chunk_start",
            "catalog_row",
        ]
    ).reset_index(
        drop=True
    )

    checkpoints = list(
        range(
            1000,
            len(ordered) + 1,
            1000,
        )
    )

    if len(ordered) not in checkpoints:
        checkpoints.append(
            len(ordered)
        )

    convergence_rows = []

    for n in checkpoints:

        Tn = (
            ordered[T_COL]
            .iloc[:n]
            .to_numpy(
                dtype=float
            )
        )

        convergence_rows.append({
            "n_h0": n,
            "q95_empirical": np.quantile(
                Tn,
                0.95,
            ),
            "q99_empirical": np.quantile(
                Tn,
                0.99,
            ),
            "alpha05_empirical": np.mean(
                Tn
                > chi2.ppf(
                    0.95,
                    df=2,
                )
            ),
            "alpha01_empirical": np.mean(
                Tn
                > chi2.ppf(
                    0.99,
                    df=2,
                )
            ),
        })

    convergence = pd.DataFrame(
        convergence_rows
    )

    convergence.to_csv(
        output_dir
        / "h0_convergence.csv",
        index=False,
    )

    # Quantile convergence

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        convergence["n_h0"],
        convergence["q95_empirical"],
        marker="o",
        label="Empirical q95",
    )

    ax.plot(
        convergence["n_h0"],
        convergence["q99_empirical"],
        marker="o",
        label="Empirical q99",
    )

    ax.axhline(
        chi2.ppf(
            0.95,
            df=2,
        ),
        linestyle="--",
        label=r"$\chi^2_2$ q95",
    )

    ax.axhline(
        chi2.ppf(
            0.99,
            df=2,
        ),
        linestyle=":",
        label=r"$\chi^2_2$ q99",
    )

    ax.set_xlabel(
        "Cumulative number of H0 realizations"
    )

    ax.set_ylabel(
        r"$T$ critical value"
    )

    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "h0_quantile_convergence.png",
        dpi=200,
    )

    fig.savefig(
        output_dir
        / "h0_quantile_convergence.pdf"
    )

    plt.close(fig)

    # False-positive convergence

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        convergence["n_h0"],
        convergence[
            "alpha05_empirical"
        ],
        marker="o",
        label=(
            r"Empirical $\alpha$ "
            "at nominal 0.05"
        ),
    )

    ax.plot(
        convergence["n_h0"],
        convergence[
            "alpha01_empirical"
        ],
        marker="o",
        label=(
            r"Empirical $\alpha$ "
            "at nominal 0.01"
        ),
    )

    ax.axhline(
        0.05,
        linestyle="--",
        label="Nominal 0.05",
    )

    ax.axhline(
        0.01,
        linestyle=":",
        label="Nominal 0.01",
    )

    ax.set_xlabel(
        "Cumulative number of H0 realizations"
    )

    ax.set_ylabel(
        "Empirical false-positive rate"
    )

    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "h0_false_positive_convergence.png",
        dpi=200,
    )

    fig.savefig(
        output_dir
        / "h0_false_positive_convergence.pdf"
    )

    plt.close(fig)

    # ========================================================
    # Stratified calibration
    # ========================================================

    stratified = stratified_calibration(
        h0,
        args.n_bins,
    )

    stratified.to_csv(
        output_dir
        / "h0_stratified_calibration.csv",
        index=False,
    )

    if len(stratified):

        print()
        print(
            "STRATIFIED VARIABLES =",
            sorted(
                stratified[
                    "variable"
                ].unique()
            ),
        )

    # ========================================================
    # Select representative events for inspection
    # ========================================================

    targets = [
        ("median", 0.50),
        ("q95", 0.95),
        ("q99", 0.99),
        ("q999", 0.999),
    ]

    selected = []

    for label, q in targets:

        target_T = np.quantile(
            T,
            q,
        )

        idx = (
            h0[T_COL] - target_T
        ).abs().idxmin()

        row = h0.loc[idx]

        selected.append({
            "label": label,
            "target_quantile": q,
            "catalog_row": int(
                row["catalog_row"]
            ),
            "T": float(
                row[T_COL]
            ),
            "chunk_start": int(
                row["_chunk_start"]
            ),
            "chunk_stop": int(
                row["_chunk_stop"]
            ),
        })

    idx = h0[T_COL].idxmax()

    row = h0.loc[idx]

    selected.append({
        "label": "maximum",
        "target_quantile": 1.0,
        "catalog_row": int(
            row["catalog_row"]
        ),
        "T": float(
            row[T_COL]
        ),
        "chunk_start": int(
            row["_chunk_start"]
        ),
        "chunk_stop": int(
            row["_chunk_stop"]
        ),
    })

    selected = pd.DataFrame(
        selected
    ).drop_duplicates(
        "catalog_row"
    )

    selected.to_csv(
        output_dir
        / "selected_h0_events.csv",
        index=False,
    )

    print()
    print("SELECTED H0 EVENTS")
    print(
        selected.to_string(
            index=False
        )
    )

    print()
    print("Saved in:")
    print(output_dir)


if __name__ == "__main__":
    main()
