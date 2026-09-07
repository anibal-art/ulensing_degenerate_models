#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2


DEFAULT_RUN_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/"
    "runs/LSSTMONTS_LRT_population_FAST"
)

DEFAULT_DONE_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/"
    "slurm_done"
)

DEFAULT_ANALYSIS_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/"
    "analysis"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the empirical H0 LRT distribution using only "
            "successfully completed production chunks."
        )
    )

    parser.add_argument(
        "--run-tag",
        required=True,
        help="Production RUN_TAG.",
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )

    parser.add_argument(
        "--done-root",
        type=Path,
        default=DEFAULT_DONE_ROOT,
    )

    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Workers encoded in production chunk directory names.",
    )

    return parser.parse_args()


def completed_chunk_summaries(args):
    done_dir = args.done_root / args.run_tag

    if not done_dir.is_dir():
        raise FileNotFoundError(
            f"DONE directory not found: {done_dir}"
        )

    chunks = []

    for done in done_dir.glob("rows_*.DONE"):
        match = re.fullmatch(
            r"rows_(\d+)_(\d+)\.DONE",
            done.name,
        )

        if match is None:
            continue

        start, stop = map(int, match.groups())

        summary = (
            args.run_root
            / (
                f"{args.run_tag}_rows_"
                f"{start}_{stop}_w{args.workers}"
            )
            / "logs"
            / "run_summary.parquet"
        )

        if not summary.is_file():
            print(
                "WARNING: DONE exists but summary is missing:",
                summary,
            )
            continue

        chunks.append(
            (start, stop, summary)
        )

    chunks.sort(key=lambda item: item[0])

    return chunks


def main():
    args = parse_args()

    output_dir = (
        args.analysis_root
        / args.run_tag
        / "empirical_h0"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = completed_chunk_summaries(args)

    if not chunks:
        raise RuntimeError(
            "No completed chunks with summaries were found."
        )

    print("=" * 72)
    print("EMPIRICAL H0 LRT DISTRIBUTION")
    print("=" * 72)
    print("run_tag       =", args.run_tag)
    print("DONE chunks   =", len(chunks))
    print("output_dir    =", output_dir)

    frames = []

    for start, stop, path in chunks:
        frame = pd.read_parquet(path)

        frame["_chunk_start"] = start
        frame["_chunk_stop"] = stop

        frames.append(frame)

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    print()
    print("logical rows   =", len(df))
    print(
        "physical events =",
        df["catalog_row"].nunique(),
    )

    print()
    print("STATUS")
    print(df["status"].value_counts())

    print()
    print("BY TRUTH")
    print(
        pd.crosstab(
            df["truth_case"],
            df["status"],
        )
    )

    # --------------------------------------------------------
    # Empirical H0 sample
    # --------------------------------------------------------

    h0_all = df[
        df["truth_case"].eq("H0")
    ].copy()

    h0_ok = h0_all[
        h0_all["status"].eq("ok")
    ].copy()

    h0_timeout = h0_all[
        h0_all["status"].eq("fit_timeout")
    ].copy()

    t_column = "lrt_delta_chi2_H0_minus_H1"

    if t_column not in h0_ok.columns:
        raise KeyError(
            f"Missing required column: {t_column}"
        )

    finite = np.isfinite(
        h0_ok[t_column].to_numpy(dtype=float)
    )

    h0 = h0_ok.loc[finite].copy()

    T = h0[t_column].to_numpy(dtype=float)

    if len(T) == 0:
        raise RuntimeError(
            "No finite H0 LRT values available."
        )

    print()
    print("H0 DISTRIBUTION")
    print("H0 ok             =", len(h0_ok))
    print("H0 finite T       =", len(T))
    print("H0 fit_timeout    =", len(h0_timeout))
    print("negative T        =", int(np.sum(T < 0)))
    print("minimum T         =", float(np.min(T)))
    print("median T          =", float(np.median(T)))
    print("maximum T         =", float(np.max(T)))

    if len(h0_timeout) + len(h0_ok) > 0:
        timeout_fraction = (
            len(h0_timeout)
            / (len(h0_timeout) + len(h0_ok))
        )
    else:
        timeout_fraction = np.nan

    print(
        "H0 timeout fraction post-gate =",
        timeout_fraction,
    )

    # --------------------------------------------------------
    # Save minimal empirical null sample
    # --------------------------------------------------------

    sample_columns = [
        col
        for col in [
            "catalog_row",
            "catalog_event_id",
            "truth_case",
            "noise_realization_id",
            "simulation_seed",
            "noise_seed",
            "status",
            "detectability_pass",
            t_column,
            "tE_catalog_days",
            "u0",
            "rho_catalog",
            "n_data_catalog",
            "_chunk_start",
            "_chunk_stop",
        ]
        if col in h0.columns
    ]

    h0[
        sample_columns
    ].to_parquet(
        output_dir / "h0_null_sample.parquet",
        index=False,
    )

    # --------------------------------------------------------
    # Quantiles
    # --------------------------------------------------------

    quantile_levels = np.array(
        [
            0.50,
            0.68,
            0.90,
            0.95,
            0.975,
            0.99,
            0.995,
            0.999,
        ]
    )

    quantiles = pd.DataFrame(
        {
            "quantile": quantile_levels,
            "T_empirical": [
                np.quantile(T, q)
                for q in quantile_levels
            ],
            "T_chi2_df2": [
                chi2.ppf(q, df=2)
                for q in quantile_levels
            ],
        }
    )

    quantiles["difference"] = (
        quantiles["T_empirical"]
        - quantiles["T_chi2_df2"]
    )

    quantiles.to_csv(
        output_dir / "h0_quantiles.csv",
        index=False,
    )

    print()
    print("EMPIRICAL QUANTILES")
    print(
        quantiles.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Critical values
    # --------------------------------------------------------

    alphas = np.array(
        [
            0.10,
            0.05,
            0.01,
            0.001,
        ]
    )

    critical = []

    for alpha in alphas:
        tcrit = np.quantile(
            T,
            1.0 - alpha,
        )

        critical.append(
            {
                "alpha": alpha,
                "quantile": 1.0 - alpha,
                "Tcrit_empirical": tcrit,
                "Tcrit_chi2_df2": chi2.ppf(
                    1.0 - alpha,
                    df=2,
                ),
                "expected_tail_count": (
                    alpha * len(T)
                ),
                "empirical_exceedances": int(
                    np.sum(T > tcrit)
                ),
            }
        )

    critical = pd.DataFrame(critical)

    critical["difference"] = (
        critical["Tcrit_empirical"]
        - critical["Tcrit_chi2_df2"]
    )

    critical.to_csv(
        output_dir / "h0_critical_values.csv",
        index=False,
    )

    print()
    print("CRITICAL VALUES")
    print(
        critical.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    xmax_hist = float(
        np.quantile(T, 0.995)
    )

    xmin_hist = min(
        0.0,
        float(np.min(T)),
    )

    x = np.linspace(
        0.0,
        xmax_hist,
        1000,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        T,
        bins=80,
        density=True,
        alpha=0.6,
        label=f"Empirical H0 (N={len(T)})",
    )

    ax.plot(
        x,
        chi2.pdf(x, df=2),
        linewidth=2,
        label=r"$\chi^2_2$",
    )

    ax.set_xlim(
        xmin_hist,
        xmax_hist,
    )

    ax.set_xlabel(
        r"$T=\chi^2_{H0}-\chi^2_{H1}$"
    )
    ax.set_ylabel(
        "Probability density"
    )

    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "h0_histogram.png",
        dpi=200,
    )

    fig.savefig(
        output_dir / "h0_histogram.pdf",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Survival function
    # --------------------------------------------------------

    Ts = np.sort(T)

    survival = (
        len(Ts)
        - np.arange(len(Ts))
    ) / len(Ts)

    xmax_sf = float(
        np.quantile(T, 0.999)
    )

    x_sf = np.linspace(
        0.0,
        xmax_sf,
        1500,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.semilogy(
        Ts,
        survival,
        label="Empirical H0",
    )

    ax.semilogy(
        x_sf,
        chi2.sf(x_sf, df=2),
        linewidth=2,
        label=r"$\chi^2_2$",
    )

    ax.set_xlim(
        min(0.0, float(np.min(T))),
        xmax_sf,
    )

    ax.set_xlabel(
        r"$T=\Delta\chi^2$"
    )
    ax.set_ylabel(
        r"$P(T_{\rm H0}\geq T)$"
    )

    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "h0_survival.png",
        dpi=200,
    )

    fig.savefig(
        output_dir / "h0_survival.pdf",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Run-level summary
    # --------------------------------------------------------

    summary = {
        "run_tag": args.run_tag,
        "done_chunks": len(chunks),
        "logical_rows": len(df),
        "physical_events": int(
            df["catalog_row"].nunique()
        ),
        "h0_ok": len(h0_ok),
        "h0_finite_T": len(T),
        "h0_fit_timeout": len(h0_timeout),
        "h0_timeout_fraction_post_gate": (
            timeout_fraction
        ),
        "negative_T": int(
            np.sum(T < 0)
        ),
        "T_min": float(
            np.min(T)
        ),
        "T_median": float(
            np.median(T)
        ),
        "T_max": float(
            np.max(T)
        ),
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        output_dir / "h0_run_summary.csv",
        index=False,
    )

    print()
    print("=" * 72)
    print("Saved:")
    for path in sorted(output_dir.iterdir()):
        print(" ", path.name)
    print("=" * 72)


if __name__ == "__main__":
    main()
