#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr


def build_parser():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--events",
        type=Path,
        required=True,
        help=(
            "Parquet table containing one row per event with "
            "LRT, pi_E recovery error, and D2_piE."
        ),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--lrt-column",
        default="delta_chi2_parallax",
    )

    p.add_argument(
        "--relative-error-column",
        default="relative_piE_vector_error",
    )

    p.add_argument(
        "--d2-column",
        default="D2_piE",
    )

    return p


def save(
    fig,
    out,
    stem,
):

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        out / f"{stem}.png",
        dpi=220,
        bbox_inches="tight",
    )

    fig.savefig(
        out / f"{stem}.pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def main():

    args = build_parser().parse_args()

    df = pd.read_parquet(
        args.events
    )

    x = pd.to_numeric(
        df[args.lrt_column],
        errors="coerce",
    ).to_numpy()

    err = pd.to_numeric(
        df[args.relative_error_column],
        errors="coerce",
    ).to_numpy()

    d2 = pd.to_numeric(
        df[args.d2_column],
        errors="coerce",
    ).to_numpy()

    good = (
        np.isfinite(x)
        & np.isfinite(err)
        & (x > 0)
        & (err > 0)
    )

    rho_s, pvalue = spearmanr(
        x[good],
        err[good],
    )

    # ========================================================
    # 1. DETECTABILITY VS CHARACTERIZATION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7.0, 5.3)
    )

    ax.scatter(
        x[good],
        err[good],
        s=5,
        alpha=0.10,
        rasterized=True,
    )

    edges = np.logspace(
        np.log10(
            x[good].min()
        ),
        np.log10(
            x[good].max()
        ),
        18,
    )

    centers = []
    medians = []

    for lo, hi in zip(
        edges[:-1],
        edges[1:],
    ):

        m = (
            good
            & (x >= lo)
            & (x < hi)
        )

        if m.sum() < 10:
            continue

        centers.append(
            np.median(
                x[m]
            )
        )

        medians.append(
            np.median(
                err[m]
            )
        )

    ax.plot(
        centers,
        medians,
        marker="o",
        linewidth=2,
        label="Median in LRT bins",
    )

    ax.axvline(
        9.21,
        linestyle="--",
        linewidth=1.3,
        label=r"$\Delta\chi^2=9.21$",
    )

    ax.axhline(
        0.30,
        linestyle=":",
        linewidth=1.2,
        label="30% relative error",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        r"$\Delta\chi^2_{\rm LRT}$"
    )

    ax.set_ylabel(
        r"$|\hat{\boldsymbol{\pi}}_E-"
        r"\boldsymbol{\pi}_{E,\rm true}|/"
        r"\pi_{E,\rm true}$"
    )

    ax.set_title(
        "Parallax detectability vs characterization\n"
        rf"Spearman $\rho={rho_s:.3f}$"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        fontsize=9
    )

    save(
        fig,
        args.output_dir,
        "01_detectability_vs_characterization",
    )


    # ========================================================
    # 2. ERROR VS DETECTION THRESHOLD
    # ========================================================

    thresholds = np.array(
        [
            9.21,
            100,
            1e3,
            1e4,
            1e5,
        ]
    )

    med_error = []
    frac_lt_030 = []
    counts = []

    for threshold in thresholds:

        m = (
            good
            & (x > threshold)
        )

        counts.append(
            int(m.sum())
        )

        if not m.any():

            med_error.append(
                np.nan
            )

            frac_lt_030.append(
                np.nan
            )

            continue

        med_error.append(
            np.median(
                err[m]
            )
        )

        frac_lt_030.append(
            np.mean(
                err[m] < 0.30
            )
        )

    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )

    ax.plot(
        thresholds,
        med_error,
        marker="o",
        linewidth=2,
        label="Median relative error",
    )

    ax.plot(
        thresholds,
        1.0 - np.asarray(
            frac_lt_030
        ),
        marker="s",
        linestyle="--",
        linewidth=1.5,
        label="Fraction with error > 30%",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        r"Minimum $\Delta\chi^2_{\rm LRT}$"
    )

    ax.set_ylabel(
        "Characterization statistic"
    )

    ax.set_title(
        "Characterization improves with detection strength"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    save(
        fig,
        args.output_dir,
        "02_characterization_vs_detection_threshold",
    )


    # ========================================================
    # 3. COVARIANCE COVERAGE VS DETECTION STRENGTH
    # ========================================================

    good_d2 = (
        np.isfinite(x)
        & np.isfinite(d2)
        & (x > 0)
        & (d2 >= 0)
    )

    q99 = chi2.ppf(
        0.99,
        df=2,
    )

    bin_center = []
    coverage = []

    for lo, hi in zip(
        edges[:-1],
        edges[1:],
    ):

        m = (
            good_d2
            & (x >= lo)
            & (x < hi)
        )

        if m.sum() < 10:
            continue

        bin_center.append(
            np.median(
                x[m]
            )
        )

        coverage.append(
            np.mean(
                d2[m] <= q99
            )
        )

    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )

    ax.plot(
        bin_center,
        coverage,
        marker="o",
        linewidth=2,
    )

    ax.axhline(
        0.99,
        linestyle="--",
        linewidth=1.4,
        label="Nominal 99%",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_ylim(
        0.0,
        1.01,
    )

    ax.set_xlabel(
        r"$\Delta\chi^2_{\rm LRT}$"
    )

    ax.set_ylabel(
        "Empirical 99% covariance coverage"
    )

    ax.set_title(
        "Detection strength vs local covariance calibration"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    save(
        fig,
        args.output_dir,
        "03_covariance_coverage_vs_detection",
    )


    # ========================================================
    # 4. FULL CALIBRATION CURVE
    # ========================================================

    d2_clean = d2[
        np.isfinite(d2)
        & (d2 >= 0)
    ]

    nominal = np.linspace(
        0.50,
        0.999,
        200,
    )

    empirical = []

    for p in nominal:

        threshold = chi2.ppf(
            p,
            df=2,
        )

        empirical.append(
            np.mean(
                d2_clean
                <= threshold
            )
        )

    fig, ax = plt.subplots(
        figsize=(6.1, 6.0)
    )

    ax.plot(
        nominal,
        empirical,
        linewidth=2,
        label="Empirical coverage",
    )

    ax.plot(
        nominal,
        nominal,
        linestyle="--",
        linewidth=1.4,
        label="Perfect calibration",
    )

    ax.set_xlabel(
        "Nominal confidence level"
    )

    ax.set_ylabel(
        "Empirical coverage"
    )

    ax.set_xlim(
        0.5,
        1.0,
    )

    ax.set_ylim(
        0.5,
        1.0,
    )

    ax.set_title(
        r"Calibration of local $C_{\pi_E}$"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    save(
        fig,
        args.output_dir,
        "04_covariance_calibration",
    )


    # ========================================================
    # NUMERICAL SUMMARY
    # ========================================================

    print(
        "N =",
        len(df),
    )

    print(
        "Spearman rho =",
        rho_s,
    )

    print(
        "Spearman p =",
        pvalue,
    )

    print(
        "99% empirical coverage =",
        np.mean(
            d2_clean <= q99
        ),
    )

    print()

    for threshold, n, median in zip(
        thresholds,
        counts,
        med_error,
    ):

        print(
            f"LRT > {threshold:g}: "
            f"N={n}, "
            f"median relative piE error={median}"
        )


if __name__ == "__main__":
    main()
