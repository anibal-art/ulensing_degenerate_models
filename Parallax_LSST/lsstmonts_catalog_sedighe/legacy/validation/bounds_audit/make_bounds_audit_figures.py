#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def build_parser():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        type=Path,
        default=(
            HERE
            / "results"
            / "refit_34_summary.csv"
        ),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=(
            HERE
            / "figures"
        ),
    )

    return p


def save(
    fig,
    output_dir,
    stem,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = (
        output_dir
        / f"{stem}.png"
    )

    pdf = (
        output_dir
        / f"{stem}.pdf"
    )

    fig.tight_layout()

    fig.savefig(
        png,
        dpi=220,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "saved:",
        png,
    )

    print(
        "saved:",
        pdf,
    )


def main():

    args = build_parser().parse_args()

    df = pd.read_csv(
        args.input
    )

    # ========================================================
    # 1. OLD VS NEW LRT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.7, 6.1)
    )

    for sample, marker in [
        ("tail", "o"),
        ("control", "s"),
    ]:

        g = df[
            df["sample"]
            == sample
        ]

        ax.scatter(
            g["old_lrt"],
            g["new_lrt"],
            marker=marker,
            s=48,
            alpha=0.8,
            label=sample,
        )

    lo = min(
        df["old_lrt"].min(),
        df["new_lrt"].min(),
    )

    hi = max(
        df["old_lrt"].max(),
        df["new_lrt"].max(),
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1.4,
        label="unchanged",
    )

    ax.axhline(
        9.21,
        linestyle=":",
        linewidth=1.3,
        label=r"$\Delta\chi^2=9.21$",
    )

    ax.axhline(
        1e4,
        linestyle="-.",
        linewidth=1.2,
        label=r"$\Delta\chi^2=10^4$",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        r"Production $\Delta\chi^2_{\rm LRT}$"
    )

    ax.set_ylabel(
        r"Truth-independent-bound $\Delta\chi^2_{\rm LRT}$"
    )

    ax.set_title(
        "Effect of truth-dependent fit bounds on parallax detectability"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        fontsize=8
    )

    save(
        fig,
        args.output_dir,
        "01_old_vs_new_lrt",
    )


    # ========================================================
    # 2. FRACTIONAL LRT REDUCTION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.2, 5.0)
    )

    groups = [
        df.loc[
            df["sample"] == "tail",
            "fractional_lrt_reduction",
        ].to_numpy(),

        df.loc[
            df["sample"] == "control",
            "fractional_lrt_reduction",
        ].to_numpy(),
    ]

    ax.boxplot(
        groups,
        tick_labels=[
            "tail",
            "control",
        ],
        showfliers=True,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.axhline(
        0.5,
        linestyle=":",
        linewidth=1.0,
    )

    ax.set_ylabel(
        r"Fractional reduction "
        r"$1-\Delta\chi^2_{\rm new}/\Delta\chi^2_{\rm old}$"
    )

    ax.set_title(
        "LRT inflation from the original fit bounds is not tail-specific"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    save(
        fig,
        args.output_dir,
        "02_fractional_lrt_reduction_tail_control",
    )


    # ========================================================
    # 3. H0 VS H1 IMPROVEMENT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.6, 5.8)
    )

    eps = 1e-4

    for sample, marker in [
        ("tail", "o"),
        ("control", "s"),
    ]:

        g = df[
            df["sample"]
            == sample
        ]

        ax.scatter(
            g["improve_h0"].clip(
                lower=eps
            ),
            g["improve_h1"].clip(
                lower=eps
            ),
            marker=marker,
            s=48,
            alpha=0.8,
            label=sample,
        )

    hi = max(
        df["improve_h0"].max(),
        df["improve_h1"].max(),
        1.0,
    )

    ax.plot(
        [eps, hi],
        [eps, hi],
        linestyle="--",
        linewidth=1.2,
        label=r"$I_0=I_1$",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        r"$I_0="
        r"\chi^2_{H0,\rm old}-"
        r"\chi^2_{H0,\rm new}$"
    )

    ax.set_ylabel(
        r"$I_1="
        r"\chi^2_{H1,\rm old}-"
        r"\chi^2_{H1,\rm new}$"
    )

    ax.set_title(
        "Wider truth-independent bounds mainly change H0"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    save(
        fig,
        args.output_dir,
        "03_h0_vs_h1_improvement",
    )


    # ========================================================
    # 4. MATCHED TAIL-CONTROL COMPARISON
    # ========================================================

    tails = df[
        df["sample"]
        == "tail"
    ].copy()

    controls = df[
        df["sample"]
        == "control"
    ].copy()

    pairs = tails.merge(
        controls,
        left_on="catalog_row",
        right_on="matched_tail_catalog_row",
        suffixes=(
            "_tail",
            "_control",
        ),
    )

    fig, ax = plt.subplots(
        figsize=(6.1, 6.0)
    )

    ax.scatter(
        pairs[
            "fractional_lrt_reduction_control"
        ],
        pairs[
            "fractional_lrt_reduction_tail"
        ],
        s=52,
        alpha=0.85,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlim(
        -0.03,
        1.03,
    )

    ax.set_ylim(
        -0.03,
        1.03,
    )

    ax.set_xlabel(
        "Matched control: fractional LRT reduction"
    )

    ax.set_ylabel(
        "Covariance tail: fractional LRT reduction"
    )

    ax.set_title(
        "Matched pairs: bound sensitivity occurs in both groups"
    )

    ax.grid(
        alpha=0.25
    )

    save(
        fig,
        args.output_dir,
        "04_matched_tail_control_lrt_reduction",
    )


if __name__ == "__main__":
    main()
