#!/usr/bin/env python3

"""
Figure 07: peak coverage of negative-Delta-chi2 events.

Scientific question
-------------------
H0 is nested in H1, so at the exact global minima:

    chi2_H1 <= chi2_H0

and therefore:

    Delta chi2_LRT = chi2_H0 - chi2_H1 >= 0.

Negative values indicate a numerical/optimization nesting violation.

This analysis tests whether those events are preferentially associated
with poor temporal coverage around the true microlensing peak.

The production diagnostic is:

    nearest_peak_distance_tE
        = min_i |(t_i - t0_true) / tE_true|

and:

    poor_peak_coverage
        = nearest_peak_distance_tE > 0.5

The flag is diagnostic only. It was not used to reject events.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================
# PRESENTATION STYLE: large fonts for projected figures
# ============================================================

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 19,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 13,
        "figure.titlesize": 19,

        "axes.linewidth": 1.0,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,

        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


import numpy as np
import pandas as pd

from common import (
    DEFAULT_DATA_PATH,
    DEFAULT_FIGURE_DIR,
    load_h1_results,
    save_figure,
)


# ============================================================
# Command-line interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    df = load_h1_results(
        args.input
    )

    finite = np.isfinite(
        df["delta_chi2_lrt"]
    )

    d = df.loc[
        finite
    ].copy()

    d["negative_delta"] = (
        d["delta_chi2_lrt"] < 0.0
    )

    # ========================================================
    # Basic counts
    # ========================================================

    n_total = len(d)

    n_negative = int(
        d["negative_delta"].sum()
    )

    n_nonnegative = (
        n_total - n_negative
    )

    poor_negative = int(
        (
            d["negative_delta"]
            & d["poor_peak_coverage"]
        ).sum()
    )

    poor_nonnegative = int(
        (
            ~d["negative_delta"]
            & d["poor_peak_coverage"]
        ).sum()
    )

    frac_poor_negative = (
        poor_negative
        / n_negative
    )

    frac_poor_nonnegative = (
        poor_nonnegative
        / n_nonnegative
    )

    enrichment = (
        frac_poor_negative
        / frac_poor_nonnegative
        if frac_poor_nonnegative > 0
        else np.nan
    )

    print(
        "============================================================"
    )
    print(
        "Negative Delta-chi2 vs peak coverage"
    )
    print(
        "============================================================"
    )

    print(
        f"N total                       = {n_total}"
    )

    print(
        f"N Delta chi2 < 0              = {n_negative}"
    )

    print(
        f"N Delta chi2 >= 0             = {n_nonnegative}"
    )

    print()

    print(
        "poor_peak_coverage among negative:"
    )

    print(
        f"  {poor_negative}/{n_negative} "
        f"= {frac_poor_negative:.6f}"
    )

    print()

    print(
        "poor_peak_coverage among non-negative:"
    )

    print(
        f"  {poor_nonnegative}/{n_nonnegative} "
        f"= {frac_poor_nonnegative:.6f}"
    )

    print()

    print(
        "relative enrichment:"
    )

    print(
        f"  {enrichment:.3f} x"
    )

    # ========================================================
    # Sampling diagnostics
    # ========================================================

    columns = [
        "nearest_peak_distance_tE",
        "n_within_0p25_tE",
        "n_within_0p5_tE",
        "n_within_1_tE",
        "n_left_within_1_tE",
        "n_right_within_1_tE",
    ]

    summary_rows = []

    for negative_flag, label in [
        (True, "delta_chi2_negative"),
        (False, "delta_chi2_nonnegative"),
    ]:
        subset = d.loc[
            d["negative_delta"]
            == negative_flag
        ]

        for column in columns:
            values = subset[
                column
            ].dropna()

            summary_rows.append(
                {
                    "group":
                        label,

                    "variable":
                        column,

                    "n":
                        len(values),

                    "mean":
                        values.mean(),

                    "median":
                        values.median(),

                    "q10":
                        values.quantile(0.10),

                    "q25":
                        values.quantile(0.25),

                    "q75":
                        values.quantile(0.75),

                    "q90":
                        values.quantile(0.90),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        args.output_dir
        / "07_negative_delta_peak_coverage_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(summary.to_string(index=False))

    print()
    print(
        f"Saved: {summary_path}"
    )

    # ========================================================
    # Figure:
    # nearest observation to true peak
    # ========================================================

    negative = d.loc[
        d["negative_delta"],
        "nearest_peak_distance_tE",
    ].dropna().to_numpy()

    nonnegative = d.loc[
        ~d["negative_delta"],
        "nearest_peak_distance_tE",
    ].dropna().to_numpy()

    bins = np.linspace(
        0.0,
        1.0,
        81,
    )

    fig, ax = plt.subplots(
        figsize=(6.8, 4.4)
    )

    ax.hist(
        nonnegative,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        label=(
            rf"$\Delta\chi^2_{{\rm LRT}}\geq0$ "
            f"(N={len(nonnegative):,})"
        ),
    )

    ax.hist(
        negative,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        label=(
            rf"$\Delta\chi^2_{{\rm LRT}}<0$ "
            f"(N={len(negative):,})"
        ),
    )

    ax.axvline(
        0.5,
        linestyle="--",
        linewidth=1.2,
        label=(
            "poor_peak_coverage threshold"
        ),
    )

    ax.set_xlabel(
        r"$\min_i |t_i-t_0|/t_E$"
    )

    ax.set_ylabel(
        "Probability density"
    )


    ax.legend()


    output = (
        args.output_dir
        / "07_negative_delta_peak_coverage"
    )

    save_figure(
        fig,
        output,
    )

    plt.show()


if __name__ == "__main__":
    main()


