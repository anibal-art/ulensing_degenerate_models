#!/usr/bin/env python3

"""
Figure 01: full positive Delta-chi2 distribution.

Purpose
-------
Visualize the full dynamic range of the LRT statistic for the
H1-generated population.

The x axis is logarithmic because Delta chi2 spans many orders
of magnitude.

Negative and zero Delta-chi2 values are not displayed on the
logarithmic x axis, but their numbers are explicitly reported
in the figure and console output.
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

from common import (
    DEFAULT_DATA_PATH,
    DEFAULT_FIGURE_DIR,
    finite_delta_chi2,
    load_h1_results,
    print_delta_chi2_summary,
    save_figure,
)


# Primary empirical H0 threshold at alpha = 1e-3.
PRIMARY_THRESHOLD = 13.15982011480088


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

    parser.add_argument(
        "--bins",
        type=int,
        default=120,
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

    print_delta_chi2_summary(
        df
    )

    delta = finite_delta_chi2(
        df
    )

    positive = delta[
        delta > 0.0
    ]

    if len(positive) == 0:
        raise RuntimeError(
            "No positive Delta chi2 values found."
        )

    bins = np.logspace(
        np.log10(
            positive.min()
        ),
        np.log10(
            positive.max()
        ),
        args.bins + 1,
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.8, 4.4)
    )

    ax.hist(
        positive,
        bins=bins,
    )

    # Primary empirical detection threshold.
    # Red is used deliberately so the operating point remains
    # visible against the blue histogram.
    ax.axvline(
        PRIMARY_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=2.0,
        zorder=5,
        label=(
            r"Primary threshold: "
            r"$\alpha=10^{-3}$, "
            rf"$c_\alpha={PRIMARY_THRESHOLD:.2f}$"
        ),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    # ========================================================
    # NEW BLOCK: final paper notation
    # ========================================================

    ax.set_xlabel(
        r"$\Delta\chi^2_{\rm LRT}$"
    )

    ax.set_ylabel(
        "Number of events"
    )


    n_negative = int(
        np.sum(delta < 0.0)
    )

    n_zero = int(
        np.sum(delta == 0.0)
    )

    annotation = (
        f"N = {len(delta):,}\n"
        f"$\\Delta\\chi^2<0$: {n_negative:,}\n"
        f"$\\Delta\\chi^2=0$: {n_zero:,}"
    )



    output = (
        args.output_dir
        / "01_h1_delta_chi2_full_distribution"
    )

    save_figure(
        fig,
        output,
    )

    plt.show()


if __name__ == "__main__":
    main()


