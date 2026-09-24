#!/usr/bin/env python3

"""
Figure 06: low-LRT H1 population with provisional chi2(df=2)
critical values.

This figure shows where the theoretical critical values for several
false-positive rates fall relative to the weak-parallax H1 population.

The vertical lines are NOT empirically calibrated thresholds.
They will later be replaced by critical values measured directly
from the H0-generated population.
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
    save_figure,
)

from chi2_reference import (
    chi2_df2_critical_value,
)


# ============================================================
# CLI
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
        "--xmin",
        type=float,
        default=-15.0,
    )

    parser.add_argument(
        "--xmax",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=130,
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

    delta = finite_delta_chi2(
        df
    )

    zoom = delta[
        (delta >= args.xmin)
        & (delta <= args.xmax)
    ]

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.8, 4.4)
    )

    ax.hist(
        zoom,
        bins=args.bins,
        range=(
            args.xmin,
            args.xmax,
        ),
    )

    ax.axvline(
        0.0,
        linestyle=":",
        linewidth=1.0,
    )

    alphas = [
        0.05,
        0.01,
        0.001,
        1e-4,
    ]

    for alpha in alphas:
        critical = float(
            chi2_df2_critical_value(
                alpha
            )
        )

        ax.axvline(
            critical,
            linestyle="--",
            linewidth=1.2,
            label=(
                rf"$\alpha={alpha:g}$: "
                rf"$c_\alpha={critical:.2f}$"
            ),
        )

    ax.set_xlabel(
        r"$\Delta\chi^2_{\rm LRT}$"
    )

    ax.set_ylabel(
        "Number of H1 events"
    )


    ax.legend()


    output = (
        args.output_dir
        / "06_h1_low_delta_chi2_with_chi2_df2_thresholds"
    )

    save_figure(
        fig,
        output,
    )

    plt.show()


if __name__ == "__main__":
    main()


