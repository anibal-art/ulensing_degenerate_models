#!/usr/bin/env python3

"""
Figure 03: empirical H1 survival function.

Purpose
-------
Plot

    S(c) = P_H1(Delta chi2_LRT > c)

as a function of the candidate critical value c.

Once the H0-generated calibration provides c_alpha, this same
curve gives the empirical detection power:

    power(alpha) = S(c_alpha)

The denominator is the complete finite H1 population, including
events with Delta chi2 <= 0. Therefore the plotted fraction is
a true population fraction rather than a fraction conditional
on Delta chi2 being positive.
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
        "--n-thresholds",
        type=int,
        default=1000,
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

    # Sort the complete finite population.
    #
    # Negative and zero values remain in the denominator and
    # therefore correctly count as events that do not exceed a
    # positive LRT threshold.
    delta_sorted = np.sort(
        delta
    )

    thresholds = np.logspace(
        np.log10(
            positive.min()
        ),
        np.log10(
            positive.max()
        ),
        args.n_thresholds,
    )

    first_above = np.searchsorted(
        delta_sorted,
        thresholds,
        side="right",
    )

    n_above = (
        len(delta_sorted)
        - first_above
    )

    survival = (
        n_above
        / len(delta_sorted)
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.8, 4.4)
    )

    ax.plot(
        thresholds,
        survival,
    )

    ax.set_xscale("log")

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        r"Candidate threshold $c$ in "
        r"$\Delta\chi^2_{\rm LRT}$"
    )

    ax.set_ylabel(
        r"$P_{H1}(\Delta\chi^2_{\rm LRT} > c)$"
    )



    output = (
        args.output_dir
        / "03_h1_delta_chi2_survival_function"
    )

    save_figure(
        fig,
        output,
    )

    plt.show()


if __name__ == "__main__":
    main()


