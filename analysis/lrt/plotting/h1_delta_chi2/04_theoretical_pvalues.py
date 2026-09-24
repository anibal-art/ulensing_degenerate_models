#!/usr/bin/env python3

"""
Figure 04: provisional p-value distribution under chi2(df=2).

The plot uses

    -log10(p)

rather than p itself because the H1 population spans extremely
small p-values.

IMPORTANT
---------
These are NOT the final calibrated p-values.

They assume

    Delta chi2_LRT ~ chi2(df=2) under H0.

They will later be replaced by empirical p-values obtained from the
H0-generated calibration sample.
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
    chi2_df2_minus_log10_pvalue,
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
        "--bins",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--xmax",
        type=float,
        default=20.0,
        help=(
            "Maximum displayed -log10(p). "
            "More significant events are collected "
            "in the final plotting bin."
        ),
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

    minus_log10_p = (
        chi2_df2_minus_log10_pvalue(
            delta
        )
    )

    # Clip only for visualization.
    # The original statistics remain unchanged.
    displayed = np.minimum(
        minus_log10_p,
        args.xmax,
    )

    print(
        "============================================================"
    )
    print(
        "Provisional chi2(df=2) p-values"
    )
    print(
        "============================================================"
    )

    for alpha in [
        0.05,
        0.01,
        0.001,
        1e-4,
        1e-5,
    ]:
        n = int(
            np.sum(
                minus_log10_p
                >= -np.log10(alpha)
            )
        )

        print(
            f"p <= {alpha:g}: "
            f"{n:8d} "
            f"({n / len(delta):.6f})"
        )

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.8, 4.4)
    )

    ax.hist(
        displayed,
        bins=args.bins,
        range=(
            0.0,
            args.xmax,
        ),
    )

    for alpha in [
        0.05,
        0.01,
        0.001,
    ]:
        x = -np.log10(
            alpha
        )

        ax.axvline(
            x,
            linestyle="--",
            linewidth=1.0,
            label=rf"$\alpha={alpha:g}$",
        )

    ax.set_xlabel(
        r"$-\log_{10}p$ "
        r"(provisional $\chi^2_2$ calibration)"
    )

    ax.set_ylabel(
        "Number of H1 events"
    )

    ax.set_yscale(
        "log"
    )


    ax.legend()


    output = (
        args.output_dir
        / "04_h1_theoretical_pvalues_chi2_df2"
    )

    save_figure(
        fig,
        output,
    )

    plt.show()


if __name__ == "__main__":
    main()


