#!/usr/bin/env python3

"""
Figure 02: low-Delta-chi2 region.

Purpose
-------
Inspect the part of the H1-generated population in which the
parallax and no-parallax FSPL models are comparatively difficult
to distinguish.

No detection threshold is imposed here. The displayed interval
is only a visualization window. The statistically calibrated
critical value will later come from the H0-generated sample.

Negative Delta chi2 values are intentionally retained because
they are useful diagnostics of finite numerical / optimization
effects in the nested-model LRT.
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
        "--xmin",
        type=float,
        default=-20.0,
    )

    parser.add_argument(
        "--xmax",
        type=float,
        default=100.0,
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

    if args.xmax <= args.xmin:
        raise ValueError(
            "--xmax must be greater than --xmin"
        )

    df = load_h1_results(
        args.input
    )

    print_delta_chi2_summary(
        df
    )

    delta = finite_delta_chi2(
        df
    )

    mask = (
        (delta >= args.xmin)
        & (delta <= args.xmax)
    )

    zoom = delta[
        mask
    ]

    print()
    print(
        f"N in [{args.xmin}, {args.xmax}] "
        f"= {len(zoom)}"
    )

    print(
        "Fraction of finite population "
        f"= {len(zoom) / len(delta):.6f}"
    )

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
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel(
        r"$\Delta\chi^2_{\rm LRT}$"
    )

    ax.set_ylabel(
        "Number of events"
    )


    annotation = (
        f"{len(zoom):,} / {len(delta):,} events\n"
        f"{100.0 * len(zoom) / len(delta):.2f}%"
    )



    output = (
        args.output_dir
        / "02_h1_delta_chi2_low_region"
    )

    save_figure(
        fig,
        output,
    )

    plt.show()


if __name__ == "__main__":
    main()


