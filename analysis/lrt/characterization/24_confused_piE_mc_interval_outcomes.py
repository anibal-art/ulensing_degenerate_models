#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decompose the Monte-Carlo |pi_E| interval coverage failures
for POSITIVE-LRT events confused with FSPL without parallax.

NO simulation.
NO fitting.
NO new Monte Carlo sampling.

For every confused event we classify the TRUE parallax amplitude as:

    below:
        |piE_true| < q16_MC

    covered:
        q16_MC <= |piE_true| <= q84_MC

    above:
        |piE_true| > q84_MC

The purpose is to determine the direction of the severe coverage failure
seen in the confused population.

If the fitted amplitude is systematically biased upward in the weak-parallax
regime, we expect most non-covered events to satisfy

    |piE_true| < q16_MC.

Only the confused positive-LRT population is plotted.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys as _paper_sys
from pathlib import Path as _PaperPath

_PAPER_LRT_DIR = next(
    parent
    for parent in _PaperPath(__file__).resolve().parents
    if parent.name == "lrt"
)
_paper_sys.path.insert(
    0,
    str(_PAPER_LRT_DIR / "plotting"),
)
from paper_style import (
    apply_paper_style,
    label_panels,
    save_pdf_companion,
)

apply_paper_style()


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "22_piE_amplitude_uncertainty_mc_vs_truth"
    / "piE_amplitude_mc_event_intervals.parquet"
)

FIGURE_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "figures"
    / "characterization"
    / "24_confused_piE_mc_interval_outcomes"
)

RESULT_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "24_confused_piE_mc_interval_outcomes"
)

FIGURE = (
    FIGURE_DIR
    / "confused_piE_mc_interval_outcomes.png"
)

TABLE = (
    RESULT_DIR
    / "confused_piE_mc_interval_outcomes_bins.csv"
)


# ============================================================
# Configuration
# ============================================================

N_BINS = 22

PIE_MIN = 0.01
PIE_MAX = 20.0

MIN_COUNT = 30

WILSON_Z = 1.959963984540054


# ============================================================
# Helpers
# ============================================================

def numeric(df, column):

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(
        dtype=float,
    )


def wilson_interval(successes, total, z=WILSON_Z):

    if total <= 0:

        return (
            np.nan,
            np.nan,
        )

    p = (
        successes
        / total
    )

    z2 = (
        z * z
    )

    denominator = (
        1.0
        + z2 / total
    )

    center = (
        p
        + z2
        / (
            2.0
            * total
        )
    ) / denominator

    half_width = (
        z
        / denominator
        * np.sqrt(
            p
            * (
                1.0
                - p
            )
            / total
            + z2
            / (
                4.0
                * total
                * total
            )
        )
    )

    return (
        center
        - half_width,
        center
        + half_width,
    )


# ============================================================
# Main
# ============================================================

def main():

    if not INPUT.exists():

        raise FileNotFoundError(
            f"Input not found:\n{INPUT}"
        )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_parquet(
        INPUT,
        columns=[
            "catalog_row",
            "positive_lrt_class",
            "piE_true_amp",
            "mc_valid_covariance",
            "mc_q16_piE_amp",
            "mc_q84_piE_amp",
        ],
    )

    piE_true = numeric(
        df,
        "piE_true_amp",
    )

    q16 = numeric(
        df,
        "mc_q16_piE_amp",
    )

    q84 = numeric(
        df,
        "mc_q84_piE_amp",
    )

    confused = (
        df[
            "positive_lrt_class"
        ].astype(str).to_numpy()
        == "confused_fspl_no_parallax"
    )

    valid_covariance = (
        df[
            "mc_valid_covariance"
        ]
        .fillna(False)
        .to_numpy(
            dtype=bool
        )
    )

    valid = (
        confused
        & valid_covariance
        & np.isfinite(
            piE_true
        )
        & (
            piE_true > 0
        )
        & np.isfinite(
            q16
        )
        & np.isfinite(
            q84
        )
        & (
            q16 <= q84
        )
    )

    below = (
        valid
        & (
            piE_true
            < q16
        )
    )

    covered = (
        valid
        & (
            piE_true
            >= q16
        )
        & (
            piE_true
            <= q84
        )
    )

    above = (
        valid
        & (
            piE_true
            > q84
        )
    )

    if not np.all(
        (
            below
            | covered
            | above
        )[
            valid
        ]
    ):

        raise RuntimeError(
            "Outcome classification is incomplete."
        )

    # ========================================================
    # Overall summary
    # ========================================================

    n_valid = int(
        valid.sum()
    )

    n_below = int(
        below.sum()
    )

    n_covered = int(
        covered.sum()
    )

    n_above = int(
        above.sum()
    )

    print(
        "=" * 100
    )

    print(
        "CONFUSED EVENTS: TRUE |pi_E| RELATIVE TO MC CENTRAL 68% INTERVAL"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "valid confused events =",
        n_valid,
    )

    print(
        "true below q16 =",
        n_below,
        f"({n_below / n_valid:.6f})",
    )

    print(
        "covered =",
        n_covered,
        f"({n_covered / n_valid:.6f})",
    )

    print(
        "true above q84 =",
        n_above,
        f"({n_above / n_valid:.6f})",
    )

    # ========================================================
    # Bins in true |pi_E|
    # ========================================================

    edges = np.geomspace(
        PIE_MIN,
        PIE_MAX,
        N_BINS + 1,
    )

    rows = []

    for ibin in range(
        N_BINS
    ):

        left = edges[
            ibin
        ]

        right = edges[
            ibin + 1
        ]

        if ibin == N_BINS - 1:

            in_bin = (
                valid
                & (
                    piE_true
                    >= left
                )
                & (
                    piE_true
                    <= right
                )
            )

        else:

            in_bin = (
                valid
                & (
                    piE_true
                    >= left
                )
                & (
                    piE_true
                    < right
                )
            )

        n = int(
            in_bin.sum()
        )

        center = np.sqrt(
            left
            * right
        )

        counts = {
            "below_q16":
                int(
                    np.sum(
                        below
                        & in_bin
                    )
                ),

            "covered":
                int(
                    np.sum(
                        covered
                        & in_bin
                    )
                ),

            "above_q84":
                int(
                    np.sum(
                        above
                        & in_bin
                    )
                ),
        }

        row = {
            "bin_index":
                ibin,

            "piE_true_left":
                left,

            "piE_true_right":
                right,

            "piE_true_center":
                center,

            "n":
                n,

            "shown_in_figure":
                (
                    n >= MIN_COUNT
                ),
        }

        for outcome, count in (
            counts.items()
        ):

            if n:

                fraction = (
                    count
                    / n
                )

                lower, upper = (
                    wilson_interval(
                        count,
                        n,
                    )
                )

            else:

                fraction = np.nan
                lower = np.nan
                upper = np.nan

            row[
                f"{outcome}_count"
            ] = count

            row[
                f"{outcome}_fraction"
            ] = fraction

            row[
                f"{outcome}_wilson95_lower"
            ] = lower

            row[
                f"{outcome}_wilson95_upper"
            ] = upper

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        TABLE,
        index=False,
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.8, 4.4)
    )

    shown = result.loc[
        result[
            "shown_in_figure"
        ]
    ].copy()

    x = shown[
        "piE_true_center"
    ].to_numpy(
        dtype=float
    )

    definitions = [
        (
            "below_q16",
            (
                r"$|\pi_{E,\rm true}| < q_{16}$"
            ),
        ),
        (
            "covered",
            (
                r"$q_{16}\leq|\pi_{E,\rm true}|"
                r"\leq q_{84}$"
            ),
        ),
        (
            "above_q84",
            (
                r"$|\pi_{E,\rm true}| > q_{84}$"
            ),
        ),
    ]

    for key, label in definitions:

        y = shown[
            f"{key}_fraction"
        ].to_numpy(
            dtype=float
        )

        lower = shown[
            f"{key}_wilson95_lower"
        ].to_numpy(
            dtype=float
        )

        upper = shown[
            f"{key}_wilson95_upper"
        ].to_numpy(
            dtype=float
        )

        good = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(lower)
            & np.isfinite(upper)
        )

        line, = ax.plot(
            x[
                good
            ],
            y[
                good
            ],
            marker="o",
            markersize=4.0,
            linewidth=1.5,
            label=label,
        )

        ax.fill_between(
            x[
                good
            ],
            lower[
                good
            ],
            upper[
                good
            ],
            alpha=0.14,
            color=line.get_color(),
        )

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        PIE_MIN,
        PIE_MAX,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        r"True $|\boldsymbol{\pi}_E|$"
    )

    ax.set_ylabel(
        "Fraction of confused events"
    )


    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        fontsize=13,
    )


    fig.savefig(
        FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    save_pdf_companion(fig, FIGURE)

    print()

    print(
        "Saved figure:",
        FIGURE,
    )

    print(
        "Saved bins:",
        TABLE,
    )


if __name__ == "__main__":
    main()


