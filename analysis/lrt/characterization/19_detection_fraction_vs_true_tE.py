#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detection fraction versus TRUE Einstein timescale
for the STRICTLY POSITIVE H1 population.

NO simulation.
NO fitting.

Population
----------
Only events satisfying

    Delta chi2_LRT > 0

are present in the input table.

Within this positive population:

    confused:
        0 < Delta chi2_LRT < c_alpha

    detected:
        Delta chi2_LRT >= c_alpha

with the empirical alpha = 1e-3 threshold

    c_alpha = 13.15982011480088.

Scientific quantity
-------------------

    f_det(tE)
        = N_detected(tE) / N_positive(tE)

where tE is the TRUE Einstein crossing time from the Sedighe catalog.

This is therefore a CONDITIONAL detection fraction within the
Delta-chi2 > 0 population, not the final unconditional population power.

The figure contains one curve with 95% Wilson binomial confidence intervals.
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
    / "18_positive_truth"
    / "h1_positive_truth_characterization.parquet"
)

FIGURE_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "figures"
    / "characterization"
    / "19_detection_fraction_vs_true_tE"
)

RESULT_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "19_detection_fraction_vs_true_tE"
)

FIGURE = (
    FIGURE_DIR
    / "detection_fraction_vs_true_tE.png"
)

TABLE = (
    RESULT_DIR
    / "detection_fraction_vs_true_tE_bins.csv"
)


# ============================================================
# Configuration
# ============================================================

PRIMARY_THRESHOLD = 13.15982011480088

N_BINS = 24

TE_MIN = 2.5
TE_MAX = 500.0

WILSON_Z = 1.959963984540054


# ============================================================
# Wilson interval
# ============================================================

def wilson_interval(
    successes,
    total,
    z=WILSON_Z,
):

    if total <= 0:

        return (
            np.nan,
            np.nan,
        )

    p = successes / total

    z2 = z * z

    denominator = (
        1.0
        + z2 / total
    )

    center = (
        p
        + z2 / (
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
            "delta_chi2_lrt",
            "positive_lrt_class",
            "tE_true_days",
        ],
    )

    T = pd.to_numeric(
        df[
            "delta_chi2_lrt"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    tE = pd.to_numeric(
        df[
            "tE_true_days"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Audits
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(T)
        & (
            T > 0
        )
    ):

        raise RuntimeError(
            "Input contains events outside the strict Delta-chi2 > 0 population."
        )

    detected = (
        T
        >= PRIMARY_THRESHOLD
    )

    confused = (
        (T > 0)
        & (
            T
            < PRIMARY_THRESHOLD
        )
    )

    if np.any(
        detected
        & confused
    ):

        raise RuntimeError(
            "Detected/confused classes overlap."
        )

    if not np.all(
        detected
        | confused
    ):

        raise RuntimeError(
            "Some positive events were not classified."
        )

    finite_tE = (
        np.isfinite(
            tE
        )
        & (
            tE > 0
        )
    )

    print(
        "=" * 90
    )

    print(
        "DETECTION FRACTION VS TRUE tE"
    )

    print(
        "=" * 90
    )

    print(
        "positive events =",
        len(
            df
        ),
    )

    print(
        "detected =",
        int(
            detected.sum()
        ),
    )

    print(
        "confused =",
        int(
            confused.sum()
        ),
    )

    print(
        "finite positive tE =",
        int(
            finite_tE.sum()
        ),
    )

    # ========================================================
    # Logarithmic true-tE bins
    # ========================================================

    edges = np.geomspace(
        TE_MIN,
        TE_MAX,
        N_BINS + 1,
    )

    rows = []

    for i in range(
        N_BINS
    ):

        left = edges[
            i
        ]

        right = edges[
            i + 1
        ]

        if i == N_BINS - 1:

            in_bin = (
                finite_tE
                & (
                    tE
                    >= left
                )
                & (
                    tE
                    <= right
                )
            )

        else:

            in_bin = (
                finite_tE
                & (
                    tE
                    >= left
                )
                & (
                    tE
                    < right
                )
            )

        n_total = int(
            np.sum(
                in_bin
            )
        )

        n_detected = int(
            np.sum(
                in_bin
                & detected
            )
        )

        n_confused = int(
            np.sum(
                in_bin
                & confused
            )
        )

        if n_total == 0:

            f_det = np.nan
            lower = np.nan
            upper = np.nan

        else:

            f_det = (
                n_detected
                / n_total
            )

            lower, upper = (
                wilson_interval(
                    n_detected,
                    n_total,
                )
            )

        rows.append(
            {
                "bin_index":
                    i,

                "tE_left_days":
                    left,

                "tE_right_days":
                    right,

                "tE_center_days":
                    np.sqrt(
                        left
                        * right
                    ),

                "n_positive":
                    n_total,

                "n_detected":
                    n_detected,

                "n_confused":
                    n_confused,

                "detection_fraction":
                    f_det,

                "confused_fraction":
                    (
                        1.0
                        - f_det
                        if np.isfinite(
                            f_det
                        )
                        else np.nan
                    ),

                "wilson95_lower":
                    lower,

                "wilson95_upper":
                    upper,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        TABLE,
        index=False,
    )

    # ========================================================
    # Overall conditional detection fraction
    # ========================================================

    overall_fraction = (
        detected.sum()
        / len(
            df
        )
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6.8, 4.4)
    )

    x = result[
        "tE_center_days"
    ].to_numpy(
        dtype=float
    )

    y = result[
        "detection_fraction"
    ].to_numpy(
        dtype=float
    )

    lower = result[
        "wilson95_lower"
    ].to_numpy(
        dtype=float
    )

    upper = result[
        "wilson95_upper"
    ].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
        & np.isfinite(
            lower
        )
        & np.isfinite(
            upper
        )
    )

    ax.fill_between(
        x[
            valid
        ],
        lower[
            valid
        ],
        upper[
            valid
        ],
        alpha=0.20,
        label="95% Wilson interval",
    )

    ax.plot(
        x[
            valid
        ],
        y[
            valid
        ],
        marker="o",
        markersize=4.0,
        linewidth=1.5,
        label=(
            r"$f_{\rm det}(t_{E,\rm true})$"
        ),
    )

    ax.axhline(
        overall_fraction,
        linestyle="--",
        linewidth=1.0,
        label=(
            "Overall positive-population "
            f"fraction = {overall_fraction:.3f}"
        ),
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        TE_MIN,
        TE_MAX,
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_xlabel(
        r"True $t_E$ [days]"
    )

    ax.set_ylabel(
        (
            r"$P(\Delta\chi^2_{\rm LRT}"
            r"\geq c_{\alpha}\mid"
            r"\Delta\chi^2_{\rm LRT}>0)$"
        )
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

    # ========================================================
    # Stdout
    # ========================================================

    print()

    print(
        "Overall conditional detection fraction =",
        overall_fraction,
    )

    print()

    print(
        result[
            [
                "tE_left_days",
                "tE_right_days",
                "n_positive",
                "n_detected",
                "n_confused",
                "detection_fraction",
                "wilson95_lower",
                "wilson95_upper",
            ]
        ].to_string(
            index=False
        )
    )

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


