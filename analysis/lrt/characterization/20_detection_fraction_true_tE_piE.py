#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conditional parallax-detection fraction in the
(true tE, true |piE|) plane.

NO simulation.
NO fitting.

Population:
    Delta chi2_LRT > 0

Detected:
    Delta chi2_LRT >= 13.15982011480088

The plotted quantity is

    f_det(tE, piE)
      = N_detected / N_positive

inside each 2D bin.

Cells containing fewer than MIN_COUNT events are masked because their
fractions are too noisy to interpret.

This is a conditional characterization of the strictly positive LRT
population, not the unconditional population power.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    / "20_detection_fraction_true_tE_piE"
)

RESULT_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "20_detection_fraction_true_tE_piE"
)

FIGURE = (
    FIGURE_DIR
    / "detection_fraction_true_tE_piE.png"
)

TABLE = (
    RESULT_DIR
    / "detection_fraction_true_tE_piE_bins.csv"
)


# ============================================================
# Configuration
# ============================================================

PRIMARY_THRESHOLD = 13.15982011480088

N_TE_BINS = 18
N_PIE_BINS = 18

TE_MIN = 2.5
TE_MAX = 500.0

PIE_MIN = 0.01
PIE_MAX = 20.0

MIN_COUNT = 50


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
            "tE_true_days",
            "piE_true_amp",
        ],
    )

    T = pd.to_numeric(
        df["delta_chi2_lrt"],
        errors="coerce",
    ).to_numpy(float)

    tE = pd.to_numeric(
        df["tE_true_days"],
        errors="coerce",
    ).to_numpy(float)

    piE = pd.to_numeric(
        df["piE_true_amp"],
        errors="coerce",
    ).to_numpy(float)

    if not np.all(
        np.isfinite(T)
        & (T > 0)
    ):
        raise RuntimeError(
            "Input is not the strict positive-LRT population."
        )

    detected = (
        T >= PRIMARY_THRESHOLD
    )

    valid = (
        np.isfinite(tE)
        & (tE > 0)
        & np.isfinite(piE)
        & (piE > 0)
    )

    print(
        "=" * 100
    )
    print(
        "DETECTION FRACTION IN TRUE tE -- TRUE piE PLANE"
    )
    print(
        "=" * 100
    )

    print(
        "rows =",
        len(df),
    )

    print(
        "valid =",
        int(valid.sum()),
    )

    print(
        "detected =",
        int(detected.sum()),
    )

    print(
        "confused =",
        int((~detected).sum()),
    )

    # ========================================================
    # Geometric bins
    # ========================================================

    tE_edges = np.geomspace(
        TE_MIN,
        TE_MAX,
        N_TE_BINS + 1,
    )

    piE_edges = np.geomspace(
        PIE_MIN,
        PIE_MAX,
        N_PIE_BINS + 1,
    )

    total, _, _ = np.histogram2d(
        tE[valid],
        piE[valid],
        bins=[
            tE_edges,
            piE_edges,
        ],
    )

    detected_hist, _, _ = np.histogram2d(
        tE[
            valid
            & detected
        ],
        piE[
            valid
            & detected
        ],
        bins=[
            tE_edges,
            piE_edges,
        ],
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        fraction = (
            detected_hist
            / total
        )

    fraction[
        total < MIN_COUNT
    ] = np.nan

    # ========================================================
    # Save long-form table
    # ========================================================

    rows = []

    for i in range(
        N_TE_BINS
    ):

        for j in range(
            N_PIE_BINS
        ):

            n_total = int(
                total[i, j]
            )

            n_detected = int(
                detected_hist[i, j]
            )

            rows.append(
                {
                    "tE_bin":
                        i,

                    "piE_bin":
                        j,

                    "tE_left_days":
                        tE_edges[i],

                    "tE_right_days":
                        tE_edges[i + 1],

                    "tE_center_days":
                        np.sqrt(
                            tE_edges[i]
                            * tE_edges[i + 1]
                        ),

                    "piE_left":
                        piE_edges[j],

                    "piE_right":
                        piE_edges[j + 1],

                    "piE_center":
                        np.sqrt(
                            piE_edges[j]
                            * piE_edges[j + 1]
                        ),

                    "n_positive":
                        n_total,

                    "n_detected":
                        n_detected,

                    "n_confused":
                        n_total
                        - n_detected,

                    "detection_fraction":
                        (
                            n_detected
                            / n_total
                            if n_total > 0
                            else np.nan
                        ),

                    "shown_in_heatmap":
                        (
                            n_total
                            >= MIN_COUNT
                        ),
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        TABLE,
        index=False,
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8.0, 6.1)
    )

    mesh = ax.pcolormesh(
        tE_edges,
        piE_edges,
        fraction.T,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlim(
        TE_MIN,
        TE_MAX,
    )

    ax.set_ylim(
        PIE_MIN,
        PIE_MAX,
    )

    ax.set_xlabel(
        r"True $t_E$ [days]"
    )

    ax.set_ylabel(
        r"True $|\boldsymbol{\pi}_E|$"
    )

    ax.set_title(
        "Parallax distinguishability in true parameter space"
    )

    colorbar = fig.colorbar(
        mesh,
        ax=ax,
    )

    colorbar.set_label(
        (
            r"$P(\Delta\chi^2_{\rm LRT}"
            r"\geq c_\alpha"
            r"\mid\Delta\chi^2_{\rm LRT}>0)$"
        )
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE,
        dpi=220,
        bbox_inches="tight",
    )

    # ========================================================
    # Diagnostics
    # ========================================================

    populated = (
        total >= MIN_COUNT
    )

    print()

    print(
        "2D cells total =",
        total.size,
    )

    print(
        f"cells with n >= {MIN_COUNT} =",
        int(
            populated.sum()
        ),
    )

    if np.any(
        populated
    ):

        print(
            "fraction range in shown cells =",
            float(
                np.nanmin(
                    fraction
                )
            ),
            "to",
            float(
                np.nanmax(
                    fraction
                )
            ),
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
