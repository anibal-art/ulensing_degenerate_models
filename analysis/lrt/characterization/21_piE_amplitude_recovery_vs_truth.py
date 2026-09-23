#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recovery of the parallax amplitude |pi_E| for the STRICTLY POSITIVE
H1-generated LRT population.

NO simulation.
NO fitting.

We compare

    |pi_E,true|

from the Sedighe catalog with

    |pi_E,fit|
        = sqrt(piEN_fit^2 + piEE_fit^2)

from the frozen H1 production fit.

The recovery diagnostic is

    delta_log10_piE
        = log10(|pi_E,fit| / |pi_E,true|)

Interpretation
--------------
    delta_log10_piE = 0:
        perfect amplitude recovery

    delta_log10_piE > 0:
        fitted parallax amplitude is overestimated

    delta_log10_piE < 0:
        fitted parallax amplitude is underestimated

We compare separately:

    confused_fspl_no_parallax
        0 < Delta chi2_LRT < c_alpha

    parallax_detected
        Delta chi2_LRT >= c_alpha

with the empirical alpha = 1e-3 threshold.

The figure shows the median and 16--84 percentile interval of the
recovery statistic as a function of TRUE |pi_E|.
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
    / "21_piE_amplitude_recovery_vs_truth"
)

RESULT_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "21_piE_amplitude_recovery_vs_truth"
)

FIGURE = (
    FIGURE_DIR
    / "piE_amplitude_recovery_vs_truth.png"
)

TABLE = (
    RESULT_DIR
    / "piE_amplitude_recovery_vs_truth_bins.csv"
)


# ============================================================
# Configuration
# ============================================================

PRIMARY_THRESHOLD = 13.15982011480088

N_BINS = 22

PIE_MIN = 0.01
PIE_MAX = 20.0

MIN_COUNT_PER_CLASS = 30


# ============================================================
# Helpers
# ============================================================

def numeric(
    df,
    column,
):

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(
        dtype=float
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

    # ========================================================
    # Load only the columns required for this diagnostic
    # ========================================================

    columns = [
        "catalog_row",
        "delta_chi2_lrt",
        "positive_lrt_class",
        "piE_true_amp",
        "h1_piEN",
        "h1_piEE",
    ]

    df = pd.read_parquet(
        INPUT,
        columns=columns,
    )

    T = numeric(
        df,
        "delta_chi2_lrt",
    )

    piE_true = numeric(
        df,
        "piE_true_amp",
    )

    piEN_fit = numeric(
        df,
        "h1_piEN",
    )

    piEE_fit = numeric(
        df,
        "h1_piEE",
    )

    classes = df[
        "positive_lrt_class"
    ].astype(
        str
    ).to_numpy()

    # ========================================================
    # Population audit
    # ========================================================

    if not np.all(
        np.isfinite(T)
        & (T > 0)
    ):

        raise RuntimeError(
            "Input contains events outside the strict "
            "Delta-chi2 > 0 population."
        )

    detected = (
        T >= PRIMARY_THRESHOLD
    )

    confused = (
        (T > 0)
        & (
            T < PRIMARY_THRESHOLD
        )
    )

    expected_class = np.where(
        detected,
        "parallax_detected",
        "confused_fspl_no_parallax",
    )

    if not np.array_equal(
        classes,
        expected_class,
    ):

        raise RuntimeError(
            "positive_lrt_class is inconsistent with "
            "delta_chi2_lrt and the frozen threshold."
        )

    # ========================================================
    # Fitted parallax amplitude
    # ========================================================

    piE_fit = np.hypot(
        piEN_fit,
        piEE_fit,
    )

    valid = (
        np.isfinite(piE_true)
        & (piE_true > 0)
        & np.isfinite(piE_fit)
        & (piE_fit > 0)
    )

    delta_log10_piE = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    delta_log10_piE[
        valid
    ] = np.log10(
        piE_fit[
            valid
        ]
        / piE_true[
            valid
        ]
    )

    fractional_error = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    fractional_error[
        valid
    ] = (
        piE_fit[
            valid
        ]
        - piE_true[
            valid
        ]
    ) / piE_true[
        valid
    ]

    absolute_fractional_error = np.abs(
        fractional_error
    )

    # ========================================================
    # Basic summary
    # ========================================================

    print(
        "=" * 100
    )

    print(
        "PARALLAX AMPLITUDE RECOVERY"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "positive events =",
        len(df),
    )

    print(
        "valid amplitude recovery =",
        int(
            valid.sum()
        ),
    )

    print(
        "invalid =",
        int(
            (~valid).sum()
        ),
    )

    print()

    # ========================================================
    # Region-level summary
    # ========================================================

    summary_rows = []

    regions = {
        "confused_fspl_no_parallax":
            confused,

        "parallax_detected":
            detected,
    }

    for name, mask in regions.items():

        use = (
            mask
            & valid
        )

        dlog = delta_log10_piE[
            use
        ]

        afe = absolute_fractional_error[
            use
        ]

        ratio = (
            piE_fit[
                use
            ]
            / piE_true[
                use
            ]
        )

        row = {
            "class":
                name,

            "n":
                int(
                    use.sum()
                ),

            "delta_log10_piE_q16":
                float(
                    np.quantile(
                        dlog,
                        0.16,
                    )
                ),

            "delta_log10_piE_median":
                float(
                    np.median(
                        dlog
                    )
                ),

            "delta_log10_piE_q84":
                float(
                    np.quantile(
                        dlog,
                        0.84,
                    )
                ),

            "piE_fit_over_true_median":
                float(
                    np.median(
                        ratio
                    )
                ),

            "abs_fractional_error_median":
                float(
                    np.median(
                        afe
                    )
                ),

            "abs_fractional_error_q90":
                float(
                    np.quantile(
                        afe,
                        0.90,
                    )
                ),
        }

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # Binning in TRUE |pi_E|
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

        center = np.sqrt(
            left
            * right
        )

        for class_name, class_mask in (
            regions.items()
        ):

            use = (
                in_bin
                & class_mask
            )

            values = delta_log10_piE[
                use
            ]

            n = int(
                len(
                    values
                )
            )

            if n > 0:

                q16, median, q84 = np.quantile(
                    values,
                    [
                        0.16,
                        0.50,
                        0.84,
                    ],
                )

            else:

                q16 = np.nan
                median = np.nan
                q84 = np.nan

            rows.append(
                {
                    "bin_index":
                        ibin,

                    "class":
                        class_name,

                    "piE_true_left":
                        left,

                    "piE_true_right":
                        right,

                    "piE_true_center":
                        center,

                    "n":
                        n,

                    "delta_log10_piE_q16":
                        q16,

                    "delta_log10_piE_median":
                        median,

                    "delta_log10_piE_q84":
                        q84,

                    "shown_in_figure":
                        (
                            n
                            >= MIN_COUNT_PER_CLASS
                        ),
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
    # Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            7.7,
            5.5,
        )
    )

    for class_name in [
        "confused_fspl_no_parallax",
        "parallax_detected",
    ]:

        sub = result.loc[
            (
                result[
                    "class"
                ]
                == class_name
            )
            & result[
                "shown_in_figure"
            ]
        ].copy()

        x = sub[
            "piE_true_center"
        ].to_numpy(
            dtype=float
        )

        median = sub[
            "delta_log10_piE_median"
        ].to_numpy(
            dtype=float
        )

        q16 = sub[
            "delta_log10_piE_q16"
        ].to_numpy(
            dtype=float
        )

        q84 = sub[
            "delta_log10_piE_q84"
        ].to_numpy(
            dtype=float
        )

        good = (
            np.isfinite(x)
            & np.isfinite(median)
            & np.isfinite(q16)
            & np.isfinite(q84)
        )

        label = (
            "Confused with FSPL without parallax"
            if class_name
            == "confused_fspl_no_parallax"
            else "Parallax detected"
        )

        line, = ax.plot(
            x[
                good
            ],
            median[
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
            q16[
                good
            ],
            q84[
                good
            ],
            alpha=0.18,
            color=line.get_color(),
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        PIE_MIN,
        PIE_MAX,
    )

    ax.set_xlabel(
        r"True $|\boldsymbol{\pi}_E|$"
    )

    ax.set_ylabel(
        (
            r"$\log_{10}"
            r"\left("
            r"|\hat{\boldsymbol{\pi}}_E|"
            r"/"
            r"|\boldsymbol{\pi}_{E,\rm true}|"
            r"\right)$"
        )
    )

    ax.set_title(
        "Recovery of the parallax amplitude"
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE,
        dpi=220,
        bbox_inches="tight",
    )

    # ========================================================
    # Output
    # ========================================================

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
