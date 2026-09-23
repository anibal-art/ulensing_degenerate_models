#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallax-detection fraction versus TRUE finite-source strength
for the STRICTLY POSITIVE H1 LRT population.

NO simulation.
NO fitting.

Finite-source strength is characterized by

    R_FS = rho_true / |u0_true|.

Interpretation:
    R_FS << 1 : source size small relative to closest approach
    R_FS ~  1 : source radius comparable to closest approach
    R_FS >  1 : lens trajectory crosses the projected source disk

We measure

    P(parallax detected | Delta chi2 > 0, R_FS)

using the frozen empirical alpha=1e-3 LRT threshold.

This is an UNCONTROLLED marginal diagnostic.  Any trend found here
can still be induced by correlations with true tE, true piE, sampling,
brightness, geometry, etc.  Those controls should only be introduced
after inspecting this first-order result.
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
    / "29_detection_fraction_vs_finite_source_strength"
)

RESULT_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "29_detection_fraction_vs_finite_source_strength"
)

FIGURE = (
    FIGURE_DIR
    / "detection_fraction_vs_finite_source_strength.png"
)

TABLE = (
    RESULT_DIR
    / "detection_fraction_vs_finite_source_strength_bins.csv"
)

REGION_TABLE = (
    RESULT_DIR
    / "detection_fraction_vs_finite_source_strength_regions.csv"
)


# ============================================================
# Frozen analysis configuration
# ============================================================

PRIMARY_THRESHOLD = 13.15982011480088

N_BINS = 30

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
        + z2 / (2.0 * total)
    ) / denominator

    half_width = (
        z
        / denominator
        * np.sqrt(
            p * (1.0 - p) / total
            + z2 / (4.0 * total**2)
        )
    )

    return (
        center - half_width,
        center + half_width,
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
    # Load positive H1 population
    # ========================================================

    df = pd.read_parquet(
        INPUT,
        columns=[
            "catalog_row",
            "delta_chi2_lrt",
            "positive_lrt_class",
            "rho_true",
            "u0_true",
            "rho_over_abs_u0_true",
        ],
    )

    T = numeric(
        df,
        "delta_chi2_lrt",
    )

    rho = numeric(
        df,
        "rho_true",
    )

    u0 = numeric(
        df,
        "u0_true",
    )

    R_FS = numeric(
        df,
        "rho_over_abs_u0_true",
    )

    classes = (
        df[
            "positive_lrt_class"
        ]
        .astype(str)
        .to_numpy()
    )

    # ========================================================
    # Positive-population audit
    # ========================================================

    if not np.all(
        np.isfinite(T)
        & (T > 0)
    ):

        raise RuntimeError(
            "Input is not the strict positive-LRT population."
        )

    detected = (
        T
        >= PRIMARY_THRESHOLD
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
            "positive_lrt_class is inconsistent with the frozen threshold."
        )

    # ========================================================
    # Audit the precomputed finite-source ratio
    # ========================================================

    recomputed = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    valid_recompute = (
        np.isfinite(rho)
        & np.isfinite(u0)
        & (np.abs(u0) > 0)
    )

    recomputed[
        valid_recompute
    ] = (
        rho[
            valid_recompute
        ]
        / np.abs(
            u0[
                valid_recompute
            ]
        )
    )

    valid_ratio = (
        np.isfinite(R_FS)
        & (R_FS > 0)
    )

    compare = (
        valid_ratio
        & np.isfinite(recomputed)
    )

    if not np.allclose(
        R_FS[compare],
        recomputed[compare],
        rtol=1.0e-10,
        atol=1.0e-12,
    ):

        raise RuntimeError(
            "rho_over_abs_u0_true does not match rho_true/abs(u0_true)."
        )

    R = R_FS[
        valid_ratio
    ]

    detected_valid = detected[
        valid_ratio
    ]

    # ========================================================
    # Global information
    # ========================================================

    overall_fraction = float(
        np.mean(
            detected_valid
        )
    )

    print(
        "=" * 110
    )

    print(
        "PARALLAX DETECTION FRACTION VS FINITE-SOURCE STRENGTH"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "strict positive-LRT events =",
        len(df),
    )

    print(
        "valid rho/|u0| =",
        len(R),
    )

    print(
        "invalid =",
        len(df) - len(R),
    )

    print(
        "overall conditional detection fraction =",
        overall_fraction,
    )

    print()

    print(
        "rho/|u0| distribution:"
    )

    for q in [
        0.00,
        0.01,
        0.10,
        0.50,
        0.90,
        0.99,
        1.00,
    ]:

        print(
            f"  q{q:4.2f} = "
            f"{np.quantile(R, q):.8g}"
        )

    print()

    # ========================================================
    # Broad physically interpretable regions
    # ========================================================

    region_definitions = [
        (
            "very_weak_FS",
            0.0,
            1.0e-2,
        ),
        (
            "weak_to_moderate_FS",
            1.0e-2,
            1.0e-1,
        ),
        (
            "strong_FS_but_no_transit",
            1.0e-1,
            1.0,
        ),
        (
            "source_transit",
            1.0,
            np.inf,
        ),
    ]

    region_rows = []

    print(
        "BROAD FINITE-SOURCE REGIONS"
    )

    print()

    for (
        label,
        left,
        right,
    ) in region_definitions:

        if np.isinf(right):

            use = (
                R >= left
            )

        else:

            use = (
                (R >= left)
                & (R < right)
            )

        n = int(
            use.sum()
        )

        k = int(
            detected_valid[
                use
            ].sum()
        )

        fraction = (
            k / n
            if n
            else np.nan
        )

        lower, upper = wilson_interval(
            k,
            n,
        )

        region_rows.append(
            {
                "region":
                    label,

                "R_FS_left":
                    left,

                "R_FS_right":
                    right,

                "n":
                    n,

                "n_detected":
                    k,

                "detection_fraction":
                    fraction,

                "wilson95_lower":
                    lower,

                "wilson95_upper":
                    upper,
            }
        )

    region_table = pd.DataFrame(
        region_rows
    )

    print(
        region_table.to_string(
            index=False
        )
    )

    region_table.to_csv(
        REGION_TABLE,
        index=False,
    )

    # ========================================================
    # Logarithmic bins spanning the actual data range
    # ========================================================

    log_min = np.floor(
        np.log10(
            np.min(R)
        )
    )

    log_max = np.ceil(
        np.log10(
            np.max(R)
        )
    )

    edges = np.geomspace(
        10.0**log_min,
        10.0**log_max,
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

            use = (
                (R >= left)
                & (R <= right)
            )

        else:

            use = (
                (R >= left)
                & (R < right)
            )

        n = int(
            use.sum()
        )

        center = np.sqrt(
            left * right
        )

        if n:

            k = int(
                detected_valid[
                    use
                ].sum()
            )

            fraction = (
                k / n
            )

            lower, upper = wilson_interval(
                k,
                n,
            )

            median_R = float(
                np.median(
                    R[
                        use
                    ]
                )
            )

        else:

            k = 0
            fraction = np.nan
            lower = np.nan
            upper = np.nan
            median_R = np.nan

        rows.append(
            {
                "bin_index":
                    ibin,

                "R_FS_left":
                    left,

                "R_FS_right":
                    right,

                "R_FS_center":
                    center,

                "R_FS_median":
                    median_R,

                "n":
                    n,

                "n_detected":
                    k,

                "detection_fraction":
                    fraction,

                "wilson95_lower":
                    lower,

                "wilson95_upper":
                    upper,

                "shown_in_figure":
                    n >= MIN_COUNT,
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
    # Single figure
    # ========================================================

    shown = result.loc[
        result[
            "shown_in_figure"
        ]
    ].copy()

    x = shown[
        "R_FS_median"
    ].to_numpy(
        dtype=float
    )

    y = shown[
        "detection_fraction"
    ].to_numpy(
        dtype=float
    )

    lower = shown[
        "wilson95_lower"
    ].to_numpy(
        dtype=float
    )

    upper = shown[
        "wilson95_upper"
    ].to_numpy(
        dtype=float
    )

    good = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(lower)
        & np.isfinite(upper)
        & (x > 0)
    )

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.6,
        )
    )

    line, = ax.plot(
        x[
            good
        ],
        y[
            good
        ],
        marker="o",
        markersize=4.5,
        linewidth=1.5,
        label="Parallax detection fraction",
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
        alpha=0.18,
        color=line.get_color(),
        label="Wilson 95% interval",
    )

    # --------------------------------------------------------
    # Geometrically important transition
    # --------------------------------------------------------

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1.1,
        label=(
            r"$\rho_{\rm true}=|u_{0,\rm true}|$"
        ),
    )

    # --------------------------------------------------------
    # Overall positive-population detection fraction
    # --------------------------------------------------------

    ax.axhline(
        overall_fraction,
        linestyle=":",
        linewidth=1.1,
        label=(
            "Overall positive-LRT detection fraction"
        ),
    )

    ax.set_xscale(
        "log"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        (
            r"True finite-source strength "
            r"$\rho_{\rm true}/|u_{0,\rm true}|$"
        )
    )

    ax.set_ylabel(
        (
            r"$P("
            r"\mathrm{parallax\ detected}"
            r"\mid \Delta\chi^2>0"
            r")$"
        )
    )

    ax.set_title(
        "Parallax detectability versus finite-source strength"
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

    print()

    print(
        "Saved figure:",
        FIGURE,
    )

    print(
        "Saved logarithmic bins:",
        TABLE,
    )

    print(
        "Saved broad-region summary:",
        REGION_TABLE,
    )


if __name__ == "__main__":
    main()
