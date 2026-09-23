#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recovery of the TRUE parallax-vector components for the STRICTLY
POSITIVE H1 LRT population.

NO simulation.
NO fitting.

The production configuration is frozen as:

    angle_basis = galactic_n1n2

and uses the catalog xi angle.

Production convention
---------------------
In the local Galactic tangent plane:

    n1 = increasing Galactic longitude (+l)
    n2 = increasing Galactic latitude  (+b)

    piE_n1 = piE * cos(xi)
    piE_n2 = piE * sin(xi)

The vector is then rotated to the local ICRS tangent basis:

    piEN = North component
    piEE = East component

using the same geometric construction as production.

Diagnostic
----------
For each fitted component we calculate

    error_N =
        |piEN_fit - piEN_true| / |piE_true|

    error_E =
        |piEE_fit - piEE_true| / |piE_true|

Normalizing by the TOTAL true parallax amplitude is intentional:
normalizing by piEN_true or piEE_true would diverge whenever one true
component approaches zero.

The single figure compares median component-recovery errors as a
function of TRUE |pi_E| for:

    confused FSPL without parallax
    parallax detected

and for both North and East components.

The output table also retains q16/q50/q84 for later diagnostics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u


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
    / "25_piE_component_recovery_vs_truth"
)

RESULT_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "25_piE_component_recovery_vs_truth"
)

FIGURE = (
    FIGURE_DIR
    / "piE_component_recovery_vs_truth.png"
)

TABLE = (
    RESULT_DIR
    / "piE_component_recovery_vs_truth_bins.csv"
)

EVENT_TABLE = (
    RESULT_DIR
    / "piE_true_components_and_recovery.parquet"
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
# Exact production geometry
# ============================================================

def true_piE_components_from_catalog(
    piE,
    xi_deg,
    l_deg,
    b_deg,
):

    """
    Reproduce the production galactic_n1n2 -> ICRS North/East
    transformation.

    Parameters
    ----------
    piE : ndarray
        True parallax amplitude.

    xi_deg : ndarray
        Catalog xi angle in degrees.

    l_deg, b_deg : ndarray
        Galactic coordinates in degrees.

    Returns
    -------
    piEN_true, piEE_true
    """

    piE = np.asarray(
        piE,
        dtype=float,
    )

    xi_deg = np.asarray(
        xi_deg,
        dtype=float,
    )

    l_deg = np.asarray(
        l_deg,
        dtype=float,
    )

    b_deg = np.asarray(
        b_deg,
        dtype=float,
    )

    xi_rad = np.mod(
        np.deg2rad(
            xi_deg
        ),
        2.0 * np.pi,
    )

    # --------------------------------------------------------
    # Event position in Galactic coordinates
    # --------------------------------------------------------

    center_gal = SkyCoord(
        l=l_deg * u.deg,
        b=b_deg * u.deg,
        frame="galactic",
    )

    center_icrs = center_gal.icrs

    # --------------------------------------------------------
    # Same small displacement used by production to construct
    # the local +l and +b tangent directions.
    # --------------------------------------------------------

    epsilon = (
        1.0e-5
        * u.deg
    )

    cos_b = np.cos(
        np.deg2rad(
            b_deg
        )
    )

    if np.any(
        np.abs(
            cos_b
        )
        < 1.0e-8
    ):

        raise RuntimeError(
            "Cannot construct Galactic +l tangent basis "
            "near a Galactic pole."
        )

    plus_n1 = SkyCoord(
        l=(
            l_deg * u.deg
            + epsilon / cos_b
        ),
        b=b_deg * u.deg,
        frame="galactic",
    ).icrs

    plus_n2 = SkyCoord(
        l=l_deg * u.deg,
        b=(
            b_deg * u.deg
            + epsilon
        ),
        frame="galactic",
    ).icrs

    # Astropy position_angle is measured east of ICRS north.
    pa_n1 = center_icrs.position_angle(
        plus_n1
    ).to_value(
        u.rad
    )

    pa_n2 = center_icrs.position_angle(
        plus_n2
    ).to_value(
        u.rad
    )

    # --------------------------------------------------------
    # Components in local Galactic tangent basis
    # --------------------------------------------------------

    piE_n1 = (
        piE
        * np.cos(
            xi_rad
        )
    )

    piE_n2 = (
        piE
        * np.sin(
            xi_rad
        )
    )

    # --------------------------------------------------------
    # Rotate to ICRS North/East
    # --------------------------------------------------------

    piEN_true = (
        piE_n1
        * np.cos(
            pa_n1
        )
        + piE_n2
        * np.cos(
            pa_n2
        )
    )

    piEE_true = (
        piE_n1
        * np.sin(
            pa_n1
        )
        + piE_n2
        * np.sin(
            pa_n2
        )
    )

    return (
        piEN_true,
        piEE_true,
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
    # Load required quantities
    # ========================================================

    df = pd.read_parquet(
        INPUT,
        columns=[
            "catalog_row",
            "delta_chi2_lrt",
            "positive_lrt_class",

            "gal_l_deg",
            "gal_b_deg",
            "xi_deg",

            "piE_true_amp",

            "h1_piEN",
            "h1_piEE",
        ],
    )

    T = numeric(
        df,
        "delta_chi2_lrt",
    )

    l_deg = numeric(
        df,
        "gal_l_deg",
    )

    b_deg = numeric(
        df,
        "gal_b_deg",
    )

    xi_deg = numeric(
        df,
        "xi_deg",
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
        & (
            T > 0
        )
    ):

        raise RuntimeError(
            "Input is not the strict positive-LRT population."
        )

    detected = (
        T
        >= PRIMARY_THRESHOLD
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
            "positive_lrt_class does not match the frozen LRT threshold."
        )

    # ========================================================
    # Reconstruct exact truth components
    # ========================================================

    valid_truth_input = (
        np.isfinite(
            piE_true
        )
        & (
            piE_true > 0
        )
        & np.isfinite(
            xi_deg
        )
        & np.isfinite(
            l_deg
        )
        & np.isfinite(
            b_deg
        )
    )

    if not np.all(
        valid_truth_input
    ):

        raise RuntimeError(
            "Non-finite inputs found in truth parallax geometry."
        )

    (
        piEN_true,
        piEE_true,
    ) = true_piE_components_from_catalog(
        piE=piE_true,
        xi_deg=xi_deg,
        l_deg=l_deg,
        b_deg=b_deg,
    )

    # ========================================================
    # Amplitude-conservation audit
    # ========================================================

    recovered_amp = np.hypot(
        piEN_true,
        piEE_true,
    )

    amp_relative_difference = (
        np.abs(
            recovered_amp
            - piE_true
        )
        / piE_true
    )

    max_amp_relative_difference = float(
        np.nanmax(
            amp_relative_difference
        )
    )

    print(
        "=" * 105
    )

    print(
        "TRUE PARALLAX COMPONENT RECONSTRUCTION"
    )

    print(
        "=" * 105
    )

    print()

    print(
        "rows =",
        len(df),
    )

    print(
        "max relative amplitude-conservation error =",
        max_amp_relative_difference,
    )

    if not np.allclose(
        recovered_amp,
        piE_true,
        rtol=1.0e-7,
        atol=1.0e-10,
    ):

        raise RuntimeError(
            "Truth-component reconstruction does not conserve |pi_E|."
        )

    # ========================================================
    # Recovery errors
    # ========================================================

    valid_fit = (
        np.isfinite(
            piEN_fit
        )
        & np.isfinite(
            piEE_fit
        )
    )

    valid = (
        valid_truth_input
        & valid_fit
    )

    signed_error_N = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    signed_error_E = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    signed_error_N[
        valid
    ] = (
        piEN_fit[
            valid
        ]
        - piEN_true[
            valid
        ]
    ) / piE_true[
        valid
    ]

    signed_error_E[
        valid
    ] = (
        piEE_fit[
            valid
        ]
        - piEE_true[
            valid
        ]
    ) / piE_true[
        valid
    ]

    abs_error_N = np.abs(
        signed_error_N
    )

    abs_error_E = np.abs(
        signed_error_E
    )

    # ========================================================
    # Save event-level truth components
    # ========================================================

    event_table = pd.DataFrame(
        {
            "catalog_row":
                df[
                    "catalog_row"
                ].to_numpy(),

            "delta_chi2_lrt":
                T,

            "positive_lrt_class":
                classes,

            "piE_true_amp":
                piE_true,

            "piEN_true":
                piEN_true,

            "piEE_true":
                piEE_true,

            "h1_piEN":
                piEN_fit,

            "h1_piEE":
                piEE_fit,

            "signed_error_piEN_over_true_piE":
                signed_error_N,

            "signed_error_piEE_over_true_piE":
                signed_error_E,

            "abs_error_piEN_over_true_piE":
                abs_error_N,

            "abs_error_piEE_over_true_piE":
                abs_error_E,

            "truth_amplitude_relative_difference":
                amp_relative_difference,
        }
    )

    event_table.to_parquet(
        EVENT_TABLE,
        index=False,
    )

    # ========================================================
    # Overall class/component summary
    # ========================================================

    regions = {
        "confused_fspl_no_parallax":
            confused,

        "parallax_detected":
            detected,
    }

    print()

    print(
        "COMPONENT RECOVERY SUMMARY"
    )

    print()

    summary_rows = []

    for class_name, class_mask in (
        regions.items()
    ):

        for component, signed_error, abs_error in [
            (
                "piEN",
                signed_error_N,
                abs_error_N,
            ),
            (
                "piEE",
                signed_error_E,
                abs_error_E,
            ),
        ]:

            use = (
                class_mask
                & valid
            )

            summary_rows.append(
                {
                    "class":
                        class_name,

                    "component":
                        component,

                    "n":
                        int(
                            use.sum()
                        ),

                    "signed_error_median":
                        float(
                            np.median(
                                signed_error[
                                    use
                                ]
                            )
                        ),

                    "abs_error_q16":
                        float(
                            np.quantile(
                                abs_error[
                                    use
                                ],
                                0.16,
                            )
                        ),

                    "abs_error_median":
                        float(
                            np.median(
                                abs_error[
                                    use
                                ]
                            )
                        ),

                    "abs_error_q84":
                        float(
                            np.quantile(
                                abs_error[
                                    use
                                ],
                                0.84,
                            )
                        ),
                }
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
    # Bin versus TRUE |pi_E|
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

            for component, values in [
                (
                    "piEN",
                    abs_error_N,
                ),
                (
                    "piEE",
                    abs_error_E,
                ),
            ]:

                use = (
                    in_bin
                    & class_mask
                )

                x = values[
                    use
                ]

                n = len(
                    x
                )

                if n:

                    q16, median, q84 = np.quantile(
                        x,
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

                        "component":
                            component,

                        "piE_true_left":
                            left,

                        "piE_true_right":
                            right,

                        "piE_true_center":
                            center,

                        "n":
                            int(
                                n
                            ),

                        "abs_error_q16":
                            q16,

                        "abs_error_median":
                            median,

                        "abs_error_q84":
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
    # Single figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.6,
        )
    )

    plot_definitions = [
        (
            "confused_fspl_no_parallax",
            "piEN",
            r"Confused: $\pi_{E,N}$",
            "-",
            "o",
        ),
        (
            "confused_fspl_no_parallax",
            "piEE",
            r"Confused: $\pi_{E,E}$",
            "--",
            "o",
        ),
        (
            "parallax_detected",
            "piEN",
            r"Detected: $\pi_{E,N}$",
            "-",
            "s",
        ),
        (
            "parallax_detected",
            "piEE",
            r"Detected: $\pi_{E,E}$",
            "--",
            "s",
        ),
    ]

    for (
        class_name,
        component,
        label,
        linestyle,
        marker,
    ) in plot_definitions:

        sub = result.loc[
            (
                result[
                    "class"
                ]
                == class_name
            )
            & (
                result[
                    "component"
                ]
                == component
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

        y = sub[
            "abs_error_median"
        ].to_numpy(
            dtype=float
        )

        good = (
            np.isfinite(x)
            & np.isfinite(y)
            & (
                y > 0
            )
        )

        ax.plot(
            x[
                good
            ],
            y[
                good
            ],
            marker=marker,
            markersize=4.0,
            linewidth=1.5,
            linestyle=linestyle,
            label=label,
        )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
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
            r"Median "
            r"$|\hat{\pi}_{E,i}-\pi_{E,i,\rm true}|"
            r"/|\boldsymbol{\pi}_{E,\rm true}|$"
        )
    )

    ax.set_title(
        "Recovery of the parallax-vector components"
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
        "Saved event table:",
        EVENT_TABLE,
    )

    print(
        "Saved bins:",
        TABLE,
    )


if __name__ == "__main__":
    main()
