#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the truth-joined table for the STRICTLY POSITIVE H1 LRT population.

NO simulation.
NO fitting.

Population
----------
Keep only events satisfying

    Delta chi2_LRT > 0.

Classification
--------------
Using the empirically calibrated alpha = 1e-3 threshold,

    0 < Delta chi2_LRT < c_alpha
        -> confused_fspl_no_parallax

    Delta chi2_LRT >= c_alpha
        -> parallax_detected

with

    c_alpha = 13.15982011480088.

Truth source
------------
Parallax_LSST/data_sedighe/LSSTMONTS.dat

The catalog has no header.  Its 80 column descriptions are supplied in

Parallax_LSST/data_sedighe/columns

and originate from Sedighe's catalog description.

Important
---------
The Sedighe catalog contains a column called "Delta chi2".
That quantity is retained as

    sedighe_delta_chi2

and MUST NOT be confused with our production LRT statistic

    delta_chi2_lrt.

The catalog contains the parallax amplitude |pi_E|, but not explicit
piEN / piEE truth components.  Those components will be reconstructed
later using the exact convention implemented in the production code.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

CATALOG_FILE = (
    ROOT
    / "Parallax_LSST"
    / "data_sedighe"
    / "LSSTMONTS.dat"
)

COLUMNS_FILE = (
    ROOT
    / "Parallax_LSST"
    / "data_sedighe"
    / "columns"
)

RESULTS_FILE = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "11_covariance_geometry"
    / "h1_covariance_geometry.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "analysis"
    / "lrt"
    / "results"
    / "characterization"
    / "18_positive_truth"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "h1_positive_truth_characterization.parquet"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "h1_positive_truth_summary.txt"
)


# ============================================================
# LRT threshold
# ============================================================

PRIMARY_THRESHOLD = 13.15982011480088


# ============================================================
# Canonical names for the 80 Sedighe columns
# ============================================================

CATALOG_COLUMNS = [
    "number_lensing",
    "NL1",
    "NL2",

    "gal_b_deg",
    "gal_l_deg",
    "ra_deg",
    "dec_deg",

    "lens_structure",
    "lens_mass_msun",
    "DL_kpc",
    "lens_velocity_kms",

    "source_structure",
    "source_mass_msun",
    "DS_kpc",
    "source_velocity_kms",

    "log10_teff_source",
    "source_luminosity_class",
    "source_radius_rsun",
    "source_age_gyr",

    "tE_true_days",
    "einstein_radius_AU",
    "t0_true_days",
    "mu_rel_mas_day",
    "v_rel_kms",

    "u0_true",
    "optical_depth_x1e6",

    "thetaE_mas",
    "piE_true_amp",
    "rho_true",
    "fwhm_days",

    "abs_mag_u",
    "abs_mag_g",
    "abs_mag_r",
    "abs_mag_i",
    "abs_mag_z",
    "abs_mag_y",

    "source_mag_u",
    "source_mag_g",
    "source_mag_r",
    "source_mag_i",
    "source_mag_z",
    "source_mag_y",

    "baseline_mag_u",
    "baseline_mag_g",
    "baseline_mag_r",
    "baseline_mag_i",
    "baseline_mag_z",
    "baseline_mag_y",

    "blend_fraction_u",
    "blend_fraction_g",
    "blend_fraction_r",
    "blend_fraction_i",
    "blend_fraction_z",
    "blend_fraction_y",

    "extinction_u",
    "extinction_g",
    "extinction_r",
    "extinction_i",
    "extinction_z",
    "extinction_y",

    "n_blend_stars_u",
    "n_blend_stars_g",
    "n_blend_stars_r",
    "n_blend_stars_i",
    "n_blend_stars_z",
    "n_blend_stars_y",

    "sedighe_n_data_points",
    "sedighe_delta_chi2",

    "rho_stars",
    "n_stars",

    "alpha_deg",

    "murel_1_mas_day",
    "murel_2_mas_day",

    "xi_deg",

    "detection_flag_u",
    "detection_flag_g",
    "detection_flag_r",
    "detection_flag_i",
    "detection_flag_z",
    "detection_flag_y",
]


# ============================================================
# Helpers
# ============================================================

def finite_quantiles(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return None

    q = np.quantile(
        values,
        [
            0.0,
            0.01,
            0.10,
            0.50,
            0.90,
            0.99,
            1.0,
        ],
    )

    return q


def print_quantiles(
    name,
    values,
):

    q = finite_quantiles(
        values
    )

    print()
    print(name)

    if q is None:
        print("  no finite values")
        return

    labels = [
        "min",
        "q01",
        "q10",
        "median",
        "q90",
        "q99",
        "max",
    ]

    for label, value in zip(
        labels,
        q,
    ):
        print(
            f"  {label:8s} = {value:.8g}"
        )


# ============================================================
# Main
# ============================================================

def main():

    for path in [
        CATALOG_FILE,
        COLUMNS_FILE,
        RESULTS_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing input:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Validate catalog descriptions
    # ========================================================

    descriptions = [
        line.strip()
        for line
        in COLUMNS_FILE.read_text().splitlines()
        if line.strip()
    ]

    print(
        "=" * 90
    )

    print(
        "SEDIGHE CATALOG DESCRIPTION"
    )

    print(
        "=" * 90
    )

    print(
        "description columns =",
        len(descriptions),
    )

    print(
        "canonical columns   =",
        len(CATALOG_COLUMNS),
    )

    if len(descriptions) != 80:

        raise RuntimeError(
            "Expected exactly 80 Sedighe column descriptions, "
            f"found {len(descriptions)}."
        )

    if len(CATALOG_COLUMNS) != 80:

        raise RuntimeError(
            "Internal canonical column list does not contain 80 names."
        )

    # ========================================================
    # Load production results
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "LOADING H1 PRODUCTION RESULTS"
    )

    print(
        "=" * 90
    )

    results = pd.read_parquet(
        RESULTS_FILE
    )

    required = [
        "catalog_row",
        "delta_chi2_lrt",
    ]

    missing = [
        col
        for col
        in required
        if col not in results.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing result columns: {missing}"
        )

    results[
        "catalog_row"
    ] = pd.to_numeric(
        results[
            "catalog_row"
        ],
        errors="raise",
    ).astype(
        np.int64
    )

    T = pd.to_numeric(
        results[
            "delta_chi2_lrt"
        ],
        errors="coerce",
    )

    print(
        "all H1 rows =",
        len(results),
    )

    print(
        "negative =",
        int(
            (T < 0).sum()
        ),
    )

    print(
        "zero     =",
        int(
            (T == 0).sum()
        ),
    )

    print(
        "positive =",
        int(
            (T > 0).sum()
        ),
    )

    # STRICT positive population.
    positive = (
        results.loc[
            T > 0
        ]
        .copy()
    )

    if positive[
        "catalog_row"
    ].duplicated().any():

        duplicated = positive.loc[
            positive[
                "catalog_row"
            ].duplicated(
                keep=False
            ),
            "catalog_row",
        ]

        raise RuntimeError(
            "Duplicate catalog_row values in positive H1 results:\n"
            f"{duplicated.head(20).to_list()}"
        )

    positive[
        "positive_lrt_class"
    ] = np.where(
        positive[
            "delta_chi2_lrt"
        ]
        >= PRIMARY_THRESHOLD,

        "parallax_detected",

        "confused_fspl_no_parallax",
    )

    print()
    print(
        "Strict positive classification:"
    )

    print(
        positive[
            "positive_lrt_class"
        ].value_counts()
    )

    # ========================================================
    # Load the catalog only as far as required
    # ========================================================

    max_catalog_row = int(
        positive[
            "catalog_row"
        ].max()
    )

    print()
    print(
        "=" * 90
    )

    print(
        "LOADING SEDIGHE TRUTH CATALOG"
    )

    print(
        "=" * 90
    )

    print(
        "maximum required catalog_row =",
        max_catalog_row,
    )

    catalog = pd.read_csv(
        CATALOG_FILE,
        sep=r"\s+",
        header=None,
        names=CATALOG_COLUMNS,
        nrows=max_catalog_row + 1,
    )

    print(
        "catalog rows read =",
        len(catalog),
    )

    print(
        "catalog columns   =",
        len(catalog.columns),
    )

    if len(catalog.columns) != 80:

        raise RuntimeError(
            "LSSTMONTS.dat did not parse as exactly 80 columns."
        )

    if len(catalog) <= max_catalog_row:

        raise RuntimeError(
            "Catalog ended before the maximum requested catalog_row."
        )

    # ========================================================
    # Establish the production catalog_row convention
    # ========================================================
    #
    # IMPORTANT:
    #
    # Number_lensing is NOT the global row identifier of LSSTMONTS.dat.
    # It must not be used to establish the join.
    #
    # Production defines catalog_row from the zero-based physical row
    # position in LSSTMONTS.dat:
    #
    #     catalog_row = catalog_row_offset + local row index
    #
    # For the complete catalog loaded from row zero:
    #
    #     catalog_row = 0, 1, ..., N-1
    #
    # This reproduces exactly the convention implemented in
    # prepare_catalog() in the production pipeline.
    # ========================================================

    catalog.insert(
        0,
        "catalog_row",
        np.arange(
            len(catalog),
            dtype=np.int64,
        ),
    )

    expected_catalog_rows = np.arange(
        len(catalog),
        dtype=np.int64,
    )

    actual_catalog_rows = catalog[
        "catalog_row"
    ].to_numpy(
        dtype=np.int64
    )

    row_alignment_valid = np.array_equal(
        actual_catalog_rows,
        expected_catalog_rows,
    )

    print()
    print(
        "catalog_row zero-based physical-row alignment:"
    )

    print(
        "catalog_row == physical row index :",
        row_alignment_valid,
    )

    print(
        "first catalog_row =",
        int(
            actual_catalog_rows[0]
        ),
    )

    print(
        "last catalog_row  =",
        int(
            actual_catalog_rows[-1]
        ),
    )

    print(
        "n catalog rows    =",
        len(
            catalog
        ),
    )

    if not row_alignment_valid:

        raise RuntimeError(
            "Zero-based physical-row catalog alignment failed."
        )

    if int(
        actual_catalog_rows[-1]
    ) != len(
        catalog
    ) - 1:

        raise RuntimeError(
            "Unexpected final catalog_row."
        )

    # --------------------------------------------------------
    # Number_lensing diagnostic only
    # --------------------------------------------------------
    #
    # Number_lensing is metadata from Sedighe's catalog.
    # It is NOT the global physical-row identifier and therefore
    # must NOT be used as the truth-join key.
    # --------------------------------------------------------

    number_lensing = pd.to_numeric(
        catalog[
            "number_lensing"
        ],
        errors="coerce",
    )

    print()
    print(
        "Number_lensing diagnostic only:"
    )

    print(
        "  finite =",
        int(
            number_lensing.notna().sum()
        ),
        "/",
        len(
            catalog
        ),
    )

    print(
        "  unique values =",
        int(
            number_lensing.nunique(
                dropna=True
            )
        ),
    )

    print(
        "  first 20 values =",
        number_lensing.head(
            20
        ).to_list(),
    )

    print(
        "  NOTE: Number_lensing is not used as the join key."
    )

    # ========================================================
    # Keep truth variables
    # ========================================================

    truth_columns = [
        "catalog_row",
        "number_lensing",

        "gal_b_deg",
        "gal_l_deg",
        "ra_deg",
        "dec_deg",

        "lens_structure",
        "lens_mass_msun",
        "DL_kpc",
        "lens_velocity_kms",

        "source_structure",
        "source_mass_msun",
        "DS_kpc",
        "source_velocity_kms",

        "source_radius_rsun",
        "source_age_gyr",

        "tE_true_days",
        "einstein_radius_AU",
        "t0_true_days",

        "mu_rel_mas_day",
        "v_rel_kms",

        "u0_true",

        "thetaE_mas",
        "piE_true_amp",
        "rho_true",
        "fwhm_days",

        "source_mag_u",
        "source_mag_g",
        "source_mag_r",
        "source_mag_i",
        "source_mag_z",
        "source_mag_y",

        "baseline_mag_u",
        "baseline_mag_g",
        "baseline_mag_r",
        "baseline_mag_i",
        "baseline_mag_z",
        "baseline_mag_y",

        "blend_fraction_u",
        "blend_fraction_g",
        "blend_fraction_r",
        "blend_fraction_i",
        "blend_fraction_z",
        "blend_fraction_y",

        "sedighe_n_data_points",

        # Retain but DO NOT use as our LRT statistic.
        "sedighe_delta_chi2",

        "rho_stars",
        "n_stars",

        "alpha_deg",

        "murel_1_mas_day",
        "murel_2_mas_day",
        "xi_deg",

        "detection_flag_u",
        "detection_flag_g",
        "detection_flag_r",
        "detection_flag_i",
        "detection_flag_z",
        "detection_flag_y",
    ]

    truth = catalog[
        truth_columns
    ].copy()

    # ========================================================
    # Derived truth quantities
    # ========================================================

    DL = pd.to_numeric(
        truth[
            "DL_kpc"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    DS = pd.to_numeric(
        truth[
            "DS_kpc"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    u0 = pd.to_numeric(
        truth[
            "u0_true"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    rho = pd.to_numeric(
        truth[
            "rho_true"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    thetaE = pd.to_numeric(
        truth[
            "thetaE_mas"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    piE = pd.to_numeric(
        truth[
            "piE_true_amp"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Finite-source geometry
    # --------------------------------------------------------

    rho_over_abs_u0 = np.full(
        len(truth),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(rho)
        & np.isfinite(u0)
        & (
            np.abs(u0)
            > 0
        )
    )

    rho_over_abs_u0[
        valid
    ] = (
        rho[
            valid
        ]
        / np.abs(
            u0[
                valid
            ]
        )
    )

    truth[
        "rho_over_abs_u0_true"
    ] = rho_over_abs_u0

    truth[
        "finite_source_dominated_true"
    ] = (
        np.isfinite(rho)
        & np.isfinite(u0)
        & (
            rho
            > np.abs(u0)
        )
    )

    # --------------------------------------------------------
    # Lens-source distances
    # --------------------------------------------------------

    DLS = (
        DS
        - DL
    )

    truth[
        "DLS_kpc"
    ] = DLS

    x_DL_DS = np.full(
        len(truth),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(DL)
        & np.isfinite(DS)
        & (
            DS > 0
        )
    )

    x_DL_DS[
        valid
    ] = (
        DL[
            valid
        ]
        / DS[
            valid
        ]
    )

    truth[
        "x_DL_over_DS"
    ] = x_DL_DS

    # --------------------------------------------------------
    # Relative parallax
    #
    # If D is in kpc, 1 / D[kpc] is parallax in mas.
    # --------------------------------------------------------

    pi_rel_from_distance = np.full(
        len(truth),
        np.nan,
        dtype=float,
    )

    valid_distance = (
        np.isfinite(DL)
        & np.isfinite(DS)
        & (DL > 0)
        & (DS > 0)
    )

    pi_rel_from_distance[
        valid_distance
    ] = (
        1.0
        / DL[
            valid_distance
        ]
        -
        1.0
        / DS[
            valid_distance
        ]
    )

    truth[
        "pi_rel_mas_from_distance"
    ] = pi_rel_from_distance

    # Independent catalog relation:
    #
    # pi_E = pi_rel / theta_E
    #
    # therefore
    #
    # pi_rel = pi_E * theta_E.
    #
    pi_rel_from_catalog = (
        piE
        * thetaE
    )

    truth[
        "pi_rel_mas_from_piE_thetaE"
    ] = pi_rel_from_catalog

    pi_rel_relative_difference = np.full(
        len(truth),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(
            pi_rel_from_distance
        )
        & np.isfinite(
            pi_rel_from_catalog
        )
        & (
            np.abs(
                pi_rel_from_catalog
            )
            > 0
        )
    )

    pi_rel_relative_difference[
        valid
    ] = (
        np.abs(
            pi_rel_from_distance[
                valid
            ]
            -
            pi_rel_from_catalog[
                valid
            ]
        )
        / np.abs(
            pi_rel_from_catalog[
                valid
            ]
        )
    )

    truth[
        "pi_rel_consistency_relative_difference"
    ] = pi_rel_relative_difference

    # ========================================================
    # Join
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "JOINING PRODUCTION RESULTS TO TRUTH"
    )

    print(
        "=" * 90
    )

    joined = positive.merge(
        truth,
        on="catalog_row",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    print(
        joined[
            "_merge"
        ].value_counts()
    )

    missing_truth = int(
        (
            joined[
                "_merge"
            ]
            != "both"
        ).sum()
    )

    if missing_truth:

        bad = joined.loc[
            joined[
                "_merge"
            ]
            != "both",
            "catalog_row",
        ]

        raise RuntimeError(
            "Missing truth rows after join. "
            f"First rows: {bad.head(20).to_list()}"
        )

    joined = joined.drop(
        columns=[
            "_merge"
        ]
    )

    # ========================================================
    # Final audits
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "FINAL POSITIVE POPULATION"
    )

    print(
        "=" * 90
    )

    print(
        "rows =",
        len(joined),
    )

    print()

    print(
        joined[
            "positive_lrt_class"
        ].value_counts()
    )

    print()

    print(
        "finite-source dominated "
        "(rho_true > |u0_true|):"
    )

    print(
        joined[
            "finite_source_dominated_true"
        ].value_counts()
    )

    print()

    print(
        "Distance-order audit:"
    )

    print(
        "DL < DS =",
        int(
            (
                joined[
                    "DL_kpc"
                ]
                <
                joined[
                    "DS_kpc"
                ]
            ).sum()
        ),
        "/",
        len(joined),
    )

    print_quantiles(
        "tE_true_days",
        joined[
            "tE_true_days"
        ],
    )

    print_quantiles(
        "rho_over_abs_u0_true",
        joined[
            "rho_over_abs_u0_true"
        ],
    )

    print_quantiles(
        "piE_true_amp",
        joined[
            "piE_true_amp"
        ],
    )

    print_quantiles(
        "DL_kpc",
        joined[
            "DL_kpc"
        ],
    )

    print_quantiles(
        "DS_kpc",
        joined[
            "DS_kpc"
        ],
    )

    print_quantiles(
        "pi_rel_mas_from_distance",
        joined[
            "pi_rel_mas_from_distance"
        ],
    )

    print_quantiles(
        "pi_rel consistency relative difference",
        joined[
            "pi_rel_consistency_relative_difference"
        ],
    )

    # ========================================================
    # Save
    # ========================================================

    joined.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Compact summary file
    # --------------------------------------------------------

    with SUMMARY_FILE.open(
        "w"
    ) as f:

        f.write(
            f"rows={len(joined)}\n"
        )

        f.write(
            f"threshold_alpha_1e-3={PRIMARY_THRESHOLD:.15g}\n"
        )

        counts = joined[
            "positive_lrt_class"
        ].value_counts()

        for key, value in counts.items():

            f.write(
                f"{key}={int(value)}\n"
            )

        f.write(
            "finite_source_dominated="
            f"{int(joined['finite_source_dominated_true'].sum())}\n"
        )

        f.write(
            "distance_order_DL_lt_DS="
            f"{int((joined['DL_kpc'] < joined['DS_kpc']).sum())}\n"
        )

    print()
    print(
        "Saved:",
        OUTPUT_FILE,
    )

    print(
        "Saved:",
        SUMMARY_FILE,
    )


if __name__ == "__main__":
    main()
