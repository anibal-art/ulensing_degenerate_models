#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the event-level lens-mass inference table for the controlled
parallax-only experiment.

NO simulation.
NO fitting.

The experiment isolates the effect of parallax recovery on lens-mass
inference by holding theta_E fixed to its true catalog value.

For each event,

    M_L = theta_E / (kappa * pi_E)

and therefore, with theta_E fixed,

    Mhat_L / M_L,true = pi_E,true / pihat_E.

Thus

    log10(Mhat_L / M_L,true)
        = -log10(pihat_E / pi_E,true).

Physical quantities lens_mass_msun and thetaE_mas are read directly
from the original LSSTMONTS catalog using catalog_row as the exact
zero-based physical catalog-row index.

The Monte-Carlo parallax-amplitude intervals are reused from script 22.
Since

    M_L propto 1 / pi_E,

the transformed central interval is

    q16(M) = M_true * pi_E,true / q84(pi_E)
    q50(M) = M_true * pi_E,true / q50(pi_E)
    q84(M) = M_true * pi_E,true / q16(pi_E)

Outputs
-------
analysis/lrt/results/characterization/
    39_lens_mass_parallax_only/
        lens_mass_parallax_only_events.parquet
        lens_mass_parallax_only_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parents[3]

POSITIVE_INPUT = (
    ROOT
    / "analysis/lrt/results/characterization/18_positive_truth"
    / "h1_positive_truth_characterization.parquet"
)

PIE_MC_INPUT = (
    ROOT
    / "analysis/lrt/results/characterization"
    / "22_piE_amplitude_uncertainty_mc_vs_truth"
    / "piE_amplitude_mc_event_intervals.parquet"
)

CATALOG_DIR = (
    ROOT
    / "Parallax_LSST/data_sedighe"
)

CATALOG_COLUMNS_FILE = (
    CATALOG_DIR
    / "columns"
)

CATALOG_DATA_FILE = (
    CATALOG_DIR
    / "LSSTMONTS.dat"
)

RESULT_DIR = (
    ROOT
    / "analysis/lrt/results/characterization"
    / "39_lens_mass_parallax_only"
)

EVENT_TABLE = (
    RESULT_DIR
    / "lens_mass_parallax_only_events.parquet"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "lens_mass_parallax_only_summary.txt"
)


# ============================================================================
# Configuration
# ============================================================================

KAPPA_MAS_PER_MSUN = 8.144

LENS_MASS_CANDIDATES = [
    "Lens_mass (solar mass)",
    "Lens mass (solar mass)",
    "lens_mass_msun",
]

THETAE_CANDIDATES = [
    "Angular Einstein radius (mas)",
    "thetaE_mas",
]


# ============================================================================
# Helpers
# ============================================================================

def numeric(df, column):
    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def require_columns(df, columns, source_name):
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns in {source_name}: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )


def find_catalog_column(columns, candidates, quantity_name):
    """
    Resolve one physical catalog column without silently guessing.
    """

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise RuntimeError(
        f"Could not identify catalog column for {quantity_name}.\n"
        f"Tried: {candidates}\n"
        f"Available catalog columns:\n{columns}"
    )


def load_catalog_physical_quantities(required_catalog_rows):
    """
    Load only lens mass and theta_E from the original LSSTMONTS catalog.

    catalog_row is defined as the exact zero-based physical row index in
    LSSTMONTS.dat, so the DataFrame row number is the join key.
    """

    if not CATALOG_COLUMNS_FILE.exists():
        raise FileNotFoundError(
            CATALOG_COLUMNS_FILE
        )

    if not CATALOG_DATA_FILE.exists():
        raise FileNotFoundError(
            CATALOG_DATA_FILE
        )

    catalog_columns = [
        line.strip()
        for line in CATALOG_COLUMNS_FILE.read_text().splitlines()
        if line.strip()
    ]

    print(
        "Catalog columns read =",
        len(catalog_columns),
    )

    mass_column = find_catalog_column(
        catalog_columns,
        LENS_MASS_CANDIDATES,
        "lens mass",
    )

    thetaE_column = find_catalog_column(
        catalog_columns,
        THETAE_CANDIDATES,
        "angular Einstein radius",
    )

    print(
        "Lens-mass catalog column =",
        mass_column,
    )

    print(
        "theta_E catalog column =",
        thetaE_column,
    )

    mass_index = catalog_columns.index(
        mass_column
    )

    thetaE_index = catalog_columns.index(
        thetaE_column
    )

    # ------------------------------------------------------------------------
    # Read only the two required positional columns.
    #
    # The original file has no header and is whitespace-delimited.
    # Reading only two columns keeps this cheap even for all 966k rows.
    # ------------------------------------------------------------------------

    catalog = pd.read_csv(
        CATALOG_DATA_FILE,
        sep=r"\s+",
        header=None,
        usecols=[
            mass_index,
            thetaE_index,
        ],
        names=[
            "lens_mass_msun",
            "thetaE_mas",
        ],
        engine="c",
    )

    catalog.insert(
        0,
        "catalog_row",
        np.arange(
            len(catalog),
            dtype=np.int64,
        ),
    )

    print(
        "Catalog rows read =",
        len(catalog),
    )

    required_catalog_rows = np.asarray(
        required_catalog_rows,
        dtype=np.int64,
    )

    if required_catalog_rows.size == 0:
        raise RuntimeError(
            "No catalog rows requested."
        )

    min_row = int(
        required_catalog_rows.min()
    )

    max_row = int(
        required_catalog_rows.max()
    )

    if min_row < 0:
        raise RuntimeError(
            f"Negative catalog_row encountered: {min_row}"
        )

    if max_row >= len(catalog):
        raise RuntimeError(
            "catalog_row exceeds LSSTMONTS catalog length: "
            f"max row={max_row}, catalog rows={len(catalog)}"
        )

    required_unique = np.unique(
        required_catalog_rows
    )

    physical = (
        catalog
        .iloc[required_unique]
        .copy()
    )

    # iloc row identity must equal catalog_row exactly.
    expected_rows = required_unique

    actual_rows = physical[
        "catalog_row"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        actual_rows,
        expected_rows,
    ):
        raise RuntimeError(
            "Physical catalog-row identity audit failed."
        )

    if not physical[
        "catalog_row"
    ].is_unique:
        raise RuntimeError(
            "catalog_row is not unique in physical catalog subset."
        )

    return physical


# ============================================================================
# Main
# ============================================================================

def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        POSITIVE_INPUT,
        PIE_MC_INPUT,
        CATALOG_COLUMNS_FILE,
        CATALOG_DATA_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    print("=" * 90)
    print("39: PARALLAX-ONLY LENS-MASS INFERENCE")
    print("=" * 90)

    # ========================================================================
    # Load positive-LRT characterization population
    # ========================================================================

    positive_columns = [
        "catalog_row",
        "delta_chi2_lrt",
        "positive_lrt_class",
        "piE_true_amp",
        "h1_piEN",
        "h1_piEE",
    ]

    positive = pd.read_parquet(
        POSITIVE_INPUT,
        columns=positive_columns,
    )

    require_columns(
        positive,
        positive_columns,
        "positive truth table",
    )

    if not positive[
        "catalog_row"
    ].is_unique:
        raise RuntimeError(
            "catalog_row is not unique in positive truth table."
        )

    print()
    print(
        "Positive events =",
        len(positive),
    )

    # ========================================================================
    # Load physical quantities from original catalog
    # ========================================================================

    physical = load_catalog_physical_quantities(
        positive[
            "catalog_row"
        ].to_numpy(dtype=np.int64)
    )

    # ========================================================================
    # Load existing MC intervals from script 22
    # ========================================================================

    mc_columns = [
        "catalog_row",
        "mc_valid_covariance",
        "mc_q16_piE_amp",
        "mc_q50_piE_amp",
        "mc_q84_piE_amp",
    ]

    mc = pd.read_parquet(
        PIE_MC_INPUT,
        columns=mc_columns,
    )

    require_columns(
        mc,
        mc_columns,
        "piE MC table",
    )

    if not mc[
        "catalog_row"
    ].is_unique:
        raise RuntimeError(
            "catalog_row is not unique in piE MC table."
        )

    # ========================================================================
    # Exact one-to-one joins
    # ========================================================================

    df = positive.merge(
        physical,
        on="catalog_row",
        how="left",
        validate="one_to_one",
    )

    df = df.merge(
        mc,
        on="catalog_row",
        how="left",
        validate="one_to_one",
    )

    if len(df) != len(positive):
        raise RuntimeError(
            "Merge changed the positive-population event count."
        )

    if df[
        [
            "lens_mass_msun",
            "thetaE_mas",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Physical catalog join produced missing values."
        )

    # ========================================================================
    # Arrays
    # ========================================================================

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

    mass_true = numeric(
        df,
        "lens_mass_msun",
    )

    thetaE_true = numeric(
        df,
        "thetaE_mas",
    )

    piE_fit = np.hypot(
        piEN_fit,
        piEE_fit,
    )

    # ========================================================================
    # Point-estimate lens mass
    # ========================================================================

    point_valid = (
        np.isfinite(piE_true)
        & (piE_true > 0.0)
        & np.isfinite(piE_fit)
        & (piE_fit > 0.0)
        & np.isfinite(mass_true)
        & (mass_true > 0.0)
        & np.isfinite(thetaE_true)
        & (thetaE_true > 0.0)
    )

    mass_ratio_point = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    mass_fit = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    delta_log10_mass = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    delta_log10_piE = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    mass_ratio_point[
        point_valid
    ] = (
        piE_true[
            point_valid
        ]
        / piE_fit[
            point_valid
        ]
    )

    mass_fit[
        point_valid
    ] = (
        mass_true[
            point_valid
        ]
        * mass_ratio_point[
            point_valid
        ]
    )

    delta_log10_mass[
        point_valid
    ] = np.log10(
        mass_ratio_point[
            point_valid
        ]
    )

    delta_log10_piE[
        point_valid
    ] = np.log10(
        piE_fit[
            point_valid
        ]
        / piE_true[
            point_valid
        ]
    )

    # Exact algebraic identity for this controlled experiment.
    identity_residual = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    identity_residual[
        point_valid
    ] = (
        delta_log10_mass[
            point_valid
        ]
        + delta_log10_piE[
            point_valid
        ]
    )

    # ========================================================================
    # Transform existing piE MC intervals into mass intervals
    # ========================================================================

    q16_piE = numeric(
        df,
        "mc_q16_piE_amp",
    )

    q50_piE = numeric(
        df,
        "mc_q50_piE_amp",
    )

    q84_piE = numeric(
        df,
        "mc_q84_piE_amp",
    )

    mc_flag = (
        df[
            "mc_valid_covariance"
        ]
        .fillna(False)
        .to_numpy(dtype=bool)
    )

    mass_mc_valid = (
        mc_flag
        & np.isfinite(piE_true)
        & (piE_true > 0.0)
        & np.isfinite(mass_true)
        & (mass_true > 0.0)
        & np.isfinite(q16_piE)
        & np.isfinite(q50_piE)
        & np.isfinite(q84_piE)
        & (q16_piE > 0.0)
        & (q50_piE > 0.0)
        & (q84_piE > 0.0)
        & (q16_piE <= q50_piE)
        & (q50_piE <= q84_piE)
    )

    # Since M / M_true = piE_true / piE,
    # no rounded kappa enters this transformation.
    mass_scale = (
        mass_true
        * piE_true
    )

    q16_mass = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    q50_mass = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    q84_mass = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    q16_mass[
        mass_mc_valid
    ] = (
        mass_scale[
            mass_mc_valid
        ]
        / q84_piE[
            mass_mc_valid
        ]
    )

    q50_mass[
        mass_mc_valid
    ] = (
        mass_scale[
            mass_mc_valid
        ]
        / q50_piE[
            mass_mc_valid
        ]
    )

    q84_mass[
        mass_mc_valid
    ] = (
        mass_scale[
            mass_mc_valid
        ]
        / q16_piE[
            mass_mc_valid
        ]
    )

    mass_mc_covered = (
        mass_mc_valid
        & (
            mass_true
            >= q16_mass
        )
        & (
            mass_true
            <= q84_mass
        )
    )

    # ========================================================================
    # Physical relation audit
    # ========================================================================
    #
    # The Sedighe catalog should approximately satisfy
    #
    #     M = theta_E / (kappa pi_E)
    #
    # Small differences can arise from catalog precision and the rounded
    # numerical value of kappa used here.  This is only an audit; all mass
    # recovery ratios above use the catalog mass directly.
    # ========================================================================

    relation_valid = (
        np.isfinite(thetaE_true)
        & (thetaE_true > 0.0)
        & np.isfinite(piE_true)
        & (piE_true > 0.0)
        & np.isfinite(mass_true)
        & (mass_true > 0.0)
    )

    mass_from_relation = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    relation_rel_diff = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    mass_from_relation[
        relation_valid
    ] = (
        thetaE_true[
            relation_valid
        ]
        / (
            KAPPA_MAS_PER_MSUN
            * piE_true[
                relation_valid
            ]
        )
    )

    relation_rel_diff[
        relation_valid
    ] = (
        mass_from_relation[
            relation_valid
        ]
        / mass_true[
            relation_valid
        ]
        - 1.0
    )

    # ========================================================================
    # Save event table
    # ========================================================================

    out = pd.DataFrame(
        {
            "catalog_row":
                df[
                    "catalog_row"
                ].to_numpy(),

            "positive_lrt_class":
                df[
                    "positive_lrt_class"
                ].astype(str).to_numpy(),

            "delta_chi2_lrt":
                numeric(
                    df,
                    "delta_chi2_lrt",
                ),

            "lens_mass_true_msun":
                mass_true,

            "thetaE_true_mas":
                thetaE_true,

            "piE_true_amp":
                piE_true,

            "piE_fit_amp":
                piE_fit,

            "mass_point_valid":
                point_valid,

            "lens_mass_fit_piE_only_msun":
                mass_fit,

            "mass_ratio_fit_over_true":
                mass_ratio_point,

            "delta_log10_mass":
                delta_log10_mass,

            "delta_log10_piE":
                delta_log10_piE,

            "mass_piE_identity_residual":
                identity_residual,

            "mass_mc_valid":
                mass_mc_valid,

            "mass_mc_q16_msun":
                q16_mass,

            "mass_mc_q50_msun":
                q50_mass,

            "mass_mc_q84_msun":
                q84_mass,

            "mass_mc_covered":
                mass_mc_covered,

            "mass_from_true_thetaE_piE_msun":
                mass_from_relation,

            "mass_truth_relation_rel_diff":
                relation_rel_diff,
        }
    )

    out.to_parquet(
        EVENT_TABLE,
        index=False,
    )

    # ========================================================================
    # Summary
    # ========================================================================

    lines = []

    lines.append(
        "LENS-MASS INFERENCE: PARALLAX-ONLY CONTROLLED EXPERIMENT"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"events = {len(out):,}"
    )

    lines.append(
        f"valid point estimates = {int(point_valid.sum()):,}"
    )

    lines.append(
        f"valid MC intervals = {int(mass_mc_valid.sum()):,}"
    )

    if np.any(
        point_valid
    ):

        lines.append("")
        lines.append(
            "Exact transformation audit"
        )

        lines.append(
            "max |delta_log10_mass + delta_log10_piE| = "
            f"{np.nanmax(np.abs(identity_residual)):.6e}"
        )

    if np.any(
        relation_valid
    ):

        absolute_difference = np.abs(
            relation_rel_diff[
                relation_valid
            ]
        )

        lines.append("")
        lines.append(
            "Catalog physical-relation audit"
        )

        lines.append(
            "median |thetaE/(kappa*piE)/Mtrue - 1| = "
            f"{np.nanmedian(absolute_difference):.6e}"
        )

        lines.append(
            "95th percentile = "
            f"{np.nanpercentile(absolute_difference, 95):.6e}"
        )

    classes = (
        df[
            "positive_lrt_class"
        ]
        .astype(str)
        .to_numpy()
    )

    for cls in [
        "confused_fspl_no_parallax",
        "parallax_detected",
    ]:

        class_mask = (
            classes
            == cls
        )

        point_mask = (
            class_mask
            & point_valid
        )

        mc_mask = (
            class_mask
            & mass_mc_valid
        )

        lines.append("")
        lines.append(
            cls
        )

        lines.append(
            "-" * len(cls)
        )

        lines.append(
            f"N = {int(class_mask.sum()):,}"
        )

        if np.any(
            point_mask
        ):

            values = (
                delta_log10_mass[
                    point_mask
                ]
            )

            lines.append(
                "median log10(Mhat/Mtrue) = "
                f"{np.nanmedian(values):.6f}"
            )

            lines.append(
                "16--84% log10(Mhat/Mtrue) = "
                f"[{np.nanpercentile(values, 16):.6f}, "
                f"{np.nanpercentile(values, 84):.6f}]"
            )

            lines.append(
                "median Mhat/Mtrue = "
                f"{np.nanmedian(10.0 ** values):.6f}"
            )

        if np.any(
            mc_mask
        ):

            coverage = np.mean(
                mass_mc_covered[
                    mc_mask
                ]
            )

            lines.append(
                "central-68% mass coverage = "
                f"{coverage:.6f}"
            )

    summary = "\n".join(
        lines
    )

    SUMMARY_FILE.write_text(
        summary
        + "\n"
    )

    print()
    print(summary)

    print()
    print("Saved:")
    print(EVENT_TABLE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
