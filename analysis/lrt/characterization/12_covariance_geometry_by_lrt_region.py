#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare covariance geometry across the COMPLETE LRT population.

NO simulation.
NO fitting.

Input
-----
analysis/lrt/results/characterization/11_covariance_geometry/
    h1_covariance_geometry.parquet

Scientific questions
--------------------
1. Are Delta-chi2 < 0 events enriched in invalid or nearly singular
   covariance matrices?

2. Do negative events have a systematically smaller correlation
   hypervolume than positive events?

3. Is the apparent non-positive-definiteness of some covariance matrices
   substantial, or is it consistent with tiny numerical negative
   eigenvalues around zero?

The preferred dimensionless diagnostics are based on the correlation matrix R:

    volume_per_dimension = det(R)^(1/(2d))

    condition_number(R)

    effective_rank(R)

    lambda_min(R) / lambda_max(R)

The last quantity is especially useful for distinguishing:
    genuinely indefinite matrices
from
    almost-singular matrices with tiny negative eigenvalues caused by
    finite numerical precision.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "characterization"
    / "11_covariance_geometry"
    / "h1_covariance_geometry.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "characterization"
    / "12_covariance_geometry_by_lrt"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "covariance_geometry_by_lrt_region.csv"
)

OUTPUT_PSD = (
    OUTPUT_DIR
    / "positive_semidefinite_tolerance_audit.csv"
)


# ============================================================
# Configuration
# ============================================================

MODELS = [
    "h0",
    "h1",
]

REGIONS = [
    "negative",
    "positive_below_threshold",
    "detected_alpha_1e-3",
]

# Relative eigenvalue tolerances:
#
# lambda_min / lambda_max >= -tol
#
# A matrix failing the strict lambda_min > 0 test but satisfying one of
# these tolerances is numerically close to positive semidefinite.
PSD_TOLERANCES = [
    0.0,
    1e-14,
    1e-12,
    1e-10,
    1e-8,
    1e-6,
]


# ============================================================
# Helpers
# ============================================================

def numeric(df, column):
    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


def summarize(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            "n_finite": 0,
            "mean": np.nan,
            "median": np.nan,
            "q10": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q90": np.nan,
        }

    q10, q25, q50, q75, q90 = np.quantile(
        values,
        [
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ],
    )

    return {
        "n_finite":
            int(
                len(values)
            ),

        "mean":
            float(
                np.mean(values)
            ),

        "median":
            float(
                q50
            ),

        "q10":
            float(
                q10
            ),

        "q25":
            float(
                q25
            ),

        "q75":
            float(
                q75
            ),

        "q90":
            float(
                q90
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input not found:\n{INPUT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_parquet(
        INPUT
    )

    print(
        "=" * 90
    )

    print(
        "COVARIANCE GEOMETRY BY LRT REGION"
    )

    print(
        "=" * 90
    )

    print(
        "N total =",
        len(df),
    )

    print()

    print(
        df[
            "lrt_region_geometry"
        ].value_counts()
    )

    summary_rows = []
    psd_rows = []

    # ========================================================
    # Model loop
    # ========================================================

    for model in MODELS:

        print()
        print(
            "=" * 90
        )

        print(
            model.upper()
        )

        print(
            "=" * 90
        )

        # ----------------------------------------------------
        # Relative minimum eigenvalue
        # ----------------------------------------------------

        min_eig = numeric(
            df,
            f"{model}_corr_min_eigenvalue",
        )

        max_eig = numeric(
            df,
            f"{model}_corr_max_eigenvalue",
        )

        relative_min_eigenvalue = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        valid_eigenvalues = (
            np.isfinite(min_eig)
            & np.isfinite(max_eig)
            & (max_eig > 0)
        )

        relative_min_eigenvalue[
            valid_eigenvalues
        ] = (
            min_eig[
                valid_eigenvalues
            ]
            / max_eig[
                valid_eigenvalues
            ]
        )

        df[
            f"{model}_corr_relative_min_eigenvalue"
        ] = relative_min_eigenvalue

        # ----------------------------------------------------
        # Log condition number
        # ----------------------------------------------------

        condition = numeric(
            df,
            f"{model}_corr_condition_number",
        )

        log10_condition = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        valid_condition = (
            np.isfinite(condition)
            & (condition > 0)
        )

        log10_condition[
            valid_condition
        ] = np.log10(
            condition[
                valid_condition
            ]
        )

        df[
            f"{model}_corr_log10_condition_number"
        ] = log10_condition

        # ----------------------------------------------------
        # Region loop
        # ----------------------------------------------------

        for region in REGIONS:

            mask = (
                df[
                    "lrt_region_geometry"
                ].eq(
                    region
                ).to_numpy()
            )

            n_region = int(
                np.sum(mask)
            )

            print()
            print(
                "-" * 70
            )

            print(
                region,
                "N =",
                n_region,
            )

            print(
                "-" * 70
            )

            # ------------------------------------------------
            # Validity fractions
            # ------------------------------------------------

            cauchy = (
                df[
                    f"{model}_corr_cauchy_valid"
                ]
                .to_numpy(
                    dtype=bool
                )
            )

            pd_strict = (
                df[
                    f"{model}_corr_positive_definite"
                ]
                .to_numpy(
                    dtype=bool
                )
            )

            cauchy_fraction = float(
                np.mean(
                    cauchy[
                        mask
                    ]
                )
            )

            pd_fraction = float(
                np.mean(
                    pd_strict[
                        mask
                    ]
                )
            )

            print(
                "Cauchy valid fraction =",
                cauchy_fraction,
            )

            print(
                "strict PD fraction    =",
                pd_fraction,
            )

            # ------------------------------------------------
            # Main metrics
            # ------------------------------------------------

            metrics = {
                "max_abs_corr":
                    numeric(
                        df,
                        f"{model}_corr_max_abs",
                    ),

                "relative_min_eigenvalue":
                    relative_min_eigenvalue,

                "log10_condition_number":
                    log10_condition,

                "log10_det_R":
                    numeric(
                        df,
                        f"{model}_corr_log10_det",
                    ),

                "sqrt_det_R":
                    numeric(
                        df,
                        f"{model}_corr_sqrt_det",
                    ),

                "volume_per_dimension":
                    numeric(
                        df,
                        f"{model}_corr_volume_per_dimension",
                    ),

                "effective_rank":
                    numeric(
                        df,
                        f"{model}_corr_effective_rank",
                    ),
            }

            for metric_name, values in (
                metrics.items()
            ):

                result = summarize(
                    values[
                        mask
                    ]
                )

                summary_rows.append(
                    {
                        "model":
                            model,

                        "lrt_region":
                            region,

                        "n_region":
                            n_region,

                        "metric":
                            metric_name,

                        **result,
                    }
                )

            # ------------------------------------------------
            # PSD tolerance audit
            # ------------------------------------------------

            rel = relative_min_eigenvalue[
                mask
            ]

            finite_rel = np.isfinite(
                rel
            )

            rel_finite = rel[
                finite_rel
            ]

            print()

            print(
                "relative lambda_min/lambda_max:"
            )

            if len(
                rel_finite
            ):

                print(
                    pd.Series(
                        rel_finite
                    ).describe(
                        percentiles=[
                            0.01,
                            0.05,
                            0.10,
                            0.50,
                            0.90,
                        ]
                    )
                )

            for tolerance in (
                PSD_TOLERANCES
            ):

                if len(
                    rel_finite
                ):

                    passes = (
                        rel_finite
                        >= -tolerance
                    )

                    fraction = float(
                        np.mean(
                            passes
                        )
                    )

                    count = int(
                        np.sum(
                            passes
                        )
                    )

                else:

                    fraction = np.nan
                    count = 0

                psd_rows.append(
                    {
                        "model":
                            model,

                        "lrt_region":
                            region,

                        "tolerance":
                            tolerance,

                        "n_with_finite_eigenvalues":
                            int(
                                len(
                                    rel_finite
                                )
                            ),

                        "n_psd_with_tolerance":
                            count,

                        "fraction_psd_with_tolerance":
                            fraction,
                    }
                )

            # ------------------------------------------------
            # Compact stdout
            # ------------------------------------------------

            for name in [
                "volume_per_dimension",
                "log10_condition_number",
                "effective_rank",
            ]:

                row = (
                    summary_rows[
                        -len(metrics):
                    ]
                )

                entry = next(
                    item
                    for item
                    in row
                    if item[
                        "metric"
                    ] == name
                )

                print(
                    f"{name:28s} "
                    f"median={entry['median']:.6g} "
                    f"q10={entry['q10']:.6g} "
                    f"q90={entry['q90']:.6g}"
                )

    # ========================================================
    # Save
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )

    psd = pd.DataFrame(
        psd_rows
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    psd.to_csv(
        OUTPUT_PSD,
        index=False,
    )

    print()
    print(
        "=" * 90
    )

    print(
        "PSD TOLERANCE SUMMARY"
    )

    print(
        "=" * 90
    )

    # Show the most useful tolerance values.
    print(
        psd[
            psd[
                "tolerance"
            ].isin(
                [
                    0.0,
                    1e-12,
                    1e-10,
                    1e-8,
                ]
            )
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        OUTPUT_SUMMARY,
    )

    print(
        "Saved:",
        OUTPUT_PSD,
    )


if __name__ == "__main__":
    main()
