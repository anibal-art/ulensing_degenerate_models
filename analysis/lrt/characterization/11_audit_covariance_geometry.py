#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit covariance geometry for the complete H1-generated population.

NO simulation.
NO fitting.

This script reads the enriched population table produced by:

    10_build_h1_parameter_covariance_table.py

and computes dimensionless covariance-geometry diagnostics using the
correlation matrix

    R = D^{-1} C D^{-1},

where

    D = diag(sigma_1, ..., sigma_d).

Why this is needed
------------------
Raw covariance matrices mix parameters with different physical units, so:

    det(C)
    condition_number(C)

are not clean dimensionless measures of degeneracy.

The correlation matrix R removes the marginal scales and allows us to study:

    det(R)
    sqrt(det(R))
    det(R)^(1/(2d))
    condition_number(R)
    eigenvalues(R)

The quantity

    sqrt(det(R))

is the hypervolume factor of the correlated error ellipsoid relative to an
uncorrelated ellipsoid with the same marginal uncertainties.

For a valid correlation matrix:

    0 < det(R) <= 1
    |R_ij| <= 1.

Events violating these conditions are explicitly flagged.

The script also computes log-determinants of C for bookkeeping, but these
must NOT be compared naively between models because they depend on units and
H0/H1 have different dimensions.
"""

from pathlib import Path
from itertools import combinations

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
    / "10_parameter_covariance"
    / "h1_parameter_covariance_characterization.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "characterization"
    / "11_covariance_geometry"
)

OUTPUT = (
    OUTPUT_DIR
    / "h1_covariance_geometry.parquet"
)

SUMMARY = (
    OUTPUT_DIR
    / "covariance_geometry_summary.csv"
)

PAIR_AUDIT = (
    OUTPUT_DIR
    / "correlation_pair_audit.csv"
)


# ============================================================
# Model definitions
# ============================================================

MODELS = {
    "h0": [
        "t0",
        "u0",
        "tE",
        "rho",
    ],

    "h1": [
        "t0",
        "u0",
        "tE",
        "rho",
        "piEN",
        "piEE",
    ],
}


PRIMARY_THRESHOLD = 13.15982011480088


# ============================================================
# Helpers
# ============================================================

def resolve_covariance_column(
    df,
    prefix,
    p1,
    p2,
):
    candidates = [
        f"{prefix}_cov_{p1}_{p2}",
        f"{prefix}_cov_{p2}_{p1}",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise KeyError(
        f"Missing covariance element: "
        f"{prefix} ({p1}, {p2})"
    )


def reconstruct_covariance(
    df,
    prefix,
    parameters,
):
    n = len(df)
    d = len(parameters)

    covariance = np.full(
        (n, d, d),
        np.nan,
        dtype=float,
    )

    for i, p1 in enumerate(
        parameters
    ):

        for j in range(
            i,
            d,
        ):

            p2 = parameters[j]

            column = resolve_covariance_column(
                df,
                prefix,
                p1,
                p2,
            )

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            covariance[:, i, j] = values
            covariance[:, j, i] = values

    return covariance


def assign_lrt_region(
    delta,
):
    region = np.full(
        len(delta),
        "positive_below_threshold",
        dtype=object,
    )

    region[
        delta < 0
    ] = "negative"

    region[
        delta >= PRIMARY_THRESHOLD
    ] = "detected_alpha_1e-3"

    return region


# ============================================================
# Geometry calculation
# ============================================================

def calculate_geometry(
    covariance,
    parameters,
    prefix,
):
    n, d, _ = covariance.shape

    diagonal = np.diagonal(
        covariance,
        axis1=1,
        axis2=2,
    )

    finite_covariance = np.all(
        np.isfinite(covariance),
        axis=(1, 2),
    )

    positive_diagonal = (
        finite_covariance
        & np.all(
            diagonal > 0,
            axis=1,
        )
    )

    # --------------------------------------------------------
    # Sigma
    # --------------------------------------------------------

    sigma = np.full_like(
        diagonal,
        np.nan,
        dtype=float,
    )

    sigma[
        positive_diagonal
    ] = np.sqrt(
        diagonal[
            positive_diagonal
        ]
    )

    # --------------------------------------------------------
    # Correlation matrix
    # --------------------------------------------------------

    correlation = np.full_like(
        covariance,
        np.nan,
        dtype=float,
    )

    denom = (
        sigma[:, :, None]
        * sigma[:, None, :]
    )

    valid_denom = (
        np.isfinite(denom)
        & (denom > 0)
    )

    correlation[
        valid_denom
    ] = (
        covariance[
            valid_denom
        ]
        / denom[
            valid_denom
        ]
    )

    # --------------------------------------------------------
    # Cauchy-Schwarz validity
    # --------------------------------------------------------

    eye = np.eye(
        d,
        dtype=bool,
    )

    offdiag = ~eye

    offdiag_values = correlation[
        :,
        offdiag,
    ]

    max_abs_corr = np.full(
        n,
        np.nan,
        dtype=float,
    )

    finite_offdiag = np.any(
        np.isfinite(
            offdiag_values
        ),
        axis=1,
    )

    safe_abs = np.where(
        np.isfinite(
            offdiag_values
        ),
        np.abs(
            offdiag_values
        ),
        -np.inf,
    )

    max_abs_corr[
        finite_offdiag
    ] = np.max(
        safe_abs[
            finite_offdiag
        ],
        axis=1,
    )

    # Small numerical tolerance only.
    correlation_cauchy_valid = (
        positive_diagonal
        & np.isfinite(
            max_abs_corr
        )
        & (
            max_abs_corr
            <= 1.0 + 1e-8
        )
    )

    # --------------------------------------------------------
    # Output arrays
    # --------------------------------------------------------

    min_eig_R = np.full(
        n,
        np.nan,
        dtype=float,
    )

    max_eig_R = np.full(
        n,
        np.nan,
        dtype=float,
    )

    condition_R = np.full(
        n,
        np.nan,
        dtype=float,
    )

    det_R = np.full(
        n,
        np.nan,
        dtype=float,
    )

    logdet_R = np.full(
        n,
        np.nan,
        dtype=float,
    )

    log10_det_R = np.full(
        n,
        np.nan,
        dtype=float,
    )

    sqrt_det_R = np.full(
        n,
        np.nan,
        dtype=float,
    )

    volume_per_dimension = np.full(
        n,
        np.nan,
        dtype=float,
    )

    effective_rank_R = np.full(
        n,
        np.nan,
        dtype=float,
    )

    positive_definite_R = np.zeros(
        n,
        dtype=bool,
    )

    # Raw C determinant bookkeeping.
    sign_C = np.full(
        n,
        np.nan,
        dtype=float,
    )

    logabsdet_C = np.full(
        n,
        np.nan,
        dtype=float,
    )

    log10_sqrt_det_C = np.full(
        n,
        np.nan,
        dtype=float,
    )

    # --------------------------------------------------------
    # Process only events with valid marginal variances
    # --------------------------------------------------------

    indices = np.flatnonzero(
        positive_diagonal
    )

    chunk_size = 20000

    for start in range(
        0,
        len(indices),
        chunk_size,
    ):

        idx = indices[
            start:
            start + chunk_size
        ]

        R = correlation[
            idx
        ]

        C = covariance[
            idx
        ]

        # --------------------------------------------
        # Raw covariance determinant
        # --------------------------------------------

        sign_c, logdet_c = np.linalg.slogdet(
            C
        )

        sign_C[
            idx
        ] = sign_c

        logabsdet_C[
            idx
        ] = logdet_c

        valid_c_det = (
            sign_c > 0
        )

        if np.any(
            valid_c_det
        ):

            valid_idx = idx[
                valid_c_det
            ]

            log10_sqrt_det_C[
                valid_idx
            ] = (
                0.5
                * logdet_c[
                    valid_c_det
                ]
                / np.log(
                    10.0
                )
            )

        # --------------------------------------------
        # Correlation eigenvalues
        # --------------------------------------------

        finite_R = np.all(
            np.isfinite(R),
            axis=(1, 2),
        )

        if not np.any(
            finite_R
        ):
            continue

        idx_R = idx[
            finite_R
        ]

        R_good = R[
            finite_R
        ]

        eigenvalues = np.linalg.eigvalsh(
            R_good
        )

        minimum = eigenvalues[:, 0]
        maximum = eigenvalues[:, -1]

        min_eig_R[
            idx_R
        ] = minimum

        max_eig_R[
            idx_R
        ] = maximum

        pd_mask = (
            np.isfinite(minimum)
            & np.isfinite(maximum)
            & (minimum > 0)
            & (maximum > 0)
        )

        pd_idx = idx_R[
            pd_mask
        ]

        positive_definite_R[
            pd_idx
        ] = True

        if len(
            pd_idx
        ) == 0:
            continue

        eig_pd = eigenvalues[
            pd_mask
        ]

        R_pd = R_good[
            pd_mask
        ]

        condition_R[
            pd_idx
        ] = (
            eig_pd[:, -1]
            / eig_pd[:, 0]
        )

        # --------------------------------------------
        # Correlation determinant
        # --------------------------------------------

        sign_r, logdet_r = np.linalg.slogdet(
            R_pd
        )

        valid_det = (
            sign_r > 0
            & np.isfinite(
                logdet_r
            )
        )

        valid_idx = pd_idx[
            valid_det
        ]

        valid_logdet = logdet_r[
            valid_det
        ]

        logdet_R[
            valid_idx
        ] = valid_logdet

        log10_det_R[
            valid_idx
        ] = (
            valid_logdet
            / np.log(
                10.0
            )
        )

        # exp(logdet) can underflow for extremely degenerate
        # matrices; the log quantity remains valid.
        det_R[
            valid_idx
        ] = np.exp(
            valid_logdet
        )

        sqrt_det_R[
            valid_idx
        ] = np.exp(
            0.5
            * valid_logdet
        )

        volume_per_dimension[
            valid_idx
        ] = np.exp(
            valid_logdet
            / (
                2.0
                * d
            )
        )

        # --------------------------------------------
        # Effective rank
        # --------------------------------------------

        eig_valid = eig_pd[
            valid_det
        ]

        eig_sum = np.sum(
            eig_valid,
            axis=1,
        )

        probabilities = (
            eig_valid
            / eig_sum[:, None]
        )

        entropy = -np.sum(
            probabilities
            * np.log(
                probabilities
            ),
            axis=1,
        )

        effective_rank_R[
            valid_idx
        ] = np.exp(
            entropy
        )

    return {
        f"{prefix}_corr_max_abs":
            max_abs_corr,

        f"{prefix}_corr_cauchy_valid":
            correlation_cauchy_valid,

        f"{prefix}_corr_positive_definite":
            positive_definite_R,

        f"{prefix}_corr_min_eigenvalue":
            min_eig_R,

        f"{prefix}_corr_max_eigenvalue":
            max_eig_R,

        f"{prefix}_corr_condition_number":
            condition_R,

        f"{prefix}_corr_det":
            det_R,

        f"{prefix}_corr_logdet":
            logdet_R,

        f"{prefix}_corr_log10_det":
            log10_det_R,

        f"{prefix}_corr_sqrt_det":
            sqrt_det_R,

        f"{prefix}_corr_volume_per_dimension":
            volume_per_dimension,

        f"{prefix}_corr_effective_rank":
            effective_rank_R,

        f"{prefix}_cov_sign_det":
            sign_C,

        f"{prefix}_cov_logabsdet":
            logabsdet_C,

        f"{prefix}_cov_log10_sqrt_det":
            log10_sqrt_det_C,
    }


# ============================================================
# Pair-level Cauchy audit
# ============================================================

def audit_pairs(
    df,
    prefix,
    parameters,
):
    rows = []

    for p1, p2 in combinations(
        parameters,
        2,
    ):

        cov_col = resolve_covariance_column(
            df,
            prefix,
            p1,
            p2,
        )

        var1_col = resolve_covariance_column(
            df,
            prefix,
            p1,
            p1,
        )

        var2_col = resolve_covariance_column(
            df,
            prefix,
            p2,
            p2,
        )

        cov = pd.to_numeric(
            df[
                cov_col
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        var1 = pd.to_numeric(
            df[
                var1_col
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        var2 = pd.to_numeric(
            df[
                var2_col
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        denominator = np.sqrt(
            var1
            * var2
        )

        correlation = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        valid = (
            np.isfinite(cov)
            & np.isfinite(denominator)
            & (denominator > 0)
        )

        correlation[
            valid
        ] = (
            cov[
                valid
            ]
            / denominator[
                valid
            ]
        )

        finite = np.isfinite(
            correlation
        )

        violations = (
            finite
            & (
                np.abs(
                    correlation
                )
                > 1.0 + 1e-8
            )
        )

        rows.append(
            {
                "model":
                    prefix,

                "parameter_1":
                    p1,

                "parameter_2":
                    p2,

                "n_finite":
                    int(
                        finite.sum()
                    ),

                "n_abs_corr_gt_1":
                    int(
                        violations.sum()
                    ),

                "fraction_abs_corr_gt_1":
                    (
                        float(
                            violations.sum()
                            / finite.sum()
                        )
                        if finite.sum()
                        else np.nan
                    ),

                "median_abs_corr":
                    (
                        float(
                            np.nanmedian(
                                np.abs(
                                    correlation
                                )
                            )
                        )
                        if finite.sum()
                        else np.nan
                    ),

                "q95_abs_corr":
                    (
                        float(
                            np.nanquantile(
                                np.abs(
                                    correlation
                                ),
                                0.95,
                            )
                        )
                        if finite.sum()
                        else np.nan
                    ),

                "max_abs_corr":
                    (
                        float(
                            np.nanmax(
                                np.abs(
                                    correlation
                                )
                            )
                        )
                        if finite.sum()
                        else np.nan
                    ),
            }
        )

    return rows


# ============================================================
# Summary by LRT region
# ============================================================

def summarize_regions(
    df,
):
    rows = []

    for model in [
        "h0",
        "h1",
    ]:

        for region in [
            "negative",
            "positive_below_threshold",
            "detected_alpha_1e-3",
        ]:

            sub = df[
                df[
                    "lrt_region_geometry"
                ].eq(
                    region
                )
            ]

            n = len(
                sub
            )

            for metric in [
                f"{model}_corr_max_abs",
                f"{model}_corr_min_eigenvalue",
                f"{model}_corr_condition_number",
                f"{model}_corr_log10_det",
                f"{model}_corr_sqrt_det",
                f"{model}_corr_volume_per_dimension",
                f"{model}_corr_effective_rank",
                f"{model}_cov_log10_sqrt_det",
            ]:

                values = pd.to_numeric(
                    sub[
                        metric
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )

                values = values[
                    np.isfinite(
                        values
                    )
                ]

                rows.append(
                    {
                        "model":
                            model,

                        "lrt_region":
                            region,

                        "n_events":
                            n,

                        "metric":
                            metric,

                        "n_finite":
                            len(
                                values
                            ),

                        "median":
                            (
                                float(
                                    np.median(
                                        values
                                    )
                                )
                                if len(
                                    values
                                )
                                else np.nan
                            ),

                        "q10":
                            (
                                float(
                                    np.quantile(
                                        values,
                                        0.10,
                                    )
                                )
                                if len(
                                    values
                                )
                                else np.nan
                            ),

                        "q90":
                            (
                                float(
                                    np.quantile(
                                        values,
                                        0.90,
                                    )
                                )
                                if len(
                                    values
                                )
                                else np.nan
                            ),
                    }
                )

            rows.append(
                {
                    "model":
                        model,

                    "lrt_region":
                        region,

                    "n_events":
                        n,

                    "metric":
                        f"{model}_corr_cauchy_valid_fraction",

                    "n_finite":
                        n,

                    "median":
                        float(
                            sub[
                                f"{model}_corr_cauchy_valid"
                            ].mean()
                        ),

                    "q10":
                        np.nan,

                    "q90":
                        np.nan,
                }
            )

            rows.append(
                {
                    "model":
                        model,

                    "lrt_region":
                        region,

                    "n_events":
                        n,

                    "metric":
                        f"{model}_corr_positive_definite_fraction",

                    "n_finite":
                        n,

                    "median":
                        float(
                            sub[
                                f"{model}_corr_positive_definite"
                            ].mean()
                        ),

                    "q10":
                        np.nan,

                    "q90":
                        np.nan,
                }
            )

    return pd.DataFrame(
        rows
    )


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

    print(
        "=" * 80
    )

    print(
        "Loading enriched H1 population"
    )

    print(
        "=" * 80
    )

    df = pd.read_parquet(
        INPUT
    )

    print(
        "rows =",
        len(
            df
        ),
    )

    delta = pd.to_numeric(
        df[
            "delta_chi2_lrt"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    df[
        "lrt_region_geometry"
    ] = assign_lrt_region(
        delta
    )

    pair_rows = []

    # --------------------------------------------------------
    # H0 and H1
    # --------------------------------------------------------

    for model, parameters in (
        MODELS.items()
    ):

        print()
        print(
            "=" * 80
        )

        print(
            f"{model.upper()} geometry"
        )

        print(
            "=" * 80
        )

        covariance = reconstruct_covariance(
            df,
            model,
            parameters,
        )

        geometry = calculate_geometry(
            covariance,
            parameters,
            model,
        )

        for column, values in (
            geometry.items()
        ):
            df[
                column
            ] = values

        pair_rows.extend(
            audit_pairs(
                df,
                model,
                parameters,
            )
        )

        print(
            "Cauchy-valid =",
            int(
                df[
                    f"{model}_corr_cauchy_valid"
                ].sum()
            ),
            "/",
            len(
                df
            ),
        )

        print(
            "Correlation matrix positive definite =",
            int(
                df[
                    f"{model}_corr_positive_definite"
                ].sum()
            ),
            "/",
            len(
                df
            ),
        )

        print()

        print(
            "max |r|:"
        )

        print(
            df[
                f"{model}_corr_max_abs"
            ].describe(
                percentiles=[
                    0.5,
                    0.9,
                    0.95,
                    0.99,
                ]
            )
        )

        print()

        print(
            "log10 det(R):"
        )

        print(
            df[
                f"{model}_corr_log10_det"
            ].describe(
                percentiles=[
                    0.5,
                    0.9,
                    0.95,
                    0.99,
                ]
            )
        )

        print()

        print(
            "sqrt(det(R)):"
        )

        print(
            df[
                f"{model}_corr_sqrt_det"
            ].describe(
                percentiles=[
                    0.5,
                    0.9,
                    0.95,
                    0.99,
                ]
            )
        )

        print()

        print(
            "condition number R:"
        )

        print(
            df[
                f"{model}_corr_condition_number"
            ].describe(
                percentiles=[
                    0.5,
                    0.9,
                    0.95,
                    0.99,
                ]
            )
        )

    # --------------------------------------------------------
    # Save event-level table
    # --------------------------------------------------------

    df.to_parquet(
        OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Pair-level audit
    # --------------------------------------------------------

    pair_table = pd.DataFrame(
        pair_rows
    )

    pair_table.to_csv(
        PAIR_AUDIT,
        index=False,
    )

    # --------------------------------------------------------
    # Region summary
    # --------------------------------------------------------

    region_summary = summarize_regions(
        df
    )

    region_summary.to_csv(
        SUMMARY,
        index=False,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "LRT REGION AUDIT"
    )

    print(
        "=" * 80
    )

    print(
        df[
            "lrt_region_geometry"
        ].value_counts()
    )

    print()
    print(
        "=" * 80
    )

    print(
        "PAIR-LEVEL |r| > 1 VIOLATIONS"
    )

    print(
        "=" * 80
    )

    violations = pair_table[
        pair_table[
            "n_abs_corr_gt_1"
        ]
        > 0
    ].sort_values(
        "n_abs_corr_gt_1",
        ascending=False,
    )

    if len(
        violations
    ):

        print(
            violations.to_string(
                index=False
            )
        )

    else:

        print(
            "No violations."
        )

    print()
    print(
        "Saved event table:",
        OUTPUT,
    )

    print(
        "Saved region summary:",
        SUMMARY,
    )

    print(
        "Saved pair audit:",
        PAIR_AUDIT,
    )


if __name__ == "__main__":
    main()
