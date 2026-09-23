#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build an enriched H1 table for population characterization.

This script performs NO simulation and NO fitting.

Input
-----
analysis/lrt/data/h1_lrt_classified_20260921.parquet

For every H1-generated event it adds:

1. fitted H0 parameters:
       t0, u0, tE, rho

2. fitted H1 parameters:
       t0, u0, tE, rho, piEN, piEE

3. formal 1-sigma uncertainties from the physical-coordinate covariance;

4. every off-diagonal covariance element;

5. every dimensionless correlation coefficient

       r_ij = C_ij / sqrt(C_ii C_jj);

6. covariance-matrix diagnostics:
       minimum eigenvalue,
       maximum eigenvalue,
       condition number,
       maximum |correlation|;

7. H0-H1 displacement in the shared parameters;

8. local parallax Wald statistic

       D_piE^2 = piE^T C_piE^{-1} piE.

The purpose is to study these quantities across the COMPLETE Delta-chi2
distribution and later determine what distinguishes the Delta-chi2 < 0 tail.
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
    / "data"
    / "h1_lrt_classified_20260921.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "characterization"
    / "10_parameter_covariance"
)

OUTPUT = (
    OUTPUT_DIR
    / "h1_parameter_covariance_characterization.parquet"
)

SUMMARY = (
    OUTPUT_DIR
    / "build_summary.txt"
)


# ============================================================
# Parameter definitions
# ============================================================

H0_PARAMETERS = [
    "t0",
    "u0",
    "tE",
    "rho",
]

H1_PARAMETERS = [
    "t0",
    "u0",
    "tE",
    "rho",
    "piEN",
    "piEE",
]


# ============================================================
# Column helpers
# ============================================================

def require_column(df, column):
    if column not in df.columns:
        raise KeyError(
            f"Required column not found: {column}\n"
            f"Available related columns:\n"
            + "\n".join(
                c
                for c in df.columns
                if (
                    "cov" in c.lower()
                    or "pi" in c.lower()
                    or "sigma" in c.lower()
                    or "optimizer" in c.lower()
                )
            )
        )


def parameter_column(prefix, parameter):
    return f"{prefix}_{parameter}"


def covariance_column(prefix, p1, p2):
    """
    Covariances were exported as the upper triangle.

    Try both orientations only for robustness.
    """

    candidates = [
        f"{prefix}_cov_{p1}_{p2}",
        f"{prefix}_cov_{p2}_{p1}",
    ]

    return candidates


def resolve_covariance_column(df, prefix, p1, p2):
    for column in covariance_column(
        prefix,
        p1,
        p2,
    ):
        if column in df.columns:
            return column

    raise KeyError(
        f"Could not find covariance element "
        f"{prefix}: ({p1}, {p2})"
    )


# ============================================================
# Build covariance-derived quantities
# ============================================================

def add_model_covariance_diagnostics(
    df,
    prefix,
    parameters,
):
    """
    Add sigmas, correlations, eigenvalue diagnostics and condition number.

    Covariance matrices are reconstructed event by event from the compact
    upper-triangle columns.
    """

    n_events = len(df)
    n_parameters = len(parameters)

    covariance = np.full(
        (
            n_events,
            n_parameters,
            n_parameters,
        ),
        np.nan,
        dtype=float,
    )

    # --------------------------------------------------------
    # Reconstruct symmetric covariance
    # --------------------------------------------------------

    for i, p1 in enumerate(parameters):

        for j in range(
            i,
            n_parameters,
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

    # --------------------------------------------------------
    # Diagonal -> formal uncertainties
    # --------------------------------------------------------

    diagonal = np.diagonal(
        covariance,
        axis1=1,
        axis2=2,
    )

    for i, parameter in enumerate(
        parameters
    ):
        variance = diagonal[:, i]

        sigma = np.full(
            n_events,
            np.nan,
            dtype=float,
        )

        valid = (
            np.isfinite(variance)
            & (variance >= 0)
        )

        sigma[valid] = np.sqrt(
            variance[valid]
        )

        df[
            f"{prefix}_sigma_{parameter}"
        ] = sigma

    # --------------------------------------------------------
    # Off-diagonal correlations
    # --------------------------------------------------------

    for i, p1 in enumerate(parameters):

        sigma_i = df[
            f"{prefix}_sigma_{p1}"
        ].to_numpy(
            dtype=float
        )

        for j in range(
            i + 1,
            n_parameters,
        ):
            p2 = parameters[j]

            sigma_j = df[
                f"{prefix}_sigma_{p2}"
            ].to_numpy(
                dtype=float
            )

            cov_ij = covariance[
                :,
                i,
                j,
            ]

            denominator = (
                sigma_i
                * sigma_j
            )

            correlation = np.full(
                n_events,
                np.nan,
                dtype=float,
            )

            valid = (
                np.isfinite(cov_ij)
                & np.isfinite(denominator)
                & (denominator > 0)
            )

            correlation[valid] = (
                cov_ij[valid]
                / denominator[valid]
            )

            df[
                f"{prefix}_corr_{p1}_{p2}"
            ] = correlation

    # --------------------------------------------------------
    # Matrix quality diagnostics
    # --------------------------------------------------------

    finite_matrix = np.all(
        np.isfinite(covariance),
        axis=(1, 2),
    )

    min_eigenvalue = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    max_eigenvalue = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    condition_number = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    positive_definite = np.zeros(
        n_events,
        dtype=bool,
    )

    valid_indices = np.flatnonzero(
        finite_matrix
    )

    # Work in chunks to avoid a large temporary allocation.
    chunk_size = 20000

    for start in range(
        0,
        len(valid_indices),
        chunk_size,
    ):

        indices = valid_indices[
            start:
            start + chunk_size
        ]

        eigenvalues = np.linalg.eigvalsh(
            covariance[indices]
        )

        minimum = eigenvalues[:, 0]
        maximum = eigenvalues[:, -1]

        min_eigenvalue[
            indices
        ] = minimum

        max_eigenvalue[
            indices
        ] = maximum

        pd_mask = (
            np.isfinite(minimum)
            & np.isfinite(maximum)
            & (minimum > 0)
            & (maximum > 0)
        )

        positive_definite[
            indices[pd_mask]
        ] = True

        condition_number[
            indices[pd_mask]
        ] = (
            maximum[pd_mask]
            / minimum[pd_mask]
        )

    df[
        f"{prefix}_cov_finite"
    ] = finite_matrix

    df[
        f"{prefix}_cov_positive_definite"
    ] = positive_definite

    df[
        f"{prefix}_cov_min_eigenvalue"
    ] = min_eigenvalue

    df[
        f"{prefix}_cov_max_eigenvalue"
    ] = max_eigenvalue

    df[
        f"{prefix}_cov_condition_number"
    ] = condition_number

    # --------------------------------------------------------
    # Strongest parameter correlation per event
    # --------------------------------------------------------

    correlation_columns = []

    correlation_labels = []

    for p1, p2 in combinations(
        parameters,
        2,
    ):

        column = (
            f"{prefix}_corr_{p1}_{p2}"
        )

        correlation_columns.append(
            column
        )

        correlation_labels.append(
            f"{p1}-{p2}"
        )

    corr_matrix = np.column_stack(
        [
            pd.to_numeric(
                df[column],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )
            for column in correlation_columns
        ]
    )

    abs_corr = np.abs(
        corr_matrix
    )

    safe = np.where(
        np.isfinite(abs_corr),
        abs_corr,
        -np.inf,
    )

    has_valid = np.any(
        np.isfinite(abs_corr),
        axis=1,
    )

    argmax = np.argmax(
        safe,
        axis=1,
    )

    maximum = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    maximum[
        has_valid
    ] = safe[
        np.arange(
            n_events
        )[
            has_valid
        ],
        argmax[
            has_valid
        ],
    ]

    labels = np.full(
        n_events,
        "",
        dtype=object,
    )

    label_array = np.asarray(
        correlation_labels,
        dtype=object,
    )

    labels[
        has_valid
    ] = label_array[
        argmax[
            has_valid
        ]
    ]

    df[
        f"{prefix}_max_abs_corr"
    ] = maximum

    df[
        f"{prefix}_max_abs_corr_pair"
    ] = labels

    return covariance


# ============================================================
# H0-H1 parameter displacement
# ============================================================

def add_h0_h1_parameter_displacement(df):
    """
    Compare the common nonlinear parameters of H0 and H1.

    The normalized displacement is a diagnostic scale only:

        |theta_H1 - theta_H0|
        / sqrt(sigma_H0^2 + sigma_H1^2)

    It is NOT a formal significance because both fits use the same data.
    """

    for parameter in H0_PARAMETERS:

        h0 = pd.to_numeric(
            df[
                f"h0_{parameter}"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        h1 = pd.to_numeric(
            df[
                f"h1_{parameter}"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        sigma_h0 = pd.to_numeric(
            df[
                f"h0_sigma_{parameter}"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        sigma_h1 = pd.to_numeric(
            df[
                f"h1_sigma_{parameter}"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        delta = (
            h1 - h0
        )

        df[
            f"delta_{parameter}_h1_minus_h0"
        ] = delta

        denominator = np.sqrt(
            sigma_h0**2
            + sigma_h1**2
        )

        normalized = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        valid = (
            np.isfinite(delta)
            & np.isfinite(denominator)
            & (denominator > 0)
        )

        normalized[valid] = (
            np.abs(
                delta[valid]
            )
            / denominator[valid]
        )

        df[
            f"separation_{parameter}_h0_h1"
        ] = normalized

    # Positive scale parameters are more naturally compared by ratios.
    for parameter in [
        "tE",
        "rho",
    ]:

        h0 = pd.to_numeric(
            df[
                f"h0_{parameter}"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        h1 = pd.to_numeric(
            df[
                f"h1_{parameter}"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        ratio = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        valid = (
            np.isfinite(h0)
            & np.isfinite(h1)
            & (h0 > 0)
            & (h1 > 0)
        )

        ratio[valid] = np.log10(
            h1[valid]
            / h0[valid]
        )

        df[
            f"log10_{parameter}_h1_over_h0"
        ] = ratio


# ============================================================
# H1 parallax covariance block
# ============================================================

def add_piE_wald(df):
    """
    Compute the local covariance-based parallax statistic:

        D^2 = piE^T C_piE^{-1} piE

    and the fitted parallax amplitude.
    """

    piEN = pd.to_numeric(
        df["h1_piEN"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    piEE = pd.to_numeric(
        df["h1_piEE"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    var_N = pd.to_numeric(
        df["h1_cov_piEN_piEN"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    cov_NE = pd.to_numeric(
        df["h1_cov_piEN_piEE"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    var_E = pd.to_numeric(
        df["h1_cov_piEE_piEE"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    determinant = (
        var_N
        * var_E
        - cov_NE**2
    )

    valid = (
        np.isfinite(piEN)
        & np.isfinite(piEE)
        & np.isfinite(var_N)
        & np.isfinite(cov_NE)
        & np.isfinite(var_E)
        & (var_N > 0)
        & (var_E > 0)
        & (determinant > 0)
    )

    D2 = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    D2[valid] = (
        var_E[valid]
        * piEN[valid] ** 2

        - 2.0
        * cov_NE[valid]
        * piEN[valid]
        * piEE[valid]

        + var_N[valid]
        * piEE[valid] ** 2
    ) / determinant[valid]

    df[
        "h1_piE_amplitude"
    ] = np.hypot(
        piEN,
        piEE,
    )

    df[
        "h1_wald_D2_piE"
    ] = D2

    df[
        "h1_corr_piEN_piEE"
    ] = (
        cov_NE
        / np.sqrt(
            var_N
            * var_E
        )
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
        "Loading H1 classified population"
    )

    print(
        "=" * 80
    )

    df = pd.read_parquet(
        INPUT
    )

    print(
        "rows =",
        len(df),
    )

    print(
        "columns =",
        len(df.columns),
    )

    require_column(
        df,
        "delta_chi2_lrt",
    )

    # --------------------------------------------------------
    # Verify fitted parameters
    # --------------------------------------------------------

    for prefix, parameters in [
        (
            "h0",
            H0_PARAMETERS,
        ),
        (
            "h1",
            H1_PARAMETERS,
        ),
    ]:

        for parameter in parameters:

            require_column(
                df,
                parameter_column(
                    prefix,
                    parameter,
                ),
            )

    # --------------------------------------------------------
    # H0 covariance
    # --------------------------------------------------------

    print()
    print(
        "Reconstructing H0 covariance..."
    )

    add_model_covariance_diagnostics(
        df=df,
        prefix="h0",
        parameters=H0_PARAMETERS,
    )

    # --------------------------------------------------------
    # H1 covariance
    # --------------------------------------------------------

    print(
        "Reconstructing H1 covariance..."
    )

    add_model_covariance_diagnostics(
        df=df,
        prefix="h1",
        parameters=H1_PARAMETERS,
    )

    # --------------------------------------------------------
    # H0-H1 parameter displacement
    # --------------------------------------------------------

    print(
        "Computing H0-H1 parameter displacement..."
    )

    add_h0_h1_parameter_displacement(
        df
    )

    # --------------------------------------------------------
    # Parallax covariance statistic
    # --------------------------------------------------------

    print(
        "Computing H1 parallax covariance diagnostics..."
    )

    add_piE_wald(
        df
    )

    # --------------------------------------------------------
    # Simple LRT-region label
    # --------------------------------------------------------

    delta = pd.to_numeric(
        df[
            "delta_chi2_lrt"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    region = np.full(
        len(df),
        "positive",
        dtype=object,
    )

    region[
        delta < 0
    ] = "negative"

    # Primary calibrated threshold.
    threshold = 13.15982011480088

    region[
        delta >= threshold
    ] = "detected_alpha_1e-3"

    df[
        "lrt_region_primary"
    ] = region

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_parquet(
        OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Audit output
    # --------------------------------------------------------

    lines = []

    lines.append(
        f"rows = {len(df)}"
    )

    lines.append(
        f"columns = {len(df.columns)}"
    )

    lines.append(
        ""
    )

    lines.append(
        "LRT regions:"
    )

    lines.append(
        str(
            df[
                "lrt_region_primary"
            ].value_counts()
        )
    )

    lines.append(
        ""
    )

    for prefix in [
        "h0",
        "h1",
    ]:

        lines.append(
            f"{prefix.upper()} covariance finite:"
        )

        lines.append(
            str(
                df[
                    f"{prefix}_cov_finite"
                ].value_counts(
                    dropna=False
                )
            )
        )

        lines.append(
            ""
        )

        lines.append(
            f"{prefix.upper()} covariance positive definite:"
        )

        lines.append(
            str(
                df[
                    f"{prefix}_cov_positive_definite"
                ].value_counts(
                    dropna=False
                )
            )
        )

        lines.append(
            ""
        )

        lines.append(
            f"{prefix.upper()} max |correlation|:"
        )

        lines.append(
            str(
                df[
                    f"{prefix}_max_abs_corr"
                ].describe(
                    percentiles=[
                        0.5,
                        0.9,
                        0.95,
                        0.99,
                    ]
                )
            )
        )

        lines.append(
            ""
        )

        lines.append(
            f"{prefix.upper()} covariance condition number:"
        )

        lines.append(
            str(
                df[
                    f"{prefix}_cov_condition_number"
                ].describe(
                    percentiles=[
                        0.5,
                        0.9,
                        0.95,
                        0.99,
                    ]
                )
            )
        )

        lines.append(
            ""
        )

    text = "\n".join(
        lines
    )

    SUMMARY.write_text(
        text
        + "\n"
    )

    print()
    print(
        "=" * 80
    )

    print(
        "DONE"
    )

    print(
        "=" * 80
    )

    print(
        text
    )

    print()
    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
