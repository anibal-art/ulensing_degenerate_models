"""
Provisional chi-square reference calibration for the LRT.

Assumption
----------
Under the Wilks reference approximation,

    Delta chi2_LRT ~ chi2(df=2)   under H0.

For df=2:

    survival(T) = exp(-T / 2)
    c_alpha     = -2 log(alpha)

This module is explicitly provisional.

Once the H0-generated calibration sample is available, the empirical
H0 distribution must replace these theoretical p-values and critical
values.

Negative Delta chi2 values are mapped to T=0 for the theoretical
chi-square reference because chi-square support is T >= 0. Therefore
they receive p=1 and are never classified as detections.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Reference p-values
# ============================================================

def chi2_df2_log_pvalue(delta_chi2):
    """
    Natural logarithm of the provisional chi2_2 survival p-value.

    Returns
        log(p) = -max(T, 0) / 2

    Working in log-space avoids numerical underflow for very large T.
    """

    delta_chi2 = np.asarray(
        delta_chi2,
        dtype=float,
    )

    t = np.maximum(
        delta_chi2,
        0.0,
    )

    return -0.5 * t


def chi2_df2_minus_log10_pvalue(delta_chi2):
    """
    Return -log10(p) under the provisional chi2_2 calibration.

    Since p = exp(-T/2),

        -log10(p) = T / (2 ln 10)

    for T >= 0.
    """

    delta_chi2 = np.asarray(
        delta_chi2,
        dtype=float,
    )

    t = np.maximum(
        delta_chi2,
        0.0,
    )

    return (
        t
        / (
            2.0
            * np.log(10.0)
        )
    )


# ============================================================
# Critical value
# ============================================================

def chi2_df2_critical_value(alpha):
    """
    Critical LRT value for false-positive rate alpha.

        P_H0(T > c_alpha) = alpha

    For chi2 with two degrees of freedom:

        c_alpha = -2 log(alpha)
    """

    alpha = np.asarray(
        alpha,
        dtype=float,
    )

    if np.any(
        (alpha <= 0.0)
        | (alpha >= 1.0)
    ):
        raise ValueError(
            "alpha must satisfy 0 < alpha < 1"
        )

    return -2.0 * np.log(
        alpha
    )
