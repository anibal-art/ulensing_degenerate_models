"""
Small dependency-free diagnostics used by the frozen LRT policy.

These helpers are intentionally separated from final_policy.py because
that historical policy imports the obsolete morphology-based fitting
stack. The definitions below are copied exactly from that policy's
goodness-of-fit diagnostics.

They do not run fits and do not alter the fitting policy.
"""

CHI2_DOF_SANITY_THRESHOLD = 50.0


def n_photometry_points(curves):
    return sum(
        len(curves[b])
        for b in ["u", "g", "r", "i", "z", "y"]
        if len(curves[b])
    )


def reduced_chi2(
    fit_record,
    n_points,
    n_params,
):
    dof = max(
        1,
        n_points - n_params,
    )

    return (
        fit_record["chi2"]
        / dof
    )


def chi2_dof_sanity_flag(
    fit_record,
    n_points,
    n_params,
):
    r = reduced_chi2(
        fit_record,
        n_points,
        n_params,
    )

    return (
        r > CHI2_DOF_SANITY_THRESHOLD,
        r,
    )
