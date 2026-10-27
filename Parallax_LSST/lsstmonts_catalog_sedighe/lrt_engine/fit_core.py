"""Fitting configuration of the hidden-parallax LRT study.

Replaces the import-time bootstrap of validation/bounds_audit/
run_bounds_audit_refit_core.py for production. BOUNDS_PROFILES,
apply_bounds_profile and the H5 light-curve reader are moved verbatim
from that file (commit deeeaae) with tools/extract_module.py. The fits run
on roman_rubin main: log(tE), log(rho) coordinates through
fit_coordinates and the bounded flux profile through flux_method.
"""

import copy

import h5py
import numpy as np

from sedighe_catalog import CONFIG


def apply_bounds_profile(
    bounds,
    *,
    h1,
    profile,
):

    if profile not in BOUNDS_PROFILES:
        raise ValueError(
            f"Unknown bounds profile: {profile!r}"
        )

    spec = BOUNDS_PROFILES[profile]

    out = copy.deepcopy(bounds)

    # Shared H0/H1 nuisance-parameter domain.
    out["u0"] = list(spec["u0"])
    out["tE"] = list(spec["tE"])
    out["rho"] = list(spec["rho"])

    # Alternative-only parallax domain.
    #
    # For audit_legacy, None means preserve the original
    # production parallax bounds exactly.
    if h1:

        if spec["piEN"] is not None:
            out["piEN"] = list(spec["piEN"])

        if spec["piEE"] is not None:
            out["piEE"] = list(spec["piEE"])

    return out


def empty_lc():
    return np.empty(
        (0, 3),
        dtype=float,
    )


def load_h5_lightcurves(path):

    curves = {
        "W149": empty_lc(),
        "u": empty_lc(),
        "g": empty_lc(),
        "r": empty_lc(),
        "i": empty_lc(),
        "z": empty_lc(),
        "y": empty_lc(),
    }

    with h5py.File(path, "r") as f:

        for band in curves:

            if band not in f:
                continue

            g = f[band]

            t = np.asarray(
                g["time"],
                dtype=float,
            )

            m = np.asarray(
                g["mag"],
                dtype=float,
            )

            e = np.asarray(
                g["err_mag"],
                dtype=float,
            )

            if "photometry_keep" in g:

                keep = np.asarray(
                    g["photometry_keep"],
                    dtype=bool,
                )

            else:

                keep = np.ones(
                    len(t),
                    dtype=bool,
                )

            good = (
                keep
                & np.isfinite(t)
                & np.isfinite(m)
                & np.isfinite(e)
                & (e > 0)
            )

            curves[band] = np.column_stack(
                [
                    t[good],
                    m[good],
                    e[good],
                ]
            )

    return curves


# ============================================================
# Production fitting configuration
# ============================================================

H0_BOUNDS = copy.deepcopy(CONFIG["fit"]["fits"]["H0"]["bounds"])
H1_BOUNDS = copy.deepcopy(CONFIG["fit"]["fits"]["H1"]["bounds"])
OPTIMIZER_OPTIONS = copy.deepcopy(CONFIG["fit"]["optimizer_options"])

BOUNDS_PROFILE = "production_candidate"
EPHEMERIDES = None  # Rubin-only study: no Roman ephemerides needed.
FLUX_METHOD = "bounded_profile"

BOUNDS_PROFILES = {

    "audit_legacy": {
        "u0": [-5.0, 5.0],
        "tE": [0.1, 20000.0],
        "rho": [1.0e-7, 10.0],

        # Historical audit kept the production piE bounds.
        "piEN": None,
        "piEE": None,
    },

    "moderate": {
        "u0": [-2.0, 2.0],
        "tE": [0.1, 750.0],
        "rho": [1.0e-7, 1.0],
        "piEN": [-10.0, 10.0],
        "piEE": [-10.0, 10.0],
    },

    "candidate": {
        "u0": [-2.0, 2.0],
        "tE": [0.1, 750.0],
        "rho": [1.0e-7, 2.0],
        "piEN": [-10.0, 10.0],
        "piEE": [-10.0, 10.0],
    },

    "wide": {
        "u0": [-3.0, 3.0],
        "tE": [0.1, 1000.0],
        "rho": [1.0e-7, 2.0],
        "piEN": [-15.0, 15.0],
        "piEE": [-15.0, 15.0],
    },

    "reference5000": {
        "u0": [-5.0, 5.0],
        "tE": [0.1, 5000.0],
        "rho": [1.0e-7, 5.0],
        "piEN": [-20.0, 20.0],
        "piEE": [-20.0, 20.0],
    },

    "control20000": {
        "u0": [-10.0, 10.0],
        "tE": [0.1, 20000.0],
        "rho": [1.0e-7, 10.0],
        "piEN": [-20.0, 20.0],
        "piEE": [-20.0, 20.0],
    },

    "te50000": {
        "u0": [-10.0, 10.0],
        "tE": [0.1, 50000.0],
        "rho": [1.0e-7, 10.0],
        "piEN": [-20.0, 20.0],
        "piEE": [-20.0, 20.0],
    },

    "te100000": {
        "u0": [-10.0, 10.0],
        "tE": [0.1, 100000.0],
        "rho": [1.0e-7, 10.0],
        "piEN": [-20.0, 20.0],
        "piEE": [-20.0, 20.0],
    },

    "te500000": {
        "u0": [-10.0, 10.0],
        "tE": [0.1, 500000.0],
        "rho": [1.0e-7, 10.0],
        "piEN": [-20.0, 20.0],
        "piEE": [-20.0, 20.0],
    },

    "production_candidate": {
        "u0": [-10.0, 10.0],
        "tE": [0.1, 500000.0],
        "rho": [1.0e-7, 10.0],
        "piEN": [-40.0, 40.0],
        "piEE": [-40.0, 40.0],
    },

    "te5000": {
        "u0": [-10.0, 10.0],
        "tE": [0.1, 5000.0],
        "rho": [1.0e-7, 10.0],
        "piEN": [-20.0, 20.0],
        "piEE": [-20.0, 20.0],
    },

    "te1000000": {
        "u0": [-10.0, 10.0],
        "tE": [0.1, 1000000.0],
        "rho": [1.0e-7, 10.0],
        "piEN": [-20.0, 20.0],
        "piEE": [-20.0, 20.0],
    },

    "stress": {
        "u0": [-5.0, 5.0],
        "tE": [0.1, 2000.0],
        "rho": [1.0e-7, 5.0],
        "piEN": [-20.0, 20.0],
        "piEE": [-20.0, 20.0],
    },
}
