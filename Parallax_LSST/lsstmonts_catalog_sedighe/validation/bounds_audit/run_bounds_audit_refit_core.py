from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import argparse
import copy
import gc
import json
import re
import sys
import traceback

import h5py
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "~/Downloads/hidden_parallax/hidden_parallax_refit_test"
).expanduser().resolve()

ARTIFACTS = ROOT / "artifacts"
MANIFEST = ROOT / "refit_manifest.csv"
OUT = ROOT / "refits"

RR = Path(
    "/home/anibal-pc/microlensing/"
    "simulation_Rubin/roman_rubin"
)

HP = Path(
    "/home/anibal-pc/ulensing_degenerate_models/"
    "Parallax_LSST/lsstmonts_catalog_sedighe"
)

CONFIG_PATH = (
    HP
    / "configs"
    / "config_lsstmonts_baseline_v5p3p5_cluster_che_multifit_LRT_H1ONLY.json"
)

EPHEMERIDES = (
    RR
    / "ephemerides"
    / "Roman_positions.npy"
)

sys.path.insert(0, str(RR))
sys.path.insert(0, str(HP))

import fit_lc

import os

# ============================================================
# EXACT PRODUCTION BOUNDED FLUX PROFILE
# ============================================================
#
# Extracted verbatim from:
# /home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/run_lsstmonts_catalog_hidden_parallax.py
#
# This diagnostic runner must use exactly the same telescope
# flux treatment as the cluster production.
# ============================================================

os.environ["HIDDEN_PARALLAX_BOUNDED_PROFILE"] = "1"

def _bounded_flux_profile_enabled():

    import os as _os

    value = str(
        _os.environ.get(
            "HIDDEN_PARALLAX_BOUNDED_PROFILE",
            "0",
        )
    ).strip().lower()

    return value in {
        "1",
        "true",
        "yes",
        "on",
    }


_BOUNDED_FLUX_PROFILE_PATCH_INSTALLED = False
_ORIGINAL_TRF_INIT_BOUNDED_PROFILE = None
_ORIGINAL_DERIVE_TELESCOPE_FLUX_BOUNDED_PROFILE = None


def _bounded_profile_ftotal_2d(
    magnification,
    flux,
    err_flux,
):
    """
    Exact weighted box-constrained least-squares solution for

        flux = fsource * (A - 1) + ftotal

    with pyLIMA's current bounds

        0 <= fsource <= max(flux)
        0 <= ftotal  <= max(flux)

    Because there are only two linear variables, solve the convex
    quadratic exactly by checking:

        - unconstrained interior solution,
        - fsource = lower,
        - fsource = upper,
        - ftotal  = lower,
        - ftotal  = upper.

    Edge minimizers are clipped to their allowed interval, so corners
    are automatically included.
    """

    import numpy as _np

    A = _np.asarray(
        magnification,
        dtype=float,
    ).reshape(-1)

    y = _np.asarray(
        flux,
        dtype=float,
    ).reshape(-1)

    sigma = _np.asarray(
        err_flux,
        dtype=float,
    ).reshape(-1)

    if not (
        len(A)
        == len(y)
        == len(sigma)
    ):
        raise RuntimeError(
            "bounded profile: incompatible array lengths."
        )

    good = (
        _np.isfinite(A)
        & _np.isfinite(y)
        & _np.isfinite(sigma)
        & (sigma > 0.0)
    )

    if not _np.all(good):
        raise RuntimeError(
            "bounded profile: non-finite photometric data."
        )

    x = A - 1.0

    w = 1.0 / (
        sigma * sigma
    )

    fmax = float(
        _np.max(y)
    )

    if (
        not _np.isfinite(fmax)
        or fmax < 0.0
    ):
        raise RuntimeError(
            "bounded profile: invalid max(flux)."
        )

    # Weighted sufficient statistics.
    sw = float(
        _np.sum(w)
    )

    sx = float(
        _np.sum(w * x)
    )

    sxx = float(
        _np.sum(w * x * x)
    )

    sy = float(
        _np.sum(w * y)
    )

    sxy = float(
        _np.sum(w * x * y)
    )

    if sw <= 0.0:
        raise RuntimeError(
            "bounded profile: non-positive total weight."
        )

    candidates = []

    def _clip(value):

        return float(
            _np.clip(
                value,
                0.0,
                fmax,
            )
        )

    def _add(fs, ft, source):

        fs = _clip(fs)
        ft = _clip(ft)

        residual = (
            y
            - fs * x
            - ft
        ) / sigma

        chi2 = float(
            _np.dot(
                residual,
                residual,
            )
        )

        candidates.append(
            (
                chi2,
                fs,
                ft,
                source,
            )
        )

    # --------------------------------------------------------
    # Unconstrained weighted least-squares interior
    # --------------------------------------------------------

    determinant = (
        sxx * sw
        - sx * sx
    )

    determinant_scale = max(
        abs(
            sxx * sw
        ),
        abs(
            sx * sx
        ),
        1.0,
    )

    if abs(
        determinant
    ) > (
        1.0e-14
        * determinant_scale
    ):

        fs_free = (
            sxy * sw
            - sy * sx
        ) / determinant

        ft_free = (
            sxx * sy
            - sx * sxy
        ) / determinant

        if (
            0.0 <= fs_free <= fmax
            and
            0.0 <= ft_free <= fmax
        ):
            _add(
                fs_free,
                ft_free,
                "interior",
            )

    # --------------------------------------------------------
    # Edge fsource = 0
    # --------------------------------------------------------

    fs = 0.0

    ft = (
        sy - fs * sx
    ) / sw

    _add(
        fs,
        ft,
        "fsource_lower",
    )

    # --------------------------------------------------------
    # Edge fsource = fmax
    # --------------------------------------------------------

    fs = fmax

    ft = (
        sy - fs * sx
    ) / sw

    _add(
        fs,
        ft,
        "fsource_upper",
    )

    # --------------------------------------------------------
    # Edges ftotal = 0 / fmax
    # --------------------------------------------------------

    if sxx > 0.0:

        ft = 0.0

        fs = (
            sxy - ft * sx
        ) / sxx

        _add(
            fs,
            ft,
            "ftotal_lower",
        )

        ft = fmax

        fs = (
            sxy - ft * sx
        ) / sxx

        _add(
            fs,
            ft,
            "ftotal_upper",
        )

    # Degenerate A ~= constant case.
    else:

        _add(
            0.0,
            sy / sw,
            "degenerate",
        )

    if not candidates:
        raise RuntimeError(
            "bounded profile: no finite candidate."
        )

    chi2, fs, ft, source = min(
        candidates,
        key=lambda item: item[0],
    )

    fb = float(
        ft - fs
    )

    bound_tol = (
        1.0e-10
        * max(
            1.0,
            fmax,
        )
    )

    if abs(
        fs
    ) <= bound_tol:
        fs_active = "lower"

    elif abs(
        fs - fmax
    ) <= bound_tol:
        fs_active = "upper"

    else:
        fs_active = "free"

    if abs(
        ft
    ) <= bound_tol:
        ft_active = "lower"

    elif abs(
        ft - fmax
    ) <= bound_tol:
        ft_active = "upper"

    else:
        ft_active = "free"

    return {
        "fsource": float(fs),
        "ftotal": float(ft),
        "fblend": float(fb),
        "chi2": float(chi2),
        "fmax": float(fmax),
        "solution_type": str(source),
        "fsource_active": fs_active,
        "ftotal_active": ft_active,
    }


def _install_bounded_flux_profile_runtime_patch():

    global _BOUNDED_FLUX_PROFILE_PATCH_INSTALLED
    global _ORIGINAL_TRF_INIT_BOUNDED_PROFILE
    global _ORIGINAL_DERIVE_TELESCOPE_FLUX_BOUNDED_PROFILE

    if _BOUNDED_FLUX_PROFILE_PATCH_INSTALLED:
        return

    if not _bounded_flux_profile_enabled():
        return

    # BOUNDED_FLUX_PROFILE_FIX_V2
    from pyLIMA.models import ML_model as _ML_model
    from pyLIMA.fits import TRF_fit as _TRF_fit

    # --------------------------------------------------------
    # Force TRF constructor into pyLIMA's "fluxes not fitted"
    # path. This is essential: fit_parameters is constructed
    # during __init__, so changing the method later is too late.
    # --------------------------------------------------------

    _ORIGINAL_TRF_INIT_BOUNDED_PROFILE = (
        _TRF_fit.TRFfit.__init__
    )

    def _trf_init_bounded_profile(
        self,
        model,
        telescopes_fluxes_method="fit",
        loss_function="chi2",
    ):

        _ORIGINAL_TRF_INIT_BOUNDED_PROFILE(
            self,
            model,
            telescopes_fluxes_method="polyfit",
            loss_function=loss_function,
        )

        # The analytical pyLIMA residual Jacobian includes explicit
        # telescope-flux columns and is therefore not the Jacobian
        # of the reduced profiled problem.
        self.model.Jacobian_flag = "Numerical"

    _TRF_fit.TRFfit.__init__ = (
        _trf_init_bounded_profile
    )

    # --------------------------------------------------------
    # Replace only the implicit-flux branch.
    #
    # If explicit flux parameters are present, preserve original
    # pyLIMA behavior exactly.
    # --------------------------------------------------------

    _ORIGINAL_DERIVE_TELESCOPE_FLUX_BOUNDED_PROFILE = (
        _ML_model.MLmodel.derive_telescope_flux
    )

    def _derive_telescope_flux_bounded_profile(
        self,
        telescope,
        pyLIMA_parameters,
        magnification,
    ):

        key_fs = (
            "fsource_"
            + telescope.name
        )

        explicit_flux = False

        try:
            value = pyLIMA_parameters[
                key_fs
            ]

            explicit_flux = (
                value is not None
            )

        except (
            TypeError,
            KeyError,
            IndexError,
        ):
            explicit_flux = False

        if explicit_flux:

            return (
                _ORIGINAL_DERIVE_TELESCOPE_FLUX_BOUNDED_PROFILE(
                    self,
                    telescope,
                    pyLIMA_parameters,
                    magnification,
                )
            )

        if (
            self.blend_flux_parameter
            != "ftotal"
        ):

            return (
                _ORIGINAL_DERIVE_TELESCOPE_FLUX_BOUNDED_PROFILE(
                    self,
                    telescope,
                    pyLIMA_parameters,
                    magnification,
                )
            )

        lightcurve = (
            telescope.lightcurve
        )

        flux = lightcurve[
            "flux"
        ].value

        err_flux = lightcurve[
            "err_flux"
        ].value

        result = (
            _bounded_profile_ftotal_2d(
                magnification,
                flux,
                err_flux,
            )
        )

        fs = result[
            "fsource"
        ]

        ft = result[
            "ftotal"
        ]

        fb = result[
            "fblend"
        ]

        pyLIMA_parameters[
            key_fs
        ] = fs

        pyLIMA_parameters[
            "fblend_"
            + telescope.name
        ] = fb

        pyLIMA_parameters[
            "ftotal_"
            + telescope.name
        ] = ft

        if fs != 0.0:

            gblend = (
                fb / fs
            )

        elif fb == 0.0:

            gblend = 0.0

        else:

            gblend = float(
                "inf"
            )

        pyLIMA_parameters[
            "gblend_"
            + telescope.name
        ] = gblend

        # Keep the most recent solution for diagnostics.
        if not hasattr(
            self,
            "_bounded_profile_last_fluxes",
        ):

            self._bounded_profile_last_fluxes = {}

        self._bounded_profile_last_fluxes[
            telescope.name
        ] = dict(
            result
        )

        return None

    _ML_model.MLmodel.derive_telescope_flux = (
        _derive_telescope_flux_bounded_profile
    )

    _BOUNDED_FLUX_PROFILE_PATCH_INSTALLED = True

    print(
        "[bounded_flux_profile] "
        "runtime patch installed; "
        "TRF telescope fluxes are profiled with exact box constraints; "
        "Jacobian=Numerical",
        flush=True,
    )


_install_bounded_flux_profile_runtime_patch()


# ============================================================
# TRF X_SCALE BENCHMARK
# ============================================================
#
# Validation-only runtime patch.
#
# Select with:
#
#   HIDDEN_PARALLAX_TRF_X_SCALE=pylima
#   HIDDEN_PARALLAX_TRF_X_SCALE=jac
#   HIDDEN_PARALLAX_TRF_X_SCALE=decade
#
# It changes ONLY scipy.optimize.least_squares(..., x_scale=...).
# Bounds, starts, objective function, loss, tolerances,
# numerical Jacobian and bounded flux profiling are unchanged.
# ============================================================

# TRF_X_SCALE_BENCHMARK_PATCH_V1

def _trf_x_scale_mode():

    import os as _os

    mode = str(
        _os.environ.get(
            "HIDDEN_PARALLAX_TRF_X_SCALE",
            "pylima",
        )
    ).strip().lower()

    allowed = {
        "pylima",
        "jac",
        "decade",
    }

    if mode not in allowed:
        raise ValueError(
            "Invalid HIDDEN_PARALLAX_TRF_X_SCALE="
            f"{mode!r}; expected one of {sorted(allowed)}"
        )

    return mode


def _install_trf_x_scale_runtime_patch():

    import inspect as _inspect
    import textwrap as _textwrap

    from pyLIMA.fits import TRF_fit as _TRF_fit_scale

    mode = _trf_x_scale_mode()

    if mode == "pylima":

        print(
            "[trf_x_scale] "
            "mode=pylima; using original pyLIMA scaling",
            flush=True,
        )

        return

    # --------------------------------------------------------
    # Start from the ACTUAL installed pyLIMA implementation
    # and modify only the single line defining `scaling`.
    # This avoids maintaining a second copy of TRFfit.fit().
    # --------------------------------------------------------

    source = _textwrap.dedent(
        _inspect.getsource(
            _TRF_fit_scale.TRFfit.fit
        )
    )

    old = (
        "    scaling = "
        "10**np.floor(np.log10(np.abs(self.guess)))+1"
    )

    if old not in source:
        raise RuntimeError(
            "Installed TRFfit.fit() does not contain the "
            "expected pyLIMA scaling expression."
        )

    if mode == "jac":

        new = (
            "    scaling = 'jac'"
        )

    elif mode == "decade":

        new = """    _abs_guess = np.abs(
        np.asarray(
            self.guess,
            dtype=float,
        )
    )

    scaling = np.ones_like(
        _abs_guess,
        dtype=float,
    )

    _nonzero = _abs_guess > 0.0

    scaling[_nonzero] = (
        10.0
        ** np.floor(
            np.log10(
                _abs_guess[_nonzero]
            )
        )
    )"""

    else:
        raise RuntimeError(mode)

    patched_source = source.replace(
        old,
        new,
        1,
    )

    namespace = {}

    exec(
        compile(
            patched_source,
            "<hidden-parallax TRF x_scale patch>",
            "exec",
        ),
        _TRF_fit_scale.__dict__,
        namespace,
    )

    _TRF_fit_scale.TRFfit.fit = (
        namespace["fit"]
    )

    print(
        "[trf_x_scale] "
        f"runtime patch installed; mode={mode}",
        flush=True,
    )


_install_trf_x_scale_runtime_patch()


# ============================================================
# TRF COORDINATE BENCHMARK
# ============================================================
#
# Validation-only runtime patch.
#
# Select with:
#
#   HIDDEN_PARALLAX_TRF_COORDS=physical
#   HIDDEN_PARALLAX_TRF_COORDS=log_te_rho
#
# In log_te_rho mode scipy.optimize.least_squares sees:
#
#   (t0, u0, log10(tE), log10(rho), ...)
#
# while pyLIMA's objective function continues to receive the
# original physical parameters:
#
#   (t0, u0, tE, rho, ...)
#
# Bounds and starts therefore remain defined in physical units.
# ============================================================

# TRF_COORDINATE_BENCHMARK_PATCH_V1

def _trf_coordinate_mode():

    import os as _os

    mode = str(
        _os.environ.get(
            "HIDDEN_PARALLAX_TRF_COORDS",
            "physical",
        )
    ).strip().lower()

    allowed = {
        "physical",
        "log_te",
        "log_rho",
        "log_te_rho",
    }

    if mode not in allowed:
        raise ValueError(
            "Invalid HIDDEN_PARALLAX_TRF_COORDS="
            f"{mode!r}; expected one of {sorted(allowed)}"
        )

    return mode


def _install_trf_coordinate_runtime_patch():

    import os as _os
    import time as _time

    import numpy as _np
    import scipy.optimize as _spopt

    from pyLIMA.fits import TRF_fit as _TRF_coord

    mode = _trf_coordinate_mode()

    if mode == "physical":

        print(
            "[trf_coords] "
            "mode=physical; using physical tE,rho coordinates",
            flush=True,
        )

        return

    # Keep x_scale fixed to the pyLIMA baseline so that this
    # experiment changes only the parameterization.
    scale_mode = str(
        _os.environ.get(
            "HIDDEN_PARALLAX_TRF_X_SCALE",
            "pylima",
        )
    ).strip().lower()

    if scale_mode != "pylima":
        raise RuntimeError(
            "For the coordinate benchmark set "
            "HIDDEN_PARALLAX_TRF_X_SCALE=pylima. "
            f"Current value={scale_mode!r}"
        )

    def _fit_log_te_rho(self):

        starting_time = _time.time()

        # ----------------------------------------------------
        # Physical initial guess and physical bounds
        # ----------------------------------------------------

        self.guess = self.initial_guess()

        if self.guess is None:
            return

        keys = list(
            self.fit_parameters.keys()
        )

        x0 = _np.asarray(
            self.guess,
            dtype=float,
        )

        bounds_min = _np.asarray(
            [
                self.fit_parameters[key][1][0]
                for key in keys
            ],
            dtype=float,
        )

        bounds_max = _np.asarray(
            [
                self.fit_parameters[key][1][1]
                for key in keys
            ],
            dtype=float,
        )

        transformed = []

        parameters_to_transform = {
            "log_te": [
                "tE",
            ],
            "log_rho": [
                "rho",
            ],
            "log_te_rho": [
                "tE",
                "rho",
            ],
        }[mode]

        for parameter in parameters_to_transform:

            if parameter in keys:
                transformed.append(
                    (
                        parameter,
                        keys.index(parameter),
                    )
                )

        if not transformed:
            raise RuntimeError(
                "log_te_rho requested but neither "
                "tE nor rho is present."
            )

        for parameter, index in transformed:

            if not (
                x0[index] > 0.0
                and bounds_min[index] > 0.0
                and bounds_max[index] > 0.0
            ):
                raise RuntimeError(
                    f"{parameter} must be strictly positive "
                    "for logarithmic coordinates: "
                    f"guess={x0[index]}, "
                    f"bounds=[{bounds_min[index]}, "
                    f"{bounds_max[index]}]"
                )

        # ----------------------------------------------------
        # Coordinate transformations
        # ----------------------------------------------------

        def _to_internal(x):

            z = _np.asarray(
                x,
                dtype=float,
            ).copy()

            for _, index in transformed:
                z[index] = _np.log10(
                    z[index]
                )

            return z

        def _to_physical(z):

            x = _np.asarray(
                z,
                dtype=float,
            ).copy()

            for _, index in transformed:
                x[index] = (
                    10.0 ** x[index]
                )

            return x

        z0 = _to_internal(x0)
        zmin = _to_internal(bounds_min)
        zmax = _to_internal(bounds_max)

        # ----------------------------------------------------
        # Same pyLIMA scaling prescription, now applied to the
        # coordinates actually seen by TRF.
        # ----------------------------------------------------

        abs_z0 = _np.abs(z0)

        scaling = _np.ones_like(
            abs_z0,
            dtype=float,
        )

        nonzero = abs_z0 > 0.0

        scaling[nonzero] = (
            10.0
            ** _np.floor(
                _np.log10(
                    abs_z0[nonzero]
                )
            )
            + 1.0
        )

        # ----------------------------------------------------
        # Same data counting as pyLIMA TRFfit.fit()
        # ----------------------------------------------------

        n_data = 0

        for telescope in self.model.event.telescopes:

            n_data += telescope.n_data(
                "flux"
            )

            n_data += telescope.n_data(
                "astrometry"
            )

        # Our bounded-flux-profile runtime patch already forces
        # Jacobian_flag="Numerical". The transformed objective
        # also requires the numerical Jacobian.
        jacobian_function = "2-point"

        if self.loss_function == "soft_l1":
            loss = "soft_l1"
        else:
            loss = "linear"

        # ----------------------------------------------------
        # Objective in internal coordinates.
        #
        # pyLIMA itself still receives physical parameters.
        # ----------------------------------------------------

        def _objective_internal(z):

            x = _to_physical(z)

            return self.objective_function(
                x
            )

        trf_fit = _spopt.least_squares(
            _objective_internal,
            z0,
            method="trf",
            bounds=(zmin, zmax),
            max_nfev=50000,
            jac=jacobian_function,
            loss=loss,
            xtol=10**-10,
            ftol=10**-10,
            gtol=10**-10,
            x_scale=scaling,
        )

        z_best = _np.asarray(
            trf_fit["x"],
            dtype=float,
        )

        x_best = _to_physical(
            z_best
        )

        fit_chi2 = (
            trf_fit["cost"]
            * 2.0
        )

        # ----------------------------------------------------
        # Covariance in internal coordinates
        # ----------------------------------------------------

        try:

            jac_z = _np.asarray(
                trf_fit["jac"],
                dtype=float,
            )

            covariance_z = _np.linalg.pinv(
                jac_z.T @ jac_z
            )

        except ValueError:

            jac_z = _np.asarray(
                trf_fit["jac"],
                dtype=float,
            )

            covariance_z = _np.zeros(
                (
                    len(self.fit_parameters),
                    len(self.fit_parameters),
                )
            )

        dof_scale = (
            fit_chi2
            / (
                n_data
                - len(
                    self.model.model_dictionnary
                )
            )
        )

        covariance_z *= dof_scale

        # ----------------------------------------------------
        # Transform covariance back to physical coordinates:
        #
        #   C_x = G C_z G^T
        #
        # where dx/dlog10(x) = ln(10) x.
        # ----------------------------------------------------

        G = _np.eye(
            len(x_best),
            dtype=float,
        )

        for _, index in transformed:

            G[index, index] = (
                _np.log(10.0)
                * x_best[index]
            )

        covariance_x = (
            G
            @ covariance_z
            @ G.T
        )

        # Physical Jacobian, useful for any downstream diagnostic.
        #
        # J_z = J_x dx/dz
        #
        # therefore
        #
        # J_x = J_z dz/dx.
        dz_dx = _np.ones(
            len(x_best),
            dtype=float,
        )

        for _, index in transformed:

            dz_dx[index] = (
                1.0
                / (
                    _np.log(10.0)
                    * x_best[index]
                )
            )

        jac_x = (
            jac_z
            * dz_dx[
                _np.newaxis,
                :
            ]
        )

        # Preserve internal information explicitly before making
        # fit_object externally consistent with physical coordinates.
        trf_fit["x_internal"] = (
            z_best.copy()
        )

        trf_fit["jac_internal"] = (
            jac_z.copy()
        )

        trf_fit["bounds_internal"] = (
            zmin.copy(),
            zmax.copy(),
        )

        trf_fit["coordinate_mode"] = (
            mode
        )

        trf_fit[
            "transformed_parameters"
        ] = [
            parameter
            for parameter, _ in transformed
        ]

        # Expose physical x/jac to downstream code.
        trf_fit["x"] = (
            x_best.copy()
        )

        trf_fit["jac"] = (
            jac_x
        )

        computation_time = (
            _time.time()
            - starting_time
        )

        print(
            self.fit_type()
            + " fit SUCCESS"
        )

        self.fit_results = {
            "best_model":
                x_best,

            self.loss_function:
                fit_chi2,

            "fit_time":
                computation_time,

            "covariance_matrix":
                covariance_x,

            "fit_object":
                trf_fit,
        }

        self.print_fit_results()

    _TRF_coord.TRFfit.fit = (
        _fit_log_te_rho
    )

    print(
        "[trf_coords] "
        "runtime patch installed; "
        f"mode={mode}",
        flush=True,
    )


_install_trf_coordinate_runtime_patch()







# ============================================================
# PRODUCTION CONFIG
# ============================================================

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

H0_BOUNDS = copy.deepcopy(
    CONFIG["fit"]["fits"]["H0"]["bounds"]
)

H1_BOUNDS = copy.deepcopy(
    CONFIG["fit"]["fits"]["H1"]["bounds"]
)

OPTIMIZER_OPTIONS = copy.deepcopy(
    CONFIG["fit"]["optimizer_options"]
)

# ============================================================
# TRUTH-INDEPENDENT BOUNDS PROFILES
# ============================================================
#
# audit_legacy:
#     Exact broad physical domain used by the historical
#     34-event bounds audit. Kept for reproducibility.
#
# moderate / wide / stress:
#     Nested truth-independent domains used for the bounds
#     convergence experiment.
#
# t0 is NOT specified here. It is set later, independently for
# every event, to the full temporal range of the actual fitted
# Rubin photometry:
#
#     t_min <= t0 <= t_max
#
# ============================================================

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


# Default preserves historical behavior.
BOUNDS_PROFILE = "audit_legacy"
FIT_SCOPE = "both"


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


def bounds_output_root():

    # Preserve the exact historical layout for the original
    # full H0+H1 audit.
    if (
        BOUNDS_PROFILE == "audit_legacy"
        and FIT_SCOPE == "both"
    ):
        return OUT

    profile_dir = BOUNDS_PROFILE

    if FIT_SCOPE != "both":
        profile_dir = (
            f"{BOUNDS_PROFILE}_{FIT_SCOPE}only"
        )

    return (
        OUT
        / "bounds_convergence"
        / profile_dir
    )


# Common rho starts.
# The old fitted rho and truth rho are added separately.
RHO_GRID = [
    1.0e-6,
    1.0e-4,
    1.0e-2,
    1.0e-1,
    1.0,
]

# Old H1 multistarts converging closer than this in piE
# are considered the same basin.
PIE_BASIN_TOL = 1.0e-3


# ============================================================
# HELPERS
# ============================================================

def load_npy(path):
    x = np.load(
        path,
        allow_pickle=True,
    )

    if (
        isinstance(x, np.ndarray)
        and x.shape == ()
    ):
        x = x.item()

    return x


def one(paths, name):
    paths = list(paths)

    if len(paths) != 1:
        raise RuntimeError(
            f"Expected 1 {name}, found {len(paths)}:\n"
            + "\n".join(map(str, paths))
        )

    return paths[0]


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


def unique_rhos(values):

    out = []

    for x in values:

        x = float(
            np.clip(
                x,
                1.0001e-7,
                9.999,
            )
        )

        if not any(
            np.isclose(
                x,
                y,
                rtol=1e-8,
                atol=1e-12,
            )
            for y in out
        ):
            out.append(x)

    return out


def safe_label(x):

    return re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        str(x),
    )[:150]


# ============================================================
# LOAD EVENT
# ============================================================

def load_case(record):

    sample = str(
        record["sample"]
    )

    row = int(
        record["catalog_row"]
    )

    case = (
        ARTIFACTS
        / sample
        / str(row)
    )

    h5 = one(
        case.rglob("Event_*.h5"),
        "H5",
    )

    h0_path = one(
        [
            p
            for p in case.rglob(
                "*TRF_FSPL_NoParallax.npy"
            )
            if "_H1_multistart" not in str(p)
        ],
        "H0",
    )

    h1_path = one(
        [
            p
            for p in case.rglob(
                "*TRF_FSPL_Parallax.npy"
            )
            if "_H1_multistart" not in str(p)
        ],
        "H1",
    )

    truth_path = one(
        case.rglob(
            "true_rr_manual_*.parquet"
        ),
        "truth parquet",
    )

    h0 = load_npy(h0_path)
    h1 = load_npy(h1_path)

    truth = pd.read_parquet(
        truth_path
    ).iloc[0]

    curves = load_h5_lightcurves(
        h5
    )

    return {
        "sample": sample,
        "row": row,
        "matched_tail":
            int(record["matched_tail_catalog_row"]),

        "case": case,

        "h0": h0,
        "h1": h1,
        "truth": truth,

        "old_h0":
            np.asarray(
                h0["best_model"],
                dtype=float,
            ),

        "old_h1":
            np.asarray(
                h1["best_model"],
                dtype=float,
            ),

        "old_h0_chi2":
            float(h0["chi2"]),

        "old_h1_chi2":
            float(h1["chi2"]),

        "curves": curves,

        "true_params":
            h1["true_params"],

        "Source":
            int(truth["Source"]),

        "rango":
            int(h1.get("rango", 1)),

        "event_ra":
            float(h1["event_ra"]),

        "event_dec":
            float(h1["event_dec"]),
    }


# ============================================================
# H1 BASINS FROM THE ORIGINAL PRODUCTION
# ============================================================

def h1_anchors(meta):

    candidates = []

    # Production winner.
    candidates.append(
        {
            "label": "old_final",
            "chi2": meta["old_h1_chi2"],
            "v": meta["old_h1"].copy(),
        }
    )

    # All converged multistart solutions.
    for p in sorted(
        meta["case"].rglob(
            "_H1_multistart/**/*Parallax.npy"
        )
    ):

        try:
            obj = load_npy(p)

            v = np.asarray(
                obj["best_model"],
                dtype=float,
            )

            if len(v) < 6:
                continue

            candidates.append(
                {
                    "label":
                        "old_ms_"
                        + p.parent.name,

                    "chi2":
                        float(
                            obj.get(
                                "chi2",
                                np.inf,
                            )
                        ),

                    "v":
                        v[:6].copy(),
                }
            )

        except Exception:
            continue

    # Truth is useful here because this is a controlled simulation,
    # and the production itself already used truth initialization.
    truth = meta["truth"]

    truth_v = np.array(
        [
            truth["t0"],
            truth["u0"],
            truth["tE"],
            truth["rho"],
            truth["piEN"],
            truth["piEE"],
        ],
        dtype=float,
    )

    candidates.append(
        {
            "label": "truth",
            "chi2": np.inf,
            "v": truth_v,
        }
    )

    # Lowest-chi2 representative of each distinct piE basin.
    candidates = sorted(
        candidates,
        key=lambda x: x["chi2"],
    )

    anchors = []

    for c in candidates:

        pie = c["v"][4:6]

        duplicate = any(
            np.linalg.norm(
                pie - a["v"][4:6]
            )
            < PIE_BASIN_TOL
            for a in anchors
        )

        if not duplicate:
            anchors.append(c)

    return anchors



# ============================================================
# OPTIONAL H0 CROSS-SEEDS FOR BOUNDS VALIDATION
# ============================================================

def load_h0_crossseeds(meta):
    """
    Load diagnostic H0 starts discovered in previous validation fits.

    These starts are used only when the environment variable

        HIDDEN_PARALLAX_H0_CROSSSEED_MANIFEST

    is defined.

    This is a validation tool, not a production initialization
    strategy. It is designed to separate parameter-domain
    convergence from optimizer-basin effects.
    """

    text = os.environ.get(
        "HIDDEN_PARALLAX_H0_CROSSSEED_MANIFEST",
        "",
    ).strip()

    if not text:
        return []

    path = Path(
        text
    ).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"H0 cross-seed manifest not found: {path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "catalog_row",
        "sample",
        "source_profile",
        "source_label",
        "source_chi2",
        "t0",
        "u0",
        "tE",
        "rho",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Cross-seed manifest missing columns: "
            + repr(sorted(missing))
        )

    x = df[
        (
            df["catalog_row"].astype(int)
            == int(meta["row"])
        )
        & (
            df["sample"].astype(str)
            == str(meta["sample"])
        )
    ].copy()

    if len(x) == 0:
        return []

    # Data-driven t0 range, identical to run_one_fit().
    all_times = []

    for band in [
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
    ]:

        lc = meta["curves"][band]

        if len(lc):
            all_times.extend(
                np.asarray(
                    lc[:, 0],
                    dtype=float,
                ).tolist()
            )

    if len(all_times) == 0:
        raise RuntimeError(
            "Cannot validate H0 cross-seeds: "
            "no Rubin times."
        )

    t_min = float(
        np.min(all_times)
    )

    t_max = float(
        np.max(all_times)
    )

    spec = BOUNDS_PROFILES[
        BOUNDS_PROFILE
    ]

    starts = []

    for _, r in x.sort_values(
        "source_chi2"
    ).iterrows():

        initial = {
            "t0": float(r["t0"]),
            "u0": float(r["u0"]),
            "tE": float(r["tE"]),
            "rho": float(r["rho"]),
        }

        if not all(
            np.isfinite(
                list(
                    initial.values()
                )
            )
        ):
            continue

        violations = []

        if not (
            t_min
            <= initial["t0"]
            <= t_max
        ):
            violations.append("t0")

        for p in [
            "u0",
            "tE",
            "rho",
        ]:

            lo, hi = spec[p]

            if not (
                lo
                <= initial[p]
                <= hi
            ):
                violations.append(p)

        if violations:

            print(
                "SKIP H0 cross-seed outside current domain:",
                r["source_profile"],
                r["source_label"],
                "violations=",
                violations,
            )

            continue

        label = (
            "crossseed_"
            + str(
                r["source_profile"]
            )
            + "_"
            + str(
                r["source_label"]
            )
        )

        starts.append(
            (
                label,
                initial,
            )
        )

    return starts


# ============================================================
# ONE FIT
# ============================================================

def run_one_fit(
    meta,
    hypothesis,
    initial,
    label,
):

    h1 = (
        hypothesis == "H1"
    )

    bounds = copy.deepcopy(
        H1_BOUNDS
        if h1
        else H0_BOUNDS
    )

    bounds = apply_bounds_profile(
        bounds,
        h1=h1,
        profile=BOUNDS_PROFILE,
    )

    # ========================================================
    # DATA-DRIVEN t0 BOUND
    # ========================================================
    #
    # The production bound t0_truth +/- 30 d is inappropriate
    # for the H0 likelihood minimization because H0 is the
    # misspecified no-parallax model and may need to shift t0
    # substantially to mimic the parallax-distorted light curve.
    #
    # Use instead the full temporal range of the actual fitted
    # photometry. This bound is independent of the truth.
    # ========================================================

    all_times = []

    for band in ["u", "g", "r", "i", "z", "y"]:

        lc = meta["curves"][band]

        if len(lc):
            all_times.extend(
                np.asarray(
                    lc[:, 0],
                    dtype=float,
                ).tolist()
            )

    all_times = np.asarray(
        all_times,
        dtype=float,
    )

    if len(all_times) == 0:
        raise RuntimeError(
            "Cannot construct data-driven t0 bounds: "
            "no Rubin photometry found."
        )

    data_t_min = float(
        np.min(all_times)
    )

    data_t_max = float(
        np.max(all_times)
    )

    t_obs = (
        data_t_max
        - data_t_min
    )

    t0_margin_factor = float(
        os.environ.get(
            "HIDDEN_PARALLAX_T0_MARGIN_FACTOR",
            "0",
        )
    )

    if t0_margin_factor < 0:
        raise ValueError(
            "HIDDEN_PARALLAX_T0_MARGIN_FACTOR "
            "must be >= 0."
        )

    t0_margin = (
        t0_margin_factor
        * t_obs
    )

    t_min = (
        data_t_min
        - t0_margin
    )

    t_max = (
        data_t_max
        + t0_margin
    )

    print(
        "[t0_bounds] "
        f"data=[{data_t_min:.10f}, {data_t_max:.10f}] "
        f"Tobs={t_obs:.10f} "
        f"margin_factor={t0_margin_factor:g} "
        f"fit=[{t_min:.10f}, {t_max:.10f}]",
        flush=True,
    )

    bounds["t0"] = {
        "type": "center_width",
        "center": 0.5 * (t_min + t_max),
        "half_width": 0.5 * (t_max - t_min),
    }

    fit_dir = (
        bounds_output_root()
        / meta["sample"]
        / str(meta["row"])
        / hypothesis
        / safe_label(label)
    )

    fit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log = (
        fit_dir
        / "fit.log"
    )

    record = {
        "sample": meta["sample"],
        "catalog_row": meta["row"],
        "bounds_profile": BOUNDS_PROFILE,
        "hypothesis": hypothesis,
        "label": label,
        "status": "failed",
        "chi2": np.nan,
        "log": str(log),
    }

    try:

        with open(log, "w") as stream:

            with (
                redirect_stdout(stream),
                redirect_stderr(stream)
            ):

                fit, ev, model = (
                    fit_lc.fit_rubin_roman(
                        Source=
                            meta["Source"],

                        event_params=
                            meta["true_params"],

                        path_save=
                            str(fit_dir),

                        path_ephemerides=
                            str(EPHEMERIDES),

                        model="FSPL",
                        algo="TRF",

                        Origin=None,

                        rango=
                            meta["rango"],

                        wfirst_lc=
                            meta["curves"]["W149"],

                        lsst_u=
                            meta["curves"]["u"],

                        lsst_g=
                            meta["curves"]["g"],

                        lsst_r=
                            meta["curves"]["r"],

                        lsst_i=
                            meta["curves"]["i"],

                        lsst_z=
                            meta["curves"]["z"],

                        lsst_y=
                            meta["curves"]["y"],

                        fit_model="FSPL",

                        fit_parallax=h1,

                        fit_defaults=None,

                        fit_bounds=
                            copy.deepcopy(bounds),

                        random_state=
                            20260909
                            + meta["row"],

                        initial_guess=
                            initial,

                        optimizer_options=
                            copy.deepcopy(
                                OPTIMIZER_OPTIONS
                            ),

                        event_ra=
                            meta["event_ra"],

                        event_dec=
                            meta["event_dec"],
                    )
                )

        res = fit.fit_results

        best = np.asarray(
            res["best_model"],
            dtype=float,
        )

        record.update(
            {
                "status": "success",
                "chi2":
                    float(res["chi2"]),

                "t0":
                    float(best[0]),

                "u0":
                    float(best[1]),

                "tE":
                    float(best[2]),

                "rho":
                    float(best[3]),

                "piEN":
                    (
                        float(best[4])
                        if h1
                        else np.nan
                    ),

                "piEE":
                    (
                        float(best[5])
                        if h1
                        else np.nan
                    ),

                "optimizer_success":
                    res.get(
                        "optimizer_success",
                        np.nan,
                    ),

                "optimizer_optimality":
                    res.get(
                        "optimizer_optimality",
                        np.nan,
                    ),

                "optimizer_active_mask":
                    res.get(
                        "optimizer_active_mask",
                        None,
                    ),
            }
        )

        del fit, ev, model
        gc.collect()

    except Exception:

        with open(
            log,
            "a",
        ) as stream:

            stream.write(
                "\n\nEXCEPTION\n"
            )

            traceback.print_exc(
                file=stream,
            )

    return record


# ============================================================
# EVENT
# ============================================================

def run_event(
    record,
    dry=False,
):

    meta = load_case(
        record
    )

    print()
    print("=" * 100)
    print(
        meta["sample"],
        meta["row"],
    )
    print("=" * 100)

    print(
        "bounds profile =",
        BOUNDS_PROFILE,
    )

    print(
        "bounds profile spec =",
        BOUNDS_PROFILES[
            BOUNDS_PROFILE
        ],
    )

    print()

    print(
        "old H0 chi2 =",
        meta["old_h0_chi2"],
    )

    print(
        "old H1 chi2 =",
        meta["old_h1_chi2"],
    )

    print(
        "old LRT =",
        meta["old_h0_chi2"]
        - meta["old_h1_chi2"],
    )

    # --------------------------------------------------------
    # H0:
    # two nuisance anchors:
    #   old H0
    #   truth nuisance
    # crossed with rho grid
    # --------------------------------------------------------

    truth = meta["truth"]

    h0_anchor_vectors = [
        (
            "old_H0",
            meta["old_h0"][:4].copy(),
        ),
        (
            "truth",
            np.array(
                [
                    truth["t0"],
                    truth["u0"],
                    truth["tE"],
                    truth["rho"],
                ],
                dtype=float,
            ),
        ),
    ]

    h0_starts = []
    h0_seen = set()

    for anchor_name, v in h0_anchor_vectors:

        rhos = unique_rhos(
            [
                v[3],
                truth["rho"],
                *RHO_GRID,
            ]
        )

        for rho in rhos:

            key = (
                round(float(v[0]), 5),
                round(float(v[1]), 6),
                round(float(v[2]), 5),
                round(float(rho), 8),
            )

            if key in h0_seen:
                continue

            h0_seen.add(key)

            h0_starts.append(
                (
                    f"{anchor_name}_rho_{rho:.6g}",
                    {
                        "t0": float(v[0]),
                        "u0": float(v[1]),
                        "tE": float(v[2]),
                        "rho": float(rho),
                    },
                )
            )


    # --------------------------------------------------------
    # H0 validation cross-seeds.
    #
    # These are exact previously discovered solutions and are
    # deliberately NOT crossed with RHO_GRID.
    # --------------------------------------------------------

    crossseed_starts = load_h0_crossseeds(
        meta
    )

    n_h0_crossseeds_added = 0

    for label, initial in crossseed_starts:

        key = (
            round(
                float(initial["t0"]),
                5,
            ),
            round(
                float(initial["u0"]),
                6,
            ),
            round(
                float(initial["tE"]),
                5,
            ),
            round(
                float(initial["rho"]),
                8,
            ),
        )

        if key in h0_seen:
            continue

        h0_seen.add(
            key
        )

        h0_starts.append(
            (
                label,
                initial,
            )
        )

        n_h0_crossseeds_added += 1

    print(
        "H0 validation cross-seeds added =",
        n_h0_crossseeds_added,
    )

    # ========================================================
    # OPTIONAL GATE-1 FINAL H0 START PLAN
    # ========================================================
    #
    # Validation-only reduced start plan selected from the
    # extreme100 4-coordinate x 13-start oracle.
    #
    # Default "all" preserves the historical behaviour exactly.
    #
    # The reduced plan is semantic: it is defined by
    #   coordinate system
    #   nuisance anchor
    #   rho initialization rule
    #
    # and does not depend on accidental start-slot numbering.
    # ========================================================

    h0_start_plan = os.environ.get(
        "HIDDEN_PARALLAX_H0_START_PLAN",
        "all",
    ).strip()

    if h0_start_plan == "gate1_final18":

        if n_h0_crossseeds_added != 0:
            raise RuntimeError(
                "gate1_final18 must be run without "
                "H0 validation cross-seeds."
            )

        coordinate_mode = os.environ.get(
            "HIDDEN_PARALLAX_TRF_COORDS",
            "physical",
        ).strip()

        gate1_plan = {
            "physical": [
                ("old_H0", 1.0),
                ("truth", 0.1),
                ("truth", 1.0),
            ],

            "log_te": [
                ("old_H0", "truth_rho"),
                ("old_H0", 0.01),
                ("old_H0", 0.1),
                ("old_H0", 1.0),
                ("truth", 0.01),
                ("truth", 1.0),
            ],

            "log_rho": [
                ("old_H0", "truth_rho"),
                ("old_H0", 0.01),
                ("old_H0", 1.0),
                ("truth", "truth_rho"),
                ("truth", 1e-6),
                ("truth", 1e-4),
                ("truth", 0.1),
                ("truth", 1.0),
            ],

            "log_te_rho": [
                ("truth", 1e-6),
            ],
        }

        if coordinate_mode not in gate1_plan:
            raise RuntimeError(
                "Unsupported coordinate mode for "
                "gate1_final18: "
                f"{coordinate_mode!r}"
            )

        anchor_lookup = {
            name: np.asarray(v, dtype=float)
            for name, v in h0_anchor_vectors
        }

        reduced_starts = []
        reduced_seen = set()

        for anchor_name, rho_rule in gate1_plan[
            coordinate_mode
        ]:

            if anchor_name not in anchor_lookup:
                raise RuntimeError(
                    f"Missing H0 anchor {anchor_name!r}"
                )

            v = anchor_lookup[
                anchor_name
            ]

            if rho_rule == "truth_rho":
                rho = float(
                    truth["rho"]
                )
                rho_tag = "truth"
            else:
                rho = float(
                    rho_rule
                )
                rho_tag = (
                    f"{rho:.6g}"
                )

            key = (
                round(float(v[0]), 5),
                round(float(v[1]), 6),
                round(float(v[2]), 5),
                round(float(rho), 8),
            )

            # In the unusual case that two semantic rules
            # generate exactly the same physical initial point,
            # run that point only once.
            if key in reduced_seen:
                continue

            reduced_seen.add(
                key
            )

            label = (
                f"{anchor_name}_"
                f"rho_{rho_tag}"
            )

            reduced_starts.append(
                (
                    label,
                    {
                        "t0":
                            float(v[0]),
                        "u0":
                            float(v[1]),
                        "tE":
                            float(v[2]),
                        "rho":
                            float(rho),
                    },
                )
            )

        h0_starts = reduced_starts

        print()
        print(
            "[gate1_final18] "
            f"coordinate_mode={coordinate_mode}"
        )

        print(
            "[gate1_final18] "
            f"H0 starts={len(h0_starts)}"
        )

        for i, (
            label,
            initial,
        ) in enumerate(
            h0_starts,
            1,
        ):
            print(
                "[gate1_final18] "
                f"{i:02d} "
                f"{label} "
                f"initial={initial}"
            )

    elif h0_start_plan != "all":

        raise RuntimeError(
            "Unknown "
            "HIDDEN_PARALLAX_H0_START_PLAN="
            f"{h0_start_plan!r}"
        )

    # --------------------------------------------------------
    # H1:
    # every distinct old piE basin + truth
    # crossed with rho grid
    # --------------------------------------------------------

    anchors = h1_anchors(
        meta
    )

    h1_starts = []
    h1_seen = set()

    for i, anchor in enumerate(
        anchors
    ):

        v = anchor["v"]

        rhos = unique_rhos(
            [
                v[3],
                truth["rho"],
                *RHO_GRID,
            ]
        )

        for rho in rhos:

            key = (
                round(float(v[0]), 5),
                round(float(v[1]), 6),
                round(float(v[2]), 5),
                round(float(rho), 8),
                round(float(v[4]), 6),
                round(float(v[5]), 6),
            )

            if key in h1_seen:
                continue

            h1_seen.add(key)

            h1_starts.append(
                (
                    f"a{i:02d}_{anchor['label']}_rho_{rho:.6g}",
                    {
                        "t0": float(v[0]),
                        "u0": float(v[1]),
                        "tE": float(v[2]),
                        "rho": float(rho),
                        "piEN": float(v[4]),
                        "piEE": float(v[5]),
                    },
                )
            )

    # ========================================================
    # EXACT H0 -> H1 NESTEDNESS ANCHOR
    # ========================================================
    #
    # This start is kept separate from h1_anchors() because
    # piE-basin deduplication must never remove the exact
    # embedded H0 solution.
    #
    # H1(theta_H0, piEN=0, piEE=0) is the exact null model
    # embedded inside H1.
    # ========================================================

    n_h0_nested_anchor_added = 0

    h0_anchor_manifest = os.environ.get(
        "HIDDEN_PARALLAX_H0_ANCHOR_MANIFEST",
        "",
    ).strip()

    if h0_anchor_manifest:

        h0_anchor_df = pd.read_csv(
            h0_anchor_manifest
        )

        x = h0_anchor_df[
            pd.to_numeric(
                h0_anchor_df["catalog_row"],
                errors="coerce",
            )
            == int(meta["row"])
        ].copy()

        if "sample" in x.columns:
            x = x[
                x["sample"].astype(str)
                == str(meta["sample"])
            ]

        if len(x) != 1:
            raise RuntimeError(
                "Expected exactly one final H0 anchor "
                f"for row={meta['row']}, "
                f"sample={meta['sample']}; found {len(x)}."
            )

        r = x.iloc[0]

        expected_margin = float(
            r["t0_margin_factor"]
        )

        current_margin = float(
            os.environ.get(
                "HIDDEN_PARALLAX_T0_MARGIN_FACTOR",
                "0",
            )
        )

        if not np.isclose(
            expected_margin,
            current_margin,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "H1 t0 domain does not match the final H0 "
                f"domain for row={meta['row']}: "
                f"H0 margin={expected_margin}, "
                f"H1 margin={current_margin}."
            )

        nested_initial = {
            "t0": float(r["t0"]),
            "u0": float(r["u0"]),
            "tE": float(r["tE"]),
            "rho": float(r["rho"]),
            "piEN": 0.0,
            "piEE": 0.0,
        }

        # Remove an accidentally identical start if one already
        # exists, then insert the explicitly labelled H0 anchor.
        def _h1_start_key(initial):
            return (
                round(float(initial["t0"]), 5),
                round(float(initial["u0"]), 6),
                round(float(initial["tE"]), 5),
                round(float(initial["rho"]), 8),
                round(float(initial["piEN"]), 6),
                round(float(initial["piEE"]), 6),
            )

        nested_key = _h1_start_key(
            nested_initial
        )

        h1_starts = [
            (label, initial)
            for label, initial in h1_starts
            if _h1_start_key(initial)
            != nested_key
        ]

        h1_starts.insert(
            0,
            (
                "H0_NESTED_piE_0",
                nested_initial,
            ),
        )

        n_h0_nested_anchor_added = 1

        print()
        print(
            "[H1 nested anchor] "
            f"row={meta['row']} "
            f"chi2_H0={float(r['chi2_h0']):.12g} "
            f"t0_margin={current_margin:g}"
        )

        print(
            "[H1 nested anchor] "
            f"initial={nested_initial}"
        )

    # ========================================================
    # OPTIONAL CONTROLLED H1 MULTISTART
    # ========================================================
    #
    # Intended for controlled simulations where the generating
    # parameters are known. This deliberately avoids using the
    # historical H1 basin catalogue as a production strategy.
    #
    # controlled4:
    #   1. exact truth
    #   2. final H0 solution embedded at piE = 0
    #   3. truth with half the true parallax vector
    #   4. approximate mirror start:
    #          u0   -> -u0
    #          piEN -> -piEN
    #          piEE unchanged
    #
    # The H0 embedded start is mandatory for numerical nestedness.
    # ========================================================

    h1_start_plan = os.environ.get(
        "HIDDEN_PARALLAX_H1_START_PLAN",
        "all",
    ).strip()

    if h1_start_plan in {
        "controlled4",
        "controlled5",
        "old_final_only",
    }:

        if n_h0_nested_anchor_added != 1:
            raise RuntimeError(
                "HIDDEN_PARALLAX_H1_START_PLAN=controlled4 "
                "requires exactly one final H0 nested anchor. "
                "Set HIDDEN_PARALLAX_H0_ANCHOR_MANIFEST."
            )

        truth_initial = {
            "t0": float(truth["t0"]),
            "u0": float(truth["u0"]),
            "tE": float(truth["tE"]),
            "rho": float(truth["rho"]),
            "piEN": float(truth["piEN"]),
            "piEE": float(truth["piEE"]),
        }

        half_pie_initial = {
            "t0": float(truth["t0"]),
            "u0": float(truth["u0"]),
            "tE": float(truth["tE"]),
            "rho": float(truth["rho"]),
            "piEN": 0.5 * float(truth["piEN"]),
            "piEE": 0.5 * float(truth["piEE"]),
        }

        mirror_initial = {
            "t0": float(truth["t0"]),
            "u0": -float(truth["u0"]),
            "tE": float(truth["tE"]),
            "rho": float(truth["rho"]),
            "piEN": -float(truth["piEN"]),
            "piEE": float(truth["piEE"]),
        }

        candidate_starts = [
            (
                "truth",
                truth_initial,
            ),
            (
                "H0_NESTED_piE_0",
                nested_initial,
            ),
            (
                "truth_half_piE",
                half_pie_initial,
            ),
            (
                "truth_mirror_u0_piEN",
                mirror_initial,
            ),
        ]

        if h1_start_plan == "controlled5":

            old_v = np.asarray(
                meta["old_h1"],
                dtype=float,
            )

            if len(old_v) < 6:
                raise RuntimeError(
                    "old_h1 has fewer than 6 parameters."
                )

            old_final_initial = {
                "t0": float(old_v[0]),
                "u0": float(old_v[1]),
                "tE": float(old_v[2]),
                "rho": float(old_v[3]),
                "piEN": float(old_v[4]),
                "piEE": float(old_v[5]),
            }

            candidate_starts.append(
                (
                    "old_final_reseed",
                    old_final_initial,
                )
            )

        if h1_start_plan == "old_final_only":

            old_v = np.asarray(
                meta["old_h1"],
                dtype=float,
            )

            if len(old_v) < 6:
                raise RuntimeError(
                    "old_h1 has fewer than 6 parameters."
                )

            candidate_starts = [
                (
                    "old_final_reseed",
                    {
                        "t0": float(old_v[0]),
                        "u0": float(old_v[1]),
                        "tE": float(old_v[2]),
                        "rho": float(old_v[3]),
                        "piEN": float(old_v[4]),
                        "piEE": float(old_v[5]),
                    },
                )
            ]

        # Defensive exact-start deduplication.
        controlled = []
        seen = set()

        for label, initial in candidate_starts:

            key = (
                round(float(initial["t0"]), 5),
                round(float(initial["u0"]), 6),
                round(float(initial["tE"]), 5),
                round(float(initial["rho"]), 8),
                round(float(initial["piEN"]), 6),
                round(float(initial["piEE"]), 6),
            )

            if key in seen:
                continue

            seen.add(key)
            controlled.append(
                (
                    label,
                    initial,
                )
            )

        h1_starts = controlled

        print()
        print(
            "[controlled4] "
            f"H1 starts={len(h1_starts)}"
        )

        for i, (label, initial) in enumerate(
            h1_starts,
            1,
        ):
            print(
                "[controlled4] "
                f"{i:02d} "
                f"{label} "
                f"initial={initial}"
            )

    elif h1_start_plan != "all":

        raise RuntimeError(
            "Unknown "
            "HIDDEN_PARALLAX_H1_START_PLAN="
            f"{h1_start_plan!r}"
        )

    print()
    print(
        "H0 starts =",
        len(h0_starts),
    )

    print(
        "H1 distinct piE anchors =",
        len(anchors),
    )

    print(
        "H1 starts =",
        len(h1_starts),
    )

    print()
    print("H1 anchors:")

    for i, a in enumerate(
        anchors
    ):

        v = a["v"]

        print(
            f" {i:02d}",
            a["label"],
            f"piE=({v[4]:+.6f},{v[5]:+.6f})",
            f"rho={v[3]:.6g}",
        )

    if dry:
        return

    results = []

    if FIT_SCOPE in ("both", "h0"):

        print()
        print("RUN H0")

        for i, (label, initial) in enumerate(
            h0_starts,
            1,
        ):

            r = run_one_fit(
                meta,
                "H0",
                initial,
                label,
            )

            results.append(r)

            print(
                f"H0 {i:02d}/{len(h0_starts):02d}",
                r["status"],
                f"chi2={r['chi2']}",
                label,
                flush=True,
            )

    if FIT_SCOPE in ("both", "h1"):

        print()
        print("RUN H1")

        for i, (label, initial) in enumerate(
            h1_starts,
            1,
        ):

            r = run_one_fit(
                meta,
                "H1",
                initial,
                label,
            )

            results.append(r)

            print(
                f"H1 {i:02d}/{len(h1_starts):02d}",
                r["status"],
                f"chi2={r['chi2']}",
                label,
                flush=True,
            )

    df = pd.DataFrame(
        results
    )

    event_out = (
        bounds_output_root()
        / meta["sample"]
        / str(meta["row"])
    )

    event_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        event_out
        / "all_refits.csv",
        index=False,
    )

    def new_min(
        hypothesis,
        old,
    ):

        x = df[
            (df["hypothesis"] == hypothesis)
            & (df["status"] == "success")
            & np.isfinite(df["chi2"])
        ]

        new = float(old)
        label = "OLD_PRODUCTION_BASELINE"

        if len(x):

            j = x["chi2"].idxmin()

            if float(
                x.loc[j, "chi2"]
            ) < new:

                new = float(
                    x.loc[j, "chi2"]
                )

                label = str(
                    x.loc[j, "label"]
                )

        return new, label

    new_h0, label_h0 = new_min(
        "H0",
        meta["old_h0_chi2"],
    )

    new_h1, label_h1 = new_min(
        "H1",
        meta["old_h1_chi2"],
    )

    improve_h0 = (
        meta["old_h0_chi2"]
        - new_h0
    )

    improve_h1 = (
        meta["old_h1_chi2"]
        - new_h1
    )

    old_lrt = (
        meta["old_h0_chi2"]
        - meta["old_h1_chi2"]
    )

    new_lrt = (
        new_h0
        - new_h1
    )

    summary = {
        "sample":
            meta["sample"],

        "input_manifest":
            str(MANIFEST_PATH),

        "fit_scope":
            FIT_SCOPE,

        "bounds_profile":
            BOUNDS_PROFILE,

        "bounds_profile_spec":
            copy.deepcopy(
                BOUNDS_PROFILES[
                    BOUNDS_PROFILE
                ]
            ),

        "catalog_row":
            meta["row"],

        "matched_tail_catalog_row":
            meta["matched_tail"],

        "old_h0_chi2":
            meta["old_h0_chi2"],

        "new_h0_chi2":
            new_h0,

        "improve_h0":
            improve_h0,

        "old_h1_chi2":
            meta["old_h1_chi2"],

        "new_h1_chi2":
            new_h1,

        "improve_h1":
            improve_h1,

        "old_lrt":
            old_lrt,

        "new_lrt":
            new_lrt,

        "delta_lrt":
            new_lrt
            - old_lrt,

        "best_h0":
            label_h0,

        "best_h1":
            label_h1,

        "n_h0_starts":
            len(h0_starts),

        "n_h0_fits_executed":
            int(
                (
                    df["hypothesis"] == "H0"
                ).sum()
            ),

        "n_h1_fits_executed":
            int(
                (
                    df["hypothesis"] == "H1"
                ).sum()
            ),

        "n_h0_crossseeds_added":
            n_h0_crossseeds_added,

        "h0_crossseed_manifest":
            os.environ.get(
                "HIDDEN_PARALLAX_H0_CROSSSEED_MANIFEST",
                None,
            ),

        "n_h1_anchors":
            len(anchors),

        "n_h1_starts":
            len(h1_starts),

        "h1_start_plan":
            h1_start_plan,

        "n_h0_nested_anchor_added":
            n_h0_nested_anchor_added,
    }

    with open(
        event_out
        / "summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("RESULT")
    print("-" * 80)

    for k, v in summary.items():
        print(
            f"{k:28s}",
            v,
        )


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--catalog-row",
    type=int,
    required=True,
)

parser.add_argument(
    "--manifest",
    type=Path,
    default=None,
    help=(
        "Explicit event manifest. If omitted, use the historical "
        "34-event bounds-audit manifest."
    ),
)

parser.add_argument(
    "--bounds-profile",
    choices=[
        "audit_legacy",
        "moderate",
        "candidate",
        "wide",
        "reference5000",
        "control20000",
        "te1000000",
        "te5000",
        "te50000",
        "te100000",
        "te500000",
        "production_candidate",
        "stress",
    ],
    default="audit_legacy",
    help=(
        "Truth-independent physical-bounds profile. "
        "Default audit_legacy preserves the historical "
        "34-event audit."
    ),
)

parser.add_argument(
    "--fit-scope",
    choices=[
        "both",
        "h0",
        "h1",
    ],
    default="both",
    help=(
        "Which hypothesis to refit. Default 'both' preserves "
        "historical behavior. Use 'h0' for shared-bounds "
        "validation without rerunning expensive H1 fits."
    ),
)

parser.add_argument(
    "--dry-run",
    action="store_true",
)

parser.add_argument(
    "--force",
    action="store_true",
)

args = parser.parse_args()

BOUNDS_PROFILE = args.bounds_profile
FIT_SCOPE = args.fit_scope

MANIFEST_PATH = (
    Path(args.manifest).expanduser().resolve()
    if args.manifest is not None
    else Path(MANIFEST).expanduser().resolve()
)

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

print(
    "manifest         =",
    MANIFEST_PATH,
)

manifest = pd.read_csv(
    MANIFEST_PATH
)

selected = manifest[
    manifest["catalog_row"]
    == args.catalog_row
]

if len(selected) != 1:
    raise RuntimeError(
        f"catalog_row {args.catalog_row}: "
        f"{len(selected)} manifest matches"
    )

record = selected.iloc[0]

sample = str(
    record["sample"]
)

summary_path = (
    bounds_output_root()
    / sample
    / str(args.catalog_row)
    / "summary.json"
)

if (
    summary_path.exists()
    and not args.force
    and not args.dry_run
):

    print(
        "Already completed:",
        summary_path,
    )

    sys.exit(0)

run_event(
    record,
    dry=args.dry_run,
)
