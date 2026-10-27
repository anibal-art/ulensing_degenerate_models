#!/usr/bin/env python3

import os

# One CPU per SLURM task. Prevent hidden BLAS oversubscription.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# H5 files are read-only in this diagnostic.
# Disable HDF5/NFS advisory locking to avoid errno=37 on CHE.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--sample", required=True)
parser.add_argument("--task-id", required=True, type=int)
parser.add_argument("--chunk-size", default=4, type=int)
parser.add_argument("--out-dir", required=True)

args = parser.parse_args()


# ============================================================
# Repository paths
# ============================================================

WORK = Path(__file__).resolve().parents[2]

PP = WORK / "validation" / "production_profiling"
BA = WORK / "validation" / "bounds_audit"

sys.path.insert(0, str(WORK))
sys.path.insert(0, str(PP))
sys.path.insert(0, str(BA))


# ============================================================
# Frozen production runtime
# ============================================================

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_TRF_X_SCALE"] = "jac"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"

manifest = BA / "data" / "refit_manifest.csv"

old_argv = list(sys.argv)

sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row", "71181",
    "--manifest", str(manifest),
    "--bounds-profile", "production_candidate",
    "--fit-scope", "h1",
    "--dry-run",
]

try:
    import run_bounds_audit_refit_core as core
finally:
    sys.argv = old_argv

import fit_lc

from load_new_event import load_new_case
from pyLIMA.fits import objective_functions


try:
    from erfa import ErfaWarning
    warnings.filterwarnings(
        "ignore",
        category=ErfaWarning,
    )
except Exception:
    pass


# ============================================================
# Flux zero points
# ============================================================

PYLIMA_ZP = 27.4

ZP = {
    "W149": 27.615,
    "u": 27.03,
    "g": 28.38,
    "r": 28.16,
    "i": 27.85,
    "z": 27.46,
    "y": 26.68,
}


def flux_to_pylima(flux, band):

    factor = 10.0 ** (
        (ZP[band] - PYLIMA_ZP)
        / 2.5
    )

    return float(flux) / factor


# ============================================================
# pyLIMA construction
# ============================================================

def build_event(meta):

    lsst_lcs = {
        band: meta["curves"][band]
        for band in [
            "u", "g", "r",
            "i", "z", "y",
        ]
    }

    return fit_lc.create_fit_event(
        meta["Source"],
        str(core.EPHEMERIDES),
        meta["curves"]["W149"],
        lsst_lcs,
        ra=meta["event_ra"],
        dec=meta["event_dec"],
        roman_name="Roman",
    )


def build_model(event, meta, fit_parallax):

    fit_params = (
        fit_lc.initial_params_for_fit_model(
            meta["true_params"],
            "FSPL",
            fit_parallax=fit_parallax,
            fit_defaults=None,
        )
    )

    return fit_lc.build_fit_pyLIMA_model(
        event,
        "FSPL",
        fit_params,
        Origin=None,
        fit_parallax=fit_parallax,
    )


def truth_vector(meta, fit_parallax):

    order = fit_lc.fit_parameter_order(
        "FSPL",
        fit_parallax=fit_parallax,
    )

    return np.asarray(
        [
            float(meta["true_params"][name])
            for name in order
        ],
        dtype=float,
    )


# ============================================================
# Flux handling
# ============================================================

def h5_band_from_telescope(name):

    if name == "Roman":
        return "W149"

    return name


def inject_generator_fluxes(
    params,
    model,
    truth_h5,
):

    result = {}

    for telescope in model.event.telescopes:

        tel = telescope.name
        band = h5_band_from_telescope(tel)

        fs_key = f"fsource_{band}"
        ft_key = f"ftotal_{band}"
        fb_key = f"fblend_{band}"

        if fs_key not in truth_h5:
            raise KeyError(
                f"Missing generator flux {fs_key}"
            )

        fs = flux_to_pylima(
            truth_h5[fs_key],
            band,
        )

        if ft_key in truth_h5:

            ft = flux_to_pylima(
                truth_h5[ft_key],
                band,
            )

        elif fb_key in truth_h5:

            fb = flux_to_pylima(
                truth_h5[fb_key],
                band,
            )

            ft = fs + fb

        else:

            raise KeyError(
                f"Missing {ft_key}/{fb_key}"
            )

        fb = ft - fs

        params[f"fsource_{tel}"] = fs
        params[f"ftotal_{tel}"] = ft
        params[f"fblend_{tel}"] = fb

        if fs != 0.0:
            params[f"gblend_{tel}"] = fb / fs

        result[tel] = {
            "fsource": float(fs),
            "ftotal": float(ft),
            "fblend": float(fb),
        }

    return result


def extract_profiled_fluxes(params, model):

    result = {}

    for telescope in model.event.telescopes:

        tel = telescope.name

        result[tel] = {
            "fsource":
                float(params[f"fsource_{tel}"]),

            "ftotal":
                float(params[f"ftotal_{tel}"]),
        }

    return result


# ============================================================
# Direct chi2 evaluations
# ============================================================

def chi2_fixed_flux(
    model,
    vector,
    truth_h5,
):

    params = model.compute_pyLIMA_parameters(
        vector
    )

    fluxes = inject_generator_fluxes(
        params,
        model,
        truth_h5,
    )

    chi2 = (
        objective_functions
        .all_telescope_photometric_chi2(
            model,
            params,
        )
    )

    return float(chi2), fluxes


def chi2_profiled_flux(
    model,
    vector,
):

    # Important:
    # no explicit fluxes are supplied here.
    # The production bounded-profile patch profiles them.
    params = model.compute_pyLIMA_parameters(
        vector
    )

    chi2 = (
        objective_functions
        .all_telescope_photometric_chi2(
            model,
            params,
        )
    )

    fluxes = extract_profiled_fluxes(
        params,
        model,
    )

    return float(chi2), fluxes


# ============================================================
# Select this task
# ============================================================

sample = pd.read_csv(args.sample)

start = args.task_id * args.chunk_size
stop = min(
    start + args.chunk_size,
    len(sample),
)

if start >= len(sample):
    print(
        f"task={args.task_id}: nothing to do",
        flush=True,
    )
    sys.exit(0)

part = sample.iloc[start:stop].copy()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

out_file = (
    out_dir
    / f"part_{args.task_id:04d}.csv"
)


# ============================================================
# Process events
# ============================================================

records = []

for _, row in part.iterrows():

    catalog_row = int(row["catalog_row"])

    print(
        f"task={args.task_id} "
        f"catalog_row={catalog_row}",
        flush=True,
    )

    try:

        json_path = Path(row["json_path"])
        h5_path = Path(row["h5_path"])

        production = json.loads(
            json_path.read_text()
        )

        meta = load_new_case(
            str(h5_path),
            core,
        )

        with h5py.File(
            h5_path,
            "r",
        ) as f:

            truth_h5 = dict(
                f["pyLIMA_parameters"].attrs
            )

        event = build_event(meta)

        h1_model = build_model(
            event,
            meta,
            fit_parallax=True,
        )

        h0_model = build_model(
            event,
            meta,
            fit_parallax=False,
        )

        x_h1_true = truth_vector(
            meta,
            fit_parallax=True,
        )

        x_h0_true = truth_vector(
            meta,
            fit_parallax=False,
        )


        # ====================================================
        # The four direct evaluations
        # ====================================================

        (
            chi2_h1_generator,
            generator_fluxes,
        ) = chi2_fixed_flux(
            h1_model,
            x_h1_true,
            truth_h5,
        )

        (
            chi2_h0_fixed,
            _,
        ) = chi2_fixed_flux(
            h0_model,
            x_h0_true,
            truth_h5,
        )

        (
            chi2_h1_profile_true,
            h1_profile_fluxes,
        ) = chi2_profiled_flux(
            h1_model,
            x_h1_true,
        )

        (
            chi2_h0_profile_true,
            h0_profile_fluxes,
        ) = chi2_profiled_flux(
            h0_model,
            x_h0_true,
        )


        # ====================================================
        # Frozen production minima
        # ====================================================

        chi2_h1_fit = float(
            production["chi2_h1"]
        )

        chi2_h0_fit = float(
            production["chi2_h0"]
        )

        T_fit = float(
            production["delta_chi2_lrt"]
        )


        # ====================================================
        # Derived decomposition
        # ====================================================

        D_fixed = (
            chi2_h0_fixed
            - chi2_h1_generator
        )

        D_profile = (
            chi2_h0_profile_true
            - chi2_h1_profile_true
        )

        h0_flux_gain = (
            chi2_h0_fixed
            - chi2_h0_profile_true
        )

        h1_flux_gain = (
            chi2_h1_generator
            - chi2_h1_profile_true
        )

        flux_absorption = (
            D_fixed
            - D_profile
        )

        A_h0_physical = (
            chi2_h0_profile_true
            - chi2_h0_fit
        )

        I_h1_physical = (
            chi2_h1_profile_true
            - chi2_h1_fit
        )

        T_decomposition = (
            D_profile
            - A_h0_physical
            + I_h1_physical
        )

        closure = (
            T_fit
            - T_decomposition
        )


        # ====================================================
        # Truth and fitted morphology
        # ====================================================

        truth = meta["true_params"]

        tE_true = float(truth["tE"])

        piEN_true = float(
            truth["piEN"]
        )

        piEE_true = float(
            truth["piEE"]
        )

        piE_true = float(
            np.hypot(
                piEN_true,
                piEE_true,
            )
        )

        X_pi = float(
            piE_true
            * (tE_true / 365.25)**2
        )

        final_h0 = production["final_h0"]
        final_h1 = production["final_h1"]

        h0_tE_fit = float(
            final_h0["tE"]
        )

        h1_tE_fit = float(
            final_h1["tE"]
        )

        n_data = int(
            production[
                "n_photometry_points"
            ]
        )


        # ====================================================
        # Output row
        # ====================================================

        records.append({
            "status": "success",
            "catalog_row": catalog_row,

            "tE_bin_index":
                int(row["tE_bin_index"]),

            "Xpi_bin_index":
                int(row["Xpi_bin_index"]),

            "n_photometry_points":
                n_data,

            "t0_true":
                float(truth["t0"]),

            "u0_true":
                float(truth["u0"]),

            "tE_true":
                tE_true,

            "rho_true":
                float(truth["rho"]),

            "piEN_true":
                piEN_true,

            "piEE_true":
                piEE_true,

            "piE_true":
                piE_true,

            "X_pi":
                X_pi,

            # SIX CENTRAL QUANTITIES
            "chi2_h1_generator":
                chi2_h1_generator,

            "chi2_h0_fixed":
                chi2_h0_fixed,

            "chi2_h1_profile_true":
                chi2_h1_profile_true,

            "chi2_h0_profile_true":
                chi2_h0_profile_true,

            "chi2_h1_fit":
                chi2_h1_fit,

            "chi2_h0_fit":
                chi2_h0_fit,

            # Useful checks
            "chi2_h1_generator_per_N":
                chi2_h1_generator / n_data,

            "D_fixed":
                D_fixed,

            "D_profile":
                D_profile,

            "h0_flux_gain":
                h0_flux_gain,

            "h1_flux_gain":
                h1_flux_gain,

            "flux_absorption":
                flux_absorption,

            "A_h0_physical":
                A_h0_physical,

            "I_h1_physical":
                I_h1_physical,

            "T_fit":
                T_fit,

            "T_from_decomposition":
                T_decomposition,

            "decomposition_closure":
                closure,

            # Morphological displacement
            "h0_tE_fit":
                h0_tE_fit,

            "h1_tE_fit":
                h1_tE_fit,

            "h0_tE_frac_bias":
                (
                    h0_tE_fit
                    - tE_true
                ) / tE_true,

            "h1_tE_frac_bias":
                (
                    h1_tE_fit
                    - tE_true
                ) / tE_true,

            # Flux details
            "generator_fluxes_json":
                json.dumps(
                    generator_fluxes,
                    sort_keys=True,
                ),

            "h1_profile_fluxes_json":
                json.dumps(
                    h1_profile_fluxes,
                    sort_keys=True,
                ),

            "h0_profile_fluxes_json":
                json.dumps(
                    h0_profile_fluxes,
                    sort_keys=True,
                ),
        })

        print(
            f"  T={T_fit:.6g} "
            f"Dprofile={D_profile:.6g} "
            f"A0={A_h0_physical:.6g} "
            f"I1={I_h1_physical:.6g} "
            f"closure={closure:+.2e}",
            flush=True,
        )

    except Exception as exc:

        records.append({
            "status": "failure",
            "catalog_row": catalog_row,
            "error":
                f"{type(exc).__name__}: {exc}",
        })

        print(
            f"  FAILED: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


pd.DataFrame(records).to_csv(
    out_file,
    index=False,
)

print(
    f"saved={out_file}",
    flush=True,
)
