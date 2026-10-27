#!/usr/bin/env python3

"""
Measure the chi2 hierarchy for a stratified sample of H1-generated events.

NO nonlinear fits are run here.

For each event we save the six central quantities:

    1. chi2_h1_generator
       H1 at true physical parameters + true generating fluxes.

    2. chi2_h0_fixed
       H0 at the same true shared physical parameters + same true
       generating fluxes, with parallax removed.

    3. chi2_h1_profile_true
       H1 at true physical parameters, with nuisance fluxes profiled.

    4. chi2_h0_profile_true
       H0 at true shared physical parameters, with nuisance fluxes profiled.

    5. chi2_h1_fit
       Stored production H1 minimum.

    6. chi2_h0_fit
       Stored production H0 minimum.

Derived quantities:

    D_fixed =
        chi2_h0_fixed - chi2_h1_generator

    D_profile =
        chi2_h0_profile_true - chi2_h1_profile_true

    flux_absorption =
        D_fixed - D_profile

    A_h0_physical =
        chi2_h0_profile_true - chi2_h0_fit

    I_h1_physical =
        chi2_h1_profile_true - chi2_h1_fit

    T_fit =
        chi2_h0_fit - chi2_h1_fit

Exact decomposition:

    T_fit = D_profile - A_h0_physical + I_h1_physical

The truth information is used only as a simulation diagnostic.
"""

import argparse
import hashlib
import heapq
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--run",
    required=True,
)

parser.add_argument(
    "--per-cell",
    type=int,
    default=5,
    help="Events per (tE, Xpi) cell.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=20260927,
)

parser.add_argument(
    "--sample-csv",
    default=None,
)

parser.add_argument(
    "--overwrite",
    action="store_true",
)

args = parser.parse_args()


# ============================================================
# Paths
# ============================================================

WORK = Path(__file__).resolve().parents[2]

PP = (
    WORK
    / "validation"
    / "production_profiling"
)

BA = (
    WORK
    / "validation"
    / "bounds_audit"
)

RUN = Path(args.run).resolve()

OUT = (
    PP
    / "results"
    / "h1_turnover_chi2_layers"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SAMPLE_PATH = (
    OUT
    / "turnover_sample_manifest.csv"
)

RESULT_PATH = (
    OUT
    / "turnover_chi2_layers.csv"
)

SUMMARY_PATH = (
    OUT
    / "turnover_chi2_layers_summary.csv"
)

TEXT_PATH = (
    OUT
    / "turnover_chi2_layers_summary.txt"
)


if not RUN.is_dir():
    raise FileNotFoundError(RUN)


# ============================================================
# Imports from production environment
# ============================================================

sys.path.insert(
    0,
    str(PP),
)

sys.path.insert(
    0,
    str(BA),
)


os.environ[
    "HIDDEN_PARALLAX_TRF_COORDS"
] = "physical"

os.environ[
    "HIDDEN_PARALLAX_TRF_X_SCALE"
] = "jac"

os.environ[
    "HIDDEN_PARALLAX_T0_MARGIN_FACTOR"
] = "0"


# Importing this core installs the exact bounded-flux-profile
# runtime patch used in production.
CORE_MANIFEST = (
    BA
    / "data"
    / "refit_manifest.csv"
)

old_argv = list(sys.argv)

sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row",
    "71181",
    "--manifest",
    str(CORE_MANIFEST),
    "--bounds-profile",
    "production_candidate",
    "--fit-scope",
    "h1",
    "--dry-run",
]

try:
    import run_bounds_audit_refit_core as core
finally:
    sys.argv = old_argv


import fit_lc

from load_new_event import load_new_case
from pyLIMA.fits import objective_functions


# Hide the harmless "dubious year" ERFA messages.
try:
    from erfa import ErfaWarning

    warnings.filterwarnings(
        "ignore",
        category=ErfaWarning,
    )
except Exception:
    pass


# ============================================================
# Photometric zero points
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


def flux_to_pylima(
    flux,
    band,
):
    """
    Convert generator flux from the survey-band zero point
    to pyLIMA's internal ZP=27.4 flux system.
    """

    factor = 10.0 ** (
        (
            ZP[band]
            - PYLIMA_ZP
        )
        / 2.5
    )

    return (
        float(flux)
        / factor
    )


def telescope_h5_band(
    telescope_name,
):
    """
    pyLIMA telescope name can be Roman while H5 uses W149.
    """

    if telescope_name == "Roman":
        return "W149"

    return telescope_name


# ============================================================
# Sampling grid
# ============================================================

TE_EDGES = np.array(
    [
        0.0,
        60.0,
        100.0,
        200.0,
        365.0,
        np.inf,
    ],
    dtype=float,
)

XPI_EDGES = np.array(
    [
        0.0,
        0.01,
        0.03,
        0.08,
        0.15,
        0.30,
        1.00,
        np.inf,
    ],
    dtype=float,
)


def interval_label(
    edges,
    i,
):
    lo = edges[i]
    hi = edges[i + 1]

    if np.isinf(hi):
        return f"[{lo:g},inf)"

    return f"[{lo:g},{hi:g})"


TE_LABELS = [
    interval_label(
        TE_EDGES,
        i,
    )
    for i
    in range(
        len(TE_EDGES) - 1
    )
]

XPI_LABELS = [
    interval_label(
        XPI_EDGES,
        i,
    )
    for i
    in range(
        len(XPI_EDGES) - 1
    )
]


# ============================================================
# Generic helpers
# ============================================================

def finite_float(
    value,
):
    try:
        x = float(value)
    except Exception:
        return None

    if not np.isfinite(x):
        return None

    return x


def truth_get(
    truth,
    name,
):
    candidates = [
        name,
        f"{name}_true",
        f"true_{name}",
    ]

    for key in candidates:
        if key in truth:
            value = finite_float(
                truth[key]
            )

            if value is not None:
                return value

    raise KeyError(
        f"Could not find {name}; "
        f"available={sorted(truth)}"
    )


def find_bin(
    value,
    edges,
):
    if not np.isfinite(value):
        return None

    if value < edges[0]:
        return None

    i = int(
        np.searchsorted(
            edges,
            value,
            side="right",
        )
        - 1
    )

    if i < 0:
        return None

    if i >= len(edges) - 1:
        i = len(edges) - 2

    return i


def deterministic_priority(
    catalog_row,
    seed,
):
    payload = (
        f"{seed}:{catalog_row}"
    ).encode()

    digest = hashlib.blake2b(
        payload,
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        "big",
        signed=False,
    )


def resolve_h5_path(
    json_path,
    data,
    catalog_row,
):
    chunk = (
        json_path
        .parent
        .parent
    )

    candidate = (
        chunk
        / "h5"
        / f"Event_{catalog_row}.h5"
    )

    if candidate.is_file():
        return candidate

    stored = data.get(
        "h5_path"
    )

    if stored:
        candidate = Path(
            str(stored)
        )

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No H5 for row={catalog_row}"
    )


# ============================================================
# Event/model construction
# ============================================================

def build_event(
    meta,
):
    lsst_lcs = {
        band:
            meta["curves"][band]
        for band in [
            "u",
            "g",
            "r",
            "i",
            "z",
            "y",
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


def build_model(
    event,
    meta,
    fit_parallax,
):
    fit_params = (
        fit_lc.initial_params_for_fit_model(
            meta["true_params"],
            "FSPL",
            fit_parallax=fit_parallax,
            fit_defaults=None,
        )
    )

    return (
        fit_lc.build_fit_pyLIMA_model(
            event,
            "FSPL",
            fit_params,
            Origin=None,
            fit_parallax=fit_parallax,
        )
    )


def physical_vector(
    meta,
    fit_parallax,
):
    order = (
        fit_lc.fit_parameter_order(
            "FSPL",
            fit_parallax=fit_parallax,
        )
    )

    vector = np.array(
        [
            float(
                meta[
                    "true_params"
                ][name]
            )
            for name in order
        ],
        dtype=float,
    )

    return (
        order,
        vector,
    )


# ============================================================
# Flux helpers
# ============================================================

def inject_generator_fluxes(
    pyparams,
    model,
    truth_h5,
):
    """
    Inject true generator fluxes explicitly.

    This prevents pyLIMA from profiling them.
    """

    used = {}

    for telescope in (
        model.event.telescopes
    ):
        tel_name = (
            telescope.name
        )

        h5_band = (
            telescope_h5_band(
                tel_name
            )
        )

        fs_h5 = (
            f"fsource_{h5_band}"
        )

        ft_h5 = (
            f"ftotal_{h5_band}"
        )

        fb_h5 = (
            f"fblend_{h5_band}"
        )

        if fs_h5 not in truth_h5:
            raise KeyError(
                f"Missing {fs_h5}"
            )

        fsource = flux_to_pylima(
            truth_h5[fs_h5],
            h5_band,
        )

        if ft_h5 in truth_h5:

            ftotal = flux_to_pylima(
                truth_h5[ft_h5],
                h5_band,
            )

        elif fb_h5 in truth_h5:

            fblend = flux_to_pylima(
                truth_h5[fb_h5],
                h5_band,
            )

            ftotal = (
                fsource
                + fblend
            )

        else:

            raise KeyError(
                f"No total/blend flux "
                f"for {h5_band}"
            )

        fblend = (
            ftotal
            - fsource
        )

        # Keys must follow pyLIMA telescope names.
        pyparams[
            f"fsource_{tel_name}"
        ] = fsource

        pyparams[
            f"ftotal_{tel_name}"
        ] = ftotal

        pyparams[
            f"fblend_{tel_name}"
        ] = fblend

        pyparams[
            f"gblend_{tel_name}"
        ] = (
            fblend / fsource
            if fsource != 0.0
            else np.nan
        )

        used[tel_name] = {
            "h5_band":
                h5_band,
            "fsource":
                float(fsource),
            "ftotal":
                float(ftotal),
            "fblend":
                float(fblend),
        }

    return used


def extract_profiled_fluxes(
    pyparams,
    model,
):
    result = {}

    for telescope in (
        model.event.telescopes
    ):

        name = telescope.name

        result[name] = {
            "fsource":
                finite_float(
                    pyparams.get(
                        f"fsource_{name}"
                    )
                ),

            "ftotal":
                finite_float(
                    pyparams.get(
                        f"ftotal_{name}"
                    )
                ),

            "fblend":
                finite_float(
                    pyparams.get(
                        f"fblend_{name}"
                    )
                ),
        }

    return result


# ============================================================
# Direct pyLIMA chi2 evaluations
# ============================================================

def fixed_generator_chi2(
    model,
    x,
    truth_h5,
):
    """
    Physical parameters fixed + generator fluxes fixed.
    """

    pyparams = (
        model.compute_pyLIMA_parameters(
            x
        )
    )

    fluxes = (
        inject_generator_fluxes(
            pyparams,
            model,
            truth_h5,
        )
    )

    chi2 = (
        objective_functions
        .all_telescope_photometric_chi2(
            model,
            pyparams,
        )
    )

    return (
        float(chi2),
        fluxes,
    )


def profiled_chi2(
    model,
    x,
):
    """
    Physical parameters fixed.

    Fluxes are deliberately NOT supplied.

    The production runtime patch on
    MLmodel.derive_telescope_flux profiles them with the
    exact bounded solution used in production.
    """

    pyparams = (
        model.compute_pyLIMA_parameters(
            x
        )
    )

    chi2 = (
        objective_functions
        .all_telescope_photometric_chi2(
            model,
            pyparams,
        )
    )

    fluxes = (
        extract_profiled_fluxes(
            pyparams,
            model,
        )
    )

    return (
        float(chi2),
        fluxes,
    )


# ============================================================
# Build deterministic T>0 stratified sample
# ============================================================

def build_sample():

    print()
    print("=" * 90)
    print("BUILDING STRATIFIED T>0 SAMPLE")
    print("=" * 90)

    reservoirs = {}

    n_json = 0
    n_positive = 0
    n_eligible = 0

    for json_path in RUN.glob(
        "rows_*/events/Event_*.json"
    ):

        n_json += 1

        if n_json % 25000 == 0:
            print(
                f"scanned = {n_json:,}",
                flush=True,
            )

        try:
            data = json.loads(
                json_path.read_text()
            )
        except Exception:
            continue

        if (
            data.get(
                "generating_model"
            )
            != "H1"
        ):
            continue

        T = finite_float(
            data.get(
                "delta_chi2_lrt"
            )
        )

        # Main turnover study:
        # strict-positive LRT population.
        if (
            T is None
            or T <= 0.0
        ):
            continue

        n_positive += 1

        truth = data.get(
            "truth",
            {},
        )

        try:
            tE = truth_get(
                truth,
                "tE",
            )

            piEN = truth_get(
                truth,
                "piEN",
            )

            piEE = truth_get(
                truth,
                "piEE",
            )

        except Exception:
            continue

        piE = float(
            np.hypot(
                piEN,
                piEE,
            )
        )

        Xpi = float(
            piE
            * (
                tE
                / 365.25
            )**2
        )

        te_i = find_bin(
            tE,
            TE_EDGES,
        )

        xp_i = find_bin(
            Xpi,
            XPI_EDGES,
        )

        if (
            te_i is None
            or xp_i is None
        ):
            continue

        catalog_row = int(
            data[
                "catalog_row"
            ]
        )

        try:
            h5_path = (
                resolve_h5_path(
                    json_path,
                    data,
                    catalog_row,
                )
            )
        except Exception:
            continue

        n_eligible += 1

        record = {
            "catalog_row":
                catalog_row,

            "json_path":
                str(
                    json_path
                ),

            "h5_path":
                str(
                    h5_path
                ),

            "tE_true":
                tE,

            "piEN_true":
                piEN,

            "piEE_true":
                piEE,

            "piE_true":
                piE,

            "X_pi":
                Xpi,

            "T_fit":
                T,

            "tE_bin_index":
                te_i,

            "Xpi_bin_index":
                xp_i,

            "tE_bin":
                TE_LABELS[
                    te_i
                ],

            "Xpi_bin":
                XPI_LABELS[
                    xp_i
                ],
        }

        priority = (
            deterministic_priority(
                catalog_row,
                args.seed,
            )
        )

        cell = (
            te_i,
            xp_i,
        )

        heap = reservoirs.setdefault(
            cell,
            [],
        )

        item = (
            -priority,
            catalog_row,
            record,
        )

        if (
            len(heap)
            < args.per_cell
        ):

            heapq.heappush(
                heap,
                item,
            )

        else:

            worst_priority = (
                -heap[0][0]
            )

            if (
                priority
                < worst_priority
            ):

                heapq.heapreplace(
                    heap,
                    item,
                )

    selected = []

    for heap in reservoirs.values():
        for _, _, record in heap:
            selected.append(
                record
            )

    sample = pd.DataFrame(
        selected
    )

    if sample.empty:
        raise RuntimeError(
            "No sample selected."
        )

    sample = sample.sort_values(
        [
            "tE_bin_index",
            "Xpi_bin_index",
            "catalog_row",
        ]
    ).reset_index(
        drop=True
    )

    sample.to_csv(
        SAMPLE_PATH,
        index=False,
    )

    print()
    print(
        f"JSON scanned     = {n_json:,}"
    )

    print(
        f"T>0 H1 events    = {n_positive:,}"
    )

    print(
        f"eligible         = {n_eligible:,}"
    )

    print(
        f"selected         = {len(sample):,}"
    )

    print(
        f"sample saved     = {SAMPLE_PATH}"
    )

    print()

    print(
        sample.groupby(
            [
                "tE_bin",
                "Xpi_bin",
            ],
            observed=True,
        )
        .size()
        .rename("N")
        .to_string()
    )

    return sample


# ============================================================
# Load/build sample
# ============================================================

if args.sample_csv:

    sample = pd.read_csv(
        Path(
            args.sample_csv
        ).resolve()
    )

elif SAMPLE_PATH.is_file():

    sample = pd.read_csv(
        SAMPLE_PATH
    )

    print(
        "Using frozen sample:",
        SAMPLE_PATH,
    )

else:

    sample = build_sample()


# ============================================================
# Resume support
# ============================================================

records = []
done = set()

if (
    RESULT_PATH.is_file()
    and not args.overwrite
):

    old = pd.read_csv(
        RESULT_PATH
    )

    records = old.to_dict(
        orient="records"
    )

    if (
        "catalog_row"
        in old
    ):

        done = set(
            pd.to_numeric(
                old[
                    "catalog_row"
                ],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )

    print(
        f"Resuming: {len(done)} "
        f"events already present."
    )


def save_checkpoint():

    df = pd.DataFrame(
        records
    )

    tmp = Path(
        str(RESULT_PATH)
        + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        RESULT_PATH,
    )


# ============================================================
# Evaluate sample
# ============================================================

sample = sample.sort_values(
    [
        "tE_bin_index",
        "Xpi_bin_index",
        "catalog_row",
    ]
).reset_index(
    drop=True
)


for index, sample_row in (
    sample.iterrows()
):

    catalog_row = int(
        sample_row[
            "catalog_row"
        ]
    )

    if (
        catalog_row in done
        and not args.overwrite
    ):
        continue

    print()
    print("=" * 90)
    print(
        f"[{index + 1}/{len(sample)}] "
        f"catalog_row={catalog_row}"
    )
    print("=" * 90)

    json_path = Path(
        sample_row[
            "json_path"
        ]
    )

    h5_path = Path(
        sample_row[
            "h5_path"
        ]
    )

    try:

        data = json.loads(
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
                f[
                    "pyLIMA_parameters"
                ].attrs
            )

        event = build_event(
            meta
        )

        # --------------------------------------------
        # Build H1 and H0 model objects
        # --------------------------------------------

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

        _, x_h1_true = (
            physical_vector(
                meta,
                fit_parallax=True,
            )
        )

        _, x_h0_true = (
            physical_vector(
                meta,
                fit_parallax=False,
            )
        )

        # ============================================
        # 1. H1 generator:
        #    true physics + true fluxes
        # ============================================

        (
            chi2_h1_generator,
            true_fluxes,
        ) = fixed_generator_chi2(
            h1_model,
            x_h1_true,
            truth_h5,
        )

        # ============================================
        # 2. H0 fixed:
        #    shared truth + piE removed + true fluxes
        # ============================================

        (
            chi2_h0_fixed,
            _,
        ) = fixed_generator_chi2(
            h0_model,
            x_h0_true,
            truth_h5,
        )

        # ============================================
        # 3. H1 truth with fluxes profiled
        # ============================================

        (
            chi2_h1_profile_true,
            h1_profile_fluxes,
        ) = profiled_chi2(
            h1_model,
            x_h1_true,
        )

        # ============================================
        # 4. H0 shared truth with fluxes profiled
        # ============================================

        (
            chi2_h0_profile_true,
            h0_profile_fluxes,
        ) = profiled_chi2(
            h0_model,
            x_h0_true,
        )

        # ============================================
        # 5-6. Stored production minima
        # ============================================

        chi2_h1_fit = float(
            data[
                "chi2_h1"
            ]
        )

        chi2_h0_fit = float(
            data[
                "chi2_h0"
            ]
        )

        T_fit = float(
            data[
                "delta_chi2_lrt"
            ]
        )

        # --------------------------------------------
        # Truth physical quantities
        # --------------------------------------------

        truth = meta[
            "true_params"
        ]

        t0_true = float(
            truth["t0"]
        )

        u0_true = float(
            truth["u0"]
        )

        tE_true = float(
            truth["tE"]
        )

        rho_true = float(
            truth["rho"]
        )

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
            * (
                tE_true
                / 365.25
            )**2
        )

        # --------------------------------------------
        # Derived hierarchy
        # --------------------------------------------

        D_fixed = (
            chi2_h0_fixed
            - chi2_h1_generator
        )

        D_profile = (
            chi2_h0_profile_true
            - chi2_h1_profile_true
        )

        h1_flux_gain = (
            chi2_h1_generator
            - chi2_h1_profile_true
        )

        h0_flux_gain = (
            chi2_h0_fixed
            - chi2_h0_profile_true
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

        predicted_T = (
            D_profile
            - A_h0_physical
            + I_h1_physical
        )

        closure = (
            T_fit
            - predicted_T
        )

        # --------------------------------------------
        # H0 fitted morphology diagnostics
        # --------------------------------------------

        final_h0 = data[
            "final_h0"
        ]

        final_h1 = data[
            "final_h1"
        ]

        h0_tE_fit = float(
            final_h0["tE"]
        )

        h1_tE_fit = float(
            final_h1["tE"]
        )

        h0_tE_frac_bias = (
            (
                h0_tE_fit
                - tE_true
            )
            / tE_true
        )

        h1_tE_frac_bias = (
            (
                h1_tE_fit
                - tE_true
            )
            / tE_true
        )

        n_data = int(
            data[
                "n_photometry_points"
            ]
        )

        # --------------------------------------------
        # Save
        # --------------------------------------------

        result = {
            "status":
                "success",

            "catalog_row":
                catalog_row,

            "json_path":
                str(
                    json_path
                ),

            "h5_path":
                str(
                    h5_path
                ),

            "tE_bin":
                sample_row[
                    "tE_bin"
                ],

            "Xpi_bin":
                sample_row[
                    "Xpi_bin"
                ],

            "n_photometry_points":
                n_data,

            # Truth
            "t0_true":
                t0_true,

            "u0_true":
                u0_true,

            "tE_true":
                tE_true,

            "rho_true":
                rho_true,

            "piEN_true":
                piEN_true,

            "piEE_true":
                piEE_true,

            "piE_true":
                piE_true,

            "X_pi":
                X_pi,

            # ========================================
            # THE SIX CENTRAL CHI2 QUANTITIES
            # ========================================

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

            # Reduced chi2-like diagnostic
            "chi2_h1_generator_per_N":
                chi2_h1_generator
                / n_data,

            # ========================================
            # DERIVED SIGNAL/ABSORPTION QUANTITIES
            # ========================================

            "D_fixed":
                D_fixed,

            "D_profile":
                D_profile,

            "h1_flux_gain":
                h1_flux_gain,

            "h0_flux_gain":
                h0_flux_gain,

            "flux_absorption":
                flux_absorption,

            "A_h0_physical":
                A_h0_physical,

            "I_h1_physical":
                I_h1_physical,

            "T_fit":
                T_fit,

            "T_from_decomposition":
                predicted_T,

            "decomposition_closure":
                closure,

            # ========================================
            # PHYSICAL FIT DISPLACEMENT
            # ========================================

            "h0_tE_fit":
                h0_tE_fit,

            "h1_tE_fit":
                h1_tE_fit,

            "h0_tE_frac_bias":
                h0_tE_frac_bias,

            "h1_tE_frac_bias":
                h1_tE_frac_bias,

            # ========================================
            # FLUXES
            # ========================================

            "generator_fluxes_json":
                json.dumps(
                    true_fluxes,
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
        }

        records.append(
            result
        )

        save_checkpoint()

        print(
            f"N={n_data}  "
            f"tE={tE_true:.3f} d  "
            f"Xpi={X_pi:.5g}"
        )

        print(
            f"H1 generator       = "
            f"{chi2_h1_generator:.6f}"
        )

        print(
            f"H0 fixed           = "
            f"{chi2_h0_fixed:.6f}"
        )

        print(
            f"H1 profile truth   = "
            f"{chi2_h1_profile_true:.6f}"
        )

        print(
            f"H0 profile truth   = "
            f"{chi2_h0_profile_true:.6f}"
        )

        print(
            f"H1 fit             = "
            f"{chi2_h1_fit:.6f}"
        )

        print(
            f"H0 fit             = "
            f"{chi2_h0_fit:.6f}"
        )

        print(
            f"D_fixed            = "
            f"{D_fixed:.6f}"
        )

        print(
            f"D_profile          = "
            f"{D_profile:.6f}"
        )

        print(
            f"flux absorption    = "
            f"{flux_absorption:.6f}"
        )

        print(
            f"A_H0 physical      = "
            f"{A_h0_physical:.6f}"
        )

        print(
            f"I_H1 physical      = "
            f"{I_h1_physical:.6f}"
        )

        print(
            f"T_fit               = "
            f"{T_fit:.6f}"
        )

        print(
            f"closure             = "
            f"{closure:+.3e}"
        )

    except Exception as exc:

        print(
            f"FAILED row={catalog_row}: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        records.append(
            {
                "status":
                    "failure",

                "catalog_row":
                    catalog_row,

                "json_path":
                    str(
                        json_path
                    ),

                "h5_path":
                    str(
                        h5_path
                    ),

                "exception_type":
                    type(
                        exc
                    ).__name__,

                "exception_message":
                    str(
                        exc
                    ),

                "traceback":
                    traceback.format_exc(),
            }
        )

        save_checkpoint()


# ============================================================
# Final summary
# ============================================================

df = pd.DataFrame(
    records
)

df.to_csv(
    RESULT_PATH,
    index=False,
)

good = df[
    df[
        "status"
    ]
    == "success"
].copy()

if good.empty:
    raise RuntimeError(
        "No successful rows."
    )


summary = (
    good
    .groupby(
        [
            "tE_bin",
            "Xpi_bin",
        ],
        observed=True,
    )
    .agg(
        N=(
            "catalog_row",
            "size",
        ),

        median_tE=(
            "tE_true",
            "median",
        ),

        median_Xpi=(
            "X_pi",
            "median",
        ),

        median_T_fit=(
            "T_fit",
            "median",
        ),

        median_D_fixed=(
            "D_fixed",
            "median",
        ),

        median_D_profile=(
            "D_profile",
            "median",
        ),

        median_flux_absorption=(
            "flux_absorption",
            "median",
        ),

        median_A_h0_physical=(
            "A_h0_physical",
            "median",
        ),

        median_I_h1_physical=(
            "I_h1_physical",
            "median",
        ),

        median_h0_tE_frac_bias=(
            "h0_tE_frac_bias",
            "median",
        ),

        median_h1_generator_per_N=(
            "chi2_h1_generator_per_N",
            "median",
        ),

        max_abs_closure=(
            "decomposition_closure",
            lambda x:
                float(
                    np.max(
                        np.abs(x)
                    )
                ),
        ),
    )
    .reset_index()
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


text = f"""
TURNOVER CHI2 LAYERS
================================================================================

Run:
{RUN}

Successful events:
{len(good)}

Six stored chi2 quantities:
  chi2_h1_generator
  chi2_h0_fixed
  chi2_h1_profile_true
  chi2_h0_profile_true
  chi2_h1_fit
  chi2_h0_fit

Definitions:

  D_fixed =
      chi2_h0_fixed
      - chi2_h1_generator

  D_profile =
      chi2_h0_profile_true
      - chi2_h1_profile_true

  flux_absorption =
      D_fixed
      - D_profile

  A_h0_physical =
      chi2_h0_profile_true
      - chi2_h0_fit

  I_h1_physical =
      chi2_h1_profile_true
      - chi2_h1_fit

  T_fit =
      chi2_h0_fit
      - chi2_h1_fit

Closure:
  T_fit =
      D_profile
      - A_h0_physical
      + I_h1_physical

Global medians:

  H1 generator / N =
      {good["chi2_h1_generator_per_N"].median():.8g}

  D_fixed =
      {good["D_fixed"].median():.8g}

  D_profile =
      {good["D_profile"].median():.8g}

  flux_absorption =
      {good["flux_absorption"].median():.8g}

  A_h0_physical =
      {good["A_h0_physical"].median():.8g}

  I_h1_physical =
      {good["I_h1_physical"].median():.8g}

  T_fit =
      {good["T_fit"].median():.8g}

Maximum absolute decomposition closure:
  {np.max(np.abs(good["decomposition_closure"])):.12e}

Outputs:
  sample  = {SAMPLE_PATH}
  events  = {RESULT_PATH}
  summary = {SUMMARY_PATH}
================================================================================
""".strip()

TEXT_PATH.write_text(
    text + "\n"
)

print()
print(text)

print()
print("=" * 90)
print("BINNED SUMMARY")
print("=" * 90)

print(
    summary.to_string(
        index=False
    )
)
