#!/usr/bin/env python3
"""
Validation-only benchmark:

    global Differential Evolution in H1
        ->
    current production TRF polish

The event is loaded from an already-materialized H5, so this benchmark
uses exactly the same photometric realization as the existing
H0/H1 basin-validation experiments.

No production policy is modified.
"""

# ============================================================
# NEW BLOCK: imports
# ============================================================

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution


# ============================================================
# NEW BLOCK: CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--rows-csv",
    required=True,
)

parser.add_argument(
    "--policy-json",
    required=True,
)

parser.add_argument(
    "--stability-json",
    required=True,
)

parser.add_argument(
    "--row",
    type=int,
    required=True,
)

parser.add_argument(
    "--seed",
    type=int,
    required=True,
)

parser.add_argument(
    "--population-size",
    type=int,
    default=10,
)

parser.add_argument(
    "--max-iteration",
    type=int,
    default=1500,
)

parser.add_argument(
    "--strategy",
    default="rand1bin",
)

parser.add_argument(
    "--out",
    required=True,
)

parser.add_argument(
    "--display-progress",
    action="store_true",
)

parser.add_argument(
    "--audit-only",
    action="store_true",
    help="Evaluate the DE objective at known reference points and exit.",
)

parser.add_argument(
    "--search-coords",
    choices=[
        "physical",
        "transformed",
    ],
    default="physical",
)

args = parser.parse_args()


# ============================================================
# NEW BLOCK: repository paths
# ============================================================

ROOT = Path.cwd().resolve()

PP = (
    ROOT
    / "validation"
    / "production_profiling"
)

BA = (
    ROOT
    / "validation"
    / "bounds_audit"
)

BC = (
    ROOT
    / "validation"
    / "bounds_convergence"
)

for p in (
    PP,
    BA,
    BC,
):
    if not p.exists():
        raise RuntimeError(
            f"Missing expected path: {p}"
        )

sys.path.insert(
    0,
    str(BA),
)

sys.path.insert(
    0,
    str(BC),
)

sys.path.insert(
    0,
    str(PP),
)


# ============================================================
# NEW BLOCK: frozen fitting environment
# ============================================================

os.environ[
    "HIDDEN_PARALLAX_BOUNDED_PROFILE"
] = "1"

os.environ[
    "HIDDEN_PARALLAX_TRF_COORDS"
] = "physical"

os.environ[
    "HIDDEN_PARALLAX_TRF_X_SCALE"
] = "jac"

os.environ[
    "HIDDEN_PARALLAX_T0_MARGIN_FACTOR"
] = "0"


# core parses argv at import time.
sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row",
    "71181",
    "--manifest",
    str(
        BA
        / "data"
        / "refit_manifest.csv"
    ),
    "--bounds-profile",
    "production_candidate",
    "--fit-scope",
    "h1",
    "--dry-run",
]


# ============================================================
# NEW BLOCK: fitting imports
# ============================================================

import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402

from load_new_event import load_new_case  # noqa: E402
from run_one_fit_full import run_one_fit_full  # noqa: E402
from pyLIMA.fits import DE_fit  # noqa: E402


if core.BOUNDS_PROFILE != "production_candidate":
    raise RuntimeError(
        "Expected production_candidate bounds."
    )

if (
    hasattr(
        core,
        "_bounded_flux_profile_enabled",
    )
    and not core._bounded_flux_profile_enabled()
):
    raise RuntimeError(
        "Bounded flux profiling is not enabled."
    )


# ============================================================
# NEW BLOCK: helpers
# ============================================================

PARAMETER_ORDER = [
    "t0",
    "u0",
    "tE",
    "rho",
    "piEN",
    "piEE",
]


def find_event(
    payload,
    row,
):
    for event in payload["events"]:
        if int(
            event["catalog_row"]
        ) == int(row):
            return event

    raise KeyError(
        f"row={row} not found"
    )


def production_h1_bounds(meta):
    """
    Construct exactly the H1 nonlinear domain used by the
    current production-candidate fitter, including the
    data-driven t0 domain.
    """

    bounds = copy.deepcopy(
        core.H1_BOUNDS
    )

    bounds = core.apply_bounds_profile(
        bounds,
        h1=True,
        profile=core.BOUNDS_PROFILE,
    )

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

    all_times = np.asarray(
        all_times,
        dtype=float,
    )

    if len(all_times) == 0:
        raise RuntimeError(
            "No Rubin photometry."
        )

    t_min = float(
        np.min(all_times)
    )

    t_max = float(
        np.max(all_times)
    )

    bounds["t0"] = {
        "type": "center_width",
        "center":
            0.5
            * (
                t_min
                + t_max
            ),
        "half_width":
            0.5
            * (
                t_max
                - t_min
            ),
    }

    return bounds


def build_de_fit(
    meta,
    bounds,
):
    """
    Build H1 FSPL+parallax DE problem using the exact same
    event, geometry and nonlinear domain as the production
    H1 fit.

    Important:
    pyLIMA DEfit defaults to loss_function='likelihood'.
    We explicitly use loss_function='chi2' because this
    benchmark is for the LRT chi2 objective.
    """

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

    event = fit_lc.create_fit_event(
        meta["Source"],
        str(
            core.EPHEMERIDES
        ),
        meta["curves"]["W149"],
        lsst_lcs,
        ra=
            meta["event_ra"],
        dec=
            meta["event_dec"],
        roman_name="Roman",
    )

    fit_params = (
        fit_lc.initial_params_for_fit_model(
            meta["true_params"],
            "FSPL",
            fit_parallax=True,
            fit_defaults=None,
        )
    )

    model = (
        fit_lc.build_fit_pyLIMA_model(
            event,
            "FSPL",
            fit_params,
            Origin=None,
            fit_parallax=True,
        )
    )

    de = DE_fit.DEfit(
        model,
        telescopes_fluxes_method=
            "polyfit",

        # CRITICAL:
        # LRT objective, not DEfit's
        # default likelihood.
        loss_function=
            "chi2",

        DE_population_size=
            args.population_size,

        max_iteration=
            args.max_iteration,

        display_progress=
            args.display_progress,

        strategy=
            args.strategy,
    )

    fit_lc.apply_fit_bounds(
        de,
        fit_params,
        "FSPL",
        meta["rango"],
        fit_parallax=True,
        fit_bounds=
            copy.deepcopy(
                bounds
            ),
    )

    order = list(
        fit_lc.fit_parameter_order(
            "FSPL",
            fit_parallax=True,
        )
    )

    if order != PARAMETER_ORDER:
        raise RuntimeError(
            "Unexpected parameter order: "
            f"{order}"
        )

    if list(
        de.fit_parameters.keys()
    ) != PARAMETER_ORDER:
        raise RuntimeError(
            "Unexpected DE fit parameters: "
            f"{list(de.fit_parameters.keys())}"
        )

    return de


# ============================================================
# NEW BLOCK: transformed global-search coordinates
# ============================================================

PIE_SCALE = 0.3


def physical_interval(spec):
    """
    Return explicit low/high bounds from the bound specification.
    """

    if isinstance(spec, dict):
        if spec.get("type") != "center_width":
            raise RuntimeError(
                f"Unsupported bound specification: {spec}"
            )

        center = float(spec["center"])
        half_width = float(spec["half_width"])

        return (
            center - half_width,
            center + half_width,
        )

    return (
        float(spec[0]),
        float(spec[1]),
    )


def make_transformed_search(
    bounds,
    h0_record,
):
    """
    Construct an invertible transformation that preserves the
    complete physical production-candidate H1 domain.

    Coordinates:
        asinh((t0-t0_H0)/tE_H0)
        asinh(u0/u_scale)
        log10(tE)
        log10(rho)
        asinh(piEN/PIE_SCALE)
        asinh(piEE/PIE_SCALE)
    """

    t0_ref = float(
        h0_record["t0"]
    )

    tE_ref = max(
        float(h0_record["tE"]),
        0.1,
    )

    u0_scale = max(
        abs(
            float(
                h0_record["u0"]
            )
        ),
        0.1,
    )

    t0_lo, t0_hi = physical_interval(
        bounds["t0"]
    )

    u0_lo, u0_hi = physical_interval(
        bounds["u0"]
    )

    tE_lo, tE_hi = physical_interval(
        bounds["tE"]
    )

    rho_lo, rho_hi = physical_interval(
        bounds["rho"]
    )

    piEN_lo, piEN_hi = physical_interval(
        bounds["piEN"]
    )

    piEE_lo, piEE_hi = physical_interval(
        bounds["piEE"]
    )

    z_bounds = [
        (
            np.arcsinh(
                (t0_lo - t0_ref)
                / tE_ref
            ),
            np.arcsinh(
                (t0_hi - t0_ref)
                / tE_ref
            ),
        ),
        (
            np.arcsinh(
                u0_lo / u0_scale
            ),
            np.arcsinh(
                u0_hi / u0_scale
            ),
        ),
        (
            np.log10(tE_lo),
            np.log10(tE_hi),
        ),
        (
            np.log10(rho_lo),
            np.log10(rho_hi),
        ),
        (
            np.arcsinh(
                piEN_lo / PIE_SCALE
            ),
            np.arcsinh(
                piEN_hi / PIE_SCALE
            ),
        ),
        (
            np.arcsinh(
                piEE_lo / PIE_SCALE
            ),
            np.arcsinh(
                piEE_hi / PIE_SCALE
            ),
        ),
    ]

    def to_physical(z):

        z = np.asarray(
            z,
            dtype=float,
        )

        x = np.array(
            [
                t0_ref
                + tE_ref
                * np.sinh(z[0]),

                u0_scale
                * np.sinh(z[1]),

                10.0 ** z[2],

                10.0 ** z[3],

                PIE_SCALE
                * np.sinh(z[4]),

                PIE_SCALE
                * np.sinh(z[5]),
            ],
            dtype=float,
        )

        # Numerical clipping only; no scientific bound changes.
        physical_bounds = [
            (t0_lo, t0_hi),
            (u0_lo, u0_hi),
            (tE_lo, tE_hi),
            (rho_lo, rho_hi),
            (piEN_lo, piEN_hi),
            (piEE_lo, piEE_hi),
        ]

        for i, (lo, hi) in enumerate(
            physical_bounds
        ):
            x[i] = np.clip(
                x[i],
                lo,
                hi,
            )

        return x

    return (
        z_bounds,
        to_physical,
        {
            "t0_ref": t0_ref,
            "tE_ref": tE_ref,
            "u0_scale": u0_scale,
            "piE_scale": PIE_SCALE,
        },
    )


# ============================================================
# NEW BLOCK: load exact frozen event
# ============================================================

manifest = pd.read_csv(
    args.rows_csv
)

manifest_row = manifest[
    manifest[
        "catalog_row"
    ].astype(int)
    == int(args.row)
]

if len(manifest_row) != 1:
    raise RuntimeError(
        f"Expected exactly one manifest entry "
        f"for row={args.row}; got "
        f"{len(manifest_row)}"
    )

h5_path = str(
    manifest_row.iloc[0][
        "h5_path"
    ]
)

meta = load_new_case(
    h5_path,
    core,
)

if int(
    meta["row"]
) != int(args.row):
    raise RuntimeError(
        "Loaded wrong catalog row."
    )

if (
    meta["generating_model"]
    != "H0"
):
    raise RuntimeError(
        "This benchmark expects "
        "an H0-generated event."
    )


# ============================================================
# NEW BLOCK: reference values
# ============================================================

policy_payload = json.loads(
    Path(
        args.policy_json
    ).read_text()
)

policy_event = find_event(
    policy_payload,
    args.row,
)

stability_payload = json.loads(
    Path(
        args.stability_json
    ).read_text()
)

stability_event = find_event(
    stability_payload,
    args.row,
)

chi2_h0 = float(
    policy_event["chi2_h0"]
)

chi2_h1_policy = float(
    policy_event["chi2_h1"]
)

chi2_target = float(
    stability_event[
        "best_local_chi2"
    ]
)


# ============================================================
# NEW BLOCK: H1 domain audit
# ============================================================

bounds = production_h1_bounds(
    meta
)

print()
print("=" * 90)
print("H0-GENERATED H1 GLOBAL-DE BENCHMARK")
print("=" * 90)

print(
    "row              =",
    args.row,
)

print(
    "seed             =",
    args.seed,
)

print(
    "population_size  =",
    args.population_size,
)

print(
    "max_iteration    =",
    args.max_iteration,
)

print(
    "strategy         =",
    args.strategy,
)

print(
    "search coords    =",
    args.search_coords,
)

print(
    "chi2 H0          =",
    chi2_h0,
)

print(
    "chi2 H1 policy   =",
    chi2_h1_policy,
)

print(
    "chi2 target      =",
    chi2_target,
)

print()
print("H1 BOUNDS")
print("---------")

for key in PARAMETER_ORDER:
    print(
        key,
        "=",
        bounds[key],
    )


# ============================================================
# NEW BLOCK: global Differential Evolution
# ============================================================

de = build_de_fit(
    meta,
    bounds,
)


# ============================================================
# NEW BLOCK: DE objective parity audit
# ============================================================

# Exact nested H1 point constructed from the settled H0 fit.
h0_record = policy_event["final_h0"]

nested_x = np.array(
    [
        float(h0_record["t0"]),
        float(h0_record["u0"]),
        float(h0_record["tE"]),
        float(h0_record["rho"]),
        0.0,
        0.0,
    ],
    dtype=float,
)


# Best locally validated H1 point.
finite_local = [
    f
    for f in stability_event["fits"]
    if (
        f.get("chi2") is not None
        and np.isfinite(float(f["chi2"]))
    )
]

best_local_record = min(
    finite_local,
    key=lambda f: float(f["chi2"]),
)

target_x = np.array(
    [
        float(best_local_record[name])
        for name in PARAMETER_ORDER
    ],
    dtype=float,
)


nested_de_objective = float(
    de.objective_function(
        nested_x
    )
)

target_de_objective = float(
    de.objective_function(
        target_x
    )
)


print()
print("=" * 90)
print("DE OBJECTIVE PARITY AUDIT")
print("=" * 90)

print(
    "stored chi2 H0         =",
    chi2_h0,
)

print(
    "DE objective nested H0 =",
    nested_de_objective,
)

print(
    "nested difference      =",
    nested_de_objective
    - chi2_h0,
)

print()

print(
    "stored local target    =",
    chi2_target,
)

print(
    "DE objective at target =",
    target_de_objective,
)

print(
    "target difference      =",
    target_de_objective
    - chi2_target,
)

print()

print(
    "nested_x =",
    nested_x.tolist(),
)

print(
    "target_x =",
    target_x.tolist(),
)


if args.audit_only:
    print()
    print(
        "AUDIT-ONLY: stopping before DE."
    )
    raise SystemExit(0)


# ============================================================
# NEW BLOCK: global search
# ============================================================

np.random.seed(
    int(args.seed)
)

random.seed(
    int(args.seed)
)

de_wall0 = time.time()


if args.search_coords == "physical":

    # Original pyLIMA DE benchmark.
    de.fit(
        computational_pool=None
    )

    if not isinstance(
        de.fit_results,
        dict,
    ):
        raise RuntimeError(
            "DE returned no fit_results."
        )

    fit_object = de.fit_results.get(
        "fit_object"
    )

    if fit_object is None:
        raise RuntimeError(
            "DE returned no scipy fit_object."
        )

    de_x = np.asarray(
        fit_object.x,
        dtype=float,
    )

    transformed_metadata = None


else:

    # Global exploration in nonlinear coordinates while
    # evaluating exactly the same physical profiled chi2.
    h0_record = policy_event[
        "final_h0"
    ]

    (
        z_bounds,
        to_physical,
        transformed_metadata,
    ) = make_transformed_search(
        bounds,
        h0_record,
    )

    def transformed_objective(z):

        physical_x = to_physical(
            z
        )

        return float(
            de.objective_function(
                physical_x
            )
        )

    fit_object = differential_evolution(
        transformed_objective,
        bounds=z_bounds,
        strategy=args.strategy,
        maxiter=args.max_iteration,
        popsize=args.population_size,
        seed=int(args.seed),
        polish=False,
        init="sobol",
        workers=1,
        updating="immediate",
        disp=args.display_progress,
    )

    de_x = to_physical(
        fit_object.x
    )


de_wall_s = (
    time.time()
    - de_wall0
)


if len(de_x) != 6:
    raise RuntimeError(
        "Expected 6D DE optimizer vector, "
        f"got length={len(de_x)}"
    )

if not np.all(
    np.isfinite(
        de_x
    )
):
    raise RuntimeError(
        "DE optimizer_x is non-finite."
    )

de_chi2 = float(
    fit_object.fun
)

de_guess = {
    name:
        float(de_x[i])
    for i, name
    in enumerate(
        PARAMETER_ORDER
    )
}

print()
print("=" * 90)
print("DE RESULT")
print("=" * 90)

print(
    "DE chi2          =",
    de_chi2,
)

print(
    "DE optimizer_x   =",
    de_x.tolist(),
)

print(
    "DE nit           =",
    getattr(
        fit_object,
        "nit",
        None,
    ),
)

print(
    "DE nfev          =",
    getattr(
        fit_object,
        "nfev",
        None,
    ),
)

print(
    "DE success       =",
    getattr(
        fit_object,
        "success",
        None,
    ),
)

print(
    "DE message       =",
    str(
        getattr(
            fit_object,
            "message",
            None,
        )
    ),
)

print(
    "DE wall_s        =",
    de_wall_s,
)


# ============================================================
# NEW BLOCK: production TRF polish from DE point
# ============================================================

trf_wall0 = time.time()

trf = run_one_fit_full(
    core,
    fit_lc,
    meta,
    "H1",
    de_guess,
    (
        f"H1_DE_seed"
        f"{args.seed}_polish"
    ),
)

trf_wall_s = (
    time.time()
    - trf_wall0
)

if (
    trf.get("status")
    != "success"
):
    raise RuntimeError(
        "TRF polish failed: "
        f"{trf}"
    )

chi2_de_trf = float(
    trf["chi2"]
)

gap_target = (
    chi2_de_trf
    - chi2_target
)

gap_policy = (
    chi2_h1_policy
    - chi2_de_trf
)

D_policy = (
    chi2_h0
    - chi2_h1_policy
)

D_de_trf = (
    chi2_h0
    - chi2_de_trf
)


# ============================================================
# NEW BLOCK: report
# ============================================================

print()
print("=" * 90)
print("DE -> TRF RESULT")
print("=" * 90)

print(
    "chi2 H1 policy   =",
    chi2_h1_policy,
)

print(
    "chi2 DE raw      =",
    de_chi2,
)

print(
    "chi2 DE+TRF      =",
    chi2_de_trf,
)

print(
    "chi2 target      =",
    chi2_target,
)

print(
    "gap to target    =",
    gap_target,
)

print(
    "gain vs policy   =",
    gap_policy,
)

print(
    "D policy         =",
    D_policy,
)

print(
    "D DE+TRF         =",
    D_de_trf,
)

print(
    "TRF wall_s       =",
    trf_wall_s,
)

print(
    "total wall_s     =",
    de_wall_s
    + trf_wall_s,
)

print()
print(
    "TRF final params =",
    {
        key:
            trf.get(key)
        for key
        in PARAMETER_ORDER
    },
)


# ============================================================
# NEW BLOCK: save compact benchmark
# ============================================================

payload = {
    "benchmark":
        "h0_generated_h1_DE_plus_TRF_v1",

    "catalog_row":
        int(args.row),

    "seed":
        int(args.seed),

    "population_size":
        int(args.population_size),

    "max_iteration":
        int(args.max_iteration),

    "strategy":
        str(args.strategy),

    "chi2_h0":
        chi2_h0,

    "chi2_h1_policy":
        chi2_h1_policy,

    "chi2_target":
        chi2_target,

    "de": {
        "chi2":
            de_chi2,

        "optimizer_x":
            de_x.tolist(),

        "parameter_order":
            list(
                PARAMETER_ORDER
            ),

        "nit":
            (
                int(
                    fit_object.nit
                )
                if getattr(
                    fit_object,
                    "nit",
                    None,
                )
                is not None
                else None
            ),

        "nfev":
            (
                int(
                    fit_object.nfev
                )
                if getattr(
                    fit_object,
                    "nfev",
                    None,
                )
                is not None
                else None
            ),

        "success":
            bool(
                getattr(
                    fit_object,
                    "success",
                    False,
                )
            ),

        "message":
            str(
                getattr(
                    fit_object,
                    "message",
                    ""
                )
            ),

        "wall_s":
            float(
                de_wall_s
            ),
    },

    "trf": {
        "chi2":
            chi2_de_trf,

        "gap_to_target":
            gap_target,

        "gain_vs_policy":
            gap_policy,

        "wall_s":
            float(
                trf_wall_s
            ),

        "parameters": {
            key:
                (
                    float(
                        trf[key]
                    )
                    if trf.get(key)
                    is not None
                    else None
                )
            for key
            in PARAMETER_ORDER
        },
    },

    "D_policy":
        D_policy,

    "D_DE_TRF":
        D_de_trf,

    "total_wall_s":
        float(
            de_wall_s
            + trf_wall_s
        ),
}

out = Path(
    args.out
)

out.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.write_text(
    json.dumps(
        payload,
        indent=2,
    )
    + "\n"
)

print()
print(
    "OUTPUT =",
    out,
)
