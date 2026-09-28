#!/usr/bin/env python3

# ============================================================
# NEW BLOCK: imports
# ============================================================

import argparse
import copy
import json
import os
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
    default=8,
)

parser.add_argument(
    "--max-iteration",
    type=int,
    default=250,
)

parser.add_argument(
    "--strategy",
    default="rand1bin",
)

parser.add_argument(
    "--out",
    required=True,
)

args = parser.parse_args()


# ============================================================
# NEW BLOCK: repository paths
# ============================================================

ROOT = Path.cwd().resolve()

PP = ROOT / "validation" / "production_profiling"
BA = ROOT / "validation" / "bounds_audit"
BC = ROOT / "validation" / "bounds_convergence"

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


sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row",
    "71181",
    "--manifest",
    str(
        BA / "data" / "refit_manifest.csv"
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


PARAMETER_ORDER = [
    "t0",
    "u0",
    "tE",
    "rho",
    "piEN",
    "piEE",
]

PIE_SCALE = 0.3


# ============================================================
# NEW BLOCK: helpers
# ============================================================

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


def build_h1_objective(
    meta,
):
    """
    Build exactly the same profiled-H1 chi2 objective used
    in the previous parity audit.
    """

    lsst_lcs = {
        band: meta["curves"][band]
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
        str(core.EPHEMERIDES),
        meta["curves"]["W149"],
        lsst_lcs,
        ra=meta["event_ra"],
        dec=meta["event_dec"],
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
        telescopes_fluxes_method="polyfit",
        loss_function="chi2",
        DE_population_size=1,
        max_iteration=1,
        display_progress=False,
        strategy="rand1bin",
    )

    bounds = copy.deepcopy(
        core.H1_BOUNDS
    )

    bounds = core.apply_bounds_profile(
        bounds,
        h1=True,
        profile=core.BOUNDS_PROFILE,
    )

    fit_lc.apply_fit_bounds(
        de,
        fit_params,
        "FSPL",
        meta["rango"],
        fit_parallax=True,
        fit_bounds=bounds,
    )

    return de


def z_to_piE(z):
    """
    Full [-40, 40] physical domain mapped through asinh.
    """

    z = np.asarray(
        z,
        dtype=float,
    )

    return (
        PIE_SCALE
        * np.sinh(z)
    )


# ============================================================
# NEW BLOCK: load frozen event and references
# ============================================================

manifest = pd.read_csv(
    args.rows_csv
)

m = manifest[
    manifest["catalog_row"].astype(int)
    == int(args.row)
]

if len(m) != 1:
    raise RuntimeError(
        f"Expected one manifest row for "
        f"{args.row}, got {len(m)}"
    )

h5_path = str(
    m.iloc[0]["h5_path"]
)

meta = load_new_case(
    h5_path,
    core,
)

if meta["generating_model"] != "H0":
    raise RuntimeError(
        "Expected H0-generated event."
    )


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


h0 = policy_event["final_h0"]

chi2_h0 = float(
    policy_event["chi2_h0"]
)

chi2_policy = float(
    policy_event["chi2_h1"]
)

chi2_target = float(
    stability_event["best_local_chi2"]
)


# ============================================================
# NEW BLOCK: common H0 morphology
# ============================================================

t0_h0 = float(
    h0["t0"]
)

u0_h0 = float(
    h0["u0"]
)

tE_h0 = float(
    h0["tE"]
)

rho_h0 = float(
    h0["rho"]
)


objective = build_h1_objective(
    meta
)


# ============================================================
# NEW BLOCK: parallax-only global discovery
# ============================================================

zmax = float(
    np.arcsinh(
        40.0 / PIE_SCALE
    )
)

z_bounds = [
    (-zmax, zmax),
    (-zmax, zmax),
]


branches = [
    (
        "same",
        u0_h0,
    ),
    (
        "mirror",
        -u0_h0,
    ),
]

branch_results = []


for branch_index, (
    branch_name,
    u0_seed,
) in enumerate(branches):

    print()
    print("=" * 90)
    print(
        f"PARALLAX GLOBAL SEARCH: "
        f"{branch_name}"
    )
    print("=" * 90)

    print(
        "anchor =",
        {
            "t0": t0_h0,
            "u0": u0_seed,
            "tE": tE_h0,
            "rho": rho_h0,
        },
    )

    def objective_2d(z):

        piEN, piEE = z_to_piE(
            z
        )

        x = np.array(
            [
                t0_h0,
                u0_seed,
                tE_h0,
                rho_h0,
                piEN,
                piEE,
            ],
            dtype=float,
        )

        return float(
            objective.objective_function(
                x
            )
        )

    wall0 = time.time()

    result = differential_evolution(
        objective_2d,
        bounds=z_bounds,
        strategy=args.strategy,
        maxiter=args.max_iteration,
        popsize=args.population_size,
        seed=(
            int(args.seed)
            + branch_index
        ),
        polish=False,
        init="sobol",
        workers=1,
        updating="immediate",
        disp=False,
    )

    de_wall = (
        time.time()
        - wall0
    )

    piEN, piEE = z_to_piE(
        result.x
    )

    initial = {
        "t0": t0_h0,
        "u0": u0_seed,
        "tE": tE_h0,
        "rho": rho_h0,
        "piEN": float(piEN),
        "piEE": float(piEE),
    }

    print(
        "DE2 chi2 =",
        float(result.fun),
    )

    print(
        "DE2 piEN =",
        float(piEN),
    )

    print(
        "DE2 piEE =",
        float(piEE),
    )

    print(
        "DE2 nit  =",
        result.nit,
    )

    print(
        "DE2 nfev =",
        result.nfev,
    )

    print(
        "DE2 wall =",
        de_wall,
    )


    # ========================================================
    # NEW BLOCK: release all six parameters with current TRF
    # ========================================================

    trf_wall0 = time.time()

    trf = run_one_fit_full(
        core,
        fit_lc,
        meta,
        "H1",
        initial,
        (
            f"H1_piE_DE_"
            f"{branch_name}"
        ),
    )

    trf_wall = (
        time.time()
        - trf_wall0
    )

    if trf.get("status") != "success":
        raise RuntimeError(
            f"TRF failed for branch "
            f"{branch_name}: {trf}"
        )

    chi2_trf = float(
        trf["chi2"]
    )

    branch_result = {
        "branch":
            branch_name,

        "u0_anchor":
            float(u0_seed),

        "de_chi2":
            float(result.fun),

        "de_piEN":
            float(piEN),

        "de_piEE":
            float(piEE),

        "de_nit":
            int(result.nit),

        "de_nfev":
            int(result.nfev),

        "de_wall_s":
            float(de_wall),

        "trf_chi2":
            chi2_trf,

        "gap_to_target":
            chi2_trf
            - chi2_target,

        "gain_vs_policy":
            chi2_policy
            - chi2_trf,

        "trf_wall_s":
            float(trf_wall),

        "final_parameters": {
            name:
                (
                    float(trf[name])
                    if trf.get(name)
                    is not None
                    else None
                )
            for name
            in PARAMETER_ORDER
        },
    }

    branch_results.append(
        branch_result
    )

    print()
    print(
        "TRF chi2     =",
        chi2_trf,
    )

    print(
        "gap target   =",
        chi2_trf
        - chi2_target,
    )

    print(
        "gain policy  =",
        chi2_policy
        - chi2_trf,
    )

    print(
        "final params =",
        branch_result[
            "final_parameters"
        ],
    )


# ============================================================
# NEW BLOCK: choose best branch
# ============================================================

best = min(
    branch_results,
    key=lambda x:
        x["trf_chi2"],
)

total_wall = sum(
    b["de_wall_s"]
    + b["trf_wall_s"]
    for b in branch_results
)


print()
print("=" * 90)
print("FINAL")
print("=" * 90)

print(
    "row             =",
    args.row,
)

print(
    "chi2 H0        =",
    chi2_h0,
)

print(
    "chi2 policy    =",
    chi2_policy,
)

print(
    "chi2 target    =",
    chi2_target,
)

print(
    "best branch    =",
    best["branch"],
)

print(
    "best chi2      =",
    best["trf_chi2"],
)

print(
    "gap target     =",
    best["gap_to_target"],
)

print(
    "gain policy    =",
    best["gain_vs_policy"],
)

print(
    "D policy       =",
    chi2_h0
    - chi2_policy,
)

print(
    "D piE-DE+TRF   =",
    chi2_h0
    - best["trf_chi2"],
)

print(
    "total wall_s   =",
    total_wall,
)


# ============================================================
# NEW BLOCK: save
# ============================================================

payload = {
    "benchmark":
        "H0_generated_piE_only_DE_plus_TRF_v1",

    "catalog_row":
        int(args.row),

    "seed":
        int(args.seed),

    "population_size":
        int(args.population_size),

    "max_iteration":
        int(args.max_iteration),

    "chi2_h0":
        chi2_h0,

    "chi2_h1_policy":
        chi2_policy,

    "chi2_target":
        chi2_target,

    "branches":
        branch_results,

    "best_branch":
        best["branch"],

    "best_chi2":
        best["trf_chi2"],

    "best_gap_to_target":
        best["gap_to_target"],

    "D_policy":
        chi2_h0
        - chi2_policy,

    "D_piE_DE_TRF":
        chi2_h0
        - best["trf_chi2"],

    "total_wall_s":
        float(total_wall),
}

out = Path(
    args.out
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
