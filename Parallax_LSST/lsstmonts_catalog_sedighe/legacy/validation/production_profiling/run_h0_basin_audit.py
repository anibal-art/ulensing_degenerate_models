#!/usr/bin/env python3

# ============================================================
# NEW BLOCK: imports
# ============================================================

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import least_squares


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
    "--reference-json",
    required=True,
)

parser.add_argument(
    "--max-nfev",
    type=int,
    default=1000,
)

parser.add_argument(
    "--out-json",
    required=True,
)

parser.add_argument(
    "--out-csv",
    required=True,
)

args = parser.parse_args()


# ============================================================
# NEW BLOCK: repository paths
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]

PP = ROOT / "validation" / "production_profiling"
BA = ROOT / "validation" / "bounds_audit"
BC = ROOT / "validation" / "bounds_convergence"

sys.path.insert(0, str(PP))
sys.path.insert(0, str(BA))
sys.path.insert(0, str(BC))


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
    "h0",
    "--dry-run",
]


# ============================================================
# NEW BLOCK: project imports
# ============================================================

import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402

from load_new_event import load_new_case  # noqa: E402
from pyLIMA.fits import DE_fit  # noqa: E402


H0_ORDER = [
    "t0",
    "u0",
    "tE",
    "rho",
]


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


def resolve_event_h5(
    stored_path,
    row,
):

    stored = Path(
        str(stored_path)
    )

    if stored.exists():

        return stored.resolve()

    local = (
        PP
        / "runtime_local"
        / "h0_h1_basin_validation_h5"
        / f"Event_{int(row)}.h5"
    )

    if local.exists():

        return local.resolve()

    raise FileNotFoundError(
        f"row={row}\n"
        f"stored H5: {stored}\n"
        f"local H5:  {local}"
    )


def vector_from_record(
    record,
):

    return np.array(
        [
            float(
                record[name]
            )
            for name
            in H0_ORDER
        ],
        dtype=float,
    )


def mirrored(
    x,
):

    y = np.asarray(
        x,
        dtype=float,
    ).copy()

    y[1] *= -1.0

    return y


def all_observation_times(
    meta,
):

    chunks = []

    for band in [
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
    ]:

        curve = meta[
            "curves"
        ][band]

        if len(curve) == 0:
            continue

        chunks.append(
            np.asarray(
                curve[:, 0],
                dtype=float,
            )
        )

    if not chunks:

        raise RuntimeError(
            "No Rubin observations."
        )

    return np.concatenate(
        chunks
    )


# ============================================================
# NEW BLOCK: exact H0 objective
# ============================================================

def build_h0_objective(
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

    event = fit_lc.create_fit_event(
        meta["Source"],
        None,
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
            fit_parallax=False,
            fit_defaults=None,
        )
    )

    model = (
        fit_lc.build_fit_pyLIMA_model(
            event,
            "FSPL",
            fit_params,
            Origin=None,
            fit_parallax=False,
        )
    )

    objective = DE_fit.DEfit(
        model,
        telescopes_fluxes_method=
            "polyfit",
        loss_function=
            "chi2",
        DE_population_size=1,
        max_iteration=1,
        display_progress=False,
        strategy="rand1bin",
    )

    actual_order = list(
        objective.fit_parameters.keys()
    )

    if actual_order != H0_ORDER:

        raise RuntimeError(
            "Unexpected H0 parameter order.\n"
            f"Expected: {H0_ORDER}\n"
            f"Actual:   {actual_order}"
        )

    return objective


# ============================================================
# NEW BLOCK: residual vector
# ============================================================

def normalized_residuals(
    objective,
    x,
):

    _, pyparams = (
        objective.model_chi2(
            np.asarray(
                x,
                dtype=float,
            )
        )
    )

    residuals, errors = (
        objective.photometric_model_residuals(
            pyparams
        )
    )

    chunks = []

    for residual, error in zip(
        residuals,
        errors,
    ):

        residual = np.asarray(
            residual,
            dtype=float,
        )

        error = np.asarray(
            error,
            dtype=float,
        )

        if len(residual) == 0:
            continue

        chunks.append(
            residual / error
        )

    return np.concatenate(
        chunks
    )


# ============================================================
# NEW BLOCK: one H0 TRF fit
# ============================================================

def fit_one_start(
    objective,
    x0,
    lower,
    upper,
    max_nfev,
):

    x0 = np.asarray(
        x0,
        dtype=float,
    ).copy()

    eps = 1e-10

    x0 = np.maximum(
        x0,
        lower + eps,
    )

    x0 = np.minimum(
        x0,
        upper - eps,
    )

    t_start = time.perf_counter()

    try:

        result = least_squares(
            lambda x:
                normalized_residuals(
                    objective,
                    x,
                ),
            x0,
            bounds=(
                lower,
                upper,
            ),
            method="trf",
            jac="2-point",
            x_scale="jac",
            diff_step=1e-6,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=max_nfev,
        )

        x_final = np.asarray(
            result.x,
            dtype=float,
        )

        chi2 = float(
            objective.objective_function(
                x_final
            )
        )

        return {
            "chi2":
                chi2,

            "t0":
                float(
                    x_final[0]
                ),

            "u0":
                float(
                    x_final[1]
                ),

            "tE":
                float(
                    x_final[2]
                ),

            "rho":
                float(
                    x_final[3]
                ),

            "success":
                bool(
                    result.success
                ),

            "status":
                int(
                    result.status
                ),

            "nfev":
                int(
                    result.nfev
                ),

            "message":
                str(
                    result.message
                ),

            "wall_s":
                float(
                    time.perf_counter()
                    - t_start
                ),
        }

    except Exception as exc:

        return {
            "chi2":
                np.inf,

            "t0":
                np.nan,

            "u0":
                np.nan,

            "tE":
                np.nan,

            "rho":
                np.nan,

            "success":
                False,

            "status":
                -999,

            "nfev":
                0,

            "message":
                repr(
                    exc
                ),

            "wall_s":
                float(
                    time.perf_counter()
                    - t_start
                ),
        }


# ============================================================
# NEW BLOCK: load inputs
# ============================================================

rows_df = pd.read_csv(
    args.rows_csv
)

policy_payload = json.loads(
    Path(
        args.policy_json
    ).read_text()
)

reference_payload = json.loads(
    Path(
        args.reference_json
    ).read_text()
)


# ============================================================
# NEW BLOCK: audit loop
# ============================================================

event_results = []

batch_start = time.perf_counter()

for index, manifest_row in rows_df.iterrows():

    row = int(
        manifest_row[
            "catalog_row"
        ]
    )

    print()
    print("=" * 100)
    print(
        f"[{index + 1}/{len(rows_df)}] "
        f"ROW {row}"
    )
    print("=" * 100)

    h5_path = resolve_event_h5(
        manifest_row[
            "h5_path"
        ],
        row,
    )

    policy_event = find_event(
        policy_payload,
        row,
    )

    reference_event = find_event(
        reference_payload,
        row,
    )

    meta = load_new_case(
        str(h5_path),
        core,
    )

    objective = build_h0_objective(
        meta
    )

    stored_h0 = vector_from_record(
        policy_event[
            "final_h0"
        ]
    )

    reference_shared = (
        vector_from_record(
            reference_event[
                "reference_winner_record"
            ]
        )
    )

    chi2_policy_h0 = float(
        policy_event[
            "chi2_h0"
        ]
    )

    chi2_h1_reference = float(
        reference_event[
            "chi2_h1_reference"
        ]
    )


    # ========================================================
    # NEW BLOCK: mandatory stored-H0 parity
    # ========================================================

    chi2_policy_direct = float(
        objective.objective_function(
            stored_h0
        )
    )

    parity_diff = (
        chi2_policy_direct
        - chi2_policy_h0
    )

    print(
        "stored H0 chi2 =",
        chi2_policy_h0,
    )

    print(
        "direct H0 chi2 =",
        chi2_policy_direct,
    )

    print(
        "parity diff     =",
        parity_diff,
    )

    if not np.isclose(
        chi2_policy_direct,
        chi2_policy_h0,
        rtol=0.0,
        atol=1e-6,
    ):

        raise RuntimeError(
            f"row={row}: "
            "stored H0 parity failure"
        )


    # ========================================================
    # NEW BLOCK: bounds
    # ========================================================

    times = all_observation_times(
        meta
    )

    lower = np.array(
        [
            float(
                np.min(times)
            ),
            -10.0,
            0.1,
            1e-7,
        ],
        dtype=float,
    )

    upper = np.array(
        [
            float(
                np.max(times)
            ),
            10.0,
            500000.0,
            10.0,
        ],
        dtype=float,
    )


    # ========================================================
    # NEW BLOCK: four morphology audit starts
    # ========================================================

    starts = [
        (
            "h0_policy",
            stored_h0,
        ),
        (
            "h0_policy_mirror",
            mirrored(
                stored_h0
            ),
        ),
        (
            "h1_reference_shared",
            reference_shared,
        ),
        (
            "h1_reference_shared_mirror",
            mirrored(
                reference_shared
            ),
        ),
    ]

    fits = []

    for label, start in starts:

        fit = fit_one_start(
            objective,
            start,
            lower,
            upper,
            args.max_nfev,
        )

        fit[
            "label"
        ] = label

        fit[
            "initial_t0"
        ] = float(
            start[0]
        )

        fit[
            "initial_u0"
        ] = float(
            start[1]
        )

        fit[
            "initial_tE"
        ] = float(
            start[2]
        )

        fit[
            "initial_rho"
        ] = float(
            start[3]
        )

        fits.append(
            fit
        )

        print(
            f"{label:29s} "
            f"chi2={fit['chi2']:.12f} "
            f"u0={fit['u0']:.6g} "
            f"tE={fit['tE']:.6g} "
            f"rho={fit['rho']:.6g}"
        )


    finite_fits = [
        fit
        for fit in fits
        if np.isfinite(
            fit[
                "chi2"
            ]
        )
    ]

    if not finite_fits:

        raise RuntimeError(
            f"row={row}: "
            "all H0 audit fits failed"
        )

    best = min(
        finite_fits,
        key=lambda x:
            x[
                "chi2"
            ],
    )

    chi2_h0_audit = float(
        best[
            "chi2"
        ]
    )

    h0_gap = (
        chi2_policy_h0
        - chi2_h0_audit
    )

    D_reference_old = (
        chi2_policy_h0
        - chi2_h1_reference
    )

    D_reference_h0_audit = (
        chi2_h0_audit
        - chi2_h1_reference
    )

    delta_D_due_h0 = (
        D_reference_h0_audit
        - D_reference_old
    )

    print()
    print(
        "BEST H0 AUDIT =",
        best[
            "label"
        ],
    )

    print(
        "chi2 H0 audit =",
        chi2_h0_audit,
    )

    print(
        "policy - audit H0 gap =",
        h0_gap,
    )

    print(
        "D old / D H0-audited =",
        D_reference_old,
        "/",
        D_reference_h0_audit,
    )

    event_results.append(
        {
            "catalog_row":
                row,

            "h5_path":
                str(
                    h5_path
                ),

            "chi2_h0_policy":
                chi2_policy_h0,

            "chi2_h0_audit":
                chi2_h0_audit,

            "h0_basin_gap":
                float(
                    h0_gap
                ),

            "h0_audit_winner":
                best[
                    "label"
                ],

            "h0_audit_winner_record":
                best,

            "chi2_h1_reference":
                chi2_h1_reference,

            "D_reference_old":
                float(
                    D_reference_old
                ),

            "D_reference_h0_audit":
                float(
                    D_reference_h0_audit
                ),

            "delta_D_due_h0":
                float(
                    delta_D_due_h0
                ),

            "parity_diff":
                float(
                    parity_diff
                ),

            "fits":
                fits,
        }
    )


# ============================================================
# NEW BLOCK: summary
# ============================================================

batch_wall_s = float(
    time.perf_counter()
    - batch_start
)

summary_df = pd.DataFrame(
    [
        {
            "catalog_row":
                event[
                    "catalog_row"
                ],

            "chi2_h0_policy":
                event[
                    "chi2_h0_policy"
                ],

            "chi2_h0_audit":
                event[
                    "chi2_h0_audit"
                ],

            "h0_basin_gap":
                event[
                    "h0_basin_gap"
                ],

            "h0_audit_winner":
                event[
                    "h0_audit_winner"
                ],

            "chi2_h1_reference":
                event[
                    "chi2_h1_reference"
                ],

            "D_reference_old":
                event[
                    "D_reference_old"
                ],

            "D_reference_h0_audit":
                event[
                    "D_reference_h0_audit"
                ],

            "delta_D_due_h0":
                event[
                    "delta_D_due_h0"
                ],
        }
        for event in event_results
    ]
)

gaps = summary_df[
    "h0_basin_gap"
].to_numpy(
    dtype=float
)

print()
print("#" * 100)
print("H0 BASIN AUDIT SUMMARY")
print("#" * 100)

print(
    summary_df.sort_values(
        "h0_basin_gap",
        ascending=False,
    ).to_string(
        index=False
    )
)

print()
print(
    "N events =",
    len(
        summary_df
    ),
)

print(
    "min gap =",
    np.min(
        gaps
    ),
)

print(
    "median gap =",
    np.median(
        gaps
    ),
)

print(
    "p90 gap =",
    np.quantile(
        gaps,
        0.90,
    ),
)

print(
    "p95 gap =",
    np.quantile(
        gaps,
        0.95,
    ),
)

print(
    "max gap =",
    np.max(
        gaps
    ),
)

for threshold in [
    1e-3,
    0.01,
    0.1,
    0.5,
    1.0,
]:

    print(
        f"gap > {threshold:g}:",
        int(
            np.sum(
                gaps
                > threshold
            )
        ),
        "/",
        len(
            gaps
        ),
    )


# ============================================================
# NEW BLOCK: save outputs
# ============================================================

out_json = Path(
    args.out_json
)

out_csv = Path(
    args.out_csv
)

out_json.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out_csv.parent.mkdir(
    parents=True,
    exist_ok=True,
)

payload = {
    "audit_name":
        "h0_morphology_basin_audit_v1",

    "audit_purpose":
        (
            "Development-only H0 basin audit "
            "on frozen N=25 H0-generated sample."
        ),

    "starts":
        [
            "h0_policy",
            "h0_policy_mirror",
            "h1_reference_shared",
            "h1_reference_shared_mirror",
        ],

    "n_events":
        len(
            event_results
        ),

    "batch_wall_s":
        batch_wall_s,

    "events":
        event_results,
}

out_json.write_text(
    json.dumps(
        payload,
        indent=2,
    )
    + "\n"
)

summary_df.to_csv(
    out_csv,
    index=False,
)

print()
print(
    "JSON =",
    out_json,
)

print(
    "CSV  =",
    out_csv,
)

print(
    "wall =",
    batch_wall_s,
    "s",
)
