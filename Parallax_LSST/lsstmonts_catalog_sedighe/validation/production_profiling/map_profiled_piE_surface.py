#!/usr/bin/env python3

# ============================================================
# NEW BLOCK: imports
# ============================================================

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    "--stability-json",
    required=True,
)

parser.add_argument(
    "--row",
    type=int,
    required=True,
)

parser.add_argument(
    "--pien-min",
    type=float,
    required=True,
)

parser.add_argument(
    "--pien-max",
    type=float,
    required=True,
)

parser.add_argument(
    "--piee-min",
    type=float,
    required=True,
)

parser.add_argument(
    "--piee-max",
    type=float,
    required=True,
)

parser.add_argument(
    "--n-grid",
    type=int,
    default=25,
)

parser.add_argument(
    "--max-nfev",
    type=int,
    default=600,
)

parser.add_argument(
    "--out-dir",
    required=True,
)

args = parser.parse_args()


# ============================================================
# NEW BLOCK: repository paths independent of current directory
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

# Script location:
#
#   lsstmonts_catalog_sedighe/
#       validation/
#           production_profiling/
#               map_profiled_piE_surface.py
#
# Therefore parents[2] is:
#
#   .../lsstmonts_catalog_sedighe

ROOT = SCRIPT_PATH.parents[2]

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

print()
print("=" * 100)
print("REPOSITORY PATHS")
print("=" * 100)

print(
    "SCRIPT_PATH =",
    SCRIPT_PATH,
)

print(
    "ROOT        =",
    ROOT,
)

print(
    "PP          =",
    PP,
)

print(
    "BA          =",
    BA,
)

print(
    "BC          =",
    BC,
)


for required_path in [
    ROOT,
    PP,
    BA,
    BC,
]:

    if not required_path.exists():

        raise FileNotFoundError(
            f"Required repository path "
            f"does not exist:\n"
            f"{required_path}"
        )


sys.path.insert(
    0,
    str(PP),
)

sys.path.insert(
    0,
    str(BA),
)

sys.path.insert(
    0,
    str(BC),
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


# ============================================================
# NEW BLOCK: local core manifest path
# ============================================================

CORE_MANIFEST = (
    BA
    / "data"
    / "refit_manifest.csv"
)

if not CORE_MANIFEST.exists():

    raise FileNotFoundError(
        "Could not find the bounds-audit "
        "manifest expected by core:\n"
        f"{CORE_MANIFEST}"
    )


# core parses argv at import time.
sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row",
    "71181",
    "--manifest",
    str(
        CORE_MANIFEST
    ),
    "--bounds-profile",
    "production_candidate",
    "--fit-scope",
    "h1",
    "--dry-run",
]


# ============================================================
# NEW BLOCK: project imports
# ============================================================

import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402

from load_new_event import load_new_case  # noqa: E402
from pyLIMA.fits import DE_fit  # noqa: E402


SHARED_ORDER = [
    "t0",
    "u0",
    "tE",
    "rho",
]

H1_ORDER = [
    "t0",
    "u0",
    "tE",
    "rho",
    "piEN",
    "piEE",
]


# ============================================================
# NEW BLOCK: generic path resolver
# ============================================================

def resolve_input_path(
    value,
    description,
):
    """
    Resolve an input file supplied on the command line.

    Resolution order:

    1. Exact path as provided.
    2. Path relative to current working directory.
    3. Path relative to project ROOT.
    """

    raw = Path(
        value
    ).expanduser()

    candidates = []

    candidates.append(
        raw
    )

    if not raw.is_absolute():

        candidates.append(
            (
                Path.cwd()
                / raw
            ).resolve()
        )

        candidates.append(
            (
                ROOT
                / raw
            ).resolve()
        )

    for candidate in candidates:

        if candidate.exists():

            return candidate.resolve()

    candidate_text = "\n".join(
        f"  - {x}"
        for x in candidates
    )

    raise FileNotFoundError(
        f"Could not resolve {description}.\n"
        f"Tried:\n"
        f"{candidate_text}"
    )


# ============================================================
# NEW BLOCK: resolve frozen H5 on CHE or local repository
# ============================================================

def resolve_event_h5(
    stored_path,
    catalog_row,
):
    """
    Use the exact stored CHE path when available.

    Otherwise use the frozen local validation copy:

      validation/production_profiling/
      runtime_local/
      h0_h1_basin_validation_h5/
      Event_<row>.h5
    """

    stored = Path(
        str(stored_path)
    ).expanduser()

    if stored.exists():

        return stored.resolve()


    local_candidate = (
        PP
        / "runtime_local"
        / "h0_h1_basin_validation_h5"
        / f"Event_{int(catalog_row)}.h5"
    )

    if local_candidate.exists():

        return local_candidate.resolve()


    # Additional conservative fallback:
    # search only below production_profiling/runtime_local.
    runtime_root = (
        PP
        / "runtime_local"
    )

    if runtime_root.exists():

        matches = list(
            runtime_root.rglob(
                f"Event_{int(catalog_row)}.h5"
            )
        )

        if len(matches) == 1:

            return matches[
                0
            ].resolve()

        if len(matches) > 1:

            matches_text = "\n".join(
                f"  - {x}"
                for x in matches
            )

            raise RuntimeError(
                "Found multiple candidate "
                f"H5 files for row="
                f"{catalog_row}:\n"
                f"{matches_text}"
            )


    raise FileNotFoundError(
        "Could not find frozen event H5.\n"
        f"Stored manifest path:\n"
        f"  {stored}\n"
        f"Expected local path:\n"
        f"  {local_candidate}"
    )


# ============================================================
# NEW BLOCK: data helpers
# ============================================================

def find_event(
    payload,
    row,
):

    for event in payload[
        "events"
    ]:

        if int(
            event[
                "catalog_row"
            ]
        ) == int(row):

            return event

    raise KeyError(
        f"row={row} not found"
    )


def finite_best_stability_fit(
    event,
):

    fits = [
        x
        for x in event[
            "fits"
        ]
        if (
            x.get(
                "chi2"
            )
            is not None
            and np.isfinite(
                float(
                    x[
                        "chi2"
                    ]
                )
            )
        )
    ]

    if not fits:

        raise RuntimeError(
            "No finite stability fit."
        )

    return min(
        fits,
        key=lambda x:
            float(
                x[
                    "chi2"
                ]
            ),
    )


def shared_vector(
    record,
):

    return np.array(
        [
            float(
                record[
                    name
                ]
            )
            for name
            in SHARED_ORDER
        ],
        dtype=float,
    )


def all_observation_times(
    meta,
):

    times = []

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

        if len(
            curve
        ) == 0:

            continue

        times.extend(
            np.asarray(
                curve[
                    :,
                    0
                ],
                dtype=float,
            ).tolist()
        )

    if not times:

        raise RuntimeError(
            "No observation times."
        )

    return np.asarray(
        times,
        dtype=float,
    )


# ============================================================
# NEW BLOCK: exact H1 objective
# ============================================================

def build_h1_objective(
    meta,
):

    lsst_lcs = {
        band:
            meta[
                "curves"
            ][band]
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
        meta[
            "Source"
        ],
        None,
        meta[
            "curves"
        ][
            "W149"
        ],
        lsst_lcs,
        ra=
            meta[
                "event_ra"
            ],
        dec=
            meta[
                "event_dec"
            ],
        roman_name=
            "Roman",
    )

    fit_params = (
        fit_lc.initial_params_for_fit_model(
            meta[
                "true_params"
            ],
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

    if actual_order != H1_ORDER:

        raise RuntimeError(
            "Unexpected H1 parameter order.\n"
            f"Expected: {H1_ORDER}\n"
            f"Actual:   {actual_order}"
        )

    return objective


# ============================================================
# NEW BLOCK: normalized residual vector at fixed piE
# ============================================================

def normalized_residuals(
    objective,
    shared,
    piEN,
    piEE,
):

    x = np.array(
        [
            shared[
                0
            ],
            shared[
                1
            ],
            shared[
                2
            ],
            shared[
                3
            ],
            float(
                piEN
            ),
            float(
                piEE
            ),
        ],
        dtype=float,
    )

    _, pyparams = (
        objective.model_chi2(
            x
        )
    )

    residuals, errors = (
        objective.photometric_model_residuals(
            pyparams
        )
    )

    chunks = []

    for (
        residual,
        error,
    ) in zip(
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

        if len(
            residual
        ) == 0:

            continue

        chunks.append(
            residual
            / error
        )

    if not chunks:

        raise RuntimeError(
            "No photometric residuals "
            "were returned."
        )

    return np.concatenate(
        chunks
    )


# ============================================================
# NEW BLOCK: fixed-piE profile fit
# ============================================================

def profile_one_point(
    objective,
    piEN,
    piEE,
    starts,
    lower,
    upper,
    max_nfev,
):

    best = None

    for (
        label,
        start,
    ) in starts:

        x0 = np.asarray(
            start,
            dtype=float,
        ).copy()

        eps = 1e-10

        x0 = np.maximum(
            x0,
            lower
            + eps,
        )

        x0 = np.minimum(
            x0,
            upper
            - eps,
        )

        try:

            result = least_squares(
                lambda x:
                    normalized_residuals(
                        objective,
                        x,
                        piEN,
                        piEE,
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
                max_nfev=
                    max_nfev,
            )

            residual = (
                normalized_residuals(
                    objective,
                    result.x,
                    piEN,
                    piEE,
                )
            )

            chi2 = float(
                np.sum(
                    residual
                    ** 2
                )
            )

            candidate = {
                "branch":
                    label,

                "chi2":
                    chi2,

                "x":
                    result.x.copy(),

                "nfev":
                    int(
                        result.nfev
                    ),

                "success":
                    bool(
                        result.success
                    ),

                "message":
                    str(
                        result.message
                    ),
            }

        except Exception as exc:

            candidate = {
                "branch":
                    label,

                "chi2":
                    np.inf,

                "x":
                    None,

                "nfev":
                    0,

                "success":
                    False,

                "message":
                    repr(
                        exc
                    ),
            }

        if (
            best is None
            or candidate[
                "chi2"
            ]
            < best[
                "chi2"
            ]
        ):

            best = candidate

    return best


# ============================================================
# NEW BLOCK: resolve input files
# ============================================================

ROWS_CSV = resolve_input_path(
    args.rows_csv,
    "rows CSV",
)

POLICY_JSON = resolve_input_path(
    args.policy_json,
    "policy JSON",
)

STABILITY_JSON = resolve_input_path(
    args.stability_json,
    "stability JSON",
)

print()
print("=" * 100)
print("INPUT FILES")
print("=" * 100)

print(
    "ROWS_CSV       =",
    ROWS_CSV,
)

print(
    "POLICY_JSON    =",
    POLICY_JSON,
)

print(
    "STABILITY_JSON =",
    STABILITY_JSON,
)


# ============================================================
# NEW BLOCK: load manifest and resolve event H5
# ============================================================

manifest = pd.read_csv(
    ROWS_CSV
)

m = manifest[
    manifest[
        "catalog_row"
    ].astype(int)
    == int(
        args.row
    )
]

if len(
    m
) != 1:

    raise RuntimeError(
        f"row={args.row}: "
        f"manifest entries="
        f"{len(m)}"
    )


stored_h5_path = str(
    m.iloc[
        0
    ][
        "h5_path"
    ]
)

H5_PATH = resolve_event_h5(
    stored_h5_path,
    args.row,
)

print()
print("=" * 100)
print("RUNTIME DATA PATHS")
print("=" * 100)

print(
    "stored H5 path =",
    stored_h5_path,
)

print(
    "resolved H5    =",
    H5_PATH,
)



# ============================================================
# NEW BLOCK: load frozen event
# ============================================================

meta = load_new_case(
    str(
        H5_PATH
    ),
    core,
)

if int(
    meta[
        "row"
    ]
) != int(
    args.row
):

    raise RuntimeError(
        "Loaded event row does not "
        "match requested catalog row.\n"
        f"Requested: {args.row}\n"
        f"Loaded:    {meta['row']}"
    )


# ============================================================
# NEW BLOCK: load stored fit results
# ============================================================

policy_payload = json.loads(
    POLICY_JSON.read_text()
)

stability_payload = json.loads(
    STABILITY_JSON.read_text()
)

policy_event = find_event(
    policy_payload,
    args.row,
)

stability_event = find_event(
    stability_payload,
    args.row,
)

oracle = (
    finite_best_stability_fit(
        stability_event
    )
)

h0 = policy_event[
    "final_h0"
]

policy = policy_event[
    "final_h1"
]


chi2_h0 = float(
    policy_event[
        "chi2_h0"
    ]
)

chi2_policy = float(
    policy_event[
        "chi2_h1"
    ]
)

chi2_oracle = float(
    stability_event[
        "best_local_chi2"
    ]
)


# ============================================================
# NEW BLOCK: shared-parameter bounds
# ============================================================

times = all_observation_times(
    meta
)

lower = np.array(
    [
        float(
            np.min(
                times
            )
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
            np.max(
                times
            )
        ),
        10.0,
        500000.0,
        10.0,
    ],
    dtype=float,
)


# ============================================================
# NEW BLOCK: diagnostic basin seeds
# ============================================================

h0_start = shared_vector(
    h0
)

h0_mirror = (
    h0_start.copy()
)

h0_mirror[
    1
] *= -1.0

policy_start = shared_vector(
    policy
)

oracle_start = shared_vector(
    oracle
)

starts = [
    (
        "H0",
        h0_start,
    ),
    (
        "H0_mirror",
        h0_mirror,
    ),
    (
        "policy",
        policy_start,
    ),
    (
        "oracle",
        oracle_start,
    ),
]


# ============================================================
# NEW BLOCK: exact H1 profiled objective
# ============================================================

objective = build_h1_objective(
    meta,
)


# ============================================================
# NEW BLOCK: control points
# ============================================================

control_points = [
    (
        "H0_nested",
        0.0,
        0.0,
        chi2_h0,
    ),
    (
        "policy",
        float(
            policy[
                "piEN"
            ]
        ),
        float(
            policy[
                "piEE"
            ]
        ),
        chi2_policy,
    ),
    (
        "oracle",
        float(
            oracle[
                "piEN"
            ]
        ),
        float(
            oracle[
                "piEE"
            ]
        ),
        chi2_oracle,
    ),
]


print()
print("=" * 100)
print(
    f"PROFILED piE MAP "
    f"row={args.row}"
)
print("=" * 100)

print(
    "H0 chi2     =",
    chi2_h0,
)

print(
    "policy chi2 =",
    chi2_policy,
)

print(
    "oracle chi2 =",
    chi2_oracle,
)

print()
print(
    "shared lower bounds =",
    dict(
        zip(
            SHARED_ORDER,
            lower,
        )
    ),
)

print(
    "shared upper bounds =",
    dict(
        zip(
            SHARED_ORDER,
            upper,
        )
    ),
)


print()
print("=" * 100)
print("CONTROL POINTS")
print("=" * 100)

control_results = []

for (
    label,
    piEN,
    piEE,
    expected,
) in control_points:

    fit = profile_one_point(
        objective,
        piEN,
        piEE,
        starts,
        lower,
        upper,
        args.max_nfev,
    )

    difference = (
        fit[
            "chi2"
        ]
        - expected
    )

    control_x = fit["x"]

    control_results.append(
        {
            "label":
                label,

            "piEN":
                piEN,

            "piEE":
                piEE,

            "profile_chi2":
                fit[
                    "chi2"
                ],

            "stored_chi2":
                expected,

            "difference":
                difference,

            "branch":
                fit[
                    "branch"
                ],

            "t0":
                (
                    float(control_x[0])
                    if control_x is not None
                    else np.nan
                ),

            "u0":
                (
                    float(control_x[1])
                    if control_x is not None
                    else np.nan
                ),

            "tE":
                (
                    float(control_x[2])
                    if control_x is not None
                    else np.nan
                ),

            "rho":
                (
                    float(control_x[3])
                    if control_x is not None
                    else np.nan
                ),
        }
    )

    print()
    print(
        label
    )

    print(
        "  piEN         =",
        piEN,
    )

    print(
        "  piEE         =",
        piEE,
    )

    print(
        "  profile chi2 =",
        fit[
            "chi2"
        ],
    )

    print(
        "  stored chi2  =",
        expected,
    )

    print(
        "  difference   =",
        difference,
    )

    print(
        "  branch       =",
        fit[
            "branch"
        ],
    )

    if fit["x"] is not None:

        print(
            "  t0           =",
            fit["x"][0],
        )

        print(
            "  u0           =",
            fit["x"][1],
        )

        print(
            "  tE           =",
            fit["x"][2],
        )

        print(
            "  rho          =",
            fit["x"][3],
        )


# ============================================================
# NEW BLOCK: output directory
# ============================================================

OUT_DIR = Path(
    args.out_dir
).expanduser()

if not OUT_DIR.is_absolute():

    OUT_DIR = (
        Path.cwd()
        / OUT_DIR
    ).resolve()

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print()
print(
    "OUT_DIR =",
    OUT_DIR,
)


# ============================================================
# NEW BLOCK: save control-point audit
# ============================================================

control_df = pd.DataFrame(
    control_results
)

control_path = (
    OUT_DIR
    / (
        f"profiled_piE_"
        f"row{args.row}_"
        f"control_points.csv"
    )
)

control_df.to_csv(
    control_path,
    index=False,
)


# ============================================================
# NEW BLOCK: grid
# ============================================================

piEN_grid = np.linspace(
    args.pien_min,
    args.pien_max,
    args.n_grid,
)

piEE_grid = np.linspace(
    args.piee_min,
    args.piee_max,
    args.n_grid,
)

csv_path = (
    OUT_DIR
    / (
        f"profiled_piE_"
        f"row{args.row}.csv"
    )
)


# ============================================================
# NEW BLOCK: resume support
# ============================================================

records = {}

if csv_path.exists():

    old = pd.read_csv(
        csv_path
    )

    required_columns = {
        "j",
        "i",
        "piEN",
        "piEE",
        "chi2",
    }

    if not required_columns.issubset(
        old.columns
    ):

        raise RuntimeError(
            "Existing resume CSV does "
            "not have expected columns:\n"
            f"{csv_path}"
        )

    for _, rec in old.iterrows():

        key = (
            int(
                rec[
                    "j"
                ]
            ),
            int(
                rec[
                    "i"
                ]
            ),
        )

        records[
            key
        ] = rec.to_dict()

    print()
    print(
        "RESUME:",
        len(
            records
        ),
        "grid points already present",
    )


# ============================================================
# NEW BLOCK: evaluate profile grid
# ============================================================

for j, piEE in enumerate(
    piEE_grid
):

    print()
    print(
        f"row {args.row}: "
        f"piEE row "
        f"{j + 1}/"
        f"{len(piEE_grid)} "
        f"piEE={piEE:.6g}"
    )

    for i, piEN in enumerate(
        piEN_grid
    ):

        key = (
            j,
            i,
        )

        if key in records:

            continue

        best = profile_one_point(
            objective,
            float(
                piEN
            ),
            float(
                piEE
            ),
            starts,
            lower,
            upper,
            args.max_nfev,
        )

        x = best[
            "x"
        ]

        record = {
            "j":
                j,

            "i":
                i,

            "piEN":
                float(
                    piEN
                ),

            "piEE":
                float(
                    piEE
                ),

            "chi2":
                float(
                    best[
                        "chi2"
                    ]
                ),

            "branch":
                best[
                    "branch"
                ],

            "success":
                best[
                    "success"
                ],

            "nfev":
                best[
                    "nfev"
                ],

            "t0":
                (
                    float(
                        x[
                            0
                        ]
                    )
                    if x
                    is not None
                    else np.nan
                ),

            "u0":
                (
                    float(
                        x[
                            1
                        ]
                    )
                    if x
                    is not None
                    else np.nan
                ),

            "tE":
                (
                    float(
                        x[
                            2
                        ]
                    )
                    if x
                    is not None
                    else np.nan
                ),

            "rho":
                (
                    float(
                        x[
                            3
                        ]
                    )
                    if x
                    is not None
                    else np.nan
                ),
        }

        records[
            key
        ] = record


    # Save after each piEE row so that the run is resumable.
    checkpoint = pd.DataFrame(
        list(
            records.values()
        )
    ).sort_values(
        [
            "j",
            "i",
        ]
    )

    checkpoint.to_csv(
        csv_path,
        index=False,
    )


# ============================================================
# NEW BLOCK: reconstruct matrices
# ============================================================

df = pd.read_csv(
    csv_path
)

expected_n = (
    args.n_grid
    * args.n_grid
)

if len(
    df
) != expected_n:

    raise RuntimeError(
        f"Expected "
        f"{expected_n} "
        f"grid points, got "
        f"{len(df)}"
    )


Z = np.full(
    (
        args.n_grid,
        args.n_grid,
    ),
    np.nan,
)

BRANCH = np.empty(
    (
        args.n_grid,
        args.n_grid,
    ),
    dtype=object,
)

for _, rec in df.iterrows():

    j = int(
        rec[
            "j"
        ]
    )

    i = int(
        rec[
            "i"
        ]
    )

    Z[
        j,
        i
    ] = float(
        rec[
            "chi2"
        ]
    )

    BRANCH[
        j,
        i
    ] = str(
        rec[
            "branch"
        ]
    )


if not np.all(
    np.isfinite(
        Z
    )
):

    bad = np.argwhere(
        ~np.isfinite(
            Z
        )
    )

    raise RuntimeError(
        "Non-finite values remain "
        "in profile surface.\n"
        f"Indices:\n{bad}"
    )


surface_min = float(
    np.min(
        Z
    )
)

relative = (
    Z
    - surface_min
)

D_vs_H0 = (
    chi2_h0
    - Z
)


# ============================================================
# NEW BLOCK: numerical summary
# ============================================================

best_index = np.unravel_index(
    np.argmin(
        Z
    ),
    Z.shape,
)

best_j, best_i = (
    best_index
)

best_piEN = float(
    piEN_grid[
        best_i
    ]
)

best_piEE = float(
    piEE_grid[
        best_j
    ]
)

best_record = df[
    (
        df[
            "j"
        ].astype(int)
        == best_j
    )
    &
    (
        df[
            "i"
        ].astype(int)
        == best_i
    )
].iloc[
    0
]


print()
print("=" * 100)
print("SURFACE SUMMARY")
print("=" * 100)

print(
    "grid minimum chi2 =",
    surface_min,
)

print(
    "grid best piEN    =",
    best_piEN,
)

print(
    "grid best piEE    =",
    best_piEE,
)

print(
    "grid best branch  =",
    best_record[
        "branch"
    ],
)

print(
    "grid best t0      =",
    best_record[
        "t0"
    ],
)

print(
    "grid best u0      =",
    best_record[
        "u0"
    ],
)

print(
    "grid best tE      =",
    best_record[
        "tE"
    ],
)

print(
    "grid best rho     =",
    best_record[
        "rho"
    ],
)

print(
    "gap to oracle     =",
    surface_min
    - chi2_oracle,
)

print(
    "D grid            =",
    chi2_h0
    - surface_min,
)


# ============================================================
# NEW BLOCK: save surface summary
# ============================================================

summary = {
    "catalog_row":
        int(
            args.row
        ),

    "h5_path":
        str(
            H5_PATH
        ),

    "n_grid":
        int(
            args.n_grid
        ),

    "pien_min":
        float(
            args.pien_min
        ),

    "pien_max":
        float(
            args.pien_max
        ),

    "piee_min":
        float(
            args.piee_min
        ),

    "piee_max":
        float(
            args.piee_max
        ),

    "chi2_h0":
        chi2_h0,

    "chi2_policy":
        chi2_policy,

    "chi2_oracle":
        chi2_oracle,

    "grid_min_chi2":
        surface_min,

    "grid_best_piEN":
        best_piEN,

    "grid_best_piEE":
        best_piEE,

    "grid_best_branch":
        str(
            best_record[
                "branch"
            ]
        ),

    "grid_best_t0":
        float(
            best_record[
                "t0"
            ]
        ),

    "grid_best_u0":
        float(
            best_record[
                "u0"
            ]
        ),

    "grid_best_tE":
        float(
            best_record[
                "tE"
            ]
        ),

    "grid_best_rho":
        float(
            best_record[
                "rho"
            ]
        ),

    "grid_gap_to_oracle":
        float(
            surface_min
            - chi2_oracle
        ),

    "D_grid":
        float(
            chi2_h0
            - surface_min
        ),
}

summary_path = (
    OUT_DIR
    / (
        f"profiled_piE_"
        f"row{args.row}_"
        f"summary.json"
    )
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)


# ============================================================
# NEW BLOCK: mesh
# ============================================================

X, Y = np.meshgrid(
    piEN_grid,
    piEE_grid,
)


# ============================================================
# NEW BLOCK: plot relative profile surface
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        7,
    )
)

cf = ax.contourf(
    X,
    Y,
    relative,
    levels=40,
)

fig.colorbar(
    cf,
    ax=ax,
    label=(
        r"$\chi^2_{\rm prof}"
        r"-\chi^2_{\rm min}$"
    ),
)

for level in [
    0.5,
    1.0,
    2.0,
    4.0,
    6.0,
    9.0,
]:

    if (
        np.nanmin(
            relative
        )
        <= level
        <= np.nanmax(
            relative
        )
    ):

        ax.contour(
            X,
            Y,
            relative,
            levels=[
                level
            ],
            linewidths=0.8,
        )


ax.scatter(
    0.0,
    0.0,
    marker="+",
    s=100,
    label="H0",
)

ax.scatter(
    float(
        policy[
            "piEN"
        ]
    ),
    float(
        policy[
            "piEE"
        ]
    ),
    marker="s",
    s=55,
    label="H1 policy",
)

ax.scatter(
    float(
        oracle[
            "piEN"
        ]
    ),
    float(
        oracle[
            "piEE"
        ]
    ),
    marker="*",
    s=110,
    label="H1 oracle",
)

ax.scatter(
    best_piEN,
    best_piEE,
    marker="x",
    s=80,
    label="grid min",
)

ax.set_xlabel(
    r"$\pi_{E,N}$"
)

ax.set_ylabel(
    r"$\pi_{E,E}$"
)

ax.set_title(
    (
        f"row {args.row}: "
        r"profiled $\chi^2(\pi_{E,N},\pi_{E,E})$"
    )
)

ax.legend()

fig.tight_layout()

relative_png = (
    OUT_DIR
    / (
        f"profiled_piE_"
        f"row{args.row}_"
        f"delta_chi2.png"
    )
)

fig.savefig(
    relative_png,
    dpi=200,
)

plt.close(
    fig
)


# ============================================================
# NEW BLOCK: plot LRT improvement relative to H0
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        7,
    )
)

cf = ax.contourf(
    X,
    Y,
    D_vs_H0,
    levels=40,
)

fig.colorbar(
    cf,
    ax=ax,
    label=(
        r"$D(\pi_E)="
        r"\chi^2_{H0}"
        r"-\chi^2_{\rm prof}$"
    ),
)

if (
    np.nanmin(
        D_vs_H0
    )
    <= 0.0
    <= np.nanmax(
        D_vs_H0
    )
):

    ax.contour(
        X,
        Y,
        D_vs_H0,
        levels=[
            0.0
        ],
        linewidths=1.0,
    )


ax.scatter(
    0.0,
    0.0,
    marker="+",
    s=100,
    label="H0",
)

ax.scatter(
    float(
        policy[
            "piEN"
        ]
    ),
    float(
        policy[
            "piEE"
        ]
    ),
    marker="s",
    s=55,
    label="H1 policy",
)

ax.scatter(
    float(
        oracle[
            "piEN"
        ]
    ),
    float(
        oracle[
            "piEE"
        ]
    ),
    marker="*",
    s=110,
    label="H1 oracle",
)

ax.scatter(
    best_piEN,
    best_piEE,
    marker="x",
    s=80,
    label="grid min",
)

ax.set_xlabel(
    r"$\pi_{E,N}$"
)

ax.set_ylabel(
    r"$\pi_{E,E}$"
)

ax.set_title(
    (
        f"row {args.row}: "
        "LRT improvement relative to H0"
    )
)

ax.legend()

fig.tight_layout()

lrt_png = (
    OUT_DIR
    / (
        f"profiled_piE_"
        f"row{args.row}_"
        f"D_vs_H0.png"
    )
)

fig.savefig(
    lrt_png,
    dpi=200,
)

plt.close(
    fig
)


# ============================================================
# NEW BLOCK: plot winning local branch
# ============================================================

branch_names = [
    "H0",
    "H0_mirror",
    "policy",
    "oracle",
]

branch_to_int = {
    name:
        i
    for i, name
    in enumerate(
        branch_names
    )
}

branch_matrix = np.full(
    (
        args.n_grid,
        args.n_grid,
    ),
    np.nan,
)

for j in range(
    args.n_grid
):

    for i in range(
        args.n_grid
    ):

        branch = BRANCH[
            j,
            i
        ]

        if branch in branch_to_int:

            branch_matrix[
                j,
                i
            ] = branch_to_int[
                branch
            ]


fig, ax = plt.subplots(
    figsize=(
        8,
        7,
    )
)

mesh = ax.pcolormesh(
    X,
    Y,
    branch_matrix,
    shading="auto",
)

cbar = fig.colorbar(
    mesh,
    ax=ax,
    ticks=np.arange(
        len(
            branch_names
        )
    ),
)

cbar.ax.set_yticklabels(
    branch_names
)

cbar.set_label(
    "winning local branch"
)

ax.scatter(
    0.0,
    0.0,
    marker="+",
    s=100,
    label="H0",
)

ax.scatter(
    float(
        policy[
            "piEN"
        ]
    ),
    float(
        policy[
            "piEE"
        ]
    ),
    marker="s",
    s=55,
    label="H1 policy",
)

ax.scatter(
    float(
        oracle[
            "piEN"
        ]
    ),
    float(
        oracle[
            "piEE"
        ]
    ),
    marker="*",
    s=110,
    label="H1 oracle",
)

ax.set_xlabel(
    r"$\pi_{E,N}$"
)

ax.set_ylabel(
    r"$\pi_{E,E}$"
)

ax.set_title(
    (
        f"row {args.row}: "
        "winning morphology branch"
    )
)

ax.legend()

fig.tight_layout()

branch_png = (
    OUT_DIR
    / (
        f"profiled_piE_"
        f"row{args.row}_"
        f"branches.png"
    )
)

fig.savefig(
    branch_png,
    dpi=200,
)

plt.close(
    fig
)


# ============================================================
# NEW BLOCK: final output
# ============================================================

print()
print("=" * 100)
print("OUTPUT FILES")
print("=" * 100)

print(
    "GRID CSV       =",
    csv_path,
)

print(
    "CONTROL CSV    =",
    control_path,
)

print(
    "SUMMARY JSON   =",
    summary_path,
)

print(
    "PROFILE PNG    =",
    relative_png,
)

print(
    "LRT PNG        =",
    lrt_png,
)

print(
    "BRANCH PNG     =",
    branch_png,
)
