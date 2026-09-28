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

from scipy.stats import qmc


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
    "--rows",
    type=int,
    nargs="+",
    required=True,
)

parser.add_argument(
    "--seed",
    type=int,
    required=True,
)

parser.add_argument(
    "--n-starts",
    type=int,
    default=32,
)

parser.add_argument(
    "--out",
    required=True,
)

args = parser.parse_args()


# ============================================================
# NEW BLOCK: repository setup
# ============================================================

ROOT = Path.cwd().resolve()

PP = ROOT / "validation" / "production_profiling"
BA = ROOT / "validation" / "bounds_audit"
BC = ROOT / "validation" / "bounds_convergence"

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


PARAMETER_ORDER = [
    "t0",
    "u0",
    "tE",
    "rho",
    "piEN",
    "piEE",
]

K_REPORT = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
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


def physical_interval(spec):

    if isinstance(spec, dict):

        if spec.get("type") != "center_width":
            raise RuntimeError(
                f"Unsupported bound spec: {spec}"
            )

        center = float(
            spec["center"]
        )

        half_width = float(
            spec["half_width"]
        )

        return (
            center - half_width,
            center + half_width,
        )

    return (
        float(spec[0]),
        float(spec[1]),
    )


def production_h1_bounds(meta):
    """
    Reconstruct the physical H1 domain used by the current
    production-candidate setup.
    """

    bounds = dict(
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


def build_transform(
    bounds,
    h0,
):
    """
    Full production physical domain, but sampled uniformly
    in transformed coordinates.

    z0 = asinh((t0 - t0_H0) / tE_H0)
    z1 = asinh(u0 / u0_scale)
    z2 = log10(tE)
    z3 = log10(rho)
    z4 = asinh(piEN / PIE_SCALE)
    z5 = asinh(piEE / PIE_SCALE)
    """

    t0_ref = float(
        h0["t0"]
    )

    tE_ref = max(
        float(
            h0["tE"]
        ),
        0.1,
    )

    u0_scale = max(
        abs(
            float(
                h0["u0"]
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

    z_bounds = np.array(
        [
            [
                np.arcsinh(
                    (t0_lo - t0_ref)
                    / tE_ref
                ),
                np.arcsinh(
                    (t0_hi - t0_ref)
                    / tE_ref
                ),
            ],
            [
                np.arcsinh(
                    u0_lo / u0_scale
                ),
                np.arcsinh(
                    u0_hi / u0_scale
                ),
            ],
            [
                np.log10(tE_lo),
                np.log10(tE_hi),
            ],
            [
                np.log10(rho_lo),
                np.log10(rho_hi),
            ],
            [
                np.arcsinh(
                    piEN_lo / PIE_SCALE
                ),
                np.arcsinh(
                    piEN_hi / PIE_SCALE
                ),
            ],
            [
                np.arcsinh(
                    piEE_lo / PIE_SCALE
                ),
                np.arcsinh(
                    piEE_hi / PIE_SCALE
                ),
            ],
        ],
        dtype=float,
    )

    physical_bounds = np.array(
        [
            [t0_lo, t0_hi],
            [u0_lo, u0_hi],
            [tE_lo, tE_hi],
            [rho_lo, rho_hi],
            [piEN_lo, piEN_hi],
            [piEE_lo, piEE_hi],
        ],
        dtype=float,
    )

    def unit_to_physical(unit_point):

        u = np.asarray(
            unit_point,
            dtype=float,
        )

        eps = np.finfo(float).eps

        u = np.clip(
            u,
            eps,
            1.0 - eps,
        )

        z = (
            z_bounds[:, 0]
            + u
            * (
                z_bounds[:, 1]
                - z_bounds[:, 0]
            )
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

        x = np.clip(
            x,
            physical_bounds[:, 0],
            physical_bounds[:, 1],
        )

        return x

    metadata = {
        "t0_ref":
            t0_ref,

        "tE_ref":
            tE_ref,

        "u0_scale":
            u0_scale,

        "piE_scale":
            PIE_SCALE,

        "z_bounds":
            z_bounds.tolist(),

        "physical_bounds":
            physical_bounds.tolist(),
    }

    return (
        unit_to_physical,
        metadata,
    )


def make_sobol_points(
    n,
    seed,
):
    """
    Deterministic scrambled Sobol sequence.

    Generate the next power of two and truncate if required.
    """

    if n < 1:
        return np.empty(
            (0, 6),
            dtype=float,
        )

    m = int(
        np.ceil(
            np.log2(n)
        )
    )

    engine = qmc.Sobol(
        d=6,
        scramble=True,
        seed=int(seed),
    )

    points = engine.random_base2(
        m=m
    )

    return points[:n]


def save_checkpoint(
    out,
    payload,
):

    out = Path(out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(out) + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n"
    )

    tmp.replace(out)


# ============================================================
# NEW BLOCK: validate requested Sobol size
# ============================================================

if args.n_starts < 1:
    raise RuntimeError(
        "--n-starts must be >= 1"
    )

if args.n_starts > 32:
    raise RuntimeError(
        "This benchmark is intentionally capped at 32 starts."
    )


# ============================================================
# NEW BLOCK: load data
# ============================================================

manifest = pd.read_csv(
    args.rows_csv
)

manifest_by_row = {
    int(r["catalog_row"]): r
    for _, r in manifest.iterrows()
}

policy_payload = json.loads(
    Path(
        args.policy_json
    ).read_text()
)

stability_payload = json.loads(
    Path(
        args.stability_json
    ).read_text()
)


# ============================================================
# NEW BLOCK: global output
# ============================================================

payload = {
    "benchmark":
        "H0_generated_Sobol6D_plus_TRF_v1",

    "seed":
        int(args.seed),

    "n_starts":
        int(args.n_starts),

    "parameter_order":
        list(
            PARAMETER_ORDER
        ),

    "sampling_coordinates": [
        "asinh((t0-t0_H0)/tE_H0)",
        "asinh(u0/u0_scale)",
        "log10(tE)",
        "log10(rho)",
        "asinh(piEN/0.3)",
        "asinh(piEE/0.3)",
    ],

    "events": [],
}


# ============================================================
# NEW BLOCK: benchmark each event
# ============================================================

for event_number, row in enumerate(
    args.rows,
    start=1,
):

    row = int(row)

    print()
    print("#" * 100)
    print(
        f"EVENT {event_number}/"
        f"{len(args.rows)} "
        f"row={row}"
    )
    print("#" * 100)

    if row not in manifest_by_row:
        raise RuntimeError(
            f"row={row} missing from manifest"
        )

    manifest_row = manifest_by_row[row]

    h5_path = str(
        manifest_row["h5_path"]
    )

    meta = load_new_case(
        h5_path,
        core,
    )

    if int(
        meta["row"]
    ) != row:
        raise RuntimeError(
            "Loaded wrong catalog row."
        )

    if (
        meta["generating_model"]
        != "H0"
    ):
        raise RuntimeError(
            f"row={row} is not H0-generated"
        )

    policy_event = find_event(
        policy_payload,
        row,
    )

    stability_event = find_event(
        stability_payload,
        row,
    )

    h0 = policy_event[
        "final_h0"
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

    chi2_target = float(
        stability_event[
            "best_local_chi2"
        ]
    )

    bounds = production_h1_bounds(
        meta
    )

    (
        unit_to_physical,
        transform_metadata,
    ) = build_transform(
        bounds,
        h0,
    )

    unit_points = make_sobol_points(
        args.n_starts,
        args.seed,
    )

    physical_points = [
        unit_to_physical(p)
        for p in unit_points
    ]

    print(
        "chi2 H0          =",
        chi2_h0,
    )

    print(
        "chi2 H1 policy   =",
        chi2_policy,
    )

    print(
        "chi2 target      =",
        chi2_target,
    )

    print(
        "policy gap       =",
        chi2_policy
        - chi2_target,
    )

    print(
        "Sobol starts     =",
        args.n_starts,
    )


    # ========================================================
    # NEW BLOCK: baseline is the already-stored production H1
    # ========================================================

    best_chi2 = chi2_policy

    best_source = "policy"

    starts = []

    cumulative = [
        {
            "k":
                0,

            "best_chi2":
                best_chi2,

            "gap_to_target":
                best_chi2
                - chi2_target,

            "best_source":
                best_source,
        }
    ]

    wall_event0 = time.time()


    # ========================================================
    # NEW BLOCK: every Sobol proposal gets a full TRF
    # ========================================================

    for i, x in enumerate(
        physical_points,
        start=1,
    ):

        initial = {
            name:
                float(x[j])
            for j, name
            in enumerate(
                PARAMETER_ORDER
            )
        }

        label = (
            f"H1_sobol6d_"
            f"{i:02d}"
        )

        print()
        print("-" * 100)

        print(
            f"row={row} "
            f"Sobol {i}/"
            f"{args.n_starts}"
        )

        print(
            "initial =",
            initial,
        )

        wall0 = time.time()

        try:

            fit = run_one_fit_full(
                core,
                fit_lc,
                meta,
                "H1",
                initial,
                label,
            )

            wall_s = (
                time.time()
                - wall0
            )

            status = fit.get(
                "status"
            )

            if status == "success":

                chi2 = float(
                    fit["chi2"]
                )

                final_parameters = {
                    name:
                        (
                            float(
                                fit[name]
                            )
                            if fit.get(name)
                            is not None
                            else None
                        )
                    for name
                    in PARAMETER_ORDER
                }

            else:

                chi2 = None

                final_parameters = None

        except Exception as exc:

            wall_s = (
                time.time()
                - wall0
            )

            status = "exception"

            chi2 = None

            final_parameters = None

            fit = {
                "exception":
                    repr(exc)
            }

        record = {
            "index":
                i,

            "label":
                label,

            "sobol_unit":
                unit_points[
                    i - 1
                ].tolist(),

            "initial_parameters":
                initial,

            "status":
                status,

            "chi2":
                chi2,

            "wall_s":
                float(
                    wall_s
                ),

            "final_parameters":
                final_parameters,
        }

        if status == "exception":
            record[
                "exception"
            ] = fit[
                "exception"
            ]

        starts.append(
            record
        )

        if (
            chi2 is not None
            and np.isfinite(chi2)
            and chi2 < best_chi2
        ):

            best_chi2 = chi2

            best_source = label

        print(
            "status          =",
            status,
        )

        print(
            "chi2            =",
            chi2,
        )

        print(
            "current best    =",
            best_chi2,
        )

        print(
            "gap target      =",
            best_chi2
            - chi2_target,
        )

        print(
            "best source     =",
            best_source,
        )

        if i in K_REPORT:

            cumulative.append(
                {
                    "k":
                        i,

                    "best_chi2":
                        float(
                            best_chi2
                        ),

                    "gap_to_target":
                        float(
                            best_chi2
                            - chi2_target
                        ),

                    "best_source":
                        best_source,
                }
            )


    event_wall_s = (
        time.time()
        - wall_event0
    )


    # ========================================================
    # NEW BLOCK: event summary
    # ========================================================

    event_output = {
        "catalog_row":
            row,

        "h5_path":
            h5_path,

        "chi2_h0":
            chi2_h0,

        "chi2_h1_policy":
            chi2_policy,

        "chi2_target":
            chi2_target,

        "policy_gap_to_target":
            chi2_policy
            - chi2_target,

        "transform":
            transform_metadata,

        "starts":
            starts,

        "cumulative":
            cumulative,

        "best_chi2":
            float(
                best_chi2
            ),

        "best_gap_to_target":
            float(
                best_chi2
                - chi2_target
            ),

        "best_source":
            best_source,

        "event_wall_s":
            float(
                event_wall_s
            ),
    }

    payload["events"].append(
        event_output
    )

    save_checkpoint(
        args.out,
        payload,
    )


    print()
    print("=" * 100)
    print(
        f"SUMMARY row={row}"
    )
    print("=" * 100)

    print(
        "policy gap =",
        chi2_policy
        - chi2_target,
    )

    for item in cumulative:

        print(
            f"K={item['k']:2d} "
            f"best_chi2="
            f"{item['best_chi2']:.12f} "
            f"gap="
            f"{item['gap_to_target']:.12f} "
            f"source="
            f"{item['best_source']}"
        )


# ============================================================
# NEW BLOCK: cross-event summary
# ============================================================

print()
print("#" * 100)
print("CROSS-EVENT SUMMARY")
print("#" * 100)

summary = []

for k in K_REPORT:

    if k > args.n_starts:
        continue

    gaps = []

    solved_01 = 0
    solved_05 = 0
    solved_10 = 0

    for event in payload["events"]:

        match = [
            x
            for x
            in event["cumulative"]
            if int(x["k"]) == int(k)
        ]

        if len(match) != 1:
            raise RuntimeError(
                f"Missing cumulative K={k} "
                f"for row="
                f"{event['catalog_row']}"
            )

        gap = float(
            match[0][
                "gap_to_target"
            ]
        )

        gaps.append(
            gap
        )

        if gap <= 0.1:
            solved_01 += 1

        if gap <= 0.5:
            solved_05 += 1

        if gap <= 1.0:
            solved_10 += 1

    gaps = np.asarray(
        gaps,
        dtype=float,
    )

    record = {
        "k":
            k,

        "median_gap":
            float(
                np.median(gaps)
            ),

        "max_gap":
            float(
                np.max(gaps)
            ),

        "N_gap_le_0p1":
            int(
                solved_01
            ),

        "N_gap_le_0p5":
            int(
                solved_05
            ),

        "N_gap_le_1":
            int(
                solved_10
            ),
    }

    summary.append(
        record
    )

    print(
        f"K={k:2d}  "
        f"median_gap="
        f"{record['median_gap']:.6f}  "
        f"max_gap="
        f"{record['max_gap']:.6f}  "
        f"<=0.1="
        f"{solved_01}/"
        f"{len(gaps)}  "
        f"<=0.5="
        f"{solved_05}/"
        f"{len(gaps)}  "
        f"<=1="
        f"{solved_10}/"
        f"{len(gaps)}"
    )

payload[
    "cross_event_summary"
] = summary

save_checkpoint(
    args.out,
    payload,
)

print()
print(
    "OUTPUT =",
    args.out,
)
