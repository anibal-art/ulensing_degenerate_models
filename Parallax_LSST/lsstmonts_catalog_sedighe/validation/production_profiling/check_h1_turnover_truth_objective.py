#!/usr/bin/env python3

"""
Cheap fixed-point diagnostic for the large-X_pi turnover.

IMPORTANT:
- Uses the EXISTING H1-generated H5 files.
- Reconstructs pyLIMA Event/Telescope/model objects in memory.
- DOES NOT run TRF, least_squares, DE optimization, multistarts, etc.
- Evaluates fixed physical parameter vectors only.
- Fluxes are profiled exactly as in the frozen production objective.

For each selected T>0 H1-generated event, evaluates:

  H1 at truth:
      chi2_h1_true

  H1 at stored H1 final:
      chi2_h1_final_direct

  H1 at stored H0 final embedded at piE=(0,0):
      chi2_h0_embedded_h1

  H0 at shared truth morphology:
      chi2_h0_shared_truth

  H0 at stored H0 final:
      chi2_h0_final_direct

This permits the decomposition

  D_truth = chi2_h0_shared_truth - chi2_h1_true

  A_H0 = chi2_h0_shared_truth - chi2_h0_fit

  I_H1 = chi2_h1_true - chi2_h1_fit

  T_fit = D_truth - A_H0 + I_H1

where:

  D_truth : H0/H1 separation at the true physical morphology
  A_H0    : amount of that separation absorbed by refitting H0
  I_H1    : improvement of H1 relative to truth on this noise realization
  T_fit   : actual production LRT statistic

The simulation truth is used ONLY as an internal diagnostic.
It does not replace the operational LRT.
"""

import argparse
import hashlib
import heapq
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--run",
    required=True,
    help="Root of the H1 production run.",
)

parser.add_argument(
    "--out-dir",
    default=(
        "validation/production_profiling/results/"
        "h1_turnover_truth_objective"
    ),
)

parser.add_argument(
    "--per-cell",
    type=int,
    default=5,
    help=(
        "Maximum events per (tE, X_pi) cell. "
        "Default 5 -> at most 175 events."
    ),
)

parser.add_argument(
    "--seed",
    type=int,
    default=20260926,
)

parser.add_argument(
    "--sample-csv",
    default=None,
    help=(
        "Optional frozen sample manifest. If supplied, skip "
        "the JSON scan and evaluate these rows."
    ),
)

parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Ignore existing diagnostic checkpoint and recompute.",
)

args = parser.parse_args()

RUN = Path(args.run).resolve()
OUT = Path(args.out_dir).resolve()

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SAMPLE_PATH = OUT / "turnover_sample_manifest.csv"
RESULT_PATH = OUT / "turnover_truth_objective_diagnostics.csv"
SUMMARY_PATH = OUT / "turnover_truth_objective_summary.csv"
TEXT_PATH = OUT / "turnover_truth_objective_summary.txt"

if not RUN.is_dir():
    raise FileNotFoundError(
        f"Run directory not found: {RUN}"
    )

if args.per_cell <= 0:
    raise ValueError(
        "--per-cell must be positive"
    )


# ============================================================
# Paths / imports
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

PP = (
    ROOT
    / "Parallax_LSST"
    / "lsstmonts_catalog_sedighe"
    / "validation"
    / "production_profiling"
)

BA = (
    ROOT
    / "Parallax_LSST"
    / "lsstmonts_catalog_sedighe"
    / "validation"
    / "bounds_audit"
)

sys.path.insert(
    0,
    str(PP),
)

sys.path.insert(
    0,
    str(BA),
)


# ============================================================
# Exact frozen production numerical environment
# ============================================================

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
# Import production fitting core
#
# This import installs the same bounded flux-profile runtime
# patch used in production. The dry-run bootstrap itself runs
# no fit.
# ============================================================

CORE_MANIFEST = (
    BA
    / "data"
    / "refit_manifest.csv"
)

if not CORE_MANIFEST.is_file():
    raise FileNotFoundError(
        f"Core manifest not found: {CORE_MANIFEST}"
    )

old_argv = list(
    sys.argv
)

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
from pyLIMA.fits import DE_fit


if core.BOUNDS_PROFILE != "production_candidate":
    raise RuntimeError(
        "Unexpected bounds profile: "
        f"{core.BOUNDS_PROFILE}"
    )


# ============================================================
# Binning used to freeze a deterministic turnover sample
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


def interval_label(edges, i):
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
    for i in range(
        len(TE_EDGES) - 1
    )
]

XPI_LABELS = [
    interval_label(
        XPI_EDGES,
        i,
    )
    for i in range(
        len(XPI_EDGES) - 1
    )
]


# ============================================================
# Utilities
# ============================================================

def finite_float(value):
    try:
        x = float(value)
    except Exception:
        return None

    if not np.isfinite(x):
        return None

    return x


def truth_get(truth, name):
    candidates = [
        name,
        f"{name}_true",
        f"true_{name}",
    ]

    for candidate in candidates:
        if candidate in truth:
            value = finite_float(
                truth[candidate]
            )
            if value is not None:
                return value

    raise KeyError(
        f"Truth parameter {name!r} not found; "
        f"keys={sorted(truth.keys())}"
    )


def find_bin(value, edges):
    if (
        not np.isfinite(value)
        or value < edges[0]
    ):
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
    row,
    seed,
):
    payload = (
        f"{seed}:{int(row)}"
    ).encode("utf-8")

    digest = hashlib.blake2b(
        payload,
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    )


def resolve_h5_path(
    event_json_path,
    data,
    catalog_row,
):
    # Prefer a path local to the same production chunk.
    chunk_dir = (
        event_json_path
        .parent
        .parent
    )

    candidate = (
        chunk_dir
        / "h5"
        / f"Event_{catalog_row}.h5"
    )

    if candidate.is_file():
        return candidate

    # Fallback to the path stored in the production JSON.
    raw = data.get(
        "h5_path"
    )

    if raw:
        candidate = Path(
            str(raw)
        )

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No H5 found for row={catalog_row}"
    )


def evaluate_objective(
    objective,
    vector,
):
    """
    pyLIMA objective_function may return either:
      - scalar chi2, or
      - a residual vector.

    Handle both safely.
    """

    value = (
        objective.objective_function(
            np.asarray(
                vector,
                dtype=float,
            )
        )
    )

    arr = np.asarray(
        value,
        dtype=float,
    )

    if arr.ndim == 0:
        return float(arr)

    return float(
        np.sum(
            arr**2
        )
    )


def build_objective(
    meta,
    fit_parallax,
):
    """
    Reconstruct the pyLIMA objective from the EXISTING H5 data.

    No synthetic photometry is generated here.
    No nonlinear optimizer is called.
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

    event = (
        fit_lc.create_fit_event(
            meta["Source"],
            str(
                core.EPHEMERIDES
            ),
            meta["curves"]["W149"],
            lsst_lcs,
            ra=meta["event_ra"],
            dec=meta["event_dec"],
            roman_name="Roman",
        )
    )

    fit_params = (
        fit_lc.initial_params_for_fit_model(
            meta["true_params"],
            "FSPL",
            fit_parallax=fit_parallax,
            fit_defaults=None,
        )
    )

    model = (
        fit_lc.build_fit_pyLIMA_model(
            event,
            "FSPL",
            fit_params,
            Origin=None,
            fit_parallax=fit_parallax,
        )
    )

    objective = DE_fit.DEfit(
        model,
        telescopes_fluxes_method="polyfit",
        loss_function="chi2",
        DE_population_size=1,
        max_iteration=1,
        display_progress=False,
        strategy="rand1bin",
    )

    return objective


def checkpoint(
    records,
):
    df = pd.DataFrame(
        records
    )

    tmp = RESULT_PATH.with_suffix(
        ".csv.tmp"
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
# Build deterministic stratified sample
#
# We scan only production Event_*.json files.
# Keep the deterministic lowest-hash rows in every cell, so
# selection is reproducible and independent of filesystem order.
# ============================================================

def build_sample():

    print()
    print("=" * 88)
    print("BUILDING STRATIFIED T>0 SAMPLE")
    print("=" * 88)

    reservoirs = {}
    n_seen = 0
    n_positive = 0
    n_eligible = 0

    for json_path in RUN.glob(
        "rows_*/events/Event_*.json"
    ):

        n_seen += 1

        if (
            n_seen % 25000
            == 0
        ):
            print(
                f"scanned JSONs = {n_seen:,}",
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

        Xpi = (
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

        x_i = find_bin(
            Xpi,
            XPI_EDGES,
        )

        if (
            te_i is None
            or x_i is None
        ):
            continue

        n_eligible += 1

        row = int(
            data["catalog_row"]
        )

        try:
            h5_path = resolve_h5_path(
                json_path,
                data,
                row,
            )
        except FileNotFoundError:
            continue

        record = {
            "catalog_row":
                row,

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
                x_i,

            "tE_bin":
                TE_LABELS[
                    te_i
                ],

            "Xpi_bin":
                XPI_LABELS[
                    x_i
                ],
        }

        cell = (
            te_i,
            x_i,
        )

        priority = (
            deterministic_priority(
                row,
                args.seed,
            )
        )

        heap = reservoirs.setdefault(
            cell,
            [],
        )

        item = (
            -priority,
            row,
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

    for cell, heap in reservoirs.items():
        for _, _, record in heap:
            selected.append(
                record
            )

    sample = pd.DataFrame(
        selected
    )

    if sample.empty:
        raise RuntimeError(
            "No eligible events selected."
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
        f"JSON files scanned      = {n_seen:,}"
    )
    print(
        f"T>0 H1 events           = {n_positive:,}"
    )
    print(
        f"eligible with truth/H5  = {n_eligible:,}"
    )
    print(
        f"selected events         = {len(sample):,}"
    )
    print(
        f"sample manifest         = {SAMPLE_PATH}"
    )

    print()
    print(
        "Selected counts by cell:"
    )

    counts = (
        sample
        .groupby(
            [
                "tE_bin",
                "Xpi_bin",
            ],
            observed=True,
        )
        .size()
        .rename("N")
        .reset_index()
    )

    print(
        counts.to_string(
            index=False
        )
    )

    return sample


# ============================================================
# Load/build sample
# ============================================================

if args.sample_csv is not None:

    sample_path = Path(
        args.sample_csv
    ).resolve()

    if not sample_path.is_file():
        raise FileNotFoundError(
            sample_path
        )

    sample = pd.read_csv(
        sample_path
    )

    print(
        f"Using supplied sample: "
        f"{sample_path}"
    )

elif SAMPLE_PATH.is_file():

    sample = pd.read_csv(
        SAMPLE_PATH
    )

    print(
        f"Using existing frozen sample: "
        f"{SAMPLE_PATH}"
    )

else:

    sample = build_sample()


# ============================================================
# Existing checkpoint / resume
# ============================================================

records = []

completed_rows = set()

if (
    RESULT_PATH.is_file()
    and not args.overwrite
):

    previous = pd.read_csv(
        RESULT_PATH
    )

    records = (
        previous
        .to_dict(
            orient="records"
        )
    )

    if (
        "catalog_row"
        in previous.columns
    ):
        completed_rows = set(
            pd.to_numeric(
                previous[
                    "catalog_row"
                ],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )

    print(
        f"Resuming from {RESULT_PATH}: "
        f"{len(completed_rows)} rows already processed."
    )


# ============================================================
# Parameter orders
# ============================================================

H1_ORDER = (
    fit_lc.fit_parameter_order(
        "FSPL",
        fit_parallax=True,
    )
)

H0_ORDER = (
    fit_lc.fit_parameter_order(
        "FSPL",
        fit_parallax=False,
    )
)

expected_h1 = [
    "t0",
    "u0",
    "tE",
    "rho",
    "piEN",
    "piEE",
]

expected_h0 = [
    "t0",
    "u0",
    "tE",
    "rho",
]

if list(H1_ORDER) != expected_h1:
    raise RuntimeError(
        f"Unexpected H1 parameter order: "
        f"{H1_ORDER}"
    )

if list(H0_ORDER) != expected_h0:
    raise RuntimeError(
        f"Unexpected H0 parameter order: "
        f"{H0_ORDER}"
    )

print()
print(
    "H1 parameter order =",
    H1_ORDER,
)

print(
    "H0 parameter order =",
    H0_ORDER,
)


# ============================================================
# Evaluate selected events
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

n_total = len(
    sample
)

for i, row_entry in sample.iterrows():

    row = int(
        row_entry[
            "catalog_row"
        ]
    )

    if (
        row in completed_rows
        and not args.overwrite
    ):
        continue

    print()
    print("=" * 88)
    print(
        f"EVENT {i + 1}/{n_total} "
        f"catalog_row={row}"
    )
    print("=" * 88)

    json_path = Path(
        row_entry[
            "json_path"
        ]
    )

    h5_path = Path(
        row_entry[
            "h5_path"
        ]
    )

    try:

        data = json.loads(
            json_path.read_text()
        )

        meta = load_new_case(
            str(
                h5_path
            ),
            core,
        )

        if (
            meta[
                "generating_model"
            ]
            != "H1"
        ):
            raise RuntimeError(
                "Expected H1-generated "
                f"event, got "
                f"{meta['generating_model']}"
            )

        # --------------------------------------------
        # Reconstruct both fixed-point objectives.
        # --------------------------------------------

        h1_objective = (
            build_objective(
                meta,
                fit_parallax=True,
            )
        )

        h0_objective = (
            build_objective(
                meta,
                fit_parallax=False,
            )
        )

        truth = dict(
            meta[
                "true_params"
            ]
        )

        final_h0 = data[
            "final_h0"
        ]

        final_h1 = data[
            "final_h1"
        ]

        # --------------------------------------------
        # Fixed vectors
        # --------------------------------------------

        x_h1_true = np.array(
            [
                float(
                    truth[name]
                )
                for name
                in H1_ORDER
            ],
            dtype=float,
        )

        x_h1_final = np.array(
            [
                float(
                    final_h1[name]
                )
                for name
                in H1_ORDER
            ],
            dtype=float,
        )

        h0_embedded = {
            "t0":
                float(
                    final_h0["t0"]
                ),
            "u0":
                float(
                    final_h0["u0"]
                ),
            "tE":
                float(
                    final_h0["tE"]
                ),
            "rho":
                float(
                    final_h0["rho"]
                ),
            "piEN":
                0.0,
            "piEE":
                0.0,
        }

        x_h0_embedded_h1 = (
            np.array(
                [
                    h0_embedded[name]
                    for name
                    in H1_ORDER
                ],
                dtype=float,
            )
        )

        shared_truth = {
            "t0":
                float(
                    truth["t0"]
                ),
            "u0":
                float(
                    truth["u0"]
                ),
            "tE":
                float(
                    truth["tE"]
                ),
            "rho":
                float(
                    truth["rho"]
                ),
        }

        x_h0_shared_truth = np.array(
            [
                shared_truth[name]
                for name
                in H0_ORDER
            ],
            dtype=float,
        )

        x_h0_final = np.array(
            [
                float(
                    final_h0[name]
                )
                for name
                in H0_ORDER
            ],
            dtype=float,
        )

        shared_truth_embedded = {
            **shared_truth,
            "piEN": 0.0,
            "piEE": 0.0,
        }

        x_h0_truth_embedded_h1 = (
            np.array(
                [
                    shared_truth_embedded[
                        name
                    ]
                    for name
                    in H1_ORDER
                ],
                dtype=float,
            )
        )

        # --------------------------------------------
        # DIRECT FIXED-POINT EVALUATIONS
        # --------------------------------------------

        chi2_h1_true = (
            evaluate_objective(
                h1_objective,
                x_h1_true,
            )
        )

        chi2_h1_final_direct = (
            evaluate_objective(
                h1_objective,
                x_h1_final,
            )
        )

        chi2_h0_embedded_h1 = (
            evaluate_objective(
                h1_objective,
                x_h0_embedded_h1,
            )
        )

        chi2_h0_shared_truth = (
            evaluate_objective(
                h0_objective,
                x_h0_shared_truth,
            )
        )

        chi2_h0_final_direct = (
            evaluate_objective(
                h0_objective,
                x_h0_final,
            )
        )

        chi2_h0_truth_embedded_h1 = (
            evaluate_objective(
                h1_objective,
                x_h0_truth_embedded_h1,
            )
        )

        # --------------------------------------------
        # Stored production values
        # --------------------------------------------

        chi2_h0_fit = float(
            data[
                "chi2_h0"
            ]
        )

        chi2_h1_fit = float(
            data[
                "chi2_h1"
            ]
        )

        T_fit = float(
            data[
                "delta_chi2_lrt"
            ]
        )

        # --------------------------------------------
        # Truth physical quantities from H5
        # --------------------------------------------

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
        # Scientific decomposition
        # --------------------------------------------

        D_truth = (
            chi2_h0_shared_truth
            - chi2_h1_true
        )

        A_H0 = (
            chi2_h0_shared_truth
            - chi2_h0_fit
        )

        G_H1_truth = (
            chi2_h1_fit
            - chi2_h1_true
        )

        I_H1_truth = (
            chi2_h1_true
            - chi2_h1_fit
        )

        decomposition_prediction = (
            D_truth
            - A_H0
            + I_H1_truth
        )

        decomposition_closure = (
            T_fit
            - decomposition_prediction
        )

        # --------------------------------------------
        # Numerical consistency checks
        # --------------------------------------------

        h1_direct_minus_stored = (
            chi2_h1_final_direct
            - chi2_h1_fit
        )

        h0_direct_minus_stored = (
            chi2_h0_final_direct
            - chi2_h0_fit
        )

        embedded_h0_minus_stored = (
            chi2_h0_embedded_h1
            - chi2_h0_fit
        )

        truth_h0_nested_difference = (
            chi2_h0_truth_embedded_h1
            - chi2_h0_shared_truth
        )

        tol = 1.0e-6

        objective_consistent = bool(
            abs(
                h1_direct_minus_stored
            )
            < tol
            and abs(
                h0_direct_minus_stored
            )
            < tol
            and abs(
                embedded_h0_minus_stored
            )
            < tol
            and abs(
                truth_h0_nested_difference
            )
            < tol
        )

        # --------------------------------------------
        # H0 morphology displacement
        # --------------------------------------------

        h0_t0_fit = float(
            final_h0["t0"]
        )

        h0_u0_fit = float(
            final_h0["u0"]
        )

        h0_tE_fit = float(
            final_h0["tE"]
        )

        h0_rho_fit = float(
            final_h0["rho"]
        )

        h0_tE_frac_bias = (
            (
                h0_tE_fit
                - tE_true
            )
            / tE_true
        )

        h0_u0_delta = (
            h0_u0_fit
            - u0_true
        )

        h0_rho_log10_ratio = (
            np.log10(
                h0_rho_fit
                / rho_true
            )
            if (
                h0_rho_fit > 0
                and rho_true > 0
            )
            else np.nan
        )

        # --------------------------------------------
        # Save row
        # --------------------------------------------

        result = {
            "status":
                "success",

            "catalog_row":
                row,

            "json_path":
                str(
                    json_path
                ),

            "h5_path":
                str(
                    h5_path
                ),

            "tE_bin":
                row_entry[
                    "tE_bin"
                ],

            "Xpi_bin":
                row_entry[
                    "Xpi_bin"
                ],

            "n_photometry_points":
                int(
                    data[
                        "n_photometry_points"
                    ]
                ),

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

            "chi2_h0_fit":
                chi2_h0_fit,

            "chi2_h1_fit":
                chi2_h1_fit,

            "T_fit":
                T_fit,

            "chi2_h1_true":
                chi2_h1_true,

            "chi2_h0_shared_truth":
                chi2_h0_shared_truth,

            "D_truth":
                D_truth,

            "A_H0":
                A_H0,

            "G_H1_truth":
                G_H1_truth,

            "I_H1_truth":
                I_H1_truth,

            "decomposition_prediction":
                decomposition_prediction,

            "decomposition_closure":
                decomposition_closure,

            "chi2_h1_final_direct":
                chi2_h1_final_direct,

            "chi2_h0_final_direct":
                chi2_h0_final_direct,

            "chi2_h0_embedded_h1":
                chi2_h0_embedded_h1,

            "chi2_h0_truth_embedded_h1":
                chi2_h0_truth_embedded_h1,

            "h1_direct_minus_stored":
                h1_direct_minus_stored,

            "h0_direct_minus_stored":
                h0_direct_minus_stored,

            "embedded_h0_minus_stored":
                embedded_h0_minus_stored,

            "truth_h0_nested_difference":
                truth_h0_nested_difference,

            "objective_consistent":
                objective_consistent,

            "h0_t0_fit":
                h0_t0_fit,

            "h0_u0_fit":
                h0_u0_fit,

            "h0_tE_fit":
                h0_tE_fit,

            "h0_rho_fit":
                h0_rho_fit,

            "h0_t0_delta_d":
                h0_t0_fit
                - t0_true,

            "h0_u0_delta":
                h0_u0_delta,

            "h0_tE_frac_bias":
                h0_tE_frac_bias,

            "h0_rho_log10_ratio":
                h0_rho_log10_ratio,
        }

        records.append(
            result
        )

        checkpoint(
            records
        )

        print(
            f"tE={tE_true:.3f} d  "
            f"Xpi={X_pi:.5g}  "
            f"T={T_fit:.6g}"
        )

        print(
            f"D_truth={D_truth:.6g}  "
            f"A_H0={A_H0:.6g}  "
            f"I_H1={I_H1_truth:.6g}"
        )

        print(
            f"closure="
            f"{decomposition_closure:+.3e}  "
            f"objective_consistent="
            f"{objective_consistent}"
        )

        print(
            f"H0 tE bias="
            f"{h0_tE_frac_bias:+.4f}"
        )

    except Exception as exc:

        print(
            f"FAILED row={row}: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        records.append(
            {
                "status":
                    "failure",

                "catalog_row":
                    row,

                "json_path":
                    str(
                        json_path
                    ),

                "h5_path":
                    str(
                        h5_path
                    ),

                "tE_bin":
                    row_entry.get(
                        "tE_bin"
                    ),

                "Xpi_bin":
                    row_entry.get(
                        "Xpi_bin"
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

        checkpoint(
            records
        )


# ============================================================
# Final table
# ============================================================

result_df = pd.DataFrame(
    records
)

result_df.to_csv(
    RESULT_PATH,
    index=False,
)

good = result_df[
    result_df[
        "status"
    ]
    == "success"
].copy()

if good.empty:
    raise RuntimeError(
        "No successful evaluations."
    )


# ============================================================
# Summary by physical cell
# ============================================================

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

        median_D_truth=(
            "D_truth",
            "median",
        ),

        median_A_H0=(
            "A_H0",
            "median",
        ),

        median_I_H1=(
            "I_H1_truth",
            "median",
        ),

        median_h0_tE_frac_bias=(
            "h0_tE_frac_bias",
            "median",
        ),

        median_abs_h0_tE_frac_bias=(
            "h0_tE_frac_bias",
            lambda x:
                float(
                    np.nanmedian(
                        np.abs(
                            x
                        )
                    )
                ),
        ),

        max_abs_closure=(
            "decomposition_closure",
            lambda x:
                float(
                    np.nanmax(
                        np.abs(
                            x
                        )
                    )
                ),
        ),

        all_objectives_consistent=(
            "objective_consistent",
            "all",
        ),
    )
    .reset_index()
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# Global diagnostic text
# ============================================================

n_success = len(
    good
)

n_fail = int(
    (
        result_df[
            "status"
        ]
        != "success"
    ).sum()
)

n_consistent = int(
    good[
        "objective_consistent"
    ].sum()
)

n_h1_worse_than_truth = int(
    (
        good[
            "G_H1_truth"
        ]
        > 1.0e-6
    ).sum()
)

n_h0_worse_than_shared_truth = int(
    (
        good[
            "A_H0"
        ]
        < -1.0e-6
    ).sum()
)

text = f"""
H1 TURNOVER FIXED-POINT TRUTH DIAGNOSTIC
================================================================================

Run:
  {RUN}

Sample:
  selected rows          = {len(sample)}
  successful             = {n_success}
  failed                 = {n_fail}
  objective consistent   = {n_consistent}/{n_success}

No nonlinear refits were run by this diagnostic.

Definitions:
  X_pi    = |piE_true| (tE_true / 365.25 d)^2

  D_truth = chi2_H0(shared truth) - chi2_H1(truth)

  A_H0    = chi2_H0(shared truth) - chi2_H0(fit)

  I_H1    = chi2_H1(truth) - chi2_H1(fit)

  G_H1    = chi2_H1(fit) - chi2_H1(truth)
          = -I_H1

  T_fit   = chi2_H0(fit) - chi2_H1(fit)

Identity:
  T_fit = D_truth - A_H0 + I_H1

Optimization sanity:
  H1 stored worse than truth by >1e-6:
    {n_h1_worse_than_truth}/{n_success}

  H0 stored worse than shared-truth H0 by >1e-6:
    {n_h0_worse_than_shared_truth}/{n_success}

Global medians:
  median(T_fit)       = {good["T_fit"].median():.12g}
  median(D_truth)     = {good["D_truth"].median():.12g}
  median(A_H0)        = {good["A_H0"].median():.12g}
  median(I_H1)        = {good["I_H1_truth"].median():.12g}

  median H0 tE frac bias =
    {good["h0_tE_frac_bias"].median():.12g}

  median |H0 tE frac bias| =
    {np.nanmedian(np.abs(good["h0_tE_frac_bias"])):.12g}

Numerical closure:
  max |T_fit - (D_truth - A_H0 + I_H1)| =
    {np.nanmax(np.abs(good["decomposition_closure"])):.12e}

Outputs:
  sample:
    {SAMPLE_PATH}

  event table:
    {RESULT_PATH}

  cell summary:
    {SUMMARY_PATH}
================================================================================
""".strip()

TEXT_PATH.write_text(
    text + "\n"
)

print()
print(text)

print()
print("=" * 88)
print("CELL SUMMARY")
print("=" * 88)

print(
    summary.to_string(
        index=False
    )
)
