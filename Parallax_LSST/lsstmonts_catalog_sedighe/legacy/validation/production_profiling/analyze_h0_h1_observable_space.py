#!/usr/bin/env python3

# ============================================================
# NEW BLOCK: imports
# ============================================================

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    "--out-dir",
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

sys.path.insert(0, str(BA))
sys.path.insert(0, str(BC))
sys.path.insert(0, str(PP))


# ============================================================
# NEW BLOCK: exact production fitting environment
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


# core parses argv at import.
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
# NEW BLOCK: project / pyLIMA imports
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

H1_ORDER = [
    "t0",
    "u0",
    "tE",
    "rho",
    "piEN",
    "piEE",
]


# ============================================================
# NEW BLOCK: utilities
# ============================================================

def find_event(payload, row):

    for event in payload["events"]:

        if int(
            event["catalog_row"]
        ) == int(row):

            return event

    raise KeyError(
        f"row={row} not found"
    )


def nested_dicts(obj, path="root"):
    """
    Yield every nested dictionary with its JSON-like path.
    """

    if isinstance(obj, dict):

        yield path, obj

        for key, value in obj.items():

            yield from nested_dicts(
                value,
                f"{path}.{key}",
            )

    elif isinstance(obj, list):

        for i, value in enumerate(obj):

            yield from nested_dicts(
                value,
                f"{path}[{i}]",
            )


def select_parameter_record(
    obj,
    parameter_names,
    expected_chi2,
    preferred_key=None,
):
    """
    Find the parameter record whose stored chi2 is closest
    to the expected fit chi2.
    """

    if (
        preferred_key is not None
        and isinstance(obj, dict)
        and preferred_key in obj
        and isinstance(
            obj[preferred_key],
            dict,
        )
    ):

        candidate = obj[
            preferred_key
        ]

        if all(
            name in candidate
            for name in parameter_names
        ):

            return (
                f"root.{preferred_key}",
                candidate,
            )

    candidates = []

    for path, record in nested_dicts(obj):

        if not all(
            name in record
            for name in parameter_names
        ):
            continue

        try:
            values = [
                float(
                    record[name]
                )
                for name
                in parameter_names
            ]
        except (
            TypeError,
            ValueError,
        ):
            continue

        if not np.all(
            np.isfinite(values)
        ):
            continue

        chi2 = record.get(
            "chi2",
            np.nan,
        )

        try:
            chi2 = float(chi2)
        except (
            TypeError,
            ValueError,
        ):
            chi2 = np.nan

        distance = (
            abs(
                chi2
                - expected_chi2
            )
            if np.isfinite(chi2)
            else np.inf
        )

        candidates.append(
            (
                distance,
                path,
                record,
            )
        )

    if not candidates:

        raise RuntimeError(
            "Could not locate parameter record "
            f"for {parameter_names}"
        )

    candidates.sort(
        key=lambda x: x[0]
    )

    _, path, record = candidates[0]

    return path, record


def vector_from_record(
    record,
    order,
):

    return np.array(
        [
            float(
                record[name]
            )
            for name in order
        ],
        dtype=float,
    )


def true_parameters_from_h5(
    h5_path,
):

    with h5py.File(
        h5_path,
        "r",
    ) as f:

        attrs = (
            f[
                "pyLIMA_parameters"
            ].attrs
        )

        return {
            "t0":
                float(
                    attrs["t0"]
                ),

            "u0":
                float(
                    attrs["u0"]
                ),

            "tE":
                float(
                    attrs["tE"]
                ),

            "rho":
                float(
                    attrs["rho"]
                ),
        }


def phase_region(
    tau,
):

    a = abs(
        float(tau)
    )

    if a <= 0.25:
        return "core_0_0p25"

    if a <= 0.5:
        return "nearpeak_0p25_0p5"

    if a <= 1.0:
        return "inner_wing_0p5_1"

    if a <= 2.0:
        return "outer_wing_1_2"

    return "baseline_gt2"


# ============================================================
# NEW BLOCK: construct exact fit objective
# ============================================================

def build_objective(
    meta,
    fit_parallax,
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
        str(
            core.EPHEMERIDES
        ),
        meta["curves"]["W149"],
        lsst_lcs,
        ra=
            meta["event_ra"],
        dec=
            meta["event_dec"],
        roman_name=
            "Roman",
    )

    fit_params = (
        fit_lc.initial_params_for_fit_model(
            meta["true_params"],
            "FSPL",
            fit_parallax=
                fit_parallax,
            fit_defaults=None,
        )
    )

    model = (
        fit_lc.build_fit_pyLIMA_model(
            event,
            "FSPL",
            fit_params,
            Origin=None,
            fit_parallax=
                fit_parallax,
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

    expected_order = (
        H1_ORDER
        if fit_parallax
        else H0_ORDER
    )

    if list(
        objective.fit_parameters.keys()
    ) != expected_order:

        raise RuntimeError(
            "Unexpected fit parameter order: "
            f"{list(objective.fit_parameters.keys())}"
        )

    return objective


# ============================================================
# NEW BLOCK: point-level evaluation
# ============================================================

def evaluate_solution(
    objective,
    x,
    label,
):
    """
    Evaluate exactly the same profiled chi2 used by the LRT,
    then recover every point's normalized residual and model
    flux.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    # Same public objective used in the previous parity audit.
    objective_chi2 = float(
        objective.objective_function(
            x
        )
    )

    # Obtain the populated pyLIMA parameter dictionary.
    model_chi2, pyparams = (
        objective.model_chi2(
            x
        )
    )

    model_chi2 = float(
        model_chi2
    )

    residuals, errors = (
        objective.photometric_model_residuals(
            pyparams
        )
    )

    active_telescopes = [
        telescope
        for telescope
        in objective.model.event.telescopes
        if telescope.lightcurve
        is not None
    ]

    if not (
        len(active_telescopes)
        == len(residuals)
        == len(errors)
    ):

        raise RuntimeError(
            "Telescope/residual list mismatch."
        )

    rows = []

    for (
        telescope,
        residual,
        error,
    ) in zip(
        active_telescopes,
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

        time = np.asarray(
            telescope.lightcurve[
                "time"
            ].value,
            dtype=float,
        )

        observed_flux = np.asarray(
            telescope.lightcurve[
                "flux"
            ].value,
            dtype=float,
        )

        if not (
            len(time)
            == len(residual)
            == len(error)
        ):

            raise RuntimeError(
                f"Length mismatch for "
                f"{telescope.name}"
            )

        model_flux = (
            observed_flux
            - residual
        )

        normalized_residual = (
            residual
            / error
        )

        chi2_point = (
            normalized_residual
            ** 2
        )

        for i in range(
            len(time)
        ):

            rows.append(
                {
                    "label":
                        label,

                    "band":
                        str(
                            telescope.name
                        ),

                    "time":
                        float(
                            time[i]
                        ),

                    "observed_flux":
                        float(
                            observed_flux[i]
                        ),

                    "model_flux":
                        float(
                            model_flux[i]
                        ),

                    "err_flux":
                        float(
                            error[i]
                        ),

                    "residual_flux":
                        float(
                            residual[i]
                        ),

                    "residual_sigma":
                        float(
                            normalized_residual[i]
                        ),

                    "chi2_point":
                        float(
                            chi2_point[i]
                        ),
                }
            )

    df = pd.DataFrame(
        rows
    )

    point_sum = float(
        df[
            "chi2_point"
        ].sum()
    )

    return {
        "objective_chi2":
            objective_chi2,

        "model_chi2":
            model_chi2,

        "point_sum":
            point_sum,

        "pyparams":
            pyparams,

        "points":
            df,
    }


def parity_check(
    label,
    evaluated,
    expected,
    atol=1e-5,
):
    """
    Require point-wise reconstruction to agree with the
    stored LRT chi2.
    """

    values = {
        "objective":
            evaluated[
                "objective_chi2"
            ],

        "model_chi2":
            evaluated[
                "model_chi2"
            ],

        "point_sum":
            evaluated[
                "point_sum"
            ],
    }

    print()
    print(
        f"PARITY {label}"
    )

    print(
        "  stored       =",
        expected,
    )

    for name, value in values.items():

        diff = (
            value
            - expected
        )

        print(
            f"  {name:12s} = "
            f"{value:.12f}  "
            f"diff={diff:+.3e}"
        )

        if not np.isclose(
            value,
            expected,
            rtol=0.0,
            atol=atol,
        ):

            raise RuntimeError(
                f"{label}: parity failure "
                f"for {name}: "
                f"{value} vs {expected}"
            )


# ============================================================
# NEW BLOCK: load inputs
# ============================================================

manifest = pd.read_csv(
    args.rows_csv
)

manifest_by_row = {
    int(row["catalog_row"]):
        row
    for _, row
    in manifest.iterrows()
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

out_dir = Path(
    args.out_dir
)

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# NEW BLOCK: analyze events
# ============================================================

event_summary_records = []

for row in args.rows:

    row = int(row)

    print()
    print("#" * 110)
    print(
        f"ROW {row}"
    )
    print("#" * 110)

    if row not in manifest_by_row:

        raise RuntimeError(
            f"row={row} missing from manifest"
        )

    h5_path = str(
        manifest_by_row[
            row
        ][
            "h5_path"
        ]
    )

    meta = load_new_case(
        h5_path,
        core,
    )

    policy_event = find_event(
        policy_payload,
        row,
    )

    stability_event = find_event(
        stability_payload,
        row,
    )

    true_params = (
        true_parameters_from_h5(
            h5_path
        )
    )

    t0_true = float(
        true_params["t0"]
    )

    tE_true = float(
        true_params["tE"]
    )


    # ========================================================
    # NEW BLOCK: locate exact parameter records
    # ========================================================

    chi2_h0_expected = float(
        policy_event[
            "chi2_h0"
        ]
    )

    chi2_policy_expected = float(
        policy_event[
            "chi2_h1"
        ]
    )

    chi2_oracle_expected = float(
        stability_event[
            "best_local_chi2"
        ]
    )


    h0_path, h0_record = (
        select_parameter_record(
            policy_event,
            H0_ORDER,
            chi2_h0_expected,
            preferred_key=
                "final_h0",
        )
    )

    policy_path, policy_record = (
        select_parameter_record(
            policy_event,
            H1_ORDER,
            chi2_policy_expected,
            preferred_key=
                "final_h1",
        )
    )


    finite_oracle_fits = [
        fit
        for fit
        in stability_event[
            "fits"
        ]
        if (
            fit.get("chi2")
            is not None
            and np.isfinite(
                float(
                    fit["chi2"]
                )
            )
        )
    ]

    if not finite_oracle_fits:

        raise RuntimeError(
            f"row={row}: no finite oracle fits"
        )

    oracle_record = min(
        finite_oracle_fits,
        key=lambda fit:
            float(
                fit["chi2"]
            ),
    )

    oracle_path = (
        "stability.fits[best]"
    )


    print()
    print("Selected records:")
    print(
        "  H0       =",
        h0_path,
    )
    print(
        "  policy   =",
        policy_path,
    )
    print(
        "  oracle   =",
        oracle_path,
    )

    x_h0 = vector_from_record(
        h0_record,
        H0_ORDER,
    )

    x_policy = vector_from_record(
        policy_record,
        H1_ORDER,
    )

    x_oracle = vector_from_record(
        oracle_record,
        H1_ORDER,
    )

    print()
    print(
        "H0 params     =",
        dict(
            zip(
                H0_ORDER,
                x_h0,
            )
        ),
    )

    print(
        "H1 policy     =",
        dict(
            zip(
                H1_ORDER,
                x_policy,
            )
        ),
    )

    print(
        "H1 oracle     =",
        dict(
            zip(
                H1_ORDER,
                x_oracle,
            )
        ),
    )


    # ========================================================
    # NEW BLOCK: exact model evaluations
    # ========================================================

    objective_h0 = build_objective(
        meta,
        fit_parallax=False,
    )

    objective_h1 = build_objective(
        meta,
        fit_parallax=True,
    )

    eval_h0 = evaluate_solution(
        objective_h0,
        x_h0,
        "H0",
    )

    eval_policy = evaluate_solution(
        objective_h1,
        x_policy,
        "H1_policy",
    )

    eval_oracle = evaluate_solution(
        objective_h1,
        x_oracle,
        "H1_oracle",
    )


    # ========================================================
    # NEW BLOCK: mandatory parity checks
    # ========================================================

    parity_check(
        "H0",
        eval_h0,
        chi2_h0_expected,
    )

    parity_check(
        "H1_policy",
        eval_policy,
        chi2_policy_expected,
    )

    parity_check(
        "H1_oracle",
        eval_oracle,
        chi2_oracle_expected,
    )


    # ========================================================
    # NEW BLOCK: merge point-level quantities
    # ========================================================

    key = [
        "band",
        "time",
    ]

    h0_points = (
        eval_h0[
            "points"
        ]
        .drop(
            columns=[
                "label",
            ]
        )
        .rename(
            columns={
                "model_flux":
                    "model_flux_h0",

                "residual_flux":
                    "residual_flux_h0",

                "residual_sigma":
                    "residual_sigma_h0",

                "chi2_point":
                    "chi2_h0",
            }
        )
    )

    policy_points = (
        eval_policy[
            "points"
        ][
            key
            + [
                "model_flux",
                "residual_flux",
                "residual_sigma",
                "chi2_point",
            ]
        ]
        .rename(
            columns={
                "model_flux":
                    "model_flux_policy",

                "residual_flux":
                    "residual_flux_policy",

                "residual_sigma":
                    "residual_sigma_policy",

                "chi2_point":
                    "chi2_policy",
            }
        )
    )

    oracle_points = (
        eval_oracle[
            "points"
        ][
            key
            + [
                "model_flux",
                "residual_flux",
                "residual_sigma",
                "chi2_point",
            ]
        ]
        .rename(
            columns={
                "model_flux":
                    "model_flux_oracle",

                "residual_flux":
                    "residual_flux_oracle",

                "residual_sigma":
                    "residual_sigma_oracle",

                "chi2_point":
                    "chi2_oracle",
            }
        )
    )

    df = h0_points.merge(
        policy_points,
        on=key,
        how="inner",
        validate="one_to_one",
    )

    df = df.merge(
        oracle_points,
        on=key,
        how="inner",
        validate="one_to_one",
    )

    df[
        "catalog_row"
    ] = row

    df[
        "tau_true"
    ] = (
        df["time"]
        - t0_true
    ) / tE_true

    df[
        "abs_tau_true"
    ] = np.abs(
        df["tau_true"]
    )

    df[
        "phase_region"
    ] = [
        phase_region(tau)
        for tau
        in df["tau_true"]
    ]

    df[
        "side"
    ] = np.where(
        df["tau_true"] < 0,
        "pre",
        "post",
    )


    # Positive = H1 is better than H0 at that point.
    df[
        "dchi2_h0_minus_policy"
    ] = (
        df["chi2_h0"]
        - df["chi2_policy"]
    )

    df[
        "dchi2_h0_minus_oracle"
    ] = (
        df["chi2_h0"]
        - df["chi2_oracle"]
    )

    # Positive = oracle is better than policy.
    df[
        "dchi2_policy_minus_oracle"
    ] = (
        df["chi2_policy"]
        - df["chi2_oracle"]
    )


    # Direct observable separation between model predictions,
    # in units of the photometric error.
    df[
        "model_shift_oracle_minus_h0_sigma"
    ] = (
        df["model_flux_oracle"]
        - df["model_flux_h0"]
    ) / df["err_flux"]

    df[
        "model_shift_oracle_minus_policy_sigma"
    ] = (
        df["model_flux_oracle"]
        - df["model_flux_policy"]
    ) / df["err_flux"]


    # ========================================================
    # NEW BLOCK: global consistency
    # ========================================================

    D_policy_points = float(
        df[
            "dchi2_h0_minus_policy"
        ].sum()
    )

    D_oracle_points = float(
        df[
            "dchi2_h0_minus_oracle"
        ].sum()
    )

    policy_oracle_gain_points = float(
        df[
            "dchi2_policy_minus_oracle"
        ].sum()
    )

    D_policy_expected = (
        chi2_h0_expected
        - chi2_policy_expected
    )

    D_oracle_expected = (
        chi2_h0_expected
        - chi2_oracle_expected
    )

    policy_oracle_gain_expected = (
        chi2_policy_expected
        - chi2_oracle_expected
    )

    print()
    print("=" * 110)
    print("GLOBAL DELTA-CHI2")
    print("=" * 110)

    print(
        "D policy stored / points =",
        D_policy_expected,
        "/",
        D_policy_points,
    )

    print(
        "D oracle stored / points =",
        D_oracle_expected,
        "/",
        D_oracle_points,
    )

    print(
        "policy -> oracle gain     =",
        policy_oracle_gain_expected,
        "/",
        policy_oracle_gain_points,
    )


    # ========================================================
    # NEW BLOCK: band summary
    # ========================================================

    band_summary = (
        df.groupby(
            "band",
            sort=True,
        )
        .agg(
            N=(
                "time",
                "size",
            ),

            chi2_h0=(
                "chi2_h0",
                "sum",
            ),

            chi2_policy=(
                "chi2_policy",
                "sum",
            ),

            chi2_oracle=(
                "chi2_oracle",
                "sum",
            ),

            dchi2_H0_policy=(
                "dchi2_h0_minus_policy",
                "sum",
            ),

            dchi2_H0_oracle=(
                "dchi2_h0_minus_oracle",
                "sum",
            ),

            gain_policy_oracle=(
                "dchi2_policy_minus_oracle",
                "sum",
            ),

            max_abs_model_shift_H0_oracle_sigma=(
                "model_shift_oracle_minus_h0_sigma",
                lambda x:
                    float(
                        np.max(
                            np.abs(x)
                        )
                    ),
            ),

            max_abs_model_shift_policy_oracle_sigma=(
                "model_shift_oracle_minus_policy_sigma",
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

    print()
    print("=" * 110)
    print("BY BAND")
    print("=" * 110)

    print(
        band_summary.to_string(
            index=False
        )
    )


    # ========================================================
    # NEW BLOCK: phase summary
    # ========================================================

    phase_order = [
        "core_0_0p25",
        "nearpeak_0p25_0p5",
        "inner_wing_0p5_1",
        "outer_wing_1_2",
        "baseline_gt2",
    ]

    phase_summary = (
        df.groupby(
            "phase_region",
            sort=False,
        )
        .agg(
            N=(
                "time",
                "size",
            ),

            dchi2_H0_policy=(
                "dchi2_h0_minus_policy",
                "sum",
            ),

            dchi2_H0_oracle=(
                "dchi2_h0_minus_oracle",
                "sum",
            ),

            gain_policy_oracle=(
                "dchi2_policy_minus_oracle",
                "sum",
            ),

            mean_abs_model_shift_H0_oracle_sigma=(
                "model_shift_oracle_minus_h0_sigma",
                lambda x:
                    float(
                        np.mean(
                            np.abs(x)
                        )
                    ),
            ),

            max_abs_model_shift_H0_oracle_sigma=(
                "model_shift_oracle_minus_h0_sigma",
                lambda x:
                    float(
                        np.max(
                            np.abs(x)
                        )
                    ),
            ),
        )
        .reindex(
            phase_order
        )
        .reset_index()
    )

    print()
    print("=" * 110)
    print("BY TRUE PHASE")
    print("=" * 110)

    print(
        phase_summary.to_string(
            index=False
        )
    )


    # ========================================================
    # NEW BLOCK: pre/post summary
    # ========================================================

    inner = df[
        df["abs_tau_true"]
        <= 2.0
    ]

    side_summary = (
        inner.groupby(
            "side"
        )
        .agg(
            N=(
                "time",
                "size",
            ),

            dchi2_H0_oracle=(
                "dchi2_h0_minus_oracle",
                "sum",
            ),

            gain_policy_oracle=(
                "dchi2_policy_minus_oracle",
                "sum",
            ),
        )
        .reset_index()
    )

    print()
    print("=" * 110)
    print("PRE / POST WITHIN |tau| <= 2")
    print("=" * 110)

    print(
        side_summary.to_string(
            index=False
        )
    )


    # ========================================================
    # NEW BLOCK: most informative individual observations
    # ========================================================

    print()
    print("=" * 110)
    print(
        "TOP 15 POINTS: H0 -> ORACLE"
    )
    print("=" * 110)

    columns_show = [
        "band",
        "time",
        "tau_true",
        "chi2_h0",
        "chi2_oracle",
        "dchi2_h0_minus_oracle",
        "model_shift_oracle_minus_h0_sigma",
    ]

    print(
        df.sort_values(
            "dchi2_h0_minus_oracle",
            ascending=False,
        )
        .head(15)[
            columns_show
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 110)
    print(
        "TOP 15 POINTS: POLICY -> ORACLE"
    )
    print("=" * 110)

    columns_show = [
        "band",
        "time",
        "tau_true",
        "chi2_policy",
        "chi2_oracle",
        "dchi2_policy_minus_oracle",
        "model_shift_oracle_minus_policy_sigma",
    ]

    print(
        df.sort_values(
            "dchi2_policy_minus_oracle",
            ascending=False,
        )
        .head(15)[
            columns_show
        ]
        .to_string(
            index=False
        )
    )


    # ========================================================
    # NEW BLOCK: save numerical products
    # ========================================================

    point_path = (
        out_dir
        / f"observable_space_row{row}_points.csv"
    )

    band_path = (
        out_dir
        / f"observable_space_row{row}_bands.csv"
    )

    phase_path = (
        out_dir
        / f"observable_space_row{row}_phases.csv"
    )

    side_path = (
        out_dir
        / f"observable_space_row{row}_prepost.csv"
    )

    df.to_csv(
        point_path,
        index=False,
    )

    band_summary.to_csv(
        band_path,
        index=False,
    )

    phase_summary.to_csv(
        phase_path,
        index=False,
    )

    side_summary.to_csv(
        side_path,
        index=False,
    )


    # ========================================================
    # NEW BLOCK: plot pointwise delta-chi2
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for band in sorted(
        df["band"].unique()
    ):

        x = df[
            df["band"] == band
        ]

        ax.scatter(
            x["tau_true"],
            x[
                "dchi2_h0_minus_oracle"
            ],
            s=24,
            alpha=0.75,
            label=band,
        )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    for value in [
        -2,
        -1,
        -0.5,
        -0.25,
        0,
        0.25,
        0.5,
        1,
        2,
    ]:

        ax.axvline(
            value,
            linewidth=0.6,
            alpha=0.35,
        )

    ax.set_xlim(
        -3,
        3,
    )

    ax.set_xlabel(
        r"$(t-t_{0,\rm true})/t_{E,\rm true}$"
    )

    ax.set_ylabel(
        r"$\chi^2_{H0,i}-\chi^2_{{\rm oracle},i}$"
    )

    ax.set_title(
        f"row {row}: pointwise H0 → oracle improvement"
    )

    ax.legend(
        title="band",
    )

    fig.tight_layout()

    fig.savefig(
        out_dir
        / f"observable_space_row{row}_dchi2.png",
        dpi=180,
    )

    plt.close(fig)


    # ========================================================
    # NEW BLOCK: standardized residual comparison
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 10),
        sharex=True,
    )

    configurations = [
        (
            axes[0],
            "residual_sigma_h0",
            "H0",
        ),
        (
            axes[1],
            "residual_sigma_policy",
            "H1 policy",
        ),
        (
            axes[2],
            "residual_sigma_oracle",
            "H1 oracle",
        ),
    ]

    for (
        ax,
        column,
        title,
    ) in configurations:

        for band in sorted(
            df["band"].unique()
        ):

            x = df[
                df["band"] == band
            ]

            ax.scatter(
                x["tau_true"],
                x[column],
                s=18,
                alpha=0.7,
                label=band,
            )

        ax.axhline(
            0.0,
            linewidth=1,
        )

        ax.axhline(
            3.0,
            linewidth=0.7,
            linestyle="--",
        )

        ax.axhline(
            -3.0,
            linewidth=0.7,
            linestyle="--",
        )

        ax.set_ylabel(
            "residual / sigma"
        )

        ax.set_title(
            title
        )

        ax.set_xlim(
            -3,
            3,
        )

    axes[-1].set_xlabel(
        r"$(t-t_{0,\rm true})/t_{E,\rm true}$"
    )

    axes[0].legend(
        title="band",
        ncol=3,
    )

    fig.suptitle(
        f"row {row}: standardized residuals"
    )

    fig.tight_layout()

    fig.savefig(
        out_dir
        / f"observable_space_row{row}_residuals.png",
        dpi=180,
    )

    plt.close(fig)


    # ========================================================
    # NEW BLOCK: cumulative localization of delta chi2
    # ========================================================

    sorted_abs = (
        df.sort_values(
            "abs_tau_true"
        )
        .copy()
    )

    sorted_abs[
        "cum_dchi2_H0_oracle"
    ] = sorted_abs[
        "dchi2_h0_minus_oracle"
    ].cumsum()

    sorted_abs[
        "cum_gain_policy_oracle"
    ] = sorted_abs[
        "dchi2_policy_minus_oracle"
    ].cumsum()

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        sorted_abs[
            "abs_tau_true"
        ],
        sorted_abs[
            "cum_dchi2_H0_oracle"
        ],
        label="H0 → oracle",
    )

    ax.plot(
        sorted_abs[
            "abs_tau_true"
        ],
        sorted_abs[
            "cum_gain_policy_oracle"
        ],
        label="policy → oracle",
    )

    for value in [
        0.25,
        0.5,
        1,
        2,
    ]:

        ax.axvline(
            value,
            linewidth=0.7,
            alpha=0.4,
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xlim(
        0,
        min(
            5,
            float(
                sorted_abs[
                    "abs_tau_true"
                ].max()
            ),
        ),
    )

    ax.set_xlabel(
        r"$|t-t_{0,\rm true}|/t_{E,\rm true}$"
    )

    ax.set_ylabel(
        r"cumulative $\Delta\chi^2$"
    )

    ax.set_title(
        f"row {row}: where the fit improvement accumulates"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_dir
        / f"observable_space_row{row}_cumulative.png",
        dpi=180,
    )

    plt.close(fig)


    # ========================================================
    # NEW BLOCK: compact event summary
    # ========================================================

    event_summary_records.append(
        {
            "catalog_row":
                row,

            "chi2_h0":
                chi2_h0_expected,

            "chi2_policy":
                chi2_policy_expected,

            "chi2_oracle":
                chi2_oracle_expected,

            "D_policy":
                D_policy_expected,

            "D_oracle":
                D_oracle_expected,

            "oracle_gain_over_policy":
                policy_oracle_gain_expected,

            "N_points":
                int(
                    len(df)
                ),

            "N_core":
                int(
                    (
                        df[
                            "abs_tau_true"
                        ]
                        <= 0.25
                    ).sum()
                ),

            "N_within_1tE":
                int(
                    (
                        df[
                            "abs_tau_true"
                        ]
                        <= 1
                    ).sum()
                ),

            "N_wings_0p5_2":
                int(
                    (
                        (
                            df[
                                "abs_tau_true"
                            ]
                            > 0.5
                        )
                        &
                        (
                            df[
                                "abs_tau_true"
                            ]
                            <= 2
                        )
                    ).sum()
                ),

            "max_model_shift_H0_oracle_sigma":
                float(
                    np.max(
                        np.abs(
                            df[
                                "model_shift_oracle_minus_h0_sigma"
                            ]
                        )
                    )
                ),

            "max_model_shift_policy_oracle_sigma":
                float(
                    np.max(
                        np.abs(
                            df[
                                "model_shift_oracle_minus_policy_sigma"
                            ]
                        )
                    )
                ),
        }
    )


# ============================================================
# NEW BLOCK: save cross-event summary
# ============================================================

summary_df = pd.DataFrame(
    event_summary_records
)

summary_path = (
    out_dir
    / "observable_space_3events_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)

print()
print("#" * 110)
print("CROSS-EVENT SUMMARY")
print("#" * 110)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print(
    "OUTPUT DIRECTORY =",
    out_dir,
)
