#!/usr/bin/env python3

# ============================================================
# NEW BLOCK: imports
# ============================================================

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr


# ============================================================
# NEW BLOCK: CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--rows-csv",
    required=True,
)

parser.add_argument(
    "--reference-json",
    required=True,
)

parser.add_argument(
    "--out-prefix",
    required=True,
)

parser.add_argument(
    "--detail-row",
    type=int,
    default=779307,
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
from pyLIMA.fits import DE_fit  # noqa: E402


PARAMETER_ORDER = [
    "t0",
    "u0",
    "tE",
    "rho",
    "piEN",
    "piEE",
]


# ============================================================
# NEW BLOCK: exact profiled-H1 objective
# ============================================================

def build_h1_objective(meta):

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


# ============================================================
# NEW BLOCK: load frozen inputs
# ============================================================

manifest = pd.read_csv(
    args.rows_csv
)

manifest_by_row = {
    int(r["catalog_row"]): r
    for _, r in manifest.iterrows()
}


reference = json.loads(
    Path(
        args.reference_json
    ).read_text()
)


# ============================================================
# NEW BLOCK: evaluate every stored initial seed
# ============================================================

seed_records = []
event_records = []
topk_records = []

TOP_K = [
    1,
    2,
    4,
    8,
    16,
    34,
]


for event_index, event in enumerate(
    reference["events"],
    start=1,
):

    row = int(
        event["catalog_row"]
    )

    print()
    print("=" * 90)
    print(
        f"EVENT "
        f"{event_index}/"
        f"{len(reference['events'])} "
        f"row={row}"
    )
    print("=" * 90)

    if row not in manifest_by_row:
        raise RuntimeError(
            f"row={row} missing from manifest"
        )

    h5_path = str(
        manifest_by_row[row][
            "h5_path"
        ]
    )

    meta = load_new_case(
        h5_path,
        core,
    )

    objective = build_h1_objective(
        meta
    )

    event_seed_rows = []

    for fit in event["fits"]:

        if fit.get("chi2") is None:
            continue

        initial = fit["initial"]

        x0 = np.array(
            [
                float(initial[name])
                for name in PARAMETER_ORDER
            ],
            dtype=float,
        )

        initial_chi2 = float(
            objective.objective_function(
                x0
            )
        )

        final_chi2 = float(
            fit["chi2"]
        )

        event_seed_rows.append(
            {
                "catalog_row":
                    row,

                "strategy":
                    fit["label"],

                "initial_chi2":
                    initial_chi2,

                "final_chi2":
                    final_chi2,

                "initial_t0":
                    float(initial["t0"]),

                "initial_u0":
                    float(initial["u0"]),

                "initial_tE":
                    float(initial["tE"]),

                "initial_rho":
                    float(initial["rho"]),

                "initial_piEN":
                    float(initial["piEN"]),

                "initial_piEE":
                    float(initial["piEE"]),
            }
        )

    df = pd.DataFrame(
        event_seed_rows
    )

    if len(df) != 34:
        raise RuntimeError(
            f"row={row}: expected 34 finite "
            f"reference seeds, got {len(df)}"
        )

    oracle_final = float(
        df["final_chi2"].min()
    )

    df["final_gap"] = (
        df["final_chi2"]
        - oracle_final
    )

    df["initial_rank"] = (
        df["initial_chi2"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    df["final_rank"] = (
        df["final_chi2"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    winner = (
        df.sort_values(
            "final_chi2"
        )
        .iloc[0]
    )

    rho_s, p_s = spearmanr(
        df["initial_chi2"],
        df["final_chi2"],
    )

    event_records.append(
        {
            "catalog_row":
                row,

            "spearman_r":
                float(rho_s),

            "spearman_p":
                float(p_s),

            "oracle_final_chi2":
                oracle_final,

            "oracle_strategy":
                winner["strategy"],

            "oracle_initial_chi2":
                float(
                    winner[
                        "initial_chi2"
                    ]
                ),

            "oracle_initial_rank":
                int(
                    winner[
                        "initial_rank"
                    ]
                ),
        }
    )

    seed_records.extend(
        df.to_dict(
            orient="records"
        )
    )

    print(
        "Spearman(initial, final) =",
        rho_s,
    )

    print(
        "oracle strategy          =",
        winner["strategy"],
    )

    print(
        "oracle initial rank      =",
        int(
            winner["initial_rank"]
        ),
        "/ 34",
    )

    print(
        "oracle initial chi2      =",
        float(
            winner["initial_chi2"]
        ),
    )

    print(
        "oracle final chi2        =",
        oracle_final,
    )


    # ========================================================
    # NEW BLOCK: top-k initial-seed selection experiment
    # ========================================================

    ordered_initial = (
        df.sort_values(
            "initial_chi2"
        )
        .reset_index(
            drop=True
        )
    )

    for k in TOP_K:

        selected = (
            ordered_initial
            .head(k)
        )

        best_selected = (
            selected.sort_values(
                "final_chi2"
            )
            .iloc[0]
        )

        regret = (
            float(
                best_selected[
                    "final_chi2"
                ]
            )
            - oracle_final
        )

        topk_records.append(
            {
                "catalog_row":
                    row,

                "k":
                    k,

                "selected_best_strategy":
                    best_selected[
                        "strategy"
                    ],

                "selected_best_final_chi2":
                    float(
                        best_selected[
                            "final_chi2"
                        ]
                    ),

                "oracle_final_chi2":
                    oracle_final,

                "regret":
                    regret,

                "oracle_seed_in_topk":
                    bool(
                        int(
                            winner[
                                "initial_rank"
                            ]
                        )
                        <= k
                    ),
            }
        )


# ============================================================
# NEW BLOCK: aggregate tables
# ============================================================

seed_df = pd.DataFrame(
    seed_records
)

event_df = pd.DataFrame(
    event_records
)

topk_df = pd.DataFrame(
    topk_records
)


# ============================================================
# NEW BLOCK: detailed diagnostic for requested event
# ============================================================

detail = (
    seed_df[
        seed_df["catalog_row"]
        == int(args.detail_row)
    ]
    .sort_values(
        "initial_chi2"
    )
    .copy()
)

print()
print("=" * 100)
print(
    f"DETAIL ROW = "
    f"{args.detail_row}"
)
print("=" * 100)

print()
print(
    detail[
        [
            "strategy",
            "initial_chi2",
            "initial_rank",
            "final_chi2",
            "final_rank",
            "final_gap",
            "initial_u0",
            "initial_rho",
            "initial_piEN",
            "initial_piEE",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# NEW BLOCK: global seed-ranking diagnostics
# ============================================================

print()
print("=" * 100)
print("ORACLE-SEED INITIAL RANK")
print("=" * 100)

print(
    event_df[
        [
            "catalog_row",
            "spearman_r",
            "oracle_strategy",
            "oracle_initial_rank",
        ]
    ]
    .sort_values(
        "oracle_initial_rank",
        ascending=False,
    )
    .to_string(
        index=False
    )
)

print()
print(
    "median oracle initial rank =",
    event_df[
        "oracle_initial_rank"
    ].median(),
)

print(
    "max oracle initial rank    =",
    event_df[
        "oracle_initial_rank"
    ].max(),
)


# ============================================================
# NEW BLOCK: top-k performance summary
# ============================================================

print()
print("=" * 100)
print("TOP-K BY INITIAL CHI2 -> FINAL-BASIN PERFORMANCE")
print("=" * 100)

summary_rows = []

for k in TOP_K:

    x = topk_df[
        topk_df["k"] == k
    ]

    g = x["regret"]

    summary_rows.append(
        {
            "k":
                k,

            "median_regret":
                g.median(),

            "p90_regret":
                g.quantile(0.90),

            "p95_regret":
                g.quantile(0.95),

            "max_regret":
                g.max(),

            "N_regret_gt_0.1":
                int(
                    (g > 0.1).sum()
                ),

            "N_regret_gt_0.5":
                int(
                    (g > 0.5).sum()
                ),

            "N_regret_gt_1":
                int(
                    (g > 1.0).sum()
                ),

            "N_oracle_seed_in_topk":
                int(
                    x[
                        "oracle_seed_in_topk"
                    ].sum()
                ),
        }
    )

summary_df = pd.DataFrame(
    summary_rows
)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# NEW BLOCK: save outputs
# ============================================================

prefix = Path(
    args.out_prefix
)

seed_path = Path(
    str(prefix)
    + "_seed_level.csv"
)

event_path = Path(
    str(prefix)
    + "_event_level.csv"
)

topk_path = Path(
    str(prefix)
    + "_topk.csv"
)

summary_path = Path(
    str(prefix)
    + "_summary.csv"
)

seed_df.to_csv(
    seed_path,
    index=False,
)

event_df.to_csv(
    event_path,
    index=False,
)

topk_df.to_csv(
    topk_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)

print()
print("SAVED:")
print(seed_path)
print(event_path)
print(topk_path)
print(summary_path)
