#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import (
    Bounds,
    LinearConstraint,
    milp,
)
from scipy.sparse import lil_matrix


P = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate1_all_start_fits.csv"
)

OUT = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate1_h0_coverage_curve.csv"
)

TOL = 0.1


df = pd.read_csv(P)

df["catalog_row"] = pd.to_numeric(
    df["catalog_row"],
    errors="raise",
).astype(int)

df["start_slot"] = pd.to_numeric(
    df["start_slot"],
    errors="raise",
).astype(int)

df["chi2"] = pd.to_numeric(
    df["chi2"],
    errors="raise",
)

df["strategy"] = (
    df["mode"].astype(str)
    + ":"
    + df["start_slot"].astype(str)
)


mat = df.pivot_table(
    index="catalog_row",
    columns="strategy",
    values="chi2",
    aggfunc="min",
)

if mat.isna().any().any():
    raise RuntimeError(
        "Incomplete event x strategy matrix."
    )


rows = list(mat.index)
strategies = list(mat.columns)

Achi = mat.to_numpy()

oracle = Achi.min(axis=1)

delta = (
    Achi
    - oracle[:, None]
)

cover = (
    delta <= TOL
).astype(float)


n_events = len(rows)
n_strategies = len(strategies)


print("N events     =", n_events)
print("N strategies =", n_strategies)
print("tolerance    =", TOL)


records = []


for k in range(1, 19):

    # Variables:
    #
    # x_s = strategy s selected
    # y_e = event e covered
    #
    # maximize sum(y_e)
    #
    # equivalent to minimizing -sum(y_e).

    nvar = (
        n_strategies
        + n_events
    )

    c = np.zeros(nvar)

    c[
        n_strategies:
    ] = -1.0


    # Constraints:
    #
    # y_e <= sum_s cover[e,s] x_s
    #
    # y_e - sum_s cover[e,s] x_s <= 0
    #
    # and
    #
    # sum_s x_s <= k

    M = lil_matrix(
        (
            n_events + 1,
            nvar,
        ),
        dtype=float,
    )

    for e in range(n_events):

        M[
            e,
            :n_strategies,
        ] = -cover[e]

        M[
            e,
            n_strategies + e,
        ] = 1.0


    M[
        n_events,
        :n_strategies,
    ] = 1.0


    lower = np.full(
        n_events + 1,
        -np.inf,
    )

    upper = np.zeros(
        n_events + 1,
    )

    upper[-1] = k


    constraint = LinearConstraint(
        M.tocsr(),
        lower,
        upper,
    )


    result = milp(
        c=c,
        integrality=np.ones(
            nvar,
            dtype=int,
        ),
        bounds=Bounds(
            np.zeros(nvar),
            np.ones(nvar),
        ),
        constraints=constraint,
        options={
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"MILP failed for k={k}: "
            f"{result.message}"
        )


    x = result.x[
        :n_strategies
    ]

    selected_idx = np.where(
        x > 0.5
    )[0]

    selected = [
        strategies[i]
        for i in selected_idx
    ]


    candidate = Achi[
        :,
        selected_idx,
    ].min(axis=1)

    d = (
        candidate
        - oracle
    )


    rec = {
        "k":
            k,

        "n_selected":
            len(selected),

        "strategies":
            " + ".join(selected),

        "n_delta_gt_0p1":
            int(
                (d > 0.1).sum()
            ),

        "n_delta_gt_1":
            int(
                (d > 1.0).sum()
            ),

        "n_delta_gt_10":
            int(
                (d > 10.0).sum()
            ),

        "max_delta":
            float(d.max()),

        "p50_delta":
            float(
                np.quantile(
                    d,
                    0.50,
                )
            ),

        "p90_delta":
            float(
                np.quantile(
                    d,
                    0.90,
                )
            ),

        "p95_delta":
            float(
                np.quantile(
                    d,
                    0.95,
                )
            ),

        "p99_delta":
            float(
                np.quantile(
                    d,
                    0.99,
                )
            ),
    }

    records.append(rec)

    print(
        f"k={k:2d} "
        f"selected={len(selected_idx):2d} "
        f">0.1={(d > 0.1).sum():3d} "
        f">1={(d > 1).sum():3d} "
        f">10={(d > 10).sum():3d} "
        f"max={d.max():.6g}"
    )

    print(
        "   ",
        " + ".join(selected),
    )


out = pd.DataFrame(records)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUT,
    index=False,
)


print()
print("=" * 120)
print("H0 COVERAGE CURVE")
print("=" * 120)

print(
    out[
        [
            "k",
            "n_delta_gt_0p1",
            "n_delta_gt_1",
            "n_delta_gt_10",
            "max_delta",
            "strategies",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8g}",
    )
)

print()
print("saved:", OUT)
