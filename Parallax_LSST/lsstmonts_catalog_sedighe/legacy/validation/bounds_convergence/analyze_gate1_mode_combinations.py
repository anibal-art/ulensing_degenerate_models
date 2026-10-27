#!/usr/bin/env python3

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


RESULTS = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate1_extreme100_coordinate_per_event.csv"
)

MODES = [
    "physical",
    "log_te",
    "log_rho",
    "log_te_rho",
]

TOL = 0.1

df = pd.read_csv(RESULTS)

records = []

for n in range(1, len(MODES) + 1):

    for subset in combinations(MODES, n):

        chi_cols = [
            f"chi2_{m}"
            for m in subset
        ]

        chi = df[chi_cols].min(
            axis=1,
            skipna=True,
        )

        delta = (
            chi
            - df["chi2_envelope"]
        )

        wall_cols = [
            f"wall_{m}"
            for m in subset
        ]

        total_wall = (
            df[wall_cols]
            .sum(axis=1)
            .sum()
        )

        records.append(
            {
                "n_modes": len(subset),
                "modes": "+".join(subset),
                "n_delta_gt_0p1":
                    int((delta > 0.1).sum()),
                "n_delta_gt_1":
                    int((delta > 1.0).sum()),
                "n_delta_gt_10":
                    int((delta > 10.0).sum()),
                "max_delta":
                    float(delta.max()),
                "p90_delta":
                    float(delta.quantile(0.90)),
                "p95_delta":
                    float(delta.quantile(0.95)),
                "p99_delta":
                    float(delta.quantile(0.99)),
                "approx_total_wall_s":
                    float(total_wall),
            }
        )

summary = pd.DataFrame(records)

summary = summary.sort_values(
    [
        "n_delta_gt_0p1",
        "n_modes",
        "approx_total_wall_s",
        "max_delta",
    ]
)

print()
print("MODE COMBINATIONS VS 4-MODE ENVELOPE")
print("=" * 130)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.7g}",
    )
)


# ============================================================
# Particularly interesting candidate combinations
# ============================================================

for subset in [
    ("physical", "log_rho"),
    ("physical", "log_te"),
    ("log_rho", "log_te"),
    ("physical", "log_rho", "log_te"),
]:

    chi = df[
        [f"chi2_{m}" for m in subset]
    ].min(axis=1)

    delta = (
        chi
        - df["chi2_envelope"]
    )

    bad = df.loc[
        delta > TOL,
        [
            "catalog_row",
            "winner",
            "chi2_envelope",
        ],
    ].copy()

    bad["delta_subset"] = (
        delta[delta > TOL]
    )

    print()
    print(
        "NOT COVERED WITH "
        + "+".join(subset)
    )
    print("=" * 100)

    if len(bad) == 0:
        print("NONE")
    else:
        print(
            bad.sort_values(
                "delta_subset",
                ascending=False,
            ).to_string(
                index=False,
                float_format=lambda x: f"{x:.8g}",
            )
        )


# ============================================================
# Runtime tails
# ============================================================

print()
print("RUNTIME TAILS")
print("=" * 100)

for mode in MODES:

    col = f"wall_{mode}"

    x = pd.to_numeric(
        df[col],
        errors="coerce",
    )

    print()
    print(mode)
    print(
        "median =",
        x.median(),
        "p90 =",
        x.quantile(0.90),
        "p95 =",
        x.quantile(0.95),
        "p99 =",
        x.quantile(0.99),
        "max =",
        x.max(),
        "sum =",
        x.sum(),
    )

    top = df.loc[
        x.sort_values(
            ascending=False
        ).head(5).index,
        [
            "catalog_row",
            "winner",
            col,
        ],
    ]

    print(
        top.to_string(
            index=False
        )
    )


out = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate1_mode_combination_summary.csv"
)

summary.to_csv(
    out,
    index=False,
)

print()
print("saved:", out)
