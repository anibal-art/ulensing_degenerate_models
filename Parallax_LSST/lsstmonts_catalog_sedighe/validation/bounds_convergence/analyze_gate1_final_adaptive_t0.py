#!/usr/bin/env python3

from pathlib import Path
import ast

import numpy as np
import pandas as pd


MODES = [
    "physical",
    "log_te",
    "log_rho",
    "log_te_rho",
]

EXPECTED = {
    "physical": 3,
    "log_te": 6,
    "log_rho": 8,
    "log_te_rho": 1,
}

TOL = 0.1

BASE_ROOT = Path(
    "~/Downloads/hidden_parallax/"
    "production_validation/gate1B_final18_t0base"
).expanduser()

RESCUE_ROOT = Path(
    "~/Downloads/hidden_parallax/"
    "production_validation/gate1B_final18"
).expanduser()

ORACLE_FILE = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate1_final_minimum_start_set_per_event.csv"
)

OUT = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate1_final_adaptive_t0.csv"
)


oracle = pd.read_csv(ORACLE_FILE)

oracle["catalog_row"] = pd.to_numeric(
    oracle["catalog_row"],
    errors="raise",
).astype(int)


def load_best(root, row, mode):

    p = (
        root
        / mode
        / "refits"
        / "bounds_convergence"
        / "te500000_h0only"
        / "extreme100"
        / str(row)
        / "all_refits.csv"
    )

    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)

    ok = df[
        (df["hypothesis"] == "H0")
        & (df["status"] == "success")
    ].copy()

    ok["chi2"] = pd.to_numeric(
        ok["chi2"],
        errors="coerce",
    )

    ok = ok[np.isfinite(ok["chi2"])]

    if len(ok) != EXPECTED[mode]:
        raise RuntimeError(
            f"row={row}, mode={mode}: "
            f"expected {EXPECTED[mode]} successful fits, "
            f"found {len(ok)}"
        )

    best = ok.loc[
        ok["chi2"].idxmin()
    ].copy()

    mask = ast.literal_eval(
        str(best["optimizer_active_mask"])
    )

    if len(mask) != 4:
        raise RuntimeError(
            f"row={row}, mode={mode}: bad mask {mask}"
        )

    return {
        "mode": mode,
        "chi2": float(best["chi2"]),
        "label": str(best["label"]),
        "t0": float(best["t0"]),
        "u0": float(best["u0"]),
        "tE": float(best["tE"]),
        "rho": float(best["rho"]),
        "active_mask": str(mask),
        "t0_active": int(mask[0]),
        "u0_active": int(mask[1]),
        "tE_active": int(mask[2]),
        "rho_active": int(mask[3]),
    }


records = []

for _, r in oracle.iterrows():

    row = int(r["catalog_row"])
    chi2_oracle = float(
        r["chi2_oracle_final"]
    )

    base_candidates = [
        load_best(
            BASE_ROOT,
            row,
            mode,
        )
        for mode in MODES
    ]

    base = min(
        base_candidates,
        key=lambda x: x["chi2"],
    )

    rescue_triggered = (
        base["t0_active"] != 0
    )

    rescue = None

    if rescue_triggered:

        rescue_candidates = [
            load_best(
                RESCUE_ROOT,
                row,
                mode,
            )
            for mode in MODES
        ]

        rescue = min(
            rescue_candidates,
            key=lambda x: x["chi2"],
        )

        final = (
            rescue
            if rescue["chi2"] < base["chi2"]
            else base
        )

    else:

        final = base

    records.append(
        {
            "catalog_row": row,
            "chi2_oracle": chi2_oracle,

            "base_chi2":
                base["chi2"],
            "base_mode":
                base["mode"],
            "base_label":
                base["label"],
            "base_active_mask":
                base["active_mask"],
            "base_t0_active":
                base["t0_active"],

            "rescue_triggered":
                rescue_triggered,

            "rescue_chi2":
                (
                    rescue["chi2"]
                    if rescue is not None
                    else np.nan
                ),

            "rescue_mode":
                (
                    rescue["mode"]
                    if rescue is not None
                    else None
                ),

            "final_chi2":
                final["chi2"],
            "final_mode":
                final["mode"],
            "final_label":
                final["label"],
            "final_active_mask":
                final["active_mask"],

            "delta_final":
                final["chi2"]
                - chi2_oracle,
        }
    )


out = pd.DataFrame(records)

out.to_csv(
    OUT,
    index=False,
)


print()
print("GATE 1 — FINAL ADAPTIVE t0 STRATEGY")
print("=" * 120)

print(
    "N events =",
    len(out),
)

print(
    "N t0 rescues triggered =",
    int(
        out[
            "rescue_triggered"
        ].sum()
    ),
)

print(
    "N delta > 0.1 =",
    int(
        (
            out["delta_final"]
            > TOL
        ).sum()
    ),
)

print(
    "N new minima delta < -0.1 =",
    int(
        (
            out["delta_final"]
            < -TOL
        ).sum()
    ),
)

print(
    "max delta =",
    out["delta_final"].max()
)

print(
    "min delta =",
    out["delta_final"].min()
)

print(
    "p50 =",
    out["delta_final"].quantile(0.50)
)

print(
    "p90 =",
    out["delta_final"].quantile(0.90)
)

print(
    "p95 =",
    out["delta_final"].quantile(0.95)
)

print(
    "p99 =",
    out["delta_final"].quantile(0.99)
)


print()
print("RESCUES")
print("=" * 120)

rescues = out[
    out["rescue_triggered"]
]

if len(rescues) == 0:
    print("NONE")
else:
    print(
        rescues[
            [
                "catalog_row",
                "chi2_oracle",
                "base_chi2",
                "base_mode",
                "base_active_mask",
                "rescue_chi2",
                "rescue_mode",
                "final_chi2",
                "delta_final",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.10g}",
        )
    )


print()
print("FAILURES")
print("=" * 120)

bad = out[
    out["delta_final"] > TOL
]

if len(bad) == 0:
    print("NONE")
else:
    print(
        bad[
            [
                "catalog_row",
                "chi2_oracle",
                "final_chi2",
                "delta_final",
                "final_mode",
                "final_label",
                "final_active_mask",
            ]
        ].sort_values(
            "delta_final",
            ascending=False,
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.10g}",
        )
    )


print()
print("FINAL WINNER MODES")
print("=" * 120)

print(
    out["final_mode"]
    .value_counts()
    .to_string()
)


print()
print("FINAL ACTIVE MASKS")
print("=" * 120)

print(
    out["final_active_mask"]
    .value_counts()
    .to_string()
)


print()
print("saved:", OUT)
