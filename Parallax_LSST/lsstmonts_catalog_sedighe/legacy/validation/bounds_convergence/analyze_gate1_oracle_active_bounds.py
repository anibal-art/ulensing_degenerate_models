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

ROOT = Path(
    "~/Downloads/hidden_parallax/"
    "production_validation/gate1_coordinates"
).expanduser()

PER_EVENT = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate1_extreme100_coordinate_per_event.csv"
)

OUT = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate1_oracle_active_bounds.csv"
)

events = pd.read_csv(PER_EVENT)

records = []

for _, ev in events.iterrows():

    row = int(ev["catalog_row"])

    candidates = []

    for mode in MODES:

        p = (
            ROOT
            / mode
            / "refits"
            / "bounds_convergence"
            / "te500000_h0only"
            / "extreme100"
            / str(row)
            / "all_refits.csv"
        )

        df = pd.read_csv(p)

        df = df[
            (df["hypothesis"] == "H0")
            & (df["status"] == "success")
        ].copy()

        df["chi2"] = pd.to_numeric(
            df["chi2"],
            errors="coerce",
        )

        df = df[np.isfinite(df["chi2"])]

        df["mode"] = mode

        candidates.append(df)

    x = pd.concat(
        candidates,
        ignore_index=True,
    )

    best = x.loc[
        x["chi2"].idxmin()
    ]

    mask = ast.literal_eval(
        str(best["optimizer_active_mask"])
    )

    if len(mask) != 4:
        raise RuntimeError(
            f"row={row}: expected H0 mask length 4, got {mask}"
        )

    records.append(
        {
            "catalog_row": row,
            "mode": best["mode"],
            "label": best["label"],
            "chi2": float(best["chi2"]),
            "t0": float(best["t0"]),
            "u0": float(best["u0"]),
            "tE": float(best["tE"]),
            "rho": float(best["rho"]),
            "optimizer_success":
                bool(best["optimizer_success"]),
            "optimizer_optimality":
                float(best["optimizer_optimality"]),
            "active_mask":
                str(mask),
            "t0_active": int(mask[0]),
            "u0_active": int(mask[1]),
            "tE_active": int(mask[2]),
            "rho_active": int(mask[3]),
        }
    )


out = pd.DataFrame(records)

out.to_csv(
    OUT,
    index=False,
)


print()
print("ORACLE ACTIVE-BOUND AUDIT")
print("=" * 100)

print("N oracle winners =", len(out))

print(
    "N optimizer_success=False =",
    int((~out["optimizer_success"]).sum()),
)

for par in [
    "t0",
    "u0",
    "tE",
    "rho",
]:

    col = f"{par}_active"

    print()
    print(
        f"{par}: "
        f"N active={int((out[col] != 0).sum())}, "
        f"N lower={int((out[col] == -1).sum())}, "
        f"N upper={int((out[col] == 1).sum())}"
    )


print()
print("FULL ACTIVE MASK COUNTS")
print("=" * 100)

print(
    out["active_mask"]
    .value_counts()
    .to_string()
)


active = out[
    (
        out[
            [
                "t0_active",
                "u0_active",
                "tE_active",
                "rho_active",
            ]
        ]
        != 0
    )
    .any(axis=1)
]

print()
print("ORACLE WINNERS WITH ANY ACTIVE BOUND")
print("=" * 100)

if len(active) == 0:
    print("NONE")
else:
    print(
        active[
            [
                "catalog_row",
                "mode",
                "label",
                "chi2",
                "t0",
                "u0",
                "tE",
                "rho",
                "active_mask",
                "optimizer_optimality",
            ]
        ].to_string(
            index=False,
            float_format=lambda z: f"{z:.10g}",
        )
    )


print()
print("OPTIMALITY")
print("=" * 100)

opt = pd.to_numeric(
    out["optimizer_optimality"],
    errors="coerce",
)

print(
    "median =",
    opt.median(),
)

print(
    "p90    =",
    opt.quantile(0.90),
)

print(
    "p95    =",
    opt.quantile(0.95),
)

print(
    "p99    =",
    opt.quantile(0.99),
)

print(
    "max    =",
    opt.max(),
)

print()
print("Largest optimality values:")
print(
    out.sort_values(
        "optimizer_optimality",
        ascending=False,
    )[
        [
            "catalog_row",
            "mode",
            "label",
            "chi2",
            "optimizer_optimality",
            "active_mask",
        ]
    ]
    .head(15)
    .to_string(
        index=False,
        float_format=lambda z: f"{z:.10g}",
    )
)

print()
print("saved:", OUT)
