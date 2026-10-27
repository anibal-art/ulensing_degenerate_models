#!/usr/bin/env python3

from pathlib import Path

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
    "production_validation/gate1B_final18"
).expanduser()

ORACLE_FILE = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate1_final_minimum_start_set_per_event.csv"
)

OUTDIR = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results"
)

TOL = 0.1


oracle = pd.read_csv(ORACLE_FILE)

oracle["catalog_row"] = pd.to_numeric(
    oracle["catalog_row"],
    errors="raise",
).astype(int)

records = []

for _, r in oracle.iterrows():

    row = int(r["catalog_row"])
    chi2_oracle = float(r["chi2_oracle_final"])

    rec = {
        "catalog_row": row,
        "chi2_oracle": chi2_oracle,
    }

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

        if not p.exists():
            rec[f"chi2_{mode}"] = np.nan
            rec[f"n_success_{mode}"] = 0
            continue

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

        rec[f"n_success_{mode}"] = len(ok)

        if len(ok):
            rec[f"chi2_{mode}"] = float(
                ok["chi2"].min()
            )
        else:
            rec[f"chi2_{mode}"] = np.nan

        wall = ROOT / mode / f"wall_{row}.txt"

        try:
            rec[f"wall_{mode}"] = float(
                wall.read_text().strip()
            )
        except Exception:
            rec[f"wall_{mode}"] = np.nan

    records.append(rec)


out = pd.DataFrame(records)

chi_cols = [
    f"chi2_{m}"
    for m in MODES
]

out["chi2_final18"] = (
    out[chi_cols]
    .min(axis=1, skipna=True)
)

out["delta"] = (
    out["chi2_final18"]
    - out["chi2_oracle"]
)

out["winner"] = (
    out[chi_cols]
    .idxmin(axis=1)
    .str.replace(
        "chi2_",
        "",
        regex=False,
    )
)

print()
print("GATE 1B FINAL18")
print("=" * 110)

print("N events =", len(out))

print(
    "N missing final chi2 =",
    int(out["chi2_final18"].isna().sum())
)

print(
    "N delta > 0.1 =",
    int((out["delta"] > TOL).sum())
)

print(
    "N new minima delta < -0.1 =",
    int((out["delta"] < -TOL).sum())
)

print(
    "max delta =",
    out["delta"].max()
)

print(
    "min delta =",
    out["delta"].min()
)

print(
    "p50 delta =",
    out["delta"].quantile(0.50)
)

print(
    "p90 delta =",
    out["delta"].quantile(0.90)
)

print(
    "p95 delta =",
    out["delta"].quantile(0.95)
)

print(
    "p99 delta =",
    out["delta"].quantile(0.99)
)


bad = out[
    out["delta"] > TOL
].sort_values(
    "delta",
    ascending=False,
)

print()
print("FAILURES: delta > 0.1")
print("=" * 110)

if len(bad) == 0:
    print("NONE")
else:
    print(
        bad[
            [
                "catalog_row",
                "chi2_oracle",
                "chi2_final18",
                "delta",
                "winner",
            ]
        ].to_string(
            index=False,
            float_format=lambda z: f"{z:.10g}",
        )
    )


better = out[
    out["delta"] < -TOL
].sort_values("delta")

print()
print("NEW MINIMA: delta < -0.1")
print("=" * 110)

if len(better) == 0:
    print("NONE")
else:
    print(
        better[
            [
                "catalog_row",
                "chi2_oracle",
                "chi2_final18",
                "delta",
                "winner",
            ]
        ].to_string(
            index=False,
            float_format=lambda z: f"{z:.10g}",
        )
    )


print()
print("WINNER COUNTS")
print("=" * 110)

print(
    out["winner"]
    .value_counts()
    .to_string()
)


print()
print("RUNTIME")
print("=" * 110)

for mode in MODES:

    x = pd.to_numeric(
        out[f"wall_{mode}"],
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

    top = out.loc[
        x.sort_values(
            ascending=False
        ).head(10).index,
        [
            "catalog_row",
            f"wall_{mode}",
            "winner",
            "delta",
        ],
    ]

    print(
        top.to_string(
            index=False,
            float_format=lambda z: f"{z:.8g}",
        )
    )


total_per_event = out[
    [f"wall_{m}" for m in MODES]
].sum(axis=1)

print()
print("FULL 18-FIT COST PER EVENT")
print("=" * 110)

print(
    "median =",
    total_per_event.median()
)

print(
    "p90 =",
    total_per_event.quantile(0.90)
)

print(
    "p95 =",
    total_per_event.quantile(0.95)
)

print(
    "p99 =",
    total_per_event.quantile(0.99)
)

print(
    "max =",
    total_per_event.max()
)

print(
    "sum =",
    total_per_event.sum()
)


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUTDIR
    / "gate1B_final18_per_event.csv",
    index=False,
)

print()
print(
    "saved:",
    OUTDIR
    / "gate1B_final18_per_event.csv"
)
