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
    "gate1_oracle_bounds.csv"
)

events = pd.read_csv(PER_EVENT)

records = []

for _, ev in events.iterrows():

    row = int(ev["catalog_row"])
    envelope = float(ev["chi2_envelope"])

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

    rec = {
        "catalog_row": row,
        "mode": best["mode"],
        "label": best["label"],
        "chi2": float(best["chi2"]),
        "delta_from_recorded_envelope":
            float(best["chi2"]) - envelope,
    }

    for par in [
        "t0",
        "u0",
        "tE",
        "rho",
    ]:
        rec[par] = float(best[par])

    records.append(rec)


out = pd.DataFrame(records)

# ============================================================
# Distances to the artificial shared H0 bounds
# ============================================================

out["u0_abs_fraction"] = (
    np.abs(out["u0"]) / 10.0
)

out["tE_upper_fraction"] = (
    out["tE"] / 500000.0
)

# rho is best inspected logarithmically because its domain spans
# eight decades: 1e-7 .. 10.
out["log10_rho"] = np.log10(
    out["rho"]
)

out["rho_decades_from_lower"] = (
    out["log10_rho"] - (-7.0)
)

out["rho_decades_from_upper"] = (
    1.0 - out["log10_rho"]
)

out.to_csv(
    OUT,
    index=False,
)

print()
print("ORACLE PARAMETER RANGES")
print("=" * 100)

for par in [
    "u0",
    "tE",
    "rho",
]:

    x = out[par]

    print(
        f"{par:>5s}: "
        f"min={x.min():.10g} "
        f"p01={x.quantile(0.01):.10g} "
        f"median={x.median():.10g} "
        f"p99={x.quantile(0.99):.10g} "
        f"max={x.max():.10g}"
    )


print()
print("BOUND PROXIMITY")
print("=" * 100)

print(
    "N |u0| > 0.90*10 =",
    int(
        (
            np.abs(out["u0"])
            > 9.0
        ).sum()
    )
)

print(
    "N |u0| > 0.99*10 =",
    int(
        (
            np.abs(out["u0"])
            > 9.9
        ).sum()
    )
)

print(
    "N tE > 0.90*500000 =",
    int(
        (
            out["tE"]
            > 450000.0
        ).sum()
    )
)

print(
    "N tE > 0.99*500000 =",
    int(
        (
            out["tE"]
            > 495000.0
        ).sum()
    )
)

print(
    "N rho within 0.1 dex of lower bound =",
    int(
        (
            out["rho_decades_from_lower"]
            < 0.1
        ).sum()
    )
)

print(
    "N rho within 0.1 dex of upper bound =",
    int(
        (
            out["rho_decades_from_upper"]
            < 0.1
        ).sum()
    )
)


print()
print("CLOSEST TO tE UPPER BOUND")
print("=" * 100)

print(
    out.sort_values(
        "tE",
        ascending=False,
    )[
        [
            "catalog_row",
            "mode",
            "label",
            "chi2",
            "u0",
            "tE",
            "rho",
            "tE_upper_fraction",
        ]
    ]
    .head(15)
    .to_string(
        index=False,
        float_format=lambda z: f"{z:.10g}",
    )
)


print()
print("CLOSEST TO rho LOWER BOUND")
print("=" * 100)

print(
    out.sort_values(
        "rho",
        ascending=True,
    )[
        [
            "catalog_row",
            "mode",
            "label",
            "chi2",
            "u0",
            "tE",
            "rho",
            "rho_decades_from_lower",
        ]
    ]
    .head(15)
    .to_string(
        index=False,
        float_format=lambda z: f"{z:.10g}",
    )
)


print()
print("CLOSEST TO |u0| BOUND")
print("=" * 100)

print(
    out.assign(
        abs_u0=np.abs(out["u0"])
    )
    .sort_values(
        "abs_u0",
        ascending=False,
    )[
        [
            "catalog_row",
            "mode",
            "label",
            "chi2",
            "u0",
            "tE",
            "rho",
            "u0_abs_fraction",
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
