#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe"
)

RESULTS = (
    PROJECT
    / "validation/bounds_convergence/results"
)

ADAPTIVE = pd.read_csv(
    RESULTS / "gate1_final_adaptive_t0.csv"
)

BASE_ROOT = Path(
    "~/Downloads/hidden_parallax/"
    "production_validation/gate1B_final18_t0base"
).expanduser()

RESCUE_ROOT = Path(
    "~/Downloads/hidden_parallax/"
    "production_validation/gate1B_final18"
).expanduser()

OUT = (
    PROJECT
    / "validation/bounds_convergence/data/"
    "gate2_h0_anchor_manifest_extreme100.csv"
)

records = []

for _, r in ADAPTIVE.iterrows():

    row = int(r["catalog_row"])
    mode = str(r["final_mode"])
    label = str(r["final_label"])
    target_chi2 = float(r["final_chi2"])

    rescue = bool(r["rescue_triggered"])

    root = (
        RESCUE_ROOT
        if rescue
        else BASE_ROOT
    )

    margin = (
        0.25
        if rescue
        else 0.0
    )

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

    x = df[
        (df["hypothesis"] == "H0")
        & (df["status"] == "success")
        & (df["label"].astype(str) == label)
    ].copy()

    if len(x) == 0:
        raise RuntimeError(
            f"row={row}: cannot find "
            f"mode={mode}, label={label}"
        )

    x["chi2_diff"] = np.abs(
        pd.to_numeric(
            x["chi2"],
            errors="raise",
        )
        - target_chi2
    )

    best = x.loc[
        x["chi2_diff"].idxmin()
    ]

    if float(best["chi2_diff"]) > 1e-5:
        raise RuntimeError(
            f"row={row}: H0 anchor mismatch: "
            f"{best['chi2']} vs {target_chi2}"
        )

    records.append(
        {
            "catalog_row": row,
            "sample": str(best["sample"]),
            "chi2_h0": float(best["chi2"]),
            "t0": float(best["t0"]),
            "u0": float(best["u0"]),
            "tE": float(best["tE"]),
            "rho": float(best["rho"]),
            "source_mode": mode,
            "source_label": label,
            "t0_margin_factor": margin,
            "rescue_triggered": rescue,
            "optimizer_active_mask":
                str(best["optimizer_active_mask"]),
        }
    )


out = pd.DataFrame(records)

if len(out) != 100:
    raise RuntimeError(
        f"Expected 100 H0 anchors, got {len(out)}"
    )

if out["catalog_row"].nunique() != 100:
    raise RuntimeError(
        "catalog_row is not unique"
    )

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUT,
    index=False,
)

print(out.to_string(index=False))
print()
print("N =", len(out))
print(
    "N rescues =",
    int(out["rescue_triggered"].sum())
)
print("saved:", OUT)
