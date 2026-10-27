#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

ROBUST = (
    ROOT
    / "results"
    / "h1_h0_robustness_full3336.parquet"
)

SAMPLE = (
    ROOT
    / "h1_h0_robustness_sample_frozen.csv"
)

OUT = (
    ROOT
    / "h1_random1500_manifest.csv"
)

RUNROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/"
    "hidden_parallax/runs/LSSTMONTS_LRT_TWO_FIT_V2"
)

MAIN = (
    RUNROOT
    / "lrt_two_fit_v2_h1_full_milliways_20260920"
)

REPAIRS = [
    RUNROOT / "lrt_two_fit_v2_h1_repair_63798_65000_20260921",
    RUNROOT / "lrt_two_fit_v2_h1_repair_244132_245000_20260921",
]


def production_json(row):
    row = int(row)

    lo = (row // 5000) * 5000
    hi = min(lo + 5000, 966000)

    p = (
        MAIN
        / f"rows_{lo}_{hi}"
        / "events"
        / f"Event_{row}.json"
    )

    if p.is_file():
        return p

    # Only expected for repaired pieces.
    for root in REPAIRS:
        matches = list(
            root.rglob(f"Event_{row}.json")
        )

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise RuntimeError(
                f"row={row}: multiple repair JSONs: {matches}"
            )

    raise FileNotFoundError(
        f"row={row}: production JSON not found"
    )


rob = pd.read_parquet(ROBUST)
sample = pd.read_csv(SAMPLE)

r = (
    rob[rob["in_random"]]
    .copy()
    .sort_values("catalog_row")
    .reset_index(drop=True)
)

assert len(r) == 1500
assert r["catalog_row"].nunique() == 1500

s = sample[
    sample["catalog_row"].isin(
        r["catalog_row"]
    )
][
    [
        "catalog_row",
        "h5_path",
        "tE_true",
        "X_pi",
    ]
].copy()

r = r.merge(
    s,
    on="catalog_row",
    how="left",
    validate="one_to_one",
    suffixes=("", "_sample"),
)

required_robust = [
    "robust_t0",
    "robust_u0",
    "robust_tE",
    "robust_rho",
]

missing = [
    c for c in required_robust
    if c not in r.columns
]

if missing:
    raise RuntimeError(
        f"missing robust parameter columns: {missing}"
    )


records = []

for _, x in r.iterrows():

    row = int(x["catalog_row"])

    p = production_json(row)

    with open(p) as f:
        e = json.load(f)

    assert int(e["catalog_row"]) == row
    assert e["policy_name"] == "lrt_two_fit_policy_v2"
    assert e["generating_model"] == "H1"

    f0 = e["final_h0"]
    f1 = e["final_h1"]

    prod_chi0 = float(e["chi2_h0"])
    prod_chi1 = float(e["chi2_h1"])
    prod_T = float(e["delta_chi2_lrt"])

    # Strong provenance checks.
    if abs(
        prod_chi0
        - float(x["chi2_h0_prod"])
    ) > 1e-7:
        raise RuntimeError(
            f"row={row}: H0 chi2 mismatch"
        )

    if abs(
        prod_T
        - float(x["T_prod"])
    ) > 1e-7:
        raise RuntimeError(
            f"row={row}: T mismatch"
        )

    if abs(
        prod_chi1
        - (
            float(x["chi2_h0_prod"])
            - float(x["T_prod"])
        )
    ) > 1e-7:
        raise RuntimeError(
            f"row={row}: H1 chi2 mismatch"
        )

    # Exact production H0 vector.
    if all(
        k in f0
        for k in ["t0", "u0", "tE", "rho"]
    ):
        prod_par = {
            "t0": float(f0["t0"]),
            "u0": float(f0["u0"]),
            "tE": float(f0["tE"]),
            "rho": float(f0["rho"]),
        }
    else:
        b = list(
            map(float, f0["best_model"][:4])
        )

        prod_par = dict(
            zip(
                ["t0", "u0", "tE", "rho"],
                b,
            )
        )

    robust_chi0 = float(
        x["chi2_h0_robust16"]
    )

    # If robust16 truly improves H0, use its vector.
    # Otherwise retain the exact stored production solution.
    if robust_chi0 < prod_chi0:

        aug_source = "robust16"

        aug_chi0 = robust_chi0

        aug_par = {
            "t0": float(x["robust_t0"]),
            "u0": float(x["robust_u0"]),
            "tE": float(x["robust_tE"]),
            "rho": float(x["robust_rho"]),
        }

    else:

        aug_source = "production"

        aug_chi0 = prod_chi0

        aug_par = prod_par

    if abs(
        aug_chi0
        - float(x["chi2_h0_aug"])
    ) > 1e-7:
        raise RuntimeError(
            f"row={row}: augmented H0 mismatch"
        )

    records.append({
        "catalog_row": row,
        "h5_path": x["h5_path"],

        "production_json": str(p),

        "tE_true": float(x["tE_true"]),
        "X_pi": float(x["X_pi"]),

        "chi2_h0_prod": prod_chi0,
        "chi2_h1_prod": prod_chi1,
        "T_prod": prod_T,

        "chi2_h0_robust16": robust_chi0,
        "chi2_h0_aug": aug_chi0,
        "T_h0only": (
            aug_chi0
            - prod_chi1
        ),

        "h0_aug_source": aug_source,

        "h0_aug_t0": aug_par["t0"],
        "h0_aug_u0": aug_par["u0"],
        "h0_aug_tE": aug_par["tE"],
        "h0_aug_rho": aug_par["rho"],

        "h0_prod_t0": prod_par["t0"],
        "h0_prod_u0": prod_par["u0"],
        "h0_prod_tE": prod_par["tE"],
        "h0_prod_rho": prod_par["rho"],
    })


out = pd.DataFrame(records)

assert len(out) == 1500
assert out["catalog_row"].nunique() == 1500

assert np.isfinite(
    out[
        [
            "chi2_h0_prod",
            "chi2_h1_prod",
            "chi2_h0_aug",
            "h0_aug_t0",
            "h0_aug_u0",
            "h0_aug_tE",
            "h0_aug_rho",
        ]
    ].to_numpy(dtype=float)
).all()

out.to_csv(
    OUT,
    index=False,
)

print("=" * 80)
print("H1 RANDOM-1500 MANIFEST")
print("=" * 80)

print("N =", len(out))

print()
print("H0 augmented source:")
print(
    out["h0_aug_source"]
    .value_counts()
    .to_string()
)

print()
print(
    "max |T_h0only - stored H0-only T| =",
    np.max(
        np.abs(
            out["T_h0only"].to_numpy()
            -
            r["T_aug"].to_numpy()
        )
    ),
)

print()
print(
    out[
        [
            "catalog_row",
            "h0_aug_source",
            "chi2_h0_prod",
            "chi2_h0_robust16",
            "chi2_h0_aug",
            "T_prod",
            "T_h0only",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print()
print("saved =", OUT)
