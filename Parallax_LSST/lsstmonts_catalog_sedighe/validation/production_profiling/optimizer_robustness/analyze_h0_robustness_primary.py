#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent

P = (
    ROOT
    / "results"
    / "h1_h0_robustness_full3336.parquet"
)

TCRIT = 13.15982011480088

df = pd.read_parquet(P)

assert len(df) == 3336
assert (df["status"] == "success").all()

# ------------------------------------------------------------
# Representative random subset ONLY
# ------------------------------------------------------------

r = df[df["in_random"]].copy()

assert len(r) == 1500

# Exact algebraic checks
assert np.allclose(
    r["T_aug"],
    r["T_prod"] - r["delta_h0"],
    atol=1e-7,
    rtol=0,
)

assert (r["delta_h0"] >= -1e-10).all()

det_prod = r["T_prod"] > TCRIT
det_aug = r["T_aug"] > TCRIT

cross_down = det_prod & ~det_aug
cross_up = ~det_prod & det_aug

assert cross_up.sum() == 0

N = len(r)
n_prod = int(det_prod.sum())
n_aug = int(det_aug.sum())
n_cross = int(cross_down.sum())

p_prod = n_prod / N
p_aug = n_aug / N
drop = n_cross / N

ci_drop = binomtest(
    n_cross,
    N,
).proportion_ci(
    confidence_level=0.95,
    method="exact",
)

print("=" * 80)
print("H0 OPTIMIZER ROBUSTNESS — REPRESENTATIVE RANDOM SUBSET")
print("=" * 80)

print(f"N                         = {N}")
print(f"Tcrit                     = {TCRIT:.12f}")

print()
print("DETECTION / POWER")
print(f"N_detect production       = {n_prod}")
print(f"N_detect H0-augmented     = {n_aug}")
print(f"N downward crossings      = {n_cross}")
print(f"N upward crossings        = {int(cross_up.sum())}")

print()
print(f"power production          = {p_prod:.8f}")
print(f"power H0-augmented        = {p_aug:.8f}")
print(f"absolute power drop       = {p_prod-p_aug:.8f}")
print(f"relative power drop       = {(p_prod-p_aug)/p_prod:.8f}")
print(f"crossing fraction         = {drop:.8f}")
print(
    "95% exact CI drop        = "
    f"[{ci_drop.low:.8f}, {ci_drop.high:.8f}]"
)

print()
print("DELTA_H0 DISTRIBUTION")

q = r["delta_h0"].quantile(
    [0, .25, .50, .75, .90, .95, .99, 1]
)

print(q.to_string())

print()

rows = []

for x in [
    0.0,
    0.1,
    1.0,
    5.0,
    10.0,
    TCRIT,
    50.0,
    100.0,
    1000.0,
]:
    n = int((r["delta_h0"] > x).sum())
    f = n / N

    rows.append({
        "threshold": x,
        "N": n,
        "fraction": f,
    })

    print(
        f"delta_h0 > {x:12.5f} : "
        f"{n:4d}/{N} = {f:.8f}"
    )

print()
print("MORPHOLOGY AVAILABILITY")

if "morphology_available" in r.columns:
    print(
        r["morphology_available"]
        .value_counts(dropna=False)
        .to_string()
    )
else:
    print(
        "morphology_available is NaN for original rows; "
        "infer from n_downstream_fits:"
    )
    print(
        r["n_downstream_fits"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

print()
print("WINNING ROBUST STRATEGIES")

print(
    r["robust_winner_label"]
    .value_counts(dropna=False)
    .to_string()
)

print()
print("DOWNWARD CROSSINGS")

cross = r.loc[
    cross_down,
    [
        "catalog_row",
        "tE_true",
        "X_pi",
        "T_prod",
        "T_aug",
        "delta_h0",
        "robust_winner_label",
        "robust_tE",
    ],
].sort_values(
    "delta_h0",
    ascending=False,
)

if len(cross):
    print(cross.to_string(index=False))
else:
    print("none")

print()
print("TOP 20 RANDOM DELTA_H0")

top = (
    r.sort_values(
        "delta_h0",
        ascending=False,
    )
    [
        [
            "catalog_row",
            "tE_true",
            "X_pi",
            "T_prod",
            "T_aug",
            "delta_h0",
            "robust_winner_label",
            "robust_tE",
        ]
    ]
    .head(20)
)

print(top.to_string(index=False))

# ------------------------------------------------------------
# Save compact outputs
# ------------------------------------------------------------

summary = pd.DataFrame(
    [{
        "N_random": N,
        "Tcrit": TCRIT,
        "N_detect_prod": n_prod,
        "N_detect_aug": n_aug,
        "N_cross_down": n_cross,
        "power_prod": p_prod,
        "power_aug": p_aug,
        "absolute_power_drop": p_prod - p_aug,
        "relative_power_drop": (
            (p_prod - p_aug) / p_prod
        ),
        "drop_ci95_low": ci_drop.low,
        "drop_ci95_high": ci_drop.high,
        "delta_h0_median": r["delta_h0"].median(),
        "delta_h0_p90": r["delta_h0"].quantile(.90),
        "delta_h0_p95": r["delta_h0"].quantile(.95),
        "delta_h0_p99": r["delta_h0"].quantile(.99),
        "delta_h0_max": r["delta_h0"].max(),
    }]
)

summary.to_csv(
    ROOT / "results" / "h0_robustness_random_primary_summary.csv",
    index=False,
)

cross.to_csv(
    ROOT / "results" / "h0_robustness_random_crossings.csv",
    index=False,
)

pd.DataFrame(rows).to_csv(
    ROOT / "results" / "h0_robustness_random_delta_thresholds.csv",
    index=False,
)

print()
print("=" * 80)
print("SAVED")
print("=" * 80)
print(
    ROOT
    / "results"
    / "h0_robustness_random_primary_summary.csv"
)
print(
    ROOT
    / "results"
    / "h0_robustness_random_crossings.csv"
)
