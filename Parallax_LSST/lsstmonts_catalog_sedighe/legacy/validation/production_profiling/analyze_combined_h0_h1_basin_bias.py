#!/usr/bin/env python3

# ============================================================
# NEW BLOCK: imports
# ============================================================

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# NEW BLOCK: CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--h0-audit-csv",
    required=True,
)

parser.add_argument(
    "--h1-reference-json",
    required=True,
)

parser.add_argument(
    "--out-csv",
    required=True,
)

parser.add_argument(
    "--out-json",
    required=True,
)

args = parser.parse_args()


# ============================================================
# NEW BLOCK: load H0 audit
# ============================================================

h0 = pd.read_csv(
    args.h0_audit_csv
)

required_h0 = {
    "catalog_row",
    "chi2_h0_policy",
    "chi2_h0_audit",
    "h0_basin_gap",
    "D_reference_h0_audit",
}

missing = (
    required_h0
    - set(h0.columns)
)

if missing:
    raise RuntimeError(
        f"H0 audit missing columns: "
        f"{sorted(missing)}"
    )


# ============================================================
# NEW BLOCK: load H1 reference
# ============================================================

payload = json.loads(
    Path(
        args.h1_reference_json
    ).read_text()
)

rows = []

for event in payload["events"]:

    rows.append(
        {
            "catalog_row":
                int(
                    event[
                        "catalog_row"
                    ]
                ),

            "chi2_h0_policy_ref":
                float(
                    event[
                        "chi2_h0_policy"
                    ]
                ),

            "chi2_h1_policy":
                float(
                    event[
                        "chi2_h1_policy"
                    ]
                ),

            "chi2_h1_reference_ref":
                float(
                    event[
                        "chi2_h1_reference"
                    ]
                ),

            "G_H1":
                float(
                    event[
                        "h1_basin_gap"
                    ]
                ),

            "D_policy":
                float(
                    event[
                        "delta_lrt_policy"
                    ]
                ),

            "D_H1ref_policyH0":
                float(
                    event[
                        "delta_lrt_reference"
                    ]
                ),
        }
    )

h1 = pd.DataFrame(
    rows
)


# ============================================================
# NEW BLOCK: merge
# ============================================================

df = h0.merge(
    h1,
    on="catalog_row",
    how="inner",
    validate="one_to_one",
)

if len(df) != len(h0):
    raise RuntimeError(
        f"Merge mismatch: "
        f"H0={len(h0)}, merged={len(df)}"
    )


# ============================================================
# NEW BLOCK: parity checks
# ============================================================

df[
    "h0_policy_parity_diff"
] = (
    df["chi2_h0_policy"]
    - df["chi2_h0_policy_ref"]
)

max_h0_parity = float(
    np.max(
        np.abs(
            df[
                "h0_policy_parity_diff"
            ]
        )
    )
)

if max_h0_parity > 1e-6:
    raise RuntimeError(
        "H0-policy mismatch between "
        "audit and H1-reference files. "
        f"max diff={max_h0_parity}"
    )


# ============================================================
# NEW BLOCK: H1-reference parity between input files
# ============================================================

df[
    "h1_reference_parity_diff"
] = (
    df["chi2_h1_reference"]
    - df["chi2_h1_reference_ref"]
)

max_h1_parity = float(
    np.max(
        np.abs(
            df[
                "h1_reference_parity_diff"
            ]
        )
    )
)

if max_h1_parity > 1e-6:
    raise RuntimeError(
        "H1-reference mismatch between "
        "audit and reference files. "
        f"max diff={max_h1_parity}"
    )


# ============================================================
# NEW BLOCK: combined basin corrections
# ============================================================

df[
    "G_H0"
] = df[
    "h0_basin_gap"
]

# Positive Delta_D:
# correcting both sides increases D.
#
# Negative Delta_D:
# correcting both sides decreases D.

df[
    "Delta_D"
] = (
    df["G_H1"]
    - df["G_H0"]
)

df[
    "D_audit"
] = (
    df["D_policy"]
    + df["Delta_D"]
)


# ============================================================
# NEW BLOCK: direct algebraic cross-check
# ============================================================

df[
    "D_audit_direct"
] = (
    df["chi2_h0_audit"]
    - df["chi2_h1_reference"]
)

df[
    "D_audit_formula_diff"
] = (
    df["D_audit"]
    - df["D_audit_direct"]
)

max_formula_diff = float(
    np.max(
        np.abs(
            df[
                "D_audit_formula_diff"
            ]
        )
    )
)

if max_formula_diff > 1e-6:
    raise RuntimeError(
        "Combined D algebra parity failed. "
        f"max diff={max_formula_diff}"
    )


# Cross-check against the value already produced by H0 audit.
df[
    "D_h0audit_parity_diff"
] = (
    df["D_audit"]
    - df["D_reference_h0_audit"]
)

max_existing_diff = float(
    np.max(
        np.abs(
            df[
                "D_h0audit_parity_diff"
            ]
        )
    )
)

if max_existing_diff > 1e-6:
    raise RuntimeError(
        "D_audit differs from H0 audit "
        "stored reference value. "
        f"max diff={max_existing_diff}"
    )


# ============================================================
# NEW BLOCK: interpretation class
# ============================================================

tol = 1e-6


def classify(delta):

    if delta > tol:
        return "H1_failure_dominates"

    if delta < -tol:
        return "H0_failure_dominates"

    return "balanced"


df[
    "net_bias_class"
] = [
    classify(x)
    for x in df[
        "Delta_D"
    ]
]


# ============================================================
# NEW BLOCK: nesting diagnostic
# ============================================================

# For exact global minima of nested models:
#
# D >= 0.
#
# Negative D_audit would therefore imply that the current
# H1 reference is still insufficient relative to the improved
# H0 solution.

df[
    "nesting_violation"
] = (
    df["D_audit"]
    < -1e-6
)


# ============================================================
# NEW BLOCK: sorted diagnostic table
# ============================================================

df[
    "abs_Delta_D"
] = np.abs(
    df["Delta_D"]
)

show = [
    "catalog_row",
    "D_policy",
    "G_H0",
    "G_H1",
    "Delta_D",
    "D_audit",
    "net_bias_class",
    "nesting_violation",
]

ordered = df.sort_values(
    "abs_Delta_D",
    ascending=False,
)


print()
print("#" * 110)
print("COMBINED H0/H1 BASIN EFFECT ON D")
print("#" * 110)

print(
    ordered[
        show
    ].to_string(
        index=False
    )
)


# ============================================================
# NEW BLOCK: summary statistics
# ============================================================

delta = df[
    "Delta_D"
].to_numpy(
    dtype=float
)

abs_delta = np.abs(
    delta
)

g0 = df[
    "G_H0"
].to_numpy(
    dtype=float
)

g1 = df[
    "G_H1"
].to_numpy(
    dtype=float
)


print()
print("#" * 110)
print("SUMMARY")
print("#" * 110)

print(
    "N =",
    len(df),
)

print()
print("H0 gaps:")
print(
    "  median =",
    np.median(g0),
)
print(
    "  p90    =",
    np.quantile(
        g0,
        0.90,
    ),
)
print(
    "  max    =",
    np.max(g0),
)

print()
print("H1 gaps:")
print(
    "  median =",
    np.median(g1),
)
print(
    "  p90    =",
    np.quantile(
        g1,
        0.90,
    ),
)
print(
    "  max    =",
    np.max(g1),
)

print()
print("Net Delta_D = G_H1 - G_H0:")
print(
    "  min       =",
    np.min(delta),
)
print(
    "  median    =",
    np.median(delta),
)
print(
    "  mean      =",
    np.mean(delta),
)
print(
    "  p10       =",
    np.quantile(
        delta,
        0.10,
    ),
)
print(
    "  p90       =",
    np.quantile(
        delta,
        0.90,
    ),
)
print(
    "  max       =",
    np.max(delta),
)

print()
print("|Delta_D|:")
print(
    "  median    =",
    np.median(
        abs_delta
    ),
)
print(
    "  p90       =",
    np.quantile(
        abs_delta,
        0.90,
    ),
)
print(
    "  p95       =",
    np.quantile(
        abs_delta,
        0.95,
    ),
)
print(
    "  max       =",
    np.max(
        abs_delta
    ),
)


print()
print(
    "H1 failure dominates:",
    int(
        np.sum(
            delta > tol
        )
    ),
    "/",
    len(df),
)

print(
    "H0 failure dominates:",
    int(
        np.sum(
            delta < -tol
        )
    ),
    "/",
    len(df),
)

print(
    "Balanced:",
    int(
        np.sum(
            np.abs(delta)
            <= tol
        )
    ),
    "/",
    len(df),
)


for threshold in [
    0.01,
    0.1,
    0.5,
    1.0,
    2.0,
]:

    print(
        f"|Delta_D| > "
        f"{threshold:g}:",
        int(
            np.sum(
                abs_delta
                > threshold
            )
        ),
        "/",
        len(df),
    )


# ============================================================
# NEW BLOCK: nesting check
# ============================================================

violations = df[
    df[
        "nesting_violation"
    ]
]

print()
print("#" * 110)
print("NESTING CHECK")
print("#" * 110)

print(
    "D_audit < -1e-6:",
    len(
        violations
    ),
    "/",
    len(df),
)

if len(
    violations
):

    print()
    print(
        violations[
            [
                "catalog_row",
                "D_policy",
                "G_H0",
                "G_H1",
                "D_audit",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# NEW BLOCK: biggest policy distortions
# ============================================================

print()
print("#" * 110)
print("TOP |Delta_D| EVENTS")
print("#" * 110)

print(
    ordered[
        show
    ].head(
        10
    ).to_string(
        index=False
    )
)


# ============================================================
# NEW BLOCK: save CSV
# ============================================================

out_csv = Path(
    args.out_csv
)

out_csv.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.sort_values(
    "catalog_row"
).to_csv(
    out_csv,
    index=False,
)


# ============================================================
# NEW BLOCK: save JSON summary
# ============================================================

summary = {
    "analysis_name":
        "combined_h0_h1_basin_bias_v1",

    "n_events":
        int(
            len(df)
        ),

    "identity":
        (
            "Delta_D = "
            "G_H1 - G_H0"
        ),

    "G_H0_median":
        float(
            np.median(
                g0
            )
        ),

    "G_H0_p90":
        float(
            np.quantile(
                g0,
                0.90,
            )
        ),

    "G_H0_max":
        float(
            np.max(
                g0
            )
        ),

    "G_H1_median":
        float(
            np.median(
                g1
            )
        ),

    "G_H1_p90":
        float(
            np.quantile(
                g1,
                0.90,
            )
        ),

    "G_H1_max":
        float(
            np.max(
                g1
            )
        ),

    "Delta_D_min":
        float(
            np.min(
                delta
            )
        ),

    "Delta_D_median":
        float(
            np.median(
                delta
            )
        ),

    "Delta_D_mean":
        float(
            np.mean(
                delta
            )
        ),

    "Delta_D_max":
        float(
            np.max(
                delta
            )
        ),

    "abs_Delta_D_median":
        float(
            np.median(
                abs_delta
            )
        ),

    "abs_Delta_D_p90":
        float(
            np.quantile(
                abs_delta,
                0.90,
            )
        ),

    "abs_Delta_D_p95":
        float(
            np.quantile(
                abs_delta,
                0.95,
            )
        ),

    "abs_Delta_D_max":
        float(
            np.max(
                abs_delta
            )
        ),

    "n_H1_failure_dominates":
        int(
            np.sum(
                delta > tol
            )
        ),

    "n_H0_failure_dominates":
        int(
            np.sum(
                delta < -tol
            )
        ),

    "n_balanced":
        int(
            np.sum(
                np.abs(delta)
                <= tol
            )
        ),

    "n_nesting_violations":
        int(
            len(
                violations
            )
        ),

    "events":
        df.to_dict(
            orient="records"
        ),
}

out_json = Path(
    args.out_json
)

out_json.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out_json.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)

print()
print(
    "CSV  =",
    out_csv,
)

print(
    "JSON =",
    out_json,
)
