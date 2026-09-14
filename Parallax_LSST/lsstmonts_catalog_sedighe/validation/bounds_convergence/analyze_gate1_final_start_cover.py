#!/usr/bin/env python3

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from scipy.optimize import milp, LinearConstraint, Bounds


# ============================================================
# CONFIG
# ============================================================

MODES = [
    "physical",
    "log_te",
    "log_rho",
    "log_te_rho",
]

N_STARTS = 13
TOL = 0.1

BASE_ROOT = Path(
    "~/Downloads/hidden_parallax/"
    "production_validation/gate1_coordinates"
).expanduser()

T0_ROOT = Path(
    "~/Downloads/hidden_parallax/"
    "production_validation/t0_margin_36103/m0.25"
).expanduser()

PER_EVENT_OLD = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate1_extreme100_coordinate_per_event.csv"
)

OLD_SELECTION = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate1_minimum_start_set.csv"
)

OUTDIR = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results"
)

OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EVENTS
# ============================================================

events = pd.read_csv(PER_EVENT_OLD)

events["catalog_row"] = pd.to_numeric(
    events["catalog_row"],
    errors="raise",
).astype(int)

event_rows = events["catalog_row"].tolist()

if 36103 not in event_rows:
    raise RuntimeError("36103 is not present in extreme100.")


# ============================================================
# LOAD ALL FITS
#
# All events except 36103:
#   original Gate 1 results, t0 margin = 0
#
# Event 36103:
#   replacement results with final t0 margin = 0.25
# ============================================================

records = []

for row in event_rows:

    for mode in MODES:

        if row == 36103:

            p = (
                T0_ROOT
                / mode
                / "refits"
                / "bounds_convergence"
                / "te500000_h0only"
                / "extreme100"
                / str(row)
                / "all_refits.csv"
            )

            source = "t0_margin_0.25"

        else:

            p = (
                BASE_ROOT
                / mode
                / "refits"
                / "bounds_convergence"
                / "te500000_h0only"
                / "extreme100"
                / str(row)
                / "all_refits.csv"
            )

            source = "original_gate1"

        if not p.exists():
            raise FileNotFoundError(p)

        df = pd.read_csv(p)

        df = df[
            (df["hypothesis"] == "H0")
            & (df["status"] == "success")
        ].copy()

        df["chi2"] = pd.to_numeric(
            df["chi2"],
            errors="coerce",
        )

        df = df[
            np.isfinite(df["chi2"])
        ].reset_index(drop=True)

        if len(df) != N_STARTS:
            raise RuntimeError(
                f"row={row}, mode={mode}: "
                f"expected {N_STARTS} successful fits, "
                f"found {len(df)}"
            )

        # Slot is the deterministic position in the start list.
        df["start_slot"] = np.arange(1, N_STARTS + 1)

        for _, r in df.iterrows():

            records.append(
                {
                    "catalog_row": row,
                    "mode": mode,
                    "start_slot":
                        int(r["start_slot"]),
                    "label":
                        str(r["label"]),
                    "chi2":
                        float(r["chi2"]),
                    "source":
                        source,
                }
            )


fits = pd.DataFrame(records)


# ============================================================
# FINAL ORACLE
# ============================================================

oracle = (
    fits.groupby(
        "catalog_row",
        as_index=False,
    )["chi2"]
    .min()
    .rename(
        columns={
            "chi2": "chi2_oracle_final"
        }
    )
)

fits = fits.merge(
    oracle,
    on="catalog_row",
    how="left",
)

fits["delta_final"] = (
    fits["chi2"]
    - fits["chi2_oracle_final"]
)


# ============================================================
# COMPARE OLD VS FINAL ORACLE
# ============================================================

old_oracle = events[
    [
        "catalog_row",
        "chi2_envelope",
    ]
].rename(
    columns={
        "chi2_envelope":
            "chi2_oracle_old"
    }
)

comparison = old_oracle.merge(
    oracle,
    on="catalog_row",
    how="left",
)

comparison["oracle_improvement"] = (
    comparison["chi2_oracle_old"]
    - comparison["chi2_oracle_final"]
)

print()
print("OLD VS FINAL ORACLE")
print("=" * 110)

changed = comparison[
    np.abs(
        comparison["oracle_improvement"]
    ) > 1e-6
]

print(
    changed.to_string(
        index=False,
        float_format=lambda z: f"{z:.10g}",
    )
)

print()
print(
    "N events whose oracle changed > 0.1 =",
    int(
        (
            comparison["oracle_improvement"]
            > TOL
        ).sum()
    )
)

print(
    "Max oracle improvement =",
    comparison[
        "oracle_improvement"
    ].max()
)


# ============================================================
# CANDIDATE COVERAGE MATRIX
# ============================================================

candidates = [
    (mode, slot)
    for mode in MODES
    for slot in range(1, N_STARTS + 1)
]

A = np.zeros(
    (
        len(event_rows),
        len(candidates),
    ),
    dtype=float,
)

for i, row in enumerate(event_rows):

    sub = fits[
        fits["catalog_row"] == row
    ]

    for j, (mode, slot) in enumerate(candidates):

        x = sub[
            (sub["mode"] == mode)
            & (sub["start_slot"] == slot)
        ]

        if len(x) != 1:
            raise RuntimeError(
                f"row={row}, mode={mode}, slot={slot}: "
                f"found {len(x)} rows"
            )

        A[i, j] = float(
            float(x.iloc[0]["delta_final"])
            <= TOL
        )


# ============================================================
# TEST OLD 18-START SET AGAINST FINAL ORACLE
# ============================================================

old_selected_df = pd.read_csv(
    OLD_SELECTION
)

old_selected = [
    (
        str(r["mode"]),
        int(r["start_slot"]),
    )
    for _, r
    in old_selected_df.iterrows()
]

old_report = []

for row in event_rows:

    x = fits[
        fits["catalog_row"] == row
    ].copy()

    mask = [
        (m, int(s)) in old_selected
        for m, s in zip(
            x["mode"],
            x["start_slot"],
        )
    ]

    x = x[mask]

    best = x.loc[
        x["chi2"].idxmin()
    ]

    oracle_row = float(
        best["chi2_oracle_final"]
    )

    old_report.append(
        {
            "catalog_row": row,
            "chi2_oracle_final":
                oracle_row,
            "chi2_old_18":
                float(best["chi2"]),
            "delta_old_18":
                float(best["chi2"])
                - oracle_row,
            "winner_mode":
                best["mode"],
            "winner_slot":
                int(best["start_slot"]),
            "winner_label":
                best["label"],
        }
    )

old_report = pd.DataFrame(
    old_report
)

print()
print("OLD 18-START SET AGAINST FINAL ORACLE")
print("=" * 110)

print(
    "N delta > 0.1 =",
    int(
        (
            old_report["delta_old_18"]
            > TOL
        ).sum()
    )
)

print(
    "max delta =",
    old_report[
        "delta_old_18"
    ].max()
)

bad_old = old_report[
    old_report["delta_old_18"]
    > TOL
]

if len(bad_old):

    print()
    print("Failures:")
    print(
        bad_old.to_string(
            index=False,
            float_format=lambda z: f"{z:.10g}",
        )
    )


# ============================================================
# EXACT MINIMUM SET COVER FOR FINAL ORACLE
# ============================================================

constraints = LinearConstraint(
    A,
    lb=np.ones(len(event_rows)),
    ub=np.full(
        len(event_rows),
        np.inf,
    ),
)

result = milp(
    c=np.ones(
        len(candidates),
        dtype=float,
    ),
    integrality=np.ones(
        len(candidates),
        dtype=int,
    ),
    bounds=Bounds(
        np.zeros(len(candidates)),
        np.ones(len(candidates)),
    ),
    constraints=constraints,
    options={
        "time_limit": 120,
    },
)

if not result.success:
    raise RuntimeError(
        "MILP failed: "
        + str(result.message)
    )

selected_indices = np.where(
    result.x > 0.5
)[0]

selected = [
    candidates[j]
    for j in selected_indices
]


# ============================================================
# FINAL SELECTION SUMMARY
# ============================================================

selection_records = []

for mode, slot in selected:

    x = fits[
        (fits["mode"] == mode)
        & (fits["start_slot"] == slot)
    ]

    labels = Counter(
        x["label"].tolist()
    )

    selection_records.append(
        {
            "mode": mode,
            "start_slot": slot,
            "n_events_covered":
                int(
                    (
                        x["delta_final"]
                        <= TOL
                    ).sum()
                ),
            "median_delta":
                float(
                    x["delta_final"].median()
                ),
            "p90_delta":
                float(
                    x["delta_final"].quantile(
                        0.90
                    )
                ),
            "representative_labels":
                "; ".join(
                    f"{k} [{v}]"
                    for k, v
                    in labels.most_common(5)
                ),
        }
    )

selection = pd.DataFrame(
    selection_records
)

print()
print("FINAL MINIMUM FIXED START SET")
print("=" * 110)

print(
    "N selected fits per event =",
    len(selection),
)

print()

print(
    selection.to_string(
        index=False,
        float_format=lambda z: f"{z:.7g}",
    )
)


# ============================================================
# VALIDATE NEW SET
# ============================================================

final_report = []

for row in event_rows:

    x = fits[
        fits["catalog_row"] == row
    ].copy()

    keep = [
        (m, int(s)) in selected
        for m, s in zip(
            x["mode"],
            x["start_slot"],
        )
    ]

    x = x[keep]

    best = x.loc[
        x["chi2"].idxmin()
    ]

    oracle_row = float(
        best["chi2_oracle_final"]
    )

    final_report.append(
        {
            "catalog_row": row,
            "chi2_oracle_final":
                oracle_row,
            "chi2_reduced":
                float(best["chi2"]),
            "delta":
                float(best["chi2"])
                - oracle_row,
            "winner_mode":
                best["mode"],
            "winner_slot":
                int(best["start_slot"]),
            "winner_label":
                best["label"],
        }
    )

final_report = pd.DataFrame(
    final_report
)

print()
print("FINAL SET VALIDATION")
print("=" * 110)

print(
    "N delta > 0.1 =",
    int(
        (
            final_report["delta"]
            > TOL
        ).sum()
    )
)

print(
    "max delta =",
    final_report["delta"].max()
)

print(
    "p99 delta =",
    final_report[
        "delta"
    ].quantile(0.99)
)


# ============================================================
# OLD VS NEW SELECTED SET
# ============================================================

old_set = set(old_selected)
new_set = set(selected)

print()
print("SELECTION CHANGES")
print("=" * 110)

print(
    "Old N =",
    len(old_set),
)

print(
    "New N =",
    len(new_set),
)

print()
print(
    "Removed:",
    sorted(
        old_set - new_set
    ),
)

print(
    "Added:",
    sorted(
        new_set - old_set
    ),
)


# ============================================================
# REMOVE-ONE NECESSITY TEST
# ============================================================

print()
print("REMOVE-ONE TEST")
print("=" * 110)

for remove in selected:

    reduced = [
        c
        for c in selected
        if c != remove
    ]

    n_bad = 0
    max_delta = 0.0

    for row in event_rows:

        x = fits[
            fits["catalog_row"] == row
        ].copy()

        keep = [
            (m, int(s)) in reduced
            for m, s in zip(
                x["mode"],
                x["start_slot"],
            )
        ]

        x = x[keep]

        if len(x) == 0:
            n_bad += 1
            max_delta = np.inf
            continue

        oracle_row = float(
            x[
                "chi2_oracle_final"
            ].iloc[0]
        )

        delta = (
            x["chi2"].min()
            - oracle_row
        )

        if delta > TOL:
            n_bad += 1

        max_delta = max(
            max_delta,
            float(delta),
        )

    print(
        f"remove {remove}: "
        f"N bad={n_bad:3d}, "
        f"max_delta={max_delta:.8g}"
    )


# ============================================================
# SAVE
# ============================================================

fits.to_csv(
    OUTDIR
    / "gate1_final_all_start_fits.csv",
    index=False,
)

comparison.to_csv(
    OUTDIR
    / "gate1_final_oracle_comparison.csv",
    index=False,
)

old_report.to_csv(
    OUTDIR
    / "gate1_old18_vs_final_oracle.csv",
    index=False,
)

selection.to_csv(
    OUTDIR
    / "gate1_final_minimum_start_set.csv",
    index=False,
)

final_report.to_csv(
    OUTDIR
    / "gate1_final_minimum_start_set_per_event.csv",
    index=False,
)

print()
print("saved final Gate-1 analysis files in:")
print(OUTDIR)
