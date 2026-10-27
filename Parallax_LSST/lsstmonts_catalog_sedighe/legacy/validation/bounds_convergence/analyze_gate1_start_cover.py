#!/usr/bin/env python3

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from scipy.optimize import (
    milp,
    LinearConstraint,
    Bounds,
)


# ============================================================
# CONFIG
# ============================================================

MODES = [
    "physical",
    "log_te",
    "log_rho",
    "log_te_rho",
]

TOL = 0.1

ROOT = Path(
    "~/Downloads/hidden_parallax/"
    "production_validation/gate1_coordinates"
).expanduser()

PER_EVENT = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate1_extreme100_coordinate_per_event.csv"
)

OUTDIR = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD EVENT ORACLE
# ============================================================

events = pd.read_csv(PER_EVENT)

events["catalog_row"] = pd.to_numeric(
    events["catalog_row"],
    errors="raise",
).astype(int)

event_rows = events[
    "catalog_row"
].tolist()

oracle = dict(
    zip(
        events["catalog_row"],
        pd.to_numeric(
            events["chi2_envelope"],
            errors="raise",
        ),
    )
)


# ============================================================
# LOAD ALL 52 FIT STRATEGIES
#
# Candidate = (coordinate mode, start slot)
#
# start_slot is the deterministic position 1..13 used by
# run_one_fit. Labels contain event-specific numerical values,
# so slot is the appropriate common identifier here.
# ============================================================

records = []

for mode in MODES:

    for row in event_rows:

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
            raise FileNotFoundError(p)

        df = pd.read_csv(p)

        df = df[
            (df["hypothesis"] == "H0")
            & (df["status"] == "success")
        ].copy()

        # Preserve file/run order.
        df = df.reset_index(drop=True)

        if len(df) != 13:
            raise RuntimeError(
                f"{mode} row={row}: expected 13 successful H0 "
                f"fits, found {len(df)}"
            )

        df["start_slot"] = (
            np.arange(len(df))
            + 1
        )

        df["chi2"] = pd.to_numeric(
            df["chi2"],
            errors="raise",
        )

        for _, r in df.iterrows():

            records.append(
                {
                    "catalog_row":
                        int(row),

                    "mode":
                        mode,

                    "start_slot":
                        int(r["start_slot"]),

                    "label":
                        str(r["label"]),

                    "chi2":
                        float(r["chi2"]),

                    "oracle":
                        float(oracle[row]),

                    "delta":
                        float(
                            r["chi2"]
                            - oracle[row]
                        ),
                }
            )


fits = pd.DataFrame(records)

fits.to_csv(
    OUTDIR
    / "gate1_all_start_fits.csv",
    index=False,
)


# ============================================================
# CANDIDATE COVERAGE MATRIX
# ============================================================

candidates = [
    (mode, slot)
    for mode in MODES
    for slot in range(1, 14)
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
                f"Unexpected candidate multiplicity: "
                f"row={row}, mode={mode}, slot={slot}, n={len(x)}"
            )

        delta = float(
            x.iloc[0]["delta"]
        )

        A[i, j] = (
            delta <= TOL
        )


# Every event must be coverable by the 52-fit oracle.
uncoverable = np.where(
    A.sum(axis=1) == 0
)[0]

if len(uncoverable):
    raise RuntimeError(
        "Events not covered by any candidate: "
        + str(
            [
                event_rows[i]
                for i in uncoverable
            ]
        )
    )


# ============================================================
# EXACT MINIMUM SET COVER
# ============================================================

n_candidates = len(candidates)

objective = np.ones(
    n_candidates,
    dtype=float,
)

constraints = LinearConstraint(
    A,
    lb=np.ones(
        len(event_rows)
    ),
    ub=np.full(
        len(event_rows),
        np.inf,
    ),
)

bounds = Bounds(
    np.zeros(n_candidates),
    np.ones(n_candidates),
)

result = milp(
    c=objective,
    integrality=np.ones(
        n_candidates,
        dtype=int,
    ),
    bounds=bounds,
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

selected_indexes = np.where(
    result.x > 0.5
)[0]

selected = [
    candidates[j]
    for j in selected_indexes
]


# ============================================================
# REPORT SELECTED CANDIDATES
# ============================================================

print()
print("MINIMUM FIXED START SET")
print("=" * 110)

print(
    "N selected fits per event =",
    len(selected),
)

print()

selection_records = []

for mode, slot in selected:

    x = fits[
        (fits["mode"] == mode)
        & (fits["start_slot"] == slot)
    ]

    n_cover = int(
        (x["delta"] <= TOL).sum()
    )

    labels = Counter(
        x["label"].tolist()
    )

    representative_labels = (
        "; ".join(
            [
                f"{label} [{count}]"
                for label, count
                in labels.most_common(5)
            ]
        )
    )

    selection_records.append(
        {
            "mode": mode,
            "start_slot": slot,
            "n_events_covered": n_cover,
            "median_delta":
                float(x["delta"].median()),
            "p90_delta":
                float(x["delta"].quantile(0.90)),
            "representative_labels":
                representative_labels,
        }
    )


selection = pd.DataFrame(
    selection_records
)

print(
    selection.to_string(
        index=False,
        float_format=lambda z: f"{z:.7g}",
    )
)


# ============================================================
# ENVELOPE FROM SELECTED SET
# ============================================================

selected_columns = []

selected_best = []

event_report = []

for row in event_rows:

    x = fits[
        fits["catalog_row"] == row
    ]

    x = x[
        [
            (m, int(s)) in selected
            for m, s in zip(
                x["mode"],
                x["start_slot"],
            )
        ]
    ]

    best = x.loc[
        x["chi2"].idxmin()
    ]

    delta = (
        float(best["chi2"])
        - oracle[row]
    )

    event_report.append(
        {
            "catalog_row":
                row,

            "oracle_chi2":
                oracle[row],

            "selected_chi2":
                float(best["chi2"]),

            "delta":
                delta,

            "winner_mode":
                best["mode"],

            "winner_slot":
                int(best["start_slot"]),

            "winner_label":
                best["label"],
        }
    )


event_report = pd.DataFrame(
    event_report
)

print()
print("SELECTED SET VALIDATION")
print("=" * 110)

print(
    "N delta > 0.1 =",
    int(
        (
            event_report["delta"]
            > TOL
        ).sum()
    )
)

print(
    "max delta =",
    event_report["delta"].max()
)

print(
    "p99 delta =",
    event_report["delta"].quantile(
        0.99
    )
)


# ============================================================
# UNIQUE NECESSITY
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
        ]

        x = x[
            [
                (m, int(s)) in reduced
                for m, s in zip(
                    x["mode"],
                    x["start_slot"],
                )
            ]
        ]

        if len(x) == 0:
            n_bad += 1
            max_delta = np.inf
            continue

        delta = (
            x["chi2"].min()
            - oracle[row]
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

selection.to_csv(
    OUTDIR
    / "gate1_minimum_start_set.csv",
    index=False,
)

event_report.to_csv(
    OUTDIR
    / "gate1_minimum_start_set_per_event.csv",
    index=False,
)

print()
print("saved:")
print(
    OUTDIR
    / "gate1_minimum_start_set.csv"
)
print(
    OUTDIR
    / "gate1_minimum_start_set_per_event.csv"
)
