#!/usr/bin/env python3

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PP = ROOT.parent

STAGE = PP / "stage_h1_controlled4.py"

EVENT_ROOT = (
    ROOT
    / "results"
    / "full3336_events"
)

OUT_ROOT = (
    ROOT
    / "results"
    / "h1_smoke3"
)

SAMPLE = (
    ROOT
    / "h1_h0_robustness_sample_frozen.csv"
)

ROWS = [
    709579,
    898250,
    245674,
]


def finite_success(record):
    try:
        return (
            record.get("status") == "success"
            and math.isfinite(float(record["chi2"]))
        )
    except Exception:
        return False


sample = pd.read_csv(SAMPLE)

by_row = (
    sample
    .set_index("catalog_row")
)

for row in ROWS:

    event_dir = EVENT_ROOT / f"row_{row}"

    all_h0 = []

    for mode in [
        "physical",
        "log_te",
        "log_rho",
        "log_te_rho",
    ]:
        p = event_dir / f"h0_{mode}.json"

        with open(p) as f:
            payload = json.load(f)

        all_h0.extend(payload["fits"])

    finite = [
        x for x in all_h0
        if finite_success(x)
    ]

    if not finite:
        raise RuntimeError(
            f"row={row}: no finite H0 fits"
        )

    best_h0 = min(
        finite,
        key=lambda x: float(x["chi2"]),
    )

    h0_final = OUT_ROOT / f"row_{row}_h0_final.json"

    with open(h0_final, "w") as f:
        json.dump(
            best_h0,
            f,
            indent=2,
        )

    h1_out = OUT_ROOT / f"row_{row}_h1_controlled4.json"
    log = OUT_ROOT / "logs" / f"row_{row}.log"

    h5 = str(
        by_row.loc[row, "h5_path"]
    )

    cmd = [
        sys.executable,
        str(STAGE),
        "--h5",
        h5,
        "--h0-final-json",
        str(h0_final),
        "--out",
        str(h1_out),
    ]

    print(
        f"RUN row={row}",
        flush=True,
    )

    with open(log, "w") as lf:
        p = subprocess.run(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )

    if p.returncode != 0:
        raise RuntimeError(
            f"row={row}: controlled4 failed; "
            f"see {log}"
        )

print("DONE")
