#!/usr/bin/env python3

# ============================================================
# NEW BLOCK: imports
# ============================================================

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# NEW BLOCK: CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--reference-json",
    required=True,
)

parser.add_argument(
    "--rows-csv",
    required=True,
)

parser.add_argument(
    "--out",
    required=True,
)

args = parser.parse_args()


# ============================================================
# NEW BLOCK: repository setup
# ============================================================

ROOT = Path.cwd().resolve()

PP = ROOT / "validation" / "production_profiling"
BA = ROOT / "validation" / "bounds_audit"
BC = ROOT / "validation" / "bounds_convergence"

sys.path.insert(0, str(BA))
sys.path.insert(0, str(BC))
sys.path.insert(0, str(PP))

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_TRF_X_SCALE"] = "jac"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"


# ============================================================
# NEW BLOCK: import fitting machinery
# ============================================================

sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row",
    "71181",
    "--manifest",
    str(
        BA / "data" / "refit_manifest.csv"
    ),
    "--bounds-profile",
    "production_candidate",
    "--fit-scope",
    "h1",
    "--dry-run",
]

import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402

from load_new_event import load_new_case  # noqa: E402
from run_one_fit_full import run_one_fit_full  # noqa: E402


# ============================================================
# NEW BLOCK: target events
# ============================================================

TARGET_ROWS = {
    608288,
    134368,
    779307,
}


# ============================================================
# NEW BLOCK: load inputs
# ============================================================

reference = json.loads(
    Path(
        args.reference_json
    ).read_text()
)

manifest = pd.read_csv(
    args.rows_csv
)

manifest_by_row = {
    int(r["catalog_row"]): r
    for _, r in manifest.iterrows()
}

reference_by_row = {
    int(e["catalog_row"]): e
    for e in reference["events"]
}


# ============================================================
# NEW BLOCK: deterministic local perturbations
# ============================================================

def build_local_starts(best):

    p0 = {
        "t0": float(best["t0"]),
        "u0": float(best["u0"]),
        "tE": float(best["tE"]),
        "rho": float(best["rho"]),
        "piEN": float(best["piEN"]),
        "piEE": float(best["piEE"]),
    }

    starts = [
        (
            "oracle_exact",
            dict(p0),
        )
    ]

    for frac in (
        0.01,
        0.05,
    ):

        # t0: perturb in units of tE, not in units of JD.
        for sign in (-1.0, 1.0):

            p = dict(p0)

            p["t0"] = (
                p0["t0"]
                + sign
                * frac
                * p0["tE"]
            )

            starts.append(
                (
                    f"t0_{sign:+.0f}_{frac:g}",
                    p,
                )
            )

        # u0: relative perturbation.
        for sign in (-1.0, 1.0):

            p = dict(p0)

            scale = max(
                abs(p0["u0"]),
                0.05,
            )

            p["u0"] = (
                p0["u0"]
                + sign
                * frac
                * scale
            )

            starts.append(
                (
                    f"u0_{sign:+.0f}_{frac:g}",
                    p,
                )
            )

        # tE: multiplicative perturbation.
        for sign in (-1.0, 1.0):

            p = dict(p0)

            p["tE"] = (
                p0["tE"]
                * math.exp(
                    sign * frac
                )
            )

            starts.append(
                (
                    f"tE_{sign:+.0f}_{frac:g}",
                    p,
                )
            )

        # rho: multiplicative perturbation.
        for sign in (-1.0, 1.0):

            p = dict(p0)

            p["rho"] = (
                p0["rho"]
                * math.exp(
                    sign * frac
                )
            )

            starts.append(
                (
                    f"rho_{sign:+.0f}_{frac:g}",
                    p,
                )
            )

        # piEN: additive perturbation.
        for sign in (-1.0, 1.0):

            p = dict(p0)

            scale = max(
                abs(p0["piEN"]),
                0.1,
            )

            p["piEN"] = (
                p0["piEN"]
                + sign
                * frac
                * scale
            )

            starts.append(
                (
                    f"piEN_{sign:+.0f}_{frac:g}",
                    p,
                )
            )

        # piEE: additive perturbation.
        for sign in (-1.0, 1.0):

            p = dict(p0)

            scale = max(
                abs(p0["piEE"]),
                0.1,
            )

            p["piEE"] = (
                p0["piEE"]
                + sign
                * frac
                * scale
            )

            starts.append(
                (
                    f"piEE_{sign:+.0f}_{frac:g}",
                    p,
                )
            )

    if len(starts) != 25:
        raise RuntimeError(
            f"Expected 25 starts, got {len(starts)}"
        )

    return starts


# ============================================================
# NEW BLOCK: run stability test
# ============================================================

all_results = []

for row in sorted(TARGET_ROWS):

    ref_event = reference_by_row[row]

    best = (
        ref_event[
            "reference_winner_record"
        ]
    )

    oracle_chi2 = float(
        best["chi2"]
    )

    h5_path = str(
        manifest_by_row[row][
            "h5_path"
        ]
    )

    meta = load_new_case(
        h5_path,
        core,
    )

    starts = build_local_starts(
        best
    )

    fits = []

    print()
    print("=" * 90)
    print(
        f"ROW={row} "
        f"oracle_chi2="
        f"{oracle_chi2:.12g}"
    )
    print("=" * 90)

    for i, (
        label,
        initial,
    ) in enumerate(
        starts,
        start=1,
    ):

        t0 = time.time()

        try:

            record = run_one_fit_full(
                core,
                fit_lc,
                meta,
                "H1",
                initial,
                label,
            )

            chi2 = float(
                record["chi2"]
            )

            result = {
                "label": label,
                "status": "success",
                "chi2": chi2,
                "gap_to_original_oracle":
                    chi2 - oracle_chi2,
                "t0": float(
                    record["t0"]
                ),
                "u0": float(
                    record["u0"]
                ),
                "tE": float(
                    record["tE"]
                ),
                "rho": float(
                    record["rho"]
                ),
                "piEN": float(
                    record["piEN"]
                ),
                "piEE": float(
                    record["piEE"]
                ),
                "wall_s":
                    time.time() - t0,
            }

        except Exception as exc:

            result = {
                "label": label,
                "status": "exception",
                "exception_type":
                    type(exc).__name__,
                "exception_message":
                    str(exc),
                "wall_s":
                    time.time() - t0,
            }

        fits.append(result)

        print(
            f"{i:02d}/25 "
            f"{label:18s} "
            f"chi2={result.get('chi2')} "
            f"gap="
            f"{result.get('gap_to_original_oracle')}"
        )

    finite = [
        f
        for f in fits
        if (
            f.get("chi2") is not None
            and np.isfinite(
                f["chi2"]
            )
        )
    ]

    best_local = min(
        finite,
        key=lambda x: x["chi2"],
    )

    all_results.append(
        {
            "catalog_row": row,
            "original_oracle_chi2":
                oracle_chi2,
            "best_local_chi2":
                float(
                    best_local["chi2"]
                ),
            "best_local_gap":
                float(
                    best_local["chi2"]
                    - oracle_chi2
                ),
            "n_finite":
                len(finite),
            "fits": fits,
        }
    )


# ============================================================
# NEW BLOCK: save output
# ============================================================

payload = {
    "test":
        "h0_h1_oracle_local_stability_v1",
    "events":
        all_results,
}

Path(
    args.out
).write_text(
    json.dumps(
        payload,
        indent=2,
    )
    + "\n"
)

print()
print(
    "OUTPUT =",
    args.out,
)
