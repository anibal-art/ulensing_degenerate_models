#!/usr/bin/env python3

import os

# One CPU per task; child processes inherit these.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PP = ROOT.parent
STAGE = PP / "stage_h1_controlled4.py"


parser = argparse.ArgumentParser()

parser.add_argument(
    "--manifest",
    required=True,
)

parser.add_argument(
    "--start",
    type=int,
    required=True,
)

parser.add_argument(
    "--stop",
    type=int,
    required=True,
)

parser.add_argument(
    "--out-root",
    required=True,
)

parser.add_argument(
    "--summary-out",
    required=True,
)

args = parser.parse_args()


manifest = pd.read_csv(args.manifest)

if not (
    0 <= args.start <= args.stop <= len(manifest)
):
    raise ValueError(
        f"invalid slice [{args.start}, {args.stop}) "
        f"for N={len(manifest)}"
    )

chunk = (
    manifest
    .iloc[args.start:args.stop]
    .copy()
)

out_root = Path(args.out_root)
out_root.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_LABELS = [
    "truth",
    "H0_NESTED_piE_0",
    "truth_half_piE",
    "truth_mirror_u0_piEN",
]


def finite_chi2(record):
    try:
        return (
            record.get("status") == "success"
            and np.isfinite(float(record["chi2"]))
        )
    except Exception:
        return False


records = []


for _, row in chunk.iterrows():

    wall0 = time.time()

    catalog_row = int(
        row["catalog_row"]
    )

    event_dir = (
        out_root
        / f"row_{catalog_row}"
    )

    event_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    h0_json = (
        event_dir
        / "h0_aug.json"
    )

    h1_json = (
        event_dir
        / "h1_controlled4.json"
    )

    log_path = (
        event_dir
        / "h1_controlled4.log"
    )

    try:

        # --------------------------------------------------------
        # Exact H0 augmented solution to embed in H1 at piE=(0,0)
        # --------------------------------------------------------

        h0_payload = {
            "catalog_row": catalog_row,
            "source": str(
                row["h0_aug_source"]
            ),
            "chi2": float(
                row["chi2_h0_aug"]
            ),
            "t0": float(
                row["h0_aug_t0"]
            ),
            "u0": float(
                row["h0_aug_u0"]
            ),
            "tE": float(
                row["h0_aug_tE"]
            ),
            "rho": float(
                row["h0_aug_rho"]
            ),
        }

        with open(
            h0_json,
            "w",
        ) as f:
            json.dump(
                h0_payload,
                f,
                indent=2,
            )

        # --------------------------------------------------------
        # Four-start H1 stage
        # --------------------------------------------------------

        cmd = [
            sys.executable,
            str(STAGE),
            "--h5",
            str(row["h5_path"]),
            "--h0-final-json",
            str(h0_json),
            "--out",
            str(h1_json),
        ]

        with open(
            log_path,
            "w",
        ) as log:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"controlled4 returned "
                f"{proc.returncode}; see {log_path}"
            )

        with open(h1_json) as f:
            payload = json.load(f)

        if int(payload["catalog_row"]) != catalog_row:
            raise RuntimeError(
                "controlled4 catalog_row mismatch"
            )

        fits = payload["controlled4"]

        if len(fits) != 4:
            raise RuntimeError(
                f"expected 4 controlled H1 fits, "
                f"found {len(fits)}"
            )

        labels = [
            x.get("label")
            for x in fits
        ]

        if set(labels) != set(EXPECTED_LABELS):
            raise RuntimeError(
                f"unexpected H1 labels: {labels}"
            )

        by_label = {
            x["label"]: x
            for x in fits
        }

        finite = [
            x
            for x in fits
            if finite_chi2(x)
        ]

        if not finite:
            raise RuntimeError(
                "no finite successful H1 fit"
            )

        best_c4 = min(
            finite,
            key=lambda x: float(
                x["chi2"]
            ),
        )

        # --------------------------------------------------------
        # Production and augmented H1
        # --------------------------------------------------------

        chi0_aug = float(
            row["chi2_h0_aug"]
        )

        chi1_prod = float(
            row["chi2_h1_prod"]
        )

        chi1_c4 = float(
            best_c4["chi2"]
        )

        # Preserve production H1 as an explicit candidate.
        chi1_aug = min(
            chi1_prod,
            chi1_c4,
        )

        delta_h1 = (
            chi1_prod
            - chi1_aug
        )

        T_prod = float(
            row["T_prod"]
        )

        T_h0only = float(
            row["T_h0only"]
        )

        T_both = (
            chi0_aug
            - chi1_aug
        )

        closure = (
            T_both
            - (
                T_h0only
                + delta_h1
            )
        )

        # --------------------------------------------------------
        # Exact nesting diagnostic
        #
        # H0 is a literal subspace of H1 at piE=(0,0), therefore
        # chi2_H1,min <= chi2_H0,min.
        #
        # Keep this separate from T_both so no correction is hidden.
        # --------------------------------------------------------

        chi1_nested_exact = min(
            chi1_aug,
            chi0_aug,
        )

        delta_h1_nested_exact = (
            chi1_prod
            - chi1_nested_exact
        )

        T_both_nested_exact = (
            chi0_aug
            - chi1_nested_exact
        )

        closure_nested_exact = (
            T_both_nested_exact
            - (
                T_h0only
                + delta_h1_nested_exact
            )
        )

        # --------------------------------------------------------
        # Individual controlled4 fits
        # --------------------------------------------------------

        def c2(label):
            x = by_label[label]
            value = x.get(
                "chi2",
                np.nan,
            )
            try:
                return float(value)
            except Exception:
                return np.nan

        rec = {
            "catalog_row": catalog_row,
            "status": "success",

            "h0_aug_source": str(
                row["h0_aug_source"]
            ),

            "chi2_h0_prod": float(
                row["chi2_h0_prod"]
            ),
            "chi2_h0_aug": chi0_aug,

            "chi2_h1_prod": chi1_prod,

            "chi2_h1_truth": c2(
                "truth"
            ),
            "chi2_h1_nested": c2(
                "H0_NESTED_piE_0"
            ),
            "chi2_h1_half": c2(
                "truth_half_piE"
            ),
            "chi2_h1_mirror": c2(
                "truth_mirror_u0_piEN"
            ),

            "chi2_h1_controlled4": chi1_c4,
            "chi2_h1_aug": chi1_aug,

            "delta_h1": delta_h1,

            "T_prod": T_prod,
            "T_h0only": T_h0only,
            "T_both": T_both,

            "closure": closure,

            "nested_gap_h1_minus_h0": (
                chi1_aug
                - chi0_aug
            ),

            "chi2_h1_nested_exact": (
                chi1_nested_exact
            ),
            "delta_h1_nested_exact": (
                delta_h1_nested_exact
            ),
            "T_both_nested_exact": (
                T_both_nested_exact
            ),
            "closure_nested_exact": (
                closure_nested_exact
            ),

            "truth_rerun_minus_prod": (
                c2("truth")
                - chi1_prod
            ),

            "h1_controlled4_winner": str(
                best_c4["label"]
            ),

            "h1_aug_source": (
                "controlled4"
                if chi1_c4 < chi1_prod
                else "production"
            ),

            "n_h1_fits": len(fits),
            "n_h1_success": len(finite),

            "wall_time_s": (
                time.time()
                - wall0
            ),

            "error": None,
        }

    except Exception as exc:

        rec = {
            "catalog_row": catalog_row,
            "status": "failed",
            "wall_time_s": (
                time.time()
                - wall0
            ),
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
            "traceback": (
                traceback.format_exc()
            ),
        }

    records.append(rec)

    print(
        f"row={catalog_row} "
        f"status={rec['status']} "
        f"T_h0only={rec.get('T_h0only')} "
        f"delta_h1={rec.get('delta_h1')} "
        f"T_both={rec.get('T_both')} "
        f"wall={rec['wall_time_s']:.2f}s",
        flush=True,
    )


summary = pd.DataFrame(
    records
)

summary_out = Path(
    args.summary_out
)

summary_out.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_parquet(
    summary_out,
    index=False,
)

print(
    "saved =",
    summary_out,
)
