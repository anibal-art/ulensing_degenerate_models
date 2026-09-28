#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PP = HERE.parent

STAGE_BRIDGE = PP / "stage_bridge_morphology.py"
STAGE_H0 = PP / "stage_h0_downstream.py"

MODES = [
    "physical",
    "log_te",
    "log_rho",
    "log_te_rho",
]

TCRIT = 13.15982011480088


def run_subprocess(cmd, log_path):

    env = os.environ.copy()

    # Important on CHE shared filesystem.
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"

    with open(log_path, "w") as logf:

        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )

    if proc.returncode != 0:

        raise RuntimeError(
            f"subprocess failed rc={proc.returncode}: "
            f"{' '.join(map(str, cmd))}; "
            f"log={log_path}"
        )


def robust_h0_for_event(row, out_root):

    catalog_row = int(row["catalog_row"])
    h5_path = str(row["h5_path"])

    event_dir = (
        out_root
        / f"row_{catalog_row}"
    )

    event_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir = event_dir / "logs"

    log_dir.mkdir(
        exist_ok=True,
    )


    # ============================================================
    # Stage 1:
    # old-H0 bridge + morphology extraction
    #
    # The bridge costs one TRF, but is NOT a final H0 candidate.
    # ============================================================

    bridge_json = (
        event_dir
        / "bridge.json"
    )

    if not bridge_json.exists():

        run_subprocess(
            [
                sys.executable,
                str(STAGE_BRIDGE),
                "--h5",
                h5_path,
                "--out",
                str(bridge_json),
            ],
            log_dir / "bridge.log",
        )


    # ============================================================
    # Stage 2:
    # 15 downstream H0 fits across the four coordinate modes.
    # ============================================================

    mode_jsons = []

    for mode in MODES:

        out_json = (
            event_dir
            / f"h0_{mode}.json"
        )

        if not out_json.exists():

            run_subprocess(
                [
                    sys.executable,
                    str(STAGE_H0),
                    "--mode",
                    mode,
                    "--h5",
                    h5_path,
                    "--bridge-json",
                    str(bridge_json),
                    "--out",
                    str(out_json),
                ],
                log_dir / f"h0_{mode}.log",
            )

        mode_jsons.append(
            (mode, out_json)
        )


    # ============================================================
    # Gather the 15 downstream candidates.
    # ============================================================

    all_fits = []

    mode_counts = {}

    for mode, path in mode_jsons:

        with open(path) as f:
            payload = json.load(f)

        fits = payload["fits"]

        mode_counts[mode] = len(fits)

        for fit in fits:

            rec = dict(fit)

            rec["_mode_file"] = mode

            all_fits.append(rec)


    # The frozen architecture must give:
    #
    # physical   2
    # log_te     4
    # log_rho    8
    # log_te_rho 1
    #
    # total      15
    #
    # plus the bridge = 16 real TRFs/event.
    with open(bridge_json) as f:
        bridge_payload = json.load(f)

    morphology_available = (
        bridge_payload.get("morphology_candidate") is not None
    )

    if morphology_available:
        expected_counts = {
            "physical": 2,
            "log_te": 4,
            "log_rho": 8,
            "log_te_rho": 1,
        }
        expected_downstream = 15
    else:
        # The frozen morphology strategies are conditional on a
        # measurable morphology seed.  When no morphology candidate
        # exists, stage_h0_downstream intentionally skips the two
        # morphology starts.
        expected_counts = {
            "physical": 2,
            "log_te": 4,
            "log_rho": 7,
            "log_te_rho": 0,
        }
        expected_downstream = 13

    if mode_counts != expected_counts:

        raise RuntimeError(
            f"row={catalog_row}: unexpected frozen-H0 "
            f"strategy counts: {mode_counts}; "
            f"expected={expected_counts}"
        )

    if len(all_fits) != expected_downstream:

        raise RuntimeError(
            f"row={catalog_row}: expected "
            f"{expected_downstream} downstream H0 fits, "
            f"got {len(all_fits)}"
        )


    # ============================================================
    # Successful finite H0 fits.
    # ============================================================

    finite = []

    for fit in all_fits:

        if fit.get("status") != "success":
            continue

        chi2 = fit.get("chi2")

        if chi2 is None:
            continue

        try:
            chi2 = float(chi2)
        except Exception:
            continue

        if not np.isfinite(chi2):
            continue

        fit = dict(fit)
        fit["chi2"] = chi2

        finite.append(fit)


    if not finite:

        raise RuntimeError(
            f"row={catalog_row}: no finite successful "
            f"robust-H0 downstream fits"
        )


    best = min(
        finite,
        key=lambda x: x["chi2"],
    )


    # ============================================================
    # Compare against PRODUCTION.
    #
    # We define the augmented H0 as the minimum of:
    #
    #   production H0
    #   robust frozen multistart H0
    #
    # so the validation can only improve or preserve H0.
    # ============================================================

    chi2_h0_prod = float(
        row["chi2_h0_fit"]
    )

    chi2_h1_prod = float(
        row["chi2_h1_fit"]
    )

    T_prod = float(
        row["T_fit"]
    )

    chi2_h0_robust = float(
        best["chi2"]
    )

    chi2_h0_aug = min(
        chi2_h0_prod,
        chi2_h0_robust,
    )

    delta_h0 = (
        chi2_h0_prod
        - chi2_h0_aug
    )

    T_aug = (
        chi2_h0_aug
        - chi2_h1_prod
    )


    # Exact expected identity:
    #
    # T_aug = T_prod - delta_h0
    closure = (
        T_aug
        -
        (
            T_prod
            - delta_h0
        )
    )


    return {
        "catalog_row": catalog_row,

        "tE_true": float(
            row["tE_true"]
        ),
        "X_pi": float(
            row["X_pi"]
        ),

        "in_random": bool(
            row["in_random"]
        ),
        "in_long_high": bool(
            row["in_long_high"]
        ),
        "in_near_threshold": bool(
            row["in_near_threshold"]
        ),

        "chi2_h0_prod": chi2_h0_prod,
        "chi2_h1_prod": chi2_h1_prod,
        "T_prod": T_prod,

        "chi2_h0_robust16": (
            chi2_h0_robust
        ),

        "chi2_h0_aug": (
            chi2_h0_aug
        ),

        "delta_h0": delta_h0,

        "T_aug": T_aug,

        "detected_prod": bool(
            T_prod > TCRIT
        ),

        "detected_aug": bool(
            T_aug > TCRIT
        ),

        "crossed_down": bool(
            (T_prod > TCRIT)
            and
            (T_aug <= TCRIT)
        ),

        "closure": closure,

        "robust_winner_mode_file": (
            best.get("_mode_file")
        ),

        "robust_winner_label": (
            best.get("label")
        ),

        "robust_t0": (
            best.get("t0")
        ),

        "robust_u0": (
            best.get("u0")
        ),

        "robust_tE": (
            best.get("tE")
        ),

        "robust_rho": (
            best.get("rho")
        ),

        "n_downstream_fits": len(
            all_fits
        ),

        "n_downstream_success": len(
            finite
        ),

        "morphology_available": morphology_available,

        "n_total_robust_trf": 1 + len(all_fits),

        "mode_count_physical": (
            mode_counts["physical"]
        ),

        "mode_count_log_te": (
            mode_counts["log_te"]
        ),

        "mode_count_log_rho": (
            mode_counts["log_rho"]
        ),

        "mode_count_log_te_rho": (
            mode_counts["log_te_rho"]
        ),

        "status": "success",
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--stop",
        type=int,
        default=None,
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


    manifest = pd.read_csv(
        args.manifest
    )


    stop = (
        len(manifest)
        if args.stop is None
        else min(
            args.stop,
            len(manifest),
        )
    )

    work = manifest.iloc[
        args.start:stop
    ].copy()


    out_root = Path(
        args.out_root
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    results = []

    for _, row in work.iterrows():

        wall0 = time.time()

        try:

            result = robust_h0_for_event(
                row,
                out_root,
            )

        except Exception as exc:

            result = {
                "catalog_row": int(
                    row["catalog_row"]
                ),
                "status": "failed",
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            }

        result["wall_time_s"] = (
            time.time()
            - wall0
        )

        results.append(
            result
        )

        print(
            f"row={int(row['catalog_row'])} "
            f"status={result['status']} "
            f"delta_h0={result.get('delta_h0')} "
            f"Tprod={result.get('T_prod')} "
            f"Taug={result.get('T_aug')} "
            f"wall={result['wall_time_s']:.2f}s",
            flush=True,
        )


    out = pd.DataFrame(
        results
    )

    summary_out = Path(
        args.summary_out
    )

    summary_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_parquet(
        summary_out,
        index=False,
    )

    print()
    print(
        "saved =",
        summary_out,
    )


if __name__ == "__main__":
    main()
