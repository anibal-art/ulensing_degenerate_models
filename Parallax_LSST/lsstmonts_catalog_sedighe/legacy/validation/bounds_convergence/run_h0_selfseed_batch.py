#!/usr/bin/env python3
import os as _os_path_setup
HERE = _os_path_setup.path.dirname(_os_path_setup.path.abspath(__file__))

import json
import os
import subprocess
import sys
import time

import pandas as pd

SCRATCH = _os_path_setup.path.join(HERE, "results")

with open(os.path.join(SCRATCH, "current_h0_anchors.json")) as f:
    anchors = json.load(f)

truth = pd.read_csv(
    "/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/"
    "lsstmonts_catalog_sedighe/validation/bounds_convergence/"
    "results/b1c_event_features.csv"
).set_index("catalog_row")

FAIL_EVENTS = [
    62786, 83189, 567800, 579320, 82728, 574423,
    557860, 558189, 567365, 35927, 563210, 79501,
]

STRATS = [
    ("physical", 1.0, "physical/current_H0_anchor/1"),
    ("log_te", "truth_rho", "log_te/current_H0_anchor/truth_rho"),
    ("log_te", 0.01, "log_te/current_H0_anchor/0.01"),
    ("log_te", 0.1, "log_te/current_H0_anchor/0.1"),
    ("log_te", 1.0, "log_te/current_H0_anchor/1"),
    ("log_rho", "truth_rho", "log_rho/current_H0_anchor/truth_rho"),
    ("log_rho", 0.01, "log_rho/current_H0_anchor/0.01"),
    ("log_rho", 1.0, "log_rho/current_H0_anchor/1"),
]

DUMMY_ROW = 71181
RUNNER = os.path.join(SCRATCH, "run_dynamic_h0_fit.py")
OUT_CSV = os.path.join(SCRATCH, "h0_selfseed_results.csv")

results = []
n_total = len(FAIL_EVENTS) * len(STRATS)
n_done = 0
t_start = time.time()

for row in FAIL_EVENTS:
    anchor = anchors[str(row)]
    rho_true = float(truth.loc[row, "rho_true"])

    for mode, rho_rule, label in STRATS:
        rho = rho_true if rho_rule == "truth_rho" else float(rho_rule)

        env = os.environ.copy()
        env["HIDDEN_PARALLAX_TRF_COORDS"] = mode
        env["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"

        cmd = [
            sys.executable, RUNNER,
            "--catalog-row", str(row),
            "--label", label,
            "--t0", repr(anchor["t0"]),
            "--u0", repr(anchor["u0"]),
            "--tE", repr(anchor["tE"]),
            "--rho", repr(rho),
            "--dummy-import-row", str(DUMMY_ROW),
        ]

        t0 = time.time()
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=300,
        )
        dt = time.time() - t0
        n_done += 1

        rec = {
            "catalog_row": row,
            "mode": mode,
            "label": label,
            "anchor_strategy": anchor["strategy"],
            "wall_time_s": dt,
        }

        line = None
        for l in proc.stdout.splitlines():
            if l.startswith("RESULT_JSON "):
                line = l[len("RESULT_JSON "):]
                break

        if line is None:
            rec["status"] = "runner_failed"
            rec["stderr_tail"] = proc.stderr[-2000:]
        else:
            r = json.loads(line)
            rec.update(r)

        results.append(rec)
        print(
            f"[{n_done}/{n_total}] row={row} label={label} "
            f"status={rec.get('status')} chi2={rec.get('chi2')} "
            f"dt={dt:.1f}s elapsed={time.time()-t_start:.0f}s",
            flush=True,
        )

        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

print("DONE. Results saved to", OUT_CSV)
