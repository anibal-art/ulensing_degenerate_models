#!/usr/bin/env python3
"""
Robust-reference builder (old_final-free) for the full 25-event
H1-generated profiling sample: 1 bridge + 15 H0-downstream (4 modes)
+ 4 controlled H1 starts = 20 TRF/event, none of which touch the
truth-relative old_final grid (avoids the 64% crash rate found in
FEASIBILITY_AUDIT.md). Reuses existing
results/pipeline_stage_json/{row}_{bridge,h0_*}.json when already on
disk (9 events from earlier orchestrator runs); computes fresh
otherwise.
"""
import sys
import os
import json
import time
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
STAGE_BRIDGE = os.path.join(HERE, "stage_bridge_morphology.py")
STAGE_H0 = os.path.join(HERE, "stage_h0_downstream.py")
STAGE_H1C4 = os.path.join(HERE, "stage_h1_controlled4.py")
MODES = ["physical", "log_te", "log_rho", "log_te_rho"]

OUT_DIR = os.path.join(HERE, "results", "pipeline_stage_json")
LOG_DIR = os.path.join(HERE, "runtime_local", "orchestrator_logs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def run_subprocess(cmd, log_path):
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess failed: {' '.join(cmd)}, see {log_path}")


def run_event(row, h5_path):
    t0 = time.time()
    bridge_json = os.path.join(OUT_DIR, f"{row}_bridge.json")
    if not os.path.exists(bridge_json):
        run_subprocess([PY, STAGE_BRIDGE, "--h5", h5_path, "--out", bridge_json],
                        os.path.join(LOG_DIR, f"{row}_ref_bridge.log"))

    h0_mode_json = {}
    for mode in MODES:
        out_json = os.path.join(OUT_DIR, f"{row}_h0_{mode}.json")
        if not os.path.exists(out_json):
            run_subprocess([PY, STAGE_H0, "--mode", mode, "--h5", h5_path,
                             "--bridge-json", bridge_json, "--out", out_json],
                            os.path.join(LOG_DIR, f"{row}_ref_h0_{mode}.log"))
        h0_mode_json[mode] = out_json

    all_h0_fits = []
    for path in h0_mode_json.values():
        with open(path) as f:
            all_h0_fits.extend(json.load(f)["fits"])
    finite_h0 = [f for f in all_h0_fits if f.get("status") == "success" and f.get("chi2") is not None]
    h0_final = min(finite_h0, key=lambda f: f["chi2"])
    h0_final_json = os.path.join(OUT_DIR, f"{row}_h0_final_ref.json")
    with open(h0_final_json, "w") as f:
        json.dump(h0_final, f, indent=2)

    h1c4_json = os.path.join(OUT_DIR, f"{row}_h1_controlled4.json")
    if not os.path.exists(h1c4_json):
        run_subprocess([PY, STAGE_H1C4, "--h5", h5_path, "--h0-final-json", h0_final_json,
                         "--out", h1c4_json], os.path.join(LOG_DIR, f"{row}_ref_h1c4.log"))
    with open(h1c4_json) as f:
        h1_data = json.load(f)

    return {
        "catalog_row": row, "h0_final_chi2": h0_final["chi2"], "h0_final_label": h0_final["label"],
        "h1_final_chi2": h1_data["h1_final_chi2"], "h1_final_label": h1_data["h1_final_label"],
        "n_h0_downstream_trf": len(all_h0_fits), "n_h1_controlled4_trf": len(h1_data["controlled4"]),
        "wall_s": time.time() - t0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pandas as pd
    manifest = pd.read_csv(args.rows_csv)

    results = []
    batch_t0 = time.time()
    for _, row_entry in manifest.iterrows():
        row = int(row_entry["catalog_row"])
        h5_path = row_entry["h5_path"]
        r = run_event(row, h5_path)
        results.append(r)
        print(f"row={row} h0_final={r['h0_final_chi2']:.4f} h1_final={r['h1_final_chi2']:.4f} "
              f"delta_lrt_ref={r['h0_final_chi2']-r['h1_final_chi2']:.4f} wall={r['wall_s']:.1f}s "
              f"(elapsed={time.time()-batch_t0:.0f}s)", flush=True)

    with open(args.out, "w") as f:
        json.dump({"n_events": len(results), "events": results,
                    "batch_wall_s": time.time() - batch_t0}, f, indent=2)
    print(f"DONE: {len(results)} events -> {args.out}")
