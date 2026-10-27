#!/usr/bin/env python3
"""
Per-event orchestrator: Stage 1 (bridge+morphology, physical mode)
-> Stage 2 x4 (H0 downstream, one subprocess per coordinate mode,
required because HIDDEN_PARALLAX_TRF_COORDS is fixed at module
import time) -> aggregate H0 final (min chi2 among the 15 real
downstream fits; the bridge itself is never a valid H0-final
candidate since it used old narrow bounds, not production_candidate)
-> Stage 3 (H1: old_final legacy producer + controlled5).

Each stage runs as its OWN subprocess (matches how this would have
to run in production anyway, given the coordinate-mode import-time
constraint), so the orchestrator measures both subprocess wall time
(includes Python/import/module-load overhead -- real cost) and child
CPU time (via resource.getrusage(RUSAGE_CHILDREN) deltas), never
just summing the per-fit optimizer timings recorded inside each
stage's own JSON.

Serial-baseline mode: --rows-csv points at the frozen profiling
sample manifest, --n selects how many of its rows (in manifest
order) to run, one at a time (no within-event or cross-event
parallelism).
"""
import sys
import os
import json
import time
import resource
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STAGE_BRIDGE = os.path.join(HERE, "stage_bridge_morphology.py")
STAGE_H0 = os.path.join(HERE, "stage_h0_downstream.py")
STAGE_H1 = os.path.join(HERE, "stage_h1.py")

MODES = ["physical", "log_te", "log_rho", "log_te_rho"]


def cpu_children_now():
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime


def run_subprocess(cmd, log_path):
    wall0 = time.time()
    cpu0 = cpu_children_now()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    wall_s = time.time() - wall0
    cpu_s = cpu_children_now() - cpu0
    return {"returncode": proc.returncode, "wall_s": wall_s, "cpu_s": cpu_s}


def run_event(row, h5_path, out_dir, log_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    event_t0 = time.time()
    stages = {}

    # ---- Stage 1: bridge + morphology ----
    bridge_json = os.path.join(out_dir, f"{row}_bridge.json")
    proc = run_subprocess(
        [PY, STAGE_BRIDGE, "--h5", h5_path, "--out", bridge_json],
        os.path.join(log_dir, f"{row}_stage1_bridge.log"),
    )
    stages["stage1_bridge_morphology"] = proc
    if proc["returncode"] != 0:
        raise RuntimeError(f"Stage 1 failed for row={row}, see {log_dir}/{row}_stage1_bridge.log")

    # ---- Stage 2: H0 downstream, one subprocess per mode ----
    h0_mode_json = {}
    for mode in MODES:
        out_json = os.path.join(out_dir, f"{row}_h0_{mode}.json")
        proc = run_subprocess(
            [PY, STAGE_H0, "--mode", mode, "--h5", h5_path,
             "--bridge-json", bridge_json, "--out", out_json],
            os.path.join(log_dir, f"{row}_stage2_h0_{mode}.log"),
        )
        stages[f"stage2_h0_{mode}"] = proc
        if proc["returncode"] != 0:
            raise RuntimeError(f"Stage 2 ({mode}) failed for row={row}, see log.")
        h0_mode_json[mode] = out_json

    # ---- Aggregate H0 final: min chi2 among the 15 real downstream fits ----
    all_h0_fits = []
    for mode, path in h0_mode_json.items():
        with open(path) as f:
            data = json.load(f)
        for fit in data["fits"]:
            all_h0_fits.append(fit)

    finite_h0 = [f for f in all_h0_fits if f.get("status") == "success" and f.get("chi2") is not None]
    if not finite_h0:
        raise RuntimeError(f"No successful H0 downstream fit for row={row}.")
    h0_final = min(finite_h0, key=lambda f: f["chi2"])
    h0_final_json = os.path.join(out_dir, f"{row}_h0_final.json")
    with open(h0_final_json, "w") as f:
        json.dump(h0_final, f, indent=2)

    n_h0_downstream_trf = len(all_h0_fits)

    # ---- Stage 3: H1 (old_final legacy producer + controlled5) ----
    h1_json = os.path.join(out_dir, f"{row}_h1.json")
    proc = run_subprocess(
        [PY, STAGE_H1, "--h5", h5_path, "--bridge-json", bridge_json,
         "--h0-final-json", h0_final_json, "--out", h1_json],
        os.path.join(log_dir, f"{row}_stage3_h1.log"),
    )
    stages["stage3_h1"] = proc
    if proc["returncode"] != 0:
        raise RuntimeError(f"Stage 3 failed for row={row}, see {log_dir}/{row}_stage3_h1.log")

    with open(h1_json) as f:
        h1_data = json.load(f)

    total_wall_s = time.time() - event_t0
    total_cpu_s = sum(s["cpu_s"] for s in stages.values())

    n_h1_trf = h1_data["old_final_n_trf"] + len(h1_data["controlled5"])
    n_bridge_trf = 1
    total_trf = n_bridge_trf + n_h0_downstream_trf + n_h1_trf

    result = {
        "catalog_row": row,
        "h5_path": h5_path,
        "stages": stages,
        "total_wall_s": total_wall_s,
        "total_cpu_s": total_cpu_s,
        "n_bridge_trf": n_bridge_trf,
        "n_h0_downstream_trf": n_h0_downstream_trf,
        "n_old_final_trf": h1_data["old_final_n_trf"],
        "n_controlled5_trf": len(h1_data["controlled5"]),
        "n_h1_trf": n_h1_trf,
        "total_trf": total_trf,
        "h0_final_chi2": h0_final["chi2"],
        "h0_final_label": h0_final["label"],
        "h1_final_chi2": h1_data["h1_final_chi2"],
        "h1_final_label": h1_data["h1_final_label"],
        "stage1_wall_s": stages["stage1_bridge_morphology"]["wall_s"],
        "stage1_cpu_s": stages["stage1_bridge_morphology"]["cpu_s"],
        "stage2_wall_s": sum(stages[f"stage2_h0_{m}"]["wall_s"] for m in MODES),
        "stage2_cpu_s": sum(stages[f"stage2_h0_{m}"]["cpu_s"] for m in MODES),
        "stage3_wall_s": stages["stage3_h1"]["wall_s"],
        "stage3_cpu_s": stages["stage3_h1"]["cpu_s"],
        "old_final_wall_s": h1_data["old_final_wall_s"],
        "old_final_cpu_s": h1_data["old_final_cpu_s"],
        "old_final_polish_ran": h1_data["old_final_polish_ran"],
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-csv", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "pipeline_stage_json"))
    ap.add_argument("--log-dir", default=os.path.join(HERE, "runtime_local", "orchestrator_logs"))
    args = ap.parse_args()

    import pandas as pd

    manifest = pd.read_csv(args.rows_csv)
    subset = manifest.iloc[: args.n]

    results = []
    failures = []
    batch_t0 = time.time()
    for _, row_entry in subset.iterrows():
        row = int(row_entry["catalog_row"])
        h5_path = row_entry["h5_path"]
        print(f"[event] row={row} starting...", flush=True)
        t0 = time.time()
        try:
            r = run_event(row, h5_path, args.out_dir, args.log_dir)
        except Exception as exc:
            # A per-event crash (e.g. a legacy-producer grid candidate whose
            # H0-embedded initial guess falls outside a truth-relative bound
            # -- a real property of the frozen recipe, not a harness bug) is
            # recorded as a failure for THIS event; the batch continues, same
            # as any real production batch runner must.
            print(f"[event] row={row} FAILED after {time.time()-t0:.1f}s: {exc}", flush=True)
            failures.append({"catalog_row": row, "error": str(exc), "wall_s_before_failure": time.time() - t0})
            continue
        print(f"[event] row={row} done: total_wall={r['total_wall_s']:.1f}s "
              f"total_cpu={r['total_cpu_s']:.1f}s total_trf={r['total_trf']} "
              f"(elapsed batch wall={time.time()-batch_t0:.0f}s)", flush=True)
        results.append(r)

    batch_wall_s = time.time() - batch_t0

    with open(args.out, "w") as f:
        json.dump({"batch_wall_s": batch_wall_s, "n_events": len(results), "events": results,
                    "n_failures": len(failures), "failures": failures}, f, indent=2)

    print()
    print("=" * 100)
    print(f"SERIAL BASELINE: {len(results)} events succeeded, {len(failures)} failed, "
          f"batch_wall_s={batch_wall_s:.1f}s -> {args.out}")
