#!/usr/bin/env python3
"""
Frozen, independent (H0_2/H1_projected-free) H0 verification panel for
the 19 events where H0_2 beat the old controlled4 reference
(Demonstration 2A). Panel is documented in full in
REFERENCE_V2_INDEPENDENT.md BEFORE this script's results were
inspected; it is fixed identically for all 19 events, not tuned per
event, and reuses only already-existing strategy/mechanism code paths
(stage_bridge_morphology.py's old_H0 bridge, stage_h0_downstream.py's
truth/old_H0-anchor construction pattern -- unmodified) extended to
one already-implemented-but-unused coordinate mode (log_te_rho) and one
already-established rho-fraction token ("truth_rho") in physical mode
that the cost-minimal 16-strategy architecture happened not to include.

Panel (4 members):
  1. old_H0_bridge       -- REUSED from the already-computed bridge fit
                             (results/pipeline_stage_json/{row}_bridge.json,
                             produced by orchestrator_robust_reference_
                             controlled4.py for the same 100-event
                             sample); truth-started, OLD narrow FAST
                             bounds. Zero new TRF.
  2. physical/truth/truth_rho -- NEW: 1 TRF, physical mode, truth-
                             anchored (t0,u0,tE from truth), rho=truth
                             rho, production_candidate bounds.
  3. log_te_rho/truth/1  -- NEW: 1 TRF, log_te_rho mode (already
                             implemented in core.py, never used for a
                             non-morphology start anywhere in this
                             project), truth-anchored, rho fraction "1"
                             (matching log_te's own "1" token),
                             production_candidate bounds.
  4. log_te_rho/old_H0/1 -- NEW: 1 TRF, log_te_rho mode, old_H0-anchored
                             (start = bridge vector), rho fraction "1",
                             production_candidate bounds.

Same H0 physical model, profiled-flux objective, and numerical
implementation as every other H0 fit in this project
(run_one_fit_full); same catastrophic chi2/dof>50 -> one same-point
continuation safeguard already adopted in final_policy.py, applied
identically here.
"""
import sys
import os
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PP = os.path.dirname(HERE)
BA = os.path.join(PP, "..", "bounds_audit")
BC = os.path.join(PP, "..", "bounds_convergence")
sys.path.insert(0, BA)
sys.path.insert(0, BC)
sys.path.insert(0, PP)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["physical", "log_te_rho"])
ap.add_argument("--events-csv", default=os.path.join(HERE, "verification_panel_events_19.csv"))
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = args.mode
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = ["run_bounds_audit_refit_core.py", "--catalog-row", "71181",
            "--bounds-profile", "production_candidate", "--fit-scope", "h0", "--dry-run"]

import pandas as pd  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402
from load_new_event import load_new_case  # noqa: E402
from run_one_fit_full import run_one_fit_full  # noqa: E402
from final_policy import n_photometry_points, chi2_dof_sanity_flag  # noqa: E402
from numerical_safeguards import run_same_point_continuation  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"

BRIDGE_JSON_DIR = os.path.join(PP, "results", "pipeline_stage_json")

events = pd.read_csv(args.events_csv)

records = []
for _, row_entry in events.iterrows():
    row = int(row_entry["catalog_row"])
    h5_path = row_entry["h5_path"]
    meta = load_new_case(h5_path, core)
    n_points = n_photometry_points(meta["curves"])

    bridge_json = os.path.join(BRIDGE_JSON_DIR, f"{row}_bridge.json")
    with open(bridge_json) as f:
        bridge_data = json.load(f)
    bridge_vector = bridge_data["bridge_vector"]

    if args.mode == "physical":
        strategies = [("physical/truth/truth_rho",
                        {"t0": meta["truth"]["t0"], "u0": meta["truth"]["u0"],
                         "tE": meta["truth"]["tE"], "rho": meta["truth"]["rho"]})]
    else:  # log_te_rho
        strategies = [
            ("log_te_rho/truth/1",
             {"t0": meta["truth"]["t0"], "u0": meta["truth"]["u0"],
              "tE": meta["truth"]["tE"], "rho": meta["truth"]["rho"]}),
            ("log_te_rho/old_H0/1",
             {"t0": bridge_vector["t0"], "u0": bridge_vector["u0"],
              "tE": bridge_vector["tE"], "rho": meta["truth"]["rho"]}),
        ]

    for label, initial in strategies:
        t0c, cpu0 = time.time(), time.process_time()
        try:
            nominal = run_one_fit_full(core, fit_lc, meta, "H0", initial, label)
        except Exception as exc:
            # A strategy-specific seed/bounds incompatibility (e.g. the
            # old_H0 bridge's own t0, fit under its OWN old narrow
            # bounds, occasionally falls just outside the CURRENT data-
            # driven H0 t0 window) is recorded as a failed panel entry,
            # not silently patched or imputed -- this is itself a real,
            # reportable domain-bounds audit finding, not a bug to fix
            # here.
            records.append({
                "event_id": row, "strategy_name": label, "chi2_H0": None,
                "t0": None, "u0": None, "tE": None, "rho": None,
                "optimizer_success": None, "optimizer_status": None,
                "optimizer_nfev": None, "optimizer_optimality": None,
                "continuation_ran": None, "wall_s": time.time() - t0c,
                "cpu_s": time.process_time() - cpu0,
                "failed": True, "failure_reason": str(exc),
            })
            print(f"row={row} strategy={label} FAILED: {exc}", flush=True)
            continue
        flagged, reduced = chi2_dof_sanity_flag(nominal, n_points, 4)
        continuation_ran = False
        final = nominal
        if flagged:
            continuation_ran = True
            cont = run_same_point_continuation(core, fit_lc, meta, "H0", nominal,
                                                label_suffix="chi2dof_continuation")
            if cont["fit_record"]["chi2"] <= nominal["chi2"]:
                final = cont["fit_record"]
        wall_s = time.time() - t0c
        cpu_s = time.process_time() - cpu0

        records.append({
            "event_id": row, "strategy_name": label, "chi2_H0": final["chi2"],
            "t0": final["t0"], "u0": final["u0"], "tE": final["tE"], "rho": final["rho"],
            "optimizer_success": final.get("optimizer_success"),
            "optimizer_status": final.get("optimizer_status"),
            "optimizer_nfev": final.get("optimizer_nfev"),
            "optimizer_optimality": final.get("optimizer_optimality"),
            "continuation_ran": continuation_ran,
            "wall_s": wall_s, "cpu_s": cpu_s,
            "failed": False, "failure_reason": None,
        })
        print(f"row={row} strategy={label} chi2={final['chi2']:.4f} "
              f"continuation_ran={continuation_ran} wall={wall_s:.2f}s", flush=True)

df = pd.DataFrame(records)
df.to_csv(args.out, index=False)
print(f"\nWrote {len(df)} rows -> {args.out}")
