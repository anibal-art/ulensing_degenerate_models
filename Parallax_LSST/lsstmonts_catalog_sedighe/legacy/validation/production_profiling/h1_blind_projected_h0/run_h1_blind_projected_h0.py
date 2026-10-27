#!/usr/bin/env python3
"""
Phase B: H1_blind -> H0_2_blind. Tests whether a truth-blind H1 start
(morphology seed + piEN=piEE=0, all six H1 parameters free) can
provide the geometric information the H1-projected H0_2 mechanism
relies on, WITHOUT the indirect truth-dependence of the original
Demonstration 2A path (truth -> H1_truth-started -> H0_2 seed).

H1_blind is used ONLY as an auxiliary seed generator for H0_2_blind.
The PRIMARY LRT statistic (D_s2_blind) still uses the OFFICIAL settled
H1 (chi2_H1_official, loaded from Demonstration 2A's frozen output,
not recomputed) -- H1_blind's own chi2 is recorded for audit only and
never substituted into the primary statistic, per instruction.

Reuses, unmodified: run_one_fit_full, final_policy.n_photometry_points
/ chi2_dof_sanity_flag, numerical_safeguards.run_same_point_continuation
/ run_nested_h1_rescue / EPSILON_NUMERIC. H0_1 and the official settled
H1 are loaded from Demonstration 1/2A's frozen outputs, not recomputed.
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PP = os.path.dirname(HERE)
BRT = os.path.join(PP, "basin_risk_trigger")
H1P = os.path.join(PP, "h1_projected_h0_start")
REF2 = os.path.join(PP, "reference_v2_independent")
BA = os.path.join(PP, "..", "bounds_audit")
BC = os.path.join(PP, "..", "bounds_convergence")
sys.path.insert(0, BA)
sys.path.insert(0, BC)
sys.path.insert(0, PP)

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = ["run_bounds_audit_refit_core.py", "--catalog-row", "71181",
            "--bounds-profile", "production_candidate", "--fit-scope", "h0", "--dry-run"]

import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402
from load_new_event import load_new_case  # noqa: E402
from run_one_fit_full import run_one_fit_full  # noqa: E402
from final_policy import n_photometry_points, chi2_dof_sanity_flag  # noqa: E402
from numerical_safeguards import run_same_point_continuation, run_nested_h1_rescue, EPSILON_NUMERIC  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"
print(f"epsilon_numeric = {EPSILON_NUMERIC} (source: numerical_safeguards.EPSILON_NUMERIC), reused unchanged.")

DEV_TABLE_PATH = os.environ.get("H1BLIND_DEV_TABLE", os.path.join(BRT, "basin_risk_development_table.csv"))
dev_table = pd.read_csv(DEV_TABLE_PATH)
h1proj = pd.read_csv(os.path.join(H1P, "h1_projected_h0_results.csv")).set_index("event_id")
ref2 = pd.read_csv(os.path.join(REF2, "reference_v2_independent.csv")).set_index("event_id")
manifest = pd.read_csv(os.path.join(PP, "results", "h1_basin_validation_sample_frozen.csv")).set_index("catalog_row")


def dangerous_label(D, D_ref):
    return bool((D_ref < 9) and (D >= 4) and (D > D_ref))


def missed_label(D, D_ref):
    return bool((D_ref >= 4) and (D < 9) and (D < D_ref))


rows = []
for _, dr in dev_table.iterrows():
    row = int(dr["event_id"])
    if bool(dr["excluded_from_development"]):
        continue  # 7 catastrophic events stay outside the main analysis (audit-only elsewhere)

    D_s = float(dr["D_s"])
    D_ref2_independent = float(ref2.loc[row, "D_ref2_independent"])
    D_s2_truthH1 = h1proj.loc[row, "D_s2"]
    D_s2_truthH1 = float(D_s2_truthH1) if pd.notna(D_s2_truthH1) else None

    chi2_H0_1 = float(h1proj.loc[row, "chi2_H0_1"])
    chi2_H0_2_truthH1 = h1proj.loc[row, "chi2_H0_2"]
    chi2_H0_2_truthH1 = float(chi2_H0_2_truthH1) if pd.notna(chi2_H0_2_truthH1) else None
    chi2_H1_official = float(h1proj.loc[row, "chi2_H1_before_second_start"])

    h5_path = manifest.loc[row, "h5_path"]
    meta = load_new_case(h5_path, core)
    n_points = n_photometry_points(meta["curves"])
    t_event0 = time.time()
    n_new_trf = 0

    # ---- H1_blind: morphology seed + piE=(0,0), all 6 free ----
    seed_blind = {"t0": float(dr["h0_seed_t0"]), "u0": float(dr["h0_seed_u0"]),
                  "tE": float(dr["h0_seed_tE"]), "rho": float(dr["h0_seed_rho"]),
                  "piEN": 0.0, "piEE": 0.0}

    h1_blind_nominal = run_one_fit_full(core, fit_lc, meta, "H1", seed_blind, "h1_blind")
    n_new_trf += 1
    flagged_h1, _ = chi2_dof_sanity_flag(h1_blind_nominal, n_points, 6)
    h1_blind_continuation_ran = False
    h1_blind_settled = h1_blind_nominal
    if flagged_h1:
        h1_blind_continuation_ran = True
        cont = run_same_point_continuation(core, fit_lc, meta, "H1", h1_blind_nominal,
                                            label_suffix="chi2dof_continuation")
        n_new_trf += 1
        if cont["fit_record"]["chi2"] <= h1_blind_nominal["chi2"]:
            h1_blind_settled = cont["fit_record"]

    chi2_H1_blind = float(h1_blind_settled["chi2"])

    # ---- H0_2_blind: project (t0,u0,tE,rho) from H1_blind, drop piE ----
    seed2_blind = {"t0": h1_blind_settled["t0"], "u0": h1_blind_settled["u0"],
                   "tE": h1_blind_settled["tE"], "rho": h1_blind_settled["rho"]}

    all_times = []
    for b in ["u", "g", "r", "i", "z", "y"]:
        lc = meta["curves"][b]
        if len(lc):
            all_times.extend(np.asarray(lc[:, 0], dtype=float).tolist())
    t_lo, t_hi = float(np.min(all_times)), float(np.max(all_times))
    bounds_check = {"t0": (t_lo, t_hi), "u0": (-10.0, 10.0), "tE": (0.1, 500000.0), "rho": (1.0e-7, 10.0)}
    violations = [p for p, (lo, hi) in bounds_check.items() if not (lo <= seed2_blind[p] <= hi)]

    if violations:
        rows.append({
            "event_id": row, "chi2_H0_1": chi2_H0_1,
            "chi2_H1_blind": chi2_H1_blind, "H1_blind_continuation_ran": h1_blind_continuation_ran,
            "h0_2_blind_seed_t0": seed2_blind["t0"], "h0_2_blind_seed_u0": seed2_blind["u0"],
            "h0_2_blind_seed_tE": seed2_blind["tE"], "h0_2_blind_seed_rho": seed2_blind["rho"],
            "h0_2_blind_seed_out_of_bounds": True, "h0_2_blind_violating_param": ",".join(violations),
            "chi2_H0_2_blind": None, "H0_2_blind_continuation_ran": None,
            "chi2_H0_final_blindseed": chi2_H0_1, "H0_2_blind_wins": False,
            "chi2_H1_official": chi2_H1_official, "chi2_H1_final_blindseed": chi2_H1_official,
            "nested_H1_rescue_ran": False,
            "D_s": D_s, "D_s2_truthH1": D_s2_truthH1, "D_s2_blind": None, "D_ref2_independent": D_ref2_independent,
            "dangerous_before": dangerous_label(D_s, D_ref2_independent),
            "dangerous_after_truthH1": dangerous_label(D_s2_truthH1, D_ref2_independent) if D_s2_truthH1 is not None else None,
            "dangerous_after_blind": None, "repaired_truthH1": None, "repaired_blind": None,
            "missed_before": missed_label(D_s, D_ref2_independent),
            "missed_after_truthH1": missed_label(D_s2_truthH1, D_ref2_independent) if D_s2_truthH1 is not None else None,
            "missed_after_blind": None, "new_missed_truthH1": None, "new_missed_blind": None,
            "delta_H0_blind_vs_truthprojected": None, "delta_H0_blind_vs_indref": None,
            "chi2_H1_blind_minus_official": chi2_H1_blind - chi2_H1_official,
            "chi2_H1_blind_minus_indref": None,
            "n_actual_new_TRFs": n_new_trf, "wall_s": time.time() - t_event0,
        })
        print(f"row={row} H0_2_BLIND SEED OUT OF BOUNDS: {violations}", flush=True)
        continue

    h0_2_blind_nominal = run_one_fit_full(core, fit_lc, meta, "H0", seed2_blind, "h0_2_blind")
    n_new_trf += 1
    flagged_h0, _ = chi2_dof_sanity_flag(h0_2_blind_nominal, n_points, 4)
    h0_2_blind_continuation_ran = False
    h0_2_blind_settled = h0_2_blind_nominal
    if flagged_h0:
        h0_2_blind_continuation_ran = True
        cont2 = run_same_point_continuation(core, fit_lc, meta, "H0", h0_2_blind_nominal,
                                             label_suffix="chi2dof_continuation")
        n_new_trf += 1
        if cont2["fit_record"]["chi2"] <= h0_2_blind_nominal["chi2"]:
            h0_2_blind_settled = cont2["fit_record"]

    chi2_H0_2_blind = float(h0_2_blind_settled["chi2"])
    chi2_H0_final_blindseed = min(chi2_H0_1, chi2_H0_2_blind)
    h0_2_blind_wins = chi2_H0_2_blind < chi2_H0_1

    # ---- re-check nesting AFTER H0 selection; primary LRT still uses chi2_H1_official ----
    nested_rescue_ran = False
    chi2_H1_final_blindseed = chi2_H1_official
    if chi2_H1_official > (chi2_H0_final_blindseed + EPSILON_NUMERIC):
        nested_rescue_ran = True
        if h0_2_blind_wins:
            winner_params = {"t0": h0_2_blind_settled["t0"], "u0": h0_2_blind_settled["u0"],
                              "tE": h0_2_blind_settled["tE"], "rho": h0_2_blind_settled["rho"]}
        else:
            winner_params = {"t0": float(dr["h0_fit_t0"]), "u0": float(dr["h0_fit_u0"]),
                              "tE": float(dr["h0_fit_tE"]), "rho": float(dr["h0_fit_rho"])}
        rescue = run_nested_h1_rescue(core, fit_lc, meta, winner_params)
        n_new_trf += 1
        chi2_H1_final_blindseed = min(chi2_H1_official, float(rescue["chi2"]))

    D_s2_blind = chi2_H0_final_blindseed - chi2_H1_final_blindseed

    dangerous_before = dangerous_label(D_s, D_ref2_independent)
    dangerous_after_truthH1 = dangerous_label(D_s2_truthH1, D_ref2_independent) if D_s2_truthH1 is not None else None
    dangerous_after_blind = dangerous_label(D_s2_blind, D_ref2_independent)
    repaired_truthH1 = (dangerous_before and dangerous_after_truthH1 is False) if D_s2_truthH1 is not None else None
    repaired_blind = dangerous_before and (not dangerous_after_blind)

    missed_before = missed_label(D_s, D_ref2_independent)
    missed_after_truthH1 = missed_label(D_s2_truthH1, D_ref2_independent) if D_s2_truthH1 is not None else None
    missed_after_blind = missed_label(D_s2_blind, D_ref2_independent)
    new_missed_truthH1 = ((not missed_before) and missed_after_truthH1) if D_s2_truthH1 is not None else None
    new_missed_blind = (not missed_before) and missed_after_blind

    chi2_H1_ref2_independent = float(ref2.loc[row, "chi2_H1_ref2_independent"])

    rows.append({
        "event_id": row, "chi2_H0_1": chi2_H0_1,
        "chi2_H1_blind": chi2_H1_blind, "H1_blind_continuation_ran": h1_blind_continuation_ran,
        "h0_2_blind_seed_t0": seed2_blind["t0"], "h0_2_blind_seed_u0": seed2_blind["u0"],
        "h0_2_blind_seed_tE": seed2_blind["tE"], "h0_2_blind_seed_rho": seed2_blind["rho"],
        "h0_2_blind_seed_out_of_bounds": False, "h0_2_blind_violating_param": None,
        "chi2_H0_2_blind": chi2_H0_2_blind, "H0_2_blind_continuation_ran": h0_2_blind_continuation_ran,
        "chi2_H0_final_blindseed": chi2_H0_final_blindseed, "H0_2_blind_wins": bool(h0_2_blind_wins),
        "chi2_H1_official": chi2_H1_official, "chi2_H1_final_blindseed": chi2_H1_final_blindseed,
        "nested_H1_rescue_ran": nested_rescue_ran,
        "D_s": D_s, "D_s2_truthH1": D_s2_truthH1, "D_s2_blind": D_s2_blind, "D_ref2_independent": D_ref2_independent,
        "dangerous_before": dangerous_before, "dangerous_after_truthH1": dangerous_after_truthH1,
        "dangerous_after_blind": dangerous_after_blind,
        "repaired_truthH1": repaired_truthH1, "repaired_blind": repaired_blind,
        "missed_before": missed_before, "missed_after_truthH1": missed_after_truthH1,
        "missed_after_blind": missed_after_blind,
        "new_missed_truthH1": new_missed_truthH1, "new_missed_blind": new_missed_blind,
        "delta_H0_blind_vs_truthprojected": (chi2_H0_2_blind - chi2_H0_2_truthH1) if chi2_H0_2_truthH1 is not None else None,
        "delta_H0_blind_vs_indref": chi2_H0_final_blindseed - float(ref2.loc[row, "chi2_H0_ref2_independent"]),
        "chi2_H1_blind_minus_official": chi2_H1_blind - chi2_H1_official,
        "chi2_H1_blind_minus_indref": chi2_H1_blind - chi2_H1_ref2_independent,
        "n_actual_new_TRFs": n_new_trf, "wall_s": time.time() - t_event0,
    })
    print(f"row={row} D_s={D_s:.3f} D_s2_blind={D_s2_blind:.3f} D_s2_truthH1={D_s2_truthH1} "
          f"D_ref2ind={D_ref2_independent:.3f} dangerous_before={dangerous_before} "
          f"repaired_blind={repaired_blind} h0_2_blind_wins={h0_2_blind_wins} "
          f"trf={n_new_trf} wall={time.time()-t_event0:.2f}s", flush=True)

df = pd.DataFrame(rows)
out_csv = os.path.join(HERE, "h1_blind_projected_h0_results.csv")
df.to_csv(out_csv, index=False)
print(f"\nWrote {len(df)} rows -> {out_csv}")
