#!/usr/bin/env python3
"""
Phase A: geometry of the 4 residual (2 H0-limited + 2 H1-limited)
cases, using ZERO new fits -- every quantity is read from already-
stored JSON/CSV produced by Demonstration 1, Demonstration 2A, and
reference_v2_independent.
"""
import os
import json
import math
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PP = os.path.dirname(HERE)
BRT = os.path.join(PP, "basin_risk_trigger")
H1P = os.path.join(PP, "h1_projected_h0_start")
REF2 = os.path.join(PP, "reference_v2_independent")

H0_LIMITED = [520062, 324312]
H1_LIMITED = [443054, 114675]

dev_table = pd.read_csv(os.path.join(BRT, "basin_risk_development_table.csv")).set_index("event_id")
h1proj = pd.read_csv(os.path.join(H1P, "h1_projected_h0_results.csv")).set_index("event_id")
ref2 = pd.read_csv(os.path.join(REF2, "reference_v2_independent.csv")).set_index("event_id")
ref_events = {e["catalog_row"]: e for e in
              json.load(open(os.path.join(PP, "results", "robust_reference_controlled4_h1basin_100events.json")))["events"]}
fp_events = {e["catalog_row"]: e for e in
             json.load(open(os.path.join(PP, "results", "final_policy_h1basinvalidation_100events.json")))["events"]}

MODES = ["physical", "log_te", "log_rho", "log_te_rho"]
PSJ = os.path.join(PP, "results", "pipeline_stage_json")


def find_reference_winner_params(row, label):
    """The winning strategy's own (t0,u0,tE,rho) -- looked up from the
    already-computed per-mode H0-downstream JSON files (never
    recomputed)."""
    if label == "old_H0_bridge":
        bridge = json.load(open(os.path.join(PSJ, f"{row}_bridge.json")))
        bv = bridge["bridge_vector"]
        return {"t0": bv["t0"], "u0": bv["u0"], "tE": bv["tE"], "rho": bv["rho"]}
    for mode in MODES:
        path = os.path.join(PSJ, f"{row}_h0_{mode}.json")
        if not os.path.exists(path):
            continue
        data = json.load(open(path))
        for fit in data["fits"]:
            if fit["label"] == label:
                return {"t0": fit["t0"], "u0": fit["u0"], "tE": fit["tE"], "rho": fit["rho"]}
    return None


def safe_log_ratio(a, b):
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return math.log(a / b)


rows_h0 = []
for row in H0_LIMITED:
    dr = dev_table.loc[row]
    hp = h1proj.loc[row]
    r2 = ref2.loc[row]
    ref = ref_events[row]

    morph = {"t0": dr["h0_seed_t0"], "u0": dr["h0_seed_u0"], "tE": dr["h0_seed_tE"], "rho": dr["h0_seed_rho"]}
    h0_1 = {"t0": dr["h0_fit_t0"], "u0": dr["h0_fit_u0"], "tE": dr["h0_fit_tE"], "rho": dr["h0_fit_rho"],
            "chi2": hp["chi2_H0_1"]}
    h0_2_seed = {"t0": hp["h0_2_seed_t0"], "u0": hp["h0_2_seed_u0"], "tE": hp["h0_2_seed_tE"], "rho": hp["h0_2_seed_rho"]}
    h0_2 = {"t0": hp["h0_2_fit_t0"], "u0": hp["h0_2_fit_u0"], "tE": hp["h0_2_fit_tE"], "rho": hp["h0_2_fit_rho"],
            "chi2": hp["chi2_H0_2"]}
    ref_winner_label = r2["best_independent_H0_strategy"]
    # "old_controlled4_reference" is this project's own bookkeeping label
    # for "the min over the 15 downstream strategies", not an actual
    # per-mode fit label -- resolve to the real winning strategy name
    # already recorded by the (pre-existing, unmodified) controlled4
    # reference pipeline before looking it up.
    lookup_label = ref["h0_final_label"] if ref_winner_label == "old_controlled4_reference" else ref_winner_label
    ref_params = find_reference_winner_params(row, lookup_label)
    ref_chi2 = r2["chi2_H0_ref2_independent"]

    def pair(name_a, a, name_b, b):
        out = {"pair": f"{name_a}_vs_{name_b}"}
        if a is None or b is None:
            out["available"] = False
            return out
        out["available"] = True
        out["dt0"] = a["t0"] - b["t0"]
        out["dt0_over_tE"] = (a["t0"] - b["t0"]) / a["tE"] if a.get("tE") else None
        out["du0"] = a["u0"] - b["u0"]
        out["signed_du0"] = a["u0"] - b["u0"]
        out["sign_u0_a"] = 1 if a["u0"] >= 0 else -1
        out["sign_u0_b"] = 1 if b["u0"] >= 0 else -1
        out["sign_u0_flip"] = out["sign_u0_a"] != out["sign_u0_b"]
        out["dlogtE"] = safe_log_ratio(a.get("tE"), b.get("tE"))
        out["dlogrho"] = safe_log_ratio(a.get("rho"), b.get("rho"))
        return out

    comparisons = [
        pair("H0_2_seed(=H1_final_shared)", h0_2_seed, "reference_winner", ref_params),
        pair("H0_2_final", h0_2, "reference_winner", ref_params),
        pair("H0_1_final", h0_1, "reference_winner", ref_params),
        pair("morph_seed", morph, "reference_winner", ref_params),
    ]

    print(f"\n=== {row} (H0-limited) ===")
    print(f"morph_seed: {morph}")
    print(f"H0_1_final: {h0_1}")
    print(f"H0_2_seed (from settled H1): {h0_2_seed}")
    print(f"H0_2_final: {h0_2}")
    print(f"independent reference winner: label={ref_winner_label} chi2={ref_chi2} params={ref_params}")
    for c in comparisons:
        print(f"  {c}")

    rows_h0.append({
        "event_id": row, "category": "H0_limited",
        "morph_t0": morph["t0"], "morph_u0": morph["u0"], "morph_tE": morph["tE"], "morph_rho": morph["rho"],
        "h0_1_t0": h0_1["t0"], "h0_1_u0": h0_1["u0"], "h0_1_tE": h0_1["tE"], "h0_1_rho": h0_1["rho"], "h0_1_chi2": h0_1["chi2"],
        "h0_2_seed_t0": h0_2_seed["t0"], "h0_2_seed_u0": h0_2_seed["u0"], "h0_2_seed_tE": h0_2_seed["tE"], "h0_2_seed_rho": h0_2_seed["rho"],
        "h0_2_t0": h0_2["t0"], "h0_2_u0": h0_2["u0"], "h0_2_tE": h0_2["tE"], "h0_2_rho": h0_2["rho"], "h0_2_chi2": h0_2["chi2"],
        "ref_winner_label": ref_winner_label, "ref_chi2": ref_chi2,
        "ref_t0": ref_params["t0"] if ref_params else None, "ref_u0": ref_params["u0"] if ref_params else None,
        "ref_tE": ref_params["tE"] if ref_params else None, "ref_rho": ref_params["rho"] if ref_params else None,
        "h0_2_vs_ref_dlogtE": comparisons[1]["dlogtE"], "h0_2_vs_ref_dlogrho": comparisons[1]["dlogrho"],
        "h0_2_vs_ref_du0": comparisons[1]["du0"], "h0_2_vs_ref_dt0_over_tE": comparisons[1]["dt0_over_tE"],
        "h0_2_vs_ref_sign_u0_flip": comparisons[1]["sign_u0_flip"],
        "h0_1_vs_ref_dlogtE": comparisons[2]["dlogtE"], "h0_1_vs_ref_dlogrho": comparisons[2]["dlogrho"],
        "h0_1_vs_ref_du0": comparisons[2]["du0"], "h0_1_vs_ref_sign_u0_flip": comparisons[2]["sign_u0_flip"],
    })

rows_h1 = []
for row in H1_LIMITED:
    hp_row = h1proj.loc[row]
    r2 = ref2.loc[row]
    fp = fp_events[row]
    ref = ref_events[row]

    cand_h1 = fp["final_h1"]
    cand = {"t0": cand_h1["t0"], "u0": cand_h1["u0"], "tE": cand_h1["tE"], "rho": cand_h1["rho"],
            "piEN": cand_h1["piEN"], "piEE": cand_h1["piEE"], "chi2": cand_h1["chi2"]}

    # independent-reference H1: since the nested reference-H1 rescue never
    # fired for any of the 93 events (REFERENCE_V2_INDEPENDENT.md sec 8-9),
    # chi2_H1_ref2_independent == chi2_H1_old_reference, and its own
    # (t0,u0,tE,rho,piEN,piEE) are simply the old controlled4 H1 reference's
    # own winning-strategy params -- looked up the same way as the H0 case,
    # from the already-computed h1_controlled4 JSON (never recomputed).
    h1c4_path = os.path.join(PSJ, f"{row}_h1_controlled4.json")
    ref_h1_params = None
    if os.path.exists(h1c4_path):
        h1c4 = json.load(open(h1c4_path))
        target_label = ref["h1_final_label"]
        for fit in h1c4["controlled4"]:
            if fit["label"] == target_label:
                ref_h1_params = {"t0": fit["t0"], "u0": fit["u0"], "tE": fit["tE"], "rho": fit["rho"],
                                  "piEN": fit["piEN"], "piEE": fit["piEE"]}
                break
    ref_chi2 = r2["chi2_H1_ref2_independent"]

    piE_cand = math.sqrt(cand["piEN"] ** 2 + cand["piEE"] ** 2)
    piE_ref = math.sqrt(ref_h1_params["piEN"] ** 2 + ref_h1_params["piEE"] ** 2) if ref_h1_params else None
    angle_cand = math.degrees(math.atan2(cand["piEE"], cand["piEN"]))
    angle_ref = math.degrees(math.atan2(ref_h1_params["piEE"], ref_h1_params["piEN"])) if ref_h1_params else None

    print(f"\n=== {row} (H1-limited) ===")
    print(f"candidate H1 (official settled): {cand}")
    print(f"independent-reference H1: label={ref['h1_final_label']} chi2={ref_chi2} params={ref_h1_params}")
    print(f"piE_candidate={piE_cand:.4f} piE_reference={piE_ref} delta_piE={piE_cand - piE_ref if piE_ref else None}")
    print(f"parallax-vector angle (deg): candidate={angle_cand:.2f} reference={angle_ref}")

    rows_h1.append({
        "event_id": row, "category": "H1_limited",
        "cand_t0": cand["t0"], "cand_u0": cand["u0"], "cand_tE": cand["tE"], "cand_rho": cand["rho"],
        "cand_piEN": cand["piEN"], "cand_piEE": cand["piEE"], "cand_chi2": cand["chi2"],
        "ref_t0": ref_h1_params["t0"] if ref_h1_params else None,
        "ref_u0": ref_h1_params["u0"] if ref_h1_params else None,
        "ref_tE": ref_h1_params["tE"] if ref_h1_params else None,
        "ref_rho": ref_h1_params["rho"] if ref_h1_params else None,
        "ref_piEN": ref_h1_params["piEN"] if ref_h1_params else None,
        "ref_piEE": ref_h1_params["piEE"] if ref_h1_params else None,
        "ref_chi2": ref_chi2, "ref_label": ref["h1_final_label"],
        "delta_t0_over_tE": (cand["t0"] - ref_h1_params["t0"]) / cand["tE"] if ref_h1_params else None,
        "delta_u0": (cand["u0"] - ref_h1_params["u0"]) if ref_h1_params else None,
        "delta_logtE": safe_log_ratio(cand["tE"], ref_h1_params["tE"]) if ref_h1_params else None,
        "delta_logrho": safe_log_ratio(cand["rho"], ref_h1_params["rho"]) if ref_h1_params else None,
        "delta_piEN": (cand["piEN"] - ref_h1_params["piEN"]) if ref_h1_params else None,
        "delta_piEE": (cand["piEE"] - ref_h1_params["piEE"]) if ref_h1_params else None,
        "piE_candidate": piE_cand, "piE_reference": piE_ref,
        "delta_piE": (piE_cand - piE_ref) if piE_ref else None,
        "angle_candidate_deg": angle_cand, "angle_reference_deg": angle_ref,
        "angle_delta_deg": (angle_cand - angle_ref) if angle_ref is not None else None,
    })

df_h0 = pd.DataFrame(rows_h0)
df_h1 = pd.DataFrame(rows_h1)
out = pd.concat([df_h0, df_h1], ignore_index=True, sort=False)
out.to_csv(os.path.join(HERE, "residual_geometry.csv"), index=False)
print(f"\nWrote {len(out)} rows -> residual_geometry.csv")
