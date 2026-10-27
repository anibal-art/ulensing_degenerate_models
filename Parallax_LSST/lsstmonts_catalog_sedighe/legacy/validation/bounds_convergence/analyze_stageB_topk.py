import os as _os_path_setup
HERE = _os_path_setup.path.dirname(_os_path_setup.path.abspath(__file__))

import pandas as pd
import numpy as np

SCRATCH = _os_path_setup.path.join(HERE, "results")

GATE1_FINAL18 = {
    "physical": [("old_H0", "1"), ("truth", "0.1"), ("truth", "1")],
    "log_te": [("old_H0", "truth_rho"), ("old_H0", "0.01"), ("old_H0", "0.1"),
               ("old_H0", "1"), ("truth", "0.01"), ("truth", "1")],
    "log_rho": [("old_H0", "truth_rho"), ("old_H0", "0.01"), ("old_H0", "1"),
                ("truth", "truth_rho"), ("truth", "1e-06"), ("truth", "0.0001"),
                ("truth", "0.1"), ("truth", "1")],
    "log_te_rho": [("truth", "1e-06")],
}
CANON18 = [f"{m}/{a}/{r}" for m, lst in GATE1_FINAL18.items() for a, r in lst]
OLD_H0_CANON18 = set(f"{m}/{a}/{r}" for m, lst in GATE1_FINAL18.items() for a, r in lst if a == "old_H0")

FULL_TRF_TIME = {  # measured calibration, seconds, mean over 6 representative events
    "physical": 0.2714, "log_te": 0.2060, "log_rho": 0.2619, "log_te_rho": 0.1840,
}
BRIDGE_TIME = float(np.mean(list(FULL_TRF_TIME.values())))  # one old_H0-equivalent TRF

BUDGETS = [3, 5, 10]


def canon_rho_tag(x):
    if isinstance(x, str) and x == "truth_rho":
        return "truth_rho"
    return f"{float(x):.10g}"


dfs = []
for mode in GATE1_FINAL18:
    d = pd.read_csv(f"{SCRATCH}/h0_scout_stageB_{mode}.csv", dtype={"rho_tag": str})
    dfs.append(d)
scouts = pd.concat(dfs, ignore_index=True)
scouts["rho_tag"] = scouts["rho_tag"].apply(canon_rho_tag)
scouts["canon"] = scouts["mode"] + "/" + scouts["anchor"] + "/" + scouts["rho_tag"]

# existing full-TRF (oracle52) results
oracle_df = pd.read_csv(
    "/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/"
    "lsstmonts_catalog_sedighe/validation/bounds_convergence/results/"
    "gate1_oracle_diagnostics.csv"
)


def canon(mode, sid):
    if "/" in sid:
        anchor, tag = sid.split("/", 1)
    else:
        if sid.startswith("old_H0_rho_"):
            anchor = "old_H0"; tag = sid[len("old_H0_rho_"):]
        else:
            anchor = "truth"; tag = sid[len("truth_rho_"):]
    if tag == "truth":
        tag = "truth_rho"
    if tag not in ("truth_rho", "rho_from_old_H0"):
        tag = f"{float(tag):.10g}"
    return f"{mode}/{anchor}/{tag}"


oracle_df["canon"] = [canon(m, s) for m, s in zip(oracle_df["mode"], oracle_df["strategy_id"])]
d52 = oracle_df[oracle_df["source"] == "gate1_oracle_52"]
piv52 = d52.pivot_table(index="catalog_row", columns="canon", values="chi2", aggfunc="min")
oracle52_min = piv52.min(axis=1)
piv18 = piv52[CANON18]
gate18_min = piv18.min(axis=1)

rows_100 = sorted(piv52.index.tolist())
assert len(rows_100) == 100

results = []

for budget in BUDGETS:
    sc_b = scouts[scouts["budget"] == budget]
    piv_scout_chi2 = sc_b.pivot_table(index="catalog_row", columns="canon", values="chi2_scout", aggfunc="min")
    piv_scout_status = sc_b.pivot_table(index="catalog_row", columns="canon", values="optimizer_status", aggfunc="min")
    piv_scout_wall = sc_b.pivot_table(index="catalog_row", columns="canon", values="wall_time_s", aggfunc="min")
    piv_scout_chi2 = piv_scout_chi2[CANON18]
    piv_scout_status = piv_scout_status[CANON18]
    piv_scout_wall = piv_scout_wall[CANON18]

    # per-event deterministic ranking by scout chi2
    orders = {}
    for row in rows_100:
        vals = piv_scout_chi2.loc[row].values
        order = sorted(range(18), key=lambda j: (vals[j], CANON18[j]))
        orders[row] = order

    total_scout_wall = float(piv_scout_wall.sum(axis=1).sum())  # all 18 scouts, all events, this budget

    # rank of the gate1_final18 winner, and rank of the first "adequate" (<=0.1 vs oracle52) strategy
    rank_of_best18 = []
    rank_of_first_adequate = []
    for row in rows_100:
        order = orders[row]
        final_vals = piv18.loc[row].values
        best18_idx = int(np.argmin(final_vals))
        rank_of_best18.append(order.index(best18_idx) + 1)

        adequate_idx = [j for j in range(18) if final_vals[j] - oracle52_min.loc[row] <= 0.1]
        if adequate_idx:
            r_first = min(order.index(j) for j in adequate_idx) + 1
        else:
            r_first = np.nan
        rank_of_first_adequate.append(r_first)

    rank_of_best18 = np.array(rank_of_best18, dtype=float)
    rank_of_first_adequate = np.array(rank_of_first_adequate, dtype=float)

    for k in range(1, 11):
        chi2_topk = np.full(100, np.inf)
        completion_wall = np.zeros(100)
        bridge_wall = np.zeros(100)

        for i, row in enumerate(rows_100):
            order = orders[row]
            chosen = [CANON18[j] for j in order[:k]]

            final_vals_topk = piv18.loc[row, chosen].values
            chi2_topk[i] = float(np.min(final_vals_topk))

            needs_bridge = any(c in OLD_H0_CANON18 for c in chosen)
            bridge_wall[i] = BRIDGE_TIME if needs_bridge else 0.0

            add_cost = 0.0
            for c in chosen:
                status = piv_scout_status.loc[row, c]
                if not (status is not None and status > 0):
                    mode_of_c = c.split("/")[0]
                    add_cost += FULL_TRF_TIME[mode_of_c]
            completion_wall[i] = add_cost

        delta = chi2_topk - oracle52_min.values
        n_gt_01 = int((delta > 0.1).sum())
        n_gt_1 = int((delta > 1.0).sum())
        n_gt_10 = int((delta > 10.0).sum())
        worst = float(delta.max())
        frac_within = float((delta <= 0.1).mean())

        total_wall = total_scout_wall + float(bridge_wall.sum()) + float(completion_wall.sum())

        results.append({
            "budget": budget,
            "k": k,
            "n_gt_0p1": n_gt_01,
            "n_gt_1": n_gt_1,
            "n_gt_10": n_gt_10,
            "worst": worst,
            "frac_within_0.1": frac_within,
            "median_rank_best18": float(np.median(rank_of_best18)),
            "frac_best18_in_topk": float((rank_of_best18 <= k).mean()),
            "median_rank_first_adequate": float(np.nanmedian(rank_of_first_adequate)),
            "frac_adequate_in_topk": float((rank_of_first_adequate <= k).mean()),
            "total_wall_time_s_100events": total_wall,
            "scout_wall_s": total_scout_wall,
            "bridge_wall_s": float(bridge_wall.sum()),
            "completion_wall_s": float(completion_wall.sum()),
        })

out = pd.DataFrame(results)
pd.set_option("display.width", 220)
print(out.to_string(index=False))
out.to_csv(f"{SCRATCH}/h0_stageB_topk_results.csv", index=False)

print()
print("=" * 100)
print("Zero-failure combinations (n_gt_0p1==0):")
print("=" * 100)
zf = out[out["n_gt_0p1"] == 0]
print(zf.to_string(index=False) if len(zf) else "NONE")

print()
print("=" * 100)
print("Reference costs")
print("=" * 100)
print("current B2 adaptive expected cost (approx, from earlier B2 work): base k=6 + escalate(gate1_final18=18) with bridge -> up to 19 TRF on escalation, ~7 on non-escalation")
print("fixed robust cost = 19 TRF (18 downstream + 1 bridge)")
avg_full_trf = float(np.mean(list(FULL_TRF_TIME.values())))
print(f"avg measured full-TRF wall time per fit: {avg_full_trf:.4f}s -> 19 TRF ~ {19*avg_full_trf:.2f}s per event -> x100 events = {19*avg_full_trf*100:.1f}s")
