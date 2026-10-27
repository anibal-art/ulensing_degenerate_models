import os as _os_path_setup
HERE = _os_path_setup.path.dirname(_os_path_setup.path.abspath(__file__))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

SCRATCH = _os_path_setup.path.join(HERE, "results")

MODES = ["physical", "log_te", "log_rho", "log_te_rho"]

starts = pd.read_csv(f"{SCRATCH}/h0_scout_stageA_starts.csv")

# canonical id matching the oracle52 'canon' scheme: mode/anchor/rho_tag
rows = []
for _, r in starts.iterrows():
    for mode in MODES:
        rows.append({
            "catalog_row": int(r["catalog_row"]),
            "mode": mode,
            "canon": f"{mode}/{r['anchor']}/{r['rho_tag']}",
            "chi2_at_start": r["chi2_at_start"],
        })
starts52 = pd.DataFrame(rows)
assert len(starts52) == 100 * 52

# ------------------------------------------------------------------
# existing oracle52 FINAL chi2 (already computed, no new fits)
# ------------------------------------------------------------------
oracle_df = pd.read_csv(
    "/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/"
    "lsstmonts_catalog_sedighe/validation/bounds_convergence/results/"
    "gate1_oracle_diagnostics.csv"
)
d52 = oracle_df[oracle_df["source"] == "gate1_oracle_52"].copy()


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


d52["canon"] = [canon(m, s) for m, s in zip(d52["mode"], d52["strategy_id"])]

piv_final = d52.pivot_table(index="catalog_row", columns="canon", values="chi2", aggfunc="min")
oracle52_min = piv_final.min(axis=1)

# sanity: canon sets match exactly
assert set(piv_final.columns) == set(starts52["canon"].unique()), (
    set(piv_final.columns) ^ set(starts52["canon"].unique())
)

piv_start = starts52.pivot_table(index="catalog_row", columns="canon", values="chi2_at_start", aggfunc="min")
piv_start = piv_start[piv_final.columns]  # align column order

cols = list(piv_final.columns)
old_cols_mask = np.array(["/old_H0/" in c for c in cols])

# ------------------------------------------------------------------
# per-event deterministic ranking by chi2_at_start (ties broken by
# canon name, alphabetical, for full reproducibility)
# ------------------------------------------------------------------
rows_idx = piv_final.index.tolist()

topk_masks = {k: np.zeros((len(rows_idx), len(cols)), dtype=bool) for k in range(1, 11)}
winner_rank = []
spearman_rhos = []
oracle_winner_col_idx = []

for i, row in enumerate(rows_idx):
    start_vals = piv_start.loc[row].values
    final_vals = piv_final.loc[row].values

    order = sorted(range(len(cols)), key=lambda j: (start_vals[j], cols[j]))

    for k in range(1, 11):
        chosen = order[:k]
        for j in chosen:
            topk_masks[k][i, j] = True

    # rank (1-indexed) of the actual oracle winner in the start-order
    winner_j = int(np.argmin(final_vals))
    oracle_winner_col_idx.append(winner_j)
    rank_of_winner = order.index(winner_j) + 1
    winner_rank.append(rank_of_winner)

    rho_s, _ = spearmanr(start_vals, final_vals)
    spearman_rhos.append(rho_s)

winner_rank = np.array(winner_rank)
spearman_rhos = np.array(spearman_rhos)

print("=" * 100)
print("Spearman correlation (chi2_at_start rank vs final optimized chi2 rank), per event")
print("=" * 100)
print(pd.Series(spearman_rhos).describe())
print()
print("=" * 100)
print("Rank (1..52) of the eventual oracle winner, by chi2_at_start ordering")
print("=" * 100)
print(pd.Series(winner_rank).describe())
print("Fraction with winner rank <= 10:", float((winner_rank <= 10).mean()))
print("Fraction with winner rank <= 5:", float((winner_rank <= 5).mean()))
print("Fraction with winner rank == 1:", float((winner_rank == 1).mean()))

print()
print("=" * 100)
print("TOP-K SIMULATION (existing full-TRF results, no new fits)")
print("=" * 100)

final_mat = piv_final.values
oracle_arr = oracle52_min.values

for k in range(1, 11):
    mask = topk_masks[k]
    masked_final = np.where(mask, final_mat, np.inf)
    chi2_topk = masked_final.min(axis=1)
    delta = chi2_topk - oracle_arr

    n_gt_01 = int((delta > 0.1).sum())
    n_gt_1 = int((delta > 1.0).sum())
    n_gt_10 = int((delta > 10.0).sum())
    worst = float(delta.max())
    frac_within = float((delta <= 0.1).mean())

    bridge_needed = mask[:, old_cols_mask].any(axis=1)
    n_bridge = int(bridge_needed.sum())
    winner_in_topk = int((winner_rank <= k).sum())

    avg_k_actual = float(mask.sum(axis=1).mean())  # accounts for tie-driven overcount

    print(
        f"k={k:2d}  n>0.1={n_gt_01:3d} n>1={n_gt_1:3d} n>10={n_gt_10:3d} "
        f"worst={worst:12.3f} frac_within_0.1={frac_within:.2f} "
        f"winner_in_topk={winner_in_topk:3d}/100 "
        f"events_needing_bridge={n_bridge:3d}/100 "
        f"avg_actual_k(ties)={avg_k_actual:.2f}"
    )

# minimal k with zero failures
for k in range(1, 11):
    mask = topk_masks[k]
    masked_final = np.where(mask, final_mat, np.inf)
    chi2_topk = masked_final.min(axis=1)
    delta = chi2_topk - oracle_arr
    if (delta > 0.1).sum() == 0:
        print()
        print(f"MINIMUM k with ZERO failures>0.1: k={k}")
        break
else:
    print()
    print("No k in 1..10 reaches zero failures.")

df_out = pd.DataFrame({
    "catalog_row": rows_idx,
    "winner_start_rank": winner_rank,
    "spearman_start_vs_final": spearman_rhos,
})
df_out.to_csv(f"{SCRATCH}/h0_scout_stageA_summary.csv", index=False)
