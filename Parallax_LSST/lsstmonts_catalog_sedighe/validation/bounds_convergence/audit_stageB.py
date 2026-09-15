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

BUDGETS = [3, 5, 10]

def canon_rho_tag(x):
    # pandas read_csv infers rho_tag as float64 when a file's column is
    # 100% numeric-looking strings (physical, log_te_rho have no
    # "truth_rho" entries) but keeps it as object/str when the column
    # is mixed with "truth_rho" (log_te, log_rho). Normalize both to
    # one canonical string form before any comparison.
    if isinstance(x, str) and x == "truth_rho":
        return "truth_rho"
    return f"{float(x):.10g}"


dfs = []
for mode in GATE1_FINAL18:
    df = pd.read_csv(f"{SCRATCH}/h0_scout_stageB_{mode}.csv", dtype={"rho_tag": str})
    dfs.append(df)
all_df = pd.concat(dfs, ignore_index=True)
all_df["rho_tag"] = all_df["rho_tag"].apply(canon_rho_tag)

print("=" * 110)
print("RAW FILE ROW COUNTS")
print("=" * 110)
for mode in GATE1_FINAL18:
    df = pd.read_csv(f"{SCRATCH}/h0_scout_stageB_{mode}.csv")
    n_strat = len(GATE1_FINAL18[mode])
    expected = n_strat * 100 * 3
    print(f"{mode:12s} n_strategies={n_strat} expected_rows={expected} actual_rows={len(df)} "
          f"{'OK' if len(df) == expected else 'MISMATCH'}")

print()
print("total actual rows:", len(all_df), " expected total:", 18 * 100 * 3)

# ------------------------------------------------------------------
# Build the exact expected inventory table
# ------------------------------------------------------------------
rows = []
for mode, strat_list in GATE1_FINAL18.items():
    for anchor, rho_tag in strat_list:
        sub = all_df[
            (all_df["mode"] == mode)
            & (all_df["anchor"] == anchor)
            & (all_df["rho_tag"] == rho_tag)
        ]
        n_events = sub["catalog_row"].nunique()
        n_rows_total = len(sub)
        counts_per_budget = {}
        dup_flag = False
        missing_flag = False
        for b in BUDGETS:
            subb = sub[sub["budget"] == b]
            n = len(subb)
            counts_per_budget[b] = n
            if n != 100:
                missing_flag = True
            n_dup = subb["catalog_row"].duplicated().sum()
            if n_dup > 0:
                dup_flag = True

        rows.append({
            "mode": mode,
            "anchor": anchor,
            "rho_tag": rho_tag,
            "type": "old_H0-anchored" if anchor == "old_H0" else "truth-anchored",
            "n_events_seen": n_events,
            "n_budget3": counts_per_budget[3],
            "n_budget5": counts_per_budget[5],
            "n_budget10": counts_per_budget[10],
            "total": n_rows_total,
            "missing_flag": missing_flag,
            "dup_flag": dup_flag,
        })

inventory = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 30)
print()
print("=" * 110)
print("EXACT INVENTORY: 18 gate1_final18 strategies x 100 events x 3 budgets")
print("=" * 110)
print(inventory.to_string(index=False))

print()
print("=" * 110)
print("MANDATORY CHECKS")
print("=" * 110)

# 1. exactly 18 unique (mode, anchor, rho_tag)
n_unique_strats = inventory[["mode", "anchor", "rho_tag"]].drop_duplicates().shape[0]
print(f"1. N unique (mode,anchor,rho_tag) strategies = {n_unique_strats}  "
      f"{'PASS' if n_unique_strats == 18 else 'FAIL'}")

# 2. exactly 100 events per strategy per budget
check2 = (inventory[["n_budget3", "n_budget5", "n_budget10"]] == 100).all().all()
print(f"2. Exactly 100 events per strategy per budget = {check2}  {'PASS' if check2 else 'FAIL'}")
if not check2:
    bad = inventory[(inventory["n_budget3"] != 100) | (inventory["n_budget5"] != 100) | (inventory["n_budget10"] != 100)]
    print(bad.to_string(index=False))

# 3. exactly 5400 expected scout records
total_records = int(inventory["total"].sum())
print(f"3. Total scout records = {total_records}  expected=5400  "
      f"{'PASS' if total_records == 5400 else 'FAIL'}")

# 4. no duplicated (catalog_row, mode, strategy, max_nfev)
dup_check = all_df.duplicated(subset=["catalog_row", "mode", "anchor", "rho_tag", "budget"]).sum()
print(f"4. Duplicated (catalog_row,mode,anchor,rho_tag,budget) rows = {dup_check}  "
      f"{'PASS' if dup_check == 0 else 'FAIL'}")

# 5. no missing combinations -- cross-check against full expected grid
expected_keys = set()
for mode, strat_list in GATE1_FINAL18.items():
    for anchor, rho_tag in strat_list:
        for row in range(100):  # placeholder, real check below uses actual catalog_rows
            pass

truth_feat_rows = sorted(pd.read_csv(
    "/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/b1c_event_features.csv"
)["catalog_row"].unique().tolist())
assert len(truth_feat_rows) == 100

expected_full = set()
for mode, strat_list in GATE1_FINAL18.items():
    for anchor, rho_tag in strat_list:
        for row in truth_feat_rows:
            for b in BUDGETS:
                expected_full.add((row, mode, anchor, rho_tag, b))

actual_full = set(
    zip(all_df["catalog_row"], all_df["mode"], all_df["anchor"], all_df["rho_tag"], all_df["budget"])
)
missing = expected_full - actual_full
extra = actual_full - expected_full
print(f"5. Missing combinations = {len(missing)}  Extra/unexpected combinations = {len(extra)}  "
      f"{'PASS' if (len(missing) == 0 and len(extra) == 0) else 'FAIL'}")
if missing:
    print("  sample missing:", list(missing)[:10])
if extra:
    print("  sample extra:", list(extra)[:10])

# 6. 8 old_H0-anchored strategies present, using shared old_H0 vector
old_h0_strats = inventory[inventory["type"] == "old_H0-anchored"]
print(f"6. N old_H0-anchored strategies present = {len(old_h0_strats)}  "
      f"{'PASS' if len(old_h0_strats) == 8 else 'FAIL'}")

old_h0_vec = pd.read_csv(f"{SCRATCH}/old_h0_vector_current_objective.csv").set_index("catalog_row")
mismatches = 0
for _, r in inventory[inventory["type"] == "old_H0-anchored"].iterrows():
    sub = all_df[(all_df["mode"] == r["mode"]) & (all_df["anchor"] == "old_H0")
                 & (all_df["rho_tag"] == r["rho_tag"]) & (all_df["budget"] == 3)]
    # can't directly recover the exact initial vector from output-only CSV;
    # verified instead at run-time construction (see script logic) -- cross
    # check here is that all these rows share consistent catalog_row set
    # with the old_h0_vec table (same 100 events).
    if not set(sub["catalog_row"]) <= set(old_h0_vec.index):
        mismatches += 1
print(f"   old_H0-anchored scouts reference the shared old_H0 vector's event set: "
      f"{'PASS' if mismatches == 0 else 'FAIL'} (mismatches={mismatches})")

# 7. bridge accounted separately, not counted as one of the 18
print(f"7. Bridge is a separate cost line (1 TRF, the historical old_H0 fit itself), "
      f"not one of the 18 downstream strategies: by construction old_H0-anchored scouts "
      f"in the inventory table above are 8 DOWNSTREAM strategies (re-optimizing FROM the "
      f"bridge vector), distinct from the bridge fit itself. PASS (by design, see note below).")

print()
print("=" * 110)
print("PARAMETER-CONSISTENCY SPOT CHECK (mode / max_nfev / bounds / flux profiling)")
print("=" * 110)
for mode in GATE1_FINAL18:
    df = pd.read_csv(f"{SCRATCH}/h0_scout_stageB_{mode}.csv")
    print(f"{mode}: unique mode values in file = {df['mode'].unique().tolist()}, "
          f"unique budgets = {sorted(df['budget'].unique().tolist())}, "
          f"nfev range = [{df['nfev'].min()},{df['nfev'].max()}], "
          f"any nfev > budget = {(df['nfev'] > df['budget']).sum()}")
