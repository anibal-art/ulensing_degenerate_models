"""
Audit the 352 new morphology TRF fits (88 events x 4 modes), combine
with the existing 48 (12 events x 4 modes) into the final 400-row
morphology100 dataset, and run the mandatory checks before any
coverage/MILP analysis is trusted.
"""
import pandas as pd
import numpy as np

MODES = ["physical", "log_te", "log_rho", "log_te_rho"]
CRITICAL_12 = [62786, 83189, 567800, 579320, 82728, 574423,
               557860, 558189, 567365, 35927, 563210, 79501]

truth_feat = pd.read_csv("results/b1c_event_features.csv")
ALL_100 = sorted(truth_feat["catalog_row"].unique().tolist())
REMAINING_88 = [r for r in ALL_100 if r not in CRITICAL_12]
assert len(ALL_100) == 100 and len(REMAINING_88) == 88

frames = []
for mode in MODES:
    d12 = pd.read_csv(f"results/h0_morphology_trf_{mode}.csv")
    d88 = pd.read_csv(f"results/h0_morphology_trf_remaining88_{mode}.csv")
    frames.append(d12)
    frames.append(d88)

full = pd.concat(frames, ignore_index=True)
full.to_csv("results/h0_morphology_trf_all100.csv", index=False)

print("=" * 100)
print("AUDIT: 400/400 morphology fits")
print("=" * 100)
print("total rows:", len(full), "expected 400:", len(full) == 400)

n_events = full["catalog_row"].nunique()
print("unique events:", n_events, "expected 100:", n_events == 100)

for mode in MODES:
    n = len(full[full["mode"] == mode])
    print(f"  mode={mode}: n={n} expected=100 {'OK' if n==100 else 'MISMATCH'}")

dup = full.duplicated(subset=["catalog_row", "mode"]).sum()
print("duplicated (catalog_row,mode):", dup, "expected 0")

expected_pairs = {(r, m) for r in ALL_100 for m in MODES}
actual_pairs = set(zip(full["catalog_row"], full["mode"]))
missing = expected_pairs - actual_pairs
extra = actual_pairs - expected_pairs
print("missing combinations:", len(missing))
print("extra combinations:", len(extra))
if missing:
    print("  sample missing:", list(missing)[:10])

# cross-check: for each row, the init_t0/u0/tE/rho used must match the
# frozen rank-1 morphology candidate for that event exactly.
cand = pd.read_csv("results/morphology_M3_M4_candidates.csv")
c1 = cand[(cand["invertible"] == True) & (cand["candidate_rank"] == 1)].set_index("catalog_row")

mismatch_init = 0
for _, r in full.iterrows():
    row = int(r["catalog_row"])
    exp = c1.loc[row]
    if not (
        np.isclose(r["init_t0"], exp["t0_morph"])
        and np.isclose(r["init_u0"], exp["u0_morph"])
        and np.isclose(r["init_tE"], exp["tE_morph"])
        and np.isclose(r["init_rho"], exp["rho_morph"])
    ):
        mismatch_init += 1
print("rows whose init vector does NOT match the frozen rank-1 morphology candidate:", mismatch_init)

print("status value_counts (all 400):")
print(full["status"].value_counts())

print()
print("PASS" if (len(full) == 400 and n_events == 100 and dup == 0 and len(missing) == 0
                  and len(extra) == 0 and mismatch_init == 0
                  and (full["status"] == "success").all())
      else "FAIL -- see above")
