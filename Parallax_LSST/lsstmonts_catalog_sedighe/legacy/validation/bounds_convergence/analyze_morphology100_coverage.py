import pandas as pd
import numpy as np

MODES = ["physical", "log_te", "log_rho", "log_te_rho"]
CRITICAL_12 = [62786, 83189, 567800, 579320, 82728, 574423,
               557860, 558189, 567365, 35927, 563210, 79501]

full = pd.read_csv("results/h0_morphology_trf_all100.csv")

oracle_df = pd.read_csv("results/gate1_oracle_diagnostics.csv")


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
oracle_min = piv52.min(axis=1)

full["oracle_min"] = full["catalog_row"].map(oracle_min)
full["delta"] = full["chi2"] - full["oracle_min"]
full["covers"] = full["delta"] <= 0.1
full["critical"] = full["catalog_row"].isin(CRITICAL_12)

full.to_csv("results/morphology100_delta.csv", index=False)

print("=" * 100)
print("PER-MODE COVERAGE, 100 events")
print("=" * 100)
for mode in MODES:
    sub = full[full["mode"] == mode]
    d = sub["delta"]
    print(f"{mode:12s} N_covered={int(sub['covers'].sum()):3d}/100  "
          f"N(>0.1)={int((d>0.1).sum()):3d} N(>1)={int((d>1).sum()):3d} "
          f"N(>10)={int((d>10).sum()):3d} worst={d.max():.3f}")

print()
print("=" * 100)
print("UNION / UNIQUE / REDUNDANCY (100 events)")
print("=" * 100)
pivot = full.pivot(index="catalog_row", columns="mode", values="covers")[MODES]
pivot["n_modes_covering"] = pivot[MODES].sum(axis=1)
pivot["any_covered"] = pivot[MODES].any(axis=1)
print("Union coverage (>=1 mode):", int(pivot["any_covered"].sum()), "/100")
for mode in MODES:
    unique_to_mode = pivot[(pivot[mode]) & (pivot["n_modes_covering"] == 1)]
    print(f"  unique to {mode}: {len(unique_to_mode)} events -> {sorted(unique_to_mode.index.tolist())}")
print("redundant (covered by >=2 modes):", int((pivot["n_modes_covering"] >= 2).sum()))

print()
print("=" * 100)
print("SPLIT: 12 critical vs 88 rest")
print("=" * 100)
for label, rows in [("critical(12)", CRITICAL_12), ("rest(88)", [r for r in pivot.index if r not in CRITICAL_12])]:
    sub = pivot.loc[rows]
    print(f"{label}: union covered = {int(sub['any_covered'].sum())}/{len(rows)}")
    for mode in MODES:
        print(f"    {mode}: {int(sub[mode].sum())}/{len(rows)}")

print()
print("=" * 100)
print("Morphology quality (rank-1 candidate) distribution, descriptive only")
print("=" * 100)
cand = pd.read_csv("results/morphology_M3_M4_candidates.csv")
c1 = cand[(cand["invertible"] == True) & (cand["candidate_rank"] == 1)].set_index("catalog_row")
pivot["mismatch"] = c1["mismatch"]
print(pivot.groupby("any_covered")["mismatch"].describe())

pivot.to_csv("results/morphology100_coverage_matrix.csv")
print()
print("DONE -> results/morphology100_delta.csv, results/morphology100_coverage_matrix.csv")
