#!/usr/bin/env python3
"""
Small independent LRT comparison: Delta_chi2_LRT_simple (the 2-fit
architecture) vs Delta_chi2_LRT_reference (the existing robust/
multistart architecture, for the events where it is known to
complete -- see FEASIBILITY_AUDIT.md), for the H1-generated regime.

Also summarizes the H0-generated regime on its own terms (no robust
reference of that kind exists in this project's history; reported as
the null-injection LRT distribution itself, which is the quantity
that actually matters for H0-calibration purposes).
"""
import json
import numpy as np
import pandas as pd

with open("results/two_fit_h1generated_9events.json") as f:
    two_fit = json.load(f)["events"]

with open("results/reference_robust_9events.json") as f:
    ref = json.load(f)["events"]

ref_by_row = {e["catalog_row"]: e for e in ref}

rows = []
for e in two_fit:
    row = e["catalog_row"]
    r = ref_by_row.get(row)
    if r is None:
        continue
    delta_simple = e["delta_chi2_lrt"]
    delta_ref = r["h0_final_chi2"] - r["h1_final_chi2"]
    rows.append({
        "catalog_row": row,
        "delta_chi2_lrt_simple": delta_simple,
        "delta_chi2_lrt_reference": delta_ref,
        "diff": delta_simple - delta_ref,
        "abs_diff": abs(delta_simple - delta_ref),
        "chi2_h0_simple": e["chi2_h0"], "chi2_h0_reference": r["h0_final_chi2"],
        "chi2_h1_simple": e["chi2_h1"], "chi2_h1_reference": r["h1_final_chi2"],
        "h0_status_simple": e["h0"]["status"], "h1_status_simple": e["h1"]["status"],
        "h0_optimality_simple": e["h0"].get("optimizer_optimality"),
        "h1_optimality_simple": e["h1"].get("optimizer_optimality"),
    })

df = pd.DataFrame(rows)
df.to_csv("results/lrt_comparison_h1generated_9events.csv", index=False)

LRT_THRESHOLD = 9.0  # ILLUSTRATIVE placeholder only (~chi2(2), delta_k=2,
# matching configs' lrt.delta_k=2), NOT a finalized production threshold --
# the actual threshold is a separate decision, to be set from the H0
# calibration sample's alpha/tail-precision requirements per the user's
# own instruction. Used here only to report how many events change
# classification, not as a proposed operating point.

df["classify_simple"] = df["delta_chi2_lrt_simple"] >= LRT_THRESHOLD
df["classify_reference"] = df["delta_chi2_lrt_reference"] >= LRT_THRESHOLD
df["reclassified"] = df["classify_simple"] != df["classify_reference"]

print("=" * 100)
print(f"H1-GENERATED REGIME: n={len(df)} (paired: same events, simplified vs robust-reference)")
print(df[["catalog_row", "delta_chi2_lrt_simple", "delta_chi2_lrt_reference", "diff"]]
      .to_string(index=False))
print()
print(f"bias (mean diff)      = {df['diff'].mean():.4f}")
print(f"median diff           = {df['diff'].median():.4f}")
print(f"p90 abs diff          = {df['abs_diff'].quantile(0.90):.4f}")
print(f"p95 abs diff          = {df['abs_diff'].quantile(0.95):.4f}")
print(f"max abs diff          = {df['abs_diff'].max():.4f}  (row {df.loc[df['abs_diff'].idxmax(),'catalog_row']})")
print(f"fraction reclassified @ threshold={LRT_THRESHOLD}: "
      f"{df['reclassified'].sum()}/{len(df)} = {df['reclassified'].mean():.1%}")
print()
print("optimizer status (simple policy):")
print(df[["h0_status_simple", "h1_status_simple"]].apply(pd.Series.value_counts))

# ---- H0-generated regime: no robust reference of this kind exists yet ----
with open("results/two_fit_h0generated_5events.json") as f:
    h0gen = json.load(f)["events"]

h0gen_df = pd.DataFrame([{
    "catalog_row": e["catalog_row"], "delta_chi2_lrt_simple": e["delta_chi2_lrt"],
    "chi2_h0": e["chi2_h0"], "chi2_h1": e["chi2_h1"],
    "h0_status": e["h0"]["status"], "h1_status": e["h1"]["status"],
} for e in h0gen])
h0gen_df.to_csv("results/lrt_h0generated_5events.csv", index=False)

print()
print("=" * 100)
print(f"H0-GENERATED REGIME (null injection): n={len(h0gen_df)} "
      "(no pre-existing robust reference for this regime -- see report)")
print(h0gen_df.to_string(index=False))
print()
print(f"mean Delta_chi2_LRT   = {h0gen_df['delta_chi2_lrt_simple'].mean():.4f}")
print(f"median                = {h0gen_df['delta_chi2_lrt_simple'].median():.4f}")
print(f"min/max               = {h0gen_df['delta_chi2_lrt_simple'].min():.4f} / "
      f"{h0gen_df['delta_chi2_lrt_simple'].max():.4f}")
n_false_positive = int((h0gen_df["delta_chi2_lrt_simple"] >= LRT_THRESHOLD).sum())
print(f"false positives @ threshold={LRT_THRESHOLD}: {n_false_positive}/{len(h0gen_df)}")
