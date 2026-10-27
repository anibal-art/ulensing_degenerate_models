#!/usr/bin/env python3
"""
Blocker-2 analysis: classification disagreement between the final
policy and the old_final-free controlled4 robust reference, on the
NEW independent H1-basin-validation sample. Report:
  - classification disagreement vs a threshold grid (esp. 4-9);
  - false-detection-direction disagreements (simple>=thr, ref<thr);
  - opposite-direction disagreements (simple<thr, ref>=thr);
  - basin-gap (diff) distribution;
  - fraction of CLEAN (unflagged) events with material disagreement,
    with a binomial CI.
"""
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import beta

ap = argparse.ArgumentParser()
ap.add_argument("--final-policy-json", required=True)
ap.add_argument("--reference-json", required=True)
ap.add_argument("--out-csv", required=True)
ap.add_argument("--material-diff-threshold", type=float, default=5.0,
                help="abs(diff) above which a disagreement is called 'material' even if not threshold-crossing")
args = ap.parse_args()

fp = json.load(open(args.final_policy_json))["events"]
ref = {e["catalog_row"]: e for e in json.load(open(args.reference_json))["events"]}

rows = []
for e in fp:
    row = e["catalog_row"]
    r = ref.get(row)
    if r is None or e.get("delta_chi2_lrt") is None:
        continue
    d_simple = e["delta_chi2_lrt"]
    d_ref = r["h0_final_chi2"] - r["h1_final_chi2"]
    flagged = bool(e["sanity_flags"].get("h0", {}).get("flagged")) or bool(e["sanity_flags"].get("h1", {}).get("flagged"))
    rows.append(dict(row=row, n_trf=e["n_total_trf"], rescue=e["rescue_ran"],
                      cont_h0=e["continuation_ran_h0"], cont_h1=e["continuation_ran_h1"],
                      d_simple=d_simple, d_ref=d_ref, diff=d_simple - d_ref, abs_diff=abs(d_simple - d_ref),
                      flagged=flagged))
df = pd.DataFrame(rows)
df.to_csv(args.out_csv, index=False)

n = len(df)
n_excl = len(fp) - n
print(f"n_valid={n} (excluded: {n_excl} estimator-failure/no-reference)")
print(f"flagged (chi2-sanity, either model): {df['flagged'].sum()}/{n}")
print()
print("basin-gap (diff) distribution, ALL events:")
print(f"  mean={df['diff'].mean():.3f} median={df['diff'].median():.3f} std={df['diff'].std():.3f}")
print(f"  p90 abs={df['abs_diff'].quantile(.9):.3f} p95 abs={df['abs_diff'].quantile(.95):.3f} "
      f"max abs={df['abs_diff'].max():.3f} (row {df.loc[df['abs_diff'].idxmax(),'row']})")

clean = df[~df["flagged"]]
print()
print(f"basin-gap distribution, CLEAN (unflagged) n={len(clean)}:")
print(f"  mean={clean['diff'].mean():.3f} median={clean['diff'].median():.3f}")
print(f"  p90 abs={clean['abs_diff'].quantile(.9):.3f} p95 abs={clean['abs_diff'].quantile(.95):.3f} "
      f"max abs={clean['abs_diff'].max():.3f}")

print()
print("=" * 100)
print("Classification disagreement vs threshold grid (clean/unflagged events only)")
print("=" * 100)
grid = [2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 25]
summary_rows = []
for thr in grid:
    cls_s = clean["d_simple"] >= thr
    cls_r = clean["d_ref"] >= thr
    false_detect = ((cls_s) & (~cls_r)).sum()   # simple says detect, reference says no -- dangerous direction
    opposite = ((~cls_s) & (cls_r)).sum()        # simple says no, reference says detect -- conservative direction
    total_dis = false_detect + opposite
    n_clean = len(clean)
    # binomial CI (Clopper-Pearson) on the material-disagreement fraction
    k = total_dis
    if k == 0:
        lo = 0.0
    else:
        lo = beta.ppf(0.025, k, n_clean - k + 1)
    if k == n_clean:
        hi = 1.0
    else:
        hi = beta.ppf(0.975, k + 1, n_clean - k)
    frac = k / n_clean
    print(f"threshold={thr:5.1f}: false_detect(simple>=thr,ref<thr)={false_detect:2d}  "
          f"opposite(simple<thr,ref>=thr)={opposite:2d}  total={total_dis:2d}/{n_clean} "
          f"({frac:.1%})  95% Clopper-Pearson CI=[{lo:.1%}, {hi:.1%}]")
    summary_rows.append(dict(threshold=thr, false_detect=false_detect, opposite=opposite,
                              total_disagree=total_dis, n_clean=n_clean, frac=frac, ci_lo=lo, ci_hi=hi))

pd.DataFrame(summary_rows).to_csv(args.out_csv.replace(".csv", "_threshold_grid.csv"), index=False)

print()
print("=" * 100)
print(f"Material (abs diff > {args.material_diff_threshold}) disagreement fraction, clean events only "
      "(threshold-independent notion of 'material'):")
material = clean[clean["abs_diff"] > args.material_diff_threshold]
k = len(material)
nn = len(clean)
lo = 0.0 if k == 0 else beta.ppf(0.025, k, nn - k + 1)
hi = 1.0 if k == nn else beta.ppf(0.975, k + 1, nn - k)
print(f"  {k}/{nn} = {k/nn:.1%}  95% CI=[{lo:.1%},{hi:.1%}]")
print("  rows:", material["row"].tolist())
