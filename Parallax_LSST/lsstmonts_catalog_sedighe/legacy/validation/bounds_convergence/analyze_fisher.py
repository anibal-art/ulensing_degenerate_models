import os as _os_path_setup
HERE = _os_path_setup.path.dirname(_os_path_setup.path.abspath(__file__))

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from itertools import combinations

SCRATCH = _os_path_setup.path.join(HERE, "results")


def zero_false_safe(pred, unsafe, higher_is_worse):
    """
    Returns (threshold, stop_mask, stop_fraction) for the direction
    given. 'stop' events must contain ZERO unsafe events.
    """
    pred = np.asarray(pred, dtype=float)
    unsafe = np.asarray(unsafe, dtype=bool)
    valid = np.isfinite(pred)
    if higher_is_worse:
        unsafe_vals = pred[unsafe & valid]
        thr = float(unsafe_vals.min()) if len(unsafe_vals) else np.inf
        stop = valid & (pred < thr)
    else:
        unsafe_vals = pred[unsafe & valid]
        thr = float(unsafe_vals.max()) if len(unsafe_vals) else -np.inf
        stop = valid & (pred > thr)
    assert not np.any(unsafe & stop), "zero-false-safe violated"
    return thr, stop, float(stop.mean())


def predictor_report(name, pred, unsafe, n_total=100):
    pred = np.asarray(pred, dtype=float)
    unsafe = np.asarray(unsafe, dtype=bool)
    valid = np.isfinite(pred)
    n_nan = int((~valid).sum())
    if valid.sum() < 3 or len(set(unsafe[valid])) < 2:
        return {"predictor": name, "n_nan": n_nan, "AUC": np.nan, "note": "degenerate"}

    auc = roc_auc_score(unsafe[valid].astype(int), pred[valid])
    higher_is_worse = auc >= 0.5
    eff_auc = auc if higher_is_worse else 1 - auc

    thr, stop, frac = zero_false_safe(pred, unsafe, higher_is_worse)

    safe_vals = pred[valid & ~unsafe]
    unsafe_vals = pred[valid & unsafe]

    return {
        "predictor": name,
        "n_nan": n_nan,
        "AUC_raw": auc,
        "AUC_effective": eff_auc,
        "direction": "high=unsafe" if higher_is_worse else "high=safe",
        "threshold": thr,
        "stop_fraction": frac,
        "n_stop": int(stop.sum()),
        "safe_median": float(np.median(safe_vals)) if len(safe_vals) else np.nan,
        "unsafe_median": float(np.median(unsafe_vals)) if len(unsafe_vals) else np.nan,
        "_stop_mask": stop,
    }


# ====================================================================
# H0
# ====================================================================
h0 = pd.read_csv(f"{SCRATCH}/fisher_h0_results.csv")
h0["unsafe1"] = (h0["chi2_step1"] - h0["chi2_oracle52"]) > 0.1
h0["unsafe2"] = (h0[["chi2_step1", "chi2_step2"]].min(axis=1) - h0["chi2_oracle52"]) > 0.1

print("=" * 100)
print(f"H0 STEP 1  (n=100, unsafe1={h0['unsafe1'].sum()})")
print("=" * 100)

PREDICTORS_1 = {
    "cond_number_step1": h0["cond_number_step1"],
    "min_eig_step1": h0["min_eig_step1"],
    "logdet_step1": h0["logdet_step1"],
    "max_abs_corr_step1": h0["max_abs_corr_step1"],
    "sigma_t0overtE_step1": h0["sigma_t0overtE_step1"],
    "sigma_u0_step1": h0["sigma_u0_step1"],
    "sigma_logtE_step1": h0["sigma_logtE_step1"],
    "sigma_logrho_step1": h0["sigma_logrho_step1"],
    "abs_weakvec_t0_step1": h0["weakvec_t0_step1"].abs(),
    "abs_weakvec_u0_step1": h0["weakvec_u0_step1"].abs(),
    "abs_weakvec_logtE_step1": h0["weakvec_logtE_step1"].abs(),
    "abs_weakvec_logrho_step1": h0["weakvec_logrho_step1"].abs(),
}

reports_1 = []
for name, series in PREDICTORS_1.items():
    r = predictor_report(name, series, h0["unsafe1"])
    reports_1.append(r)
    print(f"{name:28s} AUC={r.get('AUC_effective', np.nan):.3f} dir={r.get('direction','-'):11s} "
          f"stop_frac={r.get('stop_fraction', np.nan):.2f} n_stop={r.get('n_stop','-')} "
          f"safe_med={r.get('safe_median', np.nan):.4g} unsafe_med={r.get('unsafe_median', np.nan):.4g} "
          f"n_nan={r.get('n_nan')}")

best_step1 = max([r for r in reports_1 if not np.isnan(r.get('stop_fraction', np.nan))],
                  key=lambda r: r['stop_fraction'])
print()
print("BEST single predictor after step 1:", best_step1['predictor'],
      "stop_fraction=", best_step1['stop_fraction'])

print()
print("=" * 100)
print(f"H0 STEP 2  (n=100, unsafe2={h0['unsafe2'].sum()})")
print("=" * 100)

PREDICTORS_2 = {
    "cond_number_step2": h0["cond_number_step2"],
    "min_eig_step2": h0["min_eig_step2"],
    "logdet_step2": h0["logdet_step2"],
    "max_abs_corr_step2": h0["max_abs_corr_step2"],
    "sigma_t0overtE_step2": h0["sigma_t0overtE_step2"],
    "sigma_u0_step2": h0["sigma_u0_step2"],
    "sigma_logtE_step2": h0["sigma_logtE_step2"],
    "sigma_logrho_step2": h0["sigma_logrho_step2"],
    "delta_chi2_12": h0["delta_chi2_12"],
    "D12_sq": h0["D12_sq"],
    "weak_eigvec_alignment": h0["weak_eigvec_alignment"],
    "max_cond_1_2": h0[["cond_number_step1", "cond_number_step2"]].max(axis=1),
    "min_of_min_eig_1_2": h0[["min_eig_step1", "min_eig_step2"]].min(axis=1),
}

reports_2 = []
for name, series in PREDICTORS_2.items():
    r = predictor_report(name, series, h0["unsafe2"])
    reports_2.append(r)
    print(f"{name:28s} AUC={r.get('AUC_effective', np.nan):.3f} dir={r.get('direction','-'):11s} "
          f"stop_frac={r.get('stop_fraction', np.nan):.2f} n_stop={r.get('n_stop','-')} "
          f"safe_med={r.get('safe_median', np.nan):.4g} unsafe_med={r.get('unsafe_median', np.nan):.4g} "
          f"n_nan={r.get('n_nan')}")

valid_2 = [r for r in reports_2 if not np.isnan(r.get('stop_fraction', np.nan))]
valid_2.sort(key=lambda r: -r['stop_fraction'])
print()
print("TOP individual step-2 predictors by stop_fraction:")
for r in valid_2[:5]:
    print(" ", r['predictor'], r['stop_fraction'])

print()
print("Pairwise AND / OR combinations (top 5 predictors)")
top5 = valid_2[:5]
for ra, rb in combinations(top5, 2):
    stop_and = ra['_stop_mask'] & rb['_stop_mask']
    stop_or = ra['_stop_mask'] | rb['_stop_mask']
    assert not np.any(h0['unsafe2'].values & stop_or), "OR combo violates zero-false-safe"
    print(f"  {ra['predictor']} & {rb['predictor']}: AND={stop_and.mean():.2f}  OR={stop_or.mean():.2f}")

best_or = max(
    (ra['_stop_mask'] | rb['_stop_mask'] for ra, rb in combinations(top5, 2)),
    key=lambda m: m.mean(),
)
print()
print("BEST 2-combo OR stop fraction after step 2:", best_or.mean())

# ====================================================================
# H1
# ====================================================================
print()
print("=" * 100)
h1 = pd.read_csv(f"{SCRATCH}/fisher_h1_results.csv")
print(f"H1 TRUTH-TRF  (n=100, unsafe={h1['unsafe_H1_truth'].sum()})")
print("=" * 100)

PREDICTORS_H1 = {
    "cond_number_full": h1["cond_number_full"],
    "min_eig_full": h1["min_eig_full"],
    "max_abs_corr_full": h1["max_abs_corr_full"],
    "min_eig_schur": h1["min_eig_schur"],
    "max_eig_schur": h1["max_eig_schur"],
    "cond_number_schur": h1["cond_number_schur"],
    "logdet_schur": h1["logdet_schur"],
    "sigma_major_piE": h1["sigma_major_piE"],
    "sigma_minor_piE": h1["sigma_minor_piE"],
    "corr_piEN_piEE": h1["corr_piEN_piEE"].abs(),
    "backbone_parallax_coupling": h1["backbone_parallax_coupling"],
}

reports_h1 = []
for name, series in PREDICTORS_H1.items():
    r = predictor_report(name, series, h1["unsafe_H1_truth"])
    reports_h1.append(r)
    print(f"{name:28s} AUC={r.get('AUC_effective', np.nan):.3f} dir={r.get('direction','-'):11s} "
          f"stop_frac={r.get('stop_fraction', np.nan):.2f} n_stop={r.get('n_stop','-')} "
          f"safe_med={r.get('safe_median', np.nan):.4g} unsafe_med={r.get('unsafe_median', np.nan):.4g} "
          f"n_nan={r.get('n_nan')}")

valid_h1 = [r for r in reports_h1 if not np.isnan(r.get('stop_fraction', np.nan))]
valid_h1.sort(key=lambda r: -r['stop_fraction'])
print()
print("TOP H1 predictors by stop_fraction:")
for r in valid_h1[:5]:
    print(" ", r['predictor'], r['stop_fraction'], "AUC_eff=", r['AUC_effective'])

top5_h1 = valid_h1[:5]
print()
print("Pairwise AND/OR (top 5 H1 predictors)")
for ra, rb in combinations(top5_h1, 2):
    stop_and = ra['_stop_mask'] & rb['_stop_mask']
    stop_or = ra['_stop_mask'] | rb['_stop_mask']
    print(f"  {ra['predictor']} & {rb['predictor']}: AND={stop_and.mean():.2f}  OR={stop_or.mean():.2f}")
