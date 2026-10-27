#!/usr/bin/env python3
"""
Demonstration 1 analysis: recall-vs-activation for each production-safe
diagnostic (and a small set of interpretable combinations), on the
93-event clean development population. See
BASIN_RISK_TRIGGER_DEVELOPMENT.md for the full write-up; this script
only computes numbers.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import beta

SEED = 20260915
N_BOOTSTRAP = 2000
ACTIVATION_BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25]

# ---------------------------------------------------------------------
# Direction declarations -- fixed BEFORE any ranking/recall computation.
# "high" means larger raw value is more suspicious; "low" means smaller
# raw value is more suspicious. score = value (high) or -value (low),
# so "higher score" always means "more suspicious" uniformly.
# ---------------------------------------------------------------------
DIAGNOSTICS = {
    "chi2nu_h0": "high",       # worse fit quality -> more likely wrong basin
    "morph_mismatch": "high",  # poorer seed match -> more likely bad start
    "dlogtE": "high",          # large displacement from seed during optimization
    "dlogrho": "high",
    "du0": "high",
    "dt0_over_tE": "high",
    "d_bounds": "low",         # closer to a bound is more suspicious
    "R_max_resid": "high",     # larger peak standardized residual
    "T50_resid": "high",       # wider contiguous residual structure (ambiguous, tested as declared)
    "T75_resid": "high",
    "dw_stat": "low",          # near 0 = strong positive autocorrelation = suspicious
    "logkappa_J": "high",      # worse Jacobian conditioning
}
# A_resid: ambiguous sign: use |A_resid| as its own derived diagnostic (see below)

df = pd.read_csv("basin_risk_development_table.csv")
excluded = df[df["excluded_from_development"]].copy()
dev = df[~df["excluded_from_development"]].reset_index(drop=True)
N = len(dev)
n_dangerous = int(dev["dangerous"].sum())
print(f"development n={N}, dangerous={n_dangerous} ({n_dangerous/N:.1%}), "
      f"large_gap_abs={int(dev['large_gap_abs'].sum())}, large_gap_up={int(dev['large_gap_up'].sum())}")
print("dangerous event ids:", sorted(dev.loc[dev['dangerous'], 'event_id'].tolist()))
print()
print("=" * 100)
print("Excluded (catastrophic, chi2-sanity-flagged) events -- AUDIT ONLY, not in any metric below:")
print(excluded[["event_id", "D_s", "D_r", "dangerous", "large_gap_abs", "large_gap_up", "chi2nu_h0"]]
      .to_string(index=False))
print()


def score_column(name, direction):
    """score = higher means more suspicious; missing -> -inf (never triggered)."""
    v = dev[name].astype(float).values.copy()
    missing = ~np.isfinite(v)
    s = v if direction == "high" else -v
    s = np.where(missing, -np.inf, s)
    return s, missing


def add_derived_columns():
    dev["abs_A_resid"] = dev["A_resid"].abs()
    DIAGNOSTICS["abs_A_resid"] = "high"


add_derived_columns()


def activation_selection(score, target_frac):
    """Returns (selected_mask_definite, tied_group_mask, n_target, n_definite,
    n_slots_in_tied_group_needed) for a top-target_frac-by-score selection,
    handling ties at the cutoff explicitly."""
    n = len(score)
    k = int(round(target_frac * n))
    k = max(0, min(n, k))
    if k == 0:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), k, 0, 0
    order = np.argsort(-score, kind="stable")
    cutoff_score = score[order[k - 1]]
    strictly_above = score > cutoff_score
    tied = score == cutoff_score
    n_definite = int(strictly_above.sum())
    n_slots_needed = k - n_definite
    return strictly_above, tied, k, n_definite, n_slots_needed


def recall_precision_worst_best(dev, score, dangerous_col, target_frac):
    strictly_above, tied, k, n_definite, n_slots_needed = activation_selection(score, target_frac)
    dangerous = dev[dangerous_col].values
    n_dangerous_total = int(dangerous.sum())

    tied_idx = np.where(tied)[0]
    tied_dangerous_idx = [i for i in tied_idx if dangerous[i]]
    tied_negative_idx = [i for i in tied_idx if not dangerous[i]]
    n_tied = len(tied_idx)

    # worst case: fill remaining slots with negatives first
    n_dangerous_in_tied_worst = max(0, n_slots_needed - len(tied_negative_idx))
    n_dangerous_in_tied_worst = min(n_dangerous_in_tied_worst, len(tied_dangerous_idx))
    # best case: fill remaining slots with dangerous first
    n_dangerous_in_tied_best = min(n_slots_needed, len(tied_dangerous_idx))

    n_dangerous_definite = int(np.sum(dangerous[strictly_above]))

    n_dangerous_triggered_worst = n_dangerous_definite + n_dangerous_in_tied_worst
    n_dangerous_triggered_best = n_dangerous_definite + n_dangerous_in_tied_best

    recall_worst = n_dangerous_triggered_worst / n_dangerous_total if n_dangerous_total else np.nan
    recall_best = n_dangerous_triggered_best / n_dangerous_total if n_dangerous_total else np.nan

    n_triggered_actual = n_definite + n_slots_needed  # == k unless fewer tied available
    precision_worst = n_dangerous_triggered_worst / n_triggered_actual if n_triggered_actual else np.nan
    precision_best = n_dangerous_triggered_best / n_triggered_actual if n_triggered_actual else np.nan

    return dict(
        target_activation_rate=target_frac, actual_activation_rate=n_triggered_actual / len(dev),
        n_triggered=n_triggered_actual, n_dangerous_total=n_dangerous_total,
        n_dangerous_triggered_worst=n_dangerous_triggered_worst,
        n_dangerous_triggered_best=n_dangerous_triggered_best,
        recall_worst_case=recall_worst, recall_best_case=recall_best,
        precision_worst_case=precision_worst, precision_best_case=precision_best,
        n_boundary_tied=n_tied, n_slots_available_in_tied_group=n_slots_needed,
    )


# ======================================================================
# Section 7-8: per-diagnostic recall vs activation, with tie handling
# ======================================================================
print("=" * 100)
print("Feature-safety audit")
print("=" * 100)
audit_rows = []
for name in list(DIAGNOSTICS.keys()):
    audit_rows.append({"feature": name, "production_safe": True,
                        "source": "H0 fit / morphology seed / light curve only",
                        "truth_reference_leakage": False})
audit_df = pd.DataFrame(audit_rows)
print(audit_df.to_string(index=False))
audit_df.to_csv("feature_safety_audit.csv", index=False)

print()
print("=" * 100)
print("Continuous diagnostic summaries (development n=93)")
print("=" * 100)
summary_rows = []
for name in DIAGNOSTICS:
    v = dev[name].astype(float)
    n_missing = int(v.isna().sum())
    finite = v.dropna()
    summary_rows.append({
        "feature": name, "direction": DIAGNOSTICS[name], "n_missing": n_missing,
        "coverage": 1 - n_missing / N,
        "min": finite.min() if len(finite) else np.nan, "median": finite.median() if len(finite) else np.nan,
        "p90": finite.quantile(0.9) if len(finite) else np.nan, "max": finite.max() if len(finite) else np.nan,
    })
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv("feature_summary.csv", index=False)

print()
print("=" * 100)
print("Recall vs activation, per diagnostic (development n=93, worst-case recall is the conservative comparator)")
print("=" * 100)
per_feature_rows = []
for name, direction in DIAGNOSTICS.items():
    score, missing = score_column(name, direction)
    for frac in ACTIVATION_BUDGETS:
        r = recall_precision_worst_best(dev, score, "dangerous", frac)
        r.update({"feature": name, "direction": direction, "n_missing": int(missing.sum())})
        per_feature_rows.append(r)
        print(f"{name:15s} dir={direction:4s} target={frac:.0%} actual={r['actual_activation_rate']:.1%} "
              f"n_trig={r['n_triggered']:3d} recall_worst={r['recall_worst_case']:.1%} "
              f"recall_best={r['recall_best_case']:.1%} prec_worst={r['precision_worst_case']:.1%} "
              f"tied={r['n_boundary_tied']} slots={r['n_slots_available_in_tied_group']}")
per_feature_df = pd.DataFrame(per_feature_rows)
per_feature_df.to_csv("recall_vs_activation_per_feature.csv", index=False)

# ======================================================================
# Section 9: stratified bootstrap stability
# ======================================================================
print()
print("=" * 100)
print(f"Stratified bootstrap (seed={SEED}, resamples={N_BOOTSTRAP}, 5-95% percentile interval)")
print("=" * 100)

rng = np.random.default_rng(SEED)
dangerous_idx = np.where(dev["dangerous"].values)[0]
negative_idx = np.where(~dev["dangerous"].values)[0]

bootstrap_rows = []
for name, direction in DIAGNOSTICS.items():
    v_full = dev[name].astype(float).values
    for frac in ACTIVATION_BUDGETS:
        recalls = []
        for _b in range(N_BOOTSTRAP):
            samp_dangerous = rng.choice(dangerous_idx, size=len(dangerous_idx), replace=True)
            samp_negative = rng.choice(negative_idx, size=len(negative_idx), replace=True)
            samp_idx = np.concatenate([samp_dangerous, samp_negative])
            v = v_full[samp_idx]
            dangerous_b = np.concatenate([np.ones(len(samp_dangerous), dtype=bool),
                                           np.zeros(len(samp_negative), dtype=bool)])
            missing_b = ~np.isfinite(v)
            s_b = (v if direction == "high" else -v)
            s_b = np.where(missing_b, -np.inf, s_b)
            n_b = len(s_b)
            k_b = max(0, min(n_b, int(round(frac * n_b))))
            if k_b == 0:
                recalls.append(0.0)
                continue
            order_b = np.argsort(-s_b, kind="stable")
            selected = np.zeros(n_b, dtype=bool)
            selected[order_b[:k_b]] = True
            n_dang_b = dangerous_b.sum()
            recalls.append((selected & dangerous_b).sum() / n_dang_b if n_dang_b else np.nan)
        recalls = np.asarray(recalls)
        lo, med, hi = np.nanpercentile(recalls, [5, 50, 95])
        bootstrap_rows.append({"feature": name, "activation": frac, "median_recall": med,
                                "ci_lo_5": lo, "ci_hi_95": hi})
boot_df = pd.DataFrame(bootstrap_rows)
boot_df.to_csv("bootstrap_recall.csv", index=False)
print(boot_df.to_string(index=False))

# ======================================================================
# Section 10: small interpretable combinations (union-of-top-q-percentile rule)
# ======================================================================
print()
print("=" * 100)
print("Small diagnostic combinations (union of top-q suspicious fraction per constituent)")
print("=" * 100)

COMBINATIONS = {
    "R_max_resid_OR_morph_mismatch": ["R_max_resid", "morph_mismatch"],
    "R_max_resid_OR_dlogtE": ["R_max_resid", "dlogtE"],
    "R_max_resid_OR_d_bounds": ["R_max_resid", "d_bounds"],
    "chi2nu_h0_OR_R_max_resid": ["chi2nu_h0", "R_max_resid"],
}

combo_rows = []
for combo_name, members in COMBINATIONS.items():
    for frac in ACTIVATION_BUDGETS:
        n = N
        k = max(0, min(n, int(round(frac * n))))
        union_selected = np.zeros(n, dtype=bool)
        for m in members:
            score, _ = score_column(m, DIAGNOSTICS[m])
            if k == 0:
                continue
            order = np.argsort(-score, kind="stable")
            sel = np.zeros(n, dtype=bool)
            sel[order[:k]] = True
            union_selected |= sel
        dangerous = dev["dangerous"].values
        n_dangerous_total = int(dangerous.sum())
        n_triggered = int(union_selected.sum())
        n_dangerous_triggered = int((union_selected & dangerous).sum())
        recall = n_dangerous_triggered / n_dangerous_total if n_dangerous_total else np.nan
        precision = n_dangerous_triggered / n_triggered if n_triggered else np.nan
        combo_rows.append({"combination": combo_name, "per_member_target": frac,
                            "actual_union_activation": n_triggered / n, "n_triggered": n_triggered,
                            "recall": recall, "precision": precision})
        print(f"{combo_name:35s} per_member_q={frac:.0%} union_activation={n_triggered/n:.1%} "
              f"n_trig={n_triggered:3d} recall={recall:.1%} precision={precision:.1%}")
combo_df = pd.DataFrame(combo_rows)
combo_df.to_csv("recall_vs_activation_combinations.csv", index=False)

print()
print("=" * 100)
print("DONE")
