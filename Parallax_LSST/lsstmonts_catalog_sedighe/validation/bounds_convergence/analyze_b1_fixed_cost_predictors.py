#!/usr/bin/env python3

"""
Experiment B1 (part 2) -- fixed-cost H0 observable predictors.

STATUS: DIAGNOSTIC / CANDIDATE GENERATION, development-set analysis
(extreme100). Not a production policy, and NOT a stopping/rescue
threshold. See ADAPTIVE_H0_ANALYSIS.md, Experiment B1.

PURPOSE
    For each fixed-budget candidate base set produced by
    analyze_b1_fixed_budget_base_sets.py (one row per
    (k, candidate_rank) in b1_fixed_budget_candidate_sets.csv),
    compute, per event, a set of observables derivable ONLY from that
    candidate's own k fits for that event (chi2 spread, cross-mode
    agreement, active-bound flags, optimizer diagnostics, and
    physically-motivated parameter-agreement metrics -- see METHOD),
    then test empirically whether any of them separates
    "safe" (delta_chi2_vs_base_domain_oracle <= 0.1) from
    "unsafe" (> 0.1) events AT FIXED k -- i.e. holding the number of
    fits (and, per candidate, the exact strategy composition) constant,
    unlike Experiment A1's pooled-across-k marginal tables.

    This does not propose, tune, or select a stopping/rescue
    threshold. It only asks whether a predictor carries information at
    fixed cost.

INPUTS
    validation/bounds_convergence/results/gate1_oracle_diagnostics.csv
        (source == "gate1_oracle_52"; full per-fit fields including
        optimizer_active_mask, optimizer_optimality, t0/u0/tE/rho)
    validation/bounds_convergence/results/b1_fixed_budget_candidate_sets.csv
        (produced by analyze_b1_fixed_budget_base_sets.py; defines
        which (mode, strategy_id) strategies belong to each
        (k, candidate_rank) candidate)

OUTPUTS (validation/bounds_convergence/results/)
    b1_predictors_per_event.csv
        one row per (k, candidate_rank, catalog_row): every predictor
        computed from that candidate's own k fits for that event,
        plus the offline-only delta_chi2_vs_base_domain_oracle / safe
        label.
    b1_predictor_vs_failure_fixed_k.csv
        one row per (k, candidate_rank, predictor, metric_type,
        bucket): for metric_type="bucketed", a frequency table
        (n_obs, n_safe, n_unsafe, unsafe_rate) exactly analogous to
        Experiment A1's tables but computed AT FIXED k/candidate
        instead of pooled across k; for metric_type="continuous_separation"
        (bucket left blank), group means/medians (safe vs unsafe) and
        a rank-based separation score (AUC-style: the Mann-Whitney
        probability that a random unsafe-event value exceeds a random
        safe-event value). N_safe/N_unsafe are always reported
        alongside every statistic, per instruction -- this table never
        reports a bare rate without its denominators.

WHAT QUESTION THIS ANSWERS
    "Holding the fit budget (and, per candidate, its exact
    composition) fixed, does any observable computed only from that
    budget's own fits distinguish events that are still >0.1 from the
    base-domain oracle from events that are not?"

ORACLE USAGE
    chi2_oracle_52 is read ONLY to compute the offline
    delta_chi2_vs_base_domain_oracle / safe label used to evaluate
    predictors after the fact. No predictor column is a function of
    chi2_oracle_52 or of catalog_row. This is development-set
    (extreme100) analysis; no claim of generalization is made, and no
    threshold is fit or proposed here (that is explicitly deferred to
    Phase B2, which will also need an independent sample before any
    such threshold could be considered validated).

METHOD -- predictors computed per (candidate, event), from that
candidate's own k rows only
    - chi2_best, chi2_second_best, delta_spread = chi2_2nd - chi2_best;
    - chi2 dispersion among the k rows (std, range);
    - n_modes_in_set (distinct modes among the k strategies) and
      n_modes_agreeing (how many of those modes' own best-within-set
      chi2 is within 0.1 of the overall best-within-set chi2);
    - the winning row's optimizer_active_mask split into
      t0/u0/tE/rho_active, its optimizer_optimality, and its
      optimizer_success (kept as a pipeline sanity check even though
      Experiment 0 / A1 already found it has no variation on
      extreme100);
    - best-vs-second-best parameter agreement, using both raw
      differences and physically-motivated, degeneracy-aware
      diagnostics (per explicit instruction, NOT raw differences
      alone):
        diff_u0_signed          = u0_best - u0_2nd
        diff_abs_u0             = | |u0_best| - |u0_2nd| |
            (insensitive to the u0 -> -u0 mirror degeneracy)
        abs_log_ratio_tE        = | log(tE_best / tE_2nd) |
        abs_log_ratio_rho       = | log(rho_best / rho_2nd) |
            (tE, rho are strictly positive and span orders of
            magnitude -- a log-ratio is the physically meaningful
            agreement scale, not an absolute difference)
        diff_t0_raw             = | t0_best - t0_2nd |
        diff_t0_norm_by_tE      = diff_t0_raw / tE_best
            (t0 offsets are only meaningful relative to the event's
            own timescale)
    NOT computed here: `steps_since_improvement`. It requires an
    execution ORDER, but B1's candidate sets are evaluated as
    unordered batches (see analyze_b1_fixed_budget_base_sets.py) --
    this predictor is revisited only once Phase B2 designs an actual
    sequential policy with a defined order.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


RESULTS_DIR = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results"
)

IN_DIAGNOSTICS = RESULTS_DIR / "gate1_oracle_diagnostics.csv"
IN_CANDIDATES = RESULTS_DIR / "b1_fixed_budget_candidate_sets.csv"

OUT_PREDICTORS = RESULTS_DIR / "b1_predictors_per_event.csv"
OUT_PREDICTOR_VS_FAILURE = (
    RESULTS_DIR / "b1_predictor_vs_failure_fixed_k.csv"
)

TOL = 0.1

BUCKETED_PREDICTORS = [
    "n_modes_agreeing",
    "winner_t0_active",
    "winner_u0_active",
    "winner_tE_active",
    "winner_rho_active",
    "winner_optimizer_success",
]

CONTINUOUS_PREDICTORS = [
    "delta_spread",
    "chi2_std_among_set",
    "chi2_range_among_set",
    "winner_optimizer_optimality",
    "diff_u0_signed",
    "diff_abs_u0",
    "abs_log_ratio_tE",
    "abs_log_ratio_rho",
    "diff_t0_raw",
    "diff_t0_norm_by_tE",
]


def _parse_active_mask(raw) -> tuple[int, ...] | None:

    if not isinstance(raw, str):
        return None
    try:
        val = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    return tuple(int(x) for x in val)


def load_oracle52() -> pd.DataFrame:

    df = pd.read_csv(IN_DIAGNOSTICS)
    o = df[df["source"] == "gate1_oracle_52"].copy()

    if o.empty:
        raise RuntimeError(
            f"No gate1_oracle_52 rows found in {IN_DIAGNOSTICS}."
        )

    o["strategy_key"] = o["mode"] + "/" + o["strategy_id"]
    o["active_mask_parsed"] = o["optimizer_active_mask"].apply(
        _parse_active_mask
    )

    return o


def load_candidates() -> pd.DataFrame:

    cand = pd.read_csv(IN_CANDIDATES)

    def _strategy_set(s: str) -> tuple[str, ...]:
        return tuple(part.strip() for part in s.split(" + "))

    cand["strategy_set"] = cand["strategies"].apply(_strategy_set)

    return cand


def compute_predictors_for_candidate(
    o: pd.DataFrame,
    strategy_set: tuple[str, ...],
    oracle_chi2: pd.Series,
) -> pd.DataFrame:

    sub = o[o["strategy_key"].isin(strategy_set)]

    if sub["strategy_key"].nunique() != len(strategy_set):
        raise RuntimeError(
            "Not all strategies in this candidate were found in "
            f"gate1_oracle_diagnostics.csv: expected "
            f"{sorted(strategy_set)}, found "
            f"{sorted(sub['strategy_key'].unique())}."
        )

    records = []

    for catalog_row, g in sub.groupby("catalog_row"):

        g = g.reset_index(drop=True)
        chi2 = g["chi2"].to_numpy()

        order = np.argsort(chi2)
        best_pos = order[0]
        second_pos = order[1] if len(order) >= 2 else None

        chi2_best = float(chi2[best_pos])
        chi2_second_best = (
            float(chi2[second_pos])
            if second_pos is not None
            else float("nan")
        )
        delta_spread = (
            chi2_second_best - chi2_best
            if second_pos is not None
            else float("nan")
        )

        best_row = g.iloc[best_pos]
        mask = best_row["active_mask_parsed"]

        if mask is not None and len(mask) == 4:
            t0_active, u0_active, tE_active, rho_active = (
                int(mask[0] != 0),
                int(mask[1] != 0),
                int(mask[2] != 0),
                int(mask[3] != 0),
            )
        else:
            t0_active = u0_active = tE_active = rho_active = np.nan

        modes = g["mode"].to_numpy()
        n_modes_in_set = len(set(modes.tolist()))

        per_mode_best = {
            m: float(chi2[modes == m].min())
            for m in set(modes.tolist())
        }
        n_modes_agreeing = sum(
            1
            for v in per_mode_best.values()
            if v <= chi2_best + TOL
        )

        if second_pos is not None:
            second_row = g.iloc[second_pos]

            u0_best = float(best_row["u0"])
            u0_2nd = float(second_row["u0"])
            diff_u0_signed = u0_best - u0_2nd
            diff_abs_u0 = abs(abs(u0_best) - abs(u0_2nd))

            tE_best = float(best_row["tE"])
            tE_2nd = float(second_row["tE"])
            abs_log_ratio_tE = (
                abs(np.log(tE_best / tE_2nd))
                if tE_best > 0 and tE_2nd > 0
                else float("nan")
            )

            rho_best = float(best_row["rho"])
            rho_2nd = float(second_row["rho"])
            abs_log_ratio_rho = (
                abs(np.log(rho_best / rho_2nd))
                if rho_best > 0 and rho_2nd > 0
                else float("nan")
            )

            t0_best = float(best_row["t0"])
            t0_2nd = float(second_row["t0"])
            diff_t0_raw = abs(t0_best - t0_2nd)
            diff_t0_norm_by_tE = (
                diff_t0_raw / tE_best if tE_best > 0 else float("nan")
            )
        else:
            diff_u0_signed = float("nan")
            diff_abs_u0 = float("nan")
            abs_log_ratio_tE = float("nan")
            abs_log_ratio_rho = float("nan")
            diff_t0_raw = float("nan")
            diff_t0_norm_by_tE = float("nan")

        o_chi2 = float(oracle_chi2.get(catalog_row, float("nan")))
        delta = chi2_best - o_chi2

        records.append(
            {
                "catalog_row": int(catalog_row),
                "chi2_best": chi2_best,
                "chi2_second_best": chi2_second_best,
                "delta_spread": delta_spread,
                "chi2_std_among_set": float(np.std(chi2)),
                "chi2_range_among_set": float(
                    chi2.max() - chi2.min()
                ),
                "n_modes_in_set": n_modes_in_set,
                "n_modes_agreeing": n_modes_agreeing,
                "winner_t0_active": t0_active,
                "winner_u0_active": u0_active,
                "winner_tE_active": tE_active,
                "winner_rho_active": rho_active,
                "winner_optimizer_success": bool(
                    best_row["optimizer_success"]
                ),
                "winner_optimizer_optimality": float(
                    best_row["optimizer_optimality"]
                ),
                "diff_u0_signed": diff_u0_signed,
                "diff_abs_u0": diff_abs_u0,
                "abs_log_ratio_tE": abs_log_ratio_tE,
                "abs_log_ratio_rho": abs_log_ratio_rho,
                "diff_t0_raw": diff_t0_raw,
                "diff_t0_norm_by_tE": diff_t0_norm_by_tE,
                # OFFLINE-ONLY (oracle-derived): never a predictor
                "chi2_oracle_52": o_chi2,
                "delta_chi2_vs_base_domain_oracle": delta,
                "safe": bool(delta <= TOL),
            }
        )

    return pd.DataFrame.from_records(records)


def _bucket_n_modes_agreeing(x) -> str:
    if pd.isna(x):
        return "NA"
    return str(int(x))


def _bucket_binary(x) -> str:
    if pd.isna(x):
        return "NA"
    return str(int(x))


def bucketed_table(
    df: pd.DataFrame, predictor: str, bucket_fn
) -> pd.DataFrame:

    g = (
        df.assign(_bucket=df[predictor].apply(bucket_fn))
        .groupby("_bucket")["safe"]
        .agg(n_obs="size", n_safe="sum")
    )
    g["n_unsafe"] = g["n_obs"] - g["n_safe"]
    g["unsafe_rate"] = g["n_unsafe"] / g["n_obs"]

    out = g.reset_index().rename(columns={"_bucket": "bucket"})
    out.insert(0, "predictor", predictor)
    out.insert(1, "metric_type", "bucketed")

    return out


def continuous_separation_row(
    df: pd.DataFrame, predictor: str
) -> dict | None:

    vals = df[predictor]
    safe_vals = vals[df["safe"] & vals.notna()]
    unsafe_vals = vals[(~df["safe"]) & vals.notna()]

    n_safe = len(safe_vals)
    n_unsafe = len(unsafe_vals)

    if n_safe == 0 or n_unsafe == 0:
        auc = float("nan")
    else:
        u_stat, _ = mannwhitneyu(
            unsafe_vals, safe_vals, alternative="two-sided"
        )
        auc = float(u_stat) / (n_safe * n_unsafe)

    return {
        "predictor": predictor,
        "metric_type": "continuous_separation",
        "bucket": "",
        "n_obs": n_safe + n_unsafe,
        "n_safe": n_safe,
        "n_unsafe": n_unsafe,
        "unsafe_rate": (
            n_unsafe / (n_safe + n_unsafe)
            if (n_safe + n_unsafe)
            else float("nan")
        ),
        "mean_safe": (
            float(safe_vals.mean()) if n_safe else float("nan")
        ),
        "median_safe": (
            float(safe_vals.median()) if n_safe else float("nan")
        ),
        "mean_unsafe": (
            float(unsafe_vals.mean()) if n_unsafe else float("nan")
        ),
        "median_unsafe": (
            float(unsafe_vals.median()) if n_unsafe else float("nan")
        ),
        "auc_unsafe_gt_safe": auc,
    }


def compute_predictor_vs_failure(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for predictor in BUCKETED_PREDICTORS:
        if predictor == "n_modes_agreeing":
            bucket_fn = _bucket_n_modes_agreeing
        else:
            bucket_fn = _bucket_binary
        rows.append(bucketed_table(df, predictor, bucket_fn))

    cont_records = []
    for predictor in CONTINUOUS_PREDICTORS:
        rec = continuous_separation_row(df, predictor)
        if rec is not None:
            cont_records.append(rec)

    cont_df = pd.DataFrame.from_records(cont_records)

    return pd.concat(rows + [cont_df], ignore_index=True)


def main() -> None:

    o = load_oracle52()
    oracle_chi2 = o.groupby("catalog_row")["chi2"].min()

    candidates = load_candidates()

    all_predictors = []
    all_pvf = []

    for _, cand in candidates.iterrows():

        pred_df = compute_predictors_for_candidate(
            o, cand["strategy_set"], oracle_chi2
        )
        pred_df.insert(0, "k", cand["k"])
        pred_df.insert(1, "candidate_rank", cand["candidate_rank"])

        pvf_df = compute_predictor_vs_failure(pred_df)
        pvf_df.insert(0, "k", cand["k"])
        pvf_df.insert(1, "candidate_rank", cand["candidate_rank"])

        all_predictors.append(pred_df)
        all_pvf.append(pvf_df)

        n_safe = int(pred_df["safe"].sum())
        n_unsafe = int((~pred_df["safe"]).sum())
        print(
            f"k={cand['k']} rank={cand['candidate_rank']} "
            f"({cand['selection_reason']}): "
            f"n_safe={n_safe} n_unsafe={n_unsafe}"
        )

    predictors_out = pd.concat(all_predictors, ignore_index=True)
    pvf_out = pd.concat(all_pvf, ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    predictors_out.to_csv(OUT_PREDICTORS, index=False)
    pvf_out.to_csv(OUT_PREDICTOR_VS_FAILURE, index=False)

    print()
    print("=" * 70)
    print("WROTE")
    print("=" * 70)
    print(f"  {OUT_PREDICTORS} ({len(predictors_out)} rows)")
    print(f"  {OUT_PREDICTOR_VS_FAILURE} ({len(pvf_out)} rows)")


if __name__ == "__main__":
    main()
