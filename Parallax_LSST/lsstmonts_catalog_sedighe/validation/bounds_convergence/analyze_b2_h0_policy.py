#!/usr/bin/env python3

"""
Experiment B2 -- minimal conservative H0 adaptive-policy closure.

STATUS: policy design / evaluation on the extreme100 development
sample. See ADAPTIVE_H0_ANALYSIS.md, Experiment B2.

PURPOSE
    Test whether a small fixed H0 base set (k=4, 5, or 6, from
    Experiment B1's own candidate_rank==1 lexicographic-optimal
    candidates -- NOT re-optimized here) plus a simple, auditable
    stopping rule on Experiment B1's already-studied fixed-cost fit
    diagnostics (`abs_log_ratio_tE`, `delta_spread`, and, only to
    check for a clear improvement, a few other already-computed B1
    diagnostics) can reproduce the validated H0 reference
    (`gate1_final18` + the validated t0 rescue, i.e. Experiment A1's
    `chi2_policy_final`) within `delta_opt<=0.1` for every
    `extreme100` event, while spending fewer H0 fits on average than
    always running the full 18-strategy plan.

    This is NOT a new feature search. No new event features, no new
    ML models, no B1c pre-H0 features, no re-optimization of B1's
    candidate sets. Only threshold selection on already-computed
    diagnostics, using the already-validated reference as an offline
    design-time label.

INPUTS
    validation/bounds_convergence/results/b1_predictors_per_event.csv
        (Experiment B1; used ONLY for candidate_rank==1, k in
        {4,5,6}: chi2_best and its fixed-cost diagnostics
        `delta_spread`, `abs_log_ratio_tE`, plus
        `chi2_std_among_set`, `chi2_range_among_set`,
        `n_modes_agreeing`, `winner_t0_active`, inspected only to
        check whether any gives a clear improvement over the two
        primary diagnostics -- not a new search.)
    validation/bounds_convergence/results/b1_fixed_budget_candidate_sets.csv
        (Experiment B1; used only to confirm, not re-derive, that the
        k=4/5/6 candidate_rank==1 strategy sets are each a SUBSET of
        the 18 gate1_final18 strategies -- this determines the N_fits
        accounting for escalation, see METHOD.)
    validation/bounds_convergence/results/a1_sequential_h0_full18_policy.csv
        (Experiment A1; `chi2_policy_final` is THE validated H0
        reference -- gate1_final18's full-18 winner, with the
        validated t0 rescue applied when that winner's own
        active_mask shows t0 active. This is the target this
        experiment tries to reproduce cheaply, NOT `chi2_oracle_52`.)

OUTPUTS (validation/bounds_convergence/results/)
    b2_policy_threshold_search.csv
        one row per (k, predictor): the minimal threshold on that
        predictor achieving zero false-safe on extreme100 (escalate
        if predictor >= threshold), and the resulting escalation
        fraction / cost. This is a diagnostic table showing WHY the
        single/OR threshold approach does or does not work -- not a
        recommendation by itself.
    b2_policy_comparison.csv
        one row per candidate policy (fixed gate1_final18 baseline;
        for each k in {4,5,6}, the best single-predictor rule, the OR
        rule on {abs_log_ratio_tE, delta_spread}, and, if it changes
        the conclusion, the best rule found among the other inspected
        diagnostics): mean/p50/p90/max N_fits, fraction stopped at
        base, fraction escalated, N(delta_opt>0.1/1/10), max delta_opt,
        exact rule text. This is the table Experiment B2's decision is
        based on.

METHOD -- N_fits ACCOUNTING (verified, not assumed)
    Confirmed in this session: every one of the k=4, k=5, k=6
    candidate_rank==1 strategy sets (from
    b1_fixed_budget_candidate_sets.csv) is a SUBSET of the 18
    strategies in gate1_final18's own plan. Therefore, if a policy
    escalates, the k base fits are reused (not re-run), and
    escalation costs exactly `18 - k` additional fits to complete the
    full gate1_final18 plan, for a total of 18 -- plus 18 more, only
    for the one event (catalog_row 36103 in extreme100) whose
    full-18 winner triggers the already-validated t0 rescue (per
    Experiment A1, applied only at the full-18-winner boundary, never
    earlier). A SAFE decision costs exactly k fits.

    Because escalation always completes the exact validated reference
    procedure, an escalated event's resulting chi2 is by construction
    `chi2_policy_final` (delta_opt = 0). This means
    `N(delta_opt>0.1)` after applying any policy equals exactly the
    policy's `N_false_safe` on the SAFE-decided subset -- there is no
    additional failure mode introduced by escalation itself.

METHOD -- threshold selection (design-time use of the reference,
never a runtime feature)
    For a single predictor P assumed to increase with risk (verified,
    not assumed, per predictor: Experiment B1/B1c already found
    `delta_spread` and `abs_log_ratio_tE` positively associated with
    failure), the minimal zero-false-safe threshold is
    `t* = min(P over events with delta_vs_reference > 0.1)`, with the
    rule "escalate if P >= t*". This is the cheapest single-threshold
    rule that guarantees catching every truly-risky development-set
    event. No grid search or joint optimization is performed beyond
    this and the simple OR of two such single-predictor rules,
    per instruction to keep this auditable and not open a new search.

ORACLE/REFERENCE USAGE
    `chi2_policy_final` (the validated reference) and
    `chi2_oracle_52` are used ONLY to (a) label
    `delta_vs_reference`/`risky` for offline threshold selection and
    evaluation, and (b) account for escalation cost. No policy
    evaluated here uses `catalog_row`, `chi2_policy_final`, or
    `chi2_oracle_52` as a runtime decision input -- only
    `chi2_best`/diagnostics computable from the k base-set fits
    themselves are used in each rule's "escalate" condition.

NO FITS, NO pyLIMA, NO CHE, NO core/production changes, NO B1c
features, NO H1 work: this script only reads already-committed B1/A1
results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results"
)

B1_PREDICTORS_PATH = RESULTS_DIR / "b1_predictors_per_event.csv"
B1_CANDIDATES_PATH = RESULTS_DIR / "b1_fixed_budget_candidate_sets.csv"
A1_FULL18_POLICY_PATH = (
    RESULTS_DIR / "a1_sequential_h0_full18_policy.csv"
)

OUT_THRESHOLD_SEARCH = RESULTS_DIR / "b2_policy_threshold_search.csv"
OUT_COMPARISON = RESULTS_DIR / "b2_policy_comparison.csv"

TOL = 0.1
CANDIDATE_K_VALUES = [4, 5, 6]
CANDIDATE_RANK = 1

PRIMARY_PREDICTORS = ["abs_log_ratio_tE", "delta_spread"]

# Additional already-computed B1 diagnostics inspected ONLY to check
# for a clear improvement over the two primary predictors -- not a
# new feature search (see module docstring).
OTHER_DIAGNOSTICS_TO_CHECK = [
    "chi2_std_among_set",
    "chi2_range_among_set",
    "n_modes_agreeing",
]

N_FINAL18 = 18

# ============================================================
# FROZEN B2 CANDIDATE POLICY
# ============================================================
#
# This is the exact policy recommended in ADAPTIVE_H0_ANALYSIS.md,
# Experiment B2, status CANDIDATE -- FROZEN FOR INDEPENDENT
# VALIDATION. It is pinned here, at full float precision, as the
# single source of truth for what Gate 3 must evaluate -- prose
# elsewhere (including this script's own printed summary and the
# markdown doc) may show a ROUNDED value (e.g. "0.000412") for
# readability; this constant is the unrounded value actually used.
#
# Gate 3 must evaluate EXACTLY this (k, predictor, threshold) triple
# without modification. If Gate 3 finds a false-safe case:
#   - the conservative default is to abandon this adaptive policy and
#     produce with the fixed gate1_final18 (+ validated t0 rescue)
#     reference, which remains correct regardless;
#   - retuning this threshold (or any other part of this policy)
#     using Gate 3 observations is only valid if Gate 3 is then
#     treated as consumed development data, and the retuned policy is
#     evaluated on a NEW, independent sample before any production
#     use. Silently retuning on Gate 3 and treating that same sample
#     as having validated the retuned policy is not acceptable.
FROZEN_B2_POLICY_K = 6
FROZEN_B2_POLICY_PREDICTOR = "abs_log_ratio_tE"
FROZEN_B2_POLICY_THRESHOLD_ESCALATE_IF_GTE = 0.0004123330728713


def assert_frozen_policy_matches_computed_threshold(
    threshold_records: list[dict],
) -> None:
    """
    The frozen policy above must always equal exactly what this
    script's own (unchanged) threshold search computes for
    (k=6, abs_log_ratio_tE) -- this is a consistency check, not a
    re-derivation: if this ever fails, the frozen constant above is
    stale relative to the inputs and must be re-examined before
    trusting anything downstream, not silently updated.
    """

    matches = [
        r
        for r in threshold_records
        if r["k"] == FROZEN_B2_POLICY_K
        and r["predictor"] == FROZEN_B2_POLICY_PREDICTOR
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one threshold_search row for "
            f"k={FROZEN_B2_POLICY_K}, "
            f"predictor={FROZEN_B2_POLICY_PREDICTOR!r}; "
            f"found {len(matches)}."
        )

    computed = matches[0]["threshold_escalate_if_gte"]

    if not np.isclose(
        computed,
        FROZEN_B2_POLICY_THRESHOLD_ESCALATE_IF_GTE,
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError(
            "FROZEN_B2_POLICY_THRESHOLD_ESCALATE_IF_GTE = "
            f"{FROZEN_B2_POLICY_THRESHOLD_ESCALATE_IF_GTE!r} does "
            "not match the freshly computed threshold "
            f"{computed!r} for k={FROZEN_B2_POLICY_K}, "
            f"predictor={FROZEN_B2_POLICY_PREDICTOR!r}. Do not "
            "silently update the frozen constant -- re-examine why "
            "the inputs changed before touching the frozen policy."
        )

    print(
        "Frozen B2 policy confirmed consistent with inputs: "
        f"k={FROZEN_B2_POLICY_K}, "
        f"predictor={FROZEN_B2_POLICY_PREDICTOR}, "
        "threshold (full precision) = "
        f"{FROZEN_B2_POLICY_THRESHOLD_ESCALATE_IF_GTE!r}"
    )


def verify_subset_of_final18(k: int) -> list[str]:
    """
    Confirm (not assume) that the k, candidate_rank==1 strategy set
    is a subset of gate1_final18's own 18 strategies, so escalation
    cost accounting (18 total, not 18+k) is justified.
    """

    gate1_final18 = {
        "physical": [("old_H0", 1.0), ("truth", 0.1), ("truth", 1.0)],
        "log_te": [
            ("old_H0", "truth_rho"),
            ("old_H0", 0.01),
            ("old_H0", 0.1),
            ("old_H0", 1.0),
            ("truth", 0.01),
            ("truth", 1.0),
        ],
        "log_rho": [
            ("old_H0", "truth_rho"),
            ("old_H0", 0.01),
            ("old_H0", 1.0),
            ("truth", "truth_rho"),
            ("truth", 1e-6),
            ("truth", 1e-4),
            ("truth", 0.1),
            ("truth", 1.0),
        ],
        "log_te_rho": [("truth", 1e-6)],
    }

    def _strategy_id(anchor, rho_rule):
        if rho_rule == "truth_rho":
            return f"{anchor}/truth_rho"
        return f"{anchor}/{rho_rule:.6g}"

    final18_keys = {
        f"{mode}/{_strategy_id(anchor, rho)}"
        for mode, lst in gate1_final18.items()
        for anchor, rho in lst
    }

    if len(final18_keys) != N_FINAL18:
        raise RuntimeError(
            f"Expected {N_FINAL18} gate1_final18 keys, built "
            f"{len(final18_keys)}."
        )

    candidates = pd.read_csv(B1_CANDIDATES_PATH)
    row = candidates[
        (candidates["k"] == k)
        & (candidates["candidate_rank"] == CANDIDATE_RANK)
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Expected exactly 1 candidate row for k={k}, "
            f"candidate_rank={CANDIDATE_RANK}; found {len(row)}."
        )

    members = row.iloc[0]["strategies"].split(" + ")

    not_subset = [m for m in members if m not in final18_keys]
    if not_subset:
        raise RuntimeError(
            f"k={k} candidate_rank={CANDIDATE_RANK} contains "
            f"strategies NOT in gate1_final18: {not_subset}. "
            "Escalation cost accounting (18 total) does not hold; "
            "fix N_fits accounting before trusting this policy."
        )

    return members


def load_merged(k: int) -> pd.DataFrame:

    pred = pd.read_csv(B1_PREDICTORS_PATH)
    sub = pred[
        (pred["k"] == k) & (pred["candidate_rank"] == CANDIDATE_RANK)
    ].copy()

    ref = pd.read_csv(A1_FULL18_POLICY_PATH)[
        ["catalog_row", "chi2_policy_final", "rescue_applied"]
    ]

    m = sub.merge(ref, on="catalog_row", how="inner")

    if len(m) != 100:
        raise RuntimeError(
            f"Expected 100 merged rows for k={k}, got {len(m)}."
        )

    m["delta_vs_reference"] = m["chi2_best"] - m["chi2_policy_final"]
    m["risky"] = m["delta_vs_reference"] > TOL

    return m


def threshold_search_single(
    m: pd.DataFrame, k: int, predictor: str
) -> dict:

    risky = m[m["risky"]]

    if len(risky) == 0:
        # No risky events at all at this k: the trivial "never
        # escalate" rule already has zero false-safe.
        threshold = float("inf")
        n_escalate = 0
    else:
        threshold = float(risky[predictor].min())
        n_escalate = int((m[predictor] >= threshold).sum())

    n_events = len(m)

    return {
        "k": k,
        "predictor": predictor,
        "n_risky_at_k": len(risky),
        "min_risky_value": (
            threshold if len(risky) else float("nan")
        ),
        "threshold_escalate_if_gte": threshold,
        "n_escalate": n_escalate,
        "frac_escalate": n_escalate / n_events,
    }


def evaluate_policy(
    m: pd.DataFrame, k: int, escalate_mask: np.ndarray, rule_text: str
) -> dict:

    n_events = len(m)

    n_fits = np.where(
        escalate_mask,
        N_FINAL18 + np.where(m["rescue_applied"] == 1, N_FINAL18, 0),
        k,
    )

    # delta_opt: 0 if escalated (reproduces the reference exactly by
    # construction -- see module docstring), else delta_vs_reference.
    delta_opt = np.where(escalate_mask, 0.0, m["delta_vs_reference"])

    n_false_safe = int(((~escalate_mask) & m["risky"]).sum())

    return {
        "k": k,
        "rule": rule_text,
        "n_events": n_events,
        "frac_stopped_at_base": float((~escalate_mask).mean()),
        "frac_escalated": float(escalate_mask.mean()),
        "n_false_safe": n_false_safe,
        "n_delta_opt_gt_0p1": int((delta_opt > 0.1).sum()),
        "n_delta_opt_gt_1": int((delta_opt > 1.0).sum()),
        "n_delta_opt_gt_10": int((delta_opt > 10.0).sum()),
        "max_delta_opt": float(np.max(delta_opt)),
        "mean_n_fits": float(np.mean(n_fits)),
        "p50_n_fits": float(np.percentile(n_fits, 50)),
        "p90_n_fits": float(np.percentile(n_fits, 90)),
        "max_n_fits": float(np.max(n_fits)),
    }


def main() -> None:

    print("=" * 70)
    print(
        "Verifying k=4/5/6 candidate_rank=1 sets are subsets of "
        "gate1_final18 (required for the N_fits accounting used "
        "below)"
    )
    print("=" * 70)
    for k in CANDIDATE_K_VALUES:
        members = verify_subset_of_final18(k)
        print(f"  k={k}: {len(members)} strategies, all in "
              f"gate1_final18 -- OK")
    print()

    threshold_records = []
    comparison_records = []

    # --- Fixed gate1_final18 baseline (always "escalate", i.e.
    # always run the full validated reference procedure) ---
    ref = pd.read_csv(A1_FULL18_POLICY_PATH)
    n_fits_fixed = N_FINAL18 + np.where(
        ref["rescue_applied"] == 1, N_FINAL18, 0
    )
    comparison_records.append(
        {
            "k": "N/A (fixed)",
            "rule": "always run gate1_final18 (+ validated t0 rescue)",
            "n_events": len(ref),
            "frac_stopped_at_base": 0.0,
            "frac_escalated": 1.0,
            "n_false_safe": 0,
            "n_delta_opt_gt_0p1": 0,
            "n_delta_opt_gt_1": 0,
            "n_delta_opt_gt_10": 0,
            "max_delta_opt": 0.0,
            "mean_n_fits": float(np.mean(n_fits_fixed)),
            "p50_n_fits": float(np.percentile(n_fits_fixed, 50)),
            "p90_n_fits": float(np.percentile(n_fits_fixed, 90)),
            "max_n_fits": float(np.max(n_fits_fixed)),
        }
    )

    for k in CANDIDATE_K_VALUES:

        m = load_merged(k)

        print(
            f"k={k}: n_risky(delta_vs_reference>0.1)="
            f"{int(m['risky'].sum())}/100"
        )

        all_predictors = PRIMARY_PREDICTORS + OTHER_DIAGNOSTICS_TO_CHECK
        single_results = {}

        for predictor in all_predictors:
            res = threshold_search_single(m, k, predictor)
            threshold_records.append(res)
            single_results[predictor] = res

            print(
                f"  [{predictor}] min_risky_value="
                f"{res['min_risky_value']:.6g}, "
                f"n_escalate={res['n_escalate']}/100 "
                f"({res['frac_escalate']:.0%})"
            )

            escalate_mask = (
                m[predictor] >= res["threshold_escalate_if_gte"]
            ).to_numpy()

            comparison_records.append(
                evaluate_policy(
                    m,
                    k,
                    escalate_mask,
                    rule_text=(
                        f"escalate if {predictor} >= "
                        f"{res['threshold_escalate_if_gte']:.6g}"
                    ),
                )
            )

        # --- OR rule on the two primary predictors ---
        t_tE = single_results["abs_log_ratio_tE"][
            "threshold_escalate_if_gte"
        ]
        t_ds = single_results["delta_spread"][
            "threshold_escalate_if_gte"
        ]

        escalate_or = (
            (m["abs_log_ratio_tE"] >= t_tE)
            | (m["delta_spread"] >= t_ds)
        ).to_numpy()

        comparison_records.append(
            evaluate_policy(
                m,
                k,
                escalate_or,
                rule_text=(
                    f"escalate if abs_log_ratio_tE >= {t_tE:.6g} "
                    f"OR delta_spread >= {t_ds:.6g}"
                ),
            )
        )

        print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame.from_records(threshold_records).to_csv(
        OUT_THRESHOLD_SEARCH, index=False
    )
    pd.DataFrame.from_records(comparison_records).to_csv(
        OUT_COMPARISON, index=False
    )

    print("=" * 70)
    print("FROZEN B2 CANDIDATE POLICY CHECK")
    print("=" * 70)
    assert_frozen_policy_matches_computed_threshold(threshold_records)
    print()

    print("=" * 70)
    print("WROTE")
    print("=" * 70)
    print(f"  {OUT_THRESHOLD_SEARCH}")
    print(f"  {OUT_COMPARISON}")


if __name__ == "__main__":
    main()
