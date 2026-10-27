#!/usr/bin/env python3

"""
Experiment B1 (part 1) -- fixed-budget H0 base-set search.

STATUS: DIAGNOSTIC / CANDIDATE GENERATION, development-set analysis
(extreme100). Not a production policy. See ADAPTIVE_H0_ANALYSIS.md,
Experiment B1, for the full write-up and the Phase-B ground rules
this builds on (Experiment 0, Experiment A1).

PURPOSE
    For each fixed base-set size k in {2, 3, 4, 5, 6}, search the 52
    canonical (mode, strategy_id) H0 strategies (the same 52-strategy
    pool as the Gate-1 oracle) for the base-set(s) of size k that
    minimize the number of extreme100 events left with
    delta_chi2_vs_base_domain_oracle > 0.1 (the primary objective),
    and, among primary-tied candidates, rank by secondary severity
    criteria (N(delta>1), N(delta>10), max delta, summed positive
    excess). This produces, per k, a small set of competitive
    candidate base sets (not just one "the best set") spanning
    different mode compositions where available, plus the residual
    failing events for each candidate and a full rescue matrix
    (residual event x every remaining strategy) to support Phase-B2
    rescue-set design.

    Candidate SET SELECTION is a design-time, offline activity: it is
    legitimate to use the base-domain oracle to choose a fixed set
    that will later be applied identically to every event. This is
    different from, and must not be confused with, using the oracle
    inside a per-event runtime decision (which would be leakage) --
    see ORACLE USAGE below.

INPUTS
    validation/bounds_convergence/results/gate1_oracle_diagnostics.csv
        (source == "gate1_oracle_52"; built by
        aggregate_gate1_oracle_diagnostics.py, not regenerated here)
    validation/bounds_convergence/results/gate1_h0_coverage_curve.csv
        (pre-existing exact-MILP maximum-coverage-at-tolerance-0.1
        result for k=1..18 over the same 52-strategy pool, keyed by
        mode:start_slot; used ONLY as an independent cross-check that
        this script's primary-objective optimum at each k agrees with
        that earlier, differently-computed result -- not reused as
        this script's answer, and not blindly trusted without that
        cross-check)

OUTPUTS (validation/bounds_convergence/results/)
    b1_fixed_budget_candidate_sets.csv
        one row per (k, candidate_rank): the strategies in the
        candidate ("mode/strategy_id" list), its mode-composition
        signature, whether it is lexicographically primary-optimal,
        why it was kept (lex-optimal vs diversity alternative),
        N(delta>0.1/1/10), max_delta, sum/mean positive excess, and
        the cross-check result against gate1_h0_coverage_curve.csv.
    b1_residual_failing_events.csv
        one row per (k, candidate_rank, catalog_row) with
        delta_chi2_vs_base_domain_oracle > 0.1 under that candidate.
    b1_rescue_matrix.csv
        one row per (k, candidate_rank, catalog_row [residual only],
        remaining_strategy): chi2 if that single additional strategy
        is added to the candidate set's own result for that event,
        the resulting delta, and whether that rescues the event
        (delta <= 0.1).
    b1_rescue_strategy_summary.csv
        one row per (k, candidate_rank, remaining_strategy): how many
        of that candidate's residual events it rescues.

WHAT QUESTION THIS ANSWERS
    "At a fixed, small H0 fit budget k, which sets of k strategies
    (chosen offline, using the oracle for design only) come closest
    to the base-domain oracle, and for whichever events remain hard
    under each such set, which additional strategy would rescue
    them?" This does NOT answer "which predictor should trigger a
    rescue" -- that is Experiment B1 part 2
    (analyze_b1_fixed_cost_predictors.py).

SEARCH METHOD (exact, not heuristic, for every k in this script)
    k=2..6: FULL EXHAUSTIVE enumeration of every C(52, k) subset,
    computed in two vectorized passes so that exact exhaustive search
    stays cheap even at k=6 (C(52,6) = 20,358,520):
      pass 1 -- for every subset, compute only the cheap PRIMARY
        objective (N(delta>0.1), via a precomputed per-strategy
        boolean "covers this event at tolerance 0.1" matrix and a
        vectorized `.any()` over each subset's columns), and record
        just the smallest `TOP_M` subsets' global indices by that
        primary value (their full chi2-based metrics are NOT yet
        computed, to keep this pass cheap: this took well under 30s
        for k=6 on this machine, see ADAPTIVE_H0_ANALYSIS.md for the
        measured timings);
      pass 2 -- regenerate combinations again (same cheap
        `itertools.combinations` order, ~20s for k=6) and, only for
        the `TOP_M` global indices identified in pass 1, compute the
        full chi2-based secondary metrics needed for lexicographic
        ranking.
    This is exact (no MILP, no heuristic, no early stopping) for every
    k studied. TOP_M is generous (see TOP_M below) so it is not a
    hidden approximation of which subsets are considered for the
    primary objective; it only bounds how many near-optimal subsets
    get full secondary-metric evaluation in pass 2.

DETERMINISM
    Every step of this script is a fixed, mechanical rule with no
    manual or visual selection and no dependence on predictor results
    (analyze_b1_fixed_cost_predictors.py runs strictly afterward and
    only reads this script's output): pass-1 top-M selection uses an
    explicit stable sort (not argpartition, whose tie-handling is not
    guaranteed reproducible); pass-2 candidate order is the sorted
    (ascending global flat index) order, not Python set iteration
    order; and see select_saved_candidates() for the exact,
    documented rule that picks the saved lexicographic-optimal and
    diversity-alternative rows. Re-running this script against the
    same gate1_oracle_diagnostics.csv reproduces the same output.

ORACLE USAGE
    chi2_oracle_52 (base-domain oracle, t0_margin_factor=0, pooled
    minimum over the same 52 strategies) is used for: (a) selecting
    the candidate base sets themselves (design-time, offline -- see
    PURPOSE), and (b) labeling delta/safe-unsafe for those candidates,
    strictly for offline reporting. No per-event runtime decision is
    simulated or proposed by this script. This is development-set
    (extreme100) analysis; no claim of generalization is made.
"""

from __future__ import annotations

import itertools
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results"
)

IN_PATH = RESULTS_DIR / "gate1_oracle_diagnostics.csv"
COVERAGE_CURVE_PATH = RESULTS_DIR / "gate1_h0_coverage_curve.csv"

OUT_CANDIDATES = RESULTS_DIR / "b1_fixed_budget_candidate_sets.csv"
OUT_RESIDUAL = RESULTS_DIR / "b1_residual_failing_events.csv"
OUT_RESCUE_MATRIX = RESULTS_DIR / "b1_rescue_matrix.csv"
OUT_RESCUE_SUMMARY = RESULTS_DIR / "b1_rescue_strategy_summary.csv"

TOL = 0.1
K_VALUES = [2, 3, 4, 5, 6]

# Pass-1 candidate pool size per k: how many of the best-by-primary-
# objective subsets get full pass-2 evaluation. Generous relative to
# the handful of subsets actually saved (see select_saved_candidates).
TOP_M = 300

# Pass-1/pass-2 chunk size for itertools.combinations enumeration.
CHUNK_SIZE = 500_000

# Cap on how many diversity alternatives (beyond the exact
# lexicographic-optimal tier) are kept per k.
MAX_DIVERSITY_ALTERNATIVES = 5


def load_matrix() -> tuple[np.ndarray, list[str], list[str], np.ndarray]:

    df = pd.read_csv(IN_PATH)
    o = df[df["source"] == "gate1_oracle_52"].copy()

    if o.empty:
        raise RuntimeError(
            f"No gate1_oracle_52 rows found in {IN_PATH}."
        )

    o["strategy_key"] = o["mode"] + "/" + o["strategy_id"]

    mat = o.pivot_table(
        index="catalog_row",
        columns="strategy_key",
        values="chi2",
        aggfunc="min",
    )

    if mat.isna().any().any():
        raise RuntimeError(
            "Incomplete event x strategy matrix for gate1_oracle_52 "
            "-- expected every (catalog_row, mode, strategy_id) to "
            "be present (this was verified in Experiment 0's "
            "integrity check; re-run audit_gate1_oracle_diagnostics."
            "py if this fires)."
        )

    strategy_keys = list(mat.columns)
    modes = [s.split("/", 1)[0] for s in strategy_keys]

    if len(strategy_keys) != 52:
        raise RuntimeError(
            f"Expected 52 distinct (mode, strategy_id) strategies, "
            f"found {len(strategy_keys)}."
        )

    return mat.to_numpy(), strategy_keys, modes, np.array(mat.index)


def exhaustive_search(
    A: np.ndarray,
    oracle: np.ndarray,
    cover: np.ndarray,
    k: int,
) -> pd.DataFrame:
    """
    Exact exhaustive search over all C(n, k) subsets of the 52
    strategies. Two-pass, see module docstring SEARCH METHOD.
    Returns up to TOP_M candidates (by primary objective) with full
    lexicographic metrics, one row per candidate, NOT yet deduplicated
    or filtered down to what will actually be saved.
    """

    n = A.shape[1]
    n_events = A.shape[0]

    t_start = time.time()

    # ---------------- Pass 1: primary objective only ----------------
    best_vals: list[int] = []
    best_flat_indices: list[int] = []

    flat_offset = 0

    gen = itertools.combinations(range(n), k)

    while True:
        block = list(itertools.islice(gen, CHUNK_SIZE))
        if not block:
            break

        combos = np.asarray(block, dtype=np.int32)
        sub_cover = cover[:, combos]  # (n_events, m, k)
        covered = sub_cover.any(axis=2)  # (n_events, m)
        n_gt_0p1 = n_events - covered.sum(axis=0)  # (m,)

        best_vals.extend(n_gt_0p1.tolist())
        best_flat_indices.extend(
            range(flat_offset, flat_offset + len(block))
        )

        flat_offset += len(block)

    best_vals_arr = np.asarray(best_vals, dtype=np.int32)

    # DETERMINISM: a stable sort (ties broken by ascending flat index,
    # i.e. itertools.combinations' own fixed enumeration order) rather
    # than argpartition, whose tie-handling at the TOP_M boundary is
    # not guaranteed reproducible. This makes "the TOP_M subsets by
    # primary objective" a well-defined, reproducible set given fixed
    # input data -- not dependent on numpy/Python internals beyond a
    # documented, explicit stable sort.
    order = np.argsort(best_vals_arr, kind="stable")
    top_m_idx_local = order[: min(TOP_M, len(order))]

    # DETERMINISM: sorted(), not a bare set, so the membership check
    # below in pass 2 never depends on Python set iteration order.
    keep_flat_indices = sorted(
        int(best_flat_indices[i]) for i in top_m_idx_local
    )

    t_pass1 = time.time()

    # ---------------- Pass 2: full metrics for kept candidates -------
    kept_records = []

    flat_offset = 0
    gen = itertools.combinations(range(n), k)

    while True:
        block = list(itertools.islice(gen, CHUNK_SIZE))
        if not block:
            break

        block_start = flat_offset
        block_end = flat_offset + len(block)

        # DETERMINISM: iterate the sorted list (not a bare set) so the
        # order in which candidates are appended to kept_records is
        # always ascending global flat index, for every run.
        wanted_local = [
            i - block_start
            for i in keep_flat_indices
            if block_start <= i < block_end
        ]

        if wanted_local:

            combos = np.asarray(
                [block[i] for i in wanted_local],
                dtype=np.int32,
            )

            candidate_chi2 = A[:, combos].min(axis=2)  # (n_events, m)
            delta = candidate_chi2 - oracle[:, None]

            n_gt_0p1 = (delta > 0.1).sum(axis=0)
            n_gt_1 = (delta > 1.0).sum(axis=0)
            n_gt_10 = (delta > 10.0).sum(axis=0)
            max_delta = delta.max(axis=0)
            pos = np.clip(delta, 0.0, None)
            sum_excess = pos.sum(axis=0)
            mean_excess = pos.mean(axis=0)

            for j, combo in enumerate(combos):
                kept_records.append(
                    {
                        "combo": tuple(int(x) for x in combo),
                        "n_gt_0p1": int(n_gt_0p1[j]),
                        "n_gt_1": int(n_gt_1[j]),
                        "n_gt_10": int(n_gt_10[j]),
                        "max_delta": float(max_delta[j]),
                        "sum_excess": float(sum_excess[j]),
                        "mean_excess": float(mean_excess[j]),
                    }
                )

        flat_offset = block_end

    t_pass2 = time.time()

    print(
        f"  [k={k}] pass1={t_pass1 - t_start:.2f}s "
        f"pass2={t_pass2 - t_pass1:.2f}s "
        f"n_candidates_kept={len(kept_records)} "
        f"min_n_gt_0p1={best_vals_arr.min()}"
    )

    return pd.DataFrame.from_records(kept_records)


def select_saved_candidates(
    pool: pd.DataFrame,
    strategy_keys: list[str],
    modes: list[str],
) -> pd.DataFrame:
    """
    From the pass-2 candidate pool, select what actually gets saved.
    This is a FIXED, MECHANICAL rule -- there is no manual or
    visual selection, and no predictor computed by
    analyze_b1_fixed_cost_predictors.py is consulted here (that
    script runs strictly after this one and only reads its output):

      1. Sort the pool lexicographically on
         (n_gt_0p1, n_gt_1, n_gt_10, max_delta, sum_excess), all
         ascending, using an explicit stable sort (`kind="mergesort"`)
         so any residual ties are broken by the pool's own
         deterministic pass-2 order (itself ascending global flat
         index, see exhaustive_search's DETERMINISM notes) -- never by
         an unspecified sort algorithm's tie-handling.
      2. Save every row exactly tied with row 0 on all five criteria:
         these are "lexicographic_optimal".
      3. Walk the remaining rows in that same fixed sorted order;
         save a row as a "diversity_alternative" iff its
         mode-composition signature (the sorted (mode, count)
         multiset of its k strategies) has not already been saved,
         stopping after MAX_DIVERSITY_ALTERNATIVES such rows.

    Given the same input table, this always returns the same output
    -- re-run this script from IN_PATH and diff the result if that
    needs re-verifying.
    """

    pool = pool.sort_values(
        [
            "n_gt_0p1",
            "n_gt_1",
            "n_gt_10",
            "max_delta",
            "sum_excess",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    best_tuple = tuple(
        pool.iloc[0][
            ["n_gt_0p1", "n_gt_1", "n_gt_10", "max_delta", "sum_excess"]
        ]
    )

    def _composition(combo: tuple[int, ...]) -> tuple:
        return tuple(
            sorted(Counter(modes[i] for i in combo).items())
        )

    pool["composition"] = pool["combo"].apply(_composition)

    is_optimal = (
        (pool["n_gt_0p1"] == best_tuple[0])
        & (pool["n_gt_1"] == best_tuple[1])
        & (pool["n_gt_10"] == best_tuple[2])
        & np.isclose(pool["max_delta"], best_tuple[3])
        & np.isclose(pool["sum_excess"], best_tuple[4])
    )

    saved_rows = []
    seen_compositions = set()

    for _, row in pool[is_optimal].iterrows():
        saved_rows.append((row, "lexicographic_optimal"))
        seen_compositions.add(row["composition"])

    n_diversity_added = 0
    for _, row in pool[~is_optimal].iterrows():
        if n_diversity_added >= MAX_DIVERSITY_ALTERNATIVES:
            break
        if row["composition"] in seen_compositions:
            continue
        saved_rows.append((row, "diversity_alternative"))
        seen_compositions.add(row["composition"])
        n_diversity_added += 1

    records = []
    for rank, (row, reason) in enumerate(saved_rows, start=1):
        combo = row["combo"]
        records.append(
            {
                "candidate_rank": rank,
                "selection_reason": reason,
                "n_strategies": len(combo),
                "strategies": " + ".join(
                    strategy_keys[i] for i in combo
                ),
                "mode_composition": " + ".join(
                    f"{m}:{c}" for m, c in row["composition"]
                ),
                "combo_indices": " ".join(str(i) for i in combo),
                "n_gt_0p1": row["n_gt_0p1"],
                "n_gt_1": row["n_gt_1"],
                "n_gt_10": row["n_gt_10"],
                "max_delta": row["max_delta"],
                "sum_excess_positive": row["sum_excess"],
                "mean_excess_positive": row["mean_excess"],
                "is_lexicographic_optimal": bool(
                    reason == "lexicographic_optimal"
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def cross_check_against_existing_milp(
    k: int, n_gt_0p1_here: int
) -> str:

    curve = pd.read_csv(COVERAGE_CURVE_PATH)
    row = curve[curve["k"] == k]

    if row.empty:
        return f"no gate1_h0_coverage_curve.csv row for k={k}"

    existing = int(row.iloc[0]["n_delta_gt_0p1"])

    if existing == n_gt_0p1_here:
        return f"MATCH (existing MILP n_delta_gt_0p1={existing})"

    return (
        "MISMATCH vs existing MILP: "
        f"this_exhaustive={n_gt_0p1_here} "
        f"existing_milp={existing}"
    )


def build_residual_and_rescue(
    A: np.ndarray,
    oracle: np.ndarray,
    strategy_keys: list[str],
    rows: np.ndarray,
    candidates: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    n = A.shape[1]

    residual_records = []
    rescue_records = []

    for _, cand in candidates.iterrows():

        combo = tuple(
            int(x) for x in cand["combo_indices"].split()
        )
        remaining = [i for i in range(n) if i not in combo]

        candidate_chi2 = A[:, combo].min(axis=1)  # (n_events,)
        delta = candidate_chi2 - oracle

        residual_mask = delta > TOL
        residual_idx = np.where(residual_mask)[0]

        for ei in residual_idx:
            residual_records.append(
                {
                    "k": k,
                    "candidate_rank": cand["candidate_rank"],
                    "catalog_row": int(rows[ei]),
                    "chi2_candidate": float(candidate_chi2[ei]),
                    "chi2_oracle_52": float(oracle[ei]),
                    "delta_chi2_vs_base_domain_oracle": float(
                        delta[ei]
                    ),
                }
            )

            for si in remaining:
                chi2_after = min(
                    float(candidate_chi2[ei]), float(A[ei, si])
                )
                delta_after = chi2_after - float(oracle[ei])
                rescue_records.append(
                    {
                        "k": k,
                        "candidate_rank": cand["candidate_rank"],
                        "catalog_row": int(rows[ei]),
                        "remaining_strategy": strategy_keys[si],
                        "chi2_remaining_strategy_alone": float(
                            A[ei, si]
                        ),
                        "chi2_candidate_before": float(
                            candidate_chi2[ei]
                        ),
                        "chi2_after_adding_strategy": chi2_after,
                        "delta_after_adding_strategy": delta_after,
                        "rescued": bool(delta_after <= TOL),
                    }
                )

    residual_df = pd.DataFrame.from_records(residual_records)
    rescue_df = pd.DataFrame.from_records(rescue_records)

    if rescue_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "k",
                "candidate_rank",
                "remaining_strategy",
                "n_residual_events",
                "n_rescued",
                "rescue_rate",
            ]
        )
    else:
        summary_df = (
            rescue_df.groupby(
                ["k", "candidate_rank", "remaining_strategy"]
            )
            .agg(
                n_residual_events=("rescued", "size"),
                n_rescued=("rescued", "sum"),
            )
            .reset_index()
        )
        summary_df["rescue_rate"] = (
            summary_df["n_rescued"] / summary_df["n_residual_events"]
        )

    return residual_df, rescue_df, summary_df


def main() -> None:

    A, strategy_keys, modes, rows = load_matrix()
    oracle = A.min(axis=1)
    cover = (A - oracle[:, None]) <= TOL

    print(f"Loaded matrix: {A.shape[0]} events x {A.shape[1]} strategies")
    print(f"Base-domain oracle chi2 range: "
          f"[{oracle.min():.4g}, {oracle.max():.4g}]")
    print()

    all_candidates = []
    all_residual = []
    all_rescue = []
    all_summary = []

    for k in K_VALUES:

        print(f"=== k={k} ===")

        pool = exhaustive_search(A, oracle, cover, k)
        saved = select_saved_candidates(pool, strategy_keys, modes)

        cross_check = cross_check_against_existing_milp(
            k, int(saved.iloc[0]["n_gt_0p1"])
        )
        saved["cross_check_vs_existing_milp"] = cross_check
        saved.insert(0, "k", k)

        print(
            f"  saved {len(saved)} candidates "
            f"({(saved['is_lexicographic_optimal']).sum()} "
            f"lex-optimal, "
            f"{(~saved['is_lexicographic_optimal']).sum()} "
            f"diversity alternatives)"
        )
        print(f"  cross-check: {cross_check}")

        residual_df, rescue_df, summary_df = (
            build_residual_and_rescue(
                A, oracle, strategy_keys, rows, saved, k
            )
        )

        all_candidates.append(saved)
        all_residual.append(residual_df)
        all_rescue.append(rescue_df)
        all_summary.append(summary_df)

        print()

    candidates_out = pd.concat(all_candidates, ignore_index=True)
    residual_out = pd.concat(all_residual, ignore_index=True)
    rescue_out = pd.concat(all_rescue, ignore_index=True)
    summary_out = pd.concat(all_summary, ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    candidates_out.to_csv(OUT_CANDIDATES, index=False)
    residual_out.to_csv(OUT_RESIDUAL, index=False)
    rescue_out.to_csv(OUT_RESCUE_MATRIX, index=False)
    summary_out.to_csv(OUT_RESCUE_SUMMARY, index=False)

    print("=" * 70)
    print("WROTE")
    print("=" * 70)
    print(f"  {OUT_CANDIDATES} ({len(candidates_out)} rows)")
    print(f"  {OUT_RESIDUAL} ({len(residual_out)} rows)")
    print(f"  {OUT_RESCUE_MATRIX} ({len(rescue_out)} rows)")
    print(f"  {OUT_RESCUE_SUMMARY} ({len(summary_out)} rows)")


if __name__ == "__main__":
    main()
