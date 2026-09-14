#!/usr/bin/env python3

"""
Experiment A1 -- DIAGNOSTIC BASELINE for a sequential H0 stopping rule.

STATUS: DIAGNOSTIC BASELINE, development only. This is NOT a proposed
production policy. It exists to characterize which observables,
available during a real run, predict that a partial H0 result is
still far from the Gate-1 oracle. See ADAPTIVE_H0_ANALYSIS.md,
Experiment A1, for the full write-up.

PURPOSE
    Simulate, purely offline from already-computed fits, what a
    sequential execution of the validated `gate1_final18` H0 start
    plan would see after each of its first k=1..18 starts, for every
    extreme100 event. At each k, compute a set of predictors that use
    ONLY information available from fits 1..k of that run (no oracle,
    no information from k+1..18, no per-event hardcoded knowledge).
    Separately, operationalize the checkpoint's validated t0-rescue
    policy from the FULL 18-fit winner's own active_mask (not from
    any hardcoded catalog_row), and check that it reproduces the
    checkpoint's result (rescue needed for exactly one event,
    N(delta_chi2>0.1)=0 after the policy is applied).

INPUTS
    validation/bounds_convergence/results/gate1_oracle_diagnostics.csv
        (built by aggregate_gate1_oracle_diagnostics.py; not
        regenerated here). Uses exactly two of its three `source`
        values:
          - gate1_final18_base_t0margin0 (the 18-start base sequence)
          - gate1_final18_rescue_t0margin0.25 (the t0=0.25 rescue
            domain run, applied conditionally -- see METHOD)
        plus gate1_oracle_52, used ONLY to label delta_chi2 for
        offline evaluation (never as a predictor -- see ORACLE USAGE).

OUTPUTS
    validation/bounds_convergence/results/
        a1_sequential_h0_diagnostics_per_step.csv
            one row per (catalog_row, k) for k=1..18: predictors
            computed from fits 1..k only, plus the offline-only
            delta_chi2_vs_oracle_k / fail_k columns.
        a1_sequential_h0_full18_policy.csv
            one row per catalog_row: the full-18 base winner, its
            active_mask-derived t0_active flag, whether the t0
            rescue was applied (derived from that flag, not
            hardcoded), the resulting policy chi2, and
            delta_chi2_vs_oracle / fail against the 52-fit oracle.
        a1_predictor_vs_failure.csv
            one row per (predictor, bucket): pooled empirical
            P(fail_k | predictor in bucket) over all (catalog_row, k)
            pairs, k=1..18.

WHAT QUESTION THIS ANSWERS
    "Using only observables available from a partial or full run of
    gate1_final18's own fixed 18-start sequence, which predictors are
    associated with the running best H0 solution still being more
    than 0.1 above the Gate-1 oracle -- and does operationalizing the
    validated t0-rescue rule from the full-18 winner's own active_mask
    (rather than a hardcoded event id) reproduce the checkpoint's
    result?"

METHOD -- SEQUENCE ORDER (a stated convention for A1, NOT a validated
production execution order)
    `gate1_final18` was historically validated as four independent
    per-mode multistart runs (one HIDDEN_PARALLAX_TRF_COORDS process
    per mode), not as one single interleaved 18-fit sequence. A1
    needs *some* concrete order to define "fits 1..k", so it uses the
    simplest deterministic convention: concatenate the four modes in
    a fixed order (physical, log_te, log_rho, log_te_rho) and, within
    each mode, use that mode's own start_slot order (1..3, 1..6, 1..8,
    1..1 respectively; already verified constant across all 100
    events in gate1_oracle_diagnostics_audit_start_slot_map.csv for
    this source). This sequence is IDENTICAL for every event because
    (mode, start_slot) -> strategy_id is constant across events for
    gate1_final18_base_t0margin0 (verified in the Phase-0 audit).
    Phase B is explicitly free to study other orders; A1 does not
    claim this one is optimal.

METHOD -- THE VALIDATED t0 RESCUE, OPERATIONALIZED (not hardcoded)
    Per ADAPTIVE_H0_ANALYSIS.md section 0.4, the validated policy is:
    run the full 18-start base plan, look at THAT winner's own
    active_mask, and only if its t0 component is active, rerun the
    full 18-start plan at t0_margin_factor=0.25 for that event and
    take the better of the two winners. This script applies exactly
    that rule using the k=18 (i.e. full-plan) winner's own
    optimizer_active_mask -- catalog_row is never read to decide
    whether to rescue. Whatever event(s) this rule selects are
    reported, and compared against the checkpoint's own statement
    that only catalog_row=36103 required the rescue in extreme100 --
    that comparison is a validation check on this script's method,
    not an input to it.

    Per the user's explicit instruction, the rescue is applied ONLY
    at the full k=18 boundary. For k<18, no rescue is applied under
    any condition, even if the partial winner already has an active
    t0 bound -- firing the rescue earlier than k=18 would be a NEW,
    not-yet-validated policy, and is explicitly out of scope for A1
    (left for Phase B to propose and evaluate against the oracle on
    its own merits).

ORACLE USAGE
    The 52-fit oracle (gate1_oracle_52, pooled minimum chi2 per
    event) is used ONLY to compute delta_chi2_vs_oracle_k / fail_k
    columns for OFFLINE evaluation of the predictors computed at each
    k. It is never read by, or available to, the predictor
    computation itself -- the per-step predictor columns are exactly
    what a production run stopping at step k would have on hand.

NO CHANGES to run_bounds_audit_refit_core.py, run_bounds_audit_refit.py,
run_lsstmonts_catalog_hidden_parallax.py, production_candidate bounds,
the H1 controlled-5 plan, or LRT threshold calibration are made or
proposed by this script.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results"
)

IN_PATH = RESULTS_DIR / "gate1_oracle_diagnostics.csv"

OUT_PER_STEP = RESULTS_DIR / "a1_sequential_h0_diagnostics_per_step.csv"
OUT_FULL18_POLICY = (
    RESULTS_DIR / "a1_sequential_h0_full18_policy.csv"
)
OUT_PREDICTOR_VS_FAILURE = (
    RESULTS_DIR / "a1_predictor_vs_failure.csv"
)

MODES_ORDER = ["physical", "log_te", "log_rho", "log_te_rho"]

DELTA_CHI2_TOLERANCE = 0.1


def _parse_active_mask(raw) -> tuple[int, ...] | None:

    if not isinstance(raw, str):
        return None

    try:
        val = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None

    return tuple(int(x) for x in val)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

    df = pd.read_csv(IN_PATH)

    base = df[
        df["source"] == "gate1_final18_base_t0margin0"
    ].copy()
    rescue = df[
        df["source"] == "gate1_final18_rescue_t0margin0.25"
    ].copy()
    oracle52 = df[df["source"] == "gate1_oracle_52"].copy()

    if base.empty or rescue.empty or oracle52.empty:
        raise RuntimeError(
            "One of the required sources is missing from "
            f"{IN_PATH}. Expected gate1_final18_base_t0margin0, "
            "gate1_final18_rescue_t0margin0.25 and gate1_oracle_52."
        )

    oracle_chi2 = oracle52.groupby("catalog_row")["chi2"].min()
    oracle_chi2.name = "chi2_oracle_52"

    return base, rescue, oracle_chi2


def assign_sequence_index(base: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a fixed, event-independent sequence_index (1..18) to every
    base-run row, per the stated A1 ordering convention (see module
    docstring): modes concatenated in MODES_ORDER, start_slot
    ascending within each mode.
    """

    order_key = base[["mode", "start_slot"]].drop_duplicates()

    if len(order_key) != 18:
        raise RuntimeError(
            "Expected exactly 18 distinct (mode, start_slot) pairs "
            f"in gate1_final18_base_t0margin0, found {len(order_key)}."
        )

    mode_rank = {m: i for i, m in enumerate(MODES_ORDER)}

    unknown_modes = set(order_key["mode"]) - set(mode_rank)
    if unknown_modes:
        raise RuntimeError(
            f"Unknown mode(s) not in MODES_ORDER: {unknown_modes}"
        )

    order_key = order_key.copy()
    order_key["mode_rank"] = order_key["mode"].map(mode_rank)
    order_key = order_key.sort_values(
        ["mode_rank", "start_slot"]
    ).reset_index(drop=True)
    order_key["sequence_index"] = np.arange(len(order_key)) + 1

    out = base.merge(
        order_key[["mode", "start_slot", "sequence_index"]],
        on=["mode", "start_slot"],
        how="left",
        validate="many_to_one",
    )

    if out["sequence_index"].isna().any():
        raise RuntimeError(
            "Failed to assign sequence_index to every base row."
        )

    n_per_event = out.groupby("catalog_row")["sequence_index"].apply(
        lambda s: sorted(s.tolist())
    )
    expected = list(range(1, 19))
    bad_events = [
        row
        for row, seq in n_per_event.items()
        if seq != expected
    ]
    if bad_events:
        raise RuntimeError(
            "sequence_index is not exactly 1..18 for events: "
            f"{bad_events[:10]}"
        )

    return out


def compute_per_step_diagnostics(
    base: pd.DataFrame,
    oracle_chi2: pd.Series,
) -> pd.DataFrame:

    records = []

    for catalog_row, g in base.groupby("catalog_row"):

        g = g.sort_values("sequence_index").reset_index(drop=True)

        masks = g["optimizer_active_mask"].apply(_parse_active_mask)

        chi2 = g["chi2"].to_numpy()
        modes = g["mode"].to_numpy()

        cum_min = np.minimum.accumulate(chi2)

        # last index (0-based) at which cum_min strictly improved
        last_improve_idx = np.zeros(len(chi2), dtype=int)
        last = 0
        for i in range(len(chi2)):
            if i == 0 or cum_min[i] < cum_min[i - 1]:
                last = i
            last_improve_idx[i] = last

        o_chi2 = oracle_chi2.get(catalog_row, float("nan"))

        for k in range(1, len(g) + 1):

            sub_chi2 = chi2[:k]
            sub_modes = modes[:k]

            order = np.argsort(sub_chi2)
            best_pos = order[0]
            chi2_best = float(sub_chi2[best_pos])

            if k >= 2:
                second_pos = order[1]
                chi2_second_best = float(sub_chi2[second_pos])
                delta_spread = chi2_second_best - chi2_best
            else:
                second_pos = None
                chi2_second_best = float("nan")
                delta_spread = float("nan")

            best_row = g.iloc[best_pos]
            best_mask = masks.iloc[best_pos]

            if best_mask is not None and len(best_mask) == 4:
                t0_active, u0_active, tE_active, rho_active = (
                    int(best_mask[0] != 0),
                    int(best_mask[1] != 0),
                    int(best_mask[2] != 0),
                    int(best_mask[3] != 0),
                )
            else:
                t0_active = u0_active = tE_active = rho_active = (
                    np.nan
                )

            n_modes_seen = len(set(sub_modes.tolist()))

            per_mode_best = {}
            for m in set(sub_modes.tolist()):
                per_mode_best[m] = float(
                    sub_chi2[sub_modes == m].min()
                )
            n_modes_agreeing = sum(
                1
                for v in per_mode_best.values()
                if v <= chi2_best + DELTA_CHI2_TOLERANCE
            )

            steps_since_improvement = k - 1 - int(
                last_improve_idx[k - 1]
            )

            if second_pos is not None:
                second_row = g.iloc[second_pos]
                diff_t0_best2nd = abs(
                    float(best_row["t0"]) - float(second_row["t0"])
                )
                diff_u0_best2nd = abs(
                    float(best_row["u0"]) - float(second_row["u0"])
                )
                tE_best = float(best_row["tE"])
                tE_2nd = float(second_row["tE"])
                rho_best = float(best_row["rho"])
                rho_2nd = float(second_row["rho"])
                absdiff_log10_tE_best2nd = (
                    abs(np.log10(tE_best) - np.log10(tE_2nd))
                    if tE_best > 0 and tE_2nd > 0
                    else float("nan")
                )
                absdiff_log10_rho_best2nd = (
                    abs(np.log10(rho_best) - np.log10(rho_2nd))
                    if rho_best > 0 and rho_2nd > 0
                    else float("nan")
                )
            else:
                diff_t0_best2nd = float("nan")
                diff_u0_best2nd = float("nan")
                absdiff_log10_tE_best2nd = float("nan")
                absdiff_log10_rho_best2nd = float("nan")

            delta_chi2_vs_oracle_k = chi2_best - o_chi2
            fail_k = bool(
                delta_chi2_vs_oracle_k > DELTA_CHI2_TOLERANCE
            )

            records.append(
                {
                    "catalog_row": int(catalog_row),
                    "k": k,
                    "chi2_best_k": chi2_best,
                    "chi2_second_best_k": chi2_second_best,
                    "delta_spread_k": delta_spread,
                    "n_modes_seen_k": n_modes_seen,
                    "n_modes_agreeing_k": n_modes_agreeing,
                    "steps_since_improvement_k": (
                        steps_since_improvement
                    ),
                    "winner_t0_active_k": t0_active,
                    "winner_u0_active_k": u0_active,
                    "winner_tE_active_k": tE_active,
                    "winner_rho_active_k": rho_active,
                    "winner_optimizer_success_k": bool(
                        best_row["optimizer_success"]
                    ),
                    "winner_optimizer_optimality_k": float(
                        best_row["optimizer_optimality"]
                    ),
                    "diff_t0_best2nd_k": diff_t0_best2nd,
                    "diff_u0_best2nd_k": diff_u0_best2nd,
                    "absdiff_log10_tE_best2nd_k": (
                        absdiff_log10_tE_best2nd
                    ),
                    "absdiff_log10_rho_best2nd_k": (
                        absdiff_log10_rho_best2nd
                    ),
                    # OFFLINE-ONLY (oracle-derived): never a predictor
                    "chi2_oracle_52": o_chi2,
                    "delta_chi2_vs_oracle_k": delta_chi2_vs_oracle_k,
                    "fail_k": fail_k,
                }
            )

    return pd.DataFrame.from_records(records)


def compute_full18_policy(
    base: pd.DataFrame,
    rescue: pd.DataFrame,
    oracle_chi2: pd.Series,
) -> pd.DataFrame:

    records = []

    for catalog_row, g in base.groupby("catalog_row"):

        winner = g.loc[g["chi2"].idxmin()]
        mask = _parse_active_mask(winner["optimizer_active_mask"])

        t0_active = bool(mask is not None and mask[0] != 0)

        chi2_base_winner = float(winner["chi2"])

        rescue_applied = False
        chi2_rescue_winner = float("nan")
        chi2_policy = chi2_base_winner

        if t0_active:
            rescue_applied = True
            rg = rescue[rescue["catalog_row"] == catalog_row]
            if rg.empty:
                raise RuntimeError(
                    "t0 active for base-18 winner of catalog_row="
                    f"{catalog_row} but no "
                    "gate1_final18_rescue_t0margin0.25 rows found "
                    "for that event."
                )
            chi2_rescue_winner = float(rg["chi2"].min())
            chi2_policy = min(chi2_base_winner, chi2_rescue_winner)

        o_chi2 = oracle_chi2.get(catalog_row, float("nan"))
        delta = chi2_policy - o_chi2

        records.append(
            {
                "catalog_row": int(catalog_row),
                "chi2_base18_winner": chi2_base_winner,
                "winner_mode": winner["mode"],
                "winner_strategy_id": winner["strategy_id"],
                "winner_t0_active": int(t0_active),
                "rescue_applied": int(rescue_applied),
                "chi2_rescue18_winner": chi2_rescue_winner,
                "chi2_policy_final": chi2_policy,
                "chi2_oracle_52": o_chi2,
                "delta_chi2_vs_oracle_final": delta,
                "fail_final": bool(
                    delta > DELTA_CHI2_TOLERANCE
                ),
            }
        )

    return pd.DataFrame.from_records(records).sort_values(
        "catalog_row"
    )


def _bucket_delta_spread(x: float) -> str:

    if not np.isfinite(x):
        return "NA (k=1)"
    if x <= 0.01:
        return "[0, 0.01]"
    if x <= 0.1:
        return "(0.01, 0.1]"
    if x <= 1:
        return "(0.1, 1]"
    if x <= 10:
        return "(1, 10]"
    if x <= 100:
        return "(10, 100]"
    return "(100, inf)"


def _bucket_steps_since(x: float) -> str:

    if x == 0:
        return "0"
    if x <= 2:
        return "1-2"
    if x <= 5:
        return "3-5"
    return "6+"


def compute_predictor_vs_failure(
    per_step: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    def _add(predictor_name, bucket_series, df):
        g = (
            df.assign(_bucket=bucket_series)
            .groupby("_bucket")["fail_k"]
            .agg(n_obs="size", n_fail="sum")
        )
        g["fail_rate"] = g["n_fail"] / g["n_obs"]
        for bucket, row in g.iterrows():
            records.append(
                {
                    "predictor": predictor_name,
                    "bucket": bucket,
                    "n_obs": int(row["n_obs"]),
                    "n_fail": int(row["n_fail"]),
                    "fail_rate": float(row["fail_rate"]),
                }
            )

    _add(
        "delta_spread_k",
        per_step["delta_spread_k"].apply(_bucket_delta_spread),
        per_step,
    )

    _add(
        "n_modes_agreeing_k",
        per_step["n_modes_agreeing_k"].astype(str),
        per_step,
    )

    _add(
        "steps_since_improvement_k",
        per_step["steps_since_improvement_k"].apply(
            _bucket_steps_since
        ),
        per_step,
    )

    _add(
        "winner_t0_active_k",
        per_step["winner_t0_active_k"].astype("Int64").astype(str),
        per_step,
    )

    _add(
        "winner_rho_active_k",
        per_step["winner_rho_active_k"].astype("Int64").astype(str),
        per_step,
    )

    _add(
        "winner_optimizer_success_k",
        per_step["winner_optimizer_success_k"].astype(str),
        per_step,
    )

    _add(
        "k",
        per_step["k"].astype(str).str.zfill(2),
        per_step,
    )

    return pd.DataFrame.from_records(records)


def main() -> None:

    base, rescue, oracle_chi2 = load_tables()
    base = assign_sequence_index(base)

    per_step = compute_per_step_diagnostics(base, oracle_chi2)
    full18_policy = compute_full18_policy(base, rescue, oracle_chi2)
    predictor_vs_failure = compute_predictor_vs_failure(per_step)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_step.to_csv(OUT_PER_STEP, index=False)
    full18_policy.to_csv(OUT_FULL18_POLICY, index=False)
    predictor_vs_failure.to_csv(OUT_PREDICTOR_VS_FAILURE, index=False)

    print("=" * 70)
    print("A1 -- DIAGNOSTIC BASELINE (not a production policy)")
    print("=" * 70)

    print()
    print(
        f"per-step rows: {len(per_step)} "
        f"(= {per_step['catalog_row'].nunique()} events x 18 steps)"
    )

    print()
    print("Full-18 policy (rescue rule operationalized from the")
    print("k=18 winner's own active_mask, not hardcoded):")
    rescued = full18_policy[full18_policy["rescue_applied"] == 1]
    print(
        f"  events with rescue_applied=1: {len(rescued)} "
        f"-> {sorted(rescued['catalog_row'].tolist())}"
    )
    print(
        "  N(fail_final, delta>0.1) = "
        f"{int(full18_policy['fail_final'].sum())}"
    )
    print(
        "  max delta_chi2_vs_oracle_final = "
        f"{full18_policy['delta_chi2_vs_oracle_final'].max():.6g}"
    )

    n_fail_raw18 = int(
        (
            per_step[per_step["k"] == 18]["fail_k"]
        ).sum()
    )
    print(
        "  (for comparison) N(fail) at k=18 WITHOUT rescue = "
        f"{n_fail_raw18}"
    )

    print()
    print("N(fail_k) by k (pure running-min, no rescue):")
    by_k = per_step.groupby("k")["fail_k"].sum().astype(int)
    for k, n in by_k.items():
        print(f"  k={k:2d}: N(fail)={n}")

    print()
    print("=" * 70)
    print("WROTE")
    print("=" * 70)
    print(f"  {OUT_PER_STEP}")
    print(f"  {OUT_FULL18_POLICY}")
    print(f"  {OUT_PREDICTOR_VS_FAILURE}")


if __name__ == "__main__":
    main()
