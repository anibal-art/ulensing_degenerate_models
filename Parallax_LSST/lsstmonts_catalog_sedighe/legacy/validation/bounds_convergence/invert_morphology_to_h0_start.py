#!/usr/bin/env python3
"""
M3 (inversion) + M4 -- turn each event's observed morphology (T25/T50,
T75/T50 width ratios + T50 absolute scale + t_peak) into up to 3
genuinely distinct morphology-informed H0 start candidates
(t0, u0, tE, rho), using the frozen FSPL dimensionless lookup table
(build_fspl_dimensionless_lookup.py). No oracle/truth is used here.

Since the pure FSPL shape is symmetric (T25_left=T25_right etc.) but
real data can be censored on one side, the "effective" full width fed
into the ratio lookup is:
  - the measured full width, if both sides are measurable;
  - 2x the measurable half-width, if exactly one side is measurable
    (assumes the model's built-in left/right symmetry for the missing
    side -- an explicit, documented approximation, not a silent one);
  - not_measurable otherwise (no candidate is produced for that Tq).

An event needs an effective T25, T50 and T75 to be inverted; if any is
missing, no morphology candidate is produced for it (recorded, not
fabricated).

M4 degeneracy handling: after ranking all lookup-grid points by
mismatch = sqrt((dT25/T50)^2 + (dT75/T50)^2), the best match is
candidate 1. Additional candidates are accepted (up to 3 total) only
if they lie outside a 0.3-dex box in (log10 u0, log10 rho) around
every already-accepted candidate AND have mismatch within 3x the best
candidate's mismatch -- i.e. genuinely separate, comparably-good
basins of the inversion, not near-duplicates of the same minimum.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

DISTINCTNESS_DEX = 0.3
MISMATCH_RATIO_CAP = 3.0
MAX_CANDIDATES = 3


def effective_width(entry_value, left, right, left_c, right_c):
    if not left_c and not right_c and entry_value is not None:
        return float(entry_value), "both_sides"
    if not left_c and left is not None and right_c:
        return 2.0 * float(left), "left_only_mirrored"
    if not right_c and right is not None and left_c:
        return 2.0 * float(right), "right_only_mirrored"
    return None, "not_measurable"


def invert_one_event(obs_T25T50, obs_T75T50, lookup):
    d25 = lookup["T25_over_T50"].values - obs_T25T50
    d75 = lookup["T75_over_T50"].values - obs_T75T50
    mismatch = np.sqrt(d25 ** 2 + d75 ** 2)

    order = np.argsort(mismatch)
    best_mismatch = mismatch[order[0]]

    candidates = []
    log_u0 = np.log10(lookup["u0"].values)
    log_rho = np.log10(lookup["rho"].values)

    for idx in order:
        if mismatch[idx] > MISMATCH_RATIO_CAP * best_mismatch:
            break
        lu0, lrho = log_u0[idx], log_rho[idx]
        distinct = True
        for c in candidates:
            if abs(lu0 - c["log_u0"]) < DISTINCTNESS_DEX and abs(lrho - c["log_rho"]) < DISTINCTNESS_DEX:
                distinct = False
                break
        if not distinct:
            continue
        candidates.append({
            "u0": float(lookup["u0"].values[idx]),
            "rho": float(lookup["rho"].values[idx]),
            "log_u0": lu0,
            "log_rho": lrho,
            "T50_over_tE": float(lookup["T50_over_tE"].values[idx]),
            "mismatch": float(mismatch[idx]),
        })
        if len(candidates) >= MAX_CANDIDATES:
            break

    return candidates


def main():
    lookup = pd.read_csv(os.path.join(HERE, "results", "fspl_dimensionless_lookup.csv"))
    morph = pd.read_csv(os.path.join(HERE, "results", "morphology_M0_M1_M2_extreme100.csv"))

    rows = []
    for _, r in morph.iterrows():
        row = int(r["catalog_row"])
        if not bool(r["measurable"]):
            rows.append({"catalog_row": row, "invertible": False, "reason": "morphology_not_measurable"})
            continue

        eff25, src25 = effective_width(r["T25_value"], r["T25_left"], r["T25_right"], r["T25_left_censored"], r["T25_right_censored"])
        eff50, src50 = effective_width(r["T50_value"], r["T50_left"], r["T50_right"], r["T50_left_censored"], r["T50_right_censored"])
        eff75, src75 = effective_width(r["T75_value"], r["T75_left"], r["T75_right"], r["T75_left_censored"], r["T75_right_censored"])

        if eff25 is None or eff50 is None or eff75 is None or eff50 <= 0:
            rows.append({
                "catalog_row": row, "invertible": False,
                "reason": f"width_not_measurable (T25:{src25}, T50:{src50}, T75:{src75})",
            })
            continue

        obs_T25T50 = eff25 / eff50
        obs_T75T50 = eff75 / eff50

        candidates = invert_one_event(obs_T25T50, obs_T75T50, lookup)
        if not candidates:
            rows.append({"catalog_row": row, "invertible": False, "reason": "no_lookup_match_within_cap"})
            continue

        t_peak = float(r["t_peak_morph"])
        for rank, c in enumerate(candidates, start=1):
            tE_morph = eff50 / c["T50_over_tE"]
            rows.append({
                "catalog_row": row,
                "invertible": True,
                "candidate_rank": rank,
                "n_candidates": len(candidates),
                "obs_T25_over_T50": obs_T25T50,
                "obs_T75_over_T50": obs_T75T50,
                "T25_source": src25, "T50_source": src50, "T75_source": src75,
                "mismatch": c["mismatch"],
                "t0_morph": t_peak,
                "u0_morph": c["u0"],
                "rho_morph": c["rho"],
                "tE_morph": tE_morph,
            })

    out = pd.DataFrame(rows)
    out_csv = os.path.join(HERE, "results", "morphology_M3_M4_candidates.csv")
    out.to_csv(out_csv, index=False)

    n_events = out["catalog_row"].nunique()
    n_invertible = out.loc[out["invertible"] == True, "catalog_row"].nunique()
    print(f"events total={n_events} invertible={n_invertible}")
    if "n_candidates" in out.columns:
        print(out.loc[out["candidate_rank"] == 1, "n_candidates"].value_counts())
    print("DONE ->", out_csv)


if __name__ == "__main__":
    main()
