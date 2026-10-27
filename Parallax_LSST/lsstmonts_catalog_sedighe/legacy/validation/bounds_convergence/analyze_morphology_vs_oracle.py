#!/usr/bin/env python3
"""
M5 -- offline diagnostic, ZERO new TRF fits. Compares morphology
candidates (already computed, no oracle used to build them) against
truth, the H0 oracle52 winner, and the raw old_H0 vector, in the
dimensionless coordinates used throughout this log:

    d_t0  = (theta_cand.t0 - ref.t0) / ref.tE
    d_u0  = theta_cand.u0 - ref.u0
    d_ltE = log(theta_cand.tE / ref.tE)
    d_lrho= log(theta_cand.rho / ref.rho)
    D     = sqrt(d_t0^2 + d_u0^2 + d_ltE^2 + d_lrho^2)   (joint distance)

Critical group = the 12 events where no truth-anchored strategy
reproduces oracle52 within 0.1 and an old_H0-anchored strategy is
required (established exactly in prior sections of this log).
Rest = the other 88 events.

M6 preregistered decision criterion (frozen before this script's
output is read): morphology is promising only if it (a) clearly
improves the MEDIAN distance to the H0 oracle winner on the 12
critical events relative to truth's own distance, (b) moves a clear
MAJORITY of those 12 closer, not just 1-2 outliers, and (c) is not
simply a proxy of truth (i.e. D(morphology,truth) is not trivially
~0 while D(morphology,oracle) improvement is entirely explained by
morphology==truth).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

CRITICAL_12 = [62786, 83189, 567800, 579320, 82728, 574423,
               557860, 558189, 567365, 35927, 563210, 79501]


def dimensionless_distance(t0_c, u0_c, tE_c, rho_c, t0_r, u0_r, tE_r, rho_r):
    d_t0 = (t0_c - t0_r) / tE_r
    d_u0 = u0_c - u0_r
    d_ltE = np.log(tE_c / tE_r)
    d_lrho = np.log(rho_c / rho_r)
    D = np.sqrt(d_t0 ** 2 + d_u0 ** 2 + d_ltE ** 2 + d_lrho ** 2)
    return D, d_t0, d_u0, d_ltE, d_lrho


def main():
    cand = pd.read_csv(os.path.join(HERE, "results", "morphology_M3_M4_candidates.csv"))
    cand1 = cand[(cand["invertible"] == True) & (cand["candidate_rank"] == 1)].set_index("catalog_row")

    truth = pd.read_csv(os.path.join(HERE, "results", "b1c_event_features.csv")).set_index("catalog_row")

    oracle_df = pd.read_csv(os.path.join(HERE, "results", "gate1_oracle_diagnostics.csv"))

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
    piv_chi2 = d52.pivot_table(index="catalog_row", columns="canon", values="chi2", aggfunc="min")
    piv_t0 = d52.pivot_table(index="catalog_row", columns="canon", values="t0", aggfunc="min")
    piv_u0 = d52.pivot_table(index="catalog_row", columns="canon", values="u0", aggfunc="min")
    piv_tE = d52.pivot_table(index="catalog_row", columns="canon", values="tE", aggfunc="min")
    piv_rho = d52.pivot_table(index="catalog_row", columns="canon", values="rho", aggfunc="min")

    old_h0 = pd.read_csv(os.path.join(HERE, "results", "old_h0_vector_current_objective.csv")).set_index("catalog_row")

    morph_qual = pd.read_csv(os.path.join(HERE, "results", "morphology_M0_M1_M2_extreme100.csv")).set_index("catalog_row")

    rows = []
    for row in truth.index:
        if row not in cand1.index:
            continue
        mc = cand1.loc[row]
        t0_m, u0_m, tE_m, rho_m = mc["t0_morph"], mc["u0_morph"], mc["tE_morph"], mc["rho_morph"]

        t0_t, u0_t, tE_t, rho_t = (truth.loc[row, c] for c in
                                    ["t0_true", "u0_true", "tE_true", "rho_true"])

        winner_col = piv_chi2.loc[row].idxmin()
        t0_o = piv_t0.loc[row, winner_col]
        u0_o = piv_u0.loc[row, winner_col]
        tE_o = piv_tE.loc[row, winner_col]
        rho_o = piv_rho.loc[row, winner_col]

        t0_old, u0_old, tE_old, rho_old = (old_h0.loc[row, c] for c in
                                            ["old_h0_t0", "old_h0_u0", "old_h0_tE", "old_h0_rho"])

        D_m_oracle, *_ = dimensionless_distance(t0_m, u0_m, tE_m, rho_m, t0_o, u0_o, tE_o, rho_o)
        D_t_oracle, *_ = dimensionless_distance(t0_t, u0_t, tE_t, rho_t, t0_o, u0_o, tE_o, rho_o)
        D_m_truth, *_ = dimensionless_distance(t0_m, u0_m, tE_m, rho_m, t0_t, u0_t, tE_t, rho_t)
        D_m_oldh0, *_ = dimensionless_distance(t0_m, u0_m, tE_m, rho_m, t0_old, u0_old, tE_old, rho_old)
        D_old_oracle, *_ = dimensionless_distance(t0_old, u0_old, tE_old, rho_old, t0_o, u0_o, tE_o, rho_o)

        rows.append({
            "catalog_row": int(row),
            "critical": row in CRITICAL_12,
            "D_morph_oracle": D_m_oracle,
            "D_truth_oracle": D_t_oracle,
            "D_morph_truth": D_m_truth,
            "D_morph_oldH0": D_m_oldh0,
            "D_oldH0_oracle_ref": D_old_oracle,
            "morph_closer_than_truth": bool(D_m_oracle < D_t_oracle),
            "oracle_winner_strategy": winner_col,
            "peak_SN": morph_qual.loc[row, "peak_SN"] if row in morph_qual.index else np.nan,
            "n_contributing_bands": morph_qual.loc[row, "n_contributing_bands"] if row in morph_qual.index else np.nan,
            "A50": morph_qual.loc[row, "A50"] if row in morph_qual.index else np.nan,
        })

    out = pd.DataFrame(rows).set_index("catalog_row")
    out.to_csv(os.path.join(HERE, "results", "morphology_M5_offline_diagnostic.csv"))

    crit = out[out["critical"]]
    rest = out[~out["critical"]]

    print("=" * 100)
    print(f"CRITICAL GROUP (n={len(crit)})")
    print("=" * 100)
    print(crit[["D_morph_oracle", "D_truth_oracle", "D_morph_truth", "D_morph_oldH0", "morph_closer_than_truth"]].to_string())
    print()
    print("median D(morph,oracle)  =", crit["D_morph_oracle"].median())
    print("median D(truth,oracle)  =", crit["D_truth_oracle"].median())
    print("N morph closer than truth:", int(crit["morph_closer_than_truth"].sum()), "/", len(crit))
    print("median D(morph,truth) [proxy check]:", crit["D_morph_truth"].median())
    print("median D(oldH0-raw,oracle) [reference: how far the actual bridge vector itself sits]:", crit["D_oldH0_oracle_ref"].median())

    print()
    print("=" * 100)
    print(f"REST GROUP (n={len(rest)})")
    print("=" * 100)
    print("median D(morph,oracle)  =", rest["D_morph_oracle"].median())
    print("median D(truth,oracle)  =", rest["D_truth_oracle"].median())
    print("N morph closer than truth:", int(rest["morph_closer_than_truth"].sum()), "/", len(rest))

    print()
    print("=" * 100)
    print("Correlation of morph-improvement with quality diagnostics (descriptive only)")
    print("=" * 100)
    out["improvement"] = out["D_truth_oracle"] - out["D_morph_oracle"]
    print(out[["improvement", "peak_SN", "n_contributing_bands", "A50"]].corr(numeric_only=True)["improvement"])

    print()
    print("=" * 100)
    print("M6 PREREGISTERED DECISION")
    print("=" * 100)
    med_morph = crit["D_morph_oracle"].median()
    med_truth = crit["D_truth_oracle"].median()
    n_closer = int(crit["morph_closer_than_truth"].sum())
    n_total = len(crit)
    proxy_flag = crit["D_morph_truth"].median() < 0.05 * crit["D_truth_oracle"].median()

    clear_median_improvement = med_morph < 0.7 * med_truth  # documented "clearly improves" bar
    clear_majority = n_closer >= int(np.ceil(0.6 * n_total))  # documented "clear majority" bar

    print(f"median D(morph,oracle)={med_morph:.4f} vs median D(truth,oracle)={med_truth:.4f}")
    print(f"clear median improvement (morph < 0.7x truth distance): {clear_median_improvement}")
    print(f"N closer={n_closer}/{n_total}, clear majority (>=60%): {clear_majority}")
    print(f"morphology is a trivial truth-proxy: {proxy_flag}")

    passed = clear_median_improvement and clear_majority and not proxy_flag
    print()
    print("M6 RESULT:", "PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
