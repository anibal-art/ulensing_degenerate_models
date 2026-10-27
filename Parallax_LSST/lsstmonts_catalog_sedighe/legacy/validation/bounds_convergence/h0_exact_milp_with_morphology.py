"""
Exact dependency-aware MILP, extended with the 4 morphology-informed
H0 strategies (rank-1 morphology candidate x each of the 4 existing
coordinate modes), evaluated under two coverage assumptions for the
88 events morphology has not yet been run on:

  Scenario P (pessimistic): morphology coverage = False for all 88
      untested events, in all 4 modes. Real observed coverage used
      for the 12 tested (critical) events.
  Scenario O (optimistic lower bound): morphology coverage = True for
      all 88 untested events, in all 4 modes. The 12 tested events
      ALWAYS keep their real observed coverage/failures -- never
      overridden.

No new fits. Morphology strategies never require the old_H0 bridge
(cost=1 each, unconditionally, no x_i<=bridge constraint).
"""
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

CRITICAL_12 = [62786, 83189, 567800, 579320, 82728, 574423,
               557860, 558189, 567365, 35927, 563210, 79501]
MORPH_MODES = ["physical", "log_te", "log_rho", "log_te_rho"]
MORPH_COLS = [f"morphology/{m}" for m in MORPH_MODES]

# ------------------------------------------------------------------
# existing 52-strategy delta matrix (unchanged from h0_exact_milp.py)
# ------------------------------------------------------------------
df = pd.read_csv("results/gate1_oracle_diagnostics.csv")


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


df["canon"] = [canon(m, s) for m, s in zip(df["mode"], df["strategy_id"])]
d52 = df[df["source"] == "gate1_oracle_52"]
piv52 = d52.pivot_table(index="catalog_row", columns="canon", values="chi2", aggfunc="min")
oracle52_min = piv52.min(axis=1)
DELTA52 = piv52.sub(oracle52_min, axis=0)
cols52 = list(piv52.columns)
old_idx_52 = [i for i, c in enumerate(cols52) if "/old_H0/" in c]

rows_100 = sorted(piv52.index.tolist())
n_events = len(rows_100)
assert n_events == 100

# ------------------------------------------------------------------
# real observed morphology coverage (12 tested events only)
# ------------------------------------------------------------------
real_cov = pd.read_csv("results/morphology_M7_coverage_by_mode.csv")
real_cov_pivot = real_cov.pivot(index="row", columns="mode", values="covers")[MORPH_MODES]


def build_full_matrix(scenario):
    """scenario: 'P' or 'O'. Returns (cols, A_cov) with 52+4 columns."""
    cov_morph = np.zeros((n_events, 4), dtype=bool)
    for i, row in enumerate(rows_100):
        if row in CRITICAL_12:
            cov_morph[i, :] = real_cov_pivot.loc[row].values
        else:
            cov_morph[i, :] = (scenario == "O")

    A52 = (DELTA52[cols52].values <= 0.1)
    A_full = np.concatenate([A52, cov_morph], axis=1)
    cols_full = cols52 + MORPH_COLS
    return cols_full, A_full


def solve_milp(cols_full, A_full, allow_old_h0=True, label=""):
    n_strat = len(cols_full)
    old_idx = [i for i, c in enumerate(cols_full) if "/old_H0/" in c]
    n_vars = n_strat + 1
    BRIDGE_IDX = n_strat

    c = np.concatenate([np.ones(n_strat), [1.0]])
    integrality = np.ones(n_vars, dtype=int)

    if allow_old_h0:
        ub = np.ones(n_vars)
    else:
        ub = np.ones(n_vars)
        for i in old_idx:
            ub[i] = 0.0  # force old_H0-anchored strategies unusable
        ub[BRIDGE_IDX] = 0.0  # bridge cannot be paid for nothing
    var_bounds = Bounds(np.zeros(n_vars), ub)

    rows_dep = []
    for i in old_idx:
        row = np.zeros(n_vars)
        row[i] = 1.0
        row[BRIDGE_IDX] = -1.0
        rows_dep.append(row)
    con_dep = LinearConstraint(np.array(rows_dep), lb=-np.inf, ub=np.zeros(len(rows_dep)))

    A_cov_full = np.zeros((n_events, n_vars))
    A_cov_full[:, :n_strat] = A_full
    con_cov = LinearConstraint(A_cov_full, lb=np.ones(n_events), ub=np.full(n_events, np.inf))

    result = milp(c=c, integrality=integrality, bounds=var_bounds,
                  constraints=[con_dep, con_cov], options={"time_limit": 300})

    print("=" * 100)
    print(label)
    print("=" * 100)
    print("status:", result.status, "success:", result.success, "message:", result.message)

    if not result.success:
        print("INFEASIBLE / NOT SOLVED")
        return None

    x = result.x
    sel = [cols_full[i] for i in range(n_strat) if x[i] > 0.5]
    bridge_on = x[BRIDGE_IDX] > 0.5
    n_old_sel = sum(1 for s in sel if "/old_H0/" in s)
    n_morph_sel = sum(1 for s in sel if s.startswith("morphology/"))
    morph_modes_sel = [s.split("/")[1] for s in sel if s.startswith("morphology/")]

    print("minimum total cost:", int(round(result.fun)))
    print("bridge selected:", bridge_on)
    print("n strategies selected:", len(sel))
    print("n old_H0-anchored selected:", n_old_sel)
    print("n morphology selected:", n_morph_sel, "modes:", morph_modes_sel)
    print("selected:", sel)

    gate18_dropped = None
    return {
        "cost": int(round(result.fun)),
        "bridge": bridge_on,
        "selected": sel,
        "n_old_H0": n_old_sel,
        "n_morph": n_morph_sel,
        "morph_modes": morph_modes_sel,
        "status": result.status,
    }


# ------------------------------------------------------------------
# Baseline (no morphology) -- reconfirm existing result for reference
# ------------------------------------------------------------------
res_base = solve_milp(cols52, DELTA52[cols52].values <= 0.1, allow_old_h0=True,
                       label="BASELINE (no morphology) -- reference, should reproduce minimum=19")

# ------------------------------------------------------------------
# Scenario P (pessimistic)
# ------------------------------------------------------------------
cols_P, A_P = build_full_matrix("P")
res_P = solve_milp(cols_P, A_P, allow_old_h0=True,
                    label="SCENARIO P (pessimistic: 88 untested events, morphology coverage=False)")

# ------------------------------------------------------------------
# Scenario O (optimistic lower bound)
# ------------------------------------------------------------------
cols_O, A_O = build_full_matrix("O")
res_O = solve_milp(cols_O, A_O, allow_old_h0=True,
                    label="SCENARIO O (optimistic lower bound: 88 untested events, morphology coverage=True)")

# ------------------------------------------------------------------
# Force-no-bridge variants
# ------------------------------------------------------------------
res_P_nobridge = solve_milp(cols_P, A_P, allow_old_h0=False,
                             label="SCENARIO P, FORCE NO old_H0 BRIDGE (truth + morphology only)")
res_O_nobridge = solve_milp(cols_O, A_O, allow_old_h0=False,
                             label="SCENARIO O, FORCE NO old_H0 BRIDGE (truth + morphology only)")

# ------------------------------------------------------------------
# Coverage redundancy detail for the 12 tested events under Scenario P's optimum
# ------------------------------------------------------------------
print("=" * 100)
print("Redundancy check: which of the 12 critical events does morphology cover")
print("that are ALSO covered by a selected non-morphology strategy in the P optimum?")
print("=" * 100)
if res_P is not None:
    sel_idx = [cols_P.index(s) for s in res_P["selected"]]
    for i, row in enumerate(rows_100):
        if row not in CRITICAL_12:
            continue
        covering = [cols_P[j] for j in sel_idx if A_P[i, j]]
        print(row, "covered by:", covering)

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print("baseline minimum (no morphology):", res_base["cost"] if res_base else None)
print("Scenario P minimum:", res_P["cost"] if res_P else None)
print("Scenario O minimum:", res_O["cost"] if res_O else None)
print("Scenario P, no-bridge feasible:", res_P_nobridge is not None,
      "cost=", res_P_nobridge["cost"] if res_P_nobridge else "INFEASIBLE")
print("Scenario O, no-bridge feasible:", res_O_nobridge is not None,
      "cost=", res_O_nobridge["cost"] if res_O_nobridge else "INFEASIBLE")

if res_O and res_O["cost"] >= 19:
    decision = "A: optimistic minimum >= 19 -> morphology cannot improve cost even in the best case. Close."
elif res_P and res_P["cost"] < 19:
    decision = "B: pessimistic minimum < 19 -> already-observed rescues reduce cost. Extend to full extreme100."
elif res_P and res_P["cost"] == 19 and res_O and res_O["cost"] < 19:
    decision = "C: pessimistic==19 but optimistic<19 -> depends on untested 88 coverage. Justifies extension."
else:
    decision = "UNRESOLVED -- check res_P/res_O values manually."

print()
print("DECISION:", decision)
