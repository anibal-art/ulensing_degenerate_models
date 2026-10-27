"""
Final exact dependency-aware MILP: 52 existing oracle strategies + 4
morphology strategies, using the REAL, fully-observed 100-event
morphology coverage (no assumptions -- Scenario P/O are obsolete now
that all 100 events have been run). Also includes the re-evaluated
old_H0 bridge vector as free coverage once bridge=1 (as in the
previous exact MILP), and a force-bridge=0 variant to test whether
morphology can eliminate the legacy dependency outright.
"""
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

MORPH_MODES = ["physical", "log_te", "log_rho", "log_te_rho"]
MORPH_COLS = [f"morphology/{m}" for m in MORPH_MODES]

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

rows_100 = sorted(piv52.index.tolist())
n_events = len(rows_100)

morph_delta = pd.read_csv("results/morphology100_delta.csv")
piv_morph_cov = morph_delta.pivot(index="catalog_row", columns="mode", values="covers")[MORPH_MODES]
piv_morph_cov = piv_morph_cov.loc[rows_100]

A52 = (DELTA52[cols52].values <= 0.1)
A_morph = piv_morph_cov[MORPH_MODES].values.astype(bool)
A_full = np.concatenate([A52, A_morph], axis=1)
cols_full = cols52 + MORPH_COLS

# re-evaluated bridge vector, free coverage once bridge=1
old_h0_vec = pd.read_csv("results/old_h0_vector_current_objective.csv").set_index("catalog_row")
bridge_delta = old_h0_vec.loc[rows_100, "chi2_old_H0_vector_current_objective"] - oracle52_min.loc[rows_100]
bridge_covers = (bridge_delta.values <= 0.1)


def solve(cols, A, allow_old_h0=True, use_bridge_free_coverage=True, label=""):
    n_strat = len(cols)
    old_idx = [i for i, c in enumerate(cols) if "/old_H0/" in c]
    n_vars = n_strat + 1
    BRIDGE_IDX = n_strat

    c = np.concatenate([np.ones(n_strat), [1.0]])
    integrality = np.ones(n_vars, dtype=int)

    ub = np.ones(n_vars)
    if not allow_old_h0:
        for i in old_idx:
            ub[i] = 0.0
        ub[BRIDGE_IDX] = 0.0
    var_bounds = Bounds(np.zeros(n_vars), ub)

    rows_dep = []
    for i in old_idx:
        row = np.zeros(n_vars)
        row[i] = 1.0
        row[BRIDGE_IDX] = -1.0
        rows_dep.append(row)
    con_dep = LinearConstraint(np.array(rows_dep), lb=-np.inf, ub=np.zeros(len(rows_dep)))

    A_cov_full = np.zeros((n_events, n_vars))
    A_cov_full[:, :n_strat] = A
    if use_bridge_free_coverage:
        A_cov_full[:, BRIDGE_IDX] = bridge_covers
    con_cov = LinearConstraint(A_cov_full, lb=np.ones(n_events), ub=np.full(n_events, np.inf))

    result = milp(c=c, integrality=integrality, bounds=var_bounds,
                  constraints=[con_dep, con_cov], options={"time_limit": 300})

    print("=" * 100)
    print(label)
    print("=" * 100)
    print("status:", result.status, "success:", result.success, "message:", result.message)

    if not result.success:
        print("INFEASIBLE")
        return None

    x = result.x
    sel = [cols[i] for i in range(n_strat) if x[i] > 0.5]
    bridge_on = x[BRIDGE_IDX] > 0.5
    n_old = sum(1 for s in sel if "/old_H0/" in s)
    n_truth = sum(1 for s in sel if "/truth/" in s)
    n_morph = sum(1 for s in sel if s.startswith("morphology/"))
    morph_modes_sel = [s.split("/")[1] for s in sel if s.startswith("morphology/")]

    print("minimum total cost:", int(round(result.fun)))
    print("bridge selected:", bridge_on)
    print(f"n selected={len(sel)}  n_truth={n_truth}  n_old_H0={n_old}  n_morphology={n_morph} modes={morph_modes_sel}")
    print("selected:", sel)

    return {
        "cost": int(round(result.fun)), "bridge": bridge_on, "selected": sel,
        "n_truth": n_truth, "n_old_H0": n_old, "n_morph": n_morph,
        "morph_modes": morph_modes_sel, "status": result.status,
    }


res_final = solve(cols_full, A_full, allow_old_h0=True, use_bridge_free_coverage=True,
                   label="FINAL EXACT MILP: 52 + 4 morphology, real 100-event coverage, bridge-vector free coverage")

res_nobridge = solve(cols_full, A_full, allow_old_h0=False, use_bridge_free_coverage=False,
                      label="FORCE bridge=0 (truth + morphology only, real coverage) -- can morphology eliminate the legacy dependency entirely?")

# which gate1_final18 strategies disappear relative to the original 18
GATE1_FINAL18 = {
    "physical": [("old_H0", "1"), ("truth", "0.1"), ("truth", "1")],
    "log_te": [("old_H0", "truth_rho"), ("old_H0", "0.01"), ("old_H0", "0.1"),
               ("old_H0", "1"), ("truth", "0.01"), ("truth", "1")],
    "log_rho": [("old_H0", "truth_rho"), ("old_H0", "0.01"), ("old_H0", "1"),
                ("truth", "truth_rho"), ("truth", "1e-06"), ("truth", "0.0001"),
                ("truth", "0.1"), ("truth", "1")],
    "log_te_rho": [("truth", "1e-06")],
}
gate18 = set(f"{m}/{a}/{r}" for m, lst in GATE1_FINAL18.items() for a, r in lst)

print()
print("=" * 100)
print("COMPARISON vs baseline gate1_final18 (18 strategies, exact MILP minimum-count solution)")
print("=" * 100)
if res_final:
    sel_non_morph = set(s for s in res_final["selected"] if not s.startswith("morphology/"))
    print("gate1_final18 strategies NOT in final selection:", sorted(gate18 - sel_non_morph))
    print("selected strategies NOT in gate1_final18 (non-morphology):", sorted(sel_non_morph - gate18))

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print("Baseline (no morphology) exact minimum: 19 (reconfirmed in prior turn)")
print("Partial-morphology pessimistic bound (prior turn): 18")
print("FULL morphology100 exact minimum:", res_final["cost"] if res_final else None)
print("Force-no-bridge feasible with real 100-event morphology coverage:",
      res_nobridge is not None, "cost=" , res_nobridge["cost"] if res_nobridge else "INFEASIBLE")
