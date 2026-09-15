import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

df = pd.read_csv('/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/validation/bounds_convergence/results/gate1_oracle_diagnostics.csv')

def canon(mode, sid):
    if '/' in sid:
        anchor, tag = sid.split('/', 1)
    else:
        if sid.startswith('old_H0_rho_'):
            anchor = 'old_H0'; tag = sid[len('old_H0_rho_'):]
        else:
            anchor = 'truth'; tag = sid[len('truth_rho_'):]
    if tag == 'truth': tag = 'truth_rho'
    if tag not in ('truth_rho', 'rho_from_old_H0'): tag = f'{float(tag):.10g}'
    return f'{mode}/{anchor}/{tag}'

df['canon'] = [canon(m, s) for m, s in zip(df['mode'], df['strategy_id'])]
d52 = df[df['source'] == 'gate1_oracle_52']
piv52 = d52.pivot_table(index='catalog_row', columns='canon', values='chi2', aggfunc='min')
oracle52_min = piv52.min(axis=1)
DELTA = piv52.sub(oracle52_min, axis=0)

cols = list(piv52.columns)
old_idx = [i for i, c in enumerate(cols) if '/old_H0/' in c]
n_strat = len(cols)
n_events = len(piv52)

A_cov_strategies = (DELTA[cols].values <= 0.1).astype(float)  # (100, 52)

# --------------------------------------------------------------
# Part 2a: EXACT baseline MILP -- minimize strategy COUNT only
# (reproduces the original gate1_final18 derivation, no bridge
# concept), to confirm 18 is a proven exact minimum.
# --------------------------------------------------------------
objective = np.ones(n_strat)
constraints = LinearConstraint(A_cov_strategies, lb=np.ones(n_events), ub=np.full(n_events, np.inf))
bounds = Bounds(np.zeros(n_strat), np.ones(n_strat))
result = milp(c=objective, integrality=np.ones(n_strat, dtype=int), bounds=bounds,
              constraints=constraints, options={"time_limit": 300})

print("=" * 100)
print("BASELINE EXACT MILP: minimize strategy count (no bridge concept), 52-pool")
print("=" * 100)
print("status:", result.status, "(0=optimal)")
print("message:", result.message)
print("success:", result.success)
print("minimum N strategies:", int(round(result.fun)))
sel = [cols[i] for i in np.where(result.x > 0.5)[0]]
n_old_sel = sum(1 for c in sel if '/old_H0/' in c)
print("selected:", sel)
print("n_old_H0 in minimum set:", n_old_sel)
print()

# --------------------------------------------------------------
# Part 2b: is old_H0 mandatory? Solve the SAME MILP restricted to
# truth-only columns (should be infeasible -- we already proved
# this via exhaustive earlier, cross-check here).
# --------------------------------------------------------------
truth_idx = [i for i, c in enumerate(cols) if '/truth/' in c]
A_truth = A_cov_strategies[:, truth_idx]
objective_t = np.ones(len(truth_idx))
constraints_t = LinearConstraint(A_truth, lb=np.ones(n_events), ub=np.full(n_events, np.inf))
bounds_t = Bounds(np.zeros(len(truth_idx)), np.ones(len(truth_idx)))
result_t = milp(c=objective_t, integrality=np.ones(len(truth_idx), dtype=int), bounds=bounds_t,
                constraints=constraints_t, options={"time_limit": 60})
print("=" * 100)
print("Truth-only feasibility check (should be INFEASIBLE)")
print("=" * 100)
print("status:", result_t.status, "success:", result_t.success, "message:", result_t.message)
print()

# --------------------------------------------------------------
# Part 1 + Part 2c: dependency-aware EXACT MILP with bridge, and
# the reevaluated old_H0 vector as a free coverage option once
# bridge=1.
# --------------------------------------------------------------
raw = pd.read_csv('results/old_h0_vector_current_objective.csv').set_index('catalog_row')
raw = raw.loc[piv52.index]
bridge_delta = raw['chi2_old_H0_vector_current_objective'] - oracle52_min
bridge_covers = (bridge_delta <= 0.1).astype(float).values  # (100,)
n_bridge_covered = int(bridge_covers.sum())
print("=" * 100)
print("Bridge-vector (old_H0[:4], current objective) coverage")
print("=" * 100)
print("N events rescued by the bridge vector alone (delta<=0.1):", n_bridge_covered, "/", n_events)
print("bridge_delta describe:")
print(bridge_delta.describe())
print()

# Variables: x_1..x_52 (strategies), bridge (1 var) => 53 total
n_vars = n_strat + 1
BRIDGE_IDX = n_strat

c = np.concatenate([np.ones(n_strat), [1.0]])  # cost: each strategy 1, bridge 1
integrality = np.ones(n_vars, dtype=int)
var_bounds = Bounds(np.zeros(n_vars), np.ones(n_vars))

# Constraint 1: x_i <= bridge for every old_H0 strategy i
rows_dep = []
for i in old_idx:
    row = np.zeros(n_vars)
    row[i] = 1.0
    row[BRIDGE_IDX] = -1.0
    rows_dep.append(row)
A_dep = np.array(rows_dep)
con_dep = LinearConstraint(A_dep, lb=-np.inf, ub=np.zeros(len(rows_dep)))

# Constraint 2: coverage -- sum_s A[e,s]*x_s + bridge_covers[e]*bridge >= 1
A_cov_full = np.zeros((n_events, n_vars))
A_cov_full[:, :n_strat] = A_cov_strategies
A_cov_full[:, BRIDGE_IDX] = bridge_covers
con_cov = LinearConstraint(A_cov_full, lb=np.ones(n_events), ub=np.full(n_events, np.inf))

result_b = milp(c=c, integrality=integrality, bounds=var_bounds,
                constraints=[con_dep, con_cov], options={"time_limit": 300})

print("=" * 100)
print("EXACT dependency-aware MILP WITH bridge-vector free coverage")
print("=" * 100)
print("status:", result_b.status, "(0=optimal)")
print("message:", result_b.message)
print("success:", result_b.success)
print("minimum TOTAL cost:", int(round(result_b.fun)))
x = result_b.x
sel_s = [cols[i] for i in range(n_strat) if x[i] > 0.5]
bridge_on = x[BRIDGE_IDX] > 0.5
n_old_sel_b = sum(1 for c_ in sel_s if '/old_H0/' in c_)
print("bridge selected:", bridge_on)
print("n downstream strategies selected:", len(sel_s))
print("n old_H0 among them:", n_old_sel_b)
print("selected strategies:", sel_s)
print("n events covered by bridge-vector-alone among selected solution:",
      int(bridge_covers[np.where(bridge_covers > 0)[0]].sum()) if bridge_on else 0)
