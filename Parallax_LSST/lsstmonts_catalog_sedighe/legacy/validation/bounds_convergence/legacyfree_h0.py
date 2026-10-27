import pandas as pd
import numpy as np
from itertools import combinations

df = pd.read_csv('/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/validation/bounds_convergence/results/gate1_oracle_diagnostics.csv')

def canon(mode, sid):
    # normalize both slash and underscore formats to (mode, anchor, rho_key)
    if '/' in sid:
        anchor, tag = sid.split('/', 1)
    else:
        if sid.startswith('old_H0_rho_'):
            anchor = 'old_H0'
            tag = sid[len('old_H0_rho_'):]
        elif sid.startswith('truth_rho_'):
            anchor = 'truth'
            tag = sid[len('truth_rho_'):]
        else:
            raise ValueError(sid)
    if tag == 'truth':
        tag = 'truth_rho'
    if tag not in ('truth_rho', 'rho_from_old_H0'):
        tag = f'{float(tag):.10g}'
    return f'{mode}/{anchor}/{tag}'

df['canon'] = [canon(m, s) for m, s in zip(df['mode'], df['strategy_id'])]

d52 = df[df['source'] == 'gate1_oracle_52'].copy()
d18 = df[df['source'] == 'gate1_final18_base_t0margin0'].copy()

print('=== sanity ===')
print('d52 unique canon:', d52['canon'].nunique(), '(expect 52)')
print('d18 unique canon:', d18['canon'].nunique(), '(expect 18)')

piv52 = d52.pivot_table(index='catalog_row', columns='canon', values='chi2', aggfunc='min')
piv18 = d18.pivot_table(index='catalog_row', columns='canon', values='chi2', aggfunc='min')

# verify equivalence of the 18 overlapping canonical strategies between the two sources
common = [c for c in piv18.columns if c in piv52.columns]
print('overlap cols:', len(common), '/', piv18.shape[1])
diffs = (piv18[common] - piv52[common]).abs()
print('max abs diff d18 vs d52 on overlap:', diffs.values.max())

old_h0_cols_18 = [c for c in piv18.columns if '/old_H0/' in c]
truth_cols_18 = [c for c in piv18.columns if '/truth/' in c]
print()
print('gate1_final18 old_H0-anchored strategies (', len(old_h0_cols_18), '):')
for c in sorted(old_h0_cols_18): print(' ', c)
print('gate1_final18 truth-anchored (legacy-free) strategies (', len(truth_cols_18), '):')
for c in sorted(truth_cols_18): print(' ', c)

best18 = piv18.min(axis=1)
best_legacyfree10 = piv18[truth_cols_18].min(axis=1)

delta = best_legacyfree10 - best18
print()
print('=== legacyfree10 (subset of gate1_final18) vs gate1_final18 (18) ===')
print('N events:', len(delta))
print('N(delta>0.1):', int((delta>0.1).sum()))
print('N(delta>1):', int((delta>1.0).sum()))
print('N(delta>10):', int((delta>10.0).sum()))
print('worst:', delta.max(), 'at row', delta.idxmax())
print('mean:', delta.mean(), 'median:', delta.median())

# win counts among the 18
winner18 = piv18.idxmin(axis=1)
print()
print('=== win counts among gate1_final18 (18 strategies) ===')
print(winner18.value_counts().to_string())

# also vs full oracle52 reference
oracle52_min = piv52.min(axis=1)
delta_vs_oracle_legacyfree10 = best_legacyfree10 - oracle52_min
delta_vs_oracle_18 = best18 - oracle52_min
print()
print('=== vs full oracle52 min (all 52, incl old_H0) ===')
print('gate1_final18 (18) delta>0.1 count:', int((delta_vs_oracle_18>0.1).sum()), 'worst', delta_vs_oracle_18.max())
print('legacyfree10 delta>0.1 count:', int((delta_vs_oracle_legacyfree10>0.1).sum()), 'worst', delta_vs_oracle_legacyfree10.max())

delta.to_frame('delta_legacyfree10_vs_gate1final18').to_csv('results/legacyfree10_delta.csv')

# ---- full 24 truth-anchored search within oracle52 ----
truth_cols_52 = [c for c in piv52.columns if '/truth/' in c]
old_cols_52 = [c for c in piv52.columns if '/old_H0/' in c]
print()
print('n truth-anchored in oracle52 (candidate universe):', len(truth_cols_52))
assert len(truth_cols_52) == 24
assert len(old_cols_52) == 28

M = piv52[truth_cols_52].values  # (100, 24)
ref = oracle52_min.values        # (100,) best over ALL 52 (incl old_H0)
cols = truth_cols_52

import math
found = None
for k in range(1, 13):
    n_combo = math.comb(len(cols), k)
    best_for_k = None
    for combo in combinations(range(len(cols)), k):
        sub = M[:, combo].min(axis=1)
        d = sub - ref
        nfail = int((d > 0.1).sum())
        if nfail == 0:
            worst = float(d.max())
            key = (worst,)
            if best_for_k is None or key < best_for_k[0]:
                best_for_k = (key, combo, worst)
    print(f'k={k}: n_combos_checked={n_combo}, zero-failure sets found: {"YES" if best_for_k else "no"}' + (f', best worst-delta={best_for_k[2]:.4f}' if best_for_k else ''))
    if best_for_k is not None:
        found = (k, best_for_k[1], best_for_k[2])
        break

if found:
    k, combo, worst = found
    print()
    print(f'=== MINIMAL legacy-free-52-subset with zero failures>0.1 vs oracle52: k={k} ===')
    for i in combo:
        print(' ', cols[i])
    print('worst delta:', worst)
else:
    print('No zero-failure legacy-free subset found up to k=12')
