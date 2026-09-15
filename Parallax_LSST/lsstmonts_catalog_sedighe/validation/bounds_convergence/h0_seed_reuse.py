import pandas as pd
import numpy as np

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
d52 = df[df['source'] == 'gate1_oracle_52'].copy()

truth = pd.read_csv('/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/validation/bounds_convergence/results/b1c_event_features.csv')
truth = truth.set_index('catalog_row')[['t0_true','u0_true','tE_true','rho_true']]

FAIL_EVENTS = [62786, 83189, 567800, 579320, 82728, 574423, 557860, 558189, 567365, 35927, 563210, 79501]

winner_of_18 = {
    62786: 'log_rho/old_H0/0.01', 83189: 'log_te/old_H0/0.01', 567800: 'physical/old_H0/1',
    579320: 'physical/old_H0/1', 82728: 'log_te/old_H0/truth_rho', 574423: 'physical/old_H0/1',
    557860: 'log_te/old_H0/truth_rho', 558189: 'log_rho/old_H0/1', 567365: 'log_te/old_H0/1',
    35927: 'log_te/old_H0/0.1', 563210: 'log_rho/old_H0/truth_rho', 79501: 'log_rho/old_H0/1',
}

K6_TRUTH = ['log_rho/truth/0.1', 'log_rho/truth/1', 'log_te/truth/1', 'physical/truth/0.1']
EXTRA = ['physical/truth/1', 'log_te/truth/0.01']
CANDIDATES = K6_TRUTH + EXTRA

piv = d52.set_index(['catalog_row', 'canon'])[['t0', 'u0', 'tE', 'rho']]

rows = []
for row in FAIL_EVENTS:
    win_key = winner_of_18[row]
    try:
        win_vec = piv.loc[(row, win_key)]
    except KeyError:
        continue
    t0_true, u0_true, tE_true, rho_true = truth.loc[row]

    for cand in CANDIDATES:
        try:
            v = piv.loc[(row, cand)]
        except KeyError:
            continue
        dt0_over_tEtrue = (v['t0'] - win_vec['t0']) / tE_true
        du0 = v['u0'] - win_vec['u0']
        log_tE_ratio = np.log10(v['tE'] / win_vec['tE'])
        log_rho_ratio = np.log10(v['rho'] / win_vec['rho'])
        rows.append({
            'catalog_row': row,
            'winner': win_key,
            'candidate': cand,
            'dt0_over_tEtrue': dt0_over_tEtrue,
            'du0': du0,
            'log10_tE_ratio': log_tE_ratio,
            'log10_rho_ratio': log_rho_ratio,
            'winner_tE': win_vec['tE'],
            'cand_tE': v['tE'],
            'winner_rho': win_vec['rho'],
            'cand_rho': v['rho'],
        })

out = pd.DataFrame(rows)
pd.set_option('display.width', 220)
pd.set_option('display.max_rows', 200)
print(out.to_string(index=False))

out.to_csv('results/h0_seed_reuse_compare.csv', index=False)
