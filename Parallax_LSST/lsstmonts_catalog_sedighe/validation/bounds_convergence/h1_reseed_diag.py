from pathlib import Path
import pandas as pd
import numpy as np

ROOT4 = Path("~/Downloads/hidden_parallax/production_validation/gate2_controlled4_extreme100").expanduser()
ROOTOLD = Path("~/Downloads/hidden_parallax/production_validation/gate2_oldfinal_extreme100").expanduser()

H0_FILE = Path("/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/validation/bounds_convergence/data/gate2_h0_anchor_manifest_extreme100.csv")
h0 = pd.read_csv(H0_FILE).set_index('catalog_row')

truth = pd.read_csv('/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/validation/bounds_convergence/results/b1c_event_features.csv').set_index('catalog_row')

FAIL_EVENTS = [85380, 87786, 79700, 50179, 86451]

def read_unique(root, row):
    matches = list(root.rglob(f"extreme100/{row}/all_refits.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"row={row} root={root}: found {len(matches)}")
    return pd.read_csv(matches[0])

for row in FAIL_EVENTS:
    d4 = read_unique(ROOT4, row)
    dold = read_unique(ROOTOLD, row)
    d4 = d4[(d4['hypothesis']=='H1') & (d4['status']=='success')]
    dold = dold[(dold['hypothesis']=='H1') & (dold['status']=='success')]

    print("="*100)
    print("row", row)
    h0row = h0.loc[row]
    print("current H0 final: t0={:.6f} u0={:.6f} tE={:.6f} rho={:.6f} chi2_h0={:.6f}".format(
        h0row['t0'], h0row['u0'], h0row['tE'], h0row['rho'], h0row['chi2_h0']))

    tr = truth.loc[row]
    print("truth: t0={:.6f} u0={:.6f} tE={:.6f} rho={:.6f} piEN={:.6f} piEE={:.6f}".format(
        tr['t0_true'], tr['u0_true'], tr['tE_true'], tr['rho_true'], tr['piEN_true'], tr['piEE_true']))

    for label in ['H0_NESTED_piE_0','truth','truth_half_piE','truth_mirror_u0_piEN','old_final_reseed']:
        if label == 'old_final_reseed':
            x = dold[dold['label']==label]
        else:
            x = d4[d4['label']==label]
        if len(x) != 1:
            print(label, "NOT FOUND", len(x))
            continue
        r = x.iloc[0]
        print(f"{label:22s} chi2={r['chi2']:12.4f} t0={r['t0']:.6f} u0={r['u0']:+.6f} tE={r['tE']:10.4f} rho={r['rho']:.6f} piEN={r['piEN']:+.6f} piEE={r['piEE']:+.6f}")
