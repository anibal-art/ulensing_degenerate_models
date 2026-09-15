import sys, os, copy, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True)
args = ap.parse_args()

sys.path.insert(0, "/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/validation/bounds_audit")
os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = args.mode
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = ["x", "--catalog-row", "71181", "--bounds-profile", "production_candidate", "--fit-scope", "h0", "--dry-run"]
import run_bounds_audit_refit_core as core
import pandas as pd

truth_feat = pd.read_csv(
    "/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/b1c_event_features.csv"
).set_index("catalog_row")
rows = sorted(truth_feat.index.tolist())[:6]

times = []
for row in rows:
    meta = core.load_case({"sample": "extreme100", "catalog_row": row, "matched_tail_catalog_row": -1})
    initial = {"t0": meta["truth"]["t0"], "u0": meta["truth"]["u0"], "tE": meta["truth"]["tE"], "rho": meta["truth"]["rho"]}
    t0 = time.time()
    r = core.run_one_fit(meta, "H0", initial, f"calib_{args.mode}")
    dt = time.time() - t0
    times.append(dt)
    print(args.mode, row, dt, r["status"])

import numpy as np
print(f"MODE={args.mode} mean={np.mean(times):.4f} median={np.median(times):.4f} n={len(times)}")
