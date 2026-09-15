#!/usr/bin/env python3
"""
Block-ablation experiment, ONLY for the 2 H1-F2-confirmed genuinely
locally-stationary distinct-basin events (87786, 86451; 50179 excluded
-- not confirmed stationary). Baseline seed = old_final RAW vector
(proven to access the target basin via old_final_reseed). Three
leave-one-block-out hybrids per event, each replacing exactly one
block with the controlled4-polished value, everything else from raw
old_final. Initialization only; all 6 H1 parameters free; standard
(unbudgeted, default) optimizer options -- same convention as every
other H1 "start" test in this log (controlled4/5, H1-A/C), NOT the
tightened polish tolerances (this is a normal fit, not a polish).
"""
import sys
import os
import copy
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bounds_audit"))
sys.path.insert(0, HERE)

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row", "71181",
    "--bounds-profile", "production_candidate",
    "--fit-scope", "h1",
    "--dry-run",
]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pathlib import Path  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"
assert core.FIT_SCOPE == "h1"

STATIONARY_SURVIVORS = [87786, 86451]

f2 = pd.read_csv(os.path.join(HERE, "results", "h1_f2_convergence_audit.csv")).set_index("catalog_row")
ROOTOLD = Path("~/Downloads/hidden_parallax/production_validation/gate2_oldfinal_extreme100").expanduser()

records = []
for row in STATIONARY_SURVIVORS:
    meta = core.load_case({"sample": "extreme100", "catalog_row": int(row), "matched_tail_catalog_row": -1})
    old_raw = np.asarray(meta["old_h1"], dtype=float)[:6]
    old_raw_dict = {"t0": old_raw[0], "u0": old_raw[1], "tE": old_raw[2],
                     "rho": old_raw[3], "piEN": old_raw[4], "piEE": old_raw[5]}

    c4p = f2.loc[row]
    c4p_dict = {
        "t0": float(c4p["final_t0"]), "u0": float(c4p["final_u0"]), "tE": float(c4p["final_tE"]),
        "rho": float(c4p["final_rho"]), "piEN": float(c4p["final_piEN"]), "piEE": float(c4p["final_piEE"]),
    }

    dold = pd.read_csv(list(ROOTOLD.rglob(f"extreme100/{row}/all_refits.csv"))[0])
    dold = dold[(dold["hypothesis"] == "H1") & (dold["status"] == "success")]
    chi2_old_final_reseed = float(dold[dold["label"] == "old_final_reseed"].iloc[0]["chi2"])

    hybrids = {
        "OLD_except_BACKBONE": {**old_raw_dict, "t0": c4p_dict["t0"], "u0": c4p_dict["u0"], "tE": c4p_dict["tE"]},
        "OLD_except_RHO": {**old_raw_dict, "rho": c4p_dict["rho"]},
        "OLD_except_PIE": {**old_raw_dict, "piEN": c4p_dict["piEN"], "piEE": c4p_dict["piEE"]},
    }

    for hyb_name, initial in hybrids.items():
        t0c = time.time()
        r = core.run_one_fit(meta, "H1", initial, hyb_name)
        dt = time.time() - t0c

        chi2 = float(r["chi2"])
        delta = chi2 - chi2_old_final_reseed
        rec = {
            "catalog_row": int(row),
            "hybrid": hyb_name,
            "chi2": chi2,
            "chi2_old_final_reseed": chi2_old_final_reseed,
            "delta_vs_old_final_reseed": delta,
            "pass": bool(delta <= 0.1),
            "status": r.get("status"),
            "optimizer_active_mask": r.get("optimizer_active_mask"),
            "t0_final": r.get("t0"), "u0_final": r.get("u0"),
            "tE_final": r.get("tE"), "rho_final": r.get("rho"),
            "piEN_final": r.get("piEN"), "piEE_final": r.get("piEE"),
            "wall_time_s": dt,
            **{f"init_{k}": v for k, v in initial.items()},
        }
        records.append(rec)
        print(f"row={row} hybrid={hyb_name:22s} chi2={chi2:.4f} delta_vs_old_final_reseed={delta:.4f} "
              f"pass={rec['pass']} active_mask={r.get('optimizer_active_mask')} dt={dt:.2f}s", flush=True)

df = pd.DataFrame(records)
out_csv = os.path.join(HERE, "results", "h1_block_ablation.csv")
df.to_csv(out_csv, index=False)
print("DONE ->", out_csv, "n_fits=", len(df))
