#!/usr/bin/env python3
"""
H1-F: local convergence audit.

For each of the 4 remaining old_final-essential events, take
controlled4's own winning FINAL vector (not its initial guess) and
run exactly ONE H1 TRF polish from that exact point -- same
production_candidate bounds, same bounded_flux_profile, all 6
nonlinear parameters free, same coordinate convention (physical).
Only the optimizer tolerances change, to the historical
"adaptive polish" philosophy (xtol=ftol=1e-12, gtol=1e-8, same
max_nfev/x_scale) -- not a new start, not a new strategy family.
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

FOUR_EVENTS = [87786, 79700, 50179, 86451]
STARTS = ["truth", "H0_NESTED_piE_0", "truth_half_piE", "truth_mirror_u0_piEN"]

ROOT4 = Path("~/Downloads/hidden_parallax/production_validation/gate2_controlled4_extreme100").expanduser()
ref = pd.read_csv(os.path.join(HERE, "results", "gate2_start_ablation_per_event.csv")).set_index("catalog_row")

REAL_OPTIMIZER_OPTIONS = copy.deepcopy(core.OPTIMIZER_OPTIONS)
POLISH_OPTIMIZER_OPTIONS = {
    "xtol": 1.0e-12,
    "ftol": 1.0e-12,
    "gtol": 1.0e-8,
    "max_nfev": 50000,
    "x_scale": "jac",
}

records = []
for row in FOUR_EVENTS:
    d4 = pd.read_csv(list(ROOT4.rglob(f"extreme100/{row}/all_refits.csv"))[0])
    d4 = d4[(d4["hypothesis"] == "H1") & (d4["status"] == "success")]

    best_label = None
    best_chi2 = np.inf
    best_row = None
    for label in STARTS:
        x = d4[d4["label"] == label].iloc[0]
        if x["chi2"] < best_chi2:
            best_chi2 = x["chi2"]
            best_label = label
            best_row = x

    meta = core.load_case({"sample": "extreme100", "catalog_row": int(row), "matched_tail_catalog_row": -1})

    polish_initial = {
        "t0": float(best_row["t0"]), "u0": float(best_row["u0"]),
        "tE": float(best_row["tE"]), "rho": float(best_row["rho"]),
        "piEN": float(best_row["piEN"]), "piEE": float(best_row["piEE"]),
    }

    core.OPTIMIZER_OPTIONS = POLISH_OPTIMIZER_OPTIONS
    t0c = time.time()
    r = core.run_one_fit(meta, "H1", polish_initial, "controlled4_winner_polish")
    dt = time.time() - t0c
    core.OPTIMIZER_OPTIONS = REAL_OPTIMIZER_OPTIONS

    chi2_c5 = float(ref.loc[row, "chi2_oracle5"])
    chi2_after = float(r["chi2"])
    chi2_before = float(best_chi2)
    delta_after = chi2_after - chi2_c5

    rec = {
        "catalog_row": int(row),
        "controlled4_winner_label": best_label,
        "chi2_before_polish": chi2_before,
        "chi2_after_polish": chi2_after,
        "improvement": chi2_before - chi2_after,
        "monotonic_ok": bool(chi2_after <= chi2_before + 1e-6),
        "chi2_controlled5": chi2_c5,
        "delta_after_vs_controlled5": delta_after,
        "pass": bool(delta_after <= 0.1),
        "nfev": r.get("optimizer_nfev"),
        "njev": r.get("optimizer_njev"),
        "optimizer_optimality": r.get("optimizer_optimality"),
        "optimizer_active_mask": r.get("optimizer_active_mask"),
        "t0_final": r.get("t0"), "u0_final": r.get("u0"),
        "tE_final": r.get("tE"), "rho_final": r.get("rho"),
        "piEN_final": r.get("piEN"), "piEE_final": r.get("piEE"),
        "wall_time_s": dt,
        "init_t0": polish_initial["t0"], "init_u0": polish_initial["u0"],
        "init_tE": polish_initial["tE"], "init_rho": polish_initial["rho"],
        "init_piEN": polish_initial["piEN"], "init_piEE": polish_initial["piEE"],
    }
    records.append(rec)
    print(f"row={row} winner={best_label} chi2_before={chi2_before:.4f} "
          f"chi2_after={chi2_after:.4f} improvement={rec['improvement']:.6f} "
          f"monotonic_ok={rec['monotonic_ok']} delta_vs_c5={delta_after:.4f} "
          f"pass={rec['pass']} nfev={r.get('optimizer_nfev')} "
          f"active_mask={r.get('optimizer_active_mask')}", flush=True)

df = pd.DataFrame(records)
out_csv = os.path.join(HERE, "results", "h1_controlled4_winner_polish_4events.csv")
df.to_csv(out_csv, index=False)
print("DONE ->", out_csv)
