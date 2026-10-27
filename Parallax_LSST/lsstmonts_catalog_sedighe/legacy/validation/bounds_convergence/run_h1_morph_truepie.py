#!/usr/bin/env python3
"""
H1-C: MORPH_TRUE_PIE candidate -- run only on events H1-A did not
rescue. Initial vector: (t0,u0,tE,rho) = the frozen rank-1 morphology
candidate (H0 morphology extraction/lookup, unchanged, no retuning),
with sign(u0) set to sign(truth u0) -- H0's morphology convention is
always u0>0 (H0's exact u0-sign degeneracy), but that degeneracy does
not hold under H1 (piE breaks it), so the mirror ambiguity must be
resolved somehow; using the true sign is a documented, deliberate
initialization choice, not a retuned parameter. piEN,piEE = truth.
Initialization only -- all 6 H1 nonlinear parameters free. Same
bounds/objective/optimizer/mode as H1-A.
"""
import sys
import os
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

import pandas as pd  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"
assert core.FIT_SCOPE == "h1"

REMAINING_4 = [87786, 79700, 50179, 86451]  # not rescued by H1-A (85380 excluded)

cand = pd.read_csv(os.path.join(HERE, "results", "morphology_M3_M4_candidates.csv"))
c1 = cand[(cand["invertible"] == True) & (cand["candidate_rank"] == 1)
          & (cand["catalog_row"].isin(REMAINING_4))].set_index("catalog_row")
truth = pd.read_csv(os.path.join(HERE, "results", "b1c_event_features.csv")).set_index("catalog_row")

records = []
for row in REMAINING_4:
    meta = core.load_case({"sample": "extreme100", "catalog_row": int(row), "matched_tail_catalog_row": -1})

    u0_abs = abs(float(c1.loc[row, "u0_morph"]))
    u0_signed = u0_abs if truth.loc[row, "u0_true"] >= 0 else -u0_abs

    initial = {
        "t0": float(c1.loc[row, "t0_morph"]),
        "u0": u0_signed,
        "tE": float(c1.loc[row, "tE_morph"]),
        "rho": float(c1.loc[row, "rho_morph"]),
        "piEN": float(truth.loc[row, "piEN_true"]),
        "piEE": float(truth.loc[row, "piEE_true"]),
    }

    t0c = time.time()
    r = core.run_one_fit(meta, "H1", initial, "MORPH_TRUE_PIE")
    dt = time.time() - t0c

    r["catalog_row"] = int(row)
    r["wall_time_s"] = dt
    for k, v in initial.items():
        r[f"init_{k}"] = v
    records.append(r)
    print(f"row={row} chi2={r.get('chi2')} status={r.get('status')} "
          f"final=(t0={r.get('t0')},u0={r.get('u0')},tE={r.get('tE')},rho={r.get('rho')},"
          f"piEN={r.get('piEN')},piEE={r.get('piEE')}) "
          f"active_mask={r.get('optimizer_active_mask')} dt={dt:.2f}s", flush=True)

df = pd.DataFrame(records)
out_csv = os.path.join(HERE, "results", "h1_MORPH_TRUE_PIE_4events.csv")
df.to_csv(out_csv, index=False)
print("DONE ->", out_csv)
