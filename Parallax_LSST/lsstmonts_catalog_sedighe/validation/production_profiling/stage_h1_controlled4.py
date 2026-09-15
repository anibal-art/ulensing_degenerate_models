#!/usr/bin/env python3
"""
Robust-reference H1 stage WITHOUT old_final (avoids the 64%-crash
legacy grid entirely -- see FEASIBILITY_AUDIT.md): the 4
old_final-independent controlled5 starts (drops old_final_reseed),
production_candidate bounds. Used ONLY to build an independent,
uniformly-computable multistart reference across the full 25-event
H1-generated profiling sample for the final basin-gap validation
(TWO_FIT_ARCHITECTURE.md / NUMERICAL_SAFEGUARDS.md's n=9 reference was
old_final-based and only available for the 9 audit-PASS rows). Not
part of the frozen production candidate.
"""
import sys
import os
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
BA = os.path.join(HERE, "..", "bounds_audit")
sys.path.insert(0, BA)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--h5", required=True)
ap.add_argument("--h0-final-json", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = ["run_bounds_audit_refit_core.py", "--catalog-row", "71181",
            "--bounds-profile", "production_candidate", "--fit-scope", "h1", "--dry-run"]

import numpy as np  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402
from load_new_event import load_new_case  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"

meta = load_new_case(args.h5, core)
with open(args.h0_final_json) as f:
    h0_final = json.load(f)

truth = meta["truth"]
truth_initial = {"t0": truth["t0"], "u0": truth["u0"], "tE": truth["tE"], "rho": truth["rho"],
                  "piEN": truth["piEN"], "piEE": truth["piEE"]}
nested_initial = {"t0": h0_final["t0"], "u0": h0_final["u0"], "tE": h0_final["tE"], "rho": h0_final["rho"],
                   "piEN": 0.0, "piEE": 0.0}
half_pie_initial = {"t0": truth["t0"], "u0": truth["u0"], "tE": truth["tE"], "rho": truth["rho"],
                     "piEN": 0.5 * truth["piEN"], "piEE": 0.5 * truth["piEE"]}
mirror_initial = {"t0": truth["t0"], "u0": -truth["u0"], "tE": truth["tE"], "rho": truth["rho"],
                   "piEN": -truth["piEN"], "piEE": truth["piEE"]}

starts = [("truth", truth_initial), ("H0_NESTED_piE_0", nested_initial),
          ("truth_half_piE", half_pie_initial), ("truth_mirror_u0_piEN", mirror_initial)]

records = []
for label, initial in starts:
    t0c, cpu0 = time.time(), time.process_time()
    r = core.run_one_fit(meta, "H1", initial, label)
    records.append({**r, "wall_s": time.time() - t0c, "cpu_s": time.process_time() - cpu0})

finite = [r for r in records if np.isfinite(r.get("chi2", np.nan))]
best = min(finite, key=lambda r: r["chi2"]) if finite else records[0]

out = {"catalog_row": meta["row"], "controlled4": records,
       "h1_final_chi2": best["chi2"], "h1_final_label": best["label"]}
with open(args.out, "w") as f:
    json.dump(out, f, indent=2)

print(f"row={meta['row']} controlled4_wall={sum(r['wall_s'] for r in records):.2f}s "
      f"h1_final={best['label']} chi2={best['chi2']:.4f} -> {args.out}")
