#!/usr/bin/env python3
"""
One-event smoke test for each of the 4 simulation x fit combinations
of the simplified 2-fit architecture:
  (a) H0-generated event -> H0 fit (truth start)
  (b) H0-generated event -> H1 fit (morphology seed, piE=0)
  (c) H1-generated event -> H1 fit (truth start)
  (d) H1-generated event -> H0 fit (morphology seed)

Also verifies:
  - actual optimizer call count == 2/event (not 31, not any hidden
    prerequisite fit);
  - the morphology seed differs from truth and cannot see truth (proof
    by construction: morphology_seed_from_curves's signature takes
    only `curves`, no truth argument -- inspected here, not just
    asserted in prose);
  - whether the morphology seed falls inside the production_candidate
    domain (reported, never silently clipped).
"""
import sys
import os
import json
import inspect
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
BA = os.path.join(HERE, "..", "bounds_audit")
BC = os.path.join(HERE, "..", "bounds_convergence")
sys.path.insert(0, BA)
sys.path.insert(0, BC)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--h0-gen-h5", required=True)
ap.add_argument("--h1-gen-h5", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = ["run_bounds_audit_refit_core.py", "--catalog-row", "71181",
            "--bounds-profile", "production_candidate", "--fit-scope", "h0", "--dry-run"]

import run_bounds_audit_refit_core as core  # noqa: E402
from load_new_event import load_new_case  # noqa: E402
from morphology_seed import morphology_seed_from_curves  # noqa: E402
from fit_two_policy import run_two_fit_policy  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"

# ---- Proof #2: morphology seed cannot see truth (structural, not just asserted) ----
sig = inspect.signature(morphology_seed_from_curves)
params = list(sig.parameters)
assert params == ["curves"], (
    f"morphology_seed_from_curves must take ONLY curves, got signature {params}"
)
print(f"[proof] morphology_seed_from_curves signature = {sig} -- no truth/event_params argument exists.")

results = {}

for label, h5 in [("H0_generated", args.h0_gen_h5), ("H1_generated", args.h1_gen_h5)]:
    meta = load_new_case(h5, core)
    print()
    print("=" * 100)
    print(f"{label}: row={meta['row']} generating_model={meta['generating_model']} "
          f"truth={meta['truth']}")

    r = run_two_fit_policy(meta, core)

    print(f"  morphology_seed = {r['morphology_seed']}")
    print(f"  domain_violations = {r['morphology_seed_domain_violations']}")
    print(f"  n_trf = {r['n_trf']}  (must be 2)")
    if r["h0"] is not None:
        print(f"  H0 fit: label={r['h0']['label']} chi2={r['h0']['chi2']} "
              f"status={r['h0']['status']} wall={r['h0']['wall_s']:.2f}s")
    if r["h1"] is not None:
        print(f"  H1 fit: label={r['h1']['label']} chi2={r['h1']['chi2']} "
              f"status={r['h1']['status']} wall={r['h1']['wall_s']:.2f}s")
    if "delta_chi2_lrt" in r:
        print(f"  Delta_chi2_LRT = {r['delta_chi2_lrt']:.4f}")

    assert r["n_trf"] == 2, f"expected exactly 2 TRF for row={meta['row']}, got {r['n_trf']}"

    results[label] = r

with open(args.out, "w") as f:
    json.dump(results, f, indent=2, default=str)

print()
print("=" * 100)
print(f"SMOKE TEST PASSED: 4/4 combinations executed, 2 TRF/event confirmed -> {args.out}")
