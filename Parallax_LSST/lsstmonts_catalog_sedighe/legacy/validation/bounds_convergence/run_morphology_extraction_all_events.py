#!/usr/bin/env python3
"""
M0/M1/M2 driver: run the frozen morphology extraction (morphology_extraction.py)
on all 100 extreme100 events. No fits. No oracle/truth lookup here -- this
script only reads H5 light curves via the frozen fitter's own load_case().
"""
import sys
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bounds_audit"))
sys.path.insert(0, HERE)

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row", "71181",
    "--bounds-profile", "production_candidate",
    "--fit-scope", "h0",
    "--dry-run",
]

import pandas as pd  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402
from morphology_extraction import extract_event_morphology  # noqa: E402

truth_feat = pd.read_csv(
    os.path.join(HERE, "results", "b1c_event_features.csv")
).set_index("catalog_row")
rows_100 = sorted(truth_feat.index.tolist())
assert len(rows_100) == 100

records = []
for row in rows_100:
    meta = core.load_case({"sample": "extreme100", "catalog_row": int(row), "matched_tail_catalog_row": -1})
    r = extract_event_morphology(meta["curves"])
    r["catalog_row"] = int(row)
    if r.get("T25"):
        r["T25_value"] = r["T25"]["value"]
        r["T25_left"] = r["T25"]["left_width"]
        r["T25_right"] = r["T25"]["right_width"]
        r["T25_left_censored"] = r["T25"]["left_censored"]
        r["T25_right_censored"] = r["T25"]["right_censored"]
        del r["T25"]
    if r.get("T50"):
        r["T50_value"] = r["T50"]["value"]
        r["T50_left"] = r["T50"]["left_width"]
        r["T50_right"] = r["T50"]["right_width"]
        r["T50_left_censored"] = r["T50"]["left_censored"]
        r["T50_right_censored"] = r["T50"]["right_censored"]
        del r["T50"]
    if r.get("T75"):
        r["T75_value"] = r["T75"]["value"]
        r["T75_left"] = r["T75"]["left_width"]
        r["T75_right"] = r["T75"]["right_width"]
        r["T75_left_censored"] = r["T75"]["left_censored"]
        r["T75_right_censored"] = r["T75"]["right_censored"]
        del r["T75"]
    records.append(r)
    print(row, "measurable=", r["measurable"], r.get("reason", ""))

df = pd.DataFrame(records)
out_csv = os.path.join(HERE, "results", "morphology_M0_M1_M2_extreme100.csv")
df.to_csv(out_csv, index=False)
print("DONE ->", out_csv)
print("measurable:", int(df["measurable"].sum()), "/", len(df))
