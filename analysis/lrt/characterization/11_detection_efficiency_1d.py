#!/usr/bin/env python3
"""Measure calibrated H1 detection efficiency versus individual event properties.

The primary detection definition is the empirically calibrated alpha=1e-3 LRT
threshold. The script uses generating truth for physical parameters and the
production sampling diagnostics for observational properties.
"""

from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys as _paper_sys
from pathlib import Path as _PaperPath

_PAPER_LRT_DIR = next(
    parent
    for parent in _PaperPath(__file__).resolve().parents
    if parent.name == "lrt"
)
_paper_sys.path.insert(
    0,
    str(_PAPER_LRT_DIR / "plotting"),
)
from paper_style import (
    apply_paper_style,
    label_panels,
    save_pdf_companion,
)

apply_paper_style()

LRT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LRT_ROOT))
import lrt_common as C

parser = argparse.ArgumentParser()
parser.add_argument("--bins", type=int, default=10)
args = parser.parse_args()

C.require_file(C.TRUTH_JOIN_PATH, "truth-joined H1 table; run 10_prepare_truth_join.py")
df = pd.read_parquet(C.TRUTH_JOIN_PATH)
if "lrt_class_primary" not in df.columns:
    raise KeyError("Missing lrt_class_primary; run validation_tests/09 before the truth join")

detected = df["lrt_class_primary"].eq("detected").to_numpy()
variables = [
    "true_tE", "true_u0", "true_rho", "true_piE_abs",
    "n_photometry_points", "nearest_peak_distance_tE",
    "n_within_0p25_tE", "n_within_0p5_tE", "n_within_1_tE",
]
variables = [v for v in variables if v in df.columns]

OUT_R = C.RESULTS_DIR / "characterization" / "11_detection_efficiency_1d"
OUT_F = C.FIGURES_DIR / "characterization" / "11_detection_efficiency_1d"
C.ensure_dirs(OUT_R, OUT_F)

all_rows = []
for var in variables:
    x = pd.to_numeric(df[var], errors="coerce")
    valid = x.notna().to_numpy() & np.isfinite(x.to_numpy(float))
    x_valid = x[valid]
    d_valid = detected[valid]
    bins = pd.qcut(x_valid, q=args.bins, duplicates="drop")

    rows = []
    tmp = pd.DataFrame({"x": x_valid.to_numpy(float), "bin": bins.astype("string"), "det": d_valid})
    for group, g in tmp.groupby("bin", observed=False):
        n = len(g)
        k = int(g["det"].sum())
        lo, hi = C.wilson_interval(k, n)
        rows.append({
            "variable": var,
            "bin": str(group),
            "x_median": float(g["x"].median()),
            "x_q25": float(g["x"].quantile(0.25)),
            "x_q75": float(g["x"].quantile(0.75)),
            "n": n,
            "n_detected": k,
            "efficiency": k/n,
            "wilson_low": lo,
            "wilson_high": hi,
        })
    tab = pd.DataFrame(rows).sort_values("x_median")
    tab.to_csv(OUT_R / f"{var}.csv", index=False)
    all_rows.append(tab)

    xplot = tab["x_median"].to_numpy()
    y = tab["efficiency"].to_numpy()
    yerr = np.vstack([y-tab["wilson_low"].to_numpy(), tab["wilson_high"].to_numpy()-y])
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.errorbar(xplot, y, yerr=yerr, fmt="o-", capsize=3)
    if var in {"true_tE", "true_rho", "true_piE_abs"} and np.all(xplot > 0):
        ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel(var)
    ax.set_ylabel("Calibrated H1 detection efficiency")
    ax.grid(alpha=0.25)
    fig.savefig(OUT_F / f"{var}.png", dpi=300)
    save_pdf_companion(fig, OUT_F / f"{var}.png")
    plt.close(fig)

pd.concat(all_rows, ignore_index=True).to_csv(OUT_R / "all_1d_efficiencies.csv", index=False)
print("Saved results in:", OUT_R)


