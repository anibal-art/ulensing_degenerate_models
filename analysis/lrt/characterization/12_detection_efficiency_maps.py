#!/usr/bin/env python3
"""Two-dimensional maps of calibrated H1 parallax-detection efficiency.

Maps are computed for physically informative pairs using generating truth.
Bins with fewer than --min-count events are masked rather than overinterpreted.
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
parser.add_argument("--bins", type=int, default=20)
parser.add_argument("--min-count", type=int, default=30)
args = parser.parse_args()

C.require_file(C.TRUTH_JOIN_PATH, "truth-joined H1 table")
df = pd.read_parquet(C.TRUTH_JOIN_PATH)
detected = df["lrt_class_primary"].eq("detected").to_numpy()

pairs = [
    ("true_tE", "true_piE_abs", True, True),
    ("true_tE", "true_u0", True, False),
    ("true_tE", "true_rho", True, True),
]

OUT_R = C.RESULTS_DIR / "characterization" / "12_detection_efficiency_maps"
OUT_F = C.FIGURES_DIR / "characterization" / "12_detection_efficiency_maps"
C.ensure_dirs(OUT_R, OUT_F)

for xname, yname, logx, logy in pairs:
    if xname not in df.columns or yname not in df.columns:
        continue
    x = pd.to_numeric(df[xname], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[yname], errors="coerce").to_numpy(float)
    good = np.isfinite(x) & np.isfinite(y)
    if logx:
        good &= x > 0
    if logy:
        good &= y > 0
    x = x[good]
    y = y[good]
    d = detected[good].astype(float)

    if logx:
        xlo, xhi = np.quantile(x, [0.005, 0.995])
        xedges = np.geomspace(xlo, xhi, args.bins + 1)
    else:
        xlo, xhi = np.quantile(x, [0.005, 0.995])
        xedges = np.linspace(xlo, xhi, args.bins + 1)
    if logy:
        ylo, yhi = np.quantile(y, [0.005, 0.995])
        yedges = np.geomspace(ylo, yhi, args.bins + 1)
    else:
        ylo, yhi = np.quantile(y, [0.005, 0.995])
        yedges = np.linspace(ylo, yhi, args.bins + 1)

    counts, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    detected_counts, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=d)
    eff = np.divide(detected_counts, counts, out=np.full_like(counts, np.nan), where=counts >= args.min_count)

    rows = []
    for i in range(args.bins):
        for j in range(args.bins):
            rows.append({
                "x_lo": xedges[i], "x_hi": xedges[i+1],
                "y_lo": yedges[j], "y_hi": yedges[j+1],
                "n": int(counts[i, j]),
                "n_detected": int(detected_counts[i, j]),
                "efficiency": eff[i, j],
            })
    pd.DataFrame(rows).to_csv(OUT_R / f"{xname}_vs_{yname}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    mesh = ax.pcolormesh(xedges, yedges, eff.T, vmin=0, vmax=1, shading="auto")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    fig.colorbar(mesh, ax=ax, label="Detection efficiency")
    fig.savefig(OUT_F / f"{xname}_vs_{yname}.png", dpi=300)
    save_pdf_companion(fig, OUT_F / f"{xname}_vs_{yname}.png")
    plt.close(fig)

print("Saved maps in:", OUT_F)


