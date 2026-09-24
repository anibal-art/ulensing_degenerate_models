#!/usr/bin/env python3
"""Check 2D covariance coverage for the recovered H1 parallax vector.

D2_error = (piE_fit-piE_true)^T C_piE^{-1} (piE_fit-piE_true).
For a calibrated Gaussian 2D covariance, the nominal coverage thresholds are
those of chi-square(df=2): 68%, 90%, 95%, 99%.
"""

from pathlib import Path
import sys
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

C.require_file(C.TRUTH_JOIN_PATH, "truth-joined H1 table")
df = pd.read_parquet(C.TRUTH_JOIN_PATH)
fitN_col = C.resolve_fit_parameter_column(df, "h1", "piEN")
fitE_col = C.resolve_fit_parameter_column(df, "h1", "piEE")
cNN_col = C.resolve_covariance_column(df, "h1", "piEN", "piEN")
cNE_col = C.resolve_covariance_column(df, "h1", "piEN", "piEE")
cEE_col = C.resolve_covariance_column(df, "h1", "piEE", "piEE")

trueN = pd.to_numeric(df["true_piEN"], errors="coerce").to_numpy(float)
trueE = pd.to_numeric(df["true_piEE"], errors="coerce").to_numpy(float)
fitN = pd.to_numeric(df[fitN_col], errors="coerce").to_numpy(float)
fitE = pd.to_numeric(df[fitE_col], errors="coerce").to_numpy(float)
cNN = pd.to_numeric(df[cNN_col], errors="coerce").to_numpy(float)
cNE = pd.to_numeric(df[cNE_col], errors="coerce").to_numpy(float)
cEE = pd.to_numeric(df[cEE_col], errors="coerce").to_numpy(float)

dN = fitN - trueN
dE = fitE - trueE
det = cNN*cEE - cNE*cNE
valid = np.isfinite(dN) & np.isfinite(dE) & np.isfinite(cNN) & np.isfinite(cNE) & np.isfinite(cEE) & (cNN > 0) & (cEE > 0) & (det > 0)
D2 = np.full(len(df), np.nan)
D2[valid] = (cEE[valid]*dN[valid]**2 - 2*cNE[valid]*dN[valid]*dE[valid] + cNN[valid]*dE[valid]**2) / det[valid]
valid &= np.isfinite(D2) & (D2 >= 0)

levels = np.asarray([0.68, 0.90, 0.95, 0.99])
thresholds = -2.0*np.log(1.0-levels)
rows = []
classes = ["all", "detected", "missed", "negative"]
for cls in classes:
    if cls == "all":
        m = valid
    else:
        m = valid & df["lrt_class_primary"].eq(cls).to_numpy()
    for level, q in zip(levels, thresholds):
        n = int(m.sum())
        k = int(np.sum(D2[m] <= q))
        lo, hi = C.wilson_interval(k, n)
        rows.append({
            "class": cls,
            "nominal_level": level,
            "chi2_df2_threshold": q,
            "n": n,
            "inside": k,
            "empirical_coverage": k/n if n else np.nan,
            "wilson_low": lo,
            "wilson_high": hi,
        })

OUT_R = C.RESULTS_DIR / "characterization" / "14_covariance_coverage"
OUT_F = C.FIGURES_DIR / "characterization" / "14_covariance_coverage"
C.ensure_dirs(OUT_R, OUT_F)
summary = pd.DataFrame(rows)
summary.to_csv(OUT_R / "coverage_summary.csv", index=False)
pd.DataFrame({"catalog_row": df["catalog_row"], "D2_piE_error": D2, "valid_covariance": valid, "lrt_class_primary": df["lrt_class_primary"]}).to_parquet(
    OUT_R / "coverage_by_event.parquet", index=False
)

fig, ax = plt.subplots(figsize=(5.8, 4.8))
for cls in ["all", "detected", "missed"]:
    tab = summary[summary["class"] == cls]
    ax.plot(tab["nominal_level"], tab["empirical_coverage"], marker="o", label=cls)
ax.plot([0.65, 1.0], [0.65, 1.0], linestyle="--", linewidth=1.0)
ax.set_xlim(0.65, 1.0)
ax.set_ylim(0.65, 1.0)
ax.set_xlabel("Nominal 2D Gaussian coverage")
ax.set_ylabel("Empirical coverage")
ax.grid(alpha=0.25)
ax.legend()
fig.savefig(OUT_F / "coverage_nominal_vs_empirical.png", dpi=300)
save_pdf_companion(fig, OUT_F / "coverage_nominal_vs_empirical.png")
plt.close(fig)

print(summary.to_string(index=False))


