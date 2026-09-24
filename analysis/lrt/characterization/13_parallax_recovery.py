#!/usr/bin/env python3
"""Assess recovery of the parallax vector for detected, missed and negative H1 events."""

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

trueN = pd.to_numeric(df["true_piEN"], errors="coerce").to_numpy(float)
trueE = pd.to_numeric(df["true_piEE"], errors="coerce").to_numpy(float)
fitN = pd.to_numeric(df[fitN_col], errors="coerce").to_numpy(float)
fitE = pd.to_numeric(df[fitE_col], errors="coerce").to_numpy(float)
true_abs = np.hypot(trueN, trueE)
fit_abs = np.hypot(fitN, fitE)
vec_err = np.hypot(fitN-trueN, fitE-trueE)
rel_vec_err = np.divide(vec_err, true_abs, out=np.full_like(vec_err, np.nan), where=true_abs > 0)

out = pd.DataFrame({
    "catalog_row": df["catalog_row"].to_numpy(),
    "lrt_class_primary": df["lrt_class_primary"].astype(str).to_numpy(),
    "true_piEN": trueN,
    "true_piEE": trueE,
    "fit_piEN": fitN,
    "fit_piEE": fitE,
    "true_piE_abs": true_abs,
    "fit_piE_abs": fit_abs,
    "piE_vector_error": vec_err,
    "relative_piE_vector_error": rel_vec_err,
})

OUT_R = C.RESULTS_DIR / "characterization" / "13_parallax_recovery"
OUT_F = C.FIGURES_DIR / "characterization" / "13_parallax_recovery"
C.ensure_dirs(OUT_R, OUT_F)
out.to_parquet(OUT_R / "parallax_recovery_by_event.parquet", index=False)

rows = []
for cls, g in out.groupby("lrt_class_primary"):
    x = g["relative_piE_vector_error"].replace([np.inf, -np.inf], np.nan).dropna()
    rows.append({
        "class": cls,
        "n": len(g),
        "n_finite_relative_error": len(x),
        "median_relative_vector_error": float(x.median()) if len(x) else np.nan,
        "q16": float(x.quantile(0.16)) if len(x) else np.nan,
        "q84": float(x.quantile(0.84)) if len(x) else np.nan,
        "fraction_relerr_lt_0p5": float((x < 0.5).mean()) if len(x) else np.nan,
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT_R / "parallax_recovery_summary.csv", index=False)

mask = np.isfinite(true_abs) & np.isfinite(fit_abs) & (true_abs > 0) & (fit_abs > 0)
fig, ax = plt.subplots(figsize=(5.8, 4.8))
hb = ax.hexbin(np.log10(true_abs[mask]), np.log10(fit_abs[mask]), gridsize=70, mincnt=1, bins="log")
lo = min(np.nanpercentile(np.log10(true_abs[mask]), 0.5), np.nanpercentile(np.log10(fit_abs[mask]), 0.5))
hi = max(np.nanpercentile(np.log10(true_abs[mask]), 99.5), np.nanpercentile(np.log10(fit_abs[mask]), 99.5))
ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
ax.set_xlabel(r"$\log_{10}|\pi_{E,\rm true}|$")
ax.set_ylabel(r"$\log_{10}|\pi_{E,\rm fit}|$")
fig.colorbar(hb, ax=ax, label="log density")
fig.savefig(OUT_F / "piE_amplitude_true_vs_fit.png", dpi=300)
save_pdf_companion(fig, OUT_F / "piE_amplitude_true_vs_fit.png")
plt.close(fig)

print(summary.to_string(index=False))


