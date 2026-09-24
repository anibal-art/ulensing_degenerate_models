"""
Common utilities for the H1-generated LRT population plots.

This module centralizes:
    - repository-relative paths,
    - loading the compact H1 Parquet,
    - basic consistency checks,
    - Delta-chi2 summary diagnostics,
    - figure saving.

No scientific selection on Delta chi2 is applied here.
In particular, no detection threshold is imposed before the
H0-generated calibration is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
from paper_style import apply_paper_style

apply_paper_style()


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_DATA_PATH = (
    REPO_ROOT
    / "analysis"
    / "lrt"
    / "data"
    / "h1_lrt_results_20260920.parquet"
)

DEFAULT_FIGURE_DIR = (
    REPO_ROOT
    / "analysis"
    / "lrt"
    / "figures"
    / "h1_delta_chi2"
)


# ============================================================
# Required columns
# ============================================================

REQUIRED_COLUMNS = [
    "catalog_row",
    "delta_chi2_lrt",
    "chi2_h0",
    "chi2_h1",
    "final_nesting_ok",
    "h0_optimizer_success",
    "h1_optimizer_success",
]


# ============================================================
# Data loading
# ============================================================

def load_h1_results(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load and validate the compact H1 LRT result table.

    This function does not remove:
        - negative Delta chi2,
        - final_nesting_ok=False events,
        - optimizer-diagnostic outliers.

    Those populations are scientifically relevant for auditing
    the production and must not disappear silently.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"H1 result file not found:\n{path}"
        )

    df = pd.read_parquet(path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if df["catalog_row"].duplicated().any():
        duplicates = int(
            df["catalog_row"].duplicated().sum()
        )

        raise RuntimeError(
            f"Found {duplicates} duplicated catalog_row values."
        )

    return df


# ============================================================
# Delta-chi2 extraction
# ============================================================

def finite_delta_chi2(df: pd.DataFrame) -> np.ndarray:
    """
    Return all finite Delta chi2 values.

    No sign cut and no LRT threshold are applied.
    """

    values = df["delta_chi2_lrt"].to_numpy(
        dtype=float
    )

    return values[
        np.isfinite(values)
    ]


# ============================================================
# Console diagnostics
# ============================================================

def print_delta_chi2_summary(df: pd.DataFrame) -> None:
    """
    Print the main integrity and Delta-chi2 diagnostics.
    """

    delta = finite_delta_chi2(df)

    n_total = len(df)
    n_finite = len(delta)

    n_negative = int(
        np.sum(delta < 0.0)
    )

    n_zero = int(
        np.sum(delta == 0.0)
    )

    n_positive = int(
        np.sum(delta > 0.0)
    )

    nesting_ok = (
        df["final_nesting_ok"]
        .fillna(False)
        .astype(bool)
    )

    h0_success = (
        df["h0_optimizer_success"]
        .fillna(False)
        .astype(bool)
    )

    h1_success = (
        df["h1_optimizer_success"]
        .fillna(False)
        .astype(bool)
    )

    print(
        "============================================================"
    )
    print("H1 LRT population")
    print(
        "============================================================"
    )
    print(f"N total                  = {n_total}")
    print(f"N finite Delta chi2      = {n_finite}")
    print(f"N Delta chi2 < 0         = {n_negative}")
    print(f"N Delta chi2 = 0         = {n_zero}")
    print(f"N Delta chi2 > 0         = {n_positive}")
    print(
        "fraction Delta chi2 < 0  = "
        f"{n_negative / n_finite:.6f}"
    )
    print(
        "final_nesting_ok          = "
        f"{int(nesting_ok.sum())}/{n_total}"
    )
    print(
        "H0 optimizer success      = "
        f"{int(h0_success.sum())}/{n_total}"
    )
    print(
        "H1 optimizer success      = "
        f"{int(h1_success.sum())}/{n_total}"
    )

    print()
    print("Delta chi2 quantiles:")

    quantiles = [
        0.001,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        0.999,
    ]

    for q in quantiles:
        value = np.quantile(
            delta,
            q,
        )

        print(
            f"q={q:6.3f}: "
            f"{value:14.6g}"
        )

    print(
        "============================================================"
    )


# ============================================================
# Figure output
# ============================================================

def ensure_figure_dir(
    path: Path = DEFAULT_FIGURE_DIR,
) -> Path:
    """
    Create and return the figure directory.
    """

    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def save_figure(
    fig,
    output_stem: Path,
    dpi: int = 300,
) -> None:
    """
    Save each scientific figure both as PNG and PDF.
    """

    output_stem = Path(
        output_stem
    )

    output_stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = output_stem.with_suffix(
        ".png"
    )

    pdf_path = output_stem.with_suffix(
        ".pdf"
    )

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


