#!/usr/bin/env python3
"""Shared utilities for the LRT validation and characterization scripts.

The module is intentionally read-only with respect to the production products.
It only reads the compact H0/H1 Parquet files and writes derived analysis tables
and figures under analysis/lrt/results and analysis/lrt/figures.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd

LRT_ROOT = Path(__file__).resolve().parent
DATA_DIR = LRT_ROOT / "data"
RESULTS_DIR = LRT_ROOT / "results"
FIGURES_DIR = LRT_ROOT / "figures"

H0_PATH = DATA_DIR / "h0_lrt_results_20260921.parquet"
H1_PATH = DATA_DIR / "h1_lrt_results_20260920.parquet"
CLASSIFIED_H1_PATH = DATA_DIR / "h1_lrt_classified_20260921.parquet"
TRUTH_JOIN_PATH = DATA_DIR / "h1_lrt_truth_joined_20260921.parquet"
CALIBRATION_TABLE_PATH = DATA_DIR / "h0_empirical_calibration_thresholds_20260921.csv"

ALPHAS = np.asarray([0.05, 0.01, 0.001, 0.0001], dtype=float)
PRIMARY_ALPHA = 0.001
RNG_SEED = 707070

SAMPLING_COLUMNS = [
    "n_photometry_points",
    "nearest_peak_distance_tE",
    "n_within_0p25_tE",
    "n_within_0p5_tE",
    "n_within_1_tE",
    "n_within_2_tE",
    "n_left_within_1_tE",
    "n_right_within_1_tE",
    "poor_peak_coverage",
]

OPTIMIZER_COLUMNS = [
    "h0_optimizer_optimality",
    "h1_optimizer_optimality",
    "h0_optimizer_nfev",
    "h1_optimizer_nfev",
    "h0_optimizer_n_active_bounds",
    "h1_optimizer_n_active_bounds",
    "h0_optimizer_success",
    "h1_optimizer_success",
]


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def require_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_h0(columns=None) -> pd.DataFrame:
    require_file(H0_PATH, "H0 compact Parquet")
    return pd.read_parquet(H0_PATH, columns=columns)


def load_h1(columns=None, prefer_classified: bool = False) -> pd.DataFrame:
    path = CLASSIFIED_H1_PATH if prefer_classified and CLASSIFIED_H1_PATH.is_file() else H1_PATH
    require_file(path, "H1 compact Parquet")
    return pd.read_parquet(path, columns=columns)


def finite_lrt(df: pd.DataFrame) -> np.ndarray:
    x = pd.to_numeric(df["delta_chi2_lrt"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(x)):
        bad = int((~np.isfinite(x)).sum())
        raise RuntimeError(f"Found {bad} non-finite delta_chi2_lrt values")
    return x


def wilks_threshold(alpha: float) -> float:
    """For chi-square with 2 dof, sf(c)=exp(-c/2)."""
    alpha = float(alpha)
    return float(-2.0 * np.log(alpha))


def empirical_threshold(values: np.ndarray, alpha: float) -> float:
    """Observed conservative quantile using numpy's 'higher' convention."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite values")
    return float(np.quantile(values, 1.0 - float(alpha), method="higher"))


def empirical_pvalues(null_values: np.ndarray, test_values: np.ndarray) -> np.ndarray:
    """Population-level empirical p-values with the +1 finite-sample correction.

    p = (1 + #{T0 >= T}) / (N0 + 1)
    """
    null_sorted = np.sort(np.asarray(null_values, dtype=float))
    test_values = np.asarray(test_values, dtype=float)
    n0 = len(null_sorted)
    left = np.searchsorted(null_sorted, test_values, side="left")
    n_ge = n0 - left
    return (1.0 + n_ge) / (n0 + 1.0)


def empirical_survival(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """P(T > c) on an arbitrary threshold grid, using all draws including negatives."""
    x = np.sort(np.asarray(values, dtype=float))
    grid = np.asarray(grid, dtype=float)
    idx = np.searchsorted(x, grid, side="right")
    return (len(x) - idx) / len(x)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return center - half, center + half


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_column(columns, candidates, required_tokens=None):
    cols = list(columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate

    norm = {normalize_name(c): c for c in cols}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in norm:
            return norm[key]

    if required_tokens:
        tokens = [normalize_name(t) for t in required_tokens]
        matches = []
        for c in cols:
            nc = normalize_name(c)
            if all(t in nc for t in tokens):
                matches.append(c)
        if len(matches) == 1:
            return matches[0]

    return None


def resolve_fit_parameter_column(df: pd.DataFrame, prefix: str, parameter: str) -> str:
    candidates = [
        f"{prefix}_{parameter}",
        f"{prefix}_{parameter}_fit",
        f"{prefix}_fit_{parameter}",
        f"{prefix}_best_{parameter}",
        f"{prefix}_parameter_{parameter}",
    ]
    col = resolve_column(df.columns, candidates, required_tokens=[prefix, parameter])
    if col is None:
        matches = [c for c in df.columns if parameter.lower() in c.lower()]
        raise KeyError(
            f"Could not resolve fitted parameter {prefix}/{parameter}. "
            f"Related columns: {matches}"
        )
    return col


def resolve_covariance_column(df: pd.DataFrame, prefix: str, p1: str, p2: str) -> str:
    candidates = [
        f"{prefix}_cov_{p1}_{p2}",
        f"{prefix}_cov_{p2}_{p1}",
        f"{prefix}_covariance_{p1}_{p2}",
        f"{prefix}_covariance_{p2}_{p1}",
        f"{prefix}_{p1}_{p2}_cov",
        f"{prefix}_{p2}_{p1}_cov",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    tokens = [normalize_name(prefix), "cov", normalize_name(p1), normalize_name(p2)]
    matches = []
    for c in df.columns:
        nc = normalize_name(c)
        if all(t in nc for t in tokens):
            matches.append(c)

    if len(matches) == 1:
        return matches[0]

    cov_matches = [c for c in df.columns if "cov" in c.lower() and prefix.lower() in c.lower()]
    raise KeyError(
        f"Could not resolve covariance element {prefix} ({p1},{p2}). "
        f"Candidate covariance columns: {cov_matches}"
    )


def classify_lrt(values: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), "missed", dtype=object)
    out[values < 0] = "negative"
    out[values > threshold] = "detected"
    return out


def equal_count_bins(values: pd.Series, q: int = 5):
    clean = pd.to_numeric(values, errors="coerce")
    return pd.qcut(clean, q=q, duplicates="drop")


def save_json(data, path: Path) -> Path:
    path = Path(path)
    ensure_dirs(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
