#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select 4 representative H1 positive-population events for light-curve figures.

NO simulation.
NO fitting.

We select one event for each of the following categories:

1. parallax_detected
2. confused_fspl_no_parallax
3. anomalous_map_cell
   -> long events with intermediate / large true parallax amplitude
      where the detection map shows non-monotonic behavior
4. finite_source_dominated
   -> rho_true / |u0_true| > 1

Input
-----
analysis/lrt/results/characterization/18_positive_truth/
    h1_positive_truth_characterization.parquet

Output
------
analysis/lrt/results/characterization/32_lightcurve_examples/
    selected_lightcurve_examples.csv
    selected_lightcurve_examples.txt

Selection philosophy
--------------------
We do NOT pick arbitrary extreme outliers unless explicitly intended.

Instead:
- for "detected" and "confused", we pick representative events close to
  the class medians in key variables;
- for the anomalous map-cell case, we restrict to the suspicious region
  and then pick a representative event there;
- for finite-source dominated, we pick a representative event among the
  rho/|u0| > 1 population.

This gives examples that are interpretable in the talk and not just
pathological one-offs.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path("analysis/lrt/results/characterization")

INPUT = (
    ROOT
    / "18_positive_truth"
    / "h1_positive_truth_characterization.parquet"
)

OUTDIR = (
    ROOT
    / "32_lightcurve_examples"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

OUTCSV = OUTDIR / "selected_lightcurve_examples.csv"
OUTTXT = OUTDIR / "selected_lightcurve_examples.txt"


# ============================================================
# Utility helpers
# ============================================================

def robust_scale(x: pd.Series) -> float:
    """
    Robust scale estimate based on IQR.
    Falls back to std if IQR is zero/non-finite, and then to 1.
    """
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return 1.0

    q25 = np.nanpercentile(x, 25.0)
    q75 = np.nanpercentile(x, 75.0)
    iqr = q75 - q25

    if np.isfinite(iqr) and iqr > 0:
        return float(iqr)

    std = np.nanstd(x)

    if np.isfinite(std) and std > 0:
        return float(std)

    return 1.0


def add_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "piE_true_amp" in out.columns:
        out["log10_piE_true_amp"] = np.where(
            out["piE_true_amp"] > 0,
            np.log10(out["piE_true_amp"]),
            np.nan,
        )

    if "tE_true_days" in out.columns:
        out["log10_tE_true_days"] = np.where(
            out["tE_true_days"] > 0,
            np.log10(out["tE_true_days"]),
            np.nan,
        )

    if "rho_over_abs_u0_true" in out.columns:
        out["log10_rho_over_abs_u0_true"] = np.where(
            out["rho_over_abs_u0_true"] > 0,
            np.log10(out["rho_over_abs_u0_true"]),
            np.nan,
        )

    if "delta_chi2_lrt" in out.columns:
        positive_T = out["delta_chi2_lrt"] > 0
        out["log10_delta_chi2_lrt"] = np.nan
        out.loc[positive_T, "log10_delta_chi2_lrt"] = np.log10(
            out.loc[positive_T, "delta_chi2_lrt"]
        )

    return out


def choose_representative(
    df: pd.DataFrame,
    target_cols: list[str],
    label: str,
    hard_preference: dict | None = None,
) -> pd.Series:
    """
    Pick a representative event by minimizing the quadratic distance
    to the class medians in the specified target columns.

    Parameters
    ----------
    df : candidate subset
    target_cols : columns used for the representative distance
    label : category label to attach
    hard_preference : optional dict
        Example:
            {
                "sort_by": "delta_chi2_lrt",
                "ascending": False,
                "top_n": 500,
            }
        First keep the top_n rows according to sort_by before
        building the representative score.
    """
    if len(df) == 0:
        raise RuntimeError(f"No candidates available for category '{label}'.")

    work = df.copy()

    if hard_preference is not None:
        sort_by = hard_preference.get("sort_by", None)
        ascending = hard_preference.get("ascending", False)
        top_n = hard_preference.get("top_n", None)

        if sort_by is not None and sort_by in work.columns:
            work = work.sort_values(sort_by, ascending=ascending)

        if top_n is not None:
            work = work.head(int(top_n)).copy()

        if len(work) == 0:
            raise RuntimeError(
                f"Preference filtering removed all candidates for '{label}'."
            )

    usable_cols = []
    for c in target_cols:
        if c not in work.columns:
            continue
        values = pd.to_numeric(work[c], errors="coerce")
        if np.isfinite(values).sum() == 0:
            continue
        usable_cols.append(c)

    if len(usable_cols) == 0:
        raise RuntimeError(
            f"No usable target columns found for category '{label}'."
        )

    medians = {
        c: np.nanmedian(pd.to_numeric(work[c], errors="coerce"))
        for c in usable_cols
    }

    scales = {
        c: robust_scale(pd.to_numeric(work[c], errors="coerce"))
        for c in usable_cols
    }

    score = np.zeros(len(work), dtype=float)

    for c in usable_cols:
        x = pd.to_numeric(work[c], errors="coerce").to_numpy(dtype=float)
        med = medians[c]
        scl = scales[c]

        term = ((x - med) / scl) ** 2
        term[~np.isfinite(term)] = np.inf
        score += term

    work = work.assign(representative_score=score)
    best = work.nsmallest(1, "representative_score").iloc[0].copy()
    best["example_category"] = label

    return best


def summarize_row(row: pd.Series) -> dict:
    keep = [
        "example_category",
        "catalog_row",
        "positive_lrt_class",
        "delta_chi2_lrt",
        "tE_true_days",
        "piE_true_amp",
        "rho_over_abs_u0_true",
        "finite_source_dominated_true",
        "u0_true",
        "rho_true",
        "DL_kpc",
        "DS_kpc",
        "pi_rel_mas_from_distance",
        "h1_tE",
        "h1_u0",
        "h1_rho",
        "h1_piEN",
        "h1_piEE",
        "h0_tE",
        "h0_u0",
        "h0_rho",
    ]

    out = {}
    for c in keep:
        if c in row.index:
            out[c] = row[c]
    return out


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 90)
    print("SELECT REPRESENTATIVE LIGHT-CURVE EXAMPLES")
    print("=" * 90)

    df = pd.read_parquet(INPUT)
    df = add_log_columns(df)

    print(f"Input rows = {len(df):,}")
    print()
    print("positive_lrt_class:")
    print(df["positive_lrt_class"].value_counts(dropna=False))
    print()

    selected_rows = []

    # --------------------------------------------------------
    # 1) Detected: representative detected event
    # --------------------------------------------------------
    detected = df.loc[
        df["positive_lrt_class"] == "parallax_detected"
    ].copy()

    row_detected = choose_representative(
        detected,
        target_cols=[
            "log10_tE_true_days",
            "log10_piE_true_amp",
            "log10_delta_chi2_lrt",
        ],
        label="detected_representative",
    )

    selected_rows.append(row_detected)

    # --------------------------------------------------------
    # 2) Confused: representative confused event
    # --------------------------------------------------------
    confused = df.loc[
        df["positive_lrt_class"] == "confused_fspl_no_parallax"
    ].copy()

    row_confused = choose_representative(
        confused,
        target_cols=[
            "log10_tE_true_days",
            "log10_piE_true_amp",
            "log10_delta_chi2_lrt",
        ],
        label="confused_representative",
    )

    selected_rows.append(row_confused)

    # --------------------------------------------------------
    # 3) Anomalous map-cell:
    #    long-duration, intermediate/high piE
    # --------------------------------------------------------
    anomalous = df.loc[
        (df["tE_true_days"] > 100.0)
        & (df["piE_true_amp"] >= 2.0)
        & (df["piE_true_amp"] <= 4.0)
    ].copy()

    if len(anomalous) == 0:
        raise RuntimeError(
            "No candidates found in the requested anomalous cell "
            "(tE_true_days > 100 and 2 <= piE_true_amp <= 4)."
        )

    print("Anomalous-cell candidates:")
    print(f"  n = {len(anomalous):,}")
    print("  class counts:")
    print(anomalous["positive_lrt_class"].value_counts())
    print()

    row_anomalous = choose_representative(
        anomalous,
        target_cols=[
            "log10_tE_true_days",
            "log10_piE_true_amp",
            "log10_delta_chi2_lrt",
        ],
        label="anomalous_map_cell_representative",
    )

    selected_rows.append(row_anomalous)

    # --------------------------------------------------------
    # 4) Finite-source dominated:
    #    rho_true / |u0_true| > 1
    # --------------------------------------------------------
    fsd = df.loc[
        df["finite_source_dominated_true"] == True
    ].copy()

    if len(fsd) == 0:
        raise RuntimeError(
            "No finite-source-dominated events found."
        )

    print("Finite-source dominated candidates:")
    print(f"  n = {len(fsd):,}")
    print("  class counts:")
    print(fsd["positive_lrt_class"].value_counts())
    print()

    row_fsd = choose_representative(
        fsd,
        target_cols=[
            "log10_tE_true_days",
            "log10_piE_true_amp",
            "log10_rho_over_abs_u0_true",
            "log10_delta_chi2_lrt",
        ],
        label="finite_source_dominated_representative",
    )

    selected_rows.append(row_fsd)

    # --------------------------------------------------------
    # Build output table
    # --------------------------------------------------------
    out = pd.DataFrame(
        [summarize_row(row) for row in selected_rows]
    )

    # prevent accidental duplicates
    if out["catalog_row"].duplicated().any():
        dup = out.loc[out["catalog_row"].duplicated(), "catalog_row"].tolist()
        raise RuntimeError(
            f"Selection produced duplicate catalog_row values: {dup}"
        )

    out.to_csv(OUTCSV, index=False)

    # --------------------------------------------------------
    # Write readable text summary
    # --------------------------------------------------------
    lines = []
    lines.append("=" * 90)
    lines.append("SELECTED LIGHT-CURVE EXAMPLES")
    lines.append("=" * 90)
    lines.append(f"Input file: {INPUT}")
    lines.append("")

    for _, row in out.iterrows():
        lines.append("-" * 90)
        lines.append(f"Category   : {row['example_category']}")
        lines.append(f"catalog_row: {int(row['catalog_row'])}")
        lines.append(f"class      : {row.get('positive_lrt_class', 'NA')}")
        lines.append(f"T          : {row.get('delta_chi2_lrt', np.nan):.6g}")
        lines.append(f"tE_true    : {row.get('tE_true_days', np.nan):.6g} d")
        lines.append(f"|piE|_true : {row.get('piE_true_amp', np.nan):.6g}")
        lines.append(
            f"rho/|u0|   : {row.get('rho_over_abs_u0_true', np.nan):.6g}"
        )
        lines.append(
            f"finite src : {row.get('finite_source_dominated_true', 'NA')}"
        )
        lines.append(
            f"DL, DS [kpc]: "
            f"{row.get('DL_kpc', np.nan):.6g}, "
            f"{row.get('DS_kpc', np.nan):.6g}"
        )
        lines.append(
            f"H1 fit: tE={row.get('h1_tE', np.nan):.6g}, "
            f"u0={row.get('h1_u0', np.nan):.6g}, "
            f"rho={row.get('h1_rho', np.nan):.6g}, "
            f"piEN={row.get('h1_piEN', np.nan):.6g}, "
            f"piEE={row.get('h1_piEE', np.nan):.6g}"
        )
        lines.append(
            f"H0 fit: tE={row.get('h0_tE', np.nan):.6g}, "
            f"u0={row.get('h0_u0', np.nan):.6g}, "
            f"rho={row.get('h0_rho', np.nan):.6g}"
        )
        lines.append("")

    text = "\n".join(lines)
    OUTTXT.write_text(text)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("=" * 90)
    print("SELECTED EXAMPLES")
    print("=" * 90)
    print(out.to_string(index=False))
    print()
    print(f"Saved CSV : {OUTCSV}")
    print(f"Saved TXT : {OUTTXT}")


if __name__ == "__main__":
    main()
