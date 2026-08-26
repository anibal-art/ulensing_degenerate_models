#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select and plot LSSTMONTS events using a Delta chi2 / LRT window.

This version is adapted for the current CHE production:

    RUN_TAG = prod_multifit_20260826T041803Z

It can read either:

    1. Old candidates CSV, with columns such as:
       global_i, chi2_red, n_peak_total, rho_true, rho_fit,
       tE_true, tE_fit, u0_true, u0_fit, fit_file

    2. Current multi_fit analysis CSV, with columns such as:
       LRT, p_value_LRT, H0_chi2, H1_chi2, true_generator_chi2,
       H0_best_model, H1_best_model, t0_jd, u0, tE_catalog_days,
       rho_catalog, piEN, piEE, event_dir, simulation_seed

The script:

    - standardizes columns;
    - selects events with delta_chi2_min <= Delta chi2 < delta_chi2_max;
    - makes diagnostic plots;
    - chooses representative events;
    - calls plot_confused_lightcurves_with_inset_pipeline.py;
    - passes the correct --pipeline path;
    - overlays an annotation box with:
        true parameters,
        H0 fitted parameters,
        H1 fitted parameters,
        chi2_H0, chi2_H1, chi2_true,
        Delta chi2 / LRT,
        p_LRT,
        quality flag.
"""

from pathlib import Path
import argparse
import ast
import subprocess
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ============================================================
# Defaults for CHE
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_RUN_TAG = "prod_multifit_20260826T041803Z"

DEFAULT_ANALYSIS_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/partial_analysis"
)

DEFAULT_RUNS_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/runs"
)

DEFAULT_PLOT_SCRIPT = SCRIPT_DIR / "plot_confused_lightcurves_with_inset_pipeline.py"

DEFAULT_PIPELINE = SCRIPT_DIR / "pipeline_hidden_parallax.py"
DEFAULT_RUNNER = SCRIPT_DIR / "run_lsstmonts_catalog_hidden_parallax.py"

DEFAULT_CONFIG = SCRIPT_DIR / "config_lsstmonts_baseline_v5p3p5_cluster_che_multifit_LRT.json"


# ============================================================
# Basic helpers
# ============================================================

def as_float(x, default=np.nan):
    try:
        y = float(x)
        if np.isfinite(y):
            return y
    except Exception:
        pass
    return default


def safe_name(x):
    return (
        str(x)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("|", "_")
        .replace(":", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )


def normalize_bool_column(series):
    if series.dtype == bool:
        return series.fillna(False)

    s = series.astype(str).str.strip().str.lower()

    return s.isin(["true", "1", "yes", "y", "t"])


def resolve_existing_path(path, label):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe {label}: {p}")
    return p


def find_latest_analysis_csv(run_tag, analysis_root, filename):
    analysis_root = Path(analysis_root)

    files = sorted(
        p for p in analysis_root.glob(f"{run_tag}_*/{filename}")
        if p.is_file()
    )

    if len(files) == 0:
        return None

    return files[-1]


def find_default_input_csv(run_tag, analysis_root):
    """
    Prefer the full partial analysis table.
    Fall back to selected_lightcurve_events.csv.
    """

    full_csv = find_latest_analysis_csv(
        run_tag,
        analysis_root,
        "multi_fit_partial_with_quality_flags.csv",
    )

    if full_csv is not None:
        return full_csv

    selected_csv = find_latest_analysis_csv(
        run_tag,
        analysis_root,
        "selected_lightcurve_events.csv",
    )

    if selected_csv is not None:
        return selected_csv

    raise FileNotFoundError(
        f"No encontré multi_fit_partial_with_quality_flags.csv ni "
        f"selected_lightcurve_events.csv para run_tag={run_tag} en {analysis_root}"
    )


# ============================================================
# Best-model parsing
# ============================================================

def parse_best_model(value):
    """
    Convert H0_best_model / H1_best_model to np.array.

    Supports:
        - np.ndarray
        - list / tuple
        - string '[1, 2, 3]'
        - string '[1 2 3]'
        - string 'array([...])'
    """

    if value is None:
        return None

    if isinstance(value, np.ndarray):
        try:
            arr = value.astype(float)
            return arr if len(arr) > 0 else None
        except Exception:
            return None

    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value, dtype=float)
            return arr if len(arr) > 0 else None
        except Exception:
            return None

    if isinstance(value, str):
        s = value.strip()

        if s == "" or s.lower() in ["none", "nan"]:
            return None

        s = s.replace("array(", "")

        if s.endswith(")"):
            s = s[:-1]

        try:
            obj = ast.literal_eval(s)
            arr = np.asarray(obj, dtype=float)
            return arr if len(arr) > 0 else None
        except Exception:
            pass

        s2 = (
            s.replace("[", " ")
             .replace("]", " ")
             .replace(",", " ")
             .replace("\n", " ")
        )

        arr = np.fromstring(s2, sep=" ")

        if len(arr) > 0:
            return arr.astype(float)

    return None


def extract_fit_params(row, prefix):
    """
    Expected order:

        H0 FSPL no parallax:
            t0, u0, tE, rho, ...

        H1 FSPL parallax:
            t0, u0, tE, rho, piEN, piEE, ...
    """

    key = f"{prefix}_best_model"

    if key not in row:
        return {}

    arr = parse_best_model(row[key])

    if arr is None:
        return {}

    names = ["t0", "u0", "tE", "rho", "piEN", "piEE"]

    out = {}

    for i, name in enumerate(names):
        if i < len(arr):
            out[name] = float(arr[i])

    if prefix == "H0":
        out.pop("piEN", None)
        out.pop("piEE", None)

    return out


def add_fit_param_columns(df):
    df = df.copy()

    h0_params = []
    h1_params = []

    for _, row in df.iterrows():
        h0_params.append(extract_fit_params(row, "H0"))
        h1_params.append(extract_fit_params(row, "H1"))

    for name in ["t0", "u0", "tE", "rho", "piEN", "piEE"]:
        h0_col = f"H0_{name}_fit"
        h1_col = f"H1_{name}_fit"

        if h0_col not in df.columns:
            df[h0_col] = [p.get(name, np.nan) for p in h0_params]

        if h1_col not in df.columns:
            df[h1_col] = [p.get(name, np.nan) for p in h1_params]

    return df


# ============================================================
# Input standardization
# ============================================================

def read_delta_chi2_from_fit_file(fit_file):
    if not isinstance(fit_file, str) or fit_file.strip() == "":
        return np.nan

    try:
        fit = pd.read_parquet(fit_file)
        if len(fit) == 0:
            return np.nan

        row = fit.iloc[0]

        for col in ["delta_chi2_true", "delta_chi2", "Delta_chi2", "LRT"]:
            if col in row.index:
                return float(row[col])

    except Exception:
        return np.nan

    return np.nan


def build_fit_true_files_from_event_dir(df):
    fit_files = []
    true_files = []

    for _, row in df.iterrows():
        if "event_dir" not in row or pd.isna(row["event_dir"]):
            fit_files.append("")
            true_files.append("")
            continue

        event_dir = Path(str(row["event_dir"]))

        if "simulation_seed" in row and pd.notna(row["simulation_seed"]):
            seed = int(row["simulation_seed"])
        else:
            seed = None

        if seed is not None:
            fit_files.append(
                str(event_dir / "fit_rr" / f"fit_rr_manual_{seed}.parquet")
            )
            true_files.append(
                str(event_dir / "true" / f"true_rr_manual_{seed}.parquet")
            )
        else:
            fit_matches = sorted((event_dir / "fit_rr").glob("fit_rr_manual_*.parquet"))
            true_matches = sorted((event_dir / "true").glob("true_rr_manual_*.parquet"))

            fit_files.append(str(fit_matches[0]) if fit_matches else "")
            true_files.append(str(true_matches[0]) if true_matches else "")

    return fit_files, true_files


def ensure_columns(df):
    """
    Standardize old and current multi_fit tables into the format expected
    by the diagnostic selection and by the old light-curve plotter.
    """

    df = df.copy()
    df = add_fit_param_columns(df)

    if "global_i" not in df.columns:
        raise KeyError("Falta global_i en la tabla de entrada.")

    df["global_i"] = pd.to_numeric(df["global_i"], errors="coerce")

    if "simulation_seed" in df.columns:
        df["simulation_seed"] = pd.to_numeric(df["simulation_seed"], errors="coerce")
    else:
        df["simulation_seed"] = np.nan

    if "delta_chi2_true" not in df.columns:
        if "LRT" in df.columns:
            df["delta_chi2_true"] = pd.to_numeric(df["LRT"], errors="coerce")
        elif "delta_chi2_H0_minus_H1" in df.columns:
            df["delta_chi2_true"] = pd.to_numeric(
                df["delta_chi2_H0_minus_H1"],
                errors="coerce",
            )
        elif "fit_file" in df.columns:
            print("[info] delta_chi2_true no está; lo leo desde fit_file...")
            df["delta_chi2_true"] = df["fit_file"].apply(read_delta_chi2_from_fit_file)
        else:
            df["delta_chi2_true"] = np.nan
    else:
        df["delta_chi2_true"] = pd.to_numeric(df["delta_chi2_true"], errors="coerce")

    if "LRT" not in df.columns:
        df["LRT"] = df["delta_chi2_true"]

    if "p_value_LRT" not in df.columns:
        df["p_value_LRT"] = np.exp(-0.5 * np.maximum(df["delta_chi2_true"], 0.0))

    if "chi2_red" not in df.columns:
        if "H0_chi2_red" in df.columns:
            df["chi2_red"] = pd.to_numeric(df["H0_chi2_red"], errors="coerce")
        elif "H0_chi2" in df.columns and "H0_dof" in df.columns:
            df["chi2_red"] = (
                pd.to_numeric(df["H0_chi2"], errors="coerce")
                / pd.to_numeric(df["H0_dof"], errors="coerce")
            )
        else:
            df["chi2_red"] = np.nan
    else:
        df["chi2_red"] = pd.to_numeric(df["chi2_red"], errors="coerce")

    if "n_peak_total" not in df.columns:
        if "n_fit_points" in df.columns:
            df["n_peak_total"] = pd.to_numeric(df["n_fit_points"], errors="coerce")
        elif "H0_n_data" in df.columns:
            df["n_peak_total"] = pd.to_numeric(df["H0_n_data"], errors="coerce")
        else:
            df["n_peak_total"] = np.nan
    else:
        df["n_peak_total"] = pd.to_numeric(df["n_peak_total"], errors="coerce")

    if "rho_true" not in df.columns:
        if "rho_catalog" in df.columns:
            df["rho_true"] = pd.to_numeric(df["rho_catalog"], errors="coerce")
        else:
            df["rho_true"] = np.nan
    else:
        df["rho_true"] = pd.to_numeric(df["rho_true"], errors="coerce")

    if "tE_true" not in df.columns:
        if "tE_catalog_days" in df.columns:
            df["tE_true"] = pd.to_numeric(df["tE_catalog_days"], errors="coerce")
        else:
            df["tE_true"] = np.nan
    else:
        df["tE_true"] = pd.to_numeric(df["tE_true"], errors="coerce")

    if "u0_true" not in df.columns:
        if "u0" in df.columns:
            df["u0_true"] = pd.to_numeric(df["u0"], errors="coerce")
        else:
            df["u0_true"] = np.nan
    else:
        df["u0_true"] = pd.to_numeric(df["u0_true"], errors="coerce")

    if "piE_true" not in df.columns:
        if "piE" in df.columns:
            df["piE_true"] = pd.to_numeric(df["piE"], errors="coerce")
        elif "piEN" in df.columns and "piEE" in df.columns:
            df["piE_true"] = np.hypot(
                pd.to_numeric(df["piEN"], errors="coerce"),
                pd.to_numeric(df["piEE"], errors="coerce"),
            )
        else:
            df["piE_true"] = np.nan
    else:
        df["piE_true"] = pd.to_numeric(df["piE_true"], errors="coerce")

    if "rho_fit" not in df.columns:
        if "H0_rho_fit" in df.columns:
            df["rho_fit"] = pd.to_numeric(df["H0_rho_fit"], errors="coerce")
        elif "rho_fit_from_best_model" in df.columns:
            df["rho_fit"] = pd.to_numeric(df["rho_fit_from_best_model"], errors="coerce")
        else:
            df["rho_fit"] = np.nan
    else:
        df["rho_fit"] = pd.to_numeric(df["rho_fit"], errors="coerce")

    if "tE_fit" not in df.columns:
        if "H0_tE_fit" in df.columns:
            df["tE_fit"] = pd.to_numeric(df["H0_tE_fit"], errors="coerce")
        elif "tE_fit_from_best_model" in df.columns:
            df["tE_fit"] = pd.to_numeric(df["tE_fit_from_best_model"], errors="coerce")
        else:
            df["tE_fit"] = np.nan
    else:
        df["tE_fit"] = pd.to_numeric(df["tE_fit"], errors="coerce")

    if "u0_fit" not in df.columns:
        if "H0_u0_fit" in df.columns:
            df["u0_fit"] = pd.to_numeric(df["H0_u0_fit"], errors="coerce")
        elif "u0_fit_from_best_model" in df.columns:
            df["u0_fit"] = pd.to_numeric(df["u0_fit_from_best_model"], errors="coerce")
        else:
            df["u0_fit"] = np.nan
    else:
        df["u0_fit"] = pd.to_numeric(df["u0_fit"], errors="coerce")

    if "fit_file" not in df.columns or "true_file" not in df.columns:
        fit_files, true_files = build_fit_true_files_from_event_dir(df)

        if "fit_file" not in df.columns:
            df["fit_file"] = fit_files

        if "true_file" not in df.columns:
            df["true_file"] = true_files

    if "sigma_rho_over_rho" not in df.columns:
        if "sigma_rho_over_rho_fit" in df.columns:
            df["sigma_rho_over_rho"] = pd.to_numeric(
                df["sigma_rho_over_rho_fit"],
                errors="coerce",
            )
        else:
            df["sigma_rho_over_rho"] = np.nan
    else:
        df["sigma_rho_over_rho"] = pd.to_numeric(
            df["sigma_rho_over_rho"],
            errors="coerce",
        )

    if "bad_boundary_fit" not in df.columns:
        df["at_u0_bound"] = np.isclose(
            np.abs(df["u0_fit"]),
            5.0,
            rtol=0,
            atol=1e-2,
        )

        df["at_tE_upper_bound"] = np.isclose(
            df["tE_fit"],
            20000.0,
            rtol=0,
            atol=1e-2,
        )

        df["bad_boundary_fit"] = (
            df["at_u0_bound"]
            | df["at_tE_upper_bound"]
        )
    else:
        df["bad_boundary_fit"] = normalize_bool_column(df["bad_boundary_fit"])

    df["rho_ratio"] = df["rho_fit"] / df["rho_true"]
    df["rho_rel_error"] = (df["rho_fit"] - df["rho_true"]) / df["rho_true"]
    df["abs_rho_rel_error"] = np.abs(df["rho_rel_error"])
    df["tE_ratio"] = df["tE_fit"] / df["tE_true"]

    return df


# ============================================================
# Selection
# ============================================================

def select_delta_chi2_window(
    df,
    delta_chi2_min=0.0,
    delta_chi2_max=12.0,
    chi2_red_max=1.5,
    min_peak=5,
    require_positive_delta=True,
    use_quality_flag=False,
):
    selected = df.copy()

    mask = np.ones(len(selected), dtype=bool)

    if use_quality_flag and "quality_flag" in selected.columns:
        mask &= selected["quality_flag"].astype(str).eq("ok").to_numpy()

    mask &= np.isfinite(selected["chi2_red"])
    mask &= selected["chi2_red"] < chi2_red_max

    mask &= ~selected["bad_boundary_fit"].fillna(False).astype(bool).to_numpy()

    mask &= np.isfinite(selected["n_peak_total"])
    mask &= selected["n_peak_total"] >= min_peak

    mask &= np.isfinite(selected["delta_chi2_true"])

    if require_positive_delta:
        mask &= selected["delta_chi2_true"] >= delta_chi2_min
    else:
        mask &= selected["delta_chi2_true"] > -np.inf

    mask &= selected["delta_chi2_true"] < delta_chi2_max

    selected["is_delta_chi2_confused"] = mask

    return selected, selected[mask].copy()


def choose_representatives(
    selected,
    n_events=6,
    delta_chi2_min=0.0,
    delta_chi2_max=12.0,
    visual_chi2_red_max=1.2,
    visual_min_peak=10,
    visual_tE_ratio_min=0.3,
    visual_tE_ratio_max=3.0,
    use_visual_filter=True,
):
    if len(selected) == 0:
        return selected.copy()

    df = selected.copy()

    df = df[np.isfinite(df["delta_chi2_true"])].copy()
    df = df[df["delta_chi2_true"] >= delta_chi2_min].copy()
    df = df[df["delta_chi2_true"] < delta_chi2_max].copy()

    if len(df) == 0:
        return df.copy()

    if use_visual_filter:
        good_visual = np.ones(len(df), dtype=bool)

        good_visual &= np.isfinite(df["chi2_red"])
        good_visual &= df["chi2_red"] < visual_chi2_red_max

        good_visual &= np.isfinite(df["n_peak_total"])
        good_visual &= df["n_peak_total"] >= visual_min_peak

        if "tE_ratio" in df.columns:
            finite_ratio = np.isfinite(df["tE_ratio"])
            good_visual &= finite_ratio
            good_visual &= df["tE_ratio"] > visual_tE_ratio_min
            good_visual &= df["tE_ratio"] < visual_tE_ratio_max

        if good_visual.sum() >= min(n_events, len(df)):
            df = df[good_visual].copy()
        else:
            print(
                "[warning] Hay pocos eventos con filtro visual estricto; "
                "uso el conjunto completo seleccionado."
            )

    rows = []

    def add_one(reason, table, sort_cols, ascending=True):
        if len(table) == 0:
            return

        if isinstance(sort_cols, str):
            sort_cols = [sort_cols]

        used = set()
        if rows:
            used = set(pd.concat(rows)["global_i"].astype(int))

        tmp = table[~table["global_i"].astype(int).isin(used)].copy()

        if len(tmp) == 0:
            return

        tmp = tmp.sort_values(sort_cols, ascending=ascending)
        row = tmp.head(1).copy()
        row["selection_reason"] = reason
        rows.append(row)

    add_one(
        "smallest_positive_delta_chi2",
        df,
        ["delta_chi2_true", "chi2_red"],
        ascending=[True, True],
    )

    add_one(
        "best_chi2_red",
        df,
        ["chi2_red", "delta_chi2_true"],
        ascending=[True, True],
    )

    add_one(
        "well_sampled",
        df,
        ["n_peak_total", "delta_chi2_true"],
        ascending=[False, True],
    )

    add_one(
        "large_piE_hidden",
        df[np.isfinite(df["piE_true"])],
        ["piE_true", "delta_chi2_true"],
        ascending=[False, True],
    )

    add_one(
        "rho_approximately_recovered",
        df[np.isfinite(df["abs_rho_rel_error"])],
        ["abs_rho_rel_error", "delta_chi2_true"],
        ascending=[True, True],
    )

    add_one(
        "small_delta_large_peak",
        df,
        ["delta_chi2_true", "n_peak_total"],
        ascending=[True, False],
    )

    if rows:
        reps = pd.concat(rows, ignore_index=True)
    else:
        reps = pd.DataFrame()

    if len(reps) < n_events:
        used = set(reps["global_i"].astype(int)) if len(reps) else set()

        fill = df[~df["global_i"].astype(int).isin(used)].copy()

        fill = fill.sort_values(
            ["delta_chi2_true", "chi2_red", "n_peak_total"],
            ascending=[True, True, False],
        )

        fill = fill.head(n_events - len(reps)).copy()
        fill["selection_reason"] = "additional_delta_chi2_window"

        reps = pd.concat([reps, fill], ignore_index=True)

    return reps.head(n_events).copy()


# ============================================================
# Diagnostic plots
# ============================================================

def savefig(fig, path, dpi=250):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_delta_chi2_distribution(
    df,
    selected,
    outdir,
    delta_chi2_min=0.0,
    delta_chi2_max=12.0,
):
    fig, ax = plt.subplots(figsize=(6.8, 5.0))

    values = pd.to_numeric(df["delta_chi2_true"], errors="coerce").to_numpy(float)
    finite = values[np.isfinite(values)]

    negative = finite[finite < 0]
    nonnegative = finite[finite >= 0]

    if len(nonnegative) > 0:
        vmax = np.nanpercentile(nonnegative, 99)
        values_plot = nonnegative[nonnegative <= vmax]

        ax.hist(
            values_plot,
            bins=60,
            histtype="step",
            linewidth=1.5,
            label=r"Events with $\Delta\chi^2 \geq 0$",
        )

    selected_values = pd.to_numeric(
        selected["delta_chi2_true"],
        errors="coerce",
    ).to_numpy(float)

    selected_values = selected_values[np.isfinite(selected_values)]
    selected_values = selected_values[
        (selected_values >= delta_chi2_min)
        & (selected_values < delta_chi2_max)
    ]

    if len(selected_values) > 0:
        ax.hist(
            selected_values,
            bins=30,
            histtype="stepfilled",
            alpha=0.35,
            label=rf"Selected: ${delta_chi2_min:g} \leq \Delta\chi^2 < {delta_chi2_max:g}$",
        )

    ax.axvline(delta_chi2_min, linestyle=":", linewidth=1.2)
    ax.axvline(delta_chi2_max, linestyle="--", linewidth=1.2)

    ax.set_xlabel(r"$\Delta\chi^2$")
    ax.set_ylabel("Number of events")
    ax.set_title("Delta chi2 / LRT selection")

    text = (
        rf"$N_{{\rm total}}={len(finite)}$" "\n"
        rf"$N(\Delta\chi^2<0)={len(negative)}$" "\n"
        rf"$N_{{\rm selected}}={len(selected)}$"
    )

    ax.text(
        0.98,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    savefig(fig, outdir / "delta_chi2_distribution.png")
    savefig(fig, outdir / "delta_chi2_distribution.pdf")


def plot_chi2_vs_delta(
    df,
    selected,
    outdir,
    delta_chi2_min=0.0,
    delta_chi2_max=12.0,
    chi2_red_max=1.5,
):
    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    ax.scatter(
        df["delta_chi2_true"],
        df["chi2_red"],
        s=8,
        alpha=0.18,
        label="All candidates",
    )

    ax.scatter(
        selected["delta_chi2_true"],
        selected["chi2_red"],
        s=18,
        alpha=0.8,
        label="Selected",
    )

    ax.axvline(delta_chi2_min, linestyle=":", linewidth=1.1)
    ax.axvline(delta_chi2_max, linestyle="--", linewidth=1.1)
    ax.axhline(chi2_red_max, linestyle="--", linewidth=1.1)

    ax.set_yscale("log")

    ax.set_xlabel(r"$\Delta\chi^2$")
    ax.set_ylabel(r"$\chi^2/{\rm dof}$")
    ax.set_title("Fit quality vs Delta chi2 / LRT")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    savefig(fig, outdir / "chi2red_vs_delta_chi2.png")
    savefig(fig, outdir / "chi2red_vs_delta_chi2.pdf")


def plot_rho_recovery(df, selected, outdir):
    good = (
        np.isfinite(df["rho_true"])
        & np.isfinite(df["rho_fit"])
        & (df["rho_true"] > 0)
        & (df["rho_fit"] > 0)
    )

    sub = df[good].copy()

    good_sel = (
        np.isfinite(selected["rho_true"])
        & np.isfinite(selected["rho_fit"])
        & (selected["rho_true"] > 0)
        & (selected["rho_fit"] > 0)
    )

    sel = selected[good_sel].copy()

    if len(sub) == 0:
        print("[warning] No hay datos válidos para plot_rho_recovery.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    lo = np.nanmin([sub["rho_true"].min(), sub["rho_fit"].min()])
    hi = np.nanmax([sub["rho_true"].max(), sub["rho_fit"].max()])

    ax.scatter(
        sub["rho_true"],
        sub["rho_fit"],
        s=8,
        alpha=0.18,
        label="All candidates",
    )

    if len(sel) > 0:
        ax.scatter(
            sel["rho_true"],
            sel["rho_fit"],
            s=18,
            alpha=0.8,
            label="Selected",
        )

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Injected $\rho_{\rm true}$")
    ax.set_ylabel(r"Recovered $\rho_{\rm fit}$")
    ax.set_title(r"$\rho$ recovery")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    savefig(fig, outdir / "rho_fit_vs_rho_true.png")
    savefig(fig, outdir / "rho_fit_vs_rho_true.pdf")


def plot_piE_tE(selected, outdir):
    good = (
        np.isfinite(selected["piE_true"])
        & np.isfinite(selected["tE_true"])
        & (selected["piE_true"] > 0)
        & (selected["tE_true"] > 0)
    )

    sub = selected[good].copy()

    if len(sub) == 0:
        print("[warning] No hay piE/tE para plot_piE_tE.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.3))

    sc = ax.scatter(
        sub["tE_true"],
        sub["piE_true"],
        c=sub["delta_chi2_true"],
        s=22,
        alpha=0.85,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$t_E$ [days]")
    ax.set_ylabel(r"Injected $\pi_E$")
    ax.set_title("Selected events")

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\Delta\chi^2$")

    ax.grid(True, alpha=0.3)

    savefig(fig, outdir / "piE_vs_tE_selected.png")
    savefig(fig, outdir / "piE_vs_tE_selected.pdf")


def plot_selection_counts(df, selected, outdir, delta_chi2_max=12.0):
    n_total = len(df)

    finite_delta = np.isfinite(df["delta_chi2_true"])

    n_negative = int((finite_delta & (df["delta_chi2_true"] < 0)).sum())
    n_nonnegative = int((finite_delta & (df["delta_chi2_true"] >= 0)).sum())
    n_selected = len(selected)

    labels = [
        "Input",
        r"$\Delta\chi^2<0$",
        r"$\Delta\chi^2\geq0$",
        rf"Selected",
    ]

    counts = [
        n_total,
        n_negative,
        n_nonnegative,
        n_selected,
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.bar(labels, counts)

    ax.set_ylabel("Number of events")
    ax.set_title("Selection counts")
    ax.grid(True, axis="y", alpha=0.3)

    for i, count in enumerate(counts):
        ax.text(
            i,
            count,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    savefig(fig, outdir / "selection_counts.png")
    savefig(fig, outdir / "selection_counts.pdf")


def make_plots(
    df,
    selected,
    outdir,
    delta_chi2_min=0.0,
    delta_chi2_max=12.0,
    chi2_red_max=1.5,
):
    outdir.mkdir(parents=True, exist_ok=True)

    plot_delta_chi2_distribution(
        df,
        selected,
        outdir,
        delta_chi2_min=delta_chi2_min,
        delta_chi2_max=delta_chi2_max,
    )

    plot_chi2_vs_delta(
        df,
        selected,
        outdir,
        delta_chi2_min=delta_chi2_min,
        delta_chi2_max=delta_chi2_max,
        chi2_red_max=chi2_red_max,
    )

    plot_rho_recovery(df, selected, outdir)
    plot_piE_tE(selected, outdir)
    plot_selection_counts(df, selected, outdir, delta_chi2_max=delta_chi2_max)


# ============================================================
# Annotation for produced light-curve PNGs
# ============================================================

def format_pvalue_for_box(p):
    p = as_float(p)

    if not np.isfinite(p):
        return "nan"

    if p == 0:
        return "<1e-300"

    if p < 1e-3:
        return f"{p:.2e}"

    return f"{p:.3f}"


def annotation_text(row):
    true_t0 = as_float(row.get("t0_jd"))
    true_u0 = as_float(row.get("u0", row.get("u0_true")))
    true_tE = as_float(row.get("tE_catalog_days", row.get("tE_true")))
    true_rho = as_float(row.get("rho_catalog", row.get("rho_true")))
    true_piEN = as_float(row.get("piEN"))
    true_piEE = as_float(row.get("piEE"))
    true_piE = as_float(row.get("piE", row.get("piE_true")))

    if not np.isfinite(true_piE):
        true_piE = np.hypot(true_piEN, true_piEE)

    h0_t0 = as_float(row.get("H0_t0_fit"))
    h0_u0 = as_float(row.get("H0_u0_fit"))
    h0_tE = as_float(row.get("H0_tE_fit"))
    h0_rho = as_float(row.get("H0_rho_fit"))

    h1_t0 = as_float(row.get("H1_t0_fit"))
    h1_u0 = as_float(row.get("H1_u0_fit"))
    h1_tE = as_float(row.get("H1_tE_fit"))
    h1_rho = as_float(row.get("H1_rho_fit"))
    h1_piEN = as_float(row.get("H1_piEN_fit"))
    h1_piEE = as_float(row.get("H1_piEE_fit"))
    h1_piE = np.hypot(h1_piEN, h1_piEE)

    chi2_h0 = as_float(row.get("H0_chi2"))
    chi2_h1 = as_float(row.get("H1_chi2"))
    chi2_true = as_float(row.get("true_generator_chi2"))
    lrt = as_float(row.get("LRT", row.get("delta_chi2_true")))
    pval = row.get("p_value_LRT", np.nan)

    h1_minus_true = as_float(row.get("H1_minus_true_chi2"))

    if not np.isfinite(h1_minus_true):
        h1_minus_true = chi2_h1 - chi2_true

    category = row.get("lightcurve_category", "")
    reason = row.get("selection_reason", "")
    quality = row.get("quality_flag", "")

    lines = [
        "TRUE FSPL + parallax",
        f"t0={true_t0:.2f}, u0={true_u0:.3g}, tE={true_tE:.3g} d",
        f"rho={true_rho:.3g}, piEN={true_piEN:.3g}, piEE={true_piEE:.3g}, piE={true_piE:.3g}",
        "",
        "H0 fit: FSPL no parallax",
        f"t0={h0_t0:.2f}, u0={h0_u0:.3g}, tE={h0_tE:.3g} d, rho={h0_rho:.3g}",
        "",
        "H1 fit: FSPL + parallax",
        f"t0={h1_t0:.2f}, u0={h1_u0:.3g}, tE={h1_tE:.3g} d, rho={h1_rho:.3g}",
        f"piEN={h1_piEN:.3g}, piEE={h1_piEE:.3g}, piE={h1_piE:.3g}",
        "",
        "Hypothesis test",
        f"chi2_H0={chi2_h0:.2f}, chi2_H1={chi2_h1:.2f}, chi2_true={chi2_true:.2f}",
        f"Delta chi2={lrt:.2f}, p_LRT={format_pvalue_for_box(pval)}",
        f"chi2_H1 - chi2_true={h1_minus_true:.2f}",
    ]

    if str(category) not in ["", "nan", "None"]:
        lines.append(f"category={category}")

    if str(reason) not in ["", "nan", "None"]:
        lines.append(f"selection={reason}")

    if str(quality) not in ["", "nan", "None"]:
        lines.append(f"quality={quality}")

    return "\n".join(lines)


def annotate_png(input_png, output_png, row):
    """
    Rehace la figura agregando la caja de anotación FUERA de la imagen original.

    En vez de escribir encima del panel principal, crea un canvas más ancho:
        [figura original] [caja externa grande]

    Esto evita tapar la curva de luz, el inset y la leyenda.
    """

    input_png = Path(input_png)
    output_png = Path(output_png)

    img = mpimg.imread(input_png)
    h_px, w_px = img.shape[:2]

    dpi = 140

    # Tamaño original en pulgadas
    w_in = w_px / dpi
    h_in = h_px / dpi

    # Panel extra a la derecha para la anotación
    extra_w_in = 4.6

    fig = plt.figure(
        figsize=(w_in + extra_w_in, h_in),
        dpi=dpi,
    )

    # Fracción del canvas ocupada por la imagen original
    img_frac = w_in / (w_in + extra_w_in)

    ax_img = fig.add_axes(
        [0.0, 0.0, img_frac, 1.0]
    )

    ax_img.imshow(img)
    ax_img.axis("off")

    ax_box = fig.add_axes(
        [img_frac + 0.015, 0.05, 1.0 - img_frac - 0.035, 0.90]
    )

    ax_box.axis("off")

    txt = annotation_text(row)

    ax_box.text(
        0.0,
        1.0,
        txt,
        transform=ax_box.transAxes,
        ha="left",
        va="top",
        fontsize=13.0,
        family="monospace",
        linespacing=1.25,
        bbox=dict(
            boxstyle="round,pad=0.55",
            fc="white",
            ec="0.25",
            alpha=0.97,
        ),
    )

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_png,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)

def run_lightcurve_pipeline_one_by_one(
    representatives,
    output_dir,
    plot_script,
    pipeline,
    runner,
    config,
    reference_band,
    plot_n_tE,
    plot_inset_n_tE,
    fit_trajectory_time_mode,
    keep_temp_csv=False,
    annotate=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logs = []

    for _, row in representatives.iterrows():
        row_dict = row.to_dict()

        global_i = int(row_dict["global_i"])

        if "simulation_seed" in row_dict and np.isfinite(as_float(row_dict["simulation_seed"])):
            seed = int(row_dict["simulation_seed"])
        else:
            seed = global_i

        category = safe_name(row_dict.get("lightcurve_category", "selected"))
        reason = safe_name(row_dict.get("selection_reason", "representative"))

        event_outdir = (
            output_dir
            / f"{category}_{reason}_global_{global_i}_seed_{seed}"
        )

        event_outdir.mkdir(parents=True, exist_ok=True)

        tmp_csv = event_outdir / f"event_global_{global_i}_for_plotter.csv"

        pd.DataFrame([row_dict]).to_csv(tmp_csv, index=False)

        cmd = [
            sys.executable,
            str(plot_script),

            "--pipeline",
            str(pipeline),

            "--runner",
            str(runner),

            "--events-csv",
            str(tmp_csv),

            "--n-events",
            "1",

            "--reference-band",
            str(reference_band),

            "--plot-n-tE",
            str(plot_n_tE),

            "--plot-inset-n-tE",
            str(plot_inset_n_tE),

            "--fit-trajectory-time-mode",
            str(fit_trajectory_time_mode),

            "--output-dir",
            str(event_outdir),

            "--config",
            str(config),
        ]

        print("\nRunning light-curve pipeline:")
        print(" ".join(map(str, cmd)))

        try:
            subprocess.run(cmd, check=True)

            base_pngs = sorted(
                p for p in event_outdir.rglob("*.png")
                if not p.name.endswith("_annotated.png")
            )

            annotated_pngs = []

            if annotate:
                for png in base_pngs:
                    out_png = png.with_name(png.stem + "_annotated.png")

                    annotate_png(
                        input_png=png,
                        output_png=out_png,
                        row=row_dict,
                    )

                    annotated_pngs.append(out_png)

            if not keep_temp_csv:
                try:
                    tmp_csv.unlink()
                except Exception:
                    pass

            logs.append(
                {
                    "global_i": global_i,
                    "simulation_seed": seed,
                    "event_outdir": str(event_outdir),
                    "n_base_pngs": len(base_pngs),
                    "n_annotated_pngs": len(annotated_pngs),
                    "base_pngs": ";".join(map(str, base_pngs)),
                    "annotated_pngs": ";".join(map(str, annotated_pngs)),
                    "status": "ok",
                }
            )

            print(
                "[ok]",
                global_i,
                "base_pngs =",
                len(base_pngs),
                "annotated_pngs =",
                len(annotated_pngs),
            )

        except Exception as e:
            logs.append(
                {
                    "global_i": global_i,
                    "simulation_seed": seed,
                    "event_outdir": str(event_outdir),
                    "n_base_pngs": 0,
                    "n_annotated_pngs": 0,
                    "base_pngs": "",
                    "annotated_pngs": "",
                    "status": repr(e),
                }
            )

            print("[failed]", global_i, repr(e))

    log = pd.DataFrame(logs)

    log_path = output_dir / "lightcurve_pipeline_log.csv"
    log.to_csv(log_path, index=False)

    return log_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--analysis-root", default=str(DEFAULT_ANALYSIS_ROOT))

    parser.add_argument(
        "--input-csv",
        default=None,
        help=(
            "CSV de entrada. Si no se pasa, busca el último "
            "multi_fit_partial_with_quality_flags.csv o selected_lightcurve_events.csv."
        ),
    )

    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Directorio base de salida estilo viejo. "
            "Si no se pasa, usa la carpeta del análisis parcial."
        ),
    )

    parser.add_argument("--plot-script", default=str(DEFAULT_PLOT_SCRIPT))
    parser.add_argument("--pipeline", default=str(DEFAULT_PIPELINE))
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))

    parser.add_argument("--output-subdir", default="confused_delta_chi2_0_12_strict")

    parser.add_argument("--delta-chi2-min", type=float, default=0.0)
    parser.add_argument("--delta-chi2-max", type=float, default=12.0)

    parser.add_argument("--chi2-red-max", type=float, default=1.5)
    parser.add_argument("--min-peak", type=int, default=5)
    parser.add_argument("--n-events", type=int, default=6)

    parser.add_argument("--reference-band", default="r")
    parser.add_argument("--plot-n-tE", type=float, default=10.0)
    parser.add_argument("--plot-inset-n-tE", type=float, default=4.0)

    parser.add_argument(
        "--fit-trajectory-time-mode",
        default="same_true_window",
        choices=["same_true_window", "own_fit_tE_window"],
    )

    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--skip-lightcurves", action="store_true")
    parser.add_argument("--no-annotate-lightcurves", action="store_true")
    parser.add_argument("--keep-temp-csv", action="store_true")

    parser.add_argument("--use-quality-flag", action="store_true")
    parser.add_argument("--no-visual-filter", action="store_true")
    parser.add_argument("--visual-chi2-red-max", type=float, default=1.2)
    parser.add_argument("--visual-min-peak", type=int, default=10)
    parser.add_argument("--visual-tE-ratio-min", type=float, default=0.3)
    parser.add_argument("--visual-tE-ratio-max", type=float, default=3.0)

    args = parser.parse_args()

    plot_script = resolve_existing_path(args.plot_script, "plot_script")
    pipeline = resolve_existing_path(args.pipeline, "pipeline")
    runner = resolve_existing_path(args.runner, "runner")
    config = resolve_existing_path(args.config, "config")

    if args.input_csv is None:
        input_csv = find_default_input_csv(
            run_tag=args.run_tag,
            analysis_root=args.analysis_root,
        )
    else:
        input_csv = Path(args.input_csv)

    input_csv = resolve_existing_path(input_csv, "input_csv")

    analysis_dir = input_csv.parent

    if args.run_dir is None:
        run_dir = analysis_dir
    else:
        run_dir = Path(args.run_dir)

    outdir = run_dir / args.output_subdir
    diagnostics_dir = outdir / "diagnostics"
    lightcurves_dir = outdir / "lightcurves_with_inset_pipeline"

    outdir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    lightcurves_dir.mkdir(parents=True, exist_ok=True)

    print("Input CSV:     ", input_csv)
    print("Output dir:    ", outdir)
    print("Plot script:   ", plot_script)
    print("Pipeline:      ", pipeline)
    print("Runner:        ", runner)
    print("Config:        ", config)

    df = pd.read_csv(input_csv)
    df = ensure_columns(df)

    all_with_flags, selected = select_delta_chi2_window(
        df,
        delta_chi2_min=args.delta_chi2_min,
        delta_chi2_max=args.delta_chi2_max,
        chi2_red_max=args.chi2_red_max,
        min_peak=args.min_peak,
        require_positive_delta=True,
        use_quality_flag=args.use_quality_flag,
    )

    n_negative = int(
        (
            np.isfinite(all_with_flags["delta_chi2_true"])
            & (all_with_flags["delta_chi2_true"] < 0)
        ).sum()
    )

    print("\nSelection counts")
    print("----------------")
    print("Input events:", len(df))
    print("Negative delta chi2 excluded:", n_negative)
    print(
        f"{args.delta_chi2_min:g} <= Delta chi2 < {args.delta_chi2_max:g} selected:",
        len(selected),
    )

    all_out = outdir / "all_candidates_with_delta_chi2_flag.csv"
    sel_out = outdir / "selected_delta_chi2_events.csv"

    all_with_flags.to_csv(all_out, index=False)
    selected.to_csv(sel_out, index=False)

    representatives = choose_representatives(
        selected,
        n_events=args.n_events,
        delta_chi2_min=args.delta_chi2_min,
        delta_chi2_max=args.delta_chi2_max,
        visual_chi2_red_max=args.visual_chi2_red_max,
        visual_min_peak=args.visual_min_peak,
        visual_tE_ratio_min=args.visual_tE_ratio_min,
        visual_tE_ratio_max=args.visual_tE_ratio_max,
        use_visual_filter=(not args.no_visual_filter),
    )

    rep_out = outdir / "representative_delta_chi2_events.csv"
    representatives.to_csv(rep_out, index=False)

    print("\nSaved tables")
    print("------------")
    print(all_out)
    print(sel_out)
    print(rep_out)

    if len(representatives) > 0:
        print("\nRepresentative events")
        print("---------------------")

        show_cols = [
            "selection_reason",
            "lightcurve_category",
            "global_i",
            "simulation_seed",
            "delta_chi2_true",
            "p_value_LRT",
            "chi2_red",
            "n_peak_total",
            "rho_true",
            "rho_fit",
            "rho_rel_error",
            "sigma_rho_over_rho",
            "tE_true",
            "tE_fit",
            "tE_ratio",
            "piE_true",
            "quality_flag",
        ]

        show_cols = [c for c in show_cols if c in representatives.columns]
        print(representatives[show_cols].to_string(index=False))

    if not args.skip_diagnostics:
        make_plots(
            all_with_flags,
            selected,
            diagnostics_dir,
            delta_chi2_min=args.delta_chi2_min,
            delta_chi2_max=args.delta_chi2_max,
            chi2_red_max=args.chi2_red_max,
        )

        print("\nDiagnostic figures written to:")
        print(diagnostics_dir)

    if not args.skip_lightcurves:
        if len(representatives) == 0:
            print("[warning] No selected representatives; skipping light-curve plots.")
        else:
            log_path = run_lightcurve_pipeline_one_by_one(
                representatives=representatives,
                output_dir=lightcurves_dir,
                plot_script=plot_script,
                pipeline=pipeline,
                runner=runner,
                config=config,
                reference_band=args.reference_band,
                plot_n_tE=args.plot_n_tE,
                plot_inset_n_tE=args.plot_inset_n_tE,
                fit_trajectory_time_mode=args.fit_trajectory_time_mode,
                keep_temp_csv=args.keep_temp_csv,
                annotate=(not args.no_annotate_lightcurves),
            )

            print("\nLight-curve plots written to:")
            print(lightcurves_dir)

            print("\nLight-curve log:")
            print(log_path)

            print("\nAnnotated PNGs:")
            log = pd.read_csv(log_path)
            if "annotated_pngs" in log.columns:
                for item in log["annotated_pngs"].dropna().astype(str):
                    for path in item.split(";"):
                        if path.strip():
                            print(path)


if __name__ == "__main__":
    main()