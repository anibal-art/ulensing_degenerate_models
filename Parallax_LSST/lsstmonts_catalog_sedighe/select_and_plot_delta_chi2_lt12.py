#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select and plot events confused with FSPL no-parallax using:

    0 <= Delta chi2 < 12

This is intended as a conservative "not clearly distinguishable" criterion
for the no-parallax FSPL fit relative to the injected FSPL+parallax model.

Output directory by default:
    runs/<run_name>/figures/confused_delta_chi2_0_12_strict/

It creates:
    diagnostics/
    all_candidates_with_delta_chi2_flag.csv
    selected_delta_chi2_events.csv
    representative_delta_chi2_events.csv

and optionally calls:
    plot_confused_lightcurves_with_inset_pipeline.py
"""

from pathlib import Path
import argparse
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Defaults
# ============================================================

DEFAULT_RUN_DIR = Path(
    "/home/anibal/ulensing_degenerate_models/Parallax_LSST/runs/"
    "LSSTMONTS_xi_baseline_v5p3p5_hiddenParallax_FSPLparallax_fitFSPLNoPiE_t0pm60_DetectionFlag"
)

DEFAULT_CANDIDATES = (
    DEFAULT_RUN_DIR
    / "figures"
    / "confused_fspl_noparallax"
    / "confused_candidates_from_fitrr_truerr.csv"
)

DEFAULT_PLOT_SCRIPT = Path(
    "/home/anibal/ulensing_degenerate_models/Parallax_LSST/"
    "lsstmonts_catalog_sedighe/plot_confused_lightcurves_with_inset_pipeline.py"
)

DEFAULT_CONFIG = Path(
    "/home/anibal/ulensing_degenerate_models/Parallax_LSST/"
    "lsstmonts_catalog_sedighe/config_lsstmonts_baseline_v5p3p5.json"
)


# ============================================================
# Helpers
# ============================================================

def read_delta_chi2_from_fit_file(fit_file):
    """
    Read delta_chi2_true from fit_rr parquet if it is not already in the CSV.
    """
    if not isinstance(fit_file, str) or fit_file.strip() == "":
        return np.nan

    try:
        fit = pd.read_parquet(fit_file)
        if len(fit) == 0:
            return np.nan

        row = fit.iloc[0]

        for col in ["delta_chi2_true", "delta_chi2", "Delta_chi2"]:
            if col in row.index:
                return float(row[col])

    except Exception:
        return np.nan

    return np.nan


def normalize_bool_column(series):
    """
    Convert booleans or string booleans to actual bools.
    Unknown / NaN values are treated as False.
    """
    if series.dtype == bool:
        return series.fillna(False)

    s = series.astype(str).str.strip().str.lower()

    return s.isin(["true", "1", "yes", "y", "t"])


def ensure_columns(df):
    """
    Validate and standardize the candidates table.
    """
    df = df.copy()

    required = [
        "global_i",
        "chi2_red",
        "n_peak_total",
        "rho_true",
        "rho_fit",
        "tE_true",
        "tE_fit",
        "u0_true",
        "u0_fit",
        "fit_file",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas en candidates CSV: {missing}")

    numeric_cols = [
        "global_i",
        "chi2_red",
        "n_peak_total",
        "rho_true",
        "rho_fit",
        "tE_true",
        "tE_fit",
        "u0_true",
        "u0_fit",
        "piE_true",
        "sigma_rho_over_rho",
        "delta_chi2_true",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "delta_chi2_true" not in df.columns:
        print("[info] delta_chi2_true no está en la tabla; lo leo desde fit_file...")
        df["delta_chi2_true"] = df["fit_file"].apply(read_delta_chi2_from_fit_file)
    else:
        df["delta_chi2_true"] = pd.to_numeric(df["delta_chi2_true"], errors="coerce")

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

        df["bad_boundary_fit"] = df["at_u0_bound"] | df["at_tE_upper_bound"]
    else:
        df["bad_boundary_fit"] = normalize_bool_column(df["bad_boundary_fit"])

    if "piE_true" not in df.columns:
        df["piE_true"] = np.nan

    if "sigma_rho_over_rho" not in df.columns:
        df["sigma_rho_over_rho"] = np.nan

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
):
    """
    Main statistical selection.

    Important correction:
        require 0 <= delta_chi2_true < delta_chi2_max

    Negative delta_chi2_true values are excluded because they do not have
    the likelihood-ratio interpretation we want here.
    """
    selected = df.copy()

    mask = np.ones(len(selected), dtype=bool)

    mask &= np.isfinite(selected["chi2_red"])
    mask &= selected["chi2_red"] < chi2_red_max

    mask &= ~selected["bad_boundary_fit"].fillna(False).astype(bool)

    mask &= np.isfinite(selected["n_peak_total"])
    mask &= selected["n_peak_total"] >= min_peak

    mask &= np.isfinite(selected["delta_chi2_true"])
    mask &= selected["delta_chi2_true"] >= delta_chi2_min
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
    """
    Choose representative events for plotting.

    This function applies an additional "visual cleanliness" filter so that
    individual plotted examples do not look obviously bad, even if they pass
    the broad statistical cut.
    """
    if len(selected) == 0:
        return selected.copy()

    df = selected.copy()

    # Enforce the corrected Delta chi2 window again.
    df = df[np.isfinite(df["delta_chi2_true"])].copy()
    df = df[df["delta_chi2_true"] >= delta_chi2_min].copy()
    df = df[df["delta_chi2_true"] < delta_chi2_max].copy()

    if len(df) == 0:
        return df.copy()

    # Optional stricter visual filter for clean inset figures.
    if use_visual_filter:
        good_visual = np.ones(len(df), dtype=bool)

        good_visual &= np.isfinite(df["chi2_red"])
        good_visual &= df["chi2_red"] < visual_chi2_red_max

        good_visual &= np.isfinite(df["n_peak_total"])
        good_visual &= df["n_peak_total"] >= visual_min_peak

        good_visual &= np.isfinite(df["tE_ratio"])
        good_visual &= df["tE_ratio"] > visual_tE_ratio_min
        good_visual &= df["tE_ratio"] < visual_tE_ratio_max

        if good_visual.sum() >= min(n_events, len(df)):
            df = df[good_visual].copy()
        else:
            print(
                "[warning] Hay pocos eventos con el filtro visual estricto; "
                "uso el conjunto completo con 0 <= Delta chi2 < threshold."
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
        "best_chi2",
        df,
        ["chi2_red", "delta_chi2_true"],
        ascending=[True, True],
    )

    add_one(
        "well_sampled_peak",
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
# Plots
# ============================================================

def plot_delta_chi2_distribution(
    df,
    selected,
    outdir,
    delta_chi2_min=0.0,
    delta_chi2_max=12.0,
):
    fig, ax = plt.subplots(figsize=(6.8, 5.0))

    values = df["delta_chi2_true"].to_numpy(float)
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
            label=r"Candidates with $\Delta\chi^2 \geq 0$",
        )

    selected_values = selected["delta_chi2_true"].to_numpy(float)
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
            label=rf"Selected: $0 \leq \Delta\chi^2 < {delta_chi2_max:g}$",
        )

    ax.axvline(0.0, linestyle=":", linewidth=1.2)
    ax.axvline(delta_chi2_max, linestyle="--", linewidth=1.2)

    ax.set_xlabel(r"$\Delta\chi^2$")
    ax.set_ylabel("Number of events")
    ax.set_title(r"FSPL no-parallax confusion criterion")

    text = (
        rf"$N_{{\rm total}}={len(finite)}$" "\n"
        rf"$N(\Delta\chi^2<0)={len(negative)}$" "\n"
        rf"$N(0\leq\Delta\chi^2<{delta_chi2_max:g})={len(selected)}$"
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

    fig.tight_layout()
    fig.savefig(outdir / "delta_chi2_distribution_0_12.pdf")
    fig.savefig(outdir / "delta_chi2_distribution_0_12.png", dpi=250)
    plt.close(fig)


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
        label="Base candidates",
    )

    ax.scatter(
        selected["delta_chi2_true"],
        selected["chi2_red"],
        s=18,
        alpha=0.8,
        label=rf"Selected: $0 \leq \Delta\chi^2 < {delta_chi2_max:g}$",
    )

    ax.axvline(delta_chi2_min, linestyle=":", linewidth=1.1)
    ax.axvline(delta_chi2_max, linestyle="--", linewidth=1.1)
    ax.axhline(chi2_red_max, linestyle="--", linewidth=1.1)

    ax.set_yscale("log")

    ax.set_xlabel(r"$\Delta\chi^2$")
    ax.set_ylabel(r"$\chi^2/{\rm dof}$, FSPL no parallax")
    ax.set_title(r"No-parallax FSPL fit quality")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "chi2red_vs_delta_chi2_0_12.pdf")
    fig.savefig(outdir / "chi2red_vs_delta_chi2_0_12.png", dpi=250)
    plt.close(fig)


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
        label="Base candidates",
    )

    if len(sel) > 0:
        ax.scatter(
            sel["rho_true"],
            sel["rho_fit"],
            s=18,
            alpha=0.8,
            label=r"$0 \leq \Delta\chi^2 < 12$",
        )

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Injected $\rho_{\rm true}$")
    ax.set_ylabel(r"Recovered $\rho_{\rm fit}$")
    ax.set_title(r"$\rho$ recovery for no-parallax FSPL-confused events")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "rho_fit_vs_rho_true_delta_chi2_0_12.pdf")
    fig.savefig(outdir / "rho_fit_vs_rho_true_delta_chi2_0_12.png", dpi=250)
    plt.close(fig)


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
    ax.set_title(r"Events with $0 \leq \Delta\chi^2 < 12$")

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\Delta\chi^2$")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "piE_vs_tE_delta_chi2_0_12.pdf")
    fig.savefig(outdir / "piE_vs_tE_delta_chi2_0_12.png", dpi=250)
    plt.close(fig)


def plot_selection_counts(df, selected, outdir, delta_chi2_max=12.0):
    n_total = len(df)

    n_negative = int(
        np.isfinite(df["delta_chi2_true"]).sum()
        - ((df["delta_chi2_true"] >= 0) & np.isfinite(df["delta_chi2_true"])).sum()
    )

    n_nonnegative = int(
        ((df["delta_chi2_true"] >= 0) & np.isfinite(df["delta_chi2_true"])).sum()
    )

    n_selected = len(selected)

    labels = [
        "Base",
        r"$\Delta\chi^2<0$",
        r"$\Delta\chi^2\geq0$",
        rf"$0\leq\Delta\chi^2<{delta_chi2_max:g}$",
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
    ax.set_title(r"Selection counts for the no-parallax FSPL confusion sample")
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

    fig.tight_layout()
    fig.savefig(outdir / "selection_counts_delta_chi2_0_12.pdf")
    fig.savefig(outdir / "selection_counts_delta_chi2_0_12.png", dpi=250)
    plt.close(fig)


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
# Light-curve plotting pipeline
# ============================================================

def run_lightcurve_pipeline(
    representative_csv,
    output_dir,
    plot_script,
    config,
    n_events,
    reference_band,
    plot_n_tE,
    plot_inset_n_tE,
):
    cmd = [
        sys.executable,
        str(plot_script),
        "--events-csv",
        str(representative_csv),
        "--n-events",
        str(n_events),
        "--reference-band",
        str(reference_band),
        "--plot-n-tE",
        str(plot_n_tE),
        "--plot-inset-n_tE" if False else "--plot-inset-n-tE",
        str(plot_inset_n_tE),
        "--fit-trajectory-time-mode",
        "same_true_window",
        "--output-dir",
        str(output_dir),
        "--config",
        str(config),
    ]

    print("\nRunning light-curve pipeline:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--candidates-csv", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--plot-script", default=str(DEFAULT_PLOT_SCRIPT))
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

    parser.add_argument("--skip-lightcurves", action="store_true")

    parser.add_argument("--no-visual-filter", action="store_true")
    parser.add_argument("--visual-chi2-red-max", type=float, default=1.2)
    parser.add_argument("--visual-min-peak", type=int, default=10)
    parser.add_argument("--visual-tE-ratio-min", type=float, default=0.3)
    parser.add_argument("--visual-tE-ratio-max", type=float, default=3.0)

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    candidates_csv = Path(args.candidates_csv)

    outdir = run_dir / "figures" / args.output_subdir
    diagnostics_dir = outdir / "diagnostics"
    lightcurves_dir = outdir / "lightcurves_with_inset_pipeline"

    outdir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    print("Input candidates:", candidates_csv)
    print("Output dir:", outdir)

    df = pd.read_csv(candidates_csv)
    df = ensure_columns(df)

    all_with_flags, selected = select_delta_chi2_window(
        df,
        delta_chi2_min=args.delta_chi2_min,
        delta_chi2_max=args.delta_chi2_max,
        chi2_red_max=args.chi2_red_max,
        min_peak=args.min_peak,
    )

    n_negative = int(
        np.isfinite(all_with_flags["delta_chi2_true"]).sum()
        - (
            np.isfinite(all_with_flags["delta_chi2_true"])
            & (all_with_flags["delta_chi2_true"] >= 0)
        ).sum()
    )

    print("\nSelection counts")
    print("----------------")
    print("Base candidates:", len(df))
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

    print("\nSaved:")
    print(all_out)
    print(sel_out)
    print(rep_out)

    if len(selected) > 0:
        print("\nRepresentative events:")
        show_cols = [
            "selection_reason",
            "global_i",
            "delta_chi2_true",
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
        ]
        show_cols = [c for c in show_cols if c in representatives.columns]
        print(representatives[show_cols])

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
            run_lightcurve_pipeline(
                representative_csv=rep_out,
                output_dir=lightcurves_dir,
                plot_script=Path(args.plot_script),
                config=Path(args.config),
                n_events=args.n_events,
                reference_band=args.reference_band,
                plot_n_tE=args.plot_n_tE,
                plot_inset_n_tE=args.plot_inset_n_tE,
            )

            print("\nLight-curve plots written to:")
            print(lightcurves_dir / "plots")


if __name__ == "__main__":
    main()
