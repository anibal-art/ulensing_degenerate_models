#!/usr/bin/env python
import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--selected-csv", required=True)
    p.add_argument("--outdir", default=None)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--show-model", action="store_true")
    p.add_argument("--time-zero", choices=["t0", "min", "none"], default="t0")
    return p.parse_args()


def safe_name(x):
    return (
        str(x)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("|", "_")
        .replace(":", "_")
    )


def read_multifit_result(path):
    path = Path(path)
    df = pd.read_parquet(path)
    if len(df) == 0:
        raise RuntimeError(f"Empty multi_fit file: {path}")
    return df.iloc[0].to_dict()


def format_pvalue(p):
    p = float(p)
    if p == 0:
        return r"$p_{\rm LRT}<10^{-300}$"
    if p < 1e-3:
        return rf"$p_{{\rm LRT}}={p:.2e}$"
    return rf"$p_{{\rm LRT}}={p:.3f}$"


def hypothesis_annotation_text(row):
    lines = []

    lines.append(r"$\bf{H0\ vs\ H1}$")
    lines.append(r"$H_0$: FSPL no parallax")
    lines.append(r"$H_1$: FSPL parallax")

    if "H0_n_data" in row and "H1_n_data" in row:
        lines.append(
            rf"$N_{{H0}}={int(row['H0_n_data'])}$, "
            rf"$N_{{H1}}={int(row['H1_n_data'])}$"
        )

    if "H0_chi2" in row:
        lines.append(rf"$\chi^2_{{H0}}={float(row['H0_chi2']):.2f}$")

    if "H1_chi2" in row:
        lines.append(rf"$\chi^2_{{H1}}={float(row['H1_chi2']):.2f}$")

    if "true_generator_chi2" in row:
        lines.append(rf"$\chi^2_{{true}}={float(row['true_generator_chi2']):.2f}$")

    if "LRT" in row:
        lines.append(rf"$\Delta\chi^2={float(row['LRT']):.2f}$")

    if "p_value_LRT" in row:
        p = float(row["p_value_LRT"])
        lines.append(format_pvalue(p))

        if p < 1e-6:
            lines.append(r"strong detection")
        elif p < 1e-3:
            lines.append(r"detection")
        elif p < 0.05:
            lines.append(r"marginal detection")
        else:
            lines.append(r"non detection")

    if "H1_minus_true_chi2" in row:
        d = float(row["H1_minus_true_chi2"])
    elif "H1_chi2" in row and "true_generator_chi2" in row:
        d = float(row["H1_chi2"]) - float(row["true_generator_chi2"])
    else:
        d = None

    if d is not None:
        lines.append(rf"$\chi^2_{{H1}}-\chi^2_{{true}}={d:.2f}$")
        if d > 10:
            lines.append("warning: H1 local minimum?")

    if "quality_flag" in row:
        lines.append(f"quality: {row['quality_flag']}")

    return "\n".join(lines)


def annotate_hypothesis(ax, row):
    ax.text(
        0.98,
        0.98,
        hypothesis_annotation_text(row),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="white",
            ec="0.45",
            alpha=0.92,
        ),
        zorder=100,
    )


def find_event_dir(row):
    if "event_dir" in row and isinstance(row["event_dir"], str):
        p = Path(row["event_dir"])
        if p.exists():
            return p

    if "multi_fit_file" in row and isinstance(row["multi_fit_file"], str):
        p = Path(row["multi_fit_file"])
        if p.exists():
            if p.parent.name == "multi_fit":
                return p.parent.parent
            return p.parent

    raise RuntimeError("Could not infer event_dir.")


def find_lightcurve_tables(event_dir):
    event_dir = Path(event_dir)

    candidates = []
    for pattern in [
        "**/*lc*.parquet",
        "**/*lightcurve*.parquet",
        "**/*light_curve*.parquet",
        "**/*phot*.parquet",
        "**/*data*.parquet",
        "**/*lc*.csv",
        "**/*lightcurve*.csv",
        "**/*light_curve*.csv",
        "**/*phot*.csv",
        "**/*data*.csv",
        "**/*lc*.dat",
        "**/*lightcurve*.dat",
        "**/*phot*.dat",
    ]:
        candidates.extend(event_dir.glob(pattern))

    bad_words = [
        "multi_fit",
        "summary",
        "run_summary",
    ]

    clean = []
    for p in candidates:
        s = str(p).lower()
        if any(w in s for w in bad_words):
            continue
        clean.append(p)

    return sorted(set(clean))


def read_table(path):
    path = Path(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    return pd.read_csv(path, sep=r"\s+", comment="#", engine="python")


def infer_columns(df):
    cols = list(df.columns)
    low = {c: str(c).lower() for c in cols}

    def first_exact(names):
        for name in names:
            for c in cols:
                if low[c] == name:
                    return c
        return None

    def first_contains(tokens, reject=()):
        for c in cols:
            s = low[c]
            if all(t in s for t in tokens) and not any(r in s for r in reject):
                return c
        return None

    time_col = (
        first_exact(["time", "jd", "mjd", "hjd", "bjd"])
        or first_contains(["time"])
        or first_contains(["jd"])
        or first_contains(["mjd"])
    )

    mag_col = (
        first_exact(["mag", "magnitude", "obs_mag", "observed_mag"])
        or first_contains(["mag"], reject=("err", "error", "sigma"))
    )

    magerr_col = (
        first_exact(["err_mag", "mag_err", "sigma_mag", "mag_sigma"])
        or first_contains(["err", "mag"])
        or first_contains(["sigma", "mag"])
    )

    flux_col = (
        first_exact(["flux", "observed_flux", "obs_flux"])
        or first_contains(["flux"], reject=("err", "error", "sigma"))
    )

    fluxerr_col = (
        first_exact(["err_flux", "flux_err", "sigma_flux", "flux_sigma"])
        or first_contains(["err", "flux"])
        or first_contains(["sigma", "flux"])
    )

    band_col = (
        first_exact(["band", "filter", "camera_filter"])
        or first_contains(["band"])
        or first_contains(["filter"])
    )

    tel_col = (
        first_exact(["telescope", "tel", "instrument"])
        or first_contains(["telescope"])
        or first_contains(["instrument"])
    )

    group_col = band_col if band_col is not None else tel_col

    if time_col is None and len(cols) >= 3:
        time_col = cols[0]

    if mag_col is not None:
        y_col = mag_col
        yerr_col = magerr_col
        y_kind = "mag"
    elif flux_col is not None:
        y_col = flux_col
        yerr_col = fluxerr_col
        y_kind = "flux"
    elif len(cols) >= 3:
        y_col = cols[1]
        yerr_col = cols[2]
        y_kind = "flux"
    else:
        y_col = None
        yerr_col = None
        y_kind = None

    return {
        "time": time_col,
        "y": y_col,
        "yerr": yerr_col,
        "group": group_col,
        "y_kind": y_kind,
    }


def load_lightcurve_table(event_dir):
    candidates = find_lightcurve_tables(event_dir)

    errors = []

    for p in candidates:
        try:
            df = read_table(p)
            if len(df) == 0:
                continue

            cols = infer_columns(df)

            if cols["time"] is None or cols["y"] is None:
                errors.append((str(p), "could not infer columns"))
                continue

            return p, df, cols

        except Exception as e:
            errors.append((str(p), repr(e)))

    inventory = sorted(str(p.relative_to(event_dir)) for p in Path(event_dir).rglob("*"))

    raise RuntimeError(
        "No usable lightcurve table found in event_dir.\n"
        f"event_dir = {event_dir}\n"
        f"candidates = {candidates[:20]}\n"
        f"errors = {errors[:20]}\n"
        f"inventory first 100 = {inventory[:100]}"
    )


def choose_time_reference(df, time_col, row, mode):
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)

    if mode == "none":
        return 0.0, str(time_col)

    if mode == "min":
        t_ref = np.nanmin(t)
        return t_ref, rf"{time_col} - {t_ref:.1f}"

    if mode == "t0":
        for key in [
            "true_t0",
            "truth_t0",
            "t0_true",
            "pyLIMA_true_t0",
            "H1_t0",
            "H1_fit_t0",
            "H1_best_t0",
            "H0_t0",
            "H0_fit_t0",
            "H0_best_t0",
        ]:
            if key in row:
                try:
                    val = float(row[key])
                    if np.isfinite(val):
                        return val, r"$t-t_0$ [days]"
                except Exception:
                    pass

    t_ref = np.nanmin(t)
    return t_ref, rf"{time_col} - {t_ref:.1f}"


def plot_one_lightcurve(row, outdir, time_zero="t0"):
    event_dir = find_event_dir(row)

    multi_fit_file = row.get("multi_fit_file", "")
    if isinstance(multi_fit_file, str) and Path(multi_fit_file).exists():
        mf = read_multifit_result(multi_fit_file)
        for k, v in mf.items():
            if k not in row or pd.isna(row[k]):
                row[k] = v

    table_path, df, cols = load_lightcurve_table(event_dir)

    time_col = cols["time"]
    y_col = cols["y"]
    yerr_col = cols["yerr"]
    group_col = cols["group"]
    y_kind = cols["y_kind"]

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    if yerr_col is not None and yerr_col in df.columns:
        df[yerr_col] = pd.to_numeric(df[yerr_col], errors="coerce")

    keep = np.isfinite(df[time_col]) & np.isfinite(df[y_col])
    df = df.loc[keep].copy()

    if len(df) == 0:
        raise RuntimeError(f"No finite data in {table_path}")

    t_ref, xlabel = choose_time_reference(df, time_col, row, time_zero)
    df["_plot_time"] = df[time_col] - t_ref

    if group_col is None or group_col not in df.columns:
        df["_group"] = "all"
        group_col = "_group"

    category = row.get("lightcurve_category", "selected")
    global_i = row.get("global_i", "NA")
    seed = row.get("simulation_seed", "NA")

    fig, ax = plt.subplots(figsize=(10, 5.8))

    for group, sub in df.groupby(group_col):
        x = sub["_plot_time"].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)

        if yerr_col is not None and yerr_col in sub.columns:
            yerr = sub[yerr_col].to_numpy(dtype=float)
            if np.isfinite(yerr).any():
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt=".",
                    ms=3,
                    alpha=0.7,
                    label=str(group),
                )
            else:
                ax.plot(x, y, ".", ms=3, alpha=0.7, label=str(group))
        else:
            ax.plot(x, y, ".", ms=3, alpha=0.7, label=str(group))

    if y_kind == "mag":
        ax.invert_yaxis()
        ax.set_ylabel("Magnitude")
    else:
        ax.set_ylabel(str(y_col))

    ax.set_xlabel(xlabel)

    title = (
        f"{category} | global_i={global_i} | seed={seed}\n"
        f"{event_dir.name}"
    )

    ax.set_title(title)
    ax.grid(alpha=0.3)

    if df[group_col].nunique() <= 12:
        ax.legend(frameon=False, fontsize=8, loc="best")

    annotate_hypothesis(ax, row)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_png = (
        outdir
        / f"{safe_name(category)}_global_{safe_name(global_i)}_seed_{safe_name(seed)}.png"
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return {
        "global_i": global_i,
        "simulation_seed": seed,
        "category": category,
        "event_dir": str(event_dir),
        "table_path": str(table_path),
        "out_png": str(out_png),
        "status": "ok",
        "time_col": time_col,
        "y_col": y_col,
        "yerr_col": yerr_col,
        "group_col": group_col,
        "y_kind": y_kind,
    }


def main():
    args = parse_args()

    selected = pd.read_csv(args.selected_csv)

    if args.max_events is not None:
        selected = selected.head(args.max_events).copy()

    if args.outdir is None:
        outdir = Path(args.selected_csv).parent / "lightcurve_plots"
    else:
        outdir = Path(args.outdir)

    logs = []

    for i, row in selected.iterrows():
        row = row.to_dict()

        try:
            info = plot_one_lightcurve(
                row,
                outdir=outdir,
                time_zero=args.time_zero,
            )
            print("[ok]", info["global_i"], info["out_png"])
            logs.append(info)

        except Exception as e:
            info = {
                "global_i": row.get("global_i", "NA"),
                "simulation_seed": row.get("simulation_seed", "NA"),
                "category": row.get("lightcurve_category", "selected"),
                "event_dir": row.get("event_dir", ""),
                "out_png": "",
                "status": repr(e),
            }
            print("[failed]", info["global_i"], info["status"])
            logs.append(info)

    log = pd.DataFrame(logs)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "lightcurve_plot_log.csv"
    log.to_csv(log_path, index=False)

    print()
    print("saved:")
    print(outdir)
    print(log_path)


if __name__ == "__main__":
    main()
