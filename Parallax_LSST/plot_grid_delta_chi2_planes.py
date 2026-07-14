#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graficos de grilla para corrida single-source Rubin-only.

Lee los resultados de la corrida:
    Grid_SingleSource_directCatalog_near_lens_RubinOnly_PSPLparallax_fitNoPiE_MAFbin020

y genera visualizaciones de planos de la grilla, por ejemplo:
    D_L vs M_L coloreado por Delta chi2
    D_L vs t_E coloreado por Delta chi2
    D_L vs pi_E coloreado por Delta chi2

Incluye dos tipos de graficos:
1. raw scatter: todos los puntos de la grilla simulados.
2. collapsed grid: agrupa puntos con el mismo par (x,y) y colorea por
   mediana, maximo o fraccion detectable.

La version collapsed es importante porque en una grilla muchos puntos tienen
el mismo D_L y M_L, pero difieren en D_S, mu_rel, u0, t0 y phi_pi.
Si se grafica el raw scatter solamente, muchos puntos quedan superpuestos.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ============================================================
# Configuracion
# ============================================================

RUN_DIR = Path(
    "/home/anibal/Parallax_LSST/runs/"
    "Grid_SingleSource_directCatalog_near_lens_RubinOnly_PSPLparallax_fitNoPiE_MAFbin020"
)

RESULTS_DIR = RUN_DIR / "results"
TABLES_DIR = RUN_DIR / "tables"
PLOTS_DIR = RUN_DIR / "plots" / "grid_planes"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DETECTABLE_DELTA_CHI2_MIN = 100.0

# Si ya existe la tabla diagnostica producida por la corrida, la usa.
# Si no existe, la reconstruye desde results/true y results/fit_rr.
DIAGNOSTICS_TABLE = TABLES_DIR / "single_source_grid_diagnostics.parquet"

# Para evitar que valores negativos o cero rompan LogNorm.
COLOR_LOG_FLOOR = 1e-3


# ============================================================
# Estilo
# ============================================================

plt.rcParams.update({
    "figure.figsize": (7.2, 5.4),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "mathtext.fontset": "dejavuserif",
})


LABELS = {
    "grid_D_L_kpc": r"$D_L$ [kpc]",
    "grid_D_S_kpc": r"$D_S$ [kpc]",
    "grid_M_L_Msun": r"$M_L$ [$M_\odot$]",
    "grid_mu_rel_masyr": r"$\mu_{\rm rel}$ [mas yr$^{-1}$]",
    "grid_tE_days": r"$t_E$ [d]",
    "grid_piE": r"$\pi_E$",
    "grid_pi_rel_mas": r"$\pi_{\rm rel}$ [mas]",
    "grid_thetaE_mas": r"$\theta_E$ [mas]",
    "grid_u0": r"$u_0$",
    "grid_t0": r"$t_0$ [d]",
    "grid_phi_pi_rad": r"$\phi_{\pi_E}$ [rad]",
    "delta_chi2_true": r"$\Delta\chi^2$",
    "delta_chi2_true_per_point": r"$\Delta\chi^2/N_{\rm data}$",
    "chi2_dof_nopie": r"$\chi^2_{\rm No\,\pi_E}/{\rm dof}$",
    "delta_chi2_detectable": r"$\Delta\chi^2>100$ fraction",
}


# ============================================================
# Lectura de resultados
# ============================================================

def read_parquet_tree(path):
    path = Path(path)
    files = sorted(path.rglob("*.parquet"))

    if len(files) == 0:
        raise FileNotFoundError(f"No encontre archivos parquet en {path}")

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["__file__"] = str(f)
            dfs.append(df)
        except Exception as e:
            print(f"No pude leer {f}: {e}")

    if len(dfs) == 0:
        raise RuntimeError(f"No pude leer ningun parquet en {path}")

    return pd.concat(dfs, ignore_index=True)


def read_result_kind(results_dir, kind):
    results_dir = Path(results_dir)
    direct = results_dir / kind

    if direct.exists():
        return read_parquet_tree(direct)

    files = [
        p for p in results_dir.rglob("*.parquet")
        if kind.lower() in str(p).lower()
    ]

    if len(files) == 0:
        raise FileNotFoundError(f"No encontre resultados tipo {kind} en {results_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["__file__"] = str(f)
            dfs.append(df)
        except Exception as e:
            print(f"No pude leer {f}: {e}")

    if len(dfs) == 0:
        raise RuntimeError(f"No pude leer ningun parquet para kind={kind}")

    return pd.concat(dfs, ignore_index=True)


def safe_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_col_after_merge(df, base):
    return safe_col(df, [base, f"{base}_true", f"{base}_fit"])


def choose_merge_columns(true, fit):
    candidates = [
        ["global_i"],
        ["Source", "Set"],
        ["Source"],
        ["grid_id", "field_name", "source_index"],
    ]

    for cols in candidates:
        if all(c in true.columns for c in cols) and all(c in fit.columns for c in cols):
            return cols

    raise KeyError(
        "No pude encontrar columnas para merge entre true y fit.\n"
        f"true columns = {true.columns.tolist()}\n"
        f"fit columns = {fit.columns.tolist()}"
    )


def normalize_fit_columns(fit):
    fit = fit.copy()

    if "chi2" not in fit.columns:
        if "chichi" in fit.columns:
            fit["chi2"] = fit["chichi"]
        elif "chi2_fit" in fit.columns:
            fit["chi2"] = fit["chi2_fit"]
        else:
            raise KeyError("No encontre columna chi2/chichi en fit_rr.")

    if "dof" not in fit.columns:
        fit["dof"] = np.nan

    fit["chi2_dof"] = fit["chi2"] / fit["dof"]

    return fit


def build_diagnostics_from_results():
    true = read_result_kind(RESULTS_DIR, "true")
    fit = read_result_kind(RESULTS_DIR, "fit_rr")
    fit = normalize_fit_columns(fit)

    id_cols = choose_merge_columns(true, fit)

    print("=" * 80)
    print("Merge true-fit")
    print("id columns:", id_cols)
    print("N true:", len(true))
    print("N fit: ", len(fit))
    print("=" * 80)

    df = pd.merge(
        true,
        fit,
        on=id_cols,
        how="inner",
        suffixes=("_true", "_fit"),
    )

    if len(df) == 0:
        raise RuntimeError("El merge true-fit quedo vacio.")

    chi2_fit_col = safe_col(df, ["chi2", "chi2_fit", "chichi", "chichi_fit"])
    chi2_true_col = safe_col(df, ["chi2_true", "chi2_true_true", "chi2_true_fit"])
    n_data_col = safe_col(df, ["n_data_true", "n_data_true_true", "n_data_true_fit"])
    dof_col = safe_col(df, ["dof", "dof_fit"])
    chi2_dof_col = safe_col(df, ["chi2_dof", "chi2_dof_fit"])

    if chi2_fit_col is None or chi2_true_col is None:
        raise KeyError("No encuentro chi2 del fit o chi2_true del modelo verdadero.")

    df["chi2_nopie"] = pd.to_numeric(df[chi2_fit_col], errors="coerce")
    df["chi2_true_model"] = pd.to_numeric(df[chi2_true_col], errors="coerce")
    df["delta_chi2_true"] = df["chi2_nopie"] - df["chi2_true_model"]

    if n_data_col is not None:
        df["n_data_true_model"] = pd.to_numeric(df[n_data_col], errors="coerce")
        df["delta_chi2_true_per_point"] = df["delta_chi2_true"] / df["n_data_true_model"]
        df["chi2_true_per_point"] = df["chi2_true_model"] / df["n_data_true_model"]
    else:
        df["n_data_true_model"] = np.nan
        df["delta_chi2_true_per_point"] = np.nan
        df["chi2_true_per_point"] = np.nan

    if chi2_dof_col is not None:
        df["chi2_dof_nopie"] = pd.to_numeric(df[chi2_dof_col], errors="coerce")
    elif dof_col is not None:
        df["dof_nopie"] = pd.to_numeric(df[dof_col], errors="coerce")
        df["chi2_dof_nopie"] = df["chi2_nopie"] / df["dof_nopie"]
    else:
        df["chi2_dof_nopie"] = np.nan

    # Normalizar columnas de grilla despues del merge.
    grid_cols = [
        "grid_id",
        "grid_D_L_kpc",
        "grid_D_S_kpc",
        "grid_M_L_Msun",
        "grid_mu_rel_masyr",
        "grid_u0",
        "grid_t0",
        "grid_phi_pi_rad",
        "grid_pi_rel_mas",
        "grid_thetaE_mas",
        "grid_piE",
        "grid_piEN",
        "grid_piEE",
        "grid_tE_days",
        "field_name",
        "source_index",
        "global_i",
    ]

    for base in grid_cols:
        c = get_col_after_merge(df, base)
        if c is not None:
            df[base] = df[c]

    df["delta_chi2_detectable"] = df["delta_chi2_true"] > DETECTABLE_DELTA_CHI2_MIN

    df.to_parquet(DIAGNOSTICS_TABLE, index=False)
    df.to_csv(DIAGNOSTICS_TABLE.with_suffix(".csv"), index=False)

    print("=" * 80)
    print("Diagnostics saved:")
    print(DIAGNOSTICS_TABLE)
    print("N merged:", len(df))
    print("N detectable:", int(df["delta_chi2_detectable"].sum()))
    print("=" * 80)

    return df


def load_or_build_diagnostics():
    if DIAGNOSTICS_TABLE.exists():
        print("=" * 80)
        print("Usando tabla diagnostica existente:")
        print(DIAGNOSTICS_TABLE)
        print("=" * 80)
        df = pd.read_parquet(DIAGNOSTICS_TABLE)
    else:
        df = build_diagnostics_from_results()

    return df


# ============================================================
# Preparacion para graficos
# ============================================================

def finite_positive_mask(df, cols):
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        mask &= np.isfinite(pd.to_numeric(df[c], errors="coerce"))
        mask &= pd.to_numeric(df[c], errors="coerce") > 0
    return mask


def clean_for_plane(df, x_col, y_col, color_col):
    needed = [x_col, y_col, color_col]
    missing = [c for c in needed if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Faltan columnas para el plano: {missing}")

    out = df.copy()

    for c in needed:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    mask = (
        np.isfinite(out[x_col])
        & np.isfinite(out[y_col])
        & np.isfinite(out[color_col])
        & (out[x_col] > 0)
        & (out[y_col] > 0)
    )

    out = out.loc[mask].copy().reset_index(drop=True)

    return out


def color_norm(values, log_color):
    values = np.asarray(values, dtype=float)

    if not log_color:
        return None, values

    positive = values[np.isfinite(values) & (values > 0)]

    if len(positive) == 0:
        return None, values

    floor = max(np.nanmin(positive), COLOR_LOG_FLOOR)
    values_plot = np.where(values > 0, values, floor)
    vmax = np.nanmax(values_plot)

    if not np.isfinite(vmax) or vmax <= floor:
        return None, values_plot

    norm = mcolors.LogNorm(vmin=floor, vmax=vmax)

    return norm, values_plot


def set_axis_scales(ax, x_col, y_col):
    log_like = {
        "grid_D_L_kpc",
        "grid_D_S_kpc",
        "grid_M_L_Msun",
        "grid_mu_rel_masyr",
        "grid_tE_days",
        "grid_piE",
        "grid_pi_rel_mas",
        "grid_thetaE_mas",
    }

    if x_col in log_like:
        ax.set_xscale("log")
    if y_col in log_like:
        ax.set_yscale("log")


def format_axes(ax, x_col, y_col):
    ax.set_xlabel(LABELS.get(x_col, x_col))
    ax.set_ylabel(LABELS.get(y_col, y_col))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)


def save_raw_scatter(df, x_col, y_col, color_col, output_name, log_color=True):
    data = clean_for_plane(df, x_col, y_col, color_col)

    if len(data) == 0:
        print(f"Skip raw {output_name}: sin datos")
        return

    c = data[color_col].values
    norm, c_plot = color_norm(c, log_color=log_color)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    sc = ax.scatter(
        data[x_col],
        data[y_col],
        c=c_plot,
        s=18,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
        norm=norm,
    )

    set_axis_scales(ax, x_col, y_col)
    format_axes(ax, x_col, y_col)

    ax.set_title(
        f"Raw grid points: {LABELS.get(x_col, x_col)} vs {LABELS.get(y_col, y_col)}"
    )

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(LABELS.get(color_col, color_col))

    fig.tight_layout()
    out = PLOTS_DIR / output_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


def aggregate_plane(df, x_col, y_col):
    """
    Agrupa la grilla por el par (x,y).

    Esto evita que muchos puntos superpuestos escondan informacion.
    Para cada celda en el plano se calculan estadisticos de Delta chi2.
    """

    data = df.copy()

    for c in [x_col, y_col, "delta_chi2_true", "delta_chi2_true_per_point"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    mask = (
        np.isfinite(data[x_col])
        & np.isfinite(data[y_col])
        & (data[x_col] > 0)
        & (data[y_col] > 0)
        & np.isfinite(data["delta_chi2_true"])
    )

    data = data.loc[mask].copy()

    if len(data) == 0:
        return pd.DataFrame()

    data["delta_chi2_detectable"] = data["delta_chi2_true"] > DETECTABLE_DELTA_CHI2_MIN

    grouped = data.groupby([x_col, y_col], observed=True).agg(
        n_events=("delta_chi2_true", "size"),
        median_delta_chi2=("delta_chi2_true", "median"),
        max_delta_chi2=("delta_chi2_true", "max"),
        min_delta_chi2=("delta_chi2_true", "min"),
        mean_delta_chi2=("delta_chi2_true", "mean"),
        median_delta_chi2_per_point=("delta_chi2_true_per_point", "median"),
        fraction_detectable=("delta_chi2_detectable", "mean"),
    ).reset_index()

    return grouped


def save_collapsed_plane(
    df,
    x_col,
    y_col,
    stat_col,
    output_name,
    log_color=True,
    size_by_n=True,
):
    agg = aggregate_plane(df, x_col, y_col)

    if len(agg) == 0:
        print(f"Skip collapsed {output_name}: sin datos")
        return

    if stat_col not in agg.columns:
        raise KeyError(f"La tabla agregada no tiene {stat_col}")

    agg.to_csv(
        PLOTS_DIR / output_name.replace(".png", ".csv"),
        index=False,
    )

    c = agg[stat_col].values.astype(float)
    norm, c_plot = color_norm(c, log_color=log_color)

    if size_by_n:
        n = agg["n_events"].values.astype(float)
        n_norm = n / np.nanmax(n)
        sizes = 45.0 + 130.0 * np.sqrt(n_norm)
    else:
        sizes = 70.0

    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    sc = ax.scatter(
        agg[x_col],
        agg[y_col],
        c=c_plot,
        s=sizes,
        alpha=0.88,
        edgecolors="black",
        linewidths=0.35,
        rasterized=True,
        norm=norm,
    )

    set_axis_scales(ax, x_col, y_col)
    format_axes(ax, x_col, y_col)

    stat_label = {
        "median_delta_chi2": r"median $\Delta\chi^2$",
        "max_delta_chi2": r"max $\Delta\chi^2$",
        "min_delta_chi2": r"min $\Delta\chi^2$",
        "mean_delta_chi2": r"mean $\Delta\chi^2$",
        "median_delta_chi2_per_point": r"median $\Delta\chi^2/N_{\rm data}$",
        "fraction_detectable": r"fraction with $\Delta\chi^2>100$",
    }.get(stat_col, stat_col)

    ax.set_title(
        f"Collapsed grid: {stat_label}"
    )

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(stat_label)

    if size_by_n:
        ax.text(
            0.02,
            0.02,
            "Marker size $\\propto$ number of simulated grid points",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
        )

    fig.tight_layout()
    out = PLOTS_DIR / output_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


def save_fraction_detectable_plane(df, x_col, y_col, output_name):
    save_collapsed_plane(
        df,
        x_col,
        y_col,
        "fraction_detectable",
        output_name,
        log_color=False,
        size_by_n=True,
    )


def save_all_grid_plane_plots(df):
    """
    Genera un conjunto util de planos de la grilla.
    """

    planes = [
        ("grid_D_L_kpc", "grid_M_L_Msun", "DL_vs_ML"),
        ("grid_D_L_kpc", "grid_tE_days", "DL_vs_tE"),
        ("grid_D_L_kpc", "grid_piE", "DL_vs_piE"),
        ("grid_M_L_Msun", "grid_tE_days", "ML_vs_tE"),
        ("grid_piE", "grid_tE_days", "piE_vs_tE"),
        ("grid_mu_rel_masyr", "grid_tE_days", "murel_vs_tE"),
        ("grid_D_S_kpc", "grid_tE_days", "DS_vs_tE"),
        ("grid_D_L_kpc", "grid_thetaE_mas", "DL_vs_thetaE"),
    ]

    for x_col, y_col, tag in planes:
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Skip plane {tag}: faltan columnas")
            continue

        # Todos los puntos individuales.
        save_raw_scatter(
            df,
            x_col,
            y_col,
            "delta_chi2_true",
            f"raw_{tag}_colored_delta_chi2.png",
            log_color=True,
        )

        # Grilla colapsada por pares (x,y): mediana, maximo y fraccion detectable.
        save_collapsed_plane(
            df,
            x_col,
            y_col,
            "median_delta_chi2",
            f"collapsed_{tag}_median_delta_chi2.png",
            log_color=True,
        )

        save_collapsed_plane(
            df,
            x_col,
            y_col,
            "max_delta_chi2",
            f"collapsed_{tag}_max_delta_chi2.png",
            log_color=True,
        )

        save_fraction_detectable_plane(
            df,
            x_col,
            y_col,
            f"collapsed_{tag}_fraction_detectable_delta_chi2_gt_100.png",
        )


def print_summary(df):
    print("=" * 80)
    print("Grid diagnostic summary")
    print("=" * 80)
    print("N rows:", len(df))

    for c in [
        "delta_chi2_true",
        "delta_chi2_true_per_point",
        "grid_D_L_kpc",
        "grid_D_S_kpc",
        "grid_M_L_Msun",
        "grid_mu_rel_masyr",
        "grid_tE_days",
        "grid_piE",
    ]:
        if c in df.columns:
            print("-" * 80)
            print(c)
            print(pd.to_numeric(df[c], errors="coerce").describe())

    if "delta_chi2_true" in df.columns:
        detectable = pd.to_numeric(df["delta_chi2_true"], errors="coerce") > DETECTABLE_DELTA_CHI2_MIN
        print("-" * 80)
        print(f"N detectable Delta chi2 > {DETECTABLE_DELTA_CHI2_MIN:g}:", int(detectable.sum()))
        print("Fraction detectable:", float(detectable.mean()))

    print("=" * 80)
    print("Plots saved in:")
    print(PLOTS_DIR)
    print("=" * 80)


# ============================================================
# Main
# ============================================================

def main():
    df = load_or_build_diagnostics()

    # Asegurar tipos numericos en columnas principales.
    numeric_cols = [
        "grid_D_L_kpc",
        "grid_D_S_kpc",
        "grid_M_L_Msun",
        "grid_mu_rel_masyr",
        "grid_u0",
        "grid_t0",
        "grid_phi_pi_rad",
        "grid_pi_rel_mas",
        "grid_thetaE_mas",
        "grid_piE",
        "grid_tE_days",
        "delta_chi2_true",
        "delta_chi2_true_per_point",
        "chi2_dof_nopie",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "delta_chi2_detectable" not in df.columns and "delta_chi2_true" in df.columns:
        df["delta_chi2_detectable"] = df["delta_chi2_true"] > DETECTABLE_DELTA_CHI2_MIN

    # Guardar una copia limpia usada por estos plots.
    clean_table = TABLES_DIR / "single_source_grid_diagnostics_for_grid_planes.parquet"
    df.to_parquet(clean_table, index=False)
    df.to_csv(clean_table.with_suffix(".csv"), index=False)

    print_summary(df)
    save_all_grid_plane_plots(df)


if __name__ == "__main__":
    main()
