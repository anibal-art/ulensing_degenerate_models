#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análisis de corrida Rubin-only:
eventos generados con paralaje y ajustados con PSPL sin paralaje.

Genera:
- histogramas logarítmicos de Delta chi2
- histogramas logarítmicos de chi2/dof
- chi2 true vs chi2 NoPiE en log-log
- chi2 NoPiE vs piE true en log-log
- chi2/dof NoPiE vs piE true en log-log
- chi2 NoPiE vs tE true en log-log
- chi2/dof NoPiE vs tE true en log-log
- chi2 NoPiE vs tE fit en log-log
- chi2/dof NoPiE vs tE fit en log-log
- Delta chi2 vs piE true en log-log
- Delta chi2 vs tE true en log-log
- Delta chi2 vs tE fit en log-log
- Delta chi2 vs Delta tE = tE_fit - tE_true
- Delta chi2 vs |Delta tE|
- Delta chi2 vs masa true en log-log
- Delta chi2 por punto vs parámetros físicos
- N_data vs piE true
- Delta chi2/N_data vs N_data
- fracción de eventos Good No piE vs N_data
- bias de t0, |u0| y tE
- fit vs true para tE, u0 y t0
- tabla final de diagnóstico
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Estilo de figuras para paper
# ============================================================

PAPER_FIG_EXTENSIONS = (".png",)

plt.rcParams.update({
    "figure.figsize": (6.4, 4.8),
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

# Labels comunes de leyendas en inglés
ALL_EVENTS_LABEL = "All events"
GOOD_NOPIE_LABEL = r"Good No $\pi_E$"
THRESHOLD_LABEL = "Adopted threshold"
ZERO_LINE_LABEL = "Zero line"
ZERO_BIAS_LABEL = "Zero bias"
ONE_TO_ONE_LABEL = "1:1 relation"


# ============================================================
# Configuración
# ============================================================

RUN_DIR = Path(
    "/home/anibal/Parallax_LSST/runs/"
    "GalPlane_near_lenses_RubinOnly_PSPLparallax_fitNoPiE_MAFbin020"
)

RESULTS_DIR = RUN_DIR / "results"
PLOTS_DIR = RUN_DIR / "plots"
TABLES_DIR = RUN_DIR / "tables"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Criterios para decir que NoPiE ajustó bien
DELTA_CHI2_MAX = 100.0
DELTA_CHI2_PER_POINT_MAX = 0.1
CHI2_DOF_MAX = 2.0

# Criterio para detectar señal de paralaje en la comparación
# entre el modelo verdadero con paralaje y el mejor ajuste sin paralaje.
DETECTABLE_DELTA_CHI2_MIN = DELTA_CHI2_MAX

# Si es True, la fracción detectable se calcula solamente sobre eventos
# cuyo modelo verdadero tiene chi2/N_data consistente.
REQUIRE_GOOD_TRUE_CHI2_FOR_DETECTABLE = True

# Número de bines logarítmicos en tE para la fracción detectable.
N_TE_DETECTABLE_BINS = 12

# Bines con menos eventos se guardan en la tabla, pero no se anotan tanto
# en el gráfico para no saturarlo.
MIN_EVENTS_PER_TE_BIN_TO_ANNOTATE = 5

# Consistencia del modelo verdadero
CHI2_TRUE_PER_POINT_MIN = 0.5
CHI2_TRUE_PER_POINT_MAX = 1.5

# Para evitar que outliers arruinen histogramas de bias
CLIP_FRAC_BIAS = 5.0


# ============================================================
# Helpers de lectura
# ============================================================

def read_parquet_tree(path):
    """
    Lee todos los parquet debajo de path.
    """
    path = Path(path)
    files = sorted(path.rglob("*.parquet"))

    if len(files) == 0:
        raise FileNotFoundError(f"No encontré archivos parquet en {path}")

    dfs = []

    for f in files:
        try:
            df = pd.read_parquet(f)
            df["__file__"] = str(f)
            dfs.append(df)
        except Exception as e:
            print(f"No pude leer {f}: {e}")

    if len(dfs) == 0:
        raise RuntimeError(f"No pude leer ningún parquet en {path}")

    return pd.concat(dfs, ignore_index=True)


def read_result_kind(results_dir, kind):
    """
    Lee resultados true o fit_rr de forma flexible.

    Primero intenta results/kind.
    Si no existe, busca parquet cuyo path contenga kind.
    """
    direct = Path(results_dir) / kind

    if direct.exists():
        return read_parquet_tree(direct)

    files = [
        p for p in Path(results_dir).rglob("*.parquet")
        if kind.lower() in str(p).lower()
    ]

    if len(files) == 0:
        print("=" * 80)
        print(f"No encontré resultados tipo '{kind}'.")
        print("Archivos disponibles:")
        for p in Path(results_dir).rglob("*.parquet"):
            print(p)
        print("=" * 80)
        raise FileNotFoundError(f"No encontré parquet para kind={kind}")

    dfs = []

    for f in files:
        df = pd.read_parquet(f)
        df["__file__"] = str(f)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def choose_id_columns(true, fit):
    """
    Define columnas para merge.
    Preferimos Source + Set si existen.
    """
    if "Source" in true.columns and "Source" in fit.columns:
        if "Set" in true.columns and "Set" in fit.columns:
            return ["Source", "Set"]

        return ["Source"]

    for c in ["id", "event_id", "nevent", "i"]:
        if c in true.columns and c in fit.columns:
            return [c]

    raise KeyError(
        "No pude encontrar columnas comunes para identificar eventos.\n"
        f"true columns = {true.columns.tolist()}\n"
        f"fit columns = {fit.columns.tolist()}"
    )


def normalize_fit_columns(fit):
    """
    Normaliza nombres de chi2.
    """
    fit = fit.copy()

    if "chi2" not in fit.columns:
        if "chichi" in fit.columns:
            fit["chi2"] = fit["chichi"]
        elif "chi2_fit" in fit.columns:
            fit["chi2"] = fit["chi2_fit"]
        else:
            raise KeyError(
                "No encontré columna chi2 ni chichi en fit_rr."
            )

    if "dof" not in fit.columns:
        fit["dof"] = np.nan

    fit["chi2_dof"] = fit["chi2"] / fit["dof"]

    return fit


def safe_col(df, candidates):
    """
    Devuelve el primer nombre de columna existente.
    """
    for c in candidates:
        if c in df.columns:
            return c

    return None


def finite_series(x):
    """
    Devuelve valores finitos.
    """
    return x.replace([np.inf, -np.inf], np.nan).dropna()


def positive_finite_series(x):
    """
    Devuelve valores finitos y estrictamente positivos.
    Necesario para escalas logarítmicas.
    """
    x = finite_series(x)
    return x[x > 0]


def log_bins(x, nbins=30):
    """
    Bins logarítmicos ajustados al rango positivo de los datos.
    """
    x = positive_finite_series(x)

    if len(x) == 0:
        return None

    xmin = np.nanmin(x)
    xmax = np.nanmax(x)

    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return None

    if xmin <= 0:
        return None

    if xmax <= xmin:
        xmax = xmin * 10.0

    return np.logspace(
        np.log10(xmin),
        np.log10(xmax),
        nbins,
    )


def format_current_axes():
    """
    Aplica formato final a la figura actual.
    """
    ax = plt.gca()

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which="major",
        alpha=0.25,
        linewidth=0.6,
    )

    ax.grid(
        True,
        which="minor",
        alpha=0.12,
        linewidth=0.4,
    )


def savefig(name):
    """
    Guarda figura en PNG con calidad de paper.
    """
    format_current_axes()
    plt.tight_layout()

    output_path = PLOTS_DIR / name

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def scatter_loglog(
    data,
    xcol,
    ycol,
    xlabel,
    ylabel,
    filename,
    good_data=None,
    x_threshold=None,
    y_threshold=None,
    x_threshold_label=None,
    y_threshold_label=None,
):
    """
    Scatter con ambos ejes logarítmicos.
    Solo usa puntos con x > 0 e y > 0.
    """
    if xcol not in data.columns or ycol not in data.columns:
        print(f"Skip {filename}: faltan {xcol} o {ycol}")
        return

    mask = (
        np.isfinite(data[xcol])
        & np.isfinite(data[ycol])
        & (data[xcol] > 0)
        & (data[ycol] > 0)
    )

    if mask.sum() == 0:
        print(f"Skip {filename}: no hay puntos positivos finitos")
        return

    plt.figure(figsize=(7, 5))

    plt.scatter(
        data.loc[mask, xcol],
        data.loc[mask, ycol],
        s=18,
        alpha=0.38,
        linewidths=0,
        rasterized=True,
        label=ALL_EVENTS_LABEL,
    )

    if good_data is not None and len(good_data) > 0:
        good_mask = (
            np.isfinite(good_data[xcol])
            & np.isfinite(good_data[ycol])
            & (good_data[xcol] > 0)
            & (good_data[ycol] > 0)
        )

        if good_mask.sum() > 0:
            plt.scatter(
                good_data.loc[good_mask, xcol],
                good_data.loc[good_mask, ycol],
                s=24,
                alpha=0.85,
                linewidths=0,
                rasterized=True,
                label=GOOD_NOPIE_LABEL,
            )

    if x_threshold is not None:
        plt.axvline(
            x_threshold,
            linestyle=":",
            label=x_threshold_label or THRESHOLD_LABEL,
        )

    if y_threshold is not None:
        plt.axhline(
            y_threshold,
            linestyle="--",
            label=y_threshold_label or THRESHOLD_LABEL,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handlelength=1.2)
    savefig(filename)


def scatter_logx(
    data,
    xcol,
    ycol,
    xlabel,
    ylabel,
    filename,
    good_data=None,
    x_threshold=None,
    x_threshold_label=None,
    y_zero_line=False,
    y_zero_line_label=None,
    ylim=None,
):
    """
    Scatter con eje x logarítmico y eje y lineal.
    """
    if xcol not in data.columns or ycol not in data.columns:
        print(f"Skip {filename}: faltan {xcol} o {ycol}")
        return

    mask = (
        np.isfinite(data[xcol])
        & np.isfinite(data[ycol])
        & (data[xcol] > 0)
    )

    if mask.sum() == 0:
        print(f"Skip {filename}: no hay puntos positivos finitos")
        return

    plt.figure(figsize=(7, 5))

    plt.scatter(
        data.loc[mask, xcol],
        data.loc[mask, ycol],
        s=18,
        alpha=0.38,
        linewidths=0,
        rasterized=True,
        label=ALL_EVENTS_LABEL,
    )

    if good_data is not None and len(good_data) > 0:
        good_mask = (
            np.isfinite(good_data[xcol])
            & np.isfinite(good_data[ycol])
            & (good_data[xcol] > 0)
        )

        if good_mask.sum() > 0:
            plt.scatter(
                good_data.loc[good_mask, xcol],
                good_data.loc[good_mask, ycol],
                s=24,
                alpha=0.85,
                linewidths=0,
                rasterized=True,
                label=GOOD_NOPIE_LABEL,
            )

    if y_zero_line:
        plt.axhline(
            0,
            linestyle="--",
            label=y_zero_line_label or ZERO_BIAS_LABEL,
        )

    if x_threshold is not None:
        plt.axvline(
            x_threshold,
            linestyle=":",
            label=x_threshold_label or THRESHOLD_LABEL,
        )

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handlelength=1.2)
    savefig(filename)


def scatter_symlogx_logy(
    data,
    xcol,
    ycol,
    xlabel,
    ylabel,
    filename,
    good_data=None,
    y_threshold=None,
    y_threshold_label=None,
    x_zero_line=True,
    x_zero_line_label=None,
    linthresh=1.0,
):
    """
    Scatter con eje x symlog y eje y log.

    Sirve para cantidades que pueden ser positivas o negativas,
    como Delta tE = tE_fit - tE_true.

    El eje y debe ser positivo.
    """
    if xcol not in data.columns or ycol not in data.columns:
        print(f"Skip {filename}: faltan {xcol} o {ycol}")
        return

    mask = (
        np.isfinite(data[xcol])
        & np.isfinite(data[ycol])
        & (data[ycol] > 0)
    )

    if mask.sum() == 0:
        print(f"Skip {filename}: no hay puntos válidos")
        return

    plt.figure(figsize=(7, 5))

    plt.scatter(
        data.loc[mask, xcol],
        data.loc[mask, ycol],
        s=18,
        alpha=0.38,
        linewidths=0,
        rasterized=True,
        label=ALL_EVENTS_LABEL,
    )

    if good_data is not None and len(good_data) > 0:
        good_mask = (
            np.isfinite(good_data[xcol])
            & np.isfinite(good_data[ycol])
            & (good_data[ycol] > 0)
        )

        if good_mask.sum() > 0:
            plt.scatter(
                good_data.loc[good_mask, xcol],
                good_data.loc[good_mask, ycol],
                s=24,
                alpha=0.85,
                linewidths=0,
                rasterized=True,
                label=GOOD_NOPIE_LABEL,
            )

    if x_zero_line:
        plt.axvline(
            0,
            linestyle="--",
            label=x_zero_line_label or ZERO_LINE_LABEL,
        )

    if y_threshold is not None:
        plt.axhline(
            y_threshold,
            linestyle="--",
            label=y_threshold_label or THRESHOLD_LABEL,
        )

    plt.xscale("symlog", linthresh=linthresh)
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handlelength=1.2)
    savefig(filename)


def hist_log(
    data,
    column,
    xlabel,
    filename,
    threshold=None,
    nbins=30,
):
    """
    Histograma con bins logarítmicos y eje x logarítmico.
    Solo usa valores positivos.
    """
    if column not in data.columns:
        print(f"Skip {filename}: no existe {column}")
        return

    x = positive_finite_series(data[column])
    bins = log_bins(x, nbins=nbins)

    if bins is None or len(x) == 0:
        print(f"Skip {filename}: no hay datos positivos")
        return

    plt.figure(figsize=(7, 5))
    plt.hist(x, bins=bins, histtype="stepfilled", alpha=0.75, linewidth=1.2)

    if threshold is not None and threshold > 0:
        plt.axvline(
            threshold,
            linestyle="--",
            label=f"Threshold = {threshold:g}",
        )
        plt.legend(handlelength=1.2)

    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Number of events")
    savefig(filename)



# ============================================================
# Fracción de eventos con Delta chi2 detectable en bines de tE
# ============================================================

def make_log_edges(x, nbins):
    """
    Construye bordes logarítmicos robustos para una cantidad positiva.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]

    if len(x) == 0:
        return None

    xmin = np.nanmin(x)
    xmax = np.nanmax(x)

    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return None

    if xmin <= 0:
        return None

    if xmax <= xmin:
        xmax = xmin * 10.0

    return np.logspace(
        np.log10(xmin),
        np.log10(xmax),
        nbins + 1,
    )


def add_delta_chi2_detectable_flag(df):
    """
    Agrega columnas booleanas para diagnosticar detectabilidad de paralaje.

    Definición adoptada:
        detectable = Delta chi2 > DETECTABLE_DELTA_CHI2_MIN

    Si REQUIRE_GOOD_TRUE_CHI2_FOR_DETECTABLE=True, el denominador de la
    fracción se restringe a eventos con chi2_true/N_data consistente.
    """
    df = df.copy()

    df["delta_chi2_detectable_raw"] = (
        np.isfinite(df["delta_chi2_true"])
        & (df["delta_chi2_true"] > DETECTABLE_DELTA_CHI2_MIN)
    )

    if REQUIRE_GOOD_TRUE_CHI2_FOR_DETECTABLE and "good_true_chi2" in df.columns:
        df["detectable_denominator"] = (
            df["good_true_chi2"].astype(bool)
            & np.isfinite(df["delta_chi2_true"])
        )
    else:
        df["detectable_denominator"] = np.isfinite(df["delta_chi2_true"])

    df["delta_chi2_detectable"] = (
        df["detectable_denominator"].astype(bool)
        & df["delta_chi2_detectable_raw"].astype(bool)
    )

    return df


def build_detectable_fraction_vs_tE_table(data, te_edges, group_label="all"):
    """
    Construye tabla de fracción detectable en bines de tE.
    """
    tmp = data[
        np.isfinite(data["tE_true_used"])
        & (data["tE_true_used"] > 0)
        & data["detectable_denominator"].astype(bool)
    ].copy()

    if len(tmp) == 0:
        return pd.DataFrame()

    tmp["tE_bin"] = pd.cut(
        tmp["tE_true_used"],
        bins=te_edges,
        include_lowest=True,
    )

    grouped = tmp.groupby(
        "tE_bin",
        observed=True,
    ).agg(
        n_total=("delta_chi2_detectable", "size"),
        n_detectable=("delta_chi2_detectable", "sum"),
        median_tE=("tE_true_used", "median"),
        median_delta_chi2=("delta_chi2_true", "median"),
        median_delta_chi2_per_point=("delta_chi2_true_per_point", "median"),
        median_chi2_dof_nopie=("chi2_dof_nopie", "median"),
    )

    grouped = grouped[grouped["n_total"] > 0].copy()

    if len(grouped) == 0:
        return pd.DataFrame()

    grouped["fraction_detectable"] = (
        grouped["n_detectable"] / grouped["n_total"]
    )

    # Error binomial simple. Para n chico sirve como diagnóstico rápido.
    grouped["fraction_err"] = np.sqrt(
        grouped["fraction_detectable"]
        * (1.0 - grouped["fraction_detectable"])
        / grouped["n_total"]
    )

    grouped["tE_min"] = [interval.left for interval in grouped.index]
    grouped["tE_max"] = [interval.right for interval in grouped.index]
    grouped["group"] = group_label

    grouped = grouped.reset_index(drop=True)

    cols = [
        "group",
        "tE_min",
        "tE_max",
        "median_tE",
        "n_total",
        "n_detectable",
        "fraction_detectable",
        "fraction_err",
        "median_delta_chi2",
        "median_delta_chi2_per_point",
        "median_chi2_dof_nopie",
    ]

    return grouped[cols]


def plot_detectable_fraction_vs_tE_combined(df):
    """
    Grafica la fracción de eventos con Delta chi2 detectable en bines de tE,
    usando todos los campos combinados.
    """
    needed = [
        "tE_true_used",
        "delta_chi2_detectable",
        "detectable_denominator",
    ]

    for col in needed:
        if col not in df.columns:
            print(f"Skip binned_fraction_detectable_delta_chi2_vs_tE: falta {col}")
            return pd.DataFrame()

    valid = df[
        np.isfinite(df["tE_true_used"])
        & (df["tE_true_used"] > 0)
        & df["detectable_denominator"].astype(bool)
    ].copy()

    if len(valid) == 0:
        print("Skip binned_fraction_detectable_delta_chi2_vs_tE: no hay datos válidos.")
        return pd.DataFrame()

    te_edges = make_log_edges(
        valid["tE_true_used"],
        N_TE_DETECTABLE_BINS,
    )

    if te_edges is None:
        print("Skip binned_fraction_detectable_delta_chi2_vs_tE: no pude definir bins.")
        return pd.DataFrame()

    table = build_detectable_fraction_vs_tE_table(
        valid,
        te_edges,
        group_label="all",
    )

    if len(table) == 0:
        print("Skip binned_fraction_detectable_delta_chi2_vs_tE: bins vacíos.")
        return pd.DataFrame()

    plt.figure(figsize=(7.2, 5.2))

    plt.errorbar(
        table["median_tE"],
        table["fraction_detectable"],
        yerr=table["fraction_err"],
        marker="o",
        linestyle="-",
        capsize=3,
        label=rf"$\Delta\chi^2>{DETECTABLE_DELTA_CHI2_MIN:g}$",
    )

    for _, row in table.iterrows():
        if int(row["n_total"]) >= MIN_EVENTS_PER_TE_BIN_TO_ANNOTATE:
            plt.annotate(
                f"n={int(row['n_total'])}",
                xy=(
                    row["median_tE"],
                    row["fraction_detectable"],
                ),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xscale("log")
    plt.ylim(-0.03, 1.03)
    plt.xlabel(r"$t_{E,\mathrm{true}}$ [d]")
    plt.ylabel(r"Detectable $\Delta\chi^2$ fraction")
    plt.title(r"Fraction of events with detectable $\Delta\chi^2$ vs $t_E$")
    plt.legend(handlelength=1.5)

    savefig("binned_fraction_detectable_delta_chi2_vs_tE.png")

    table.to_csv(
        TABLES_DIR / "binned_fraction_detectable_delta_chi2_vs_tE.csv",
        index=False,
    )

    table.to_parquet(
        TABLES_DIR / "binned_fraction_detectable_delta_chi2_vs_tE.parquet",
        index=False,
    )

    return table


def plot_detectable_fraction_vs_tE_by_field(df):
    """
    Grafica la fracción detectable por bines de tE separando por campo,
    si existe una columna de campo.
    """
    field_col = safe_col(
        df,
        [
            "field_name",
            "field_name_true",
            "field",
            "field_true",
        ],
    )

    if field_col is None:
        print("Skip binned_fraction_detectable_delta_chi2_vs_tE_by_field: no hay columna de campo.")
        return pd.DataFrame()

    if "tE_true_used" not in df.columns:
        print("Skip binned_fraction_detectable_delta_chi2_vs_tE_by_field: falta tE_true_used.")
        return pd.DataFrame()

    valid = df[
        np.isfinite(df["tE_true_used"])
        & (df["tE_true_used"] > 0)
        & df["detectable_denominator"].astype(bool)
    ].copy()

    if len(valid) == 0:
        print("Skip binned_fraction_detectable_delta_chi2_vs_tE_by_field: no hay datos válidos.")
        return pd.DataFrame()

    te_edges = make_log_edges(
        valid["tE_true_used"],
        N_TE_DETECTABLE_BINS,
    )

    if te_edges is None:
        print("Skip binned_fraction_detectable_delta_chi2_vs_tE_by_field: no pude definir bins.")
        return pd.DataFrame()

    all_tables = []

    plt.figure(figsize=(7.8, 5.6))

    for field_value, g in valid.groupby(field_col):
        table = build_detectable_fraction_vs_tE_table(
            g,
            te_edges,
            group_label=str(field_value),
        )

        if len(table) == 0:
            continue

        all_tables.append(table)

        plt.errorbar(
            table["median_tE"],
            table["fraction_detectable"],
            yerr=table["fraction_err"],
            marker="o",
            linestyle="-",
            capsize=2,
            alpha=0.85,
            label=str(field_value),
        )

    if len(all_tables) == 0:
        plt.close()
        print("Skip binned_fraction_detectable_delta_chi2_vs_tE_by_field: tablas vacías.")
        return pd.DataFrame()

    full_table = pd.concat(
        all_tables,
        ignore_index=True,
    )

    plt.xscale("log")
    plt.ylim(-0.03, 1.03)
    plt.xlabel(r"$t_{E,\mathrm{true}}$ [d]")
    plt.ylabel(r"Detectable $\Delta\chi^2$ fraction")
    plt.title(r"Detectable $\Delta\chi^2$ fraction vs $t_E$ by field")
    plt.legend(handlelength=1.3, ncol=2)

    savefig("binned_fraction_detectable_delta_chi2_vs_tE_by_field.png")

    full_table.to_csv(
        TABLES_DIR / "binned_fraction_detectable_delta_chi2_vs_tE_by_field.csv",
        index=False,
    )

    full_table.to_parquet(
        TABLES_DIR / "binned_fraction_detectable_delta_chi2_vs_tE_by_field.parquet",
        index=False,
    )

    return full_table


def add_detectable_delta_chi2_vs_tE_plots(df):
    """
    Agrega diagnósticos de fracción detectable en función de tE.
    """
    df = add_delta_chi2_detectable_flag(df)

    denom = int(df["detectable_denominator"].sum())
    ndet = int(df["delta_chi2_detectable"].sum())

    print("=" * 80)
    print("Detectable Delta chi2 diagnostic")
    print("=" * 80)
    print(f"Threshold: Delta chi2 > {DETECTABLE_DELTA_CHI2_MIN:g}")
    print(
        "Require good_true_chi2 denominator:",
        REQUIRE_GOOD_TRUE_CHI2_FOR_DETECTABLE,
    )
    print(f"N denominator: {denom}")
    print(f"N detectable:  {ndet}")

    if denom > 0:
        print(f"Fraction detectable: {ndet / denom:.4f}")

    print("=" * 80)

    plot_detectable_fraction_vs_tE_combined(df)
    plot_detectable_fraction_vs_tE_by_field(df)

    return df



# ============================================================
# Leer resultados
# ============================================================

true = read_result_kind(RESULTS_DIR, "true")
fit = read_result_kind(RESULTS_DIR, "fit_rr")

fit = normalize_fit_columns(fit)

print("=" * 80)
print("Columnas TRUE")
print(true.columns.tolist())
print("=" * 80)
print("Columnas FIT_RR")
print(fit.columns.tolist())
print("=" * 80)

id_cols = choose_id_columns(true, fit)

print("Merge usando:", id_cols)

df = pd.merge(
    true,
    fit,
    on=id_cols,
    how="inner",
    suffixes=("_true", "_fit"),
)

if len(df) == 0:
    raise RuntimeError("El merge true-fit quedó vacío.")

print(f"Eventos en true: {len(true)}")
print(f"Eventos en fit:  {len(fit)}")
print(f"Eventos merge:   {len(df)}")


# ============================================================
# Columnas principales
# ============================================================

chi2_col = safe_col(df, ["chi2", "chi2_fit", "chichi", "chichi_fit"])
dof_col = safe_col(df, ["dof", "dof_fit"])
chi2_dof_col = safe_col(df, ["chi2_dof", "chi2_dof_fit"])

chi2_true_col = safe_col(
    df,
    ["chi2_true", "chi2_true_true", "chi2_true_fit"],
)

n_data_true_col = safe_col(
    df,
    ["n_data_true", "n_data_true_true", "n_data_true_fit"],
)

if chi2_true_col is None:
    raise KeyError(
        "No encuentro chi2_true. Revisá que extract_data_event lo esté guardando."
    )

df["chi2_nopie"] = df[chi2_col]
df["chi2_fit"] = df[chi2_col]
df["chi2_true_model"] = df[chi2_true_col]
df["delta_chi2_true"] = df["chi2_nopie"] - df["chi2_true_model"]

if dof_col is not None:
    df["dof_nopie"] = df[dof_col]
else:
    df["dof_nopie"] = np.nan

if chi2_dof_col is not None:
    df["chi2_dof_nopie"] = df[chi2_dof_col]
else:
    df["chi2_dof_nopie"] = df["chi2_nopie"] / df["dof_nopie"]

df["chi2_dof_fit"] = df["chi2_dof_nopie"]

if n_data_true_col is not None:
    df["n_data_true_model"] = df[n_data_true_col]
else:
    df["n_data_true_model"] = np.nan

df["delta_chi2_true_per_point"] = (
    df["delta_chi2_true"] / df["n_data_true_model"]
)

df["chi2_true_per_point"] = (
    df["chi2_true_model"] / df["n_data_true_model"]
)


# ============================================================
# Biases
# ============================================================

for par in ["t0", "tE"]:

    true_col = safe_col(df, [f"{par}_true"])
    fit_col = safe_col(df, [f"{par}_fit", par])

    if true_col is not None and fit_col is not None:

        df[f"bias_{par}"] = df[fit_col] - df[true_col]

        df[f"frac_bias_{par}"] = (
            df[f"bias_{par}"]
            / df[true_col].replace(0, np.nan)
        )

        df[f"{par}_true_used"] = df[true_col]
        df[f"{par}_fit_used"] = df[fit_col]


# ============================================================
# Diferencias específicas en tE
# ============================================================

if "tE_true_used" in df.columns and "tE_fit_used" in df.columns:

    df["delta_tE_fit_minus_true"] = (
        df["tE_fit_used"]
        - df["tE_true_used"]
    )

    df["abs_delta_tE_fit_minus_true"] = np.abs(
        df["delta_tE_fit_minus_true"]
    )

    df["frac_delta_tE_fit_minus_true"] = (
        df["delta_tE_fit_minus_true"]
        / df["tE_true_used"].replace(0, np.nan)
    )

    df["abs_frac_delta_tE_fit_minus_true"] = np.abs(
        df["frac_delta_tE_fit_minus_true"]
    )


# Para u0 comparamos valor absoluto
u0_true_col = safe_col(df, ["u0_true"])
u0_fit_col = safe_col(df, ["u0_fit", "u0"])

if u0_true_col is not None and u0_fit_col is not None:

    df["absu0_true_used"] = np.abs(df[u0_true_col])
    df["absu0_fit_used"] = np.abs(df[u0_fit_col])

    df["bias_absu0"] = (
        df["absu0_fit_used"]
        - df["absu0_true_used"]
    )

    df["frac_bias_absu0"] = (
        df["bias_absu0"]
        / df["absu0_true_used"].replace(0, np.nan)
    )


# piE true
piE_true_col = safe_col(df, ["piE_true", "piE_true_true"])

if piE_true_col is not None:
    df["piE_true_used"] = df[piE_true_col]


# mass true
mass_true_col = safe_col(
    df,
    ["mass_true", "mass_true_true", "mass"],
)

if mass_true_col is not None:
    df["mass_true_used"] = df[mass_true_col]


# ============================================================
# Definir eventos bien ajustados
# ============================================================

df["good_true_chi2"] = (
    np.isfinite(df["chi2_true_per_point"])
    & (df["chi2_true_per_point"] > CHI2_TRUE_PER_POINT_MIN)
    & (df["chi2_true_per_point"] < CHI2_TRUE_PER_POINT_MAX)
)

df["good_nopie"] = (
    df["good_true_chi2"]
    & np.isfinite(df["delta_chi2_true"])
    & np.isfinite(df["delta_chi2_true_per_point"])
    & np.isfinite(df["chi2_dof_nopie"])
    & (df["chi2_dof_nopie"] < CHI2_DOF_MAX)
    & (df["delta_chi2_true_per_point"] < DELTA_CHI2_PER_POINT_MAX)
)

df["good_nopie_strict"] = (
    df["good_nopie"]
    & (df["delta_chi2_true"] < DELTA_CHI2_MAX)
)

df["nopie_fails"] = (
    df["good_true_chi2"]
    & (
        (df["chi2_dof_nopie"] > CHI2_DOF_MAX)
        | (df["delta_chi2_true_per_point"] > 1.0)
    )
)

good = df[df["good_nopie"]].copy()
good_strict = df[df["good_nopie_strict"]].copy()

# ============================================================
# Fracción de eventos con Delta chi2 detectable en bines de tE
# ============================================================

df = add_detectable_delta_chi2_vs_tE_plots(df)

# Recalcular subconjuntos después de agregar columnas nuevas.
good = df[df["good_nopie"]].copy()
good_strict = df[df["good_nopie_strict"]].copy()

# ============================================================
# Prints de diagnóstico
# ============================================================

print("=" * 80)
print(f"Eventos totales mergeados: {len(df)}")
print(f"Eventos con chi2_true consistente: {df['good_true_chi2'].sum()}")
print(f"Good No pi_E events: {len(good)}")
print(f"Strict good No pi_E events: {len(good_strict)}")
print(f"Good No pi_E fraction: {len(good) / len(df):.3f}")
print("=" * 80)

cols_print = (
    id_cols
    + [
        "chi2_true_model",
        "n_data_true_model",
        "chi2_true_per_point",
        "chi2_nopie",
        "dof_nopie",
        "chi2_dof_nopie",
        "delta_chi2_true",
        "delta_chi2_true_per_point",
        "good_nopie",
        "good_nopie_strict",
    ]
)

if "piE_true_used" in df.columns:
    cols_print += ["piE_true_used"]

if "tE_true_used" in df.columns:
    cols_print += ["tE_true_used"]

if "tE_fit_used" in df.columns:
    cols_print += ["tE_fit_used"]

if "delta_tE_fit_minus_true" in df.columns:
    cols_print += [
        "delta_tE_fit_minus_true",
        "frac_delta_tE_fit_minus_true",
    ]

cols_print = [c for c in cols_print if c in df.columns]

print(df[cols_print].head(20))

print("=" * 80)
print("Resumen chi2_true/N_data")
print(df["chi2_true_per_point"].describe())
print("=" * 80)

if "n_data_true_model" in df.columns:
    print("Resumen N_data")
    print(df["n_data_true_model"].describe())
    print("=" * 80)

if "piE_true_used" in df.columns:
    print("Resumen piE_true")
    print(df["piE_true_used"].describe())
    print("=" * 80)

if "tE_true_used" in df.columns:
    print("Resumen tE_true")
    print(df["tE_true_used"].describe())
    print("=" * 80)

if "tE_fit_used" in df.columns:
    print("Resumen tE_fit")
    print(df["tE_fit_used"].describe())
    print("=" * 80)

if "delta_tE_fit_minus_true" in df.columns:
    print("Resumen Delta tE = tE_fit - tE_true")
    print(df["delta_tE_fit_minus_true"].describe())
    print("=" * 80)

print("Eventos con menor Delta chi2 por punto")
print(
    df[cols_print]
    .sort_values("delta_chi2_true_per_point")
    .head(20)
)

print("=" * 80)
print("Eventos con mayor Delta chi2 por punto")
print(
    df[cols_print]
    .sort_values("delta_chi2_true_per_point", ascending=False)
    .head(20)
)


# ============================================================
# Guardar tablas diagnóstico
# ============================================================

df.to_parquet(
    TABLES_DIR / "diagnostico_NoPiE_vs_true.parquet",
    index=False,
)

df.to_csv(
    TABLES_DIR / "diagnostico_NoPiE_vs_true.csv",
    index=False,
)

good.to_parquet(
    TABLES_DIR / "diagnostico_NoPiE_vs_true_good.parquet",
    index=False,
)

good.to_csv(
    TABLES_DIR / "diagnostico_NoPiE_vs_true_good.csv",
    index=False,
)

good_strict.to_parquet(
    TABLES_DIR / "diagnostico_NoPiE_vs_true_good_strict.parquet",
    index=False,
)

good_strict.to_csv(
    TABLES_DIR / "diagnostico_NoPiE_vs_true_good_strict.csv",
    index=False,
)


# ============================================================
# Plots: histogramas chi2 en escala log
# ============================================================

hist_log(
    df,
    "delta_chi2_true",
    r"$\Delta\chi^2=\chi^2_{\mathrm{No}\,\pi_E}-\chi^2_{\mathrm{true\ model}}$",
    "hist_delta_chi2_true_log.png",
    threshold=DELTA_CHI2_MAX,
    nbins=30,
)

hist_log(
    df,
    "delta_chi2_true_per_point",
    r"$\Delta\chi^2/N_{\rm data}$",
    "hist_delta_chi2_true_per_point_log.png",
    threshold=DELTA_CHI2_PER_POINT_MAX,
    nbins=30,
)

hist_log(
    df,
    "chi2_dof_nopie",
    r"$\chi^2_{\mathrm{No}\,\pi_E}/{\rm dof}$",
    "hist_chi2_dof_nopie_log.png",
    threshold=CHI2_DOF_MAX,
    nbins=30,
)

hist_log(
    df,
    "chi2_true_per_point",
    r"$\chi^2_{\mathrm{true\ model}}/N_{\rm data}$",
    "hist_chi2_true_per_point_log.png",
    threshold=1.0,
    nbins=30,
)

hist_log(
    df,
    "chi2_true_model",
    r"$\chi^2_{\mathrm{true\ model}}$",
    "hist_chi2_true_log.png",
    threshold=None,
    nbins=30,
)

hist_log(
    df,
    "chi2_nopie",
    r"$\chi^2_{\mathrm{No}\,\pi_E}$",
    "hist_chi2_nopie_log.png",
    threshold=None,
    nbins=30,
)


# ============================================================
# chi2 NoPiE vs chi2 true, log-log
# ============================================================

mask_chi2 = (
    np.isfinite(df["chi2_true_model"])
    & np.isfinite(df["chi2_nopie"])
    & (df["chi2_true_model"] > 0)
    & (df["chi2_nopie"] > 0)
)

if mask_chi2.sum() > 0:

    plt.figure(figsize=(7, 5))

    plt.scatter(
        df.loc[mask_chi2, "chi2_true_model"],
        df.loc[mask_chi2, "chi2_nopie"],
        s=18,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
        label=ALL_EVENTS_LABEL,
    )

    good_mask = (
        mask_chi2
        & df["good_nopie"]
    )

    if good_mask.sum() > 0:
        plt.scatter(
            df.loc[good_mask, "chi2_true_model"],
            df.loc[good_mask, "chi2_nopie"],
            s=24,
            alpha=0.85,
            linewidths=0,
            rasterized=True,
            label=GOOD_NOPIE_LABEL,
        )

    vmin = np.nanmin(
        [
            df.loc[mask_chi2, "chi2_true_model"].min(),
            df.loc[mask_chi2, "chi2_nopie"].min(),
        ]
    )

    vmax = np.nanmax(
        [
            df.loc[mask_chi2, "chi2_true_model"].max(),
            df.loc[mask_chi2, "chi2_nopie"].max(),
        ]
    )

    plt.plot(
        [vmin, vmax],
        [vmin, vmax],
        linestyle="--",
        label=r"$\chi^2_{\mathrm{No}\,\pi_E}=\chi^2_{\mathrm{true\ model}}$",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\chi^2_{\mathrm{true\ model}}$")
    plt.ylabel(r"$\chi^2_{\mathrm{No}\,\pi_E}$")
    plt.legend(handlelength=1.2)
    savefig("chi2_nopie_vs_chi2_true_loglog.png")


# ============================================================
# chi2 del fit NoPiE vs piE true
# ============================================================

if "piE_true_used" in df.columns:

    scatter_loglog(
        df,
        "piE_true_used",
        "chi2_nopie",
        r"$\pi_E$ true",
        r"$\chi^2_{\mathrm{No}\,\pi_E}$",
        "chi2_nopie_vs_piE_true_loglog.png",
        good_data=good,
    )

    scatter_loglog(
        df,
        "piE_true_used",
        "chi2_dof_nopie",
        r"$\pi_E$ true",
        r"$\chi^2_{\mathrm{No}\,\pi_E}/{\rm dof}$",
        "chi2_dof_nopie_vs_piE_true_loglog.png",
        good_data=good,
        y_threshold=CHI2_DOF_MAX,
    )

    scatter_loglog(
        df,
        "piE_true_used",
        "chi2_fit",
        r"$\pi_E$ true",
        r"$\chi^2_{\rm fit}$",
        "chi2_fit_vs_piE_true_loglog.png",
        good_data=good,
    )

    scatter_loglog(
        df,
        "piE_true_used",
        "chi2_dof_fit",
        r"$\pi_E$ true",
        r"$\chi^2_{\rm fit}/{\rm dof}$",
        "chi2_dof_fit_vs_piE_true_loglog.png",
        good_data=good,
        y_threshold=CHI2_DOF_MAX,
    )

    scatter_loglog(
        df,
        "piE_true_used",
        "chi2_true_model",
        r"$\pi_E$ true",
        r"$\chi^2_{\mathrm{true\ model}}$",
        "chi2_true_vs_piE_true_loglog.png",
        good_data=good,
    )

    scatter_loglog(
        df,
        "piE_true_used",
        "chi2_true_per_point",
        r"$\pi_E$ true",
        r"$\chi^2_{\mathrm{true\ model}}/N_{\rm data}$",
        "chi2_true_per_point_vs_piE_true_loglog.png",
        good_data=good,
        y_threshold=1.0,
    )



# ============================================================
# Diagnostics involving number of data points
# ============================================================

if "piE_true_used" in df.columns and "n_data_true_model" in df.columns:

    scatter_loglog(
        df,
        "piE_true_used",
        "n_data_true_model",
        r"$\pi_E$ true",
        r"$N_{\rm data}$",
        "n_data_vs_piE_true_loglog.png",
        good_data=good,
    )


if "n_data_true_model" in df.columns:

    scatter_loglog(
        df,
        "n_data_true_model",
        "delta_chi2_true_per_point",
        r"$N_{\rm data}$",
        r"$\Delta\chi^2/N_{\rm data}$",
        "delta_chi2_per_point_vs_n_data_loglog.png",
        good_data=good,
        y_threshold=DELTA_CHI2_PER_POINT_MAX,
        y_threshold_label=r"Adopted threshold",
    )

    scatter_loglog(
        df,
        "n_data_true_model",
        "chi2_true_per_point",
        r"$N_{\rm data}$",
        r"$\chi^2_{\mathrm{true\ model}}/N_{\rm data}$",
        "chi2_true_per_point_vs_n_data_loglog.png",
        good_data=good,
        y_threshold=1.0,
        y_threshold_label=r"Expected value",
    )

    scatter_loglog(
        df,
        "n_data_true_model",
        "chi2_dof_nopie",
        r"$N_{\rm data}$",
        r"$\chi^2_{\mathrm{No}\,\pi_E}/{\rm dof}$",
        "chi2_dof_nopie_vs_n_data_loglog.png",
        good_data=good,
        y_threshold=CHI2_DOF_MAX,
        y_threshold_label=r"Adopted threshold",
    )


# ============================================================
# Fraction of Good No pi_E events vs number of data points
# ============================================================

if "n_data_true_model" in df.columns:

    tmp = df[
        np.isfinite(df["n_data_true_model"])
        & (df["n_data_true_model"] > 0)
        & np.isfinite(df["good_nopie"])
    ].copy()

    if len(tmp) > 0 and tmp["n_data_true_model"].nunique() > 1:

        n_min = tmp["n_data_true_model"].min()
        n_max = tmp["n_data_true_model"].max()

        bins = np.logspace(
            np.log10(n_min),
            np.log10(n_max),
            8,
        )

        # Avoid duplicated bin edges for very small or discrete samples.
        bins = np.unique(bins)

        if len(bins) >= 3:

            tmp["n_data_bin"] = pd.cut(
                tmp["n_data_true_model"],
                bins=bins,
                include_lowest=True,
            )

            grouped = tmp.groupby("n_data_bin", observed=True).agg(
                n_events=("good_nopie", "size"),
                n_good=("good_nopie", "sum"),
                median_n_data=("n_data_true_model", "median"),
            )

            grouped = grouped[grouped["n_events"] > 0].copy()

            if len(grouped) > 0:

                grouped["fraction_good"] = (
                    grouped["n_good"] / grouped["n_events"]
                )

                grouped["fraction_good_err"] = np.sqrt(
                    grouped["fraction_good"]
                    * (1.0 - grouped["fraction_good"])
                    / grouped["n_events"]
                )

                plt.figure(figsize=(7, 5))

                plt.errorbar(
                    grouped["median_n_data"],
                    grouped["fraction_good"],
                    yerr=grouped["fraction_good_err"],
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    label=GOOD_NOPIE_LABEL + " fraction",
                )

                plt.xscale("log")
                plt.xlabel(r"$N_{\rm data}$")
                plt.ylabel(GOOD_NOPIE_LABEL + " fraction")
                plt.ylim(-0.03, 1.03)
                plt.legend(handlelength=1.5)

                savefig("fraction_good_nopie_vs_n_data.png")

# ============================================================
# chi2 del fit NoPiE vs tE true y tE fit
# ============================================================

if "tE_true_used" in df.columns:

    scatter_loglog(
        df,
        "tE_true_used",
        "chi2_nopie",
        r"$t_{E,\mathrm{true}}$ [d]",
        r"$\chi^2_{\mathrm{No}\,\pi_E}$",
        "chi2_nopie_vs_tE_true_loglog.png",
        good_data=good,
    )

    scatter_loglog(
        df,
        "tE_true_used",
        "chi2_dof_nopie",
        r"$t_{E,\mathrm{true}}$ [d]",
        r"$\chi^2_{\mathrm{No}\,\pi_E}/{\rm dof}$",
        "chi2_dof_nopie_vs_tE_true_loglog.png",
        good_data=good,
        y_threshold=CHI2_DOF_MAX,
    )

    scatter_loglog(
        df,
        "tE_true_used",
        "chi2_fit",
        r"$t_{E,\mathrm{true}}$ [d]",
        r"$\chi^2_{\rm fit}$",
        "chi2_fit_vs_tE_true_loglog.png",
        good_data=good,
    )

    scatter_loglog(
        df,
        "tE_true_used",
        "chi2_dof_fit",
        r"$t_{E,\mathrm{true}}$ [d]",
        r"$\chi^2_{\rm fit}/{\rm dof}$",
        "chi2_dof_fit_vs_tE_true_loglog.png",
        good_data=good,
        y_threshold=CHI2_DOF_MAX,
    )


if "tE_fit_used" in df.columns:

    scatter_loglog(
        df,
        "tE_fit_used",
        "chi2_nopie",
        r"$t_{E,\mathrm{fit}}$ [d]",
        r"$\chi^2_{\mathrm{No}\,\pi_E}$",
        "chi2_nopie_vs_tE_fit_loglog.png",
        good_data=good,
    )

    scatter_loglog(
        df,
        "tE_fit_used",
        "chi2_dof_nopie",
        r"$t_{E,\mathrm{fit}}$ [d]",
        r"$\chi^2_{\mathrm{No}\,\pi_E}/{\rm dof}$",
        "chi2_dof_nopie_vs_tE_fit_loglog.png",
        good_data=good,
        y_threshold=CHI2_DOF_MAX,
    )

    scatter_loglog(
        df,
        "tE_fit_used",
        "chi2_fit",
        r"$t_{E,\mathrm{fit}}$ [d]",
        r"$\chi^2_{\rm fit}$",
        "chi2_fit_vs_tE_fit_loglog.png",
        good_data=good,
    )

    scatter_loglog(
        df,
        "tE_fit_used",
        "chi2_dof_fit",
        r"$t_{E,\mathrm{fit}}$ [d]",
        r"$\chi^2_{\rm fit}/{\rm dof}$",
        "chi2_dof_fit_vs_tE_fit_loglog.png",
        good_data=good,
        y_threshold=CHI2_DOF_MAX,
    )


# ============================================================
# Delta chi2 vs parámetros verdaderos y ajustados
# ============================================================

if "tE_true_used" in df.columns:

    scatter_loglog(
        df,
        "tE_true_used",
        "delta_chi2_true",
        r"$t_{E,\mathrm{true}}$ [d]",
        r"$\Delta\chi^2$",
        "delta_chi2_true_vs_tE_true_loglog.png",
        good_data=good,
        y_threshold=DELTA_CHI2_MAX,
    )

    scatter_loglog(
        df,
        "tE_true_used",
        "delta_chi2_true_per_point",
        r"$t_{E,\mathrm{true}}$ [d]",
        r"$\Delta\chi^2/N_{\rm data}$",
        "delta_chi2_true_per_point_vs_tE_true_loglog.png",
        good_data=good,
        y_threshold=DELTA_CHI2_PER_POINT_MAX,
    )


if "tE_fit_used" in df.columns:

    scatter_loglog(
        df,
        "tE_fit_used",
        "delta_chi2_true",
        r"$t_{E,\mathrm{fit}}$ [d]",
        r"$\Delta\chi^2$",
        "delta_chi2_true_vs_tE_fit_loglog.png",
        good_data=good,
        y_threshold=DELTA_CHI2_MAX,
    )

    scatter_loglog(
        df,
        "tE_fit_used",
        "delta_chi2_true_per_point",
        r"$t_{E,\mathrm{fit}}$ [d]",
        r"$\Delta\chi^2/N_{\rm data}$",
        "delta_chi2_true_per_point_vs_tE_fit_loglog.png",
        good_data=good,
        y_threshold=DELTA_CHI2_PER_POINT_MAX,
    )


if "piE_true_used" in df.columns:

    scatter_loglog(
        df,
        "piE_true_used",
        "delta_chi2_true",
        r"$\pi_E$ true",
        r"$\Delta\chi^2$",
        "delta_chi2_true_vs_piE_true_loglog.png",
        good_data=good,
        y_threshold=DELTA_CHI2_MAX,
    )

    scatter_loglog(
        df,
        "piE_true_used",
        "delta_chi2_true_per_point",
        r"$\pi_E$ true",
        r"$\Delta\chi^2/N_{\rm data}$",
        "delta_chi2_true_per_point_vs_piE_true_loglog.png",
        good_data=good,
        y_threshold=DELTA_CHI2_PER_POINT_MAX,
    )


if "mass_true_used" in df.columns:

    scatter_loglog(
        df,
        "mass_true_used",
        "delta_chi2_true",
        r"$M_{\rm lens}$ true [$M_\odot$]",
        r"$\Delta\chi^2$",
        "delta_chi2_true_vs_mass_true_loglog.png",
        good_data=good,
        y_threshold=DELTA_CHI2_MAX,
    )

    scatter_loglog(
        df,
        "mass_true_used",
        "delta_chi2_true_per_point",
        r"$M_{\rm lens}$ true [$M_\odot$]",
        r"$\Delta\chi^2/N_{\rm data}$",
        "delta_chi2_true_per_point_vs_mass_true_loglog.png",
        good_data=good,
        y_threshold=DELTA_CHI2_PER_POINT_MAX,
    )


# ============================================================
# Delta chi2 vs Delta tE = tE_fit - tE_true
# ============================================================

if "delta_tE_fit_minus_true" in df.columns:

    scatter_symlogx_logy(
        df,
        "delta_tE_fit_minus_true",
        "delta_chi2_true",
        r"$\Delta t_E = t_{E,\mathrm{fit}}-t_{E,\mathrm{true}}$ [d]",
        r"$\Delta\chi^2$",
        "delta_chi2_true_vs_delta_tE_symlogx_logy.png",
        good_data=good,
        y_threshold=DELTA_CHI2_MAX,
        linthresh=1.0,
    )

    scatter_symlogx_logy(
        df,
        "delta_tE_fit_minus_true",
        "delta_chi2_true_per_point",
        r"$\Delta t_E = t_{E,\mathrm{fit}}-t_{E,\mathrm{true}}$ [d]",
        r"$\Delta\chi^2/N_{\rm data}$",
        "delta_chi2_true_per_point_vs_delta_tE_symlogx_logy.png",
        good_data=good,
        y_threshold=DELTA_CHI2_PER_POINT_MAX,
        linthresh=1.0,
    )

    scatter_loglog(
        df,
        "abs_delta_tE_fit_minus_true",
        "delta_chi2_true",
        r"$|\Delta t_E| = |t_{E,\mathrm{fit}}-t_{E,\mathrm{true}}|$ [d]",
        r"$\Delta\chi^2$",
        "delta_chi2_true_vs_abs_delta_tE_loglog.png",
        good_data=good,
        y_threshold=DELTA_CHI2_MAX,
    )

    scatter_loglog(
        df,
        "abs_delta_tE_fit_minus_true",
        "delta_chi2_true_per_point",
        r"$|\Delta t_E| = |t_{E,\mathrm{fit}}-t_{E,\mathrm{true}}|$ [d]",
        r"$\Delta\chi^2/N_{\rm data}$",
        "delta_chi2_true_per_point_vs_abs_delta_tE_loglog.png",
        good_data=good,
        y_threshold=DELTA_CHI2_PER_POINT_MAX,
    )

    scatter_symlogx_logy(
        df,
        "frac_delta_tE_fit_minus_true",
        "delta_chi2_true",
        r"$\Delta t_E/t_{E,\mathrm{true}}$",
        r"$\Delta\chi^2$",
        "delta_chi2_true_vs_frac_delta_tE_symlogx_logy.png",
        good_data=good,
        y_threshold=DELTA_CHI2_MAX,
        linthresh=0.01,
    )

    scatter_symlogx_logy(
        df,
        "frac_delta_tE_fit_minus_true",
        "delta_chi2_true_per_point",
        r"$\Delta t_E/t_{E,\mathrm{true}}$",
        r"$\Delta\chi^2/N_{\rm data}$",
        "delta_chi2_true_per_point_vs_frac_delta_tE_symlogx_logy.png",
        good_data=good,
        y_threshold=DELTA_CHI2_PER_POINT_MAX,
        linthresh=0.01,
    )

    scatter_loglog(
        df,
        "abs_frac_delta_tE_fit_minus_true",
        "delta_chi2_true",
        r"$|\Delta t_E|/t_{E,\mathrm{true}}$",
        r"$\Delta\chi^2$",
        "delta_chi2_true_vs_abs_frac_delta_tE_loglog.png",
        good_data=good,
        y_threshold=DELTA_CHI2_MAX,
    )

    scatter_loglog(
        df,
        "abs_frac_delta_tE_fit_minus_true",
        "delta_chi2_true_per_point",
        r"$|\Delta t_E|/t_{E,\mathrm{true}}$",
        r"$\Delta\chi^2/N_{\rm data}$",
        "delta_chi2_true_per_point_vs_abs_frac_delta_tE_loglog.png",
        good_data=good,
        y_threshold=DELTA_CHI2_PER_POINT_MAX,
    )


# ============================================================
# Bias histograms para eventos good
# ============================================================

def plot_bias_hist(column, xlabel, filename, use_good=True):
    """
    Histograma de bias.
    Para bias fraccional recorta outliers.
    """
    data = good if use_good else df

    if column not in data.columns:
        print(f"Skip {filename}: no existe {column}")
        return

    x = finite_series(data[column])

    if len(x) == 0:
        print(f"Skip {filename}: sin datos finitos")
        return

    if "frac" in column:
        x = x[(x > -CLIP_FRAC_BIAS) & (x < CLIP_FRAC_BIAS)]

    if len(x) == 0:
        print(f"Skip {filename}: sin datos tras clipping")
        return

    plt.figure(figsize=(7, 5))
    plt.hist(x, bins=30, histtype="stepfilled", alpha=0.75, linewidth=1.2)
    plt.axvline(
        0,
        linestyle="--",
        label=ZERO_BIAS_LABEL,
    )
    plt.xlabel(xlabel)
    plt.ylabel(r"Number of good No $\pi_E$ events" if use_good else "Number of events")
    plt.legend(handlelength=1.2)
    savefig(filename)


plot_bias_hist(
    "bias_t0",
    r"$t_{0,\mathrm{No}\,\pi_E}-t_{0,\mathrm{true}}$ [d]",
    "hist_bias_t0_good_NoPiE.png",
)

plot_bias_hist(
    "frac_bias_tE",
    r"$(t_{E,\mathrm{No}\,\pi_E}-t_{E,\mathrm{true}})/t_{E,\mathrm{true}}$",
    "hist_frac_bias_tE_good_NoPiE.png",
)

plot_bias_hist(
    "frac_bias_absu0",
    r"$(|u_{0,\mathrm{No}\,\pi_E}|-|u_{0,\mathrm{true}}|)/|u_{0,\mathrm{true}}|$",
    "hist_frac_bias_absu0_good_NoPiE.png",
)

plot_bias_hist(
    "delta_tE_fit_minus_true",
    r"$\Delta t_E = t_{E,\mathrm{fit}}-t_{E,\mathrm{true}}$ [d]",
    "hist_delta_tE_good_NoPiE.png",
)

plot_bias_hist(
    "frac_delta_tE_fit_minus_true",
    r"$\Delta t_E/t_{E,\mathrm{true}}$",
    "hist_frac_delta_tE_good_NoPiE.png",
)


# ============================================================
# Fit vs true
# ============================================================

def plot_fit_vs_true(
    true_col,
    fit_col,
    xlabel,
    ylabel,
    filename,
    loglog=False,
):
    """
    Plot fit vs true.
    """
    if true_col not in df.columns or fit_col not in df.columns:
        print(f"Skip {filename}: faltan {true_col} o {fit_col}")
        return

    mask = (
        np.isfinite(df[true_col])
        & np.isfinite(df[fit_col])
    )

    if loglog:
        mask = (
            mask
            & (df[true_col] > 0)
            & (df[fit_col] > 0)
        )

    if mask.sum() == 0:
        print(f"Skip {filename}: sin puntos válidos")
        return

    plt.figure(figsize=(6, 6))

    plt.scatter(
        df.loc[mask, true_col],
        df.loc[mask, fit_col],
        s=16,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
        label=ALL_EVENTS_LABEL,
    )

    good_mask = (
        mask
        & df["good_nopie"]
    )

    if good_mask.sum() > 0:
        plt.scatter(
            df.loc[good_mask, true_col],
            df.loc[good_mask, fit_col],
            s=22,
            alpha=0.85,
            linewidths=0,
            rasterized=True,
            label=GOOD_NOPIE_LABEL,
        )

    x = df.loc[mask, true_col]
    y = df.loc[mask, fit_col]

    vmin = np.nanmin([x.min(), y.min()])
    vmax = np.nanmax([x.max(), y.max()])

    plt.plot(
        [vmin, vmax],
        [vmin, vmax],
        linestyle="--",
        label=ONE_TO_ONE_LABEL,
    )

    if loglog:
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handlelength=1.2)
    savefig(filename)


plot_fit_vs_true(
    "tE_true_used",
    "tE_fit_used",
    r"$t_{E,\mathrm{true}}$ [d]",
    r"$t_{E,\mathrm{No}\,\pi_E}$ [d]",
    "tE_NoPiE_vs_true.png",
    loglog=True,
)

plot_fit_vs_true(
    "absu0_true_used",
    "absu0_fit_used",
    r"$|u_{0,\mathrm{true}}|$",
    r"$|u_{0,\mathrm{No}\,\pi_E}|$",
    "absu0_NoPiE_vs_true.png",
    loglog=True,
)

plot_fit_vs_true(
    "t0_true_used",
    "t0_fit_used",
    r"$t_{0,\mathrm{true}}$ [d]",
    r"$t_{0,\mathrm{No}\,\pi_E}$ [d]",
    "t0_NoPiE_vs_true.png",
    loglog=False,
)


# ============================================================
# Bias vs Delta chi2
# ============================================================

if "frac_bias_tE" in df.columns:

    scatter_logx(
        df,
        "delta_chi2_true",
        "frac_bias_tE",
        r"$\Delta\chi^2$",
        r"Fractional bias in $t_E$",
        "frac_bias_tE_vs_delta_chi2_true_logx.png",
        good_data=good,
        x_threshold=DELTA_CHI2_MAX,
        y_zero_line=True,
        ylim=(-CLIP_FRAC_BIAS, CLIP_FRAC_BIAS),
    )

    scatter_logx(
        df,
        "delta_chi2_true_per_point",
        "frac_bias_tE",
        r"$\Delta\chi^2/N_{\rm data}$",
        r"Fractional bias in $t_E$",
        "frac_bias_tE_vs_delta_chi2_true_per_point_logx.png",
        good_data=good,
        x_threshold=DELTA_CHI2_PER_POINT_MAX,
        y_zero_line=True,
        ylim=(-CLIP_FRAC_BIAS, CLIP_FRAC_BIAS),
    )


if "frac_bias_absu0" in df.columns:

    scatter_logx(
        df,
        "delta_chi2_true",
        "frac_bias_absu0",
        r"$\Delta\chi^2$",
        r"Fractional bias in $|u_0|$",
        "frac_bias_absu0_vs_delta_chi2_true_logx.png",
        good_data=good,
        x_threshold=DELTA_CHI2_MAX,
        y_zero_line=True,
        ylim=(-CLIP_FRAC_BIAS, CLIP_FRAC_BIAS),
    )

    scatter_logx(
        df,
        "delta_chi2_true_per_point",
        "frac_bias_absu0",
        r"$\Delta\chi^2/N_{\rm data}$",
        r"Fractional bias in $|u_0|$",
        "frac_bias_absu0_vs_delta_chi2_true_per_point_logx.png",
        good_data=good,
        x_threshold=DELTA_CHI2_PER_POINT_MAX,
        y_zero_line=True,
        ylim=(-CLIP_FRAC_BIAS, CLIP_FRAC_BIAS),
    )


# ============================================================
# Resumen numérico
# ============================================================

summary = {
    "N_total": len(df),
    "N_good_true_chi2": int(df["good_true_chi2"].sum()),
    "N_good_nopie": len(good),
    "N_good_nopie_strict": len(good_strict),
    "fraction_good_nopie": len(good) / len(df) if len(df) > 0 else np.nan,
    "CHI2_DOF_MAX": CHI2_DOF_MAX,
    "DELTA_CHI2_MAX": DELTA_CHI2_MAX,
    "DELTA_CHI2_PER_POINT_MAX": DELTA_CHI2_PER_POINT_MAX,
    "CHI2_TRUE_PER_POINT_MIN": CHI2_TRUE_PER_POINT_MIN,
    "CHI2_TRUE_PER_POINT_MAX": CHI2_TRUE_PER_POINT_MAX,
    "median_chi2_true_per_point": np.nanmedian(df["chi2_true_per_point"]),
    "median_chi2_dof_nopie": np.nanmedian(df["chi2_dof_nopie"]),
    "median_n_data_true_model": (
        np.nanmedian(df["n_data_true_model"])
        if "n_data_true_model" in df.columns
        else np.nan
    ),
    "median_delta_chi2_true": np.nanmedian(df["delta_chi2_true"]),
    "median_delta_chi2_true_per_point": np.nanmedian(
        df["delta_chi2_true_per_point"]
    ),
    "median_delta_chi2_true_good": (
        np.nanmedian(good["delta_chi2_true"])
        if len(good) > 0
        else np.nan
    ),
    "median_delta_chi2_true_per_point_good": (
        np.nanmedian(good["delta_chi2_true_per_point"])
        if len(good) > 0
        else np.nan
    ),
}

if "piE_true_used" in df.columns:
    summary["median_piE_true"] = np.nanmedian(df["piE_true_used"])
    summary["min_piE_true"] = np.nanmin(df["piE_true_used"])
    summary["max_piE_true"] = np.nanmax(df["piE_true_used"])

if "tE_true_used" in df.columns:
    summary["median_tE_true"] = np.nanmedian(df["tE_true_used"])
    summary["min_tE_true"] = np.nanmin(df["tE_true_used"])
    summary["max_tE_true"] = np.nanmax(df["tE_true_used"])

if "tE_fit_used" in df.columns:
    summary["median_tE_fit"] = np.nanmedian(df["tE_fit_used"])
    summary["min_tE_fit"] = np.nanmin(df["tE_fit_used"])
    summary["max_tE_fit"] = np.nanmax(df["tE_fit_used"])

if "delta_tE_fit_minus_true" in df.columns:
    summary["median_delta_tE_fit_minus_true"] = np.nanmedian(
        df["delta_tE_fit_minus_true"]
    )
    summary["median_abs_delta_tE_fit_minus_true"] = np.nanmedian(
        df["abs_delta_tE_fit_minus_true"]
    )
    summary["median_frac_delta_tE_fit_minus_true"] = np.nanmedian(
        df["frac_delta_tE_fit_minus_true"]
    )
    summary["median_abs_frac_delta_tE_fit_minus_true"] = np.nanmedian(
        df["abs_frac_delta_tE_fit_minus_true"]
    )

if "frac_bias_tE" in good.columns and len(good) > 0:
    summary["median_frac_bias_tE_good"] = np.nanmedian(good["frac_bias_tE"])
    summary["std_frac_bias_tE_good"] = np.nanstd(good["frac_bias_tE"])

if "frac_bias_absu0" in good.columns and len(good) > 0:
    summary["median_frac_bias_absu0_good"] = np.nanmedian(
        good["frac_bias_absu0"]
    )
    summary["std_frac_bias_absu0_good"] = np.nanstd(
        good["frac_bias_absu0"]
    )

if "delta_tE_fit_minus_true" in good.columns and len(good) > 0:
    summary["median_delta_tE_good"] = np.nanmedian(
        good["delta_tE_fit_minus_true"]
    )
    summary["median_abs_delta_tE_good"] = np.nanmedian(
        good["abs_delta_tE_fit_minus_true"]
    )
    summary["median_frac_delta_tE_good"] = np.nanmedian(
        good["frac_delta_tE_fit_minus_true"]
    )
    summary["median_abs_frac_delta_tE_good"] = np.nanmedian(
        good["abs_frac_delta_tE_fit_minus_true"]
    )

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    TABLES_DIR / "summary_NoPiE_vs_true.csv",
    index=False,
)

print("=" * 80)
print("Resumen")
print(summary_df.T)
print("=" * 80)
print(f"Tablas guardadas en: {TABLES_DIR}")
print(f"Plots guardados en:  {PLOTS_DIR}")
print("=" * 80)
