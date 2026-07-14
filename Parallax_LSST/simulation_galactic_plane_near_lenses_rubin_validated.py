#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Corrida Rubin-only en paralelo para estudiar paralaje en microlensing
hacia varias lineas de vision del plano galactico, lejos del bulbo,
con lentes cercanas.

Objetivo fisico:
- Lentes cercanas => pi_rel grande => pi_E grande.
- Eventos de larga duracion, especialmente t_E > ~6 meses, pueden mostrar
  desviaciones de paralaje por la aceleracion orbital de la Tierra.
- Se simulan curvas PSPL con paralaje y se ajustan con PSPL sin paralaje.
- Se evalua si el evento seria confundible con PSPL y si los parametros
  ajustados quedan sesgados.

Version con:
- Multiples campos definidos en coordenadas galacticas (l,b).
- Conversion automatica (l,b) -> (RA,Dec) para AstroDataLab y Rubin/MAF.
- Catalogo pareado AstroDataLab independiente para cada linea de vision.
- Rubin-only usando coordenadas fuente por fuente.
- Binning angular para MAF/Rubin, para no llamar a rubin_sim una vez por fuente.
- Opcional: prewarming de cache MAF antes de abrir el pool paralelo.
- Procesamiento balanceado: N_EVENTS_PER_FIELD eventos por linea de vision.
"""

# ============================================================
# Evitar sobre-suscripcion de threads dentro de cada proceso
# ============================================================

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
sys.path.append("/home/anibal/microlensing/simulation_Rubin/roman_rubin/")

from pathlib import Path
import json
import traceback
import importlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u

import ulens_params
importlib.reload(ulens_params)

from ulens_params import download_astrodatalab_pair_catalog
from functions_roman_rubin import sim_fit


# ============================================================
# Variables globales para workers
# ============================================================

GLOBAL_PAIR_CATALOGS = None
GLOBAL_CFG = None


def init_worker(pair_catalogs, cfg):
    """
    Inicializador de cada proceso.

    Guarda los catalogos por campo y la configuracion como variables globales
    para no tener que pasarlas en cada evento.
    """

    global GLOBAL_PAIR_CATALOGS
    global GLOBAL_CFG

    GLOBAL_PAIR_CATALOGS = pair_catalogs
    GLOBAL_CFG = cfg


# ============================================================
# Helpers de campos / coordenadas
# ============================================================

def galactic_to_radec(l_deg, b_deg):
    """
    Convierte coordenadas galacticas (l,b) en grados a ICRS (RA,Dec).
    """

    c = SkyCoord(
        l=float(l_deg) * u.deg,
        b=float(b_deg) * u.deg,
        frame="galactic",
    )

    icrs = c.icrs

    return float(icrs.ra.deg), float(icrs.dec.deg)


def make_field_name(l_deg, b_deg):
    """
    Nombre corto y estable para una linea de vision.
    """

    l_label = int(round(float(l_deg)))
    b_label = int(round(float(b_deg)))

    l_str = f"l{l_label:03d}"

    if b_label >= 0:
        b_str = f"bp{abs(b_label):02d}"
    else:
        b_str = f"bm{abs(b_label):02d}"

    return f"{l_str}_{b_str}"


def prepare_galactic_fields(fields):
    """
    Agrega RA/Dec y nombre a cada campo galactico.
    """

    out = []

    for field in fields:
        f = dict(field)
        l_deg = float(f["l_deg"])
        b_deg = float(f.get("b_deg", 0.0))

        ra_deg, dec_deg = galactic_to_radec(l_deg, b_deg)

        f["ra_center"] = ra_deg
        f["dec_center"] = dec_deg
        f["field_name"] = f.get("field_name", make_field_name(l_deg, b_deg))

        out.append(f)

    return out


# ============================================================
# Helpers MAF binning
# ============================================================

def quantize_maf_cells(df, cell_deg):
    """
    Calcula las celdas MAF que se usarian para un catalogo dado.

    El evento sigue teniendo ra/dec reales de la fuente, pero MAF usa:
        maf_ra  = round(ra/cell_deg)*cell_deg
        maf_dec = round(dec/cell_deg)*cell_deg
    """

    ra = np.asarray(df["ra"], dtype=float)
    dec = np.asarray(df["dec"], dtype=float)

    if cell_deg is None:
        maf_ra = np.round(ra, 5)
        maf_dec = np.round(dec, 5)
    else:
        cell_deg = float(cell_deg)
        maf_ra = np.round(ra / cell_deg) * cell_deg
        maf_dec = np.round(dec / cell_deg) * cell_deg
        maf_ra = maf_ra % 360.0
        maf_dec = np.clip(maf_dec, -90.0, 90.0)

        maf_ra = np.round(maf_ra, 6)
        maf_dec = np.round(maf_dec, 6)

    cells = pd.DataFrame(
        {
            "maf_ra": maf_ra,
            "maf_dec": maf_dec,
        }
    )

    return cells


def selected_pair_row_for_seed(seed, n_pair_rows):
    """
    Reproduce la seleccion de fila que hace sim_fit para
    catalog_mode='astrodatalab_pairs':

        np.random.seed(i)
        ROW_P = np.random.randint(0, len(pair_catalog))

    """

    rng = np.random.RandomState(int(seed))
    return int(rng.randint(0, int(n_pair_rows)))


def selected_pair_rows_for_tasks(event_tasks, pair_catalogs):
    """
    Devuelve las filas de catalogo que serian seleccionadas para cada tarea.
    """

    rows = []

    for task in event_tasks:
        field_name = task["field_name"]
        global_i = int(task["global_i"])
        n_rows = len(pair_catalogs[field_name])
        rows.append(selected_pair_row_for_seed(global_i, n_rows))

    return np.asarray(rows, dtype=int)


def print_maf_binning_diagnostics(pair_catalogs, event_tasks, rubin_cache_cell_deg):
    """
    Imprime cuantas celdas MAF se usaran aproximadamente por campo.
    """

    print("=" * 80)
    print("Diagnostico de binning MAF")
    print("=" * 80)
    print(f"rubin_cache_cell_deg = {rubin_cache_cell_deg}")
    print(f"N total events        = {len(event_tasks)}")
    print("=" * 80)

    total_cells_expected = []

    for field_name, catalog in pair_catalogs.items():
        tasks_field = [t for t in event_tasks if t["field_name"] == field_name]

        cells_all = quantize_maf_cells(
            catalog,
            rubin_cache_cell_deg,
        )

        n_cells_all = len(cells_all.drop_duplicates())

        selected_rows = [
            selected_pair_row_for_seed(t["global_i"], len(catalog))
            for t in tasks_field
        ]

        if len(selected_rows) > 0:
            catalog_sub = catalog.iloc[selected_rows].copy()

            cells_sub = quantize_maf_cells(
                catalog_sub,
                rubin_cache_cell_deg,
            )

            n_cells_sub = len(cells_sub.drop_duplicates())
        else:
            n_cells_sub = 0

        total_cells_expected.append(n_cells_sub)

        print(f"Campo {field_name}")
        print(f"  N pair catalog: {len(catalog)}")
        print(f"  N events:       {len(tasks_field)}")
        print(f"  MAF cells catalog completo: {n_cells_all}")
        print(f"  MAF cells esperadas corrida: {n_cells_sub}")
        print("-" * 80)

    print(f"Total MAF cells esperadas, sumando campos: {sum(total_cells_expected)}")
    print("=" * 80)



def _maf_cells_for_catalog(catalog, rubin_cache_cell_deg):
    """
    Devuelve una tabla catalog+maf_ra_bin+maf_dec_bin.
    """

    cells = quantize_maf_cells(
        catalog,
        rubin_cache_cell_deg,
    )

    out = catalog.copy()
    out["maf_ra_bin"] = cells["maf_ra"].values
    out["maf_dec_bin"] = cells["maf_dec"].values

    return out


def _probe_maf_cell(path_ephemerides, maf_ra, maf_dec, rubin_cache_cell_deg):
    """
    Prueba si una celda MAF/Rubin es utilizable.

    Algunas lineas de vision pueden caer fuera del footprint efectivo de Rubin.
    En esos casos MAF puede devolver un dataSlice vacio o con una estructura
    no compatible con la construccion de las curvas Rubin. En lugar de dejar
    que la corrida se caiga, marcamos esa celda como invalida.
    """

    import set_telescopes_pyLIMA as stp

    record = {
        "maf_ra_bin": float(maf_ra),
        "maf_dec_bin": float(maf_dec),
        "rubin_valid": False,
        "rubin_error": "",
        "maf_mode": "",
        "maf_cache_mode": "",
        "maf_cache_source": "",
        "maf_n_obs": np.nan,
    }

    try:
        stp.tel_roman_rubin(
            path_ephemerides,
            time_window=None,
            use_roman=False,
            use_rubin=True,
            Ra=float(maf_ra),
            Dec=float(maf_dec),
            rubin_pointing_mode="source",
            rubin_cache_cell_deg=rubin_cache_cell_deg,
        )

        info = getattr(stp, "LAST_DATASLICE_INFO", {})

        n_obs = info.get("n_obs", np.nan)

        record["maf_mode"] = info.get("mode", "")
        record["maf_cache_mode"] = info.get("cache_mode", "")
        record["maf_cache_source"] = info.get("source", "")
        record["maf_n_obs"] = n_obs

        try:
            n_obs_float = float(n_obs)
        except Exception:
            n_obs_float = np.nan

        record["rubin_valid"] = np.isfinite(n_obs_float) and (n_obs_float > 0)

        if not record["rubin_valid"]:
            record["rubin_error"] = f"MAF cell returned n_obs={n_obs}"

    except Exception as e:
        record["rubin_valid"] = False
        record["rubin_error"] = repr(e)

    return record


def validate_and_filter_pair_catalogs_for_rubin(
    pair_catalogs,
    rubin_cache_cell_deg,
    path_ephemerides,
    catalogs_dir,
):
    """
    Valida las celdas MAF/Rubin de cada campo y filtra catalogos.

    Esto evita que una linea de vision sin observaciones Rubin, o con un
    dataSlice invalido, mate la corrida completa. El caso tipico es un campo
    demasiado al norte para Rubin.

    Devuelve:
        filtered_pair_catalogs, validation_table
    """

    print("=" * 80)
    print("Validando cobertura Rubin/MAF por celda")
    print("=" * 80)

    catalogs_dir = Path(catalogs_dir)
    filtered_pair_catalogs = {}
    validation_rows = []

    for field_name, catalog in pair_catalogs.items():

        catalog_with_bins = _maf_cells_for_catalog(
            catalog,
            rubin_cache_cell_deg,
        )

        cells_unique = (
            catalog_with_bins[["maf_ra_bin", "maf_dec_bin"]]
            .drop_duplicates()
            .sort_values(["maf_ra_bin", "maf_dec_bin"])
            .reset_index(drop=True)
        )

        print("-" * 80)
        print(f"Campo {field_name}")
        print(f"N pairs antes de filtro: {len(catalog_with_bins)}")
        print(f"N celdas MAF a probar:   {len(cells_unique)}")

        field_validation_rows = []

        for k, cell in cells_unique.iterrows():
            maf_ra = float(cell["maf_ra_bin"])
            maf_dec = float(cell["maf_dec_bin"])

            print(
                f"  [{k + 1}/{len(cells_unique)}] "
                f"RA={maf_ra:.6f}, Dec={maf_dec:.6f} ... ",
                end="",
                flush=True,
            )

            rec = _probe_maf_cell(
                path_ephemerides,
                maf_ra,
                maf_dec,
                rubin_cache_cell_deg,
            )

            rec["field_name"] = field_name
            field_validation_rows.append(rec)
            validation_rows.append(rec)

            if rec["rubin_valid"]:
                print(f"OK, n_obs={rec['maf_n_obs']}")
            else:
                print(f"INVALID: {rec['rubin_error']}")

        validation_field = pd.DataFrame(field_validation_rows)

        validation_field.to_csv(
            catalogs_dir / f"maf_validation_{field_name}.csv",
            index=False,
        )

        validation_field.to_parquet(
            catalogs_dir / f"maf_validation_{field_name}.parquet",
            index=False,
        )

        valid_cells = validation_field[validation_field["rubin_valid"]].copy()

        if len(valid_cells) == 0:
            print(f"[warning] Campo {field_name} sin celdas Rubin validas. Se saltea completo.")
            continue

        valid_pairs = set(
            zip(
                valid_cells["maf_ra_bin"].astype(float),
                valid_cells["maf_dec_bin"].astype(float),
            )
        )

        keep = [
            (float(ra), float(dec)) in valid_pairs
            for ra, dec in zip(
                catalog_with_bins["maf_ra_bin"],
                catalog_with_bins["maf_dec_bin"],
            )
        ]

        filtered = catalog_with_bins.loc[keep].copy().reset_index(drop=True)

        print(f"N celdas validas:        {len(valid_cells)}")
        print(f"N pairs despues filtro:  {len(filtered)}")

        if len(filtered) == 0:
            print(f"[warning] Campo {field_name} quedo sin pares luego del filtro Rubin. Se saltea.")
            continue

        filtered_pair_catalogs[field_name] = filtered

        filtered.to_parquet(
            catalogs_dir / f"astrodatalab_pairs_{field_name}_near_lens_rubin_valid.parquet",
            index=False,
        )

    validation_table = pd.DataFrame(validation_rows)

    if len(validation_table) > 0:
        validation_table.to_csv(
            catalogs_dir / "maf_validation_all_fields.csv",
            index=False,
        )

        validation_table.to_parquet(
            catalogs_dir / "maf_validation_all_fields.parquet",
            index=False,
        )

    print("=" * 80)
    print("Resumen validacion Rubin/MAF")
    print("=" * 80)

    if len(validation_table) > 0:
        print(validation_table.groupby("field_name")["rubin_valid"].agg(["sum", "count"]))
    else:
        print("No se evaluo ninguna celda.")

    print("Campos que quedan para simular:")
    print(list(filtered_pair_catalogs.keys()))
    print("=" * 80)

    if len(filtered_pair_catalogs) == 0:
        raise RuntimeError(
            "Ningun campo tiene cobertura Rubin/MAF valida. "
            "Revisa las lineas de vision o el footprint usado."
        )

    return filtered_pair_catalogs, validation_table


def prewarm_maf_cache(pair_catalogs, event_tasks, rubin_cache_cell_deg, path_ephemerides):
    """
    Precalienta el cache de MAF para las celdas que van a usar los eventos.

    Esta version es robusta: si una celda falla, no aborta inmediatamente.
    De todos modos, si usaste validate_and_filter_pair_catalogs_for_rubin
    antes, no deberian quedar celdas invalidas.
    """

    print("=" * 80)
    print("Prewarming MAF cache")
    print("=" * 80)

    rows = []

    for task in event_tasks:
        field_name = task["field_name"]
        global_i = int(task["global_i"])
        catalog = pair_catalogs[field_name]
        row_idx = selected_pair_row_for_seed(global_i, len(catalog))
        row = catalog.iloc[row_idx]

        rows.append(
            {
                "field_name": field_name,
                "source_ra": float(row["ra"]),
                "source_dec": float(row["dec"]),
            }
        )

    if len(rows) == 0:
        print("No hay tareas para prewarm.")
        return

    selected = pd.DataFrame(rows)

    cells = quantize_maf_cells(
        selected.rename(
            columns={
                "source_ra": "ra",
                "source_dec": "dec",
            }
        ),
        rubin_cache_cell_deg,
    )

    selected["maf_ra"] = cells["maf_ra"].values
    selected["maf_dec"] = cells["maf_dec"].values

    cells_unique = (
        selected[["field_name", "maf_ra", "maf_dec"]]
        .drop_duplicates()
        .sort_values(["field_name", "maf_ra", "maf_dec"])
        .reset_index(drop=True)
    )

    print(f"N events:           {len(event_tasks)}")
    print(f"N unique MAF cells: {len(cells_unique)}")
    print("=" * 80)

    failed_cells = []

    for k, row in cells_unique.iterrows():
        maf_ra = float(row["maf_ra"])
        maf_dec = float(row["maf_dec"])
        field_name = row["field_name"]

        print(
            f"[{k + 1}/{len(cells_unique)}] "
            f"{field_name}: prewarm MAF cell "
            f"RA={maf_ra:.6f}, Dec={maf_dec:.6f}"
        )

        rec = _probe_maf_cell(
            path_ephemerides,
            maf_ra,
            maf_dec,
            rubin_cache_cell_deg,
        )

        if not rec["rubin_valid"]:
            rec["field_name"] = field_name
            failed_cells.append(rec)
            print("  [warning] celda invalida durante prewarm:", rec["rubin_error"])

    if len(failed_cells) > 0:
        failed = pd.DataFrame(failed_cells)
        print("=" * 80)
        print("WARNING: algunas celdas fallaron durante prewarm")
        print(failed[["field_name", "maf_ra_bin", "maf_dec_bin", "rubin_error"]])
        print("=" * 80)

    print("=" * 80)
    print("MAF cache prewarming finished")
    print("=" * 80)

# ============================================================
# Catalogos por linea de vision
# ============================================================

def build_or_load_pair_catalog_for_field(field, catalog_config_template, catalogs_dir):
    """
    Crea o carga el catalogo pareado para una linea de vision.
    """

    field_name = field["field_name"]

    path_pair_catalog = (
        Path(catalogs_dir)
        / f"astrodatalab_pairs_{field_name}_near_lens.parquet"
    )

    if path_pair_catalog.exists():
        print("=" * 80)
        print(f"Usando catalogo existente para {field_name}:")
        print(path_pair_catalog)
        print("=" * 80)

        pair_catalog = pd.read_parquet(path_pair_catalog)

    else:
        print("=" * 80)
        print(f"Generando catalogo AstroDataLab para {field_name}")
        print(f"l,b       = {field['l_deg']:.3f}, {field['b_deg']:.3f} deg")
        print(f"RA,Dec    = {field['ra_center']:.6f}, {field['dec_center']:.6f} deg")
        print(f"radius    = {field['radius']:.4f} deg")
        print("=" * 80)

        cfg = dict(catalog_config_template)
        cfg["ra_center"] = float(field["ra_center"])
        cfg["dec_center"] = float(field["dec_center"])
        cfg["radius"] = float(field["radius"])
        cfg["random_state"] = int(field.get("random_state", cfg["random_state"]))

        pair_catalog = download_astrodatalab_pair_catalog(**cfg)

        pair_catalog.to_parquet(
            path_pair_catalog,
            index=False,
        )

        print("=" * 80)
        print("Catalogo guardado en:")
        print(path_pair_catalog)
        print("=" * 80)

    pair_catalog = pair_catalog.copy()

    pair_catalog["field_name"] = field_name
    pair_catalog["field_l_center"] = float(field["l_deg"])
    pair_catalog["field_b_center"] = float(field["b_deg"])
    pair_catalog["field_ra_center"] = float(field["ra_center"])
    pair_catalog["field_dec_center"] = float(field["dec_center"])
    pair_catalog["field_radius_deg"] = float(field["radius"])

    return field_name, pair_catalog, path_pair_catalog


def build_or_load_all_pair_catalogs(fields, catalog_config_template, catalogs_dir):
    """
    Crea o carga catalogos por linea de vision.
    """

    pair_catalogs = {}
    catalog_paths = {}

    for field in fields:
        field_name, pair_catalog, path_pair_catalog = build_or_load_pair_catalog_for_field(
            field,
            catalog_config_template,
            catalogs_dir,
        )

        if len(pair_catalog) == 0:
            print(f"[warning] Catalogo vacio para {field_name}. Se saltea.")
            continue

        pair_catalogs[field_name] = pair_catalog
        catalog_paths[field_name] = str(path_pair_catalog)

    if len(pair_catalogs) == 0:
        raise RuntimeError("No se pudo construir ningun catalogo pareado.")

    return pair_catalogs, catalog_paths


# ============================================================
# Tareas de eventos
# ============================================================

def build_event_tasks(fields, pair_catalogs, n_events_per_field):
    """
    Arma una lista balanceada de tareas.

    Cada campo recibe n_events_per_field eventos.
    global_i es unico y se usa como semilla en sim_fit.
    """

    tasks = []
    global_i = 0

    for field in fields:
        field_name = field["field_name"]

        if field_name not in pair_catalogs:
            continue

        for local_i in range(int(n_events_per_field)):
            tasks.append(
                {
                    "global_i": int(global_i),
                    "local_i": int(local_i),
                    "field_name": field_name,
                    "field_l_center": float(field["l_deg"]),
                    "field_b_center": float(field["b_deg"]),
                    "field_ra_center": float(field["ra_center"]),
                    "field_dec_center": float(field["dec_center"]),
                }
            )
            global_i += 1

    return tasks


# ============================================================
# Corrida de un evento
# ============================================================

def run_single_event(task):
    """
    Ejecuta un unico evento dentro de un worker.

    Cada evento escribe en sus propias carpetas:
        models/<field>/event_XXXXXX/
        fits/<field>/event_XXXXXX/
        results/<field>/event_XXXXXX/
        logs/<field>/event_XXXXXX.log
    """

    cfg = GLOBAL_CFG
    pair_catalogs = GLOBAL_PAIR_CATALOGS

    global_i = int(task["global_i"])
    local_i = int(task["local_i"])
    field_name = task["field_name"]

    pair_catalog = pair_catalogs[field_name]

    event_tag = f"event_{global_i:06d}"

    event_model_dir = Path(cfg["models_dir"]) / field_name / event_tag
    event_fit_dir = Path(cfg["fits_dir"]) / field_name / event_tag
    event_results_dir = Path(cfg["results_dir"]) / field_name / event_tag
    event_log_file = Path(cfg["logs_dir"]) / field_name / f"{event_tag}.log"

    event_model_dir.mkdir(parents=True, exist_ok=True)
    event_fit_dir.mkdir(parents=True, exist_ok=True)
    event_results_dir.mkdir(parents=True, exist_ok=True)
    event_log_file.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "global_i": global_i,
        "local_i": local_i,
        "field_name": field_name,
        "field_l_center": task.get("field_l_center", np.nan),
        "field_b_center": task.get("field_b_center", np.nan),
        "field_ra_center": task.get("field_ra_center", np.nan),
        "field_dec_center": task.get("field_dec_center", np.nan),
        "status": "started",
        "error": "",
        "log_file": str(event_log_file),
        "model_dir": str(event_model_dir),
        "fit_dir": str(event_fit_dir),
        "results_dir": str(event_results_dir),
        "rubin_pointing_mode": cfg["rubin_pointing_mode"],
        "rubin_cache_cell_deg": cfg["rubin_cache_cell_deg"],
        "maf_mode": "",
        "maf_cache_mode": "",
        "maf_source_ra": np.nan,
        "maf_source_dec": np.nan,
        "maf_ra": np.nan,
        "maf_dec": np.nan,
        "maf_cache_source": "",
        "maf_n_obs": np.nan,
    }

    try:
        with open(event_log_file, "w") as log:
            with redirect_stdout(log), redirect_stderr(log):

                print("=" * 80)
                print(f"Evento global_i={global_i}, local_i={local_i}, field={field_name}")
                print("=" * 80)
                print("Rubin pointing mode:", cfg["rubin_pointing_mode"])
                print("Rubin cache cell deg:", cfg["rubin_cache_cell_deg"])
                print("N pair catalog field:", len(pair_catalog))
                print("=" * 80)

                result = sim_fit(
                    global_i,
                    cfg["system_type"],
                    model=cfg["model"],
                    algo=cfg["algo"],

                    path_TRILEGAL_set=None,
                    path_GENULENS_set=None,

                    path_to_save_model=str(event_model_dir) + "/",
                    path_to_save_fit=str(event_fit_dir) + "/",
                    path_ephemerides=cfg["path_ephemerides"],
                    path_to_save_results=str(event_results_dir) + "/",

                    catalog_mode="astrodatalab_pairs",
                    pair_catalog=pair_catalog,

                    use_roman=cfg["use_roman"],
                    use_rubin=cfg["use_rubin"],

                    param_samplers=cfg["param_samplers_near_lenses"],

                    fit_model=cfg["fit_model"],
                    fit_parallax=cfg["fit_parallax"],
                    fit_bounds=cfg["fit_bounds_nopie"],

                    rubin_pointing_mode=cfg["rubin_pointing_mode"],
                    rubin_cache_cell_deg=cfg["rubin_cache_cell_deg"],
                )

                try:
                    import set_telescopes_pyLIMA as stp

                    info = getattr(stp, "LAST_DATASLICE_INFO", {})

                    print("=" * 80)
                    print("LAST_DATASLICE_INFO")
                    print(info)
                    print("=" * 80)

                    row["maf_mode"] = info.get("mode", "")
                    row["maf_cache_mode"] = info.get("cache_mode", "")
                    row["maf_source_ra"] = info.get("source_Ra", np.nan)
                    row["maf_source_dec"] = info.get("source_Dec", np.nan)
                    row["maf_ra"] = info.get("maf_Ra", info.get("Ra", np.nan))
                    row["maf_dec"] = info.get("maf_Dec", info.get("Dec", np.nan))
                    row["maf_cache_source"] = info.get("source", "")
                    row["maf_n_obs"] = info.get("n_obs", np.nan)

                except Exception as e:
                    print("[warning] No pude leer LAST_DATASLICE_INFO:", repr(e))

                print("=" * 80)
                print("Evento terminado correctamente")
                print("=" * 80)

        row["status"] = "ok"

    except Exception as e:

        err = traceback.format_exc()

        with open(event_log_file, "a") as log:
            log.write("\n")
            log.write("=" * 80 + "\n")
            log.write("ERROR\n")
            log.write("=" * 80 + "\n")
            log.write(err)
            log.write("\n")

        row["status"] = "failed"
        row["error"] = str(e)

    return row


# ============================================================
# Resumen incremental
# ============================================================

def save_summary(summary_rows, logs_dir):
    """
    Guarda el resumen incremental de la corrida.
    """

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)

    if len(summary_df) > 0:
        sort_cols = [c for c in ["global_i", "field_name", "local_i"] if c in summary_df.columns]
        summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)

    summary_df.to_csv(
        logs_dir / "run_summary.csv",
        index=False,
    )

    summary_df.to_parquet(
        logs_dir / "run_summary.parquet",
        index=False,
    )

    return summary_df


# ============================================================
# Main
# ============================================================

def main():

    # ============================================================
    # Configuracion general
    # ============================================================

    BASE_DIR = Path("/home/anibal/Parallax_LSST")

    RUN_NAME = "GalPlane_near_lenses_RubinOnly_PSPLparallax_fitNoPiE_MAFbin020"

    RUN_DIR = BASE_DIR / "runs" / RUN_NAME

    DIRS = {
        "catalogs": RUN_DIR / "catalogs",
        "models": RUN_DIR / "models",
        "fits": RUN_DIR / "fits",
        "results": RUN_DIR / "results",
        "plots": RUN_DIR / "plots",
        "logs": RUN_DIR / "logs",
        "config": RUN_DIR / "config",
    }

    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Paths externos
    # ============================================================

    path_ephemerides = (
        "/home/anibal/microlensing/simulation_Rubin/roman_rubin/"
        "ephemerides/Roman_positions.npy"
    )

    # ============================================================
    # Parametros de corrida
    # ============================================================

    system_type = "BH"
    model = "PSPL"
    fit_model = "PSPL"
    fit_parallax = False
    algo = "TRF"

    use_roman = False
    use_rubin = True

    # Numero de eventos por linea de vision.
    # Para prueba: 50-100.
    # Para corrida real: 500-2000 por campo.
    N_EVENTS_PER_FIELD = 1000
    N_PAIRS_PER_FIELD = 5000

    # En muchas maquinas 6-8 workers puede ser mas rapido que 16
    # por I/O y memoria. Subilo si ves buena escalabilidad.
    N_WORKERS = 8

    # ============================================================
    # Lineas de vision del plano galactico, lejos del bulbo
    # ============================================================
    # Excluimos |l| pequeno alrededor del bulbo.
    # Campos de ejemplo visibles razonablemente por Rubin/LSST.
    # Se pueden agregar b=+-2 para explorar fuera del plano.

    GALACTIC_FIELDS_INPUT = [
        {"l_deg": 30.0,  "b_deg": 0.0, "radius": 0.20, "random_state": 1001},
        {"l_deg": 60.0,  "b_deg": 0.0, "radius": 0.20, "random_state": 1002},
        {"l_deg": 210.0, "b_deg": 0.0, "radius": 0.20, "random_state": 1003},
        {"l_deg": 240.0, "b_deg": 0.0, "radius": 0.20, "random_state": 1004},
        {"l_deg": 270.0, "b_deg": 0.0, "radius": 0.20, "random_state": 1005},
        {"l_deg": 300.0, "b_deg": 0.0, "radius": 0.20, "random_state": 1006},
        {"l_deg": 330.0, "b_deg": 0.0, "radius": 0.20, "random_state": 1007},
    ]

    fields = prepare_galactic_fields(GALACTIC_FIELDS_INPUT)

    # ============================================================
    # Configuracion MAF/Rubin
    # ============================================================
    # source: evento en coordenadas reales de fuente.
    # rubin_cache_cell_deg agrupa llamadas a MAF.
    # Para multiples campos conviene empezar con 0.2 deg.

    rubin_pointing_mode = "source"
    rubin_cache_cell_deg = 0.20

    # Precalentar MAF antes del paralelo. Puede tardar al comienzo,
    # pero evita que cada worker reconstruya la misma celda.
    PREWARM_MAF_CACHE = True

    # Valida y filtra celdas/campos que no tengan observaciones Rubin.
    # Esto evita fallas en campos fuera del footprint efectivo de LSST.
    VALIDATE_RUBIN_COVERAGE = True

    # ============================================================
    # Samplers fisicos: lentes cercanas
    # ============================================================
    # Con D_L pequeno, pi_rel y pi_E crecen.
    # El rango de masa permite eventos largos sin limitarse solo a BH masivos.

    param_samplers_near_lenses = {
        "star_mass": {
            "type": "loguniform",
            "low": 0.1,
            "high": 30.0,
        },
        "mass_planet": {
            "type": "fixed",
            "value": 0.0,
        },
    }

    fit_bounds_nopie = {
        "t0": {
            "type": "center_width",
            "half_width": 800.0,
        },
        "u0": [
            -5.0,
            5.0,
        ],
        "tE": [
            1.0,
            10000.0,
        ],
    }

    # ============================================================
    # Configuracion del catalogo AstroDataLab
    # ============================================================
    # Lentes cercanas:
    #   lens_D_range_kpc = (0.05, 1.0)
    # Fuentes de fondo:
    #   source_D_range_kpc = (2.0, 12.0)
    # Distancias internas min_D/offset estan en pc.

    catalog_config_template = {
        # Se sobreescriben por campo:
        "ra_center": None,
        "dec_center": None,
        "radius": None,

        "N": N_PAIRS_PER_FIELD,
        "Ds_max": 12000,

        # Distancias internas en pc
        "min_D": 50.0,
        "offset": 200.0,

        "random_state": 123,
        "limit_extra": 0,
        "table": "lsst_sim.simdr2",
        "mu_rel_mode": "random_angle",
        "w149_from": "Y",

        "select_cols": [
            "ra",
            "dec",
            "mu0",
            "pmracosd",
            "pmdec",
            "umag",
            "gmag",
            "rmag",
            "imag",
            "zmag",
            "ymag",
            "logl",
            "logte",
            "gc",
            "galb",
            "gall",
        ],

        # Disco fino + disco grueso. Para halo se podria agregar gc=3.
        "extra_where": "gc IN (1, 2)",

        "extra_keep_cols": [
            "gc",
            "galb",
            "gall",
            "D_S_kpc",
            "D_L_kpc",
            "lens_gc",
            "lens_galb",
            "lens_gall",
        ],

        "chunk_query": True,
        "n_mu0_chunks": 20,
        "limit_per_chunk": 50,
        "max_retries": 0,
        "timeout": 300,

        "lens_D_range_kpc": (0.05, 1.0),
        "source_D_range_kpc": (2.0, 12.0),
    }

    # ============================================================
    # Preparar y guardar configuracion
    # ============================================================

    run_config = {
        "RUN_NAME": RUN_NAME,
        "RUN_DIR": str(RUN_DIR),
        "system_type": system_type,
        "model": model,
        "fit_model": fit_model,
        "fit_parallax": fit_parallax,
        "algo": algo,
        "use_roman": use_roman,
        "use_rubin": use_rubin,
        "N_EVENTS_PER_FIELD": N_EVENTS_PER_FIELD,
        "N_PAIRS_PER_FIELD": N_PAIRS_PER_FIELD,
        "N_WORKERS": N_WORKERS,
        "path_ephemerides": path_ephemerides,
        "param_samplers_near_lenses": param_samplers_near_lenses,
        "fit_bounds_nopie": fit_bounds_nopie,
        "catalog_config_template": catalog_config_template,
        "rubin_pointing_mode": rubin_pointing_mode,
        "rubin_cache_cell_deg": rubin_cache_cell_deg,
        "PREWARM_MAF_CACHE": PREWARM_MAF_CACHE,
        "VALIDATE_RUBIN_COVERAGE": VALIDATE_RUBIN_COVERAGE,
        "fields": fields,
    }

    with open(DIRS["config"] / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)

    # Tabla de campos
    fields_df = pd.DataFrame(fields)
    fields_df.to_csv(DIRS["config"] / "fields.csv", index=False)
    fields_df.to_parquet(DIRS["config"] / "fields.parquet", index=False)

    print("=" * 80)
    print("Campos galacticos")
    print("=" * 80)
    print(fields_df[["field_name", "l_deg", "b_deg", "ra_center", "dec_center", "radius"]])
    print("=" * 80)

    # ============================================================
    # Crear o cargar catalogos por linea de vision
    # ============================================================

    pair_catalogs, catalog_paths = build_or_load_all_pair_catalogs(
        fields,
        catalog_config_template,
        DIRS["catalogs"],
    )

    # ============================================================
    # Validar cobertura Rubin/MAF y filtrar campos/celdas invalidas
    # ============================================================

    if VALIDATE_RUBIN_COVERAGE:
        pair_catalogs, maf_validation_table = validate_and_filter_pair_catalogs_for_rubin(
            pair_catalogs,
            rubin_cache_cell_deg,
            path_ephemerides,
            DIRS["catalogs"],
        )
    else:
        maf_validation_table = pd.DataFrame()

    # Guardar catalogo combinado para auditoria/analisis
    all_pair_catalog = pd.concat(
        pair_catalogs.values(),
        ignore_index=True,
    )

    all_pair_catalog.to_parquet(
        DIRS["catalogs"] / "all_pair_catalogs_combined_rubin_valid.parquet",
        index=False,
    )

    # ============================================================
    # Chequeo de catalogos
    # ============================================================

    print("=" * 80)
    print("Resumen de catalogos pareados")
    print("=" * 80)

    for field_name, catalog in pair_catalogs.items():
        print(f"Campo: {field_name}")
        print("N pairs:", len(catalog))
        cols_summary = [c for c in ["D_L_kpc", "D_S_kpc", "mu_rel", "gall", "galb"] if c in catalog.columns]
        if len(cols_summary) > 0:
            print(catalog[cols_summary].describe())
        print("-" * 80)

    # ============================================================
    # Construir tareas balanceadas
    # ============================================================

    event_tasks = build_event_tasks(
        fields,
        pair_catalogs,
        N_EVENTS_PER_FIELD,
    )

    tasks_df = pd.DataFrame(event_tasks)
    tasks_df.to_csv(DIRS["config"] / "event_tasks.csv", index=False)
    tasks_df.to_parquet(DIRS["config"] / "event_tasks.parquet", index=False)

    N_events_total = len(event_tasks)

    print("=" * 80)
    print("Tareas de simulacion")
    print("=" * 80)
    print(f"N fields:       {len(pair_catalogs)}")
    print(f"N events total: {N_events_total}")
    print(tasks_df.groupby("field_name").size())
    print("=" * 80)

    # ============================================================
    # Diagnostico MAF y prewarm
    # ============================================================

    print_maf_binning_diagnostics(
        pair_catalogs,
        event_tasks,
        rubin_cache_cell_deg,
    )

    # Guardar celdas MAF por catalogo completo
    maf_bin_tables = []

    for field_name, catalog in pair_catalogs.items():
        cells = quantize_maf_cells(
            catalog,
            rubin_cache_cell_deg,
        )
        tmp = catalog.copy()
        tmp["maf_ra_bin"] = cells["maf_ra"].values
        tmp["maf_dec_bin"] = cells["maf_dec"].values
        maf_bin_tables.append(tmp)

        tmp.to_parquet(
            DIRS["catalogs"] / f"pair_catalog_with_maf_bins_{field_name}.parquet",
            index=False,
        )

    pd.concat(maf_bin_tables, ignore_index=True).to_parquet(
        DIRS["catalogs"] / "all_pair_catalogs_with_maf_bins.parquet",
        index=False,
    )

    # Crear contexto fork antes del prewarm para saber si se compartira memoria.
    try:
        mp_context = mp.get_context("fork")
    except ValueError:
        mp_context = None

    if PREWARM_MAF_CACHE:
        prewarm_maf_cache(
            pair_catalogs,
            event_tasks,
            rubin_cache_cell_deg,
            path_ephemerides,
        )

    # ============================================================
    # Config para workers
    # ============================================================

    worker_cfg = {
        "system_type": system_type,
        "model": model,
        "fit_model": fit_model,
        "fit_parallax": fit_parallax,
        "algo": algo,
        "use_roman": use_roman,
        "use_rubin": use_rubin,
        "path_ephemerides": path_ephemerides,
        "models_dir": str(DIRS["models"]),
        "fits_dir": str(DIRS["fits"]),
        "results_dir": str(DIRS["results"]),
        "logs_dir": str(DIRS["logs"]),
        "param_samplers_near_lenses": param_samplers_near_lenses,
        "fit_bounds_nopie": fit_bounds_nopie,
        "rubin_pointing_mode": rubin_pointing_mode,
        "rubin_cache_cell_deg": rubin_cache_cell_deg,
    }

    # ============================================================
    # Corrida paralela
    # ============================================================

    print("=" * 80)
    print("Iniciando corrida paralela")
    print(f"N_events_total:        {N_events_total}")
    print(f"N_WORKERS:             {N_WORKERS}")
    print(f"rubin_pointing_mode:   {rubin_pointing_mode}")
    print(f"rubin_cache_cell_deg:  {rubin_cache_cell_deg}")
    print(f"PREWARM_MAF_CACHE:     {PREWARM_MAF_CACHE}")
    print(f"VALIDATE_RUBIN_COVERAGE: {VALIDATE_RUBIN_COVERAGE}")
    print("=" * 80)

    summary_rows = []

    if mp_context is not None:
        executor_kwargs = {
            "max_workers": N_WORKERS,
            "initializer": init_worker,
            "initargs": (
                pair_catalogs,
                worker_cfg,
            ),
            "mp_context": mp_context,
        }
    else:
        executor_kwargs = {
            "max_workers": N_WORKERS,
            "initializer": init_worker,
            "initargs": (
                pair_catalogs,
                worker_cfg,
            ),
        }

    with ProcessPoolExecutor(**executor_kwargs) as executor:

        future_to_task = {
            executor.submit(run_single_event, task): task
            for task in event_tasks
        }

        for future in as_completed(future_to_task):

            task = future_to_task[future]
            global_i = int(task["global_i"])
            field_name = task["field_name"]

            try:
                row = future.result()

            except Exception as e:

                err = traceback.format_exc()

                log_dir = DIRS["logs"] / field_name
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / f"event_{global_i:06d}_executor_error.log"

                with open(log_file, "w") as f:
                    f.write(err)

                row = {
                    "global_i": global_i,
                    "local_i": task.get("local_i", np.nan),
                    "field_name": field_name,
                    "field_l_center": task.get("field_l_center", np.nan),
                    "field_b_center": task.get("field_b_center", np.nan),
                    "field_ra_center": task.get("field_ra_center", np.nan),
                    "field_dec_center": task.get("field_dec_center", np.nan),
                    "status": "executor_failed",
                    "error": str(e),
                    "log_file": str(log_file),
                    "model_dir": "",
                    "fit_dir": "",
                    "results_dir": "",
                    "rubin_pointing_mode": rubin_pointing_mode,
                    "rubin_cache_cell_deg": rubin_cache_cell_deg,
                    "maf_mode": "",
                    "maf_cache_mode": "",
                    "maf_source_ra": np.nan,
                    "maf_source_dec": np.nan,
                    "maf_ra": np.nan,
                    "maf_dec": np.nan,
                    "maf_cache_source": "",
                    "maf_n_obs": np.nan,
                }

            summary_rows.append(row)

            summary_df = save_summary(
                summary_rows,
                DIRS["logs"],
            )

            n_done = len(summary_rows)
            n_ok = (summary_df["status"] == "ok").sum()
            n_failed = (
                summary_df["status"].isin(
                    [
                        "failed",
                        "executor_failed",
                    ]
                )
            ).sum()

            print(
                f"[{n_done}/{N_events_total}] "
                f"{field_name} event {global_i} -> {row['status']} | "
                f"OK={n_ok}, failed={n_failed}"
            )

    # ============================================================
    # Resumen final
    # ============================================================

    summary_df = save_summary(
        summary_rows,
        DIRS["logs"],
    )

    n_ok = (summary_df["status"] == "ok").sum()
    n_failed = (
        summary_df["status"].isin(
            [
                "failed",
                "executor_failed",
            ]
        )
    ).sum()

    print("=" * 80)
    print("Corrida terminada")
    print(f"OK:     {n_ok}")
    print(f"Failed: {n_failed}")
    print(f"Run dir: {RUN_DIR}")
    print("=" * 80)

    if len(summary_df) > 0:
        print("Resumen por campo:")
        print(summary_df.groupby(["field_name", "status"]).size())
        print("=" * 80)

        print("Resumen MAF bins usados:")
        cols = [
            "field_name",
            "maf_mode",
            "maf_cache_mode",
            "maf_ra",
            "maf_dec",
            "maf_cache_source",
        ]
        existing_cols = [c for c in cols if c in summary_df.columns]
        print(summary_df[existing_cols].drop_duplicates().head(50))


if __name__ == "__main__":
    main()
