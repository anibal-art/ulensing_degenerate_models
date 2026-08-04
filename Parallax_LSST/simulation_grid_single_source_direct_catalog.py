#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grid search Rubin-only para estudiar paralaje anual en eventos de microlensing
para una unica fuente bien observada en todas las bandas de Rubin,
explorando una grilla fisica de lentes cercanas.

Objetivo:
- Construir una grilla fisica controlada en D_L, D_S, M_L, mu_rel, u0,
  t0 y direccion del vector de paralaje.
- Calcular de forma consistente pi_rel, theta_E, pi_E y t_E en cada punto.
- Simular curvas PSPL con paralaje.
- Ajustar las mismas curvas con PSPL sin paralaje.
- Medir si el modelo sin paralaje falla mediante Delta chi2 y si sesga
  los parametros ajustados.

Diferencia respecto a la version Monte Carlo:
- El catalogo AstroDataLab se usa solamente para elegir una fuente template
  con magnitudes finitas y suficientemente brillantes en ugrizy.
- Se conserva una unica posicion/cadencia/fuente para toda la corrida.
- Las propiedades fisicas del lente y de la geometria se imponen desde
  una grilla, no se toman aleatoriamente del catalogo pareado.
- Al final se generan graficos de D_L vs t_E coloreados por Delta chi2.
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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from astropy.coordinates import SkyCoord
import astropy.units as u

import ulens_params
importlib.reload(ulens_params)

from functions_roman_rubin import sim_fit

# AstroDataLab: se usa solo para obtener una fuente template.
# No construimos pares fuente-lente para este experimento de grilla.
try:
    from dl import queryClient as qc
    from dl.helpers.utils import convert
    ASTRODATALAB_AVAILABLE = True
except Exception as _e:
    qc = None
    convert = None
    ASTRODATALAB_AVAILABLE = False
    ASTRODATALAB_IMPORT_ERROR = repr(_e)


# ============================================================
# Constantes fisicas
# ============================================================

KAPPA_MAS_PER_MSUN = 8.144  # mas / Msun


# ============================================================
# Variables globales para workers
# ============================================================

GLOBAL_SOURCE_CATALOGS = None
GLOBAL_CFG = None


def init_worker(source_catalogs, cfg):
    """
    Inicializador de cada proceso.

    Guarda los catalogos fuente por campo y la configuracion como variables
    globales para no tener que pasarlas en cada evento.
    """

    global GLOBAL_SOURCE_CATALOGS
    global GLOBAL_CFG

    GLOBAL_SOURCE_CATALOGS = source_catalogs
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


def validate_and_filter_source_catalogs_for_rubin(
    source_catalogs,
    rubin_cache_cell_deg,
    path_ephemerides,
    catalogs_dir,
):
    """
    Valida las celdas MAF/Rubin de cada campo y filtra catalogos fuente.

    Esto evita que campos/celdas fuera del footprint efectivo de Rubin maten
    la corrida.
    """

    print("=" * 80)
    print("Validando cobertura Rubin/MAF por celda")
    print("=" * 80)

    catalogs_dir = Path(catalogs_dir)
    filtered_source_catalogs = {}
    validation_rows = []

    for field_name, catalog in source_catalogs.items():

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
        print(f"N fuentes antes de filtro: {len(catalog_with_bins)}")
        print(f"N celdas MAF a probar:     {len(cells_unique)}")

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

        print(f"N celdas validas:          {len(valid_cells)}")
        print(f"N fuentes despues filtro:  {len(filtered)}")

        if len(filtered) == 0:
            print(f"[warning] Campo {field_name} quedo sin fuentes luego del filtro Rubin. Se saltea.")
            continue

        filtered_source_catalogs[field_name] = filtered

        filtered.to_parquet(
            catalogs_dir / f"source_templates_{field_name}_rubin_valid.parquet",
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
    print(list(filtered_source_catalogs.keys()))
    print("=" * 80)

    if len(filtered_source_catalogs) == 0:
        raise RuntimeError(
            "Ningun campo tiene cobertura Rubin/MAF valida. "
            "Revisa las lineas de vision o el footprint usado."
        )

    return filtered_source_catalogs, validation_table


def prewarm_maf_cache(source_catalogs, event_tasks, rubin_cache_cell_deg, path_ephemerides):
    """
    Precalienta el cache de MAF para las celdas que van a usar los eventos.
    """

    print("=" * 80)
    print("Prewarming MAF cache")
    print("=" * 80)

    rows = []

    for task in event_tasks:
        field_name = task["field_name"]
        source_index = int(task["source_index"])
        catalog = source_catalogs[field_name]
        source = catalog.iloc[source_index]

        rows.append(
            {
                "field_name": field_name,
                "ra": float(source["ra"]),
                "dec": float(source["dec"]),
            }
        )

    if len(rows) == 0:
        print("No hay tareas para prewarm.")
        return

    selected = pd.DataFrame(rows)

    cells = quantize_maf_cells(
        selected,
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
# Catalogos fuente por linea de vision
# ============================================================

def _quote_sql_value(x):
    """
    Formatea valores simples para SQL.
    """

    if isinstance(x, str):
        return "'" + x.replace("'", "''") + "'"

    return str(x)


def _normalize_astrodatalab_source_columns(df):
    """
    Normaliza nombres de columnas de AstroDataLab para que sean compatibles
    con el simulador.
    """

    df = df.copy()

    rename_map = {
        "umag": "u",
        "gmag": "g",
        "rmag": "r",
        "imag": "i",
        "zmag": "z",
        "ymag": "Y",
        "logl": "logL",
        "logte": "logTe",
    }

    for old, new in rename_map.items():
        if old in df.columns:
            if new in df.columns and new != old:
                df = df.drop(columns=[new])
            df = df.rename(columns={old: new})

    if "W149" not in df.columns:
        if "Y" in df.columns:
            df["W149"] = df["Y"]
        elif "y" in df.columns:
            df["W149"] = df["y"]

    return df


def download_astrodatalab_source_catalog(
    ra_center,
    dec_center,
    radius,
    table="lsst_sim.simdr2",
    select_cols=None,
    extra_where="gc IN (1, 2)",
    mag_limits=None,
    limit=5000,
    timeout=300,
):
    """
    Descarga directamente estrellas de AstroDataLab en una region chica del cielo.

    Importante:
    - Esta funcion NO construye pares fuente-lente.
    - Solo se usa para elegir una unica fuente template con ra/dec y magnitudes.
    - Las propiedades fisicas del lente se imponen despues con la grilla.
    """

    if not ASTRODATALAB_AVAILABLE:
        raise ImportError(
            "No pude importar AstroDataLab dl.queryClient. "
            f"Error original: {globals().get('ASTRODATALAB_IMPORT_ERROR', '')}"
        )

    if select_cols is None:
        select_cols = [
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
        ]

    ra_center = float(ra_center)
    dec_center = float(dec_center)
    radius = float(radius)

    # Caja rectangular que contiene el circulo. Para estos campos no cruzamos RA=0.
    cos_dec = np.cos(np.deg2rad(dec_center))
    if abs(cos_dec) < 1e-3:
        dra = radius
    else:
        dra = radius / abs(cos_dec)

    ra_min = ra_center - dra
    ra_max = ra_center + dra
    dec_min = dec_center - radius
    dec_max = dec_center + radius

    where_terms = [
        f"ra >= {ra_min:.10f}",
        f"ra <= {ra_max:.10f}",
        f"dec >= {dec_min:.10f}",
        f"dec <= {dec_max:.10f}",
    ]

    # Corte circular aproximado, en grados, para no quedarnos con toda la caja.
    where_terms.append(
        "POWER((ra - {ra0:.10f})*COS({dec0:.10f}*PI()/180.0), 2) "
        "+ POWER(dec - {dec0:.10f}, 2) <= POWER({rad:.10f}, 2)".format(
            ra0=ra_center,
            dec0=dec_center,
            rad=radius,
        )
    )

    if extra_where is not None and str(extra_where).strip() != "":
        where_terms.append(f"({extra_where})")

    if mag_limits is not None:
        mag_col_map = {
            "u": "umag",
            "g": "gmag",
            "r": "rmag",
            "i": "imag",
            "z": "zmag",
            "Y": "ymag",
            "y": "ymag",
        }
        for band, lim in mag_limits.items():
            col = mag_col_map.get(band, band)
            where_terms.append(f"{col} IS NOT NULL")
            where_terms.append(f"{col} < {float(lim):.6f}")

    # Pedimos magnitudes finitas/no nulas en todas las bandas Rubin de interes.
    for col in ["umag", "gmag", "rmag", "imag", "zmag", "ymag"]:
        if col in select_cols:
            where_terms.append(f"{col} IS NOT NULL")

    sql = f"""
        SELECT {', '.join(select_cols)}
        FROM {table}
        WHERE {' AND '.join(where_terms)}
        LIMIT {int(limit)}
    """

    print("=" * 80)
    print("Descargando catalogo directo de fuentes desde AstroDataLab")
    print("NO se construyen pares fuente-lente en esta version.")
    print("=" * 80)
    print(sql)
    print("=" * 80)

    try:
        result = qc.query(sql=sql, fmt="csv", timeout=int(timeout))
    except TypeError:
        result = qc.query(sql=sql, timeout=int(timeout))

    try:
        df = convert(result, "pandas")
    except Exception:
        from io import StringIO
        df = pd.read_csv(StringIO(result))

    df = _normalize_astrodatalab_source_columns(df)

    for col in ["ra", "dec", "u", "g", "r", "i", "z", "Y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)

    print(f"Downloaded {len(df)} source candidates from AstroDataLab.")

    return df


def build_or_load_source_catalog_for_field(field, catalog_config_template, catalogs_dir):
    """
    Crea o carga un catalogo directo de fuentes para una linea de vision.

    Esta version evita completamente la busqueda de pairing.
    El catalogo contiene estrellas candidatas con posicion y magnitudes;
    la lente cercana se define luego por la grilla fisica.
    """

    field_name = field["field_name"]

    path_catalog = (
        Path(catalogs_dir)
        / f"source_candidates_direct_{field_name}.parquet"
    )

    if path_catalog.exists():
        print("=" * 80)
        print(f"Usando catalogo directo de fuentes existente para {field_name}:")
        print(path_catalog)
        print("=" * 80)

        catalog = pd.read_parquet(path_catalog)

    else:
        print("=" * 80)
        print(f"Generando catalogo directo de fuentes AstroDataLab para {field_name}")
        print(f"l,b       = {field['l_deg']:.3f}, {field['b_deg']:.3f} deg")
        print(f"RA,Dec    = {field['ra_center']:.6f}, {field['dec_center']:.6f} deg")
        print(f"radius    = {field['radius']:.4f} deg")
        print("No se hace pairing fuente-lente.")
        print("=" * 80)

        cfg = dict(catalog_config_template)
        cfg["ra_center"] = float(field["ra_center"])
        cfg["dec_center"] = float(field["dec_center"])
        cfg["radius"] = float(field["radius"])

        catalog = download_astrodatalab_source_catalog(**cfg)

        catalog.to_parquet(
            path_catalog,
            index=False,
        )

        print("=" * 80)
        print("Catalogo directo de fuentes guardado en:")
        print(path_catalog)
        print("=" * 80)

    catalog = catalog.copy().reset_index(drop=True)

    catalog = _normalize_astrodatalab_source_columns(catalog)

    if "W149" not in catalog.columns:
        if "Y" in catalog.columns:
            catalog["W149"] = catalog["Y"]
        elif "y" in catalog.columns:
            catalog["W149"] = catalog["y"]
        else:
            raise KeyError("No encuentro Y/y para construir W149.")

    for band in ["u", "g", "r", "i", "z", "Y", "W149"]:
        if band not in catalog.columns:
            raise KeyError(f"El catalogo fuente no tiene la banda {band}.")
        catalog[band] = pd.to_numeric(catalog[band], errors="coerce")

    catalog["field_name"] = field_name
    catalog["field_l_center"] = float(field["l_deg"])
    catalog["field_b_center"] = float(field["b_deg"])
    catalog["field_ra_center"] = float(field["ra_center"])
    catalog["field_dec_center"] = float(field["dec_center"])
    catalog["field_radius_deg"] = float(field["radius"])

    return field_name, catalog, path_catalog

def build_or_load_all_source_catalogs(fields, catalog_config_template, catalogs_dir):
    """
    Crea o carga catalogos fuente por linea de vision.
    """

    source_catalogs = {}
    catalog_paths = {}

    for field in fields:
        field_name, catalog, path_catalog = build_or_load_source_catalog_for_field(
            field,
            catalog_config_template,
            catalogs_dir,
        )

        if len(catalog) == 0:
            print(f"[warning] Catalogo fuente vacio para {field_name}. Se saltea.")
            continue

        source_catalogs[field_name] = catalog
        catalog_paths[field_name] = str(path_catalog)

    if len(source_catalogs) == 0:
        raise RuntimeError("No se pudo construir ningun catalogo fuente.")

    return source_catalogs, catalog_paths


# ============================================================
# Seleccion de una unica fuente template
# ============================================================

def select_single_good_rubin_source(
    source_catalogs,
    catalogs_dir,
    mag_limits=None,
    require_nearby_catalog_lens=False,
):
    """
    Selecciona una unica fuente template para correr toda la grilla.

    Criterios:
    - Coordenadas finitas.
    - Magnitudes finitas en u,g,r,i,z,Y.
    - Magnitudes mas brillantes que los limites especificados.
    - Si require_nearby_catalog_lens=True y existe D_L_kpc, se exige D_L_kpc <= 1.0.

    Entre las fuentes que pasan los cortes, se elige la de menor suma de
    magnitudes en las bandas Rubin, es decir, la mas brillante en promedio.
    """

    if mag_limits is None:
        mag_limits = {
            "u": 23.5,
            "g": 24.5,
            "r": 24.0,
            "i": 23.5,
            "z": 23.0,
            "Y": 22.5,
        }

    catalogs_dir = Path(catalogs_dir)

    candidate_tables = []

    for field_name, catalog in source_catalogs.items():

        df = catalog.copy().reset_index(drop=True)
        df["source_index_original"] = np.arange(len(df), dtype=int)
        df["selected_field_name"] = field_name

        mask = np.ones(len(df), dtype=bool)

        for col in ["ra", "dec"]:
            if col not in df.columns:
                raise KeyError(f"El catalogo no tiene columna {col}.")
            mask &= np.isfinite(pd.to_numeric(df[col], errors="coerce"))

        for band, limit in mag_limits.items():
            if band not in df.columns:
                raise KeyError(f"El catalogo no tiene banda {band}.")

            mag = pd.to_numeric(df[band], errors="coerce")
            mask &= np.isfinite(mag)
            mask &= mag < float(limit)

        if require_nearby_catalog_lens and "D_L_kpc" in df.columns:
            dl = pd.to_numeric(df["D_L_kpc"], errors="coerce")
            mask &= np.isfinite(dl)
            mask &= dl <= 1.0

        cand = df.loc[mask].copy()

        if len(cand) == 0:
            print(f"[warning] Campo {field_name}: ninguna fuente pasa cortes multibanda.")
            continue

        score = np.zeros(len(cand), dtype=float)
        for band in mag_limits:
            score += pd.to_numeric(cand[band], errors="coerce").values

        cand["source_selection_score"] = score
        candidate_tables.append(cand)

    if len(candidate_tables) == 0:
        raise RuntimeError(
            "No encontre ninguna fuente buena en todas las bandas Rubin. "
            "Relaja SOURCE_MAG_LIMITS o cambia el campo."
        )

    candidates = pd.concat(candidate_tables, ignore_index=True)
    candidates = candidates.sort_values("source_selection_score").reset_index(drop=True)

    selected = candidates.iloc[0].copy()
    selected_field = str(selected["selected_field_name"])

    selected_df = pd.DataFrame([selected])

    selected_df.to_csv(
        catalogs_dir / "selected_single_source_template.csv",
        index=False,
    )

    selected_df.to_parquet(
        catalogs_dir / "selected_single_source_template.parquet",
        index=False,
    )

    candidates.head(100).to_csv(
        catalogs_dir / "selected_source_candidates_top100.csv",
        index=False,
    )

    print("=" * 80)
    print("Fuente unica seleccionada para la grilla")
    print("=" * 80)
    print(f"Campo: {selected_field}")
    print(f"Indice original: {int(selected['source_index_original'])}")
    print(f"RA, Dec: {float(selected['ra']):.8f}, {float(selected['dec']):.8f}")
    if "gall" in selected and "galb" in selected:
        print(f"l, b: {float(selected['gall']):.8f}, {float(selected['galb']):.8f}")
    for band in mag_limits:
        print(f"{band}: {float(selected[band]):.4f}")
    if "D_L_kpc" in selected:
        print(f"D_L catalog lens [kpc]: {float(selected['D_L_kpc']):.4f}")
    if "D_S_kpc" in selected:
        print(f"D_S catalog source [kpc]: {float(selected['D_S_kpc']):.4f}")
    print("=" * 80)

    # El catalogo que usara sim_fit contiene una sola fila.
    source_catalog_single = selected_df.copy().reset_index(drop=True)
    source_catalog_single["source_index_original"] = int(selected["source_index_original"])

    return {selected_field: source_catalog_single}, selected_df


# ============================================================
# Grilla fisica
# ============================================================

def microlensing_grid_quantities(DL_kpc, DS_kpc, ML_Msun, mu_rel_masyr):
    """
    Calcula pi_rel, theta_E, pi_E y t_E para un punto de la grilla.

    Para distancias en kpc:
        pi_rel[mas] = 1/DL[kpc] - 1/DS[kpc]
    """

    DL_kpc = float(DL_kpc)
    DS_kpc = float(DS_kpc)
    ML_Msun = float(ML_Msun)
    mu_rel_masyr = float(mu_rel_masyr)

    pi_rel_mas = (1.0 / DL_kpc) - (1.0 / DS_kpc)

    if pi_rel_mas <= 0.0:
        return None

    thetaE_mas = np.sqrt(KAPPA_MAS_PER_MSUN * ML_Msun * pi_rel_mas)
    piE = pi_rel_mas / thetaE_mas
    tE_days = thetaE_mas / mu_rel_masyr * 365.25

    return {
        "grid_pi_rel_mas": float(pi_rel_mas),
        "grid_thetaE_mas": float(thetaE_mas),
        "grid_piE": float(piE),
        "grid_tE_days": float(tE_days),
    }


def build_grid_points(
    DL_grid_kpc,
    DS_grid_kpc,
    ML_grid_Msun,
    mu_rel_grid_masyr,
    u0_grid,
    t0_grid,
    phi_pi_grid_rad,
    tE_min_days=None,
    tE_max_days=None,
):
    """
    Construye la grilla fisica completa, con filtro opcional en t_E.
    """

    rows = []

    for DL_kpc in DL_grid_kpc:
        for DS_kpc in DS_grid_kpc:

            if float(DS_kpc) <= float(DL_kpc):
                continue

            for ML_Msun in ML_grid_Msun:
                for mu_rel_masyr in mu_rel_grid_masyr:

                    q = microlensing_grid_quantities(
                        DL_kpc,
                        DS_kpc,
                        ML_Msun,
                        mu_rel_masyr,
                    )

                    if q is None:
                        continue

                    tE_days = q["grid_tE_days"]

                    if tE_min_days is not None and tE_days < float(tE_min_days):
                        continue

                    if tE_max_days is not None and tE_days > float(tE_max_days):
                        continue

                    for u0 in u0_grid:
                        for t0 in t0_grid:
                            for phi_pi_rad in phi_pi_grid_rad:

                                piE = q["grid_piE"]

                                row = {
                                    "grid_D_L_kpc": float(DL_kpc),
                                    "grid_D_S_kpc": float(DS_kpc),
                                    "grid_M_L_Msun": float(ML_Msun),
                                    "grid_mu_rel_masyr": float(mu_rel_masyr),
                                    "grid_u0": float(u0),
                                    "grid_t0": float(t0),
                                    "grid_phi_pi_rad": float(phi_pi_rad),
                                    "grid_piEN": float(piE * np.cos(phi_pi_rad)),
                                    "grid_piEE": float(piE * np.sin(phi_pi_rad)),
                                }

                                row.update(q)
                                rows.append(row)

    grid = pd.DataFrame(rows)

    if len(grid) == 0:
        raise RuntimeError("La grilla fisica quedo vacia. Revisa rangos/filtros.")

    grid["grid_id"] = np.arange(len(grid), dtype=int)

    return grid


def thin_grid_points(grid, max_points=None, random_state=12345):
    """
    Reduce la grilla de forma deterministica si max_points no es None.

    Esto es util para hacer una corrida inicial manejable. Para correr la
    grilla completa, poner max_points=None.
    """

    if max_points is None:
        return grid.copy().reset_index(drop=True)

    max_points = int(max_points)

    if len(grid) <= max_points:
        return grid.copy().reset_index(drop=True)

    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(len(grid), size=max_points, replace=False)
    idx = np.sort(idx)

    out = grid.iloc[idx].copy().reset_index(drop=True)
    out["grid_thinned_from_n"] = len(grid)

    return out


def build_event_tasks(fields, source_catalogs, grid_points, max_grid_points_per_field=None, thinning_random_state=12345):
    """
    Arma una lista de tareas de simulacion en grilla.

    Cada campo recibe la misma grilla fisica. En esta version el catalogo
    de fuentes ya fue reducido a una unica fuente, por lo que todos los
    puntos de la grilla se simulan con la misma fuente/cadencia/magnitudes.
    """

    tasks = []
    global_i = 0

    for field in fields:
        field_name = field["field_name"]

        if field_name not in source_catalogs:
            continue

        source_catalog = source_catalogs[field_name]
        grid_field = thin_grid_points(
            grid_points,
            max_points=max_grid_points_per_field,
            random_state=thinning_random_state + int(round(float(field["l_deg"]) * 10.0)),
        )

        for local_i, (_, g) in enumerate(grid_field.iterrows()):

            # Usamos siempre la unica fuente seleccionada.
            source_index = 0

            task = {
                "global_i": int(global_i),
                "local_i": int(local_i),
                "field_name": field_name,
                "source_index": source_index,
                "field_l_center": float(field["l_deg"]),
                "field_b_center": float(field["b_deg"]),
                "field_ra_center": float(field["ra_center"]),
                "field_dec_center": float(field["dec_center"]),
            }

            for key, val in g.to_dict().items():
                if isinstance(val, (np.integer,)):
                    task[key] = int(val)
                elif isinstance(val, (np.floating,)):
                    task[key] = float(val)
                else:
                    task[key] = val

            tasks.append(task)
            global_i += 1

    return tasks


# ============================================================
# Construccion de fila sintetica para sim_fit
# ============================================================

def build_single_row_pair_catalog_from_grid(source_row, task):
    """
    Construye un catalogo de una sola fila compatible con sim_fit.

    La fuente aporta:
        - posicion
        - magnitudes
        - metadatos de campo

    La grilla aporta:
        - D_L, D_S
        - mu_rel
        - theta_rad, usado como direccion del vector de paralaje
        - parametros fisicos fijos
    """

    row = source_row.copy()

    DL_kpc = float(task["grid_D_L_kpc"])
    DS_kpc = float(task["grid_D_S_kpc"])
    mu_rel = float(task["grid_mu_rel_masyr"])
    phi = float(task["grid_phi_pi_rad"])

    # Distancias esperadas internamente por las rutinas: pc.
    row["D_L"] = 1000.0 * DL_kpc
    row["D_S"] = 1000.0 * DS_kpc
    row["D_L_kpc"] = DL_kpc
    row["D_S_kpc"] = DS_kpc

    # Cinematica impuesta por la grilla.
    row["mu_rel"] = mu_rel
    row["theta_rad"] = phi
    row["traj_angle"] = phi

    # Coordenadas de lente: no afectan la curva en este nivel,
    # pero se guardan para trazabilidad.
    row["lens_ra"] = row.get("ra", np.nan)
    row["lens_dec"] = row.get("dec", np.nan)

    if "gall" in row:
        row["lens_gall"] = row["gall"]
    if "galb" in row:
        row["lens_galb"] = row["galb"]

    # Magnitudes requeridas.
    if "Y" not in row and "ymag" in row:
        row["Y"] = row["ymag"]

    if "W149" not in row:
        row["W149"] = row["Y"]

    for band in ["u", "g", "r", "i", "z", "Y", "W149"]:
        if band not in row:
            raise KeyError(f"Falta banda {band} en source_row.")

    # Metadatos de grilla para guardar luego en los parquet.
    for key, val in task.items():
        if key.startswith("grid_") or key in [
            "field_name",
            "field_l_center",
            "field_b_center",
            "field_ra_center",
            "field_dec_center",
            "source_index",
            "global_i",
            "local_i",
        ]:
            row[key] = val

    return pd.DataFrame([row])


def fixed_param_samplers_from_task(task):
    """
    Samplers fijos para que cada evento use exactamente el punto de la grilla.
    """

    return {
        "star_mass": {
            "type": "fixed",
            "value": float(task["grid_M_L_Msun"]),
        },
        "mass_planet": {
            "type": "fixed",
            "value": 0.0,
        },
        "u0": {
            "type": "fixed",
            "value": float(task["grid_u0"]),
        },
        "t0": {
            "type": "fixed",
            "value": float(task["grid_t0"]),
        },
    }


def task_metadata_for_outputs(task):
    """
    Metadatos que se agregan a los parquet true/fit para facilitar el analisis.
    """

    keys = [
        "global_i",
        "local_i",
        "field_name",
        "source_index",
        "field_l_center",
        "field_b_center",
        "field_ra_center",
        "field_dec_center",
        "grid_id",
        "grid_D_L_kpc",
        "grid_D_S_kpc",
        "grid_M_L_Msun",
        "grid_mu_rel_masyr",
        "grid_u0",
        "grid_t0",
        "grid_t0_phase_days",
        "grid_phi_pi_rad",
        "grid_pi_rel_mas",
        "grid_thetaE_mas",
        "grid_piE",
        "grid_piEN",
        "grid_piEE",
        "grid_tE_days",
    ]

    meta = {}

    for key in keys:
        meta[key] = task.get(key, np.nan)

    return meta


def add_metadata_to_result_parquets(event_results_dir, metadata):
    """
    Agrega metadatos de grilla/campo a todos los parquet de resultados
    generados para un evento.
    """

    event_results_dir = Path(event_results_dir)

    files = sorted(event_results_dir.rglob("*.parquet"))

    for f in files:
        try:
            df = pd.read_parquet(f)

            for key, val in metadata.items():
                df[key] = val

            df.to_parquet(f, index=False)

        except Exception as e:
            print(f"[warning] No pude agregar metadatos a {f}: {repr(e)}")


# ============================================================
# Corrida de un evento
# ============================================================

def run_single_event(task):
    """
    Ejecuta un unico evento de la grilla dentro de un worker.
    """

    cfg = GLOBAL_CFG
    source_catalogs = GLOBAL_SOURCE_CATALOGS

    global_i = int(task["global_i"])
    local_i = int(task["local_i"])
    field_name = task["field_name"]
    source_index = int(task["source_index"])

    source_catalog = source_catalogs[field_name]
    source_row = source_catalog.iloc[source_index]

    single_pair_catalog = build_single_row_pair_catalog_from_grid(
        source_row,
        task,
    )

    event_tag = f"grid_{global_i:07d}"

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
        "source_index": source_index,
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

    # Agregar columnas de grilla al resumen.
    row.update(task_metadata_for_outputs(task))

    try:
        with open(event_log_file, "w") as log:
            with redirect_stdout(log), redirect_stderr(log):

                print("=" * 80)
                print(
                    f"Evento global_i={global_i}, local_i={local_i}, "
                    f"field={field_name}, source_index={source_index}"
                )
                print("=" * 80)
                print("Rubin pointing mode:", cfg["rubin_pointing_mode"])
                print("Rubin cache cell deg:", cfg["rubin_cache_cell_deg"])
                print("Grid point:")
                print(json.dumps(task_metadata_for_outputs(task), indent=4))
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
                    pair_catalog=single_pair_catalog,

                    use_roman=cfg["use_roman"],
                    use_rubin=cfg["use_rubin"],

                    param_samplers=fixed_param_samplers_from_task(task),

                    fit_model=cfg["fit_model"],
                    fit_parallax=cfg["fit_parallax"],
                    fit_bounds=cfg["fit_bounds_nopie"],

                    rubin_pointing_mode=cfg["rubin_pointing_mode"],
                    rubin_cache_cell_deg=cfg["rubin_cache_cell_deg"],

                    # Importante para saber si el evento fue realmente fiteado
                    # o si fue rechazado por los criterios de detección.
                    return_data=True,
                )

                row["sim_fit_return_type"] = type(result).__name__

                if isinstance(result, dict):
                    row["sim_fit_status"] = result.get("status", "")
                    if result.get("status", "") == "rejected":
                        row["status"] = "rejected"
                    elif result.get("status", "") == "fitted":
                        row["status"] = "ok"
                else:
                    row["sim_fit_status"] = "legacy_return"

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

                # Agregar metadatos de grilla a los parquet de resultados.
                metadata = task_metadata_for_outputs(task)
                metadata.update(
                    {
                        "maf_mode": row["maf_mode"],
                        "maf_cache_mode": row["maf_cache_mode"],
                        "maf_source_ra": row["maf_source_ra"],
                        "maf_source_dec": row["maf_source_dec"],
                        "maf_ra": row["maf_ra"],
                        "maf_dec": row["maf_dec"],
                        "maf_cache_source": row["maf_cache_source"],
                        "maf_n_obs": row["maf_n_obs"],
                    }
                )

                add_metadata_to_result_parquets(
                    event_results_dir,
                    metadata,
                )

                print("=" * 80)
                print("Evento terminado correctamente")
                print("=" * 80)

        if row.get("status") not in ["rejected", "failed", "executor_failed"]:
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
# Analisis rapido de la grilla: D_L vs t_E coloreado por Delta chi2
# ============================================================

def read_parquet_tree(path):
    """
    Lee todos los parquet debajo de path.
    """

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
    """
    Lee resultados true o fit_rr de forma flexible.
    """

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


def safe_col_local(df, candidates):
    """
    Devuelve la primera columna existente.
    """

    for c in candidates:
        if c in df.columns:
            return c
    return None


def choose_merge_columns_for_grid(true, fit):
    """
    Columnas para unir true y fit.
    Preferimos global_i porque es unico en la grilla.
    """

    for cols in [
        ["global_i"],
        ["Source", "Set"],
        ["Source"],
        ["grid_id", "field_name", "source_index"],
    ]:
        if all(c in true.columns for c in cols) and all(c in fit.columns for c in cols):
            return cols

    raise KeyError(
        "No pude encontrar columnas para merge entre true y fit.\n"
        f"true columns = {true.columns.tolist()}\n"
        f"fit columns = {fit.columns.tolist()}"
    )


def normalize_fit_for_grid(fit):
    """
    Normaliza nombres de chi2 en fit.
    """

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


def get_col_after_merge(df, base):
    """
    Busca columna base, base_true o base_fit despues del merge.
    """

    return safe_col_local(df, [base, f"{base}_true", f"{base}_fit"])


def save_grid_scatter_plot(
    data,
    color_col,
    color_label,
    filename,
    plots_dir,
    log_color=True,
):
    """
    Grafico D_L vs t_E coloreado por una cantidad escalar.
    """

    needed = ["grid_D_L_kpc_used", "grid_tE_days_used", color_col]
    for col in needed:
        if col not in data.columns:
            print(f"Skip {filename}: falta {col}")
            return

    mask = (
        np.isfinite(data["grid_D_L_kpc_used"])
        & np.isfinite(data["grid_tE_days_used"])
        & np.isfinite(data[color_col])
        & (data["grid_D_L_kpc_used"] > 0)
        & (data["grid_tE_days_used"] > 0)
    )

    if mask.sum() == 0:
        print(f"Skip {filename}: no hay puntos validos")
        return

    x = data.loc[mask, "grid_D_L_kpc_used"].values
    y = data.loc[mask, "grid_tE_days_used"].values
    c = data.loc[mask, color_col].values

    norm = None
    c_plot = c.copy()

    if log_color:
        positive = c_plot[np.isfinite(c_plot) & (c_plot > 0)]
        if len(positive) > 0:
            floor = max(np.nanmin(positive), 1e-6)
            c_plot = np.where(c_plot > 0, c_plot, floor)
            norm = mcolors.LogNorm(
                vmin=floor,
                vmax=np.nanmax(c_plot),
            )

    plt.figure(figsize=(7.4, 5.6))

    sc = plt.scatter(
        x,
        y,
        c=c_plot,
        s=28,
        alpha=0.82,
        linewidths=0,
        rasterized=True,
        norm=norm,
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$D_L$ [kpc]")
    plt.ylabel(r"$t_E$ [d]")
    plt.title(r"Single-source grid: $D_L$ vs $t_E$")

    cb = plt.colorbar(sc)
    cb.set_label(color_label)

    plt.grid(True, which="major", alpha=0.25, linewidth=0.6)
    plt.grid(True, which="minor", alpha=0.12, linewidth=0.4)
    plt.tight_layout()

    out = Path(plots_dir) / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


def make_single_source_grid_diagnostic_plots(run_dir):
    """
    Lee resultados de la grilla y genera graficos de D_L vs t_E coloreados por Delta chi2.
    """

    run_dir = Path(run_dir)
    results_dir = run_dir / "results"
    plots_dir = run_dir / "plots"
    tables_dir = run_dir / "tables"

    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    try:
        true = read_result_kind(results_dir, "true")
        fit = read_result_kind(results_dir, "fit_rr")
    except Exception as e:
        print("No pude generar plots de diagnostico porque faltan resultados:", repr(e))
        return None

    fit = normalize_fit_for_grid(fit)

    id_cols = choose_merge_columns_for_grid(true, fit)

    df = pd.merge(
        true,
        fit,
        on=id_cols,
        how="inner",
        suffixes=("_true", "_fit"),
    )

    if len(df) == 0:
        print("Merge true-fit vacio. No genero plots.")
        return None

    chi2_fit_col = safe_col_local(df, ["chi2", "chi2_fit", "chichi", "chichi_fit"])
    chi2_true_col = safe_col_local(df, ["chi2_true", "chi2_true_true", "chi2_true_fit"])
    n_data_col = safe_col_local(df, ["n_data_true", "n_data_true_true", "n_data_true_fit"])

    if chi2_fit_col is None or chi2_true_col is None:
        print("No encuentro chi2 fit o chi2 true. No genero plots.")
        return None

    dl_col = get_col_after_merge(df, "grid_D_L_kpc")
    te_col = get_col_after_merge(df, "grid_tE_days")

    if dl_col is None or te_col is None:
        print("No encuentro grid_D_L_kpc o grid_tE_days. No genero plots.")
        return None

    df["grid_D_L_kpc_used"] = pd.to_numeric(df[dl_col], errors="coerce")
    df["grid_tE_days_used"] = pd.to_numeric(df[te_col], errors="coerce")
    df["chi2_nopie"] = pd.to_numeric(df[chi2_fit_col], errors="coerce")
    df["chi2_true_model"] = pd.to_numeric(df[chi2_true_col], errors="coerce")
    df["delta_chi2_true"] = df["chi2_nopie"] - df["chi2_true_model"]

    if n_data_col is not None:
        df["n_data_true_model"] = pd.to_numeric(df[n_data_col], errors="coerce")
        df["delta_chi2_true_per_point"] = df["delta_chi2_true"] / df["n_data_true_model"]
    else:
        df["n_data_true_model"] = np.nan
        df["delta_chi2_true_per_point"] = np.nan

    df["delta_chi2_detectable"] = df["delta_chi2_true"] > 100.0

    df.to_parquet(
        tables_dir / "single_source_grid_diagnostics.parquet",
        index=False,
    )

    df.to_csv(
        tables_dir / "single_source_grid_diagnostics.csv",
        index=False,
    )

    save_grid_scatter_plot(
        df,
        "delta_chi2_true",
        r"$\Delta\chi^2$",
        "grid_single_source_DL_vs_tE_colored_delta_chi2.png",
        plots_dir,
        log_color=True,
    )

    save_grid_scatter_plot(
        df,
        "delta_chi2_true_per_point",
        r"$\Delta\chi^2/N_{\rm data}$",
        "grid_single_source_DL_vs_tE_colored_delta_chi2_per_point.png",
        plots_dir,
        log_color=True,
    )

    save_grid_scatter_plot(
        df,
        "delta_chi2_detectable",
        r"$\Delta\chi^2>100$",
        "grid_single_source_DL_vs_tE_detectable_flag.png",
        plots_dir,
        log_color=False,
    )

    print("=" * 80)
    print("Single-source grid diagnostics")
    print("=" * 80)
    print(f"N merged events: {len(df)}")
    print(f"N detectable Delta chi2 > 100: {int(df['delta_chi2_detectable'].sum())}")
    print(f"Diagnostics table: {tables_dir / 'single_source_grid_diagnostics.parquet'}")
    print("=" * 80)

    return df


# ============================================================
# Main
# ============================================================

def main():

    # ============================================================
    # Configuracion general
    # ============================================================

    BASE_DIR = Path("/home/anibal/Parallax_LSST")

    RUN_NAME = "Grid_SingleSource_directCatalog_near_lens_RubinOnly_PSPLparallax_fitNoPiE_MAFbin020_JDt0"

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
    # Parametros de modelo y fit
    # ============================================================

    system_type = "BH"
    model = "PSPL"
    fit_model = "PSPL"
    fit_parallax = False
    algo = "TRF"

    use_roman = False
    use_rubin = True

    # En muchas maquinas 6-8 workers puede ser mas rapido que 16
    # por I/O y memoria.
    N_WORKERS = 8

    # ============================================================
    # Lineas de vision del plano galactico
    # ============================================================

    # Para este experimento inicial usamos UNA sola linea de vision y
    # una sola fuente template. Cambia estos valores si queres probar otro campo.
    GALACTIC_FIELDS_INPUT = [
        {"l_deg": 30.0, "b_deg": 0.0, "radius": 0.20, "random_state": 1001},
    ]

    fields = prepare_galactic_fields(GALACTIC_FIELDS_INPUT)

    # ============================================================
    # Configuracion MAF/Rubin
    # ============================================================

    rubin_pointing_mode = "source"
    rubin_cache_cell_deg = 0.20

    PREWARM_MAF_CACHE = True
    VALIDATE_RUBIN_COVERAGE = True

    # ============================================================
    # Seleccion de fuente unica
    # ============================================================
    # La fuente template debe ser suficientemente brillante en todas las
    # bandas Rubin para que la comparacion entre PSPL con y sin paralaje
    # no este dominada por una banda sin S/N.

    SOURCE_MAG_LIMITS = {
        "u": 23.5,
        "g": 24.5,
        "r": 24.0,
        "i": 23.5,
        "z": 23.0,
        "Y": 22.5,
    }

    # ============================================================
    # Grilla fisica
    # ============================================================
    # Se eligen lentes cercanas y se incluye un rango amplio de masas y
    # movimientos propios para producir eventos cortos y largos.
    #
    # Para correr toda la grilla, poner MAX_GRID_POINTS_PER_FIELD = None.
    # Para una corrida inicial manejable, dejarlo en 1000 o menos.

    # Lentes cercanas. Se usa una grilla relativamente densa en D_L
    # porque el objetivo principal es ver como cambia Delta chi2 en el
    # plano D_L--t_E.
    DL_GRID_KPC = np.array([0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00])
    DS_GRID_KPC = np.array([2.0, 4.0, 8.0, 12.0])

    ML_GRID_MSUN = np.array([0.3, 1.0, 3.0, 10.0, 30.0])
    MU_REL_GRID_MASYR = np.array([2.0, 5.0, 10.0, 20.0])

    U0_GRID = np.array([0.10, 0.30, 0.60])

    # Fases anuales del maximo del evento.
    # Ojo: pyLIMA/Rubin trabaja en JD absolutos.
    # Si usamos t0 = 0, 91, ... el pico cae miles de días fuera
    # de la ventana Rubin y todos los eventos son rechazados.
    T0_REFERENCE_JD = 2460413.013828608
    T0_PHASE_GRID_DAYS = np.array([0.0, 91.3125, 182.625, 273.9375])
    T0_GRID = T0_REFERENCE_JD + T0_PHASE_GRID_DAYS

    # Direccion del vector de paralaje / trayectoria.
    PHI_PI_GRID_RAD = np.linspace(
        0.0,
        2.0 * np.pi,
        4,
        endpoint=False,
    )

    # Filtramos solo eventos dentro de un rango manejable de duraciones.
    TE_MIN_DAYS = 10.0
    TE_MAX_DAYS = 10000.0

    # Para una unica fuente, por defecto corremos toda la grilla.
    # Si queres una prueba rapida, poner por ejemplo MAX_GRID_POINTS_PER_FIELD=200.
    MAX_GRID_POINTS_PER_FIELD = None
    GRID_THINNING_RANDOM_STATE = 20260713

    grid_full = build_grid_points(
        DL_GRID_KPC,
        DS_GRID_KPC,
        ML_GRID_MSUN,
        MU_REL_GRID_MASYR,
        U0_GRID,
        T0_GRID,
        PHI_PI_GRID_RAD,
        tE_min_days=TE_MIN_DAYS,
        tE_max_days=TE_MAX_DAYS,
    )

    grid_full["grid_t0_phase_days"] = grid_full["grid_t0"] - T0_REFERENCE_JD

    # ============================================================
    # Bounds del fit NoPiE
    # ============================================================

    fit_bounds_nopie = {
        "t0": {
            "type": "center_width",
            "half_width": 1000.0,
        },
        "u0": [
            -5.0,
            5.0,
        ],
        "tE": [
            1.0,
            20000.0,
        ],
    }

    # ============================================================
    # Configuracion del catalogo AstroDataLab
    # ============================================================
    # Este catalogo se usa solo como fuente de magnitudes y posiciones.
    # Las distancias fisicas del evento son reemplazadas por la grilla.

    # Catalogo directo de fuentes: sin pairing fuente-lente.
    # Pedimos candidatos que ya pasen los cortes de magnitud para no descargar
    # miles de estrellas innecesarias.
    N_SOURCE_CANDIDATES = 2000

    catalog_config_template = {
        "ra_center": None,
        "dec_center": None,
        "radius": None,
        "table": "lsst_sim.simdr2",
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
        "extra_where": "gc IN (1, 2)",
        "mag_limits": SOURCE_MAG_LIMITS,
        "limit": N_SOURCE_CANDIDATES,
        "timeout": 300,
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
        "N_WORKERS": N_WORKERS,
        "path_ephemerides": path_ephemerides,
        "fit_bounds_nopie": fit_bounds_nopie,
        "catalog_config_template": catalog_config_template,
        "rubin_pointing_mode": rubin_pointing_mode,
        "rubin_cache_cell_deg": rubin_cache_cell_deg,
        "PREWARM_MAF_CACHE": PREWARM_MAF_CACHE,
        "VALIDATE_RUBIN_COVERAGE": VALIDATE_RUBIN_COVERAGE,
        "SOURCE_MAG_LIMITS": SOURCE_MAG_LIMITS,
        "fields": fields,
        "DL_GRID_KPC": DL_GRID_KPC.tolist(),
        "DS_GRID_KPC": DS_GRID_KPC.tolist(),
        "ML_GRID_MSUN": ML_GRID_MSUN.tolist(),
        "MU_REL_GRID_MASYR": MU_REL_GRID_MASYR.tolist(),
        "U0_GRID": U0_GRID.tolist(),
        "T0_REFERENCE_JD": float(T0_REFERENCE_JD),
        "T0_PHASE_GRID_DAYS": T0_PHASE_GRID_DAYS.tolist(),
        "T0_GRID": T0_GRID.tolist(),
        "PHI_PI_GRID_RAD": PHI_PI_GRID_RAD.tolist(),
        "TE_MIN_DAYS": TE_MIN_DAYS,
        "TE_MAX_DAYS": TE_MAX_DAYS,
        "MAX_GRID_POINTS_PER_FIELD": MAX_GRID_POINTS_PER_FIELD,
        "GRID_THINNING_RANDOM_STATE": GRID_THINNING_RANDOM_STATE,
        "N_GRID_FULL": int(len(grid_full)),
    }

    with open(DIRS["config"] / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)

    fields_df = pd.DataFrame(fields)
    fields_df.to_csv(DIRS["config"] / "fields.csv", index=False)
    fields_df.to_parquet(DIRS["config"] / "fields.parquet", index=False)

    grid_full.to_csv(DIRS["config"] / "grid_full.csv", index=False)
    grid_full.to_parquet(DIRS["config"] / "grid_full.parquet", index=False)

    print("=" * 80)
    print("Campos galacticos")
    print("=" * 80)
    print(fields_df[["field_name", "l_deg", "b_deg", "ra_center", "dec_center", "radius"]])
    print("=" * 80)

    print("=" * 80)
    print("Grilla fisica completa")
    print("=" * 80)
    print(f"N grid full: {len(grid_full)}")
    print(grid_full[[
        "grid_D_L_kpc",
        "grid_D_S_kpc",
        "grid_M_L_Msun",
        "grid_mu_rel_masyr",
        "grid_tE_days",
        "grid_piE",
    ]].describe())
    print("=" * 80)

    # ============================================================
    # Crear o cargar catalogos fuente por linea de vision
    # ============================================================

    source_catalogs, catalog_paths = build_or_load_all_source_catalogs(
        fields,
        catalog_config_template,
        DIRS["catalogs"],
    )

    # ============================================================
    # Validar cobertura Rubin/MAF y filtrar campos/celdas invalidas
    # ============================================================

    if VALIDATE_RUBIN_COVERAGE:
        source_catalogs, maf_validation_table = validate_and_filter_source_catalogs_for_rubin(
            source_catalogs,
            rubin_cache_cell_deg,
            path_ephemerides,
            DIRS["catalogs"],
        )
    else:
        maf_validation_table = pd.DataFrame()

    # Reducir a una unica fuente bien observada en todas las bandas Rubin.
    source_catalogs, selected_source_table = select_single_good_rubin_source(
        source_catalogs,
        DIRS["catalogs"],
        mag_limits=SOURCE_MAG_LIMITS,
        require_nearby_catalog_lens=False,
    )

    # Mantener solo el campo donde vive la fuente seleccionada.
    fields = [
        field for field in fields
        if field["field_name"] in source_catalogs
    ]

    all_source_catalog = pd.concat(
        source_catalogs.values(),
        ignore_index=True,
    )

    all_source_catalog.to_parquet(
        DIRS["catalogs"] / "all_source_templates_combined_rubin_valid.parquet",
        index=False,
    )

    # ============================================================
    # Chequeo de catalogos fuente
    # ============================================================

    print("=" * 80)
    print("Resumen de catalogos fuente")
    print("=" * 80)

    for field_name, catalog in source_catalogs.items():
        print(f"Campo: {field_name}")
        print("N fuentes:", len(catalog))
        cols_summary = [c for c in ["D_L_kpc", "D_S_kpc", "mu_rel", "gall", "galb"] if c in catalog.columns]
        if len(cols_summary) > 0:
            print(catalog[cols_summary].describe())
        print("-" * 80)

    # ============================================================
    # Construir tareas de grilla
    # ============================================================

    event_tasks = build_event_tasks(
        fields,
        source_catalogs,
        grid_full,
        max_grid_points_per_field=MAX_GRID_POINTS_PER_FIELD,
        thinning_random_state=GRID_THINNING_RANDOM_STATE,
    )

    tasks_df = pd.DataFrame(event_tasks)
    tasks_df.to_csv(DIRS["config"] / "event_tasks_grid.csv", index=False)
    tasks_df.to_parquet(DIRS["config"] / "event_tasks_grid.parquet", index=False)

    N_events_total = len(event_tasks)

    print("=" * 80)
    print("Tareas de simulacion en grilla")
    print("=" * 80)
    print(f"N fields:       {len(source_catalogs)}")
    print(f"N grid full:    {len(grid_full)}")
    print(f"Max grid/field: {MAX_GRID_POINTS_PER_FIELD}")
    print(f"N events total: {N_events_total}")

    if len(tasks_df) > 0:
        print(tasks_df.groupby("field_name").size())

    print("=" * 80)

    # ============================================================
    # Diagnostico MAF y prewarm
    # ============================================================

    # Guardar celdas MAF por catalogo completo
    maf_bin_tables = []

    for field_name, catalog in source_catalogs.items():
        cells = quantize_maf_cells(
            catalog,
            rubin_cache_cell_deg,
        )
        tmp = catalog.copy()
        tmp["maf_ra_bin"] = cells["maf_ra"].values
        tmp["maf_dec_bin"] = cells["maf_dec"].values
        maf_bin_tables.append(tmp)

        tmp.to_parquet(
            DIRS["catalogs"] / f"source_catalog_with_maf_bins_{field_name}.parquet",
            index=False,
        )

    pd.concat(maf_bin_tables, ignore_index=True).to_parquet(
        DIRS["catalogs"] / "all_source_catalogs_with_maf_bins.parquet",
        index=False,
    )

    try:
        mp_context = mp.get_context("fork")
    except ValueError:
        mp_context = None

    if PREWARM_MAF_CACHE:
        prewarm_maf_cache(
            source_catalogs,
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
        "fit_bounds_nopie": fit_bounds_nopie,
        "rubin_pointing_mode": rubin_pointing_mode,
        "rubin_cache_cell_deg": rubin_cache_cell_deg,
    }

    # ============================================================
    # Corrida paralela
    # ============================================================

    print("=" * 80)
    print("Iniciando corrida paralela en grilla")
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
                source_catalogs,
                worker_cfg,
            ),
            "mp_context": mp_context,
        }
    else:
        executor_kwargs = {
            "max_workers": N_WORKERS,
            "initializer": init_worker,
            "initargs": (
                source_catalogs,
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
                log_file = log_dir / f"grid_{global_i:07d}_executor_error.log"

                with open(log_file, "w") as f:
                    f.write(err)

                row = {
                    "global_i": global_i,
                    "local_i": task.get("local_i", np.nan),
                    "field_name": field_name,
                    "source_index": task.get("source_index", np.nan),
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

                row.update(task_metadata_for_outputs(task))

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
                f"{field_name} grid event {global_i} -> {row['status']} | "
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
    print("Corrida en grilla terminada")
    print(f"OK:     {n_ok}")
    print(f"Failed: {n_failed}")
    print(f"Run dir: {RUN_DIR}")
    print("=" * 80)

    # Generar diagnosticos automaticos de la grilla.
    make_single_source_grid_diagnostic_plots(RUN_DIR)

    if len(summary_df) > 0:
        print("Resumen por campo:")
        print(summary_df.groupby(["field_name", "status"]).size())
        print("=" * 80)

        print("Resumen por tE grid:")
        cols_te = ["grid_tE_days", "grid_piE", "grid_D_L_kpc", "grid_M_L_Msun"]
        existing = [c for c in cols_te if c in summary_df.columns]
        if len(existing) > 0:
            print(summary_df[existing].describe())

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
