#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Corrida Rubin-only en paralelo para eventos PSPL generados con paralaje
y ajustados con PSPL sin paralaje.

Versión con:
- AstroDataLab pair catalog.
- Rubin-only usando coordenadas fuente por fuente.
- Binning angular para MAF/Rubin, para no llamar a rubin_sim una vez por cada fuente.
- Cache de MAF por celda angular.
"""

import os
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

import ulens_params
importlib.reload(ulens_params)

from ulens_params import download_astrodatalab_pair_catalog
from functions_roman_rubin import sim_fit


# ============================================================
# Variables globales para workers
# ============================================================

GLOBAL_PAIR_CATALOG = None
GLOBAL_CFG = None


def init_worker(pair_catalog, cfg):
    """
    Inicializador de cada proceso.
    Guarda el catálogo y la configuración como variables globales
    para no tener que pasarlas en cada evento.
    """

    global GLOBAL_PAIR_CATALOG
    global GLOBAL_CFG

    GLOBAL_PAIR_CATALOG = pair_catalog
    GLOBAL_CFG = cfg


def quantize_maf_cells(df, cell_deg):
    """
    Calcula las celdas MAF que se usarían para un catálogo dado.

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


def selected_pair_rows_for_seeds(n_events, n_pair_rows):
    """
    Reproduce la selección de filas que hace sim_fit para catalog_mode='astrodatalab_pairs':

        np.random.seed(i)
        ROW_P = np.random.randint(0, n_pair_rows)

    """

    rows = []

    for i in range(n_events):
        rng = np.random.RandomState(i)
        rows.append(rng.randint(0, n_pair_rows))

    return np.asarray(rows, dtype=int)


def print_maf_binning_diagnostics(pair_catalog, n_events, rubin_cache_cell_deg):
    """
    Imprime cuántas celdas MAF se usarán aproximadamente.
    """

    print("=" * 80)
    print("Diagnóstico de binning MAF")
    print("=" * 80)

    print(f"rubin_cache_cell_deg = {rubin_cache_cell_deg}")
    print(f"N pair catalog        = {len(pair_catalog)}")
    print(f"N events              = {n_events}")

    # Catálogo completo
    cells_all = quantize_maf_cells(
        pair_catalog,
        rubin_cache_cell_deg,
    )

    n_cells_all = len(cells_all.drop_duplicates())

    print(f"Celdas MAF únicas en catálogo completo: {n_cells_all}")

    # Filas que efectivamente se van a sortear para i = 0...N_events-1
    rows = selected_pair_rows_for_seeds(
        n_events,
        len(pair_catalog),
    )

    pair_sub = pair_catalog.iloc[rows].copy()

    cells_sub = quantize_maf_cells(
        pair_sub,
        rubin_cache_cell_deg,
    )

    n_cells_sub = len(cells_sub.drop_duplicates())

    print(f"Filas únicas seleccionadas: {len(np.unique(rows))}")
    print(f"Celdas MAF únicas esperadas en la corrida: {n_cells_sub}")

    print("-" * 80)
    print("Primeras celdas MAF:")
    print(cells_sub.drop_duplicates().head(20))
    print("=" * 80)


def run_single_event(i):
    """
    Ejecuta un único evento dentro de un worker.

    Cada evento escribe en sus propias carpetas:
        models/event_XXXXX/
        fits/event_XXXXX/
        results/event_XXXXX/
        logs/event_XXXXX.log
    """

    cfg = GLOBAL_CFG
    pair_catalog = GLOBAL_PAIR_CATALOG

    event_tag = f"event_{i:05d}"

    event_model_dir = Path(cfg["models_dir"]) / event_tag
    event_fit_dir = Path(cfg["fits_dir"]) / event_tag
    event_results_dir = Path(cfg["results_dir"]) / event_tag
    event_log_file = Path(cfg["logs_dir"]) / f"{event_tag}.log"

    event_model_dir.mkdir(parents=True, exist_ok=True)
    event_fit_dir.mkdir(parents=True, exist_ok=True)
    event_results_dir.mkdir(parents=True, exist_ok=True)
    event_log_file.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "i": i,
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
                print(f"Evento {i}")
                print("=" * 80)
                print("Rubin pointing mode:", cfg["rubin_pointing_mode"])
                print("Rubin cache cell deg:", cfg["rubin_cache_cell_deg"])
                print("=" * 80)

                result = sim_fit(
                    i,
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

                    param_samplers=cfg["param_samplers_bh_lowmass"],

                    fit_model=cfg["fit_model"],
                    fit_parallax=cfg["fit_parallax"],
                    fit_bounds=cfg["fit_bounds_nopie"],

                    # ====================================================
                    # Clave para consistencia + binning MAF
                    # ====================================================
                    rubin_pointing_mode=cfg["rubin_pointing_mode"],
                    rubin_cache_cell_deg=cfg["rubin_cache_cell_deg"],
                )

                # Diagnóstico del dataSlice usado por este worker/evento
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


def save_summary(summary_rows, logs_dir):
    """
    Guarda el resumen incremental de la corrida.
    """

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values("i").reset_index(drop=True)

    summary_df.to_csv(
        logs_dir / "run_summary.csv",
        index=False,
    )

    summary_df.to_parquet(
        logs_dir / "run_summary.parquet",
        index=False,
    )

    return summary_df


def main():

    # ============================================================
    # Configuración general
    # ============================================================

    BASE_DIR = Path("/home/anibal/Parallax_LSST")

    RUN_NAME = "BH_lowmass_RubinOnly_PSPL_NoPiE_lens1_2kpc_parallel_MAFbin005"

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
    # Parámetros de corrida
    # ============================================================

    system_type = "BH"
    model = "PSPL"
    fit_model = "PSPL"
    fit_parallax = False
    algo = "TRF"

    use_roman = False
    use_rubin = True

    N_events = 10000
    N_pairs_catalog = 10000

    N_WORKERS = 16

    # ============================================================
    # Configuración MAF/Rubin
    # ============================================================
    # Rubin-only con AstroDataLab:
    # - source: usa ra/dec de la fuente para el evento.
    # - rubin_cache_cell_deg: agrupa llamadas a MAF en celdas angulares.
    #
    # 0.05 deg = 3 arcmin.
    # Más rápido: 0.1
    # Más fino:   0.02

    rubin_pointing_mode = "source"
    rubin_cache_cell_deg = 0.05

    # ============================================================
    # Samplers físicos
    # ============================================================

    param_samplers_bh_lowmass = {
        "star_mass": {
            "type": "loguniform",
            "low": 0.1,
            "high": 10.0,
        },
        "mass_planet": {
            "type": "fixed",
            "value": 0.0,
        },
    }

    fit_bounds_nopie = {
        "t0": {
            "type": "center_width",
            "half_width": 500.0,
        },
        "u0": [
            -5.0,
            5.0,
        ],
        "tE": [
            0.1,
            5000.0,
        ],
    }

    # ============================================================
    # Configuración del catálogo AstroDataLab
    # ============================================================

    catalog_config = {
        "ra_center": 267.925,
        "dec_center": -29.152,
        "radius": 0.05,

        "N": N_pairs_catalog,
        "Ds_max": 8000,

        # Distancias internas en pc
        "min_D": 1000.0,
        "offset": 100.0,

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
        "limit_per_chunk": 10,
        "max_retries": 0,
        "timeout": 300,

        "lens_D_range_kpc": (1.0, 2.0),
        "source_D_range_kpc": (2.0, 8.0),
    }

    path_pair_catalog = (
        DIRS["catalogs"]
        / "astrodatalab_pairs_lens1_2kpc_source2_8kpc.parquet"
    )

    # ============================================================
    # Guardar configuración de corrida
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
        "N_events": N_events,
        "N_pairs_catalog": N_pairs_catalog,
        "N_WORKERS": N_WORKERS,
        "path_ephemerides": path_ephemerides,
        "path_pair_catalog": str(path_pair_catalog),
        "param_samplers_bh_lowmass": param_samplers_bh_lowmass,
        "fit_bounds_nopie": fit_bounds_nopie,
        "catalog_config": catalog_config,
        "rubin_pointing_mode": rubin_pointing_mode,
        "rubin_cache_cell_deg": rubin_cache_cell_deg,
    }

    with open(DIRS["config"] / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)

    # ============================================================
    # Crear o cargar catálogo pareado
    # ============================================================

    if path_pair_catalog.exists():

        print("=" * 80)
        print("Usando catálogo existente:")
        print(path_pair_catalog)
        print("=" * 80)

        pair_catalog = pd.read_parquet(path_pair_catalog)

    else:

        print("=" * 80)
        print("No existe catálogo pareado. Generando desde AstroDataLab...")
        print("=" * 80)

        pair_catalog = download_astrodatalab_pair_catalog(**catalog_config)

        pair_catalog.to_parquet(
            path_pair_catalog,
            index=False,
        )

        print("=" * 80)
        print("Catálogo guardado en:")
        print(path_pair_catalog)
        print("=" * 80)

    # ============================================================
    # Chequeo del catálogo
    # ============================================================

    print("=" * 80)
    print("Resumen del catálogo pareado")
    print("=" * 80)

    print(
        pair_catalog[
            [
                "D_L_kpc",
                "D_S_kpc",
                "mu_rel",
            ]
        ].describe()
    )

    print("D_L min:", pair_catalog["D_L_kpc"].min())
    print("D_L max:", pair_catalog["D_L_kpc"].max())
    print("D_S min:", pair_catalog["D_S_kpc"].min())
    print("D_S max:", pair_catalog["D_S_kpc"].max())
    print("N pairs:", len(pair_catalog))

    # ============================================================
    # Diagnóstico del binning MAF antes de correr
    # ============================================================

    print_maf_binning_diagnostics(
        pair_catalog,
        N_events,
        rubin_cache_cell_deg,
    )

    # Guardar tabla auxiliar de celdas para auditoría
    pair_cells = quantize_maf_cells(
        pair_catalog,
        rubin_cache_cell_deg,
    )

    pair_cells_out = pair_catalog.copy()
    pair_cells_out["maf_ra_bin"] = pair_cells["maf_ra"].values
    pair_cells_out["maf_dec_bin"] = pair_cells["maf_dec"].values

    pair_cells_out.to_parquet(
        DIRS["catalogs"] / "pair_catalog_with_maf_bins.parquet",
        index=False,
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
        "param_samplers_bh_lowmass": param_samplers_bh_lowmass,
        "fit_bounds_nopie": fit_bounds_nopie,

        # ========================================================
        # MAF binning
        # ========================================================
        "rubin_pointing_mode": rubin_pointing_mode,
        "rubin_cache_cell_deg": rubin_cache_cell_deg,
    }

    # ============================================================
    # Corrida paralela de eventos
    # ============================================================

    print("=" * 80)
    print("Iniciando corrida paralela")
    print(f"N_events:              {N_events}")
    print(f"N_WORKERS:             {N_WORKERS}")
    print(f"rubin_pointing_mode:   {rubin_pointing_mode}")
    print(f"rubin_cache_cell_deg:  {rubin_cache_cell_deg}")
    print("=" * 80)

    summary_rows = []

    try:
        mp_context = mp.get_context("fork")
    except ValueError:
        mp_context = None

    if mp_context is not None:

        executor_kwargs = {
            "max_workers": N_WORKERS,
            "initializer": init_worker,
            "initargs": (
                pair_catalog,
                worker_cfg,
            ),
            "mp_context": mp_context,
        }

    else:

        executor_kwargs = {
            "max_workers": N_WORKERS,
            "initializer": init_worker,
            "initargs": (
                pair_catalog,
                worker_cfg,
            ),
        }

    with ProcessPoolExecutor(**executor_kwargs) as executor:

        future_to_i = {
            executor.submit(run_single_event, i): i
            for i in range(N_events)
        }

        for future in as_completed(future_to_i):

            i = future_to_i[future]

            try:
                row = future.result()

            except Exception as e:

                err = traceback.format_exc()

                log_file = DIRS["logs"] / f"event_{i:05d}_executor_error.log"

                with open(log_file, "w") as f:
                    f.write(err)

                row = {
                    "i": i,
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
                f"[{n_done}/{N_events}] "
                f"evento {i} -> {row['status']} | "
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
        print("Resumen MAF bins usados:")
        cols = [
            "maf_mode",
            "maf_cache_mode",
            "maf_ra",
            "maf_dec",
            "maf_cache_source",
        ]
        existing_cols = [c for c in cols if c in summary_df.columns]
        print(summary_df[existing_cols].drop_duplicates().head(30))


if __name__ == "__main__":
    main()
