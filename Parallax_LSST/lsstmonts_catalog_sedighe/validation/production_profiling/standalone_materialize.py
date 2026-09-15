#!/usr/bin/env python3
"""
Standalone H5+truth materialization, reusing the real production
driver's own setup/config machinery (paths, worker_config, runtime
patches) but calling ONLY the pure-simulation functions
(simulate_event_for_fit -> save_simulated_event_if_selected), never
sim_fit/sim_fit_multi_fits/run_single_event -- so the old FAST H1
fit is never invoked as a side effect.

Config: configs/validation/production_profiling/PROFILING_SIM_LOCAL.json
(a copy of FAST.json with paths corrected for this machine; only the
paths/observing/truth/simulation/rubin/selection sections are used --
the "fit" section is never read by anything this script calls).
"""
import sys
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER_DIR = os.path.dirname(os.path.dirname(HERE))  # lsstmonts_catalog_sedighe
sys.path.insert(0, DRIVER_DIR)

CONFIG_PATH = os.path.join(HERE, "..", "..", "configs", "validation",
                            "production_profiling", "PROFILING_SIM_LOCAL.json")
CONFIG_PATH = os.path.abspath(CONFIG_PATH)

sys.argv = ["run_lsstmonts_catalog_hidden_parallax.py", "--config", CONFIG_PATH]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run_lsstmonts_catalog_hidden_parallax as driver  # noqa: E402
import functions_roman_rubin as frr  # noqa: E402

_WORKER_INITIALIZED = False


def build_worker_config():
    return {
        "system_type": driver.SYSTEM_TYPE,
        "model": driver.MODEL,
        "fit_model": driver.FIT_MODEL,
        "fit_parallax": driver.FIT_PARALLAX,
        "truth_parallax": driver.TRUTH_PARALLAX,
        "run_multiple_fits": driver.RUN_MULTIPLE_FITS,
        "fit_specs": driver.FIT_SPECS,
        "primary_fit": driver.PRIMARY_FIT,
        "lrt_config": driver.LRT_CONFIG,
        "algo": driver.ALGO,
        "use_roman": driver.USE_ROMAN,
        "use_rubin": driver.USE_RUBIN,
        "rubin_sim_data_dir": str(driver.RUBIN_SIM_DATA_DIR),
        "rubin_throughputs_dir": str(driver.RUBIN_THROUGHPUTS_DIR),
        "rubin_opsim_db_path": str(driver.RUBIN_OPSIM_DB_PATH),
        "path_ephemerides": str(driver.PATH_EPHEMERIDES),
        "models_dir": str(driver.DIRS["models"]),
        "fits_dir": str(driver.DIRS["fits"]),
        "results_dir": str(driver.DIRS["results"]),
        "logs_dir": str(driver.DIRS["logs"]),
        "fit_bounds": driver.FIT_BOUNDS_NOPIE,
        "initial_guess": driver.FIT_INITIAL_GUESS,
        "optimizer_options": driver.FIT_OPTIMIZER_OPTIONS,
        "rubin_pointing_mode": driver.RUBIN_POINTING_MODE,
        "rubin_cache_cell_deg": driver.RUBIN_CACHE_CELL_DEG,
        "apply_detection_criteria": driver.APPLY_DETECTION_CRITERIA,
        "apply_photometric_filter": driver.APPLY_PHOTOMETRIC_FILTER,
        "band_availability_mode": driver.BAND_AVAILABILITY_MODE,
        "fit_window_enabled": driver.FIT_WINDOW_ENABLED,
        "fit_window_half_width_tE": driver.FIT_WINDOW_HALF_WIDTH_TE,
        "fit_window_minimum_total_points": driver.FIT_WINDOW_MINIMUM_TOTAL_POINTS,
        "catalog_row_start": None,
        "catalog_row_stop": None,
        "chunk_output_label": driver.CHUNK_OUTPUT_LABEL,
    }


def load_one_catalog_row(catalog_row):
    """Load exactly one raw catalog row and prepare it -- no fits."""
    raw = driver.load_raw_catalog(
        driver.COLUMNS_FILE, driver.DATA_FILE,
        nrows=1, catalog_row_start=catalog_row, catalog_row_stop=catalog_row + 1,
    )
    prepared, invalid = driver.prepare_catalog(
        raw, max_base_events=1, catalog_row_offset=catalog_row,
    )
    return prepared, invalid


def build_task_for_row(prepared_row_df):
    """prepared_row_df: a 1-row prepared_catalog DataFrame."""
    tasks = driver._build_tasks_single_realization_legacy(prepared_row_df)
    assert len(tasks) == 1
    return tasks[0]


def materialize_event(catalog_row, out_dir, timing=None, generating_model="H1"):
    """
    Full standalone materialization for one catalog_row:
    load -> prepare -> t0 resolve -> pair_catalog -> param_samplers ->
    simulate -> (if detectable) save H5. Returns a dict with outcome
    and per-stage timings (if `timing` dict is passed, fills it in).
    No fit of any kind is run.

    generating_model: "H1" (default, matches the real production
    catalog -- real annual-parallax physics, piEN/piEE from the
    catalog row) or "H0" (forces truth_parallax=False in
    simulate_event_for_fit, so the simulator builds the light curve
    with parallax=["None", 0.0] -- a genuine no-parallax simulation,
    not a truth override. pyLIMA_parameters then has no piEN/piEE at
    all, matching "no truth piE exists and none should be invented").
    """
    if generating_model not in ("H0", "H1"):
        raise ValueError(f"generating_model must be 'H0' or 'H1', got {generating_model!r}")
    global _WORKER_INITIALIZED
    t = timing if timing is not None else {}

    t0 = time.time()
    prepared, invalid = load_one_catalog_row(catalog_row)
    t["load_prepare_s"] = time.time() - t0

    if len(prepared) == 0:
        return {"catalog_row": catalog_row, "status": "invalid_catalog_row",
                "invalid_reason": invalid.iloc[0].to_dict() if len(invalid) else None}

    task = build_task_for_row(prepared)

    if not _WORKER_INITIALIZED:
        t0 = time.time()
        driver.init_worker(prepared, build_worker_config())
        t["init_worker_s"] = time.time() - t0
        _WORKER_INITIALIZED = True
    else:
        # keep GLOBAL_PREPARED_CATALOG in sync with THIS row's prepared table
        driver.GLOBAL_PREPARED_CATALOG = prepared

    config = driver.GLOBAL_WORKER_CONFIG

    base_row = prepared.iloc[int(task["prepared_index"])].copy()

    t0 = time.time()
    base_row = driver.apply_t0_from_first_maf_timestamp(base_row, config)
    t["t0_resolve_s"] = time.time() - t0

    driver.set_runtime_event_context(base_row)
    driver.set_runtime_noise_realization_context(task)

    t0 = time.time()
    pair_catalog = driver.build_single_row_pair_catalog(base_row, task)
    param_samplers = driver.fixed_param_samplers(base_row, task)
    t["setup_s"] = time.time() - t0

    seed = int(task["simulation_seed"])
    np.random.seed(seed)

    pair_row = frr.load_pair_catalog_row(pair_catalog=pair_catalog, row_index=0)
    TRILEGAL_row = pd.DataFrame([pair_row])
    GENULENS_row = pd.DataFrame([pair_row])
    ROW_T = ROW_G = 0

    event_params = frr.build_event_params_from_pair_row(
        task["global_i"], pair_row, config["system_type"],
        param_samplers=param_samplers,
        custom_system=None,
    )

    path_TRILEGAL_set = "astrodatalab_pairs"
    path_GENULENS_set = "astrodatalab_pairs"

    t0 = time.time()
    my_own_model, pyLIMA_parameters, decision = frr.simulate_event_for_fit(
        task["global_i"], event_params, config["path_ephemerides"], config["model"],
        time_window=None,
        use_roman=config["use_roman"], use_rubin=config["use_rubin"],
        truth_parallax=(generating_model == "H1"),
        rubin_pointing_mode=config["rubin_pointing_mode"],
        rubin_cache_cell_deg=config["rubin_cache_cell_deg"],
        apply_detection_criteria=config["apply_detection_criteria"],
        apply_photometric_filter=config["apply_photometric_filter"],
    )
    t["simulate_s"] = time.time() - t0

    if not decision:
        driver.clear_runtime_event_context()
        return {"catalog_row": catalog_row, "status": "not_detectable", **t}

    t0 = time.time()
    frr.replace_flux_by_noisy_magnitude_flux_ZP(my_own_model, verbose=False)
    t["noise_s"] = time.time() - t0

    t0 = time.time()
    frr.save_simulated_event_if_selected(
        task["global_i"], ROW_G, ROW_T, path_TRILEGAL_set, path_GENULENS_set,
        str(out_dir), my_own_model, pyLIMA_parameters, event_params,
        GENULENS_row, TRILEGAL_row,
    )
    t["save_h5_s"] = time.time() - t0

    driver.clear_runtime_event_context()

    h5_path = os.path.join(str(out_dir), f"Event_{task['global_i']}.h5")

    return {"catalog_row": catalog_row, "status": "materialized",
            "h5_path": h5_path, "global_i": task["global_i"], **t}
