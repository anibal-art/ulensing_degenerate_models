"""Materialize one Sedighe catalogue event with roman_rubin main.

catalog row -> prepared row (sedighe_catalog) -> t0 from the first MAF
timestamp -> materialized roman_rubin row (fsource/ftotal/blend_ratio for
the catalogue-visible bands, piEN/piEE from xi) -> pre-fit detectability on
a no-parallax reference simulation -> H0 or H1 simulation with
simulation.simulate_event.sim_event -> Event_<row>.h5 (read_save.save_sim).

Same API as the historical validation/production_profiling/
standalone_materialize.py: preload_catalog_window() and materialize_event().
The photometric-noise seed is the catalogue row, as before.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sedighe_catalog as sc
import detectability as det
from photometry.constants import SIMULATION_BAND_ZERO_POINTS as ZP
from read_save import save_sim
from simulation.simulate_event import sim_event

CONFIG = sc.CONFIG
SIMULATION = CONFIG["simulation"]


def _expand(value):
    return os.path.expandvars(os.path.expanduser(str(value)))


COLUMNS_FILE = Path(_expand(CONFIG["input"]["columns_file"]))
DATA_FILE = Path(_expand(CONFIG["input"]["data_file"]))

WORKER_CONFIG = {
    "path_ephemerides": None,
    "use_roman": False,
    "use_rubin": True,
    "rubin_pointing_mode": SIMULATION["rubin_pointing_mode"],
    "rubin_cache_cell_deg": SIMULATION["rubin_cache_cell_deg"],
    "rubin_opsim_db_path": _expand(CONFIG["rubin"]["opsim_db_path"]),
    "rubin_sim_data_dir": _expand(CONFIG["rubin"]["sim_data_dir"]),
    "rubin_throughputs_dir": _expand(CONFIG["rubin"]["throughputs_dir"]),
}

if bool(CONFIG["observing"].get("use_roman", False)):
    raise NotImplementedError("This study is Rubin-only (use_roman=false).")

_RUBIN_CONFIGURED = False
_RAW = None
_RAW_START = None
_RAW_STOP = None


def configure_rubin():
    """Point roman_rubin to the Rubin data of the production config."""
    global _RUBIN_CONFIGURED

    if not _RUBIN_CONFIGURED:
        sc.stp.configure_rubin_paths(
            rubin_sim_data_dir=WORKER_CONFIG["rubin_sim_data_dir"],
            rubin_throughputs_dir=WORKER_CONFIG["rubin_throughputs_dir"],
            rubin_opsim_db_path=WORKER_CONFIG["rubin_opsim_db_path"],
        )
        _RUBIN_CONFIGURED = True


def preload_catalog_window(row_start, row_stop):
    """Preload one contiguous raw-catalog window for repeated row access."""
    global _RAW, _RAW_START, _RAW_STOP

    row_start, row_stop = int(row_start), int(row_stop)

    if row_start < 0 or row_stop <= row_start:
        raise ValueError(f"invalid window [{row_start}, {row_stop})")

    raw = sc.load_raw_catalog(
        COLUMNS_FILE,
        DATA_FILE,
        nrows=row_stop - row_start,
        catalog_row_start=row_start,
        catalog_row_stop=row_stop,
    ).reset_index(drop=True)

    if len(raw) != row_stop - row_start:
        raise RuntimeError(
            f"preload returned {len(raw)} rows for [{row_start}, {row_stop})"
        )

    _RAW, _RAW_START, _RAW_STOP = raw, row_start, row_stop
    print(f"[catalog-cache] preloaded raw rows [{row_start}, {row_stop}) n={len(raw)}", flush=True)


def load_one_catalog_row(catalog_row):
    """Load and prepare exactly one raw catalogue row."""
    catalog_row = int(catalog_row)

    if _RAW is not None and _RAW_START <= catalog_row < _RAW_STOP:
        raw = _RAW.iloc[[catalog_row - _RAW_START]].copy().reset_index(drop=True)
    else:
        raw = sc.load_raw_catalog(
            COLUMNS_FILE,
            DATA_FILE,
            nrows=1,
            catalog_row_start=catalog_row,
            catalog_row_stop=catalog_row + 1,
        )

    return sc.prepare_catalog(raw, max_base_events=1, catalog_row_offset=catalog_row)


def event_row_from_catalog(base_row):
    """
    Materialized roman_rubin row of one prepared catalogue event.

    tE, u0, rho come from the catalogue; t0 = first visible MAF timestamp
    + t0_catalog_days; piEN/piEE from piE and xi. For every visible band,
    F_source = 10^(0.4 (ZP - m_source)) and F_total = F_source / f_s, with
    f_s = F_source / F_total the catalogue blending factor.
    """
    piEN, piEE, _ = sc.parallax_components_from_catalog(base_row)
    active = sc._visible_bands_from_row(base_row, mode=sc.BAND_AVAILABILITY_MODE)
    mags = {band: float(base_row[band]) for band in sc.SOURCE_MAG_COLUMNS}

    row = {
        "catalog_row": int(base_row["catalog_row"]),
        "t0": float(base_row["t0_jd"]),
        "u0": float(base_row["u0"]),
        "tE": float(base_row["tE_catalog_days"]),
        "rho": float(base_row["rho"]),
        "piEN": float(piEN),
        "piEE": float(piEE),
        "ra": float(base_row["ra"]),
        "dec": float(base_row["dec"]),
        "maf_ra": float(base_row["ra"]),
        "maf_dec": float(base_row["dec"]),
        **mags,
        "y": mags["Y"],
        "W149": mags["Y"],
        "active_bands": list(active),
    }

    for band in active:
        source_fraction = float(base_row[f"blend_{band}"])

        if not (np.isfinite(source_fraction) and 0.0 < source_fraction <= 1.0):
            raise ValueError(f"invalid source fraction for {band}: {source_fraction}")

        fsource = sc._mag_to_flux(mags["Y" if band == "y" else band], ZP[band])
        ftotal = fsource / source_fraction
        row[f"fsource_{band}"] = fsource
        row[f"ftotal_{band}"] = ftotal
        row[f"blend_ratio_{band}"] = (ftotal - fsource) / fsource

    return row


def _simulate(i, row, truth_parallax):
    return sim_event(
        i,
        row,
        None,
        "FSPL",
        time_window=SIMULATION.get("time_window", None),
        use_roman=False,
        use_rubin=True,
        truth_parallax=bool(truth_parallax),
        rubin_pointing_mode=SIMULATION["rubin_pointing_mode"],
        rubin_cache_cell_deg=SIMULATION["rubin_cache_cell_deg"],
        apply_detection_criteria=False,
        apply_photometric_filter=bool(SIMULATION["apply_photometric_filter"]),
    )


def _detectability(i, row, generating_model):
    """Pre-fit gate on a reference simulation; the RNG state is restored."""
    detect_cfg = det._prefit_detectability_config()

    if not detect_cfg["enabled"]:
        return True, {}, None

    current_truth = detect_cfg["reference_model"] == "current_truth"
    reference_parallax = generating_model == "H1" if current_truth else False

    state = np.random.get_state()

    try:
        reference = _simulate(i, row, reference_parallax)
    finally:
        np.random.set_state(state)

    model, _, valid = reference
    metrics = det._compute_prefit_detectability_metrics(model, row, detect_cfg)
    metrics["detectability_reference_has_required_data"] = bool(valid)

    if not valid:
        metrics = det._prefit_force_failure(metrics, "no_required_data")

    passed = bool(metrics.get("detectability_pass", False)) and not detect_cfg["audit_only"]

    return passed, metrics, reference if current_truth else None


def materialize_event(catalog_row, out_dir, timing=None, generating_model="H1"):
    """Materialize one event; same return contract as standalone_materialize."""
    if generating_model not in ("H0", "H1"):
        raise ValueError(f"generating_model must be 'H0' or 'H1', got {generating_model!r}")

    t = timing if timing is not None else {}
    configure_rubin()

    t0 = time.time()
    prepared, invalid = load_one_catalog_row(catalog_row)
    t["load_prepare_s"] = time.time() - t0

    if len(prepared) == 0:
        return {
            "catalog_row": catalog_row,
            "status": "invalid_catalog_row",
            "invalid_reason": invalid.iloc[0].to_dict() if len(invalid) else None,
        }

    t0 = time.time()
    base_row = sc.apply_t0_from_first_maf_timestamp(prepared.iloc[0].copy(), WORKER_CONFIG)
    t["t0_resolve_s"] = time.time() - t0

    t0 = time.time()
    row = event_row_from_catalog(base_row)
    i = int(catalog_row)
    t["setup_s"] = time.time() - t0

    t0 = time.time()
    passed, metrics, reused = _detectability(i, row, generating_model)

    if not passed:
        t["simulate_s"] = time.time() - t0
        return {"catalog_row": catalog_row, "status": "not_detectable", **t}

    if reused is not None:
        model, params, valid = reused
    else:
        model, params, valid = _simulate(i, row, generating_model == "H1")

    t["simulate_s"] = time.time() - t0

    if not valid:
        return {"catalog_row": catalog_row, "status": "not_detectable", **t}

    t0 = time.time()
    catalog_frame = pd.DataFrame([dict(base_row)])
    save_sim(
        i, 0, 0, "sedighe_lsstmonts", "sedighe_lsstmonts", str(out_dir),
        model, params, row, catalog_frame, catalog_frame,
    )
    t["save_h5_s"] = time.time() - t0

    return {
        "catalog_row": catalog_row,
        "status": "materialized",
        "h5_path": os.path.join(str(out_dir), f"Event_{i}.h5"),
        "global_i": i,
        **t,
    }
