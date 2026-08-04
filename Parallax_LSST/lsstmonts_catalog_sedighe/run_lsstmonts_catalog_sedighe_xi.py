#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the LSSTMONTS catalog through the current Rubin-only pipeline.

Experiment implemented here
---------------------------
- one simulation per catalog row;
- source apparent magnitudes are read from the catalog;
- catalog blending factors are interpreted as
      f_s = F_source / (F_source + F_blend);
- the catalog column xi is read explicitly and used as the
  trajectory angle from Equations 5-6 of Sajadian & Sahu (2023);
- xi is measured in the local Galactic tangent plane from +l toward +b;
- the same xi fixes the source-lens trajectory and piEN/piEE;
- the true event is FSPL with annual parallax;
- the fit is FSPL without parallax;
- the event is generated with the complete MAF cadence;
- only the fit is restricted to t0 +/- k*tE (k=3.5 by default);
- the catalog was already detection-selected, so the internal
  deviation-from-constant criterion is disabled.

Scientific convention used in this runner
-----------------------------------------
The catalog quantity xi is used directly as the source-lens trajectory
angle. The alpha column, if present, is kept only as metadata. Thus
    pi_E,n1 = pi_E cos(xi)
    pi_E,n2 = pi_E sin(xi)
with n1 along increasing Galactic longitude and n2 along increasing
Galactic latitude. The local vector is then rotated to ICRS North/East.

The runner installs runtime patches inside each worker:
1. catalog flux/blending replaces the pipeline's random blending;
2. the fit light curves are cropped after the complete event is simulated.

No source file in the Roman-Rubin pipeline is overwritten.

Usage
-----
    python run_lsstmonts_catalog_sedighe_xi.py \
        --config lsstmonts_catalog_sedighe_xi.yaml

Preparation/validation only
---------------------------
    python run_lsstmonts_catalog_sedighe_xi.py \
        --config lsstmonts_catalog_sedighe_xi.yaml \
        --prepare-only
"""

# ============================================================================
# Avoid thread oversubscription inside multiprocessing workers
# ============================================================================

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import json
import argparse
import traceback
import inspect
import shutil
import multiprocessing as mp
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.constants import L_sun, sigma_sb

try:
    import yaml
except ImportError:
    yaml = None


# ============================================================================
# Paths
# ============================================================================

HOME = Path.home()
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "config_sedighe_lsstmonts_xi.json"


def log_step(message):
    """Print progress immediately, useful for large catalog reads."""
    print(message, flush=True)


import numpy as np


def normalize_angle_to_rad(angle, unit="degrees"):
    """
    Convierte un ángulo a radianes y lo normaliza a [0, 2*pi).
    """

    unit = str(unit).lower()

    if unit in {"deg", "degree", "degrees"}:
        angle_rad = np.deg2rad(float(angle))

    elif unit in {"rad", "radian", "radians"}:
        angle_rad = float(angle)

    else:
        raise ValueError(
            "angle unit debe ser 'degrees' o 'radians'. "
            f"Recibido: {unit}"
        )

    return float(np.mod(angle_rad, 2.0 * np.pi))


def galactic_tangent_basis_to_icrs(l_deg, b_deg):
    """
    Construye la base tangente local galáctica y la base North/East
    ecuatorial en el punto (l, b).

    Convención:
        n1 = dirección de longitud galáctica creciente, +l
        n2 = dirección de latitud galáctica creciente, +b

    Devuelve vectores cartesianos unitarios en ICRS.
    """

    # Matriz ICRS -> Galactic, estándar J2000.
    # Su transpuesta hace Galactic -> ICRS.
    r_eq_to_gal = np.array(
        [
            [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
            [ 0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
            [-0.8676661490190047, -0.1980763734312015,  0.4559837761750669],
        ],
        dtype=float,
    )

    r_gal_to_eq = r_eq_to_gal.T

    l_rad = np.deg2rad(float(l_deg))
    b_rad = np.deg2rad(float(b_deg))

    cos_l = np.cos(l_rad)
    sin_l = np.sin(l_rad)
    cos_b = np.cos(b_rad)
    sin_b = np.sin(b_rad)

    # Vector radial en coordenadas galácticas cartesianas.
    radial_gal = np.array(
        [
            cos_b * cos_l,
            cos_b * sin_l,
            sin_b,
        ],
        dtype=float,
    )

    # Base tangente local galáctica.
    n1_gal = np.array(
        [
            -sin_l,
            cos_l,
            0.0,
        ],
        dtype=float,
    )

    n2_gal = np.array(
        [
            -sin_b * cos_l,
            -sin_b * sin_l,
            cos_b,
        ],
        dtype=float,
    )

    # Rotar a ICRS.
    radial_eq = r_gal_to_eq @ radial_gal
    n1_eq = r_gal_to_eq @ n1_gal
    n2_eq = r_gal_to_eq @ n2_gal

    radial_eq /= np.linalg.norm(radial_eq)
    n1_eq /= np.linalg.norm(n1_eq)
    n2_eq /= np.linalg.norm(n2_eq)

    # Coordenadas ecuatoriales del punto.
    ra_rad = np.mod(
        np.arctan2(radial_eq[1], radial_eq[0]),
        2.0 * np.pi,
    )

    dec_rad = np.arcsin(
        np.clip(radial_eq[2], -1.0, 1.0)
    )

    # Base local ICRS: East y North.
    east_eq = np.array(
        [
            -np.sin(ra_rad),
            np.cos(ra_rad),
            0.0,
        ],
        dtype=float,
    )

    north_eq = np.array(
        [
            -np.cos(ra_rad) * np.sin(dec_rad),
            -np.sin(ra_rad) * np.sin(dec_rad),
            np.cos(dec_rad),
        ],
        dtype=float,
    )

    east_eq /= np.linalg.norm(east_eq)
    north_eq /= np.linalg.norm(north_eq)

    return {
        "ra_deg": float(np.rad2deg(ra_rad)),
        "dec_deg": float(np.rad2deg(dec_rad)),
        "n1_eq": n1_eq,
        "n2_eq": n2_eq,
        "east_eq": east_eq,
        "north_eq": north_eq,
    }


def piE_xi_to_piEN_piEE(
    piE,
    xi,
    l_deg,
    b_deg,
    xi_unit="degrees",
):
    """
    Convierte amplitud piE y ángulo xi del catálogo a componentes pyLIMA.

    Entrada:
        piE  : amplitud de parallax, escalar positivo.
        xi   : ángulo de la dirección lens-source.
        l,b  : coordenadas galácticas del evento, en grados.

    Convención asumida para xi:
        xi = 0      apunta hacia +l
        xi = pi/2   apunta hacia +b

    pyLIMA espera:
        piEN = componente North
        piEE = componente East
    """

    piE = float(piE)

    if not np.isfinite(piE):
        raise ValueError(f"piE no finito: {piE}")

    if piE < 0:
        raise ValueError(f"piE debe ser >= 0. Recibido: {piE}")

    xi_rad = normalize_angle_to_rad(
        xi,
        unit=xi_unit,
    )

    basis = galactic_tangent_basis_to_icrs(
        l_deg=l_deg,
        b_deg=b_deg,
    )

    # Componentes en la base galáctica local.
    piE_n1 = piE * np.cos(xi_rad)
    piE_n2 = piE * np.sin(xi_rad)

    # Vector parallax en coordenadas cartesianas ICRS.
    vector_eq = (
        piE_n1 * basis["n1_eq"]
        + piE_n2 * basis["n2_eq"]
    )

    # Proyección a la base local ICRS.
    piEN = float(np.dot(vector_eq, basis["north_eq"]))
    piEE = float(np.dot(vector_eq, basis["east_eq"]))

    recovered_piE = float(np.hypot(piEN, piEE))

    if not np.isclose(
        recovered_piE,
        piE,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "La rotación xi/piE -> piEN/piEE no conservó la amplitud: "
            f"piE input={piE}, piE output={recovered_piE}"
        )

    return piEN, piEE
    
def _early_config_path():
    """
    Read --config before the remaining module-level configuration is built.
    parse_known_args() leaves the normal CLI arguments for main().
    """

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_FILE),
    )

    args, _ = parser.parse_known_args()

    return Path(args.config).expanduser().resolve()


def _load_config(path):
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de configuración: {path}\n"
            "Usá --config /ruta/al/archivo.json"
        )

    suffix = path.suffix.lower()

    with open(path, "r", encoding="utf-8") as file:
        if suffix == ".json":
            config = json.load(file)
        else:
            if yaml is None:
                raise ImportError(
                    "Para leer YAML instalá PyYAML o usá un config JSON."
                )
            config = yaml.safe_load(file)

    if config is None:
        config = {}

    if not isinstance(config, dict):
        raise TypeError(
            "La raíz del archivo de configuración debe ser un diccionario."
        )

    return config


CONFIG_PATH = _early_config_path()
CONFIG_DIR = CONFIG_PATH.parent
CONFIG = _load_config(CONFIG_PATH)


def cfg(section, key, default=None):
    value = CONFIG.get(section, {})

    if value is None:
        return default

    if not isinstance(value, dict):
        raise TypeError(
            f"La sección '{section}' del config debe ser un diccionario."
        )

    return value.get(key, default)


def topcfg(key, default=None):
    return CONFIG.get(key, default)


def first_config_value(*values, default=None):
    for value in values:
        if value not in (None, ""):
            return value
    return default


def resolve_config_path(value, default=None):
    """
    Expand ~ and environment variables. Relative paths are interpreted
    relative to the YAML file, not relative to the current shell directory.
    """

    if value is None:
        value = default

    if value is None:
        return None

    expanded = os.path.expandvars(
        os.path.expanduser(str(value))
    )

    path = Path(expanded)

    if not path.is_absolute():
        path = CONFIG_DIR / path

    return path.resolve()


def configure_environment_from_config():
    mapping = {
        "project_base": "PARALLAX_LSST_BASE",
        "roman_rubin_dir": "ROMAN_RUBIN_DIR",
        "rubin_sim_data_dir": "RUBIN_SIM_DATA_DIR",
        "roman_ephemerides": "ROMAN_EPHEMERIDES",
    }

    for yaml_key, env_key in mapping.items():
        value = cfg("paths", yaml_key)

        if value not in (None, ""):
            os.environ[env_key] = str(
                resolve_config_path(value)
            )


configure_environment_from_config()


def configure_opsim_from_config():
    """
    Configura la OpSim desde el config unificado.

    Prioridad:
      1. rubin.opsim_db_path
      2. simulation.opsim_db_path
      3. variables de entorno ya existentes
      4. get_baseline() dentro de set_telescopes_pyLIMA
    """

    value = first_config_value(
        cfg("rubin", "opsim_db_path", None),
        cfg("simulation", "opsim_db_path", None),
        default=None,
    )

    if value in (None, "", "default", "auto"):
        return None

    path = resolve_config_path(value)

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo OpSim especificado en el config:\n{path}"
        )

    os.environ["RUBIN_OPSIM_DB_PATH"] = str(path)

    print(f"[config] RUBIN_OPSIM_DB_PATH = {path}", flush=True)

    return path


configure_opsim_from_config()


def find_project_base():
    env = os.environ.get("PARALLAX_LSST_BASE", "").strip()

    candidates = [
        Path(env).expanduser() if env else None,
        SCRIPT_DIR,
        *SCRIPT_DIR.parents,
        HOME / "ulensing_degenerate_models" / "Parallax_LSST",
        HOME / "Parallax_LSST",
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        candidate = candidate.resolve()

        if candidate.name == "Parallax_LSST":
            return candidate

        if (candidate / "runs").exists() and (
            (candidate / "data_sedighe").exists()
            or (candidate / "data").exists()
        ):
            return candidate

    return SCRIPT_DIR


def find_roman_rubin_dir(base_dir):
    env = os.environ.get("ROMAN_RUBIN_DIR", "").strip()

    candidates = [
        Path(env).expanduser() if env else None,
        base_dir / "roman_rubin",
        base_dir / "simulation_Rubin" / "roman_rubin",
        base_dir.parent / "simulation_Rubin" / "roman_rubin",
        HOME / "microlensing" / "simulation_Rubin" / "roman_rubin",
        HOME / "simulation_Rubin" / "roman_rubin",
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        candidate = candidate.resolve()

        if (candidate / "functions_roman_rubin.py").exists():
            return candidate

    raise FileNotFoundError(
        "No pude encontrar functions_roman_rubin.py.\n"
        "Definí, por ejemplo:\n"
        "export ROMAN_RUBIN_DIR=/home/anibal/microlensing/"
        "simulation_Rubin/roman_rubin"
    )


def configure_rubin_sim_data(base_dir):
    """
    Find a Rubin data directory containing the throughput tables and configure
    both environment-variable names used by different rubin_sim versions.
    """

    env_candidates = [
        os.environ.get("RUBIN_SIM_DATA_DIR", "").strip(),
        os.environ.get("SIMS_DATA_DIR", "").strip(),
    ]

    candidates = [
        Path(value).expanduser() for value in env_candidates if value
    ]

    candidates.extend(
        [
            HOME / "rubin_sim_data",
            HOME / "rubin_sim" / "rubin_sim_data",
            base_dir / "rubin_sim_data",
            base_dir.parent / "rubin_sim_data",
            HOME / "ulensing_degenerate_models" / "rubin_sim_data",
            HOME / "microlensing" / "rubin_sim_data",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()

        required = candidate / "throughputs" / "baseline" / "total_u.dat"

        if required.exists():
            os.environ["RUBIN_SIM_DATA_DIR"] = str(candidate)
            os.environ["SIMS_DATA_DIR"] = str(candidate)
            return candidate

    raise FileNotFoundError(
        "No pude encontrar rubin_sim_data/throughputs/baseline/total_u.dat.\n"
        "Definí RUBIN_SIM_DATA_DIR o SIMS_DATA_DIR con la ruta correcta."
    )


def find_ephemerides(roman_rubin_dir):
    env = os.environ.get("ROMAN_EPHEMERIDES", "").strip()

    candidates = [
        Path(env).expanduser() if env else None,
        roman_rubin_dir / "ephemerides" / "Roman_positions.npy",
        roman_rubin_dir / "Roman_positions.npy",
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        candidate = candidate.resolve()

        if candidate.exists():
            return candidate

    # The interface currently requests this path even in Rubin-only mode.
    raise FileNotFoundError(
        "No pude encontrar Roman_positions.npy.\n"
        "Definí ROMAN_EPHEMERIDES con la ruta correcta."
    )


BASE_DIR = find_project_base()
RUBIN_SIM_DATA_DIR = configure_rubin_sim_data(BASE_DIR)
ROMAN_RUBIN_DIR = find_roman_rubin_dir(BASE_DIR)
PATH_EPHEMERIDES = find_ephemerides(ROMAN_RUBIN_DIR)

sys.path.insert(0, str(ROMAN_RUBIN_DIR))

import functions_roman_rubin as frr  # noqa: E402
sim_fit = frr.sim_fit


# ============================================================================
# General configuration
# ============================================================================

DEFAULT_COLUMNS_FILE = BASE_DIR / "data_sedighe" / "columns"
DEFAULT_DATA_FILE = BASE_DIR / "data_sedighe" / "LSSTMONTS.dat"

COLUMNS_FILE = resolve_config_path(
    first_config_value(
        cfg("input", "columns_file", None),
        cfg("paths", "columns_file", None),
        default=None,
    ),
    default=DEFAULT_COLUMNS_FILE,
)

DATA_FILE = resolve_config_path(
    first_config_value(
        cfg("input", "data_file", None),
        cfg("paths", "data_file", None),
        default=None,
    ),
    default=DEFAULT_DATA_FILE,
)

RUN_NAME = str(
    first_config_value(
        topcfg("run_name", None),
        cfg("output", "run_name", None),
        default="LSSTMONTS_catalogXi_catalogBlending_FSPLparallax_fitFSPLNoPiE",
    )
)

OUTPUT_ROOT = resolve_config_path(
    first_config_value(
        topcfg("path_storage", None),
        cfg("output", "root_dir", None),
        default=None,
    ),
    default=BASE_DIR / "runs",
)

RUN_DIR = OUTPUT_ROOT / RUN_NAME

DIRS = {
    "catalogs": RUN_DIR / "catalogs",
    "models": RUN_DIR / "models",
    "fits": RUN_DIR / "fits",
    "results": RUN_DIR / "results",
    "logs": RUN_DIR / "logs",
    "config": RUN_DIR / "config",
}

for directory in DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)


# One catalog event produces exactly one simulation.
# The new Sedighe catalog contains an explicit xi column.  The alpha column,
# if present, is stored only as metadata and is not used to define parallax.
PARALLAX_ANGLE_COLUMN = str(
    first_config_value(
        cfg("sedighe", "angle_column", None),
        cfg("parallax", "angle_column", None),
        default="xi",
    )
)

PARALLAX_ANGLE_SEMANTICS = str(
    first_config_value(
        cfg("sedighe", "angle_semantics", None),
        cfg("parallax", "angle_semantics", None),
        default="xi_catalog_column",
    )
).lower()

if PARALLAX_ANGLE_SEMANTICS not in {
    "xi",
    "xi_catalog",
    "xi_catalog_column",
}:
    raise ValueError(
        "Este runner está diseñado para usar la columna xi explícita. "
        "Configure parallax.angle_column: xi y, opcionalmente, "
        "parallax.angle_semantics: xi_catalog_column."
    )

PARALLAX_ANGLE_UNIT = str(
    first_config_value(
        cfg("sedighe", "angle_unit", None),
        cfg("parallax", "angle_unit", None),
        default="auto",
    )
).lower()
PARALLAX_ANGLE_BASIS = str(
    first_config_value(
        cfg("sedighe", "angle_basis", None),
        cfg("parallax", "angle_basis", None),
        default="galactic_n1n2",
    )
).lower()

PARALLAX_COMPONENT_CONVENTION = str(
    first_config_value(
        cfg("sedighe", "component_convention", None),
        cfg("parallax", "component_convention", None),
        default="east_cos_north_sin",
    )
).lower()

if PARALLAX_ANGLE_UNIT not in {"auto", "rad", "radian", "radians", "deg", "degree", "degrees"}:
    raise ValueError(
        "parallax.angle_unit debe ser auto, radians o degrees."
    )

# Backward-compatible aliases for the local basis used in the paper:
# n1 = increasing Galactic longitude, n2 = increasing Galactic latitude.
if PARALLAX_ANGLE_BASIS == "galactic_lb":
    PARALLAX_ANGLE_BASIS = "galactic_n1n2"

if PARALLAX_ANGLE_BASIS not in {
    "galactic_n1n2",
    "icrs_en",
}:
    raise ValueError(
        "parallax.angle_basis debe ser galactic_n1n2 o icrs_en."
    )

if PARALLAX_COMPONENT_CONVENTION not in {
    "east_cos_north_sin",
    "north_cos_east_sin",
}:
    raise ValueError(
        "parallax.component_convention debe ser "
        "east_cos_north_sin o north_cos_east_sin."
    )

T0_ZERO_JD = float(
    first_config_value(
        cfg("input", "t0_zero_jd", None),
        cfg("catalog", "t0_zero_jd", None),
        default=2460413.013828608,
    )
)

RANDOM_SEED = int(
    cfg("execution", "random_seed", 20260728)
)

N_WORKERS = int(
    cfg("execution", "workers", 8)
)

MAX_BASE_EVENTS_CONFIG = first_config_value(
    topcfg("Nevents", None),
    cfg("selection", "max_base_events", None),
    default=25,
)

READ_NROWS_CONFIG = first_config_value(
    cfg("input", "read_nrows", None),
    cfg("catalog", "read_nrows", None),
    default=None,
)

PREPARE_ONLY_CONFIG = bool(
    cfg("execution", "prepare_only", False)
)


def parse_max_events(value):
    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in {"all", "none", ""}:
        return None

    parsed = int(value)

    if parsed <= 0:
        raise ValueError("max-base-events debe ser positivo o 'all'.")

    return parsed


def parse_optional_positive_int(value, name):
    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in {"all", "none", ""}:
        return None

    parsed = int(value)

    if parsed <= 0:
        raise ValueError(f"{name} debe ser positivo, None o 'all'.")

    return parsed


MAX_BASE_EVENTS = parse_max_events(MAX_BASE_EVENTS_CONFIG)


# Current pipeline settings
# The FFP branch is only used as an internal FSPL container because the
# current event-parameter builder inserts rho in that branch. star_mass is
# fixed event by event to the catalog lens mass and mass_planet is fixed to 0.
SYSTEM_TYPE = str(
    first_config_value(
        cfg("simulation", "system_type_internal", None),
        topcfg("system_type", None),
        default="FFP",
    )
)

MODEL = str(
    first_config_value(
        cfg("simulation", "model", None),
        topcfg("model", None),
        default="FSPL",
    )
)

FIT_MODEL = str(
    cfg("fit", "model", "FSPL")
)

FIT_PARALLAX = bool(
    cfg("fit", "parallax", False)
)

ALGO = str(
    first_config_value(
        cfg("fit", "algorithm", None),
        topcfg("algo", None),
        default="TRF",
    )
)

APPLY_DETECTION_CRITERIA = bool(
    first_config_value(
        cfg("simulation", "apply_detection_criteria", None),
        cfg("selection", "apply_detection_criteria", None),
        default=False,
    )
)

USE_ROMAN = bool(
    first_config_value(
        cfg("observing", "use_roman", None),
        cfg("observatories", "use_roman", None),
        default=False,
    )
)

USE_RUBIN = bool(
    first_config_value(
        cfg("observing", "use_rubin", None),
        cfg("observatories", "use_rubin", None),
        default=True,
    )
)

RUBIN_POINTING_MODE = str(
    first_config_value(
        cfg("simulation", "rubin_pointing_mode", None),
        cfg("rubin", "pointing_mode", None),
        default="source",
    )
)

_cache_cell_value = first_config_value(
    cfg("simulation", "rubin_cache_cell_deg", None),
    cfg("rubin", "cache_cell_deg", None),
    default=0.20,
)

RUBIN_CACHE_CELL_DEG = (
    None
    if _cache_cell_value is None
    else float(_cache_cell_value)
)

FIT_BOUNDS_NOPIE = cfg(
    "fit",
    "bounds",
    {
        "t0": {
            "type": "center_width",
            "half_width": 1000.0,
        },
        "u0": [-5.0, 5.0],
        "tE": [0.1, 20000.0],
    },
)

# The complete light curve is generated. This window is applied only to the
# arrays passed to the fitter by a runtime patch of extract_lightcurves_for_fit.
_fit_time_window_cfg = cfg("fit", "time_window", None)
if _fit_time_window_cfg is None:
    _fit_time_window_cfg = {}

if _fit_time_window_cfg is not None and not isinstance(_fit_time_window_cfg, dict):
    raise TypeError("fit.time_window debe ser un diccionario o null.")

FIT_WINDOW_ENABLED = bool(
    first_config_value(
        _fit_time_window_cfg.get("enabled", None),
        cfg("fit_window", "enabled", None),
        default=True,
    )
)

FIT_WINDOW_HALF_WIDTH_TE = float(
    first_config_value(
        _fit_time_window_cfg.get("factor", None),
        _fit_time_window_cfg.get("half_width_tE", None),
        cfg("fit_window", "half_width_tE", None),
        default=3.5,
    )
)

FIT_WINDOW_MINIMUM_TOTAL_POINTS = int(
    first_config_value(
        _fit_time_window_cfg.get("minimum_total_points", None),
        cfg("fit_window", "minimum_total_points", None),
        default=4,
    )
)

if FIT_WINDOW_ENABLED and FIT_WINDOW_HALF_WIDTH_TE <= 0.0:
    raise ValueError(
        "fit_window.half_width_tE debe ser mayor que cero."
    )

if FIT_WINDOW_MINIMUM_TOTAL_POINTS < 1:
    raise ValueError(
        "fit_window.minimum_total_points debe ser al menos 1."
    )

BLENDING_MODE = str(
    first_config_value(
        cfg("sedighe", "blending_mode", None),
        cfg("blending", "mode", None),
        default="catalog_source_fraction",
    )
).lower()

BLENDING_STRICT = bool(
    cfg("blending", "strict", True)
)

BLENDING_ZERO_MEANS_UNAVAILABLE_FILTER = bool(
    first_config_value(
        cfg("sedighe", "zero_means_unavailable_filter", None),
        cfg("blending", "zero_means_unavailable_filter", None),
        default=True,
    )
)

BLENDING_MINIMUM_VISIBLE_FILTERS = int(
    first_config_value(
        cfg("sedighe", "minimum_visible_filters", None),
        cfg("blending", "minimum_visible_filters", None),
        default=3,
    )
)

if BLENDING_MINIMUM_VISIBLE_FILTERS < 1:
    raise ValueError(
        "blending.minimum_visible_filters debe ser al menos 1."
    )

if BLENDING_MODE != "catalog_source_fraction":
    raise NotImplementedError(
        "Esta versión requiere blending.mode: catalog_source_fraction."
    )

BLENDING_ASSUMPTION = {
    "mode": BLENDING_MODE,
    "definition": "f_s = F_source / (F_source + F_blend)",
    "source": "catalog columns",
    "zero_means_unavailable_filter":
        BLENDING_ZERO_MEANS_UNAVAILABLE_FILTER,
    "minimum_visible_filters":
        BLENDING_MINIMUM_VISIBLE_FILTERS,
}


# ============================================================================
# Input-column mapping
# ============================================================================

COLUMN_ALIASES = {
    "Number_lensing": "catalog_event_id",
    "Galactic latitude (degree)": "b_deg",
    "Galactic Longitude (degree)": "l_deg",
    "Right ascention (degree)": "ra_catalog_deg",
    "Right ascention  (degree)": "ra_catalog_deg",
    "Declination (degree)": "dec_catalog_deg",
    "Lens_mass (solar mass)": "lens_mass_msun",
    "Lens_Distance (kpc)": "lens_distance_kpc",
    "Source_distance (kpc)": "source_distance_kpc",
    "Log10[Teff_source]": "logTe",
    "Einstein crossing time (days)": "tE_catalog_days",
    "time of closest approach (days)": "t0_catalog_days",
    "lens-source relative angular velocity (mas/years)":
        "mu_rel_catalog_masyr",
    "lens-source relative angular velocity (mas/days)":
        "mu_rel_catalog_masday",
    "Lens impact parameter": "u0",
    "Angular Einstein radius (mas)": "thetaE_mas",
    "Normalized Parallax amplitude": "piE",
    "Normalized Parallax amplitude (piE)": "piE",
    "Normalized source radius": "rho",
    r"Normalized source radius (i.e., \rho*)": "rho",
    "Apparent magnitude of source star in u-band": "u",
    "Apparent magnitude of source star in g-band": "g",
    "Apparent magnitude of source star in r-band": "r",
    "Apparent magnitude of source star in i-band": "i",
    "Apparent magnitude of source star in z-band": "z",
    "Apparent magnitude of source star in y-band": "Y",
    "Blending factor in u-band": "blend_u",
    "Blending factor in g-band": "blend_g",
    "Blending factor in r-band": "blend_r",
    "Blending factor in i-band": "blend_i",
    "Blending factor in z-band": "blend_z",
    "Blending factor in y-band": "blend_y",
    "Number of data points": "n_data_catalog",
    "Delta chi2": "delta_chi2_catalog",
    "FWHM (days)": "fwhm_catalog_days",
    "alpha": "alpha_catalog",
    "Alpha": "alpha_catalog",
    "ALPHA": "alpha_catalog",
    r"\alpha(degree)": "alpha_catalog",
    r"\alpha (degree)": "alpha_catalog",
    "alpha (rad)": "alpha_catalog",
    "alpha(rad)": "alpha_catalog",
    "alpha (degree)": "alpha_catalog",
    "alpha (degrees)": "alpha_catalog",
    "trajectory angle": "xi_catalog",
    "parallax angle": "xi_catalog",
    "xi": "xi_catalog",
    "Xi": "xi_catalog",
    "XI": "xi_catalog",
    r"\xi": "xi_catalog",
    r"\xi (degree)": "xi_catalog",
    r"\xi(degree)": "xi_catalog",
    r"\xi (rad)": "xi_catalog",
    r"\xi(rad)": "xi_catalog",
    "murel_1 (mas/days): x-component of lens-source relative angular velocity":
        "murel_1_mas_per_day",
    "murel_2 (mas/days): x-component of lens-source relative angular velocity":
        "murel_2_mas_per_day",
    "DetectionFlag_u: if source is visible in u-band =1 and zero otherwise":
        "DetectionFlag_u",
    "DetectionFlag_g: if source is visible in g-band =1 and zero otherwise":
        "DetectionFlag_g",
    "DetectionFlag_r: if source is visible in r-band =1 and zero otherwise":
        "DetectionFlag_r",
    "DetectionFlag_i: if source is visible in i-band =1 and zero otherwise":
        "DetectionFlag_i",
    "DetectionFlag_z: if source is visible in z-band =1 and zero otherwise":
        "DetectionFlag_z",
    "DetectionFlag_y: if source is visible in y-band =1 and zero otherwise":
        "DetectionFlag_y",
}

REQUIRED_COLUMNS = [
    "catalog_event_id",
    "l_deg",
    "b_deg",
    "lens_mass_msun",
    "lens_distance_kpc",
    "source_distance_kpc",
    "logTe",
    "tE_catalog_days",
    "t0_catalog_days",
    "u0",
    "thetaE_mas",
    "piE",
    "rho",
    "u",
    "g",
    "r",
    "i",
    "z",
    "Y",
    "blend_u",
    "blend_g",
    "blend_r",
    "blend_i",
    "blend_z",
    "blend_y",
    "xi_catalog",
]

BLEND_COLUMNS = [
    "blend_u",
    "blend_g",
    "blend_r",
    "blend_i",
    "blend_z",
    "blend_y",
]

SOURCE_MAG_COLUMNS = ["u", "g", "r", "i", "z", "Y"]


# ============================================================================
# Worker globals
# ============================================================================

GLOBAL_PREPARED_CATALOG = None
GLOBAL_WORKER_CONFIG = None

# Per-process runtime context. Each ProcessPool worker executes one event at a
# time, so these values are not shared between simultaneous events.
_RUNTIME_BLEND_SOURCE_FRACTION = None
_RUNTIME_FIT_WINDOW = None
_RUNTIME_FIT_MIN_POINTS = 1
_RUNTIME_LAST_FIT_COUNTS = {}
_ORIGINAL_EXTRACT_LIGHTCURVES = None
_ORIGINAL_MODEL_CHOICE = None
_RUNTIME_PATCHES_INSTALLED = False


class FitWindowRejected(RuntimeError):
    """The simulated event exists, but the requested fit window has too few points."""


def _as_float_array(values):
    return np.asarray(getattr(values, "value", values), dtype=float)


def _mag_to_flux(magnitude, zeropoint):
    return 10.0 ** (0.4 * (float(zeropoint) - float(magnitude)))


def _canonical_catalog_band(telescope_name):
    """
    Map pyLIMA/Rubin telescope names to the six catalog filter names.
    """

    name = str(telescope_name).strip()
    lower = name.lower()

    if lower in {"w149", "roman", "roman_w149", "y"}:
        return "y"

    if lower in {"u", "g", "r", "i", "z"}:
        return lower

    for band in ("u", "g", "r", "i", "z", "y"):
        if lower.endswith("_" + band):
            return band

    return None


def model_choice_catalog_visible_filters(
    event,
    *args,
    **kwargs,
):
    """
    Remove catalog filters with f_s == 0 before pyLIMA constructs flux bounds.

    The catalog was selected to be visible in at least three filters. A zero
    source fraction in an individual band is therefore treated as an
    unavailable band, not as an invalid physical event.
    """

    if _ORIGINAL_MODEL_CHOICE is None:
        raise RuntimeError(
            "No se instaló el model_choice original."
        )

    if _RUNTIME_BLEND_SOURCE_FRACTION is None:
        return _ORIGINAL_MODEL_CHOICE(
            event,
            *args,
            **kwargs,
        )

    kept_telescopes = []
    removed_telescopes = []

    for telescope in event.telescopes:
        catalog_band = _canonical_catalog_band(
            telescope.name
        )

        if catalog_band is None:
            kept_telescopes.append(telescope)
            continue

        source_fraction = float(
            _RUNTIME_BLEND_SOURCE_FRACTION[
                catalog_band
            ]
        )

        if (
            BLENDING_ZERO_MEANS_UNAVAILABLE_FILTER
            and source_fraction == 0.0
        ):
            removed_telescopes.append(
                str(telescope.name)
            )
            continue

        kept_telescopes.append(telescope)

    event.telescopes = kept_telescopes

    if removed_telescopes:
        print(
            "Catalog filters removed because f_s == 0:",
            removed_telescopes,
        )

    if len(event.telescopes) == 0:
        raise FitWindowRejected(
            "No catalog-visible filters remain after removing "
            "bands with f_s == 0."
        )

    print(
        "Catalog-visible active telescopes:",
        [
            telescope.name
            for telescope in event.telescopes
        ],
    )

    return _ORIGINAL_MODEL_CHOICE(
        event,
        *args,
        **kwargs,
    )


def catalog_flux_parameters_model(
    magstar,
    ZP,
    my_own_model,
    band_order=None,
):
    """
    Replacement for the pipeline random-blending function.

    The catalog quantity is
        source_fraction = F_source / (F_source + F_blend).

    For blend_flux_parameter='ftotal', pyLIMA receives for every active band
        [F_source, F_total].
    """

    if _RUNTIME_BLEND_SOURCE_FRACTION is None:
        raise RuntimeError(
            "No se configuró el blending del evento antes de sim_fit."
        )

    active_bands = [
        telescope.name
        for telescope in my_own_model.event.telescopes
    ]

    blend_parameter = str(
        getattr(
            my_own_model,
            "blend_flux_parameter",
            "ftotal",
        )
    ).lower()

    flux_parameters = []
    source_fluxes = []
    blend_ratios = []
    total_fluxes = []

    for band in active_bands:
        catalog_band = _canonical_catalog_band(
            band
        )

        if catalog_band is None:
            raise KeyError(
                f"No pude asociar el telescopio {band!r} "
                "con una banda del catálogo."
            )

        magnitude_band = (
            "Y" if catalog_band == "y" and "Y" in magstar
            else catalog_band
        )

        if magnitude_band not in magstar:
            if band == "W149" and "y" in magstar:
                magnitude_band = "y"
            else:
                raise KeyError(
                    f"No encontré magnitud de fuente para banda {band}."
                )

        fraction_band = catalog_band

        if (
            fraction_band
            not in _RUNTIME_BLEND_SOURCE_FRACTION
        ):
            raise KeyError(
                f"No encontré source fraction para banda {band}."
            )

        source_fraction = float(
            _RUNTIME_BLEND_SOURCE_FRACTION[
                fraction_band
            ]
        )

        if not np.isfinite(source_fraction) or not (0.0 < source_fraction <= 1.0):
            raise ValueError(
                f"Blending inválido para {band}: {source_fraction}. "
                "Se requiere 0 < F_source/F_total <= 1."
            )

        source_flux = _mag_to_flux(
            magstar[magnitude_band],
            ZP[band],
        )
        total_flux = source_flux / source_fraction
        blend_flux = total_flux - source_flux
        blend_ratio = blend_flux / source_flux

        if blend_parameter in {"ftotal", "total", "total_flux"}:
            second_flux_parameter = total_flux
        elif blend_parameter in {"fblend", "blend", "blend_flux"}:
            second_flux_parameter = blend_flux
        elif blend_parameter in {"gblend", "g", "blend_ratio"}:
            second_flux_parameter = blend_ratio
        else:
            raise ValueError(
                "blend_flux_parameter no reconocido en el modelo: "
                f"{blend_parameter!r}"
            )

        flux_parameters.extend(
            [float(source_flux), float(second_flux_parameter)]
        )
        source_fluxes.append(float(source_flux))
        blend_ratios.append(float(blend_ratio))
        total_fluxes.append(float(total_flux))

    print("Catalog blending used:")
    print(
        {
            band: {
                "source_fraction": float(
                    _RUNTIME_BLEND_SOURCE_FRACTION[
                        _canonical_catalog_band(band)
                    ]
                ),
                "F_source": source_fluxes[k],
                "F_total": total_fluxes[k],
                "F_blend_over_F_source": blend_ratios[k],
            }
            for k, band in enumerate(active_bands)
        }
    )

    return (
        flux_parameters,
        source_fluxes,
        blend_ratios,
        total_fluxes,
    )


def _crop_array_to_window(array, t_min, t_max):
    array = np.asarray(array)
    if array.size == 0:
        return array
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(
            f"Curva para fit con forma inesperada: {array.shape}"
        )
    mask = (
        np.isfinite(array[:, 0])
        & (array[:, 0] >= t_min)
        & (array[:, 0] <= t_max)
    )
    return array[mask]


def extract_lightcurves_for_fit_catalog_window(pyLIMA_model):
    """
    Wrapper around the current pipeline extractor.

    The complete event has already been simulated and saved. Only the arrays
    sent to the fitter, and the corresponding analysis tables, are cropped.
    """

    global _RUNTIME_LAST_FIT_COUNTS

    lc_to_fit, lc_to_save = _ORIGINAL_EXTRACT_LIGHTCURVES(
        pyLIMA_model
    )

    if _RUNTIME_FIT_WINDOW is None:
        _RUNTIME_LAST_FIT_COUNTS = {
            band: int(len(values))
            for band, values in lc_to_fit.items()
        }
        return lc_to_fit, lc_to_save

    t_min, t_max = map(float, _RUNTIME_FIT_WINDOW)
    counts = {}

    for band in list(lc_to_fit):
        cropped = _crop_array_to_window(
            lc_to_fit[band],
            t_min,
            t_max,
        )
        lc_to_fit[band] = cropped
        counts[band] = int(len(cropped))

    for band in list(lc_to_save):
        table = lc_to_save[band]
        times = _as_float_array(table["time"])
        mask = (
            np.isfinite(times)
            & (times >= t_min)
            & (times <= t_max)
        )
        lc_to_save[band] = table[mask]

    _RUNTIME_LAST_FIT_COUNTS = counts
    total_points = int(sum(counts.values()))

    print("Fit-only time window [JD]:", (t_min, t_max))
    print("Fit points by band:", counts)
    print("Fit points total:", total_points)

    if total_points < int(_RUNTIME_FIT_MIN_POINTS):
        raise FitWindowRejected(
            "Insufficient observations inside fit-only window: "
            f"N={total_points}, required={_RUNTIME_FIT_MIN_POINTS}."
        )

    return lc_to_fit, lc_to_save


def install_runtime_patches():
    global _ORIGINAL_EXTRACT_LIGHTCURVES
    global _ORIGINAL_MODEL_CHOICE
    global _RUNTIME_PATCHES_INSTALLED

    if _RUNTIME_PATCHES_INSTALLED:
        return

    if not hasattr(frr, "flux_parameters_model"):
        raise RuntimeError(
            "functions_roman_rubin no expone flux_parameters_model."
        )

    if not hasattr(frr, "extract_lightcurves_for_fit"):
        raise RuntimeError(
            "functions_roman_rubin no expone extract_lightcurves_for_fit; "
            "no puedo aplicar la ventana únicamente al fit."
        )

    if not hasattr(frr, "model_choice"):
        raise RuntimeError(
            "functions_roman_rubin no expone model_choice; "
            "no puedo excluir filtros con f_s == 0 antes de crear "
            "el modelo pyLIMA."
        )

    _ORIGINAL_EXTRACT_LIGHTCURVES = (
        frr.extract_lightcurves_for_fit
    )

    _ORIGINAL_MODEL_CHOICE = (
        frr.model_choice
    )

    frr.flux_parameters_model = catalog_flux_parameters_model
    frr.model_choice = model_choice_catalog_visible_filters
    frr.extract_lightcurves_for_fit = (
        extract_lightcurves_for_fit_catalog_window
    )

    _RUNTIME_PATCHES_INSTALLED = True


def set_runtime_event_context(base_row):
    global _RUNTIME_BLEND_SOURCE_FRACTION
    global _RUNTIME_FIT_WINDOW
    global _RUNTIME_FIT_MIN_POINTS
    global _RUNTIME_LAST_FIT_COUNTS

    _RUNTIME_BLEND_SOURCE_FRACTION = {
        "u": float(base_row["blend_u"]),
        "g": float(base_row["blend_g"]),
        "r": float(base_row["blend_r"]),
        "i": float(base_row["blend_i"]),
        "z": float(base_row["blend_z"]),
        "y": float(base_row["blend_y"]),
    }

    _RUNTIME_FIT_WINDOW = build_event_fit_window(base_row)
    _RUNTIME_FIT_MIN_POINTS = FIT_WINDOW_MINIMUM_TOTAL_POINTS
    _RUNTIME_LAST_FIT_COUNTS = {}


def clear_runtime_event_context():
    global _RUNTIME_BLEND_SOURCE_FRACTION
    global _RUNTIME_FIT_WINDOW
    global _RUNTIME_LAST_FIT_COUNTS

    _RUNTIME_BLEND_SOURCE_FRACTION = None
    _RUNTIME_FIT_WINDOW = None
    _RUNTIME_LAST_FIT_COUNTS = {}


def init_worker(prepared_catalog, worker_config):
    global GLOBAL_PREPARED_CATALOG
    global GLOBAL_WORKER_CONFIG

    GLOBAL_PREPARED_CATALOG = prepared_catalog
    GLOBAL_WORKER_CONFIG = worker_config

    install_runtime_patches()


# ============================================================================
# Catalog preparation
# ============================================================================

def _count_data_columns(data_file):
    with open(data_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                return len(line.split())
    raise ValueError(f"El catálogo está vacío: {data_file}")


def _normalized_column_name(name):
    return " ".join(str(name).strip().split())


def _compact_column_name(name):
    return (
        _normalized_column_name(name)
        .lower()
        .replace("\\", "")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def rename_sedighe_xi_columns(data):
    """
    Detecta las columnas nuevas de Sedighe aunque el archivo columns
    contenga barras, espacios dobles o descripciones largas.

    En particular, mapea la columna
        \\xi (degree): The angle of the lens-source trajectory ...
    a xi_catalog.
    """

    rename = {}

    for col in data.columns:
        clean = _normalized_column_name(col)
        compact = _compact_column_name(col)

        # La columna xi explícita del nuevo catálogo.
        if (
            compact.startswith("xi(degree):")
            or compact.startswith("xi(degree)")
            or compact.startswith("xidegree")
            or compact.startswith("xi(rad):")
            or compact.startswith("xirad")
            or (
                "angleofthelenssourcetrajectory" in compact
                and compact.startswith("xi")
            )
        ):
            rename[col] = "xi_catalog"

        # Alpha queda como metadato, no como fuente de xi.
        elif compact.startswith("alpha(degree)") or compact.startswith("alphadegree"):
            rename[col] = "alpha_catalog"

        elif compact.startswith("murel1(mas/days)"):
            rename[col] = "murel_1_mas_per_day"

        elif compact.startswith("murel2(mas/days)"):
            rename[col] = "murel_2_mas_per_day"

        elif compact.startswith("detectionflagu"):
            rename[col] = "DetectionFlag_u"
        elif compact.startswith("detectionflagg"):
            rename[col] = "DetectionFlag_g"
        elif compact.startswith("detectionflagr"):
            rename[col] = "DetectionFlag_r"
        elif compact.startswith("detectionflagi"):
            rename[col] = "DetectionFlag_i"
        elif compact.startswith("detectionflagz"):
            rename[col] = "DetectionFlag_z"
        elif compact.startswith("detectionflagy"):
            rename[col] = "DetectionFlag_y"

    return data.rename(columns=rename)


def load_raw_catalog(columns_file, data_file, nrows=None):
    log_step(f"[catalog] columns_file = {columns_file}")
    log_step(f"[catalog] data_file    = {data_file}")
    if nrows is not None:
        log_step(f"[catalog] read_nrows   = {nrows}")
    else:
        log_step("[catalog] read_nrows   = all")

    if not columns_file.exists():
        raise FileNotFoundError(
            f"No existe el archivo de columnas: {columns_file}"
        )

    if not data_file.exists():
        raise FileNotFoundError(
            f"No existe el catálogo: {data_file}"
        )

    # Leer el archivo de columnas línea por línea. No usamos pd.read_csv
    # porque algunas descripciones contienen comas, por ejemplo "i.e.,".
    with open(columns_file, "r", encoding="utf-8") as file:
        column_names = [
            _normalized_column_name(line)
            for line in file
            if line.strip()
        ]

    log_step(f"[catalog] columns read = {len(column_names)}")

    n_data_columns = _count_data_columns(data_file)
    log_step(f"[catalog] data columns in first non-empty row = {n_data_columns}")

    # If the data file has one extra value with respect to the columns file,
    # assume the missing last column is the configured parallax angle column.
    # For the new Sedighe catalog this should normally be xi.
    if n_data_columns == len(column_names) + 1:
        column_names.append(PARALLAX_ANGLE_COLUMN)
        print(
            "[catalog] Added missing last-column name from YAML:",
            PARALLAX_ANGLE_COLUMN,
        )
    elif n_data_columns != len(column_names):
        raise ValueError(
            f"El catálogo tiene {n_data_columns} valores por fila, pero "
            f"columns contiene {len(column_names)} nombres."
        )

    log_step("[catalog] starting pd.read_csv ...")

    data = pd.read_csv(
        data_file,
        sep=r"\s+",
        header=None,
        names=column_names,
        engine="c",
        nrows=nrows,
    )

    log_step(f"[catalog] finished pd.read_csv. shape = {data.shape}")

    # Exact configured angle-column name takes precedence.
    if PARALLAX_ANGLE_COLUMN in data.columns:
        data = data.rename(
            columns={PARALLAX_ANGLE_COLUMN: "xi_catalog"}
        )

    data = data.rename(columns=COLUMN_ALIASES)
    data = rename_sedighe_xi_columns(data)

    if "alpha_catalog" not in data.columns:
        data["alpha_catalog"] = np.nan

    log_step("[catalog] column renaming finished.")

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            "No pude identificar estas columnas:\n"
            + "\n".join(f"- {column}" for column in missing)
            + "\n\nColumnas disponibles:\n"
            + "\n".join(map(str, data.columns))
        )

    log_step("[catalog] required columns found.")
    return data


def _resolve_angle_unit(values):
    requested = PARALLAX_ANGLE_UNIT

    if requested in {"rad", "radian", "radians"}:
        return "radians"
    if requested in {"deg", "degree", "degrees"}:
        return "degrees"

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]

    if len(finite) == 0:
        raise ValueError("No hay valores finitos en la columna alpha/xi.")

    if np.nanmax(np.abs(finite)) <= 2.0 * np.pi + 0.1:
        return "radians"

    return "degrees"


def prepare_catalog(raw_catalog, max_base_events):
    data = raw_catalog.copy().reset_index(drop=True)

    data["catalog_row"] = np.arange(
        len(data),
        dtype=int,
    )

    numeric_columns = list(
        dict.fromkeys(
            REQUIRED_COLUMNS
            + [
                "mu_rel_catalog_masyr",
                "mu_rel_catalog_masday",
                "murel_1_mas_per_day",
                "murel_2_mas_per_day",
                "alpha_catalog",
                "xi_catalog",
                "n_data_catalog",
                "delta_chi2_catalog",
                "fwhm_catalog_days",
                "DetectionFlag_u",
                "DetectionFlag_g",
                "DetectionFlag_r",
                "DetectionFlag_i",
                "DetectionFlag_z",
                "DetectionFlag_y",
            ]
        )
    )

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    # Explicit scientific assumption for this run:
    # use the catalog xi column as the relative trajectory angle.
    data["xi_catalog"] = data[
        "xi_catalog"
    ].to_numpy(dtype=float)

    resolved_angle_unit = _resolve_angle_unit(
        data["xi_catalog"].to_numpy(dtype=float)
    )

    if resolved_angle_unit == "degrees":
        data["xi_rad"] = np.deg2rad(
            data["xi_catalog"].to_numpy(dtype=float)
        )
    else:
        data["xi_rad"] = data[
            "xi_catalog"
        ].to_numpy(dtype=float)

    # Normalize only the angle used in trigonometric calculations.
    data["xi_rad"] = np.mod(
        data["xi_rad"].to_numpy(dtype=float),
        2.0 * np.pi,
    )

    data["xi_deg"] = np.rad2deg(
        data["xi_rad"].to_numpy(dtype=float)
    )

    # Backward-compatible aliases used by a few generic output routines.
    data["alpha_rad"] = data["xi_rad"]
    data["alpha_deg"] = data["xi_deg"]
    data["alpha_unit_resolved"] = resolved_angle_unit
    data["angle_semantics"] = "xi_catalog_column"

    # The test catalog had RA=Dec=0. Derive ICRS coordinates from l,b.
    coordinates = SkyCoord(
        l=data["l_deg"].to_numpy(dtype=float) * u.deg,
        b=data["b_deg"].to_numpy(dtype=float) * u.deg,
        frame="galactic",
    )

    data["ra"] = coordinates.icrs.ra.deg
    data["dec"] = coordinates.icrs.dec.deg

    # Preserve the catalog tE even if the column labeled mu_rel uses a
    # different unit/convention.
    data["mu_rel_for_pipeline_masyr"] = (
        data["thetaE_mas"]
        * 365.25
        / data["tE_catalog_days"]
    )

    data["t0_jd"] = (
        T0_ZERO_JD
        + data["t0_catalog_days"]
    )

    data["D_L"] = 1000.0 * data["lens_distance_kpc"]
    data["D_S"] = 1000.0 * data["source_distance_kpc"]
    data["D_L_kpc"] = data["lens_distance_kpc"]
    data["D_S_kpc"] = data["source_distance_kpc"]
    data["mu_rel"] = data["mu_rel_for_pipeline_masyr"]

    # Source radius/luminosity consistent with the catalog rho.
    theta_s_rad = (
        data["rho"].to_numpy(dtype=float)
        * data["thetaE_mas"].to_numpy(dtype=float)
        * u.mas
    ).to(u.rad).value

    source_radius = (
        theta_s_rad
        * data["source_distance_kpc"].to_numpy(dtype=float)
        * u.kpc
    ).to(u.R_sun)

    source_temperature = (
        10.0 ** data["logTe"].to_numpy(dtype=float)
    ) * u.K

    luminosity_ratio = (
        4.0
        * np.pi
        * sigma_sb
        * source_radius.to(u.m) ** 2
        * source_temperature ** 4
        / L_sun
    ).decompose().value

    data["source_radius_rsun_catalog"] = source_radius.value
    data["logL"] = np.log10(luminosity_ratio)
    data["W149"] = data["Y"]

    data["gall"] = data["l_deg"]
    data["galb"] = data["b_deg"]
    data["lens_ra"] = data["ra"]
    data["lens_dec"] = data["dec"]

    finite_required = [
        "ra",
        "dec",
        "D_L",
        "D_S",
        "mu_rel",
        "logTe",
        "logL",
        "lens_mass_msun",
        "u0",
        "t0_jd",
        "piE",
        "rho",
        "alpha_rad",
        *SOURCE_MAG_COLUMNS,
        *BLEND_COLUMNS,
    ]

    valid = np.ones(len(data), dtype=bool)

    for column in finite_required:
        valid &= np.isfinite(
            data[column].to_numpy(dtype=float)
        )

    valid &= data["D_L"].to_numpy(dtype=float) > 0.0
    valid &= (
        data["D_S"].to_numpy(dtype=float)
        > data["D_L"].to_numpy(dtype=float)
    )
    valid &= data["mu_rel"].to_numpy(dtype=float) > 0.0
    valid &= data["lens_mass_msun"].to_numpy(dtype=float) > 0.0
    valid &= data["tE_catalog_days"].to_numpy(dtype=float) > 0.0
    valid &= data["piE"].to_numpy(dtype=float) >= 0.0
    valid &= data["rho"].to_numpy(dtype=float) > 0.0

    blend_matrix = data[
        BLEND_COLUMNS
    ].to_numpy(dtype=float)

    blend_range_valid = np.all(
        (blend_matrix >= 0.0)
        & (blend_matrix <= 1.0),
        axis=1,
    )

    visible_filter_count = np.sum(
        blend_matrix > 0.0,
        axis=1,
    )

    data["catalog_visible_filter_count"] = (
        visible_filter_count.astype(int)
    )

    enough_visible_filters = (
        visible_filter_count
        >= BLENDING_MINIMUM_VISIBLE_FILTERS
    )

    blend_valid = (
        blend_range_valid
        & enough_visible_filters
    )

    if BLENDING_STRICT and np.any(~blend_range_valid):
        bad = data.loc[
            ~blend_range_valid,
            ["catalog_row", *BLEND_COLUMNS],
        ].head(20)

        raise ValueError(
            "Los factores de blending deben cumplir "
            "0 <= f_s <= 1. Primeros casos fuera de rango:\n"
            + bad.to_string(index=False)
        )

    if BLENDING_STRICT and np.any(~enough_visible_filters):
        bad = data.loc[
            ~enough_visible_filters,
            [
                "catalog_row",
                "catalog_visible_filter_count",
                *BLEND_COLUMNS,
            ],
        ].head(20)

        raise ValueError(
            "Hay eventos con menos filtros visibles que el mínimo "
            f"configurado ({BLENDING_MINIMUM_VISIBLE_FILTERS}). "
            "Primeros casos:\n"
            + bad.to_string(index=False)
        )

    valid &= blend_valid

    invalid = data.loc[~valid].copy()
    data = data.loc[valid].copy().reset_index(drop=True)

    if (
        max_base_events is not None
        and len(data) > max_base_events
    ):
        selected_indices = (
            data.sample(
                n=max_base_events,
                random_state=RANDOM_SEED,
            )
            .index
            .sort_values()
        )

        data = (
            data.loc[selected_indices]
            .copy()
            .reset_index(drop=True)
        )

    data.attrs["xi_unit_resolved"] = resolved_angle_unit
    data.attrs["alpha_unit_resolved"] = resolved_angle_unit

    return data, invalid


def make_field_name(l_deg, b_deg):
    l_signed = (
        (float(l_deg) + 180.0)
        % 360.0
    ) - 180.0

    name = (
        f"l{l_signed:+07.2f}_"
        f"b{float(b_deg):+06.2f}"
    )

    return name.replace(".", "p")


def parallax_components_from_catalog(row):
    """
    Convert the catalog amplitude and xi into pyLIMA (North, East)
    parallax components.

    Assumption
    ----------
    The catalog column ``xi`` is used explicitly.  The column ``alpha``,
    if present, is retained only as metadata and is not used here.

    For the paper-local Galactic tangent basis:

        n1 = direction of increasing Galactic longitude
        n2 = direction of increasing Galactic latitude

    the vector is

        piE_n1 = piE * cos(xi)
        piE_n2 = piE * sin(xi)

    and is then rotated into the ICRS North/East basis expected by pyLIMA.
    """

    xi_rad = float(row["xi_rad"])
    piE = float(row["piE"])

    diagnostics = {
        "galactic_n1_basis_pa_icrs_deg": np.nan,
        "galactic_n2_basis_pa_icrs_deg": np.nan,
        # Retained aliases for compatibility with previous result readers.
        "galactic_l_basis_pa_icrs_deg": np.nan,
        "galactic_b_basis_pa_icrs_deg": np.nan,
    }

    if PARALLAX_ANGLE_BASIS == "icrs_en":
        if PARALLAX_COMPONENT_CONVENTION == "east_cos_north_sin":
            piEE = piE * np.cos(xi_rad)
            piEN = piE * np.sin(xi_rad)
        else:
            piEN = piE * np.cos(xi_rad)
            piEE = piE * np.sin(xi_rad)

        return float(piEN), float(piEE), diagnostics

    # Paper-local basis:
    # n1 = +l and n2 = +b on the sky.
    # Astropy position_angle is measured east of ICRS north.
    l_deg = float(row["l_deg"])
    b_deg = float(row["b_deg"])

    center_gal = SkyCoord(
        l=l_deg * u.deg,
        b=b_deg * u.deg,
        frame="galactic",
    )
    center_icrs = center_gal.icrs

    epsilon = 1.0e-5 * u.deg
    cos_b = np.cos(np.deg2rad(b_deg))

    if abs(cos_b) < 1.0e-8:
        raise ValueError(
            "No se puede construir la base n1 cerca "
            "de un polo galáctico."
        )

    plus_n1 = SkyCoord(
        l=(l_deg * u.deg + epsilon / cos_b),
        b=b_deg * u.deg,
        frame="galactic",
    ).icrs

    plus_n2 = SkyCoord(
        l=l_deg * u.deg,
        b=(b_deg * u.deg + epsilon),
        frame="galactic",
    ).icrs

    pa_n1 = float(
        center_icrs.position_angle(plus_n1).to_value(u.rad)
    )
    pa_n2 = float(
        center_icrs.position_angle(plus_n2).to_value(u.rad)
    )

    piE_n1 = piE * np.cos(xi_rad)
    piE_n2 = piE * np.sin(xi_rad)

    # A tangent vector with position angle PA has components
    # North = amplitude*cos(PA), East = amplitude*sin(PA).
    piEN = (
        piE_n1 * np.cos(pa_n1)
        + piE_n2 * np.cos(pa_n2)
    )
    piEE = (
        piE_n1 * np.sin(pa_n1)
        + piE_n2 * np.sin(pa_n2)
    )

    pa_n1_deg = float(np.degrees(pa_n1))
    pa_n2_deg = float(np.degrees(pa_n2))

    diagnostics["galactic_n1_basis_pa_icrs_deg"] = pa_n1_deg
    diagnostics["galactic_n2_basis_pa_icrs_deg"] = pa_n2_deg
    diagnostics["galactic_l_basis_pa_icrs_deg"] = pa_n1_deg
    diagnostics["galactic_b_basis_pa_icrs_deg"] = pa_n2_deg

    recovered = float(np.hypot(piEN, piEE))

    if not np.isclose(
        recovered,
        piE,
        rtol=1.0e-7,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "La transformación de piE desde la base (n1,n2) "
            "a (North,East) no conservó la amplitud: "
            f"input={piE}, output={recovered}."
        )

    return float(piEN), float(piEE), diagnostics


def build_tasks(prepared_catalog):
    """Create exactly one task per catalog row."""

    tasks = []

    for prepared_index, row in prepared_catalog.iterrows():
        simulation_seed = int(
            (
                RANDOM_SEED
                + int(row["catalog_row"])
            )
            % (2**32 - 1)
        )

        piEN, piEE, direction_info = (
            parallax_components_from_catalog(row)
        )

        task = {
            "global_i": int(prepared_index),
            "simulation_seed": simulation_seed,
            "prepared_index": int(prepared_index),
            "catalog_row": int(row["catalog_row"]),
            "catalog_event_id": int(row["catalog_event_id"]),
            "alpha_catalog": float(row["alpha_catalog"]),
            "xi_catalog": float(row["xi_catalog"]),
            "xi_rad": float(row["xi_rad"]),
            "xi_deg": float(row["xi_deg"]),
            # Compatibility aliases.
            "alpha_rad": float(row["xi_rad"]),
            "alpha_deg": float(row["xi_deg"]),
            "angle_semantics": "xi_catalog_column",
            "piE": float(row["piE"]),
            "piEN": piEN,
            "piEE": piEE,
            "field_name": make_field_name(
                row["l_deg"],
                row["b_deg"],
            ),
            **direction_info,
        }

        tasks.append(task)

    return tasks


# ============================================================================
# Single-row catalog passed to sim_fit
# ============================================================================

PAIR_ROW_REQUIRED = [
    "D_S",
    "D_L",
    "D_S_kpc",
    "D_L_kpc",
    "mu_rel",
    "logL",
    "logTe",
    "ra",
    "dec",
    "lens_ra",
    "lens_dec",
    "gall",
    "galb",
    "u",
    "g",
    "r",
    "i",
    "z",
    "Y",
    "W149",
]


def build_single_row_pair_catalog(base_row, task):
    row = base_row.copy()

    # The same xi sets the source-lens trajectory direction.
    row["theta_rad"] = float(
        task["xi_rad"]
    )

    row["traj_angle"] = float(
        task["xi_rad"]
    )

    row["xi_rad"] = float(
        task["xi_rad"]
    )

    row["maf_ra"] = float(row["ra"])
    row["maf_dec"] = float(row["dec"])

    missing = [
        column for column in PAIR_ROW_REQUIRED
        if column not in row.index
    ]

    if missing:
        raise KeyError(
            f"Faltan columnas para sim_fit: {missing}"
        )

    return pd.DataFrame([row])


def build_event_fit_window(base_row):
    """
    Return the absolute-JD interval applied only to the fit arrays.

    For half_width_tE = 3.5:
        [t0 - 3.5*tE, t0 + 3.5*tE]

    If the option is disabled, return None and use the complete light curve.
    """

    if not FIT_WINDOW_ENABLED:
        return None

    t0_jd = float(base_row["t0_jd"])
    tE_days = float(base_row["tE_catalog_days"])
    half_width_days = FIT_WINDOW_HALF_WIDTH_TE * tE_days

    return (
        t0_jd - half_width_days,
        t0_jd + half_width_days,
    )


def fixed_param_samplers(base_row, task):
    """
    Force the catalog event and the selected parallax orientation.
    """

    return {
        "star_mass": {
            "type": "fixed",
            "value": float(
                base_row["lens_mass_msun"]
            ),
        },
        "mass_planet": {
            "type": "fixed",
            "value": 0.0,
        },
        "u0": {
            "type": "fixed",
            "value": float(
                base_row["u0"]
            ),
        },
        "t0": {
            "type": "fixed",
            "value": float(
                base_row["t0_jd"]
            ),
        },
        "piEN": {
            "type": "fixed",
            "value": float(
                task["piEN"]
            ),
        },
        "piEE": {
            "type": "fixed",
            "value": float(
                task["piEE"]
            ),
        },
        "rho": {
            "type": "fixed",
            "value": float(
                base_row["rho"]
            ),
        },
    }


def task_metadata(base_row, task):
    fit_window = build_event_fit_window(base_row)

    metadata = {
        **task,
        "l_deg": float(base_row["l_deg"]),
        "b_deg": float(base_row["b_deg"]),
        "ra": float(base_row["ra"]),
        "dec": float(base_row["dec"]),
        "lens_mass_msun": float(base_row["lens_mass_msun"]),
        "D_L_kpc": float(base_row["lens_distance_kpc"]),
        "D_S_kpc": float(base_row["source_distance_kpc"]),
        "tE_catalog_days": float(base_row["tE_catalog_days"]),
        "thetaE_mas": float(base_row["thetaE_mas"]),
        "mu_rel_catalog_masyr": float(
            base_row.get("mu_rel_catalog_masyr", np.nan)
        ),
        "mu_rel_for_pipeline_masyr": float(
            base_row["mu_rel_for_pipeline_masyr"]
        ),
        "t0_catalog_days": float(base_row["t0_catalog_days"]),
        "t0_jd": float(base_row["t0_jd"]),
        "t0_origin": str(
            base_row.get("t0_origin", "global_zero_jd")
        ),
        "t0_reference_jd": float(
            base_row.get("t0_reference_jd", np.nan)
        ),
        "t0_reference_mjd": float(
            base_row.get("t0_reference_mjd", np.nan)
        ),
        "t0_jd_original_global_zero": float(
            base_row.get("t0_jd_original_global_zero", np.nan)
        ),
        "u0": float(base_row["u0"]),
        "rho_catalog": float(base_row["rho"]),
        "source_radius_rsun_catalog": float(
            base_row["source_radius_rsun_catalog"]
        ),
        "logL_from_rho_thetaE_DS_Teff": float(base_row["logL"]),
        "source_mag_u": float(base_row["u"]),
        "source_mag_g": float(base_row["g"]),
        "source_mag_r": float(base_row["r"]),
        "source_mag_i": float(base_row["i"]),
        "source_mag_z": float(base_row["z"]),
        "source_mag_y": float(base_row["Y"]),
        "source_fraction_u": float(base_row["blend_u"]),
        "source_fraction_g": float(base_row["blend_g"]),
        "source_fraction_r": float(base_row["blend_r"]),
        "source_fraction_i": float(base_row["blend_i"]),
        "source_fraction_z": float(base_row["blend_z"]),
        "source_fraction_y": float(base_row["blend_y"]),
        "catalog_visible_filter_count": int(
            base_row["catalog_visible_filter_count"]
        ),
        "catalog_zero_fraction_bands": ",".join(
            band
            for band, column in zip(
                ("u", "g", "r", "i", "z", "y"),
                BLEND_COLUMNS,
            )
            if float(base_row[column]) == 0.0
        ),
        "blend_definition": BLENDING_ASSUMPTION["definition"],
        "alpha_catalog": float(base_row["alpha_catalog"]),
        "alpha_interpretation": "not_used_for_xi",
        "xi_catalog": float(base_row["xi_catalog"]),
        "xi_rad": float(base_row["xi_rad"]),
        "xi_deg": float(base_row["xi_deg"]),
        "alpha_unit_resolved": str(base_row["alpha_unit_resolved"]),
        "xi_unit_resolved": str(base_row["alpha_unit_resolved"]),
        "parallax_angle_basis": PARALLAX_ANGLE_BASIS,
        "parallax_component_convention": PARALLAX_COMPONENT_CONVENTION,
        "n_data_catalog": float(
            base_row.get("n_data_catalog", np.nan)
        ),
        "delta_chi2_catalog": float(
            base_row.get("delta_chi2_catalog", np.nan)
        ),
        "fwhm_catalog_days": float(
            base_row.get("fwhm_catalog_days", np.nan)
        ),
        "fit_window_enabled": FIT_WINDOW_ENABLED,
        "fit_window_half_width_tE": (
            FIT_WINDOW_HALF_WIDTH_TE
            if FIT_WINDOW_ENABLED
            else np.nan
        ),
        "fit_window_minimum_total_points": (
            FIT_WINDOW_MINIMUM_TOTAL_POINTS
            if FIT_WINDOW_ENABLED
            else np.nan
        ),
        "fit_window_t_min_jd": (
            float(fit_window[0])
            if fit_window is not None
            else np.nan
        ),
        "fit_window_t_max_jd": (
            float(fit_window[1])
            if fit_window is not None
            else np.nan
        ),
        "fit_window_total_days": (
            float(fit_window[1] - fit_window[0])
            if fit_window is not None
            else np.nan
        ),
    }

    return metadata


def add_metadata_to_result_parquets(
    event_results_dir,
    metadata,
):
    event_results_dir = Path(
        event_results_dir
    )

    files = sorted(
        event_results_dir.rglob("*.parquet")
    )

    for file in files:
        try:
            data = pd.read_parquet(file)

            for key, value in metadata.items():
                data[key] = value

            data.to_parquet(
                file,
                index=False,
            )

        except Exception as error:
            print(
                "[warning] No pude agregar "
                f"metadatos a {file}: {error!r}"
            )


# ============================================================================
# Worker
# ============================================================================
def apply_t0_from_first_maf_timestamp(base_row, config):
    """
    Redefine t0_jd como:

        t0_jd = primer timestamp Rubin/MAF usado por este evento
                + t0_catalog_days

    Usa el mismo dataSlice que después usará sim_event, incluyendo
    rubin_pointing_mode='source' y rubin_cache_cell_deg.
    """

    import set_telescopes_pyLIMA as stp

    base_row = base_row.copy()

    if not bool(config.get("use_rubin", True)):
        raise RuntimeError(
            "t0_from_first_maf_timestamp requiere use_rubin=True."
        )

    source_ra = float(base_row["ra"])
    source_dec = float(base_row["dec"])

    _, dataSlice, _ = stp.tel_roman_rubin(
        config["path_ephemerides"],
        time_window=None,
        use_roman=bool(config.get("use_roman", False)),
        use_rubin=bool(config.get("use_rubin", True)),
        Ra=source_ra,
        Dec=source_dec,
        rubin_pointing_mode=config.get("rubin_pointing_mode", "source"),
        rubin_cache_cell_deg=config.get("rubin_cache_cell_deg", None),
    )

    if dataSlice is None or len(dataSlice) == 0:
        raise RuntimeError(
            "No hay observaciones Rubin/MAF para calcular el primer timestamp."
        )

    mjd = np.asarray(
        dataSlice["observationStartMJD"],
        dtype=float,
    )

    jd = mjd + 2400000.5

    finite = jd[np.isfinite(jd)]

    if len(finite) == 0:
        raise RuntimeError(
            "dataSlice no contiene observationStartMJD finitos."
        )

    first_timestamp_jd = float(np.min(finite))

    old_t0_jd = float(base_row["t0_jd"])
    new_t0_jd = first_timestamp_jd + float(base_row["t0_catalog_days"])

    base_row["t0_jd_original_global_zero"] = old_t0_jd
    base_row["t0_reference_jd"] = first_timestamp_jd
    base_row["t0_reference_mjd"] = first_timestamp_jd - 2400000.5
    base_row["t0_origin"] = "first_maf_timestamp"
    base_row["t0_jd"] = new_t0_jd

    return base_row
    
def run_single_event(task):
    config = GLOBAL_WORKER_CONFIG
    catalog = GLOBAL_PREPARED_CATALOG
    base_row = catalog.iloc[
        int(task["prepared_index"])
    ].copy()

    # Redefinir t0 usando el primer timestamp real del dataSlice MAF
    base_row = apply_t0_from_first_maf_timestamp(
        base_row,
        config,
    )

    global_i = int(task["global_i"])
    simulation_seed = int(task["simulation_seed"])
    field_name = str(task["field_name"])
    event_tag = f"event_{global_i:07d}"

    model_dir = (
        Path(config["models_dir"])
        / field_name
        / event_tag
    )
    fit_dir = (
        Path(config["fits_dir"])
        / field_name
        / event_tag
    )
    results_dir = (
        Path(config["results_dir"])
        / field_name
        / event_tag
    )
    log_file = (
        Path(config["logs_dir"])
        / field_name
        / f"{event_tag}.log"
    )

    for directory in [
        model_dir,
        fit_dir,
        results_dir,
        log_file.parent,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    metadata = task_metadata(base_row, task)

    summary = {
        **metadata,
        "status": "started",
        "sim_fit_status": "",
        "error": "",
        "log_file": str(log_file),
        "model_dir": str(model_dir),
        "fit_dir": str(fit_dir),
        "results_dir": str(results_dir),
    }

    set_runtime_event_context(base_row)

    try:
        pair_catalog = build_single_row_pair_catalog(
            base_row,
            task,
        )
        param_samplers = fixed_param_samplers(
            base_row,
            task,
        )

        with open(log_file, "w") as log:
            with redirect_stdout(log), redirect_stderr(log):
                print("=" * 80)
                print("LSSTMONTS catalog event")
                print("=" * 80)
                print(json.dumps(metadata, indent=2, default=str))
                print("param_samplers")
                print(json.dumps(param_samplers, indent=2))
                print("=" * 80)
                print(
                    "Simulation cadence: complete MAF light curve; "
                    "fit window is applied after simulation."
                )
                if _RUNTIME_FIT_WINDOW is not None:
                    print(
                        "fit-only window [JD]: "
                        f"{_RUNTIME_FIT_WINDOW[0]:.8f}, "
                        f"{_RUNTIME_FIT_WINDOW[1]:.8f}"
                    )

                sim_fit_kwargs = {
                    "model": config["model"],
                    "algo": config["algo"],
                    "path_TRILEGAL_set": None,
                    "path_GENULENS_set": None,
                    "path_to_save_model": str(model_dir) + "/",
                    "path_to_save_fit": str(fit_dir) + "/",
                    "path_ephemerides": config["path_ephemerides"],
                    "path_to_save_results": str(results_dir) + "/",
                    "catalog_mode": "astrodatalab_pairs",
                    "pair_catalog": pair_catalog,
                    "use_roman": config["use_roman"],
                    "use_rubin": config["use_rubin"],
                    "param_samplers": param_samplers,
                    "fit_model": config["fit_model"],
                    "fit_parallax": config["fit_parallax"],
                    "fit_bounds": config["fit_bounds"],
                    "rubin_pointing_mode": config["rubin_pointing_mode"],
                    "rubin_cache_cell_deg": config["rubin_cache_cell_deg"],
                    "return_data": True,
                    # Critical: the simulation is not temporally cropped.
                    "time_window": None,
                }

                sim_fit_parameters = inspect.signature(sim_fit).parameters

                if "apply_detection_criteria" in sim_fit_parameters:
                    sim_fit_kwargs["apply_detection_criteria"] = config[
                        "apply_detection_criteria"
                    ]

                result = sim_fit(
                    simulation_seed,
                    config["system_type"],
                    **sim_fit_kwargs,
                )

                if isinstance(result, dict):
                    sim_status = str(result.get("status", ""))
                    summary["sim_fit_status"] = sim_status

                    if sim_status == "fitted":
                        summary["status"] = "ok"
                    elif sim_status == "rejected":
                        summary["status"] = "rejected_pipeline"
                    else:
                        summary["status"] = "returned_dict"
                else:
                    summary["status"] = "ok"
                    summary["sim_fit_status"] = type(result).__name__

                fit_counts = dict(_RUNTIME_LAST_FIT_COUNTS)
                summary["fit_n_points_total"] = int(
                    sum(fit_counts.values())
                )
                for band in ["W149", "u", "g", "r", "i", "z", "y"]:
                    summary[f"fit_n_points_{band}"] = int(
                        fit_counts.get(band, 0)
                    )

                metadata_with_fit = {
                    **metadata,
                    "fit_n_points_total": summary["fit_n_points_total"],
                    **{
                        f"fit_n_points_{band}": summary[
                            f"fit_n_points_{band}"
                        ]
                        for band in ["W149", "u", "g", "r", "i", "z", "y"]
                    },
                }
                add_metadata_to_result_parquets(
                    results_dir,
                    metadata_with_fit,
                )

                try:
                    import set_telescopes_pyLIMA as stp

                    info = getattr(stp, "LAST_DATASLICE_INFO", {})
                    summary["maf_n_obs"] = info.get("n_obs", np.nan)
                    summary["maf_ra"] = info.get(
                        "maf_Ra", info.get("Ra", np.nan)
                    )
                    summary["maf_dec"] = info.get(
                        "maf_Dec", info.get("Dec", np.nan)
                    )
                    print("LAST_DATASLICE_INFO")
                    print(info)
                except Exception as error:
                    print(
                        "[warning] LAST_DATASLICE_INFO: "
                        f"{error!r}"
                    )

    except FitWindowRejected as error:
        summary["status"] = "rejected_fit_window"
        summary["sim_fit_status"] = "fit_window_rejected"
        summary["error"] = str(error)
        fit_counts = dict(_RUNTIME_LAST_FIT_COUNTS)
        summary["fit_n_points_total"] = int(sum(fit_counts.values()))
        for band in ["W149", "u", "g", "r", "i", "z", "y"]:
            summary[f"fit_n_points_{band}"] = int(
                fit_counts.get(band, 0)
            )

        with open(log_file, "a") as log:
            log.write("\n" + "=" * 80 + "\n")
            log.write("FIT WINDOW REJECTED\n")
            log.write("=" * 80 + "\n")
            log.write(str(error) + "\n")
            log.write(f"Counts: {fit_counts}\n")

    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = str(error)

        with open(log_file, "a") as log:
            log.write("\n" + "=" * 80 + "\n")
            log.write("ERROR\n")
            log.write("=" * 80 + "\n")
            log.write(traceback.format_exc())

    finally:
        clear_runtime_event_context()

    return summary


# ============================================================================
# Outputs
# ============================================================================

def save_summary(summary_rows):
    summary = pd.DataFrame(
        summary_rows
    )

    if len(summary) > 0:
        summary = (
            summary.sort_values("global_i")
            .reset_index(drop=True)
        )

    summary.to_csv(
        DIRS["logs"] / "run_summary.csv",
        index=False,
    )

    summary.to_parquet(
        DIRS["logs"] / "run_summary.parquet",
        index=False,
    )

    return summary


def print_diagnostics(
    prepared_catalog,
    invalid_catalog,
    tasks,
    workers,
):
    print("=" * 80)
    print("LSSTMONTS preparation")
    print("=" * 80)
    print(f"Valid selected events:  {len(prepared_catalog)}")
    print(f"Invalid excluded rows:  {len(invalid_catalog)}")
    print("Tasks per event:        1")
    print(f"Total simulation tasks: {len(tasks)}")
    print(f"N workers:              {workers}")
    print(f"Config file:            {CONFIG_PATH}")
    print(f"T0_ZERO_JD:             {T0_ZERO_JD}")
    print(f"Catalog angle column:   {PARALLAX_ANGLE_COLUMN}")
    print("Angle interpretation:   xi column used directly")
    print(
        "Trajectory angle unit:  "
        f"{prepared_catalog.attrs.get('alpha_unit_resolved', 'unknown')}"
    )
    print(f"Xi tangent-plane basis: {PARALLAX_ANGLE_BASIS}")
    print(
        "Parallax components:    "
        f"{PARALLAX_COMPONENT_CONVENTION}"
    )
    print("Simulation time range:  complete MAF cadence")
    if FIT_WINDOW_ENABLED:
        print(
            "Fit-only window:      "
            f"t0 +/- {FIT_WINDOW_HALF_WIDTH_TE:g} tE "
            f"(minimum {FIT_WINDOW_MINIMUM_TOTAL_POINTS} total points)"
        )
    else:
        print("Fit-only window:      disabled")
    print(f"Run directory:          {RUN_DIR}")
    print(f"Blending:               {BLENDING_ASSUMPTION}")
    print("=" * 80)

    if (
        "mu_rel_catalog_masyr" in prepared_catalog.columns
        and len(prepared_catalog) > 0
    ):
        denominator = prepared_catalog[
            "mu_rel_catalog_masyr"
        ].replace(0.0, np.nan)

        ratio = (
            prepared_catalog["mu_rel_for_pipeline_masyr"]
            / denominator
        )

        print("Reconstructed mu_rel [mas/yr]:")
        print(
            prepared_catalog[
                "mu_rel_for_pipeline_masyr"
            ].describe()
        )
        print("Reconstructed/catalog-labeled ratio:")
        print(
            ratio.replace([np.inf, -np.inf], np.nan).describe()
        )
        print("=" * 80)


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run LSSTMONTS FSPL simulations using a JSON/YAML "
            "configuration file."
        )
    )

    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help=(
            "JSON/YAML configuration file. It is loaded before "
            "the rest of the program."
        ),
    )

    parser.add_argument(
        "--max-base-events",
        default=None,
        help=(
            "Optional CLI override for selection.max_base_events. "
            "Use 'all' for the complete catalog."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Optional CLI override for execution.workers."
        ),
    )

    parser.add_argument(
        "--read-nrows",
        default=None,
        help=(
            "Optional CLI override for catalog.read_nrows. "
            "Use this for quick prepare-only tests with huge files."
        ),
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Override the config and only prepare the catalog/tasks."
        ),
    )

    return parser


# ============================================================================
# Main
# ============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    max_base_events = parse_max_events(
        MAX_BASE_EVENTS_CONFIG
        if args.max_base_events is None
        else args.max_base_events
    )

    workers = (
        N_WORKERS
        if args.workers is None
        else int(args.workers)
    )

    prepare_only = (
        PREPARE_ONLY_CONFIG
        or bool(args.prepare_only)
    )

    read_nrows = parse_optional_positive_int(
        READ_NROWS_CONFIG
        if args.read_nrows is None
        else args.read_nrows,
        name="catalog.read_nrows",
    )

    if workers <= 0:
        raise ValueError(
            "execution.workers debe ser positivo."
        )

    sim_fit_parameters = inspect.signature(
        sim_fit
    ).parameters

    if not hasattr(frr, "extract_lightcurves_for_fit"):
        raise RuntimeError(
            "Tu functions_roman_rubin.py no expone "
            "extract_lightcurves_for_fit. Esta función es necesaria para "
            "aplicar t0 +/- k*tE solamente al ajuste."
        )

    if "time_window" not in sim_fit_parameters:
        raise RuntimeError(
            "Tu sim_fit no acepta time_window. El runner pasa "
            "time_window=None para asegurar que la simulación use toda "
            "la cadencia MAF."
        )

    if (
        not APPLY_DETECTION_CRITERIA
        and "apply_detection_criteria"
        not in sim_fit_parameters
    ):
        raise RuntimeError(
            "El config solicita apply_detection_criteria: false, "
            "pero tu función sim_fit todavía no acepta ese argumento. "
            "Propagalo por sim_fit -> simulate_event_for_fit -> sim_event, "
            "o usá apply_detection_criteria: true."
        )

    log_step("[main] Loading raw catalog ...")
    raw_catalog = load_raw_catalog(
        COLUMNS_FILE,
        DATA_FILE,
        nrows=read_nrows,
    )

    log_step("[main] Preparing catalog ...")
    prepared_catalog, invalid_catalog = (
        prepare_catalog(
            raw_catalog,
            max_base_events=max_base_events,
        )
    )
    log_step(
        "[main] Preparation done. "
        f"valid={len(prepared_catalog)}, invalid={len(invalid_catalog)}"
    )

    if len(prepared_catalog) == 0:
        raise RuntimeError(
            "No quedó ningún evento válido."
        )

    tasks = build_tasks(
        prepared_catalog
    )

    print_diagnostics(
        prepared_catalog,
        invalid_catalog,
        tasks,
        workers,
    )

    prepared_catalog.to_parquet(
        DIRS["catalogs"]
        / "lsstmonts_prepared.parquet",
        index=False,
    )

    prepared_catalog.to_csv(
        DIRS["catalogs"]
        / "lsstmonts_prepared.csv",
        index=False,
    )

    if len(invalid_catalog) > 0:
        invalid_catalog.to_csv(
            DIRS["catalogs"]
            / "lsstmonts_invalid_rows.csv",
            index=False,
        )

    tasks_table = pd.DataFrame(tasks)

    tasks_table.to_csv(
        DIRS["config"]
        / "event_tasks.csv",
        index=False,
    )

    tasks_table.to_parquet(
        DIRS["config"]
        / "event_tasks.parquet",
        index=False,
    )

    run_config = {
        "CONFIG_PATH": str(CONFIG_PATH),
        "BASE_DIR": str(BASE_DIR),
        "RUN_DIR": str(RUN_DIR),
        "COLUMNS_FILE": str(COLUMNS_FILE),
        "DATA_FILE": str(DATA_FILE),
        "ROMAN_RUBIN_DIR": str(
            ROMAN_RUBIN_DIR
        ),
        "RUBIN_SIM_DATA_DIR": str(
            RUBIN_SIM_DATA_DIR
        ),
        "RUBIN_OPSIM_DB_PATH": os.environ.get("RUBIN_OPSIM_DB_PATH", ""),
        "PATH_EPHEMERIDES": str(
            PATH_EPHEMERIDES
        ),
        "TASKS_PER_EVENT": 1,
        "PARALLAX_ANGLE_COLUMN": PARALLAX_ANGLE_COLUMN,
        "PARALLAX_ANGLE_SEMANTICS": PARALLAX_ANGLE_SEMANTICS,
        "ALPHA_ASSUMED_EQUAL_TO_XI": False,
        "XI_COLUMN_USED_DIRECTLY": True,
        "PARALLAX_ANGLE_UNIT_REQUESTED": PARALLAX_ANGLE_UNIT,
        "PARALLAX_ANGLE_UNIT_RESOLVED": prepared_catalog.attrs.get(
            "xi_unit_resolved", "unknown"
        ),
        "PARALLAX_ANGLE_BASIS": PARALLAX_ANGLE_BASIS,
        "PARALLAX_COMPONENT_CONVENTION":
            PARALLAX_COMPONENT_CONVENTION,
        "T0_ZERO_JD": T0_ZERO_JD,
        "MAX_BASE_EVENTS": max_base_events,
        "READ_NROWS": read_nrows,
        "N_BASE_EVENTS_SELECTED":
            len(prepared_catalog),
        "N_TASKS": len(tasks),
        "N_WORKERS": workers,
        "RANDOM_SEED": RANDOM_SEED,
        "SYSTEM_TYPE": SYSTEM_TYPE,
        "MODEL": MODEL,
        "FIT_MODEL": FIT_MODEL,
        "FIT_PARALLAX": FIT_PARALLAX,
        "APPLY_DETECTION_CRITERIA":
            APPLY_DETECTION_CRITERIA,
        "ALGO": ALGO,
        "USE_ROMAN": USE_ROMAN,
        "USE_RUBIN": USE_RUBIN,
        "RUBIN_POINTING_MODE":
            RUBIN_POINTING_MODE,
        "RUBIN_CACHE_CELL_DEG":
            RUBIN_CACHE_CELL_DEG,
        "FIT_BOUNDS_NOPIE":
            FIT_BOUNDS_NOPIE,
        "SIMULATION_TIME_RANGE": "complete MAF cadence",
        "FIT_WINDOW_ENABLED": FIT_WINDOW_ENABLED,
        "FIT_WINDOW_HALF_WIDTH_TE": FIT_WINDOW_HALF_WIDTH_TE,
        "FIT_WINDOW_MINIMUM_TOTAL_POINTS":
            FIT_WINDOW_MINIMUM_TOTAL_POINTS,
        "FIT_WINDOW_DEFINITION":
            "[t0-k*tE, t0+k*tE] in absolute JD, fit only",
        "BLENDING_ASSUMPTION":
            BLENDING_ASSUMPTION,
        "BLENDING_ZERO_MEANS_UNAVAILABLE_FILTER":
            BLENDING_ZERO_MEANS_UNAVAILABLE_FILTER,
        "BLENDING_MINIMUM_VISIBLE_FILTERS":
            BLENDING_MINIMUM_VISIBLE_FILTERS,
        "MU_REL_ASSUMPTION":
            "thetaE_mas * 365.25 / tE_catalog_days",
        "SOURCE_LUMINOSITY_CONSTRUCTION":
            "thetaS=rho*thetaE; R=thetaS*DS; "
            "L=4*pi*sigma_SB*R^2*Teff^4",
        "SYSTEM_TYPE_INTERNAL_NOTE":
            "FFP branch used only because current event_param adds rho "
            "for that branch; star_mass is fixed to catalog lens mass "
            "and mass_planet is fixed to zero.",
    }

    shutil.copy2(
        CONFIG_PATH,
        DIRS["config"] / ("input_config" + CONFIG_PATH.suffix),
    )

    with open(
        DIRS["config"] / "run_config.json",
        "w",
    ) as file:
        json.dump(
            run_config,
            file,
            indent=2,
        )

    if prepare_only:
        print(
            "Preparation finished. "
            "No simulations were run."
        )

        return

    worker_config = {
        "system_type": SYSTEM_TYPE,
        "model": MODEL,
        "fit_model": FIT_MODEL,
        "fit_parallax": FIT_PARALLAX,
        "algo": ALGO,
        "use_roman": USE_ROMAN,
        "use_rubin": USE_RUBIN,
        "path_ephemerides": str(
            PATH_EPHEMERIDES
        ),
        "models_dir": str(
            DIRS["models"]
        ),
        "fits_dir": str(
            DIRS["fits"]
        ),
        "results_dir": str(
            DIRS["results"]
        ),
        "logs_dir": str(
            DIRS["logs"]
        ),
        "fit_bounds":
            FIT_BOUNDS_NOPIE,
        "rubin_pointing_mode":
            RUBIN_POINTING_MODE,
        "rubin_cache_cell_deg":
            RUBIN_CACHE_CELL_DEG,
        "apply_detection_criteria":
            APPLY_DETECTION_CRITERIA,
        "fit_window_enabled": FIT_WINDOW_ENABLED,
        "fit_window_half_width_tE": FIT_WINDOW_HALF_WIDTH_TE,
        "fit_window_minimum_total_points":
            FIT_WINDOW_MINIMUM_TOTAL_POINTS,
    }

    try:
        mp_context = mp.get_context(
            "fork"
        )

    except ValueError:
        mp_context = None

    executor_kwargs = {
        "max_workers": workers,
        "initializer": init_worker,
        "initargs": (
            prepared_catalog,
            worker_config,
        ),
    }

    if mp_context is not None:
        executor_kwargs[
            "mp_context"
        ] = mp_context

    summary_rows = []

    with ProcessPoolExecutor(
        **executor_kwargs
    ) as executor:

        future_to_task = {
            executor.submit(
                run_single_event,
                task,
            ): task
            for task in tasks
        }

        for completed, future in enumerate(
            as_completed(future_to_task),
            start=1,
        ):
            task = future_to_task[future]

            try:
                row = future.result()

            except Exception as error:
                row = {
                    **task,
                    "status": "executor_failed",
                    "error": str(error),
                }

            summary_rows.append(row)
            save_summary(summary_rows)

            if (
                completed == 1
                or completed % 25 == 0
                or completed == len(tasks)
            ):
                status_counts = pd.Series(
                    [
                        item.get("status", "")
                        for item in summary_rows
                    ]
                ).value_counts()

                status_text = ", ".join(
                    f"{status}={count}"
                    for status, count
                    in status_counts.items()
                )

                print(
                    f"[{completed}/{len(tasks)}] "
                    f"{status_text}"
                )

    summary = save_summary(
        summary_rows
    )

    print("=" * 80)
    print("Run finished")
    print("=" * 80)

    print(
        summary["status"]
        .value_counts(
            dropna=False
        )
    )

    print(
        "Summary: "
        f"{DIRS['logs'] / 'run_summary.parquet'}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
