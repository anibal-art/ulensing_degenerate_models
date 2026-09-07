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
- the catalog t0 is interpreted as days after the first OpSim/MAF
  timestamp read for the selected field/source;
- only the fit is restricted to t0 +/- k*tE (k=3.5 by default);
- the catalog was already detection-selected, so the internal
  deviation-from-constant criterion is disabled;
- band availability is controlled by DetectionFlag_* by default;
- the t0 reference timestamp is the first MAF timestamp in the
  catalog-visible bands, not necessarily the first raw MAF timestamp;
- the photometric m5/5-sigma point filter is controlled explicitly
  from simulation.apply_photometric_filter in the configuration file.

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
    python run_lsstmonts_catalog_hidden_parallax.py \
        --config configs/config_lsstmonts_baseline_v5p3p5.json

Preparation/validation only
---------------------------
    python run_lsstmonts_catalog_hidden_parallax.py \
        --config configs/config_lsstmonts_baseline_v5p3p5.json \
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
import time
import shutil
import signal
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
    
def _parse_early_cli_args():
    """
    Read CLI options needed before module-level paths are built.

    parse_known_args() leaves the normal CLI arguments for main().
    The chunk/output arguments are parsed early so each SLURM array task can
    write to an independent subdirectory without needing a second config file.
    """

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_FILE),
    )

    parser.add_argument(
        "--catalog-row-start",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--catalog-row-stop",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--chunk-id",
        default=None,
    )

    parser.add_argument(
        "--run-name-suffix",
        default=None,
    )

    args, _ = parser.parse_known_args()

    return args


def _early_config_path():
    args = _parse_early_cli_args()
    return Path(args.config).expanduser().resolve()


def _sanitize_output_label(value):
    """Return a conservative filesystem-safe output label."""

    if value in (None, ""):
        return ""

    text = str(value).strip()
    allowed = []

    for char in text:
        if char.isalnum() or char in {"_", "-", "."}:
            allowed.append(char)
        else:
            allowed.append("_")

    return "".join(allowed).strip("._-")


def _early_chunk_output_label(args):
    """
    Determine the chunk subdirectory name from CLI arguments.

    Priority:
      1. explicit --run-name-suffix
      2. explicit --chunk-id
      3. row window label when --catalog-row-start/stop are given

    If no chunk option is given, returns an empty string and the runner keeps
    the original one-config, one-run directory behavior.
    """

    explicit = _sanitize_output_label(getattr(args, "run_name_suffix", None))

    if explicit:
        return explicit

    chunk_id = getattr(args, "chunk_id", None)

    if chunk_id not in (None, ""):
        try:
            return f"chunk_{int(chunk_id):06d}"
        except Exception:
            return "chunk_" + _sanitize_output_label(chunk_id)

    start = getattr(args, "catalog_row_start", None)
    stop = getattr(args, "catalog_row_stop", None)

    if start is not None or stop is not None:
        start_label = int(start) if start is not None else 0
        stop_label = "end" if stop is None else f"{int(stop):07d}"
        return f"rows_{start_label:07d}_{stop_label}"

    return ""


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


EARLY_CLI_ARGS = _parse_early_cli_args()
CONFIG_PATH = Path(EARLY_CLI_ARGS.config).expanduser().resolve()
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


def _expand_config_variables(value, extra_env=None):
    """
    Expande ~, variables de entorno y placeholders tipo ${VAR}.

    A diferencia de os.path.expandvars, permite pasar variables extra
    que todavía no están necesariamente en os.environ. Esto es útil para
    paths como:

        ${RUBIN_SIM_DATA_DIR}/sim_baseline/baseline_v5.3.5_10yrs.db

    dentro del config.
    """

    text = os.path.expanduser(str(value))

    env = dict(os.environ)
    env.setdefault("HOME", str(HOME))

    if extra_env is not None:
        env.update({
            str(key): str(val)
            for key, val in extra_env.items()
            if val not in (None, "")
        })

    # Sustitución explícita de ${VAR}.
    for key, val in env.items():
        text = text.replace("${" + key + "}", str(val))

    # Fallback para $VAR.
    text = os.path.expandvars(text)

    return text


def resolve_config_path(value, default=None, extra_env=None):
    """
    Expand ~ and environment variables. Relative paths are interpreted
    relative to the config file, not relative to the current shell directory.
    """

    if value is None:
        value = default

    if value is None:
        return None

    expanded = _expand_config_variables(
        value,
        extra_env=extra_env,
    )

    path = Path(expanded)

    if not path.is_absolute():
        path = CONFIG_DIR / path

    return path.resolve()


def _paths_section():
    value = CONFIG.get("paths", {})

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(
            "La sección 'paths' del config debe ser un diccionario."
        )

    return value


def configure_paths_from_config():
    """
    Resuelve y exporta los paths machine-dependent del proyecto.

    Esquema recomendado en el JSON/YAML:

        paths:
          microlensing_root: /home/anibalvarela/microlensing
          ulensing_degenerate_models_root: /export/.../ulensing_degenerate_models
          output_root: /export/.../hidden_parallax

    A partir de esos roots se derivan:
        PARALLAX_LSST_BASE = ${ULENSING_DEGENERATE_MODELS_ROOT}/Parallax_LSST
        ROMAN_RUBIN_DIR   = ${MICROLENSING_ROOT}/simulation_Rubin/roman_rubin

    Estos valores se exportan a os.environ antes de resolver otros paths,
    para que el resto del config pueda usar placeholders como:
        ${ULENSING_DEGENERATE_MODELS_ROOT}/Parallax_LSST/data_sedighe/columns
        ${OUTPUT_ROOT}/runs
    """

    paths_cfg = _paths_section()

    base_env = {
        "HOME": HOME,
    }

    microlensing_value = first_config_value(
        paths_cfg.get("microlensing_root", None),
        os.environ.get("MICROLENSING_ROOT", ""),
        default="${HOME}/microlensing",
    )

    microlensing_root = resolve_config_path(
        microlensing_value,
        extra_env=base_env,
    )

    env_after_microlensing = {
        **base_env,
        "MICROLENSING_ROOT": microlensing_root,
    }

    ulensing_value = first_config_value(
        paths_cfg.get("ulensing_degenerate_models_root", None),
        paths_cfg.get("ulensing_root", None),
        os.environ.get("ULENSING_DEGENERATE_MODELS_ROOT", ""),
        default="${HOME}/ulensing_degenerate_models",
    )

    ulensing_degenerate_models_root = resolve_config_path(
        ulensing_value,
        extra_env=env_after_microlensing,
    )

    env_after_ulensing = {
        **env_after_microlensing,
        "ULENSING_DEGENERATE_MODELS_ROOT": ulensing_degenerate_models_root,
    }

    parallax_lsst_value = first_config_value(
        paths_cfg.get("parallax_lsst_root", None),
        paths_cfg.get("project_base", None),
        os.environ.get("PARALLAX_LSST_BASE", ""),
        default=None,
    )

    if parallax_lsst_value in (None, "", "default", "auto"):
        parallax_lsst_base = (
            ulensing_degenerate_models_root
            / "Parallax_LSST"
        ).resolve()
    else:
        parallax_lsst_base = resolve_config_path(
            parallax_lsst_value,
            extra_env=env_after_ulensing,
        )

    env_after_parallax = {
        **env_after_ulensing,
        "PARALLAX_LSST_BASE": parallax_lsst_base,
    }

    roman_rubin_value = first_config_value(
        paths_cfg.get("roman_rubin_dir", None),
        os.environ.get("ROMAN_RUBIN_DIR", ""),
        default=None,
    )

    if roman_rubin_value in (None, "", "default", "auto"):
        roman_rubin_dir = (
            microlensing_root
            / "simulation_Rubin"
            / "roman_rubin"
        ).resolve()
    else:
        roman_rubin_dir = resolve_config_path(
            roman_rubin_value,
            extra_env=env_after_parallax,
        )

    env_after_roman = {
        **env_after_parallax,
        "ROMAN_RUBIN_DIR": roman_rubin_dir,
    }

    output_root_value = first_config_value(
        paths_cfg.get("output_root", None),
        os.environ.get("OUTPUT_ROOT", ""),
        default="${HOME}/hidden_parallax",
    )

    output_root = resolve_config_path(
        output_root_value,
        extra_env=env_after_roman,
    )

    final_env = {
        **env_after_roman,
        "OUTPUT_ROOT": output_root,
    }

    roman_ephemerides_value = first_config_value(
        paths_cfg.get("roman_ephemerides", None),
        os.environ.get("ROMAN_EPHEMERIDES", ""),
        default=None,
    )

    roman_ephemerides = None

    if roman_ephemerides_value not in (None, "", "default", "auto"):
        roman_ephemerides = resolve_config_path(
            roman_ephemerides_value,
            extra_env=final_env,
        )
        final_env["ROMAN_EPHEMERIDES"] = roman_ephemerides

    for key, value in final_env.items():
        os.environ[str(key)] = str(value)

    print(f"[config] MICROLENSING_ROOT              = {microlensing_root}", flush=True)
    print(f"[config] ULENSING_DEGENERATE_MODELS_ROOT = {ulensing_degenerate_models_root}", flush=True)
    print(f"[config] PARALLAX_LSST_BASE             = {parallax_lsst_base}", flush=True)
    print(f"[config] ROMAN_RUBIN_DIR                = {roman_rubin_dir}", flush=True)
    print(f"[config] OUTPUT_ROOT                    = {output_root}", flush=True)

    if roman_ephemerides is not None:
        print(f"[config] ROMAN_EPHEMERIDES             = {roman_ephemerides}", flush=True)

    return {
        "microlensing_root": microlensing_root,
        "ulensing_degenerate_models_root": ulensing_degenerate_models_root,
        "parallax_lsst_base": parallax_lsst_base,
        "roman_rubin_dir": roman_rubin_dir,
        "output_root": output_root,
        "roman_ephemerides": roman_ephemerides,
    }


PATHS_FROM_CONFIG = configure_paths_from_config()

MICROLENSING_ROOT = PATHS_FROM_CONFIG["microlensing_root"]
ULENSING_DEGENERATE_MODELS_ROOT = PATHS_FROM_CONFIG[
    "ulensing_degenerate_models_root"
]
OUTPUT_ROOT_FROM_CONFIG = PATHS_FROM_CONFIG["output_root"]


def configure_environment_from_config():
    """
    Exporta variables de entorno adicionales desde el config.

    Los roots principales se resuelven antes en configure_paths_from_config().
    Los paths de Rubin se terminan de resolver en configure_rubin_paths(),
    porque pueden depender unos de otros, por ejemplo:
        rubin.opsim_db_path = ${RUBIN_SIM_DATA_DIR}/...
    """

    mapping = [
        ("paths", "project_base", "PARALLAX_LSST_BASE"),
        ("paths", "parallax_lsst_root", "PARALLAX_LSST_BASE"),
        ("paths", "roman_rubin_dir", "ROMAN_RUBIN_DIR"),
        ("paths", "roman_ephemerides", "ROMAN_EPHEMERIDES"),
        # Alias backward-compatible: si el config viejo lo tiene en paths,
        # lo copiamos a RUBIN_SIM_DATA_DIR.
        ("paths", "rubin_sim_data_dir", "RUBIN_SIM_DATA_DIR"),
        # Nuevo esquema recomendado.
        ("rubin", "sim_data_dir", "RUBIN_SIM_DATA_DIR"),
    ]

    extra_env = {
        "HOME": HOME,
        "MICROLENSING_ROOT": MICROLENSING_ROOT,
        "ULENSING_DEGENERATE_MODELS_ROOT": ULENSING_DEGENERATE_MODELS_ROOT,
        "PARALLAX_LSST_BASE": PATHS_FROM_CONFIG["parallax_lsst_base"],
        "ROMAN_RUBIN_DIR": PATHS_FROM_CONFIG["roman_rubin_dir"],
        "OUTPUT_ROOT": OUTPUT_ROOT_FROM_CONFIG,
    }

    for section, key, env_key in mapping:
        value = cfg(section, key)

        if value not in (None, ""):
            os.environ[env_key] = str(
                resolve_config_path(
                    value,
                    extra_env=extra_env,
                )
            )

            if env_key == "RUBIN_SIM_DATA_DIR":
                os.environ["SIMS_DATA_DIR"] = os.environ[env_key]


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


# La OpSim y los throughputs se configuran juntos más abajo, luego de resolver BASE_DIR.

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


def configure_rubin_paths(base_dir=None, validate=True):
    """
    Configura todos los paths de Rubin desde el config.

    Nuevo esquema recomendado en JSON/YAML:

        rubin:
          sim_data_dir: ${HOME}/rubin_sim_data
          opsim_db_path: ${RUBIN_SIM_DATA_DIR}/sim_baseline/baseline_v5.3.5_10yrs.db
          throughputs_dir: ${RUBIN_SIM_DATA_DIR}/throughputs/baseline

    No busca automáticamente bajo /home ni bajo el repo. Para correr en un
    cluster, cambiá solo el config o exportá las variables de entorno.
    """

    sim_data_value = first_config_value(
        cfg("rubin", "sim_data_dir", None),
        cfg("paths", "rubin_sim_data_dir", None),
        os.environ.get("RUBIN_SIM_DATA_DIR", ""),
        os.environ.get("SIMS_DATA_DIR", ""),
        default=None,
    )

    if sim_data_value in (None, "", "default", "auto"):
        raise FileNotFoundError(
            "No está definido rubin.sim_data_dir en el config.\n"
            "Agregá, por ejemplo:\n"
            "  rubin.sim_data_dir: ${HOME}/rubin_sim_data\n"
            "o exportá RUBIN_SIM_DATA_DIR=/ruta/a/rubin_sim_data."
        )

    rubin_sim_data_dir = resolve_config_path(sim_data_value)

    os.environ["RUBIN_SIM_DATA_DIR"] = str(rubin_sim_data_dir)
    os.environ["SIMS_DATA_DIR"] = str(rubin_sim_data_dir)

    extra_env = {
        "RUBIN_SIM_DATA_DIR": rubin_sim_data_dir,
        "SIMS_DATA_DIR": rubin_sim_data_dir,
    }

    throughputs_value = first_config_value(
        cfg("rubin", "throughputs_dir", None),
        cfg("paths", "rubin_throughputs_dir", None),
        os.environ.get("RUBIN_THROUGHPUTS_DIR", ""),
        default=None,
    )

    if throughputs_value in (None, "", "default", "auto"):
        rubin_throughputs_dir = (
            rubin_sim_data_dir
            / "throughputs"
            / "baseline"
        ).resolve()
    else:
        rubin_throughputs_dir = resolve_config_path(
            throughputs_value,
            extra_env=extra_env,
        )

    os.environ["RUBIN_THROUGHPUTS_DIR"] = str(rubin_throughputs_dir)

    extra_env["RUBIN_THROUGHPUTS_DIR"] = rubin_throughputs_dir

    opsim_value = first_config_value(
        cfg("rubin", "opsim_db_path", None),
        cfg("simulation", "opsim_db_path", None),
        cfg("paths", "opsim_db_path", None),
        os.environ.get("RUBIN_OPSIM_DB_PATH", ""),
        os.environ.get("RUBIN_OPSIM_DB", ""),
        default=None,
    )

    if opsim_value in (None, "", "default", "auto"):
        raise FileNotFoundError(
            "No está definido rubin.opsim_db_path en el config.\n"
            "Agregá, por ejemplo:\n"
            "  rubin.opsim_db_path: "
            "${RUBIN_SIM_DATA_DIR}/sim_baseline/baseline_v5.3.5_10yrs.db"
        )

    rubin_opsim_db_path = resolve_config_path(
        opsim_value,
        extra_env=extra_env,
    )

    os.environ["RUBIN_OPSIM_DB_PATH"] = str(rubin_opsim_db_path)
    os.environ["RUBIN_OPSIM_DB"] = str(rubin_opsim_db_path)

    if validate:
        if not rubin_sim_data_dir.exists():
            raise FileNotFoundError(
                f"No existe rubin.sim_data_dir: {rubin_sim_data_dir}"
            )

        if not rubin_throughputs_dir.exists():
            raise FileNotFoundError(
                f"No existe rubin.throughputs_dir: {rubin_throughputs_dir}"
            )

        missing = []
        for band in "ugrizy":
            throughput_file = rubin_throughputs_dir / f"total_{band}.dat"
            throughput_file_gz = rubin_throughputs_dir / f"total_{band}.dat.gz"
            if not throughput_file.exists() and not throughput_file_gz.exists():
                missing.append(str(throughput_file))

        if missing:
            raise FileNotFoundError(
                "Faltan archivos de throughput Rubin:\n"
                + "\n".join(missing)
            )

        if not rubin_opsim_db_path.exists():
            raise FileNotFoundError(
                f"No existe rubin.opsim_db_path: {rubin_opsim_db_path}"
            )

    print(f"[config] RUBIN_SIM_DATA_DIR    = {rubin_sim_data_dir}", flush=True)
    print(f"[config] RUBIN_THROUGHPUTS_DIR = {rubin_throughputs_dir}", flush=True)
    print(f"[config] RUBIN_OPSIM_DB_PATH   = {rubin_opsim_db_path}", flush=True)

    return rubin_sim_data_dir, rubin_throughputs_dir, rubin_opsim_db_path


def configure_set_telescopes_module(
    rubin_sim_data_dir,
    rubin_throughputs_dir,
    rubin_opsim_db_path,
    reset_caches=True,
):
    """
    Configura set_telescopes_pyLIMA si la versión instalada expone
    configure_rubin_paths(). Si no, deja las variables de entorno seteadas.
    """

    os.environ["RUBIN_SIM_DATA_DIR"] = str(rubin_sim_data_dir)
    os.environ["SIMS_DATA_DIR"] = str(rubin_sim_data_dir)
    os.environ["RUBIN_THROUGHPUTS_DIR"] = str(rubin_throughputs_dir)
    os.environ["RUBIN_OPSIM_DB_PATH"] = str(rubin_opsim_db_path)
    os.environ["RUBIN_OPSIM_DB"] = str(rubin_opsim_db_path)

    try:
        import set_telescopes_pyLIMA as stp

        if hasattr(stp, "configure_rubin_paths"):
            stp.configure_rubin_paths(
                rubin_sim_data_dir=rubin_sim_data_dir,
                rubin_throughputs_dir=rubin_throughputs_dir,
                rubin_opsim_db_path=rubin_opsim_db_path,
                reset_caches=reset_caches,
                validate=True,
            )
        else:
            print(
                "[warning] set_telescopes_pyLIMA no expone "
                "configure_rubin_paths(). Se usarán solo variables de entorno. "
                "Actualizá set_telescopes_pyLIMA.py para evitar paths hardcodeados.",
                flush=True,
            )

    except ImportError as error:
        print(
            "[warning] todavía no pude importar set_telescopes_pyLIMA "
            f"para configurarlo: {error!r}",
            flush=True,
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
(
    RUBIN_SIM_DATA_DIR,
    RUBIN_THROUGHPUTS_DIR,
    RUBIN_OPSIM_DB_PATH,
) = configure_rubin_paths(BASE_DIR)
ROMAN_RUBIN_DIR = find_roman_rubin_dir(BASE_DIR)
PATH_EPHEMERIDES = find_ephemerides(ROMAN_RUBIN_DIR)

sys.path.insert(0, str(ROMAN_RUBIN_DIR))

configure_set_telescopes_module(
    rubin_sim_data_dir=RUBIN_SIM_DATA_DIR,
    rubin_throughputs_dir=RUBIN_THROUGHPUTS_DIR,
    rubin_opsim_db_path=RUBIN_OPSIM_DB_PATH,
    reset_caches=True,
)

import functions_roman_rubin as frr  # noqa: E402

# ---------------------------------------------------------------------------
# Fit backend
# ---------------------------------------------------------------------------
# Standard behavior: one simulation + one fit through frr.sim_fit.
# New hypothesis-test behavior: one simulation + multiple fits through
# frr.sim_fit_multi_fits.  The choice is controlled only by the config file.
RUN_MULTIPLE_FITS = bool(cfg("fit", "run_multiple_fits", False))
FIT_SPECS = cfg("fit", "fits", None)
PRIMARY_FIT = cfg("fit", "primary_fit", None)
LRT_CONFIG = cfg("fit", "lrt", None)

if RUN_MULTIPLE_FITS:
    if not hasattr(frr, "sim_fit_multi_fits"):
        raise RuntimeError(
            "El config tiene fit.run_multiple_fits=true, pero "
            "functions_roman_rubin.py no expone sim_fit_multi_fits. "
            "Reemplazá functions_roman_rubin.py por la versión multi-fit."
        )
    sim_fit = frr.sim_fit_multi_fits
else:
    sim_fit = frr.sim_fit

print(f"[config] RUN_MULTIPLE_FITS = {RUN_MULTIPLE_FITS}", flush=True)
if RUN_MULTIPLE_FITS:
    print(f"[config] PRIMARY_FIT = {PRIMARY_FIT}", flush=True)
    print(f"[config] FIT_SPECS keys = {list(FIT_SPECS.keys()) if isinstance(FIT_SPECS, dict) else FIT_SPECS}", flush=True)
    print(f"[config] LRT_CONFIG = {LRT_CONFIG}", flush=True)


# ============================================================================
# General configuration
# ============================================================================

DEFAULT_COLUMNS_FILE = BASE_DIR / "data_sedighe" / "columns"
DEFAULT_DATA_FILE = BASE_DIR / "data_sedighe" / "LSSTMONTS.dat"

_PROJECT_PATH_ENV = {
    "HOME": HOME,
    "MICROLENSING_ROOT": MICROLENSING_ROOT,
    "ULENSING_DEGENERATE_MODELS_ROOT": ULENSING_DEGENERATE_MODELS_ROOT,
    "PARALLAX_LSST_BASE": BASE_DIR,
    "ROMAN_RUBIN_DIR": ROMAN_RUBIN_DIR,
    "OUTPUT_ROOT": OUTPUT_ROOT_FROM_CONFIG,
}

COLUMNS_FILE = resolve_config_path(
    first_config_value(
        cfg("input", "columns_file", None),
        cfg("paths", "columns_file", None),
        default=None,
    ),
    default=DEFAULT_COLUMNS_FILE,
    extra_env=_PROJECT_PATH_ENV,
)

DATA_FILE = resolve_config_path(
    first_config_value(
        cfg("input", "data_file", None),
        cfg("paths", "data_file", None),
        default=None,
    ),
    default=DEFAULT_DATA_FILE,
    extra_env=_PROJECT_PATH_ENV,
)

RUN_NAME_BASE = str(
    first_config_value(
        topcfg("run_name", None),
        cfg("output", "run_name", None),
        default="LSSTMONTS_catalogXi_catalogBlending_FSPLparallax_fitFSPLNoPiE",
    )
)

# Keep one scientific config file.  For cluster jobs, each chunk writes to an
# independent subdirectory under the same RUN_NAME_BASE.
CHUNK_OUTPUT_LABEL = _early_chunk_output_label(EARLY_CLI_ARGS)
RUN_NAME = RUN_NAME_BASE

OUTPUT_ROOT = resolve_config_path(
    first_config_value(
        topcfg("path_storage", None),
        cfg("output", "root_dir", None),
        cfg("paths", "path_storage", None),
        default="${OUTPUT_ROOT}/runs",
    ),
    default=BASE_DIR / "runs",
    extra_env={
        "OUTPUT_ROOT": OUTPUT_ROOT_FROM_CONFIG,
        "MICROLENSING_ROOT": MICROLENSING_ROOT,
        "ULENSING_DEGENERATE_MODELS_ROOT": ULENSING_DEGENERATE_MODELS_ROOT,
        "PARALLAX_LSST_BASE": BASE_DIR,
        "ROMAN_RUBIN_DIR": ROMAN_RUBIN_DIR,
    },
)

if CHUNK_OUTPUT_LABEL:
    RUN_DIR = OUTPUT_ROOT / RUN_NAME_BASE / CHUNK_OUTPUT_LABEL
else:
    RUN_DIR = OUTPUT_ROOT / RUN_NAME_BASE

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

# Legacy value kept only for diagnostics/backward-compatible metadata.
# It is NOT allowed to define the simulated event t0.
# Scientific convention enforced below:
#     t0_jd = first_OpSim_MAF_timestamp_for_this_field + t0_catalog_days
T0_ZERO_JD = float(
    first_config_value(
        cfg("input", "t0_zero_jd", None),
        cfg("catalog", "t0_zero_jd", None),
        default=2460413.013828608,
    )
)

T0_ORIGIN_POLICY = str(
    first_config_value(
        cfg("input", "t0_origin", None),
        cfg("catalog", "t0_origin", None),
        cfg("sedighe", "t0_origin", None),
        default="first_maf_timestamp",
    )
).lower()

if T0_ORIGIN_POLICY not in {"first_maf_timestamp", "first_opsim_timestamp"}:
    raise ValueError(
        "Este runner solo permite interpretar t0_catalog_days desde el "
        "primer timestamp OpSim/MAF del campo. Configure "
        "catalog.t0_origin: first_maf_timestamp."
    )

T0_ORIGIN_POLICY = "first_maf_timestamp"

RANDOM_SEED = int(
    cfg("execution", "random_seed", 20260728)
)

N_WORKERS = int(
    cfg("execution", "workers", 8)
)

_TIMEOUT_RAW = cfg(
    "execution",
    "post_detectability_timeout_s",
    None,
)

if (
    _TIMEOUT_RAW is None
    or str(_TIMEOUT_RAW).strip().lower()
    in {"", "none", "off", "false", "0", "0.0"}
):
    POST_DETECTABILITY_TIMEOUT_S = None
else:
    POST_DETECTABILITY_TIMEOUT_S = float(_TIMEOUT_RAW)

    if POST_DETECTABILITY_TIMEOUT_S <= 0.0:
        raise ValueError(
            "execution.post_detectability_timeout_s "
            "must be positive, null, or off."
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


def parse_optional_nonnegative_int(value, name):
    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in {"all", "none", ""}:
        return None

    parsed = int(value)

    if parsed < 0:
        raise ValueError(f"{name} debe ser >= 0, None o 'all'.")

    return parsed


def resolve_catalog_row_window(start, stop):
    start = parse_optional_nonnegative_int(start, "catalog_row_start")
    stop = parse_optional_nonnegative_int(stop, "catalog_row_stop")

    if start is None:
        start = 0

    if stop is not None and stop <= start:
        raise ValueError(
            "catalog_row_stop debe ser mayor que catalog_row_start. "
            f"Recibido start={start}, stop={stop}."
        )

    return int(start), stop


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


# ============================================================
# Initial guess del optimizador
# ============================================================

FIT_INITIAL_GUESS = cfg(
    "fit",
    "initial_guess",
    None,
)

if isinstance(FIT_INITIAL_GUESS, str):

    _initial_guess_mode = FIT_INITIAL_GUESS.strip().lower()

    if _initial_guess_mode in {
        "",
        "none",
        "null",
        "random",
    }:
        FIT_INITIAL_GUESS = None

    elif _initial_guess_mode in {
        "truth",
        "fit_params",
    }:
        FIT_INITIAL_GUESS = _initial_guess_mode

    else:
        raise ValueError(
            "fit.initial_guess no reconocido: "
            f"{FIT_INITIAL_GUESS!r}"
        )

elif (
    FIT_INITIAL_GUESS is not None
    and not isinstance(
        FIT_INITIAL_GUESS,
        (dict, list, tuple),
    )
):
    raise TypeError(
        "fit.initial_guess debe ser "
        "null, string, dict, list o tuple."
    )

# ============================================================
# Opciones del optimizador TRF
# ============================================================

FIT_OPTIMIZER_OPTIONS = cfg(
    "fit",
    "optimizer_options",
    None,
)

if FIT_OPTIMIZER_OPTIONS is not None:

    if not isinstance(FIT_OPTIMIZER_OPTIONS, dict):
        raise TypeError(
            "fit.optimizer_options debe ser null o dict."
        )

    FIT_OPTIMIZER_OPTIONS = dict(
        FIT_OPTIMIZER_OPTIONS
    )

    allowed_optimizer_keys = {
        "xtol",
        "ftol",
        "gtol",
        "max_nfev",
        "x_scale",
    }

    unknown = (
        set(FIT_OPTIMIZER_OPTIONS)
        - allowed_optimizer_keys
    )

    if unknown:
        raise ValueError(
            "Claves no soportadas en fit.optimizer_options: "
            f"{sorted(unknown)}"
        )

    for key in ("xtol", "ftol", "gtol"):

        if key in FIT_OPTIMIZER_OPTIONS:

            value = float(
                FIT_OPTIMIZER_OPTIONS[key]
            )

            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"fit.optimizer_options.{key} "
                    "debe ser finito y > 0."
                )

            FIT_OPTIMIZER_OPTIONS[key] = value

    if "max_nfev" in FIT_OPTIMIZER_OPTIONS:

        value = int(
            FIT_OPTIMIZER_OPTIONS["max_nfev"]
        )

        if value <= 0:
            raise ValueError(
                "fit.optimizer_options.max_nfev debe ser > 0."
            )

        FIT_OPTIMIZER_OPTIONS[
            "max_nfev"
        ] = value

    if "x_scale" in FIT_OPTIMIZER_OPTIONS:

        value = FIT_OPTIMIZER_OPTIONS["x_scale"]

        if isinstance(value, str):

            if value.lower() != "jac":
                raise ValueError(
                    "fit.optimizer_options.x_scale "
                    "debe ser 'jac' o un escalar positivo."
                )

            FIT_OPTIMIZER_OPTIONS["x_scale"] = "jac"

        else:

            value = float(value)

            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "fit.optimizer_options.x_scale "
                    "debe ser 'jac' o un escalar positivo."
                )

            FIT_OPTIMIZER_OPTIONS["x_scale"] = value


TRUTH_PARALLAX = bool(
    cfg("truth", "parallax", True)
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

APPLY_PHOTOMETRIC_FILTER = bool(
    first_config_value(
        cfg("simulation", "apply_photometric_filter", None),
        cfg("selection", "apply_photometric_filter", None),
        default=True,
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
            "half_width": 60.0,
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

BAND_AVAILABILITY_MODE = str(
    first_config_value(
        cfg("sedighe", "band_availability", None),
        cfg("selection", "band_availability", None),
        cfg("blending", "band_availability", None),
        default="detection_flag",
    )
).lower()

if BAND_AVAILABILITY_MODE not in {
    "detection_flag",
    "blend_positive",
    "all",
}:
    raise ValueError(
        "sedighe.band_availability debe ser 'detection_flag', "
        "'blend_positive' o 'all'. "
        f"Recibido: {BAND_AVAILABILITY_MODE!r}."
    )

if BLENDING_MODE != "catalog_source_fraction":
    raise NotImplementedError(
        "Esta versión requiere blending.mode: catalog_source_fraction."
    )

BLENDING_ASSUMPTION = {
    "mode": BLENDING_MODE,
    "definition": "f_s = F_source / (F_source + F_blend)",
    "source": "catalog columns",
    "band_availability_mode": BAND_AVAILABILITY_MODE,
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

CATALOG_BANDS = ["u", "g", "r", "i", "z", "y"]

DETECTION_FLAG_COLUMNS = [
    f"DetectionFlag_{band}"
    for band in CATALOG_BANDS
]


# ============================================================================
# Worker globals
# ============================================================================

GLOBAL_PREPARED_CATALOG = None
GLOBAL_WORKER_CONFIG = None

# Per-process runtime context. Each ProcessPool worker executes one event at a
# time, so these values are not shared between simultaneous events.
_RUNTIME_BLEND_SOURCE_FRACTION = None
_RUNTIME_AVAILABLE_BANDS = None
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


def _canonical_maf_filter(filter_name):
    """
    Canonicalize OpSim/MAF filter names to u,g,r,i,z,y.

    This also accepts names with suffixes such as r_57.
    """

    lower = str(filter_name).strip().lower()

    if lower in set(CATALOG_BANDS):
        return lower

    for band in CATALOG_BANDS:
        if lower.startswith(band + "_") or lower.endswith("_" + band):
            return band

    if len(lower) > 0 and lower[0] in set(CATALOG_BANDS):
        return lower[0]

    return lower


def _find_dataslice_filter_column(dataSlice):
    """
    Return the filter-column name used by the current MAF/dataSlice.
    """

    names = dataSlice.dtype.names

    if names is None:
        raise RuntimeError(
            "dataSlice no tiene dtype.names; no puedo identificar filtros."
        )

    for candidate in [
        "filter",
        "band",
        "filtername",
        "filterName",
        "filter_name",
    ]:
        if candidate in names:
            return candidate

    raise KeyError(
        "No encontré columna de filtro en dataSlice. "
        f"Columnas disponibles: {names}"
    )


def _visible_bands_from_row(base_row, mode=None):
    """
    Define which catalog bands are allowed to enter the simulated event.

    Preferred mode:
        detection_flag:
            DetectionFlag_band == 1 means the source is visible in that band.

    The blending factors are not used as availability flags in this mode.
    They are only used as source fractions for active bands.
    """

    if mode is None:
        mode = BAND_AVAILABILITY_MODE

    mode = str(mode).lower()

    if mode == "all":
        return list(CATALOG_BANDS)

    if mode == "blend_positive":
        return [
            band for band in CATALOG_BANDS
            if float(base_row.get(f"blend_{band}", 0.0)) > 0.0
        ]

    if mode == "detection_flag":
        missing = [
            f"DetectionFlag_{band}"
            for band in CATALOG_BANDS
            if f"DetectionFlag_{band}" not in base_row.index
        ]

        if missing:
            raise KeyError(
                "band_availability='detection_flag' requiere estas columnas: "
                f"{missing}"
            )

        visible_bands = []

        for band in CATALOG_BANDS:
            flag = float(base_row[f"DetectionFlag_{band}"])

            if not np.isfinite(flag):
                raise ValueError(
                    f"DetectionFlag_{band} no es finito: {flag}"
                )

            if int(round(flag)) == 1:
                visible_bands.append(band)

        return visible_bands

    raise ValueError(
        f"band availability mode desconocido: {mode!r}"
    )


def model_choice_catalog_visible_filters(
    event,
    *args,
    **kwargs,
):
    """
    Keep only the catalog-visible filters before pyLIMA constructs flux bounds.

    The preferred availability criterion is DetectionFlag_band == 1.
    Blending factors are used only as source fractions for active bands.
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

    if _RUNTIME_AVAILABLE_BANDS is None:
        available_bands = list(CATALOG_BANDS)
    else:
        available_bands = list(_RUNTIME_AVAILABLE_BANDS)

    kept_telescopes = []
    removed_telescopes = []

    for telescope in event.telescopes:
        catalog_band = _canonical_catalog_band(
            telescope.name
        )

        if catalog_band is None:
            kept_telescopes.append(telescope)
            continue

        if catalog_band not in available_bands:
            removed_telescopes.append(
                {
                    "telescope": str(telescope.name),
                    "catalog_band": catalog_band,
                    "reason": f"band_availability={BAND_AVAILABILITY_MODE}",
                }
            )
            continue

        source_fraction = float(
            _RUNTIME_BLEND_SOURCE_FRACTION[
                catalog_band
            ]
        )

        if not np.isfinite(source_fraction) or not (0.0 < source_fraction <= 1.0):
            raise ValueError(
                f"Blending/source fraction inválido para banda activa "
                f"{catalog_band}: {source_fraction}. "
                "Para bandas con DetectionFlag=1 se requiere "
                "0 < F_source/F_total <= 1."
            )

        kept_telescopes.append(telescope)

    event.telescopes = kept_telescopes

    if removed_telescopes:
        print(
            "Catalog filters removed by band availability:",
            removed_telescopes,
        )

    if len(event.telescopes) == 0:
        raise FitWindowRejected(
            "No catalog-visible filters remain after applying "
            f"band_availability={BAND_AVAILABILITY_MODE}."
        )

    print(
        "Catalog-visible active telescopes:",
        [
            telescope.name
            for telescope in event.telescopes
        ],
    )
    print(
        "Catalog available bands:",
        available_bands,
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



# ============================================================================
# PREFIT_DETECTABILITY_RUNTIME_PATCH_V1
#
# Noise-independent pre-fit microlensing detectability audit/gate.
#
# Scientific purpose
# ------------------
# Decide whether the Rubin cadence actually contains an appreciable
# microlensing signal BEFORE attempting H0/H1 fits.
#
# Selection is deliberately based on a no-parallax reference FSPL by default,
# so the sample definition does not depend on the parallax signal that will
# subsequently be tested with the LRT.
#
# The criterion uses noiseless mag_model and the expected err_mag at the
# actual simulated Rubin timestamps after the photometric filtering.
# ============================================================================

_ORIGINAL_SIM_EVENT_PREFIT_DETECTABILITY = None
_PREFIT_DETECTABILITY_PATCH_INSTALLED = False
_RUNTIME_LAST_DETECTABILITY = {}

_RUNTIME_TIMEOUT_ARMED = False
_RUNTIME_TIMEOUT_STARTED_WALL = None


class PostDetectabilityTimeout(BaseException):
    """Wall-clock timeout after a positive detectability decision."""

    def __init__(self, timeout_s, elapsed_s):
        self.timeout_s = float(timeout_s)
        self.elapsed_s = float(elapsed_s)

        super().__init__(
            "Post-detectability logical-task timeout: "
            f"limit={self.timeout_s:.3f}s, "
            f"elapsed={self.elapsed_s:.3f}s"
        )


def _post_detectability_timeout_handler(signum, frame):
    if _RUNTIME_TIMEOUT_STARTED_WALL is None:
        elapsed = 0.0
    else:
        elapsed = (
            time.perf_counter()
            - _RUNTIME_TIMEOUT_STARTED_WALL
        )

    raise PostDetectabilityTimeout(
        POST_DETECTABILITY_TIMEOUT_S,
        elapsed,
    )


def _arm_post_detectability_timeout():
    global _RUNTIME_TIMEOUT_ARMED
    global _RUNTIME_TIMEOUT_STARTED_WALL

    if POST_DETECTABILITY_TIMEOUT_S is None:
        return

    if _RUNTIME_TIMEOUT_ARMED:
        return

    if (
        not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    ):
        raise RuntimeError(
            "post-detectability timeout requires "
            "SIGALRM/setitimer on this platform."
        )

    signal.signal(
        signal.SIGALRM,
        _post_detectability_timeout_handler,
    )

    _RUNTIME_TIMEOUT_STARTED_WALL = (
        time.perf_counter()
    )

    _RUNTIME_TIMEOUT_ARMED = True

    signal.setitimer(
        signal.ITIMER_REAL,
        float(POST_DETECTABILITY_TIMEOUT_S),
    )

    print(
        "[fit timeout] armed after detectability PASS: "
        f"{POST_DETECTABILITY_TIMEOUT_S:.3f} s",
        flush=True,
    )


def _cancel_post_detectability_timeout():
    global _RUNTIME_TIMEOUT_ARMED
    global _RUNTIME_TIMEOUT_STARTED_WALL

    if (
        _RUNTIME_TIMEOUT_ARMED
        and hasattr(signal, "setitimer")
    ):
        signal.setitimer(
            signal.ITIMER_REAL,
            0.0,
        )

    _RUNTIME_TIMEOUT_ARMED = False
    _RUNTIME_TIMEOUT_STARTED_WALL = None



def _prefit_detectability_config():
    """
    Resolve selection.prefit_detectability.

    Fallback:
        simulation.prefit_detectability

    Disabled by default, preserving all historical runs.
    """

    raw = cfg(
        "selection",
        "prefit_detectability",
        None,
    )

    if raw is None:
        raw = cfg(
            "simulation",
            "prefit_detectability",
            None,
        )

    if raw is None:
        raw = {}

    if isinstance(raw, bool):
        raw = {
            "enabled": bool(raw),
        }

    if not isinstance(raw, dict):
        raise TypeError(
            "selection.prefit_detectability must be a dict, bool, or null."
        )

    out = {
        "enabled": bool(
            raw.get(
                "enabled",
                False,
            )
        ),

        # If true:
        #   simulate + audit + reject before fit, independently of pass/fail.
        "audit_only": bool(
            raw.get(
                "audit_only",
                False,
            )
        ),

        # Recommended:
        #   no_parallax
        #
        # Alternative diagnostic:
        #   current_truth
        "reference_model": str(
            raw.get(
                "reference_model",
                "no_parallax",
            )
        ).strip().lower(),

        # "rubin", "active", or explicit list of bands.
        "bands": raw.get(
            "bands",
            "rubin",
        ),

        "peak_window_tE": float(
            raw.get(
                "peak_window_tE",
                1.0,
            )
        ),

        "min_total_points": int(
            raw.get(
                "min_total_points",
                10,
            )
        ),

        "min_bands": int(
            raw.get(
                "min_bands",
                3,
            )
        ),

        "min_peak_points": int(
            raw.get(
                "min_peak_points",
                5,
            )
        ),

        "min_left_peak_points": int(
            raw.get(
                "min_left_peak_points",
                1,
            )
        ),

        "min_right_peak_points": int(
            raw.get(
                "min_right_peak_points",
                1,
            )
        ),

        "nsigma": float(
            raw.get(
                "nsigma",
                3.0,
            )
        ),

        "min_nsigma_points": int(
            raw.get(
                "min_nsigma_points",
                6,
            )
        ),

        "min_delta_chi2_per_point": float(
            raw.get(
                "min_delta_chi2_per_point",
                2.0,
            )
        ),

        # None means diagnostic only.
        "max_nearest_peak_distance_tE": raw.get(
            "max_nearest_peak_distance_tE",
            None,
        ),
    }

    if out["reference_model"] not in {
        "no_parallax",
        "current_truth",
    }:
        raise ValueError(
            "prefit_detectability.reference_model must be "
            "'no_parallax' or 'current_truth'."
        )

    if out["peak_window_tE"] <= 0:
        raise ValueError(
            "peak_window_tE must be > 0."
        )

    if out["min_total_points"] < 0:
        raise ValueError(
            "min_total_points must be >= 0."
        )

    if out["min_bands"] < 0:
        raise ValueError(
            "min_bands must be >= 0."
        )

    if out["nsigma"] <= 0:
        raise ValueError(
            "nsigma must be > 0."
        )

    if out["min_delta_chi2_per_point"] < 0:
        raise ValueError(
            "min_delta_chi2_per_point must be >= 0."
        )

    nearest = out[
        "max_nearest_peak_distance_tE"
    ]

    if nearest is not None:
        nearest = float(nearest)

        if nearest <= 0:
            raise ValueError(
                "max_nearest_peak_distance_tE must be > 0 or null."
            )

        out[
            "max_nearest_peak_distance_tE"
        ] = nearest

    return out


def _prefit_plain_array(values):
    if hasattr(values, "value"):
        values = values.value

    return np.asarray(
        values,
        dtype=float,
    )


def _prefit_selected_band(
    band_name,
    bands_config,
):
    band_name = str(
        band_name
    )

    if isinstance(
        bands_config,
        str,
    ):
        mode = bands_config.strip().lower()

        if mode == "rubin":
            return band_name in {
                "u",
                "g",
                "r",
                "i",
                "z",
                "y",
            }

        if mode == "active":
            return True

        return (
            band_name.lower()
            == mode
        )

    if isinstance(
        bands_config,
        (list, tuple, set),
    ):
        allowed = {
            str(x)
            for x in bands_config
        }

        return (
            band_name
            in allowed
        )

    raise TypeError(
        "prefit_detectability.bands must be "
        "'rubin', 'active', a band name, or a list."
    )


def _compute_prefit_detectability_metrics(
    pyLIMA_model,
    event_data,
    detect_cfg,
):
    """
    Compute a noise-independent microlensing-vs-constant diagnostic.

    For each band, fit the best weighted constant magnitude:

        m0 = sum(w_i m_i) / sum(w_i)

    using noiseless model magnitudes.

    Then:

        Delta chi2_const =
            sum_i [(m_model_i - m0_band)/sigma_i]^2

    This is an Asimov expected signal metric evaluated at the
    ACTUAL Rubin timestamps and photometric uncertainties.
    """

    out = {
        "detectability_enabled": True,
        "detectability_audit_only": bool(
            detect_cfg[
                "audit_only"
            ]
        ),
        "detectability_reference_model": str(
            detect_cfg[
                "reference_model"
            ]
        ),
        "detectability_noise_independent": True,

        "detectability_n_obs": 0,
        "detectability_n_bands": 0,

        "detectability_n_peak": 0,
        "detectability_n_left_peak": 0,
        "detectability_n_right_peak": 0,

        "detectability_nearest_dt_over_tE": np.nan,

        "detectability_delta_chi2_const_asimov": 0.0,
        "detectability_delta_chi2_per_point": np.nan,

        "detectability_n_nsigma": 0,
        "detectability_max_snr": 0.0,

        "detectability_pass": False,
        "detectability_reasons": "",
    }

    # Save thresholds explicitly.
    out[
        "detectability_threshold_min_total_points"
    ] = int(
        detect_cfg[
            "min_total_points"
        ]
    )

    out[
        "detectability_threshold_min_bands"
    ] = int(
        detect_cfg[
            "min_bands"
        ]
    )

    out[
        "detectability_threshold_min_peak_points"
    ] = int(
        detect_cfg[
            "min_peak_points"
        ]
    )

    out[
        "detectability_threshold_min_left_peak_points"
    ] = int(
        detect_cfg[
            "min_left_peak_points"
        ]
    )

    out[
        "detectability_threshold_min_right_peak_points"
    ] = int(
        detect_cfg[
            "min_right_peak_points"
        ]
    )

    out[
        "detectability_threshold_nsigma"
    ] = float(
        detect_cfg[
            "nsigma"
        ]
    )

    out[
        "detectability_threshold_min_nsigma_points"
    ] = int(
        detect_cfg[
            "min_nsigma_points"
        ]
    )

    out[
        "detectability_threshold_min_delta_chi2_per_point"
    ] = float(
        detect_cfg[
            "min_delta_chi2_per_point"
        ]
    )

    nearest_limit = detect_cfg[
        "max_nearest_peak_distance_tE"
    ]

    out[
        "detectability_threshold_max_nearest_dt_over_tE"
    ] = (
        np.nan
        if nearest_limit is None
        else float(nearest_limit)
    )

    if pyLIMA_model is None:
        out[
            "detectability_reasons"
        ] = "no_model"

        return out

    if not isinstance(
        event_data,
        dict,
    ):
        out[
            "detectability_reasons"
        ] = "event_data_not_dict"

        return out

    try:
        t0 = float(
            event_data[
                "t0"
            ]
        )

        tE = abs(
            float(
                event_data[
                    "tE"
                ]
            )
        )

    except Exception:
        out[
            "detectability_reasons"
        ] = "invalid_t0_or_tE"

        return out

    if (
        not np.isfinite(t0)
        or not np.isfinite(tE)
        or tE <= 0.0
    ):
        out[
            "detectability_reasons"
        ] = "invalid_t0_or_tE"

        return out

    all_times = []

    total_chi2 = 0.0
    total_nsigma = 0
    total_max_snr = 0.0
    n_bands = 0
    n_obs = 0

    for telescope in pyLIMA_model.event.telescopes:

        band = str(
            telescope.name
        )

        if not _prefit_selected_band(
            band,
            detect_cfg[
                "bands"
            ],
        ):
            continue

        lc = getattr(
            telescope,
            "lightcurve",
            None,
        )

        if lc is None or len(lc) == 0:
            continue

        colnames = list(
            getattr(
                lc,
                "colnames",
                [],
            )
        )

        required = {
            "time",
            "mag_model",
            "err_mag",
        }

        if not required.issubset(
            set(colnames)
        ):
            continue

        times = _prefit_plain_array(
            lc[
                "time"
            ]
        )

        model_mag = _prefit_plain_array(
            lc[
                "mag_model"
            ]
        )

        sigma_mag = _prefit_plain_array(
            lc[
                "err_mag"
            ]
        )

        valid = (
            np.isfinite(times)
            & np.isfinite(model_mag)
            & np.isfinite(sigma_mag)
            & (sigma_mag > 0.0)
        )

        times = times[
            valid
        ]

        model_mag = model_mag[
            valid
        ]

        sigma_mag = sigma_mag[
            valid
        ]

        n = int(
            len(times)
        )

        if n == 0:
            continue

        weights = (
            1.0
            / sigma_mag**2
        )

        sum_w = float(
            np.sum(weights)
        )

        if (
            not np.isfinite(sum_w)
            or sum_w <= 0.0
        ):
            continue

        m0 = float(
            np.sum(
                weights
                * model_mag
            )
            / sum_w
        )

        residual_sigma = (
            model_mag
            - m0
        ) / sigma_mag

        residual_sigma = np.asarray(
            residual_sigma,
            dtype=float,
        )

        chi2_band = float(
            np.sum(
                residual_sigma**2
            )
        )

        abs_snr = np.abs(
            residual_sigma
        )

        n_nsigma_band = int(
            np.sum(
                abs_snr
                >= float(
                    detect_cfg[
                        "nsigma"
                    ]
                )
            )
        )

        max_snr_band = float(
            np.max(
                abs_snr
            )
        )

        n_bands += 1
        n_obs += n

        total_chi2 += (
            chi2_band
        )

        total_nsigma += (
            n_nsigma_band
        )

        total_max_snr = max(
            total_max_snr,
            max_snr_band,
        )

        all_times.append(
            times
        )

        prefix = (
            f"detectability_band_{band}"
        )

        out[
            f"{prefix}_n"
        ] = n

        out[
            f"{prefix}_constant_mag"
        ] = m0

        out[
            f"{prefix}_delta_chi2_const"
        ] = chi2_band

        out[
            f"{prefix}_n_nsigma"
        ] = n_nsigma_band

        out[
            f"{prefix}_max_snr"
        ] = max_snr_band

    out[
        "detectability_n_obs"
    ] = int(
        n_obs
    )

    out[
        "detectability_n_bands"
    ] = int(
        n_bands
    )

    out[
        "detectability_delta_chi2_const_asimov"
    ] = float(
        total_chi2
    )

    out[
        "detectability_n_nsigma"
    ] = int(
        total_nsigma
    )

    out[
        "detectability_max_snr"
    ] = float(
        total_max_snr
    )

    if n_obs > 0:

        delta_per_point = (
            total_chi2
            / float(n_obs)
        )

        out[
            "detectability_delta_chi2_per_point"
        ] = float(
            delta_per_point
        )

    else:

        delta_per_point = np.nan

    if all_times:

        times_all = np.concatenate(
            all_times
        )

        dt = (
            times_all
            - t0
        )

        peak_half_width = (
            float(
                detect_cfg[
                    "peak_window_tE"
                ]
            )
            * tE
        )

        peak = (
            np.abs(dt)
            <= peak_half_width
        )

        left = (
            (dt < 0.0)
            & (
                dt
                >= -peak_half_width
            )
        )

        right = (
            (dt > 0.0)
            & (
                dt
                <= peak_half_width
            )
        )

        n_peak = int(
            np.sum(
                peak
            )
        )

        n_left = int(
            np.sum(
                left
            )
        )

        n_right = int(
            np.sum(
                right
            )
        )

        nearest = float(
            np.min(
                np.abs(
                    dt
                )
            )
            / tE
        )

    else:

        n_peak = 0
        n_left = 0
        n_right = 0
        nearest = np.nan

    out[
        "detectability_n_peak"
    ] = n_peak

    out[
        "detectability_n_left_peak"
    ] = n_left

    out[
        "detectability_n_right_peak"
    ] = n_right

    out[
        "detectability_nearest_dt_over_tE"
    ] = nearest

    reasons = []

    if (
        n_obs
        < detect_cfg[
            "min_total_points"
        ]
    ):
        reasons.append(
            "min_total_points"
        )

    if (
        n_bands
        < detect_cfg[
            "min_bands"
        ]
    ):
        reasons.append(
            "min_bands"
        )

    if (
        n_peak
        < detect_cfg[
            "min_peak_points"
        ]
    ):
        reasons.append(
            "min_peak_points"
        )

    if (
        n_left
        < detect_cfg[
            "min_left_peak_points"
        ]
    ):
        reasons.append(
            "min_left_peak_points"
        )

    if (
        n_right
        < detect_cfg[
            "min_right_peak_points"
        ]
    ):
        reasons.append(
            "min_right_peak_points"
        )

    if (
        total_nsigma
        < detect_cfg[
            "min_nsigma_points"
        ]
    ):
        reasons.append(
            "min_nsigma_points"
        )

    if (
        not np.isfinite(
            delta_per_point
        )
        or (
            delta_per_point
            < detect_cfg[
                "min_delta_chi2_per_point"
            ]
        )
    ):
        reasons.append(
            "min_delta_chi2_per_point"
        )

    if nearest_limit is not None:

        if (
            not np.isfinite(
                nearest
            )
            or (
                nearest
                > float(
                    nearest_limit
                )
            )
        ):
            reasons.append(
                "max_nearest_peak_distance_tE"
            )

    out[
        "detectability_pass"
    ] = (
        len(reasons)
        == 0
    )

    out[
        "detectability_reasons"
    ] = ";".join(
        reasons
    )

    return out


def _prefit_force_failure(
    metrics,
    reason,
):
    reasons = [
        item
        for item in str(
            metrics.get(
                "detectability_reasons",
                "",
            )
        ).split(";")
        if item
    ]

    if reason not in reasons:
        reasons.append(
            str(reason)
        )

    metrics[
        "detectability_pass"
    ] = False

    metrics[
        "detectability_reasons"
    ] = ";".join(
        reasons
    )

    return metrics


def _sim_event_with_prefit_detectability(
    *args,
    **kwargs,
):
    """
    Runtime wrapper around functions_roman_rubin.sim_event.

    When disabled:
        exact historical behavior.

    When enabled:
        1. simulate reference event with old noisy detection disabled;
        2. compute Asimov microlensing detectability;
        3a. audit_only=True:
                return decision=False, so no fitting occurs;
        3b. audit_only=False:
                simulate requested truth event and allow the fit only if
                the prefit detectability criterion passed.
    """

    global _RUNTIME_LAST_DETECTABILITY

    detect_cfg = (
        _prefit_detectability_config()
    )

    _RUNTIME_LAST_DETECTABILITY = {
        "detectability_enabled":
            bool(
                detect_cfg[
                    "enabled"
                ]
            ),
    }

    if not detect_cfg[
        "enabled"
    ]:
        return (
            _ORIGINAL_SIM_EVENT_PREFIT_DETECTABILITY(
                *args,
                **kwargs,
            )
        )

    import inspect as _inspect
    import random as _random

    signature = _inspect.signature(
        _ORIGINAL_SIM_EVENT_PREFIT_DETECTABILITY
    )

    bound = signature.bind_partial(
        *args,
        **kwargs,
    )

    bound.apply_defaults()

    call_args = dict(
        bound.arguments
    )

    event_data = call_args.get(
        "data",
        None,
    )

    # We replace the historical noise-dependent detection criterion.
    call_args[
        "apply_detection_criteria"
    ] = False

    reference_mode = detect_cfg[
        "reference_model"
    ]

    reference_args = dict(
        call_args
    )

    if (
        reference_mode
        == "no_parallax"
    ):
        reference_args[
            "truth_parallax"
        ] = False

    # ------------------------------------------------------------
    # Reference simulation must not perturb the RNG state used by
    # the actual H0/H1 simulation.
    # ------------------------------------------------------------

    np_state = np.random.get_state()

    try:
        py_state = _random.getstate()
    except Exception:
        py_state = None

    try:

        (
            reference_model,
            reference_parameters,
            reference_valid_data,
        ) = (
            _ORIGINAL_SIM_EVENT_PREFIT_DETECTABILITY(
                **reference_args
            )
        )

    finally:

        np.random.set_state(
            np_state
        )

        if py_state is not None:
            try:
                _random.setstate(
                    py_state
                )
            except Exception:
                pass

    metrics = (
        _compute_prefit_detectability_metrics(
            reference_model,
            event_data,
            detect_cfg,
        )
    )

    metrics[
        "detectability_reference_has_required_data"
    ] = bool(
        reference_valid_data
    )

    if not reference_valid_data:
        metrics = _prefit_force_failure(
            metrics,
            "no_required_data",
        )

    _RUNTIME_LAST_DETECTABILITY = dict(
        metrics
    )

    print("=" * 80)
    print("[prefit detectability]")
    print(
        "reference_model =",
        metrics.get(
            "detectability_reference_model"
        ),
    )
    print(
        "Nobs =",
        metrics.get(
            "detectability_n_obs"
        ),
    )
    print(
        "Nbands =",
        metrics.get(
            "detectability_n_bands"
        ),
    )
    print(
        "Npeak =",
        metrics.get(
            "detectability_n_peak"
        ),
    )
    print(
        "Nleft/Nright =",
        metrics.get(
            "detectability_n_left_peak"
        ),
        metrics.get(
            "detectability_n_right_peak"
        ),
    )
    print(
        "nearest |dt|/tE =",
        metrics.get(
            "detectability_nearest_dt_over_tE"
        ),
    )
    print(
        "Delta chi2 const Asimov =",
        metrics.get(
            "detectability_delta_chi2_const_asimov"
        ),
    )
    print(
        "Delta chi2 / Nobs =",
        metrics.get(
            "detectability_delta_chi2_per_point"
        ),
    )
    print(
        "N >= nsigma =",
        metrics.get(
            "detectability_n_nsigma"
        ),
    )
    print(
        "max SNR =",
        metrics.get(
            "detectability_max_snr"
        ),
    )
    print(
        "PASS =",
        metrics.get(
            "detectability_pass"
        ),
    )
    print(
        "reasons =",
        metrics.get(
            "detectability_reasons"
        ),
    )
    print("=" * 80)

    # ------------------------------------------------------------
    # Audit mode: deliberately stop before any fit.
    # ------------------------------------------------------------

    if detect_cfg[
        "audit_only"
    ]:

        return (
            reference_model,
            reference_parameters,
            False,
        )

    if bool(
        metrics.get(
            "detectability_pass",
            False,
        )
    ):
        _arm_post_detectability_timeout()

    # ------------------------------------------------------------
    # Gate mode.
    #
    # If the reference IS the requested truth simulation, reuse it.
    # Otherwise run the original requested H0/H1 truth simulation.
    # ------------------------------------------------------------

    if (
        reference_mode
        == "current_truth"
    ):

        actual_model = (
            reference_model
        )

        actual_parameters = (
            reference_parameters
        )

        actual_valid_data = bool(
            reference_valid_data
        )

    else:

        actual_args = dict(
            call_args
        )

        (
            actual_model,
            actual_parameters,
            actual_valid_data,
        ) = (
            _ORIGINAL_SIM_EVENT_PREFIT_DETECTABILITY(
                **actual_args
            )
        )

    final_decision = (
        bool(
            metrics[
                "detectability_pass"
            ]
        )
        and bool(
            actual_valid_data
        )
    )

    return (
        actual_model,
        actual_parameters,
        final_decision,
    )


def install_prefit_detectability_runtime_patch():
    global _ORIGINAL_SIM_EVENT_PREFIT_DETECTABILITY
    global _PREFIT_DETECTABILITY_PATCH_INSTALLED

    if (
        _PREFIT_DETECTABILITY_PATCH_INSTALLED
    ):
        return

    if not hasattr(
        frr,
        "sim_event",
    ):
        raise RuntimeError(
            "functions_roman_rubin does not expose sim_event."
        )

    _ORIGINAL_SIM_EVENT_PREFIT_DETECTABILITY = (
        frr.sim_event
    )

    frr.sim_event = (
        _sim_event_with_prefit_detectability
    )

    _PREFIT_DETECTABILITY_PATCH_INSTALLED = True

    detect_cfg = (
        _prefit_detectability_config()
    )

    print(
        "[prefit detectability patch] installed:",
        detect_cfg,
    )


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
    global _RUNTIME_AVAILABLE_BANDS
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

    _RUNTIME_AVAILABLE_BANDS = _visible_bands_from_row(
        base_row,
        mode=BAND_AVAILABILITY_MODE,
    )

    _RUNTIME_FIT_WINDOW = build_event_fit_window(base_row)
    _RUNTIME_FIT_MIN_POINTS = FIT_WINDOW_MINIMUM_TOTAL_POINTS
    _RUNTIME_LAST_FIT_COUNTS = {}


def clear_runtime_event_context():
    _cancel_post_detectability_timeout()

    global _RUNTIME_LAST_DETECTABILITY
    global _RUNTIME_BLEND_SOURCE_FRACTION
    global _RUNTIME_AVAILABLE_BANDS
    global _RUNTIME_FIT_WINDOW
    global _RUNTIME_LAST_FIT_COUNTS

    _RUNTIME_BLEND_SOURCE_FRACTION = None
    _RUNTIME_AVAILABLE_BANDS = None
    _RUNTIME_FIT_WINDOW = None
    _RUNTIME_LAST_FIT_COUNTS = {}
    _RUNTIME_LAST_DETECTABILITY = {}


def init_worker(prepared_catalog, worker_config):
    global GLOBAL_PREPARED_CATALOG
    global GLOBAL_WORKER_CONFIG

    GLOBAL_PREPARED_CATALOG = prepared_catalog
    GLOBAL_WORKER_CONFIG = worker_config

    configure_set_telescopes_module(
        rubin_sim_data_dir=worker_config["rubin_sim_data_dir"],
        rubin_throughputs_dir=worker_config["rubin_throughputs_dir"],
        rubin_opsim_db_path=worker_config["rubin_opsim_db_path"],
        reset_caches=False,
    )

    install_runtime_patches()
    install_prefit_detectability_runtime_patch()
    install_noise_realization_runtime_patch()


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


def load_raw_catalog(
    columns_file,
    data_file,
    nrows=None,
    catalog_row_start=0,
    catalog_row_stop=None,
):
    catalog_row_start, catalog_row_stop = resolve_catalog_row_window(
        catalog_row_start,
        catalog_row_stop,
    )

    if catalog_row_stop is not None:
        window_nrows = int(catalog_row_stop - catalog_row_start)
        if nrows is None:
            nrows = window_nrows
        else:
            nrows = min(int(nrows), window_nrows)

    log_step(f"[catalog] columns_file = {columns_file}")
    log_step(f"[catalog] data_file    = {data_file}")
    log_step(
        "[catalog] row window   = "
        f"[{catalog_row_start}, "
        f"{catalog_row_stop if catalog_row_stop is not None else 'EOF'})"
    )
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
        skiprows=catalog_row_start if catalog_row_start > 0 else None,
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


def prepare_catalog(raw_catalog, max_base_events, catalog_row_offset=0):
    data = raw_catalog.copy().reset_index(drop=True)

    catalog_row_offset = int(catalog_row_offset)

    data["catalog_row"] = (
        catalog_row_offset
        + np.arange(
            len(data),
            dtype=int,
        )
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

    # IMPORTANT:
    # t0_catalog_days is a relative time from Sedighe's catalog.
    # It must NOT be converted with a fixed global JD origin.
    # The absolute t0 is resolved per event after reading the relevant
    # OpSim/MAF dataSlice:
    #
    #     t0_jd = first_maf_timestamp_jd + t0_catalog_days
    #
    # Keep the old fixed-origin value only as a diagnostic.
    data["t0_jd_original_global_zero"] = (
        T0_ZERO_JD
        + data["t0_catalog_days"]
    )
    data["t0_reference_jd"] = np.nan
    data["t0_reference_mjd"] = np.nan
    data["t0_origin"] = "unresolved_requires_first_maf_timestamp"
    data["t0_jd"] = np.nan

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
        "t0_catalog_days",
        "piE",
        "rho",
        "alpha_rad",
        *SOURCE_MAG_COLUMNS,
        *BLEND_COLUMNS,
    ]

    valid = np.ones(len(data), dtype=bool)
    data["invalid_reason"] = ""

    def mark_invalid(mask, reason):
        """Attach a row-level invalid reason without aborting the full run."""

        mask = np.asarray(mask, dtype=bool)

        if len(mask) != len(data):
            raise ValueError(
                f"Máscara inválida para {reason}: "
                f"len(mask)={len(mask)}, len(data)={len(data)}"
            )

        if not np.any(mask):
            return

        previous = data.loc[mask, "invalid_reason"].astype(str)
        empty = previous.isin(["", "nan", "None"])

        data.loc[mask, "invalid_reason"] = np.where(
            empty,
            reason,
            previous + ";" + reason,
        )

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
    valid &= data["t0_catalog_days"].to_numpy(dtype=float) >= 0.0
    valid &= data["piE"].to_numpy(dtype=float) >= 0.0
    valid &= data["rho"].to_numpy(dtype=float) > 0.0

    mark_invalid(
        ~valid,
        "invalid_basic_finite_or_physical_quantity",
    )

    blend_matrix = data[
        BLEND_COLUMNS
    ].to_numpy(dtype=float)

    blend_range_valid = np.all(
        np.isfinite(blend_matrix)
        & (blend_matrix >= 0.0)
        & (blend_matrix <= 1.0),
        axis=1,
    )

    if BAND_AVAILABILITY_MODE == "detection_flag":
        missing_flags = [
            column for column in DETECTION_FLAG_COLUMNS
            if column not in data.columns
        ]

        if missing_flags:
            raise KeyError(
                "sedighe.band_availability='detection_flag' requiere "
                f"estas columnas: {missing_flags}"
            )

        flag_matrix = data[
            DETECTION_FLAG_COLUMNS
        ].to_numpy(dtype=float)

        flag_finite = np.all(
            np.isfinite(flag_matrix),
            axis=1,
        )

        flag_binary = np.all(
            np.isin(
                np.rint(flag_matrix).astype(int),
                [0, 1],
            ),
            axis=1,
        )

        flag_valid = flag_finite & flag_binary
        availability_matrix = (
            np.rint(flag_matrix).astype(int) == 1
        )

    elif BAND_AVAILABILITY_MODE == "blend_positive":
        flag_valid = np.ones(len(data), dtype=bool)
        availability_matrix = blend_matrix > 0.0

    elif BAND_AVAILABILITY_MODE == "all":
        flag_valid = np.ones(len(data), dtype=bool)
        availability_matrix = np.ones_like(
            blend_matrix,
            dtype=bool,
        )

    else:
        raise ValueError(
            f"band_availability desconocido: {BAND_AVAILABILITY_MODE!r}"
        )

    visible_filter_count = np.sum(
        availability_matrix,
        axis=1,
    )

    data["catalog_visible_filter_count"] = (
        visible_filter_count.astype(int)
    )

    data["catalog_available_bands"] = [
        ",".join(
            band
            for band, is_available in zip(CATALOG_BANDS, row)
            if is_available
        )
        for row in availability_matrix
    ]

    for k, band in enumerate(CATALOG_BANDS):
        data[f"catalog_band_available_{band}"] = (
            availability_matrix[:, k].astype(int)
        )

    enough_visible_filters = (
        visible_filter_count
        >= BLENDING_MINIMUM_VISIBLE_FILTERS
    )

    # Only active/available bands need strictly positive source fraction.
    # Inactive bands may carry a catalog value, but it is not used.
    source_fraction_positive_in_available = np.ones(
        len(data),
        dtype=bool,
    )

    for k, band in enumerate(CATALOG_BANDS):
        source_fraction_positive_in_available &= (
            (~availability_matrix[:, k])
            | (blend_matrix[:, k] > 0.0)
        )

    mark_invalid(~blend_range_valid, "invalid_blend_source_fraction_range")
    mark_invalid(~flag_valid, "invalid_detection_flag")
    mark_invalid(~enough_visible_filters, "too_few_visible_filters")
    mark_invalid(
        ~source_fraction_positive_in_available,
        "visible_band_with_nonpositive_source_fraction",
    )

    blend_valid = (
        blend_range_valid
        & flag_valid
        & enough_visible_filters
        & source_fraction_positive_in_available
    )

    if BLENDING_STRICT and np.any(~blend_range_valid):
        bad = data.loc[
            ~blend_range_valid,
            ["catalog_row", "invalid_reason", *BLEND_COLUMNS],
        ].head(20)

        print(
            "[warning] Hay filas con factores de blending/source fraction "
            "fuera de rango o no finitos. Estas filas se marcarán como "
            "inválidas y no se simularán. Primeros casos:"
        )
        print(bad.to_string(index=False))

    if BLENDING_STRICT and np.any(~flag_valid):
        bad = data.loc[
            ~flag_valid,
            [
                "catalog_row",
                "invalid_reason",
                *[
                    column for column in DETECTION_FLAG_COLUMNS
                    if column in data.columns
                ],
            ],
        ].head(20)

        print(
            "[warning] Hay filas con DetectionFlag_* no finitas o no "
            "binarias. Estas filas se marcarán como inválidas y no se "
            "simularán. Primeros casos:"
        )
        print(bad.to_string(index=False))

    if BLENDING_STRICT and np.any(~enough_visible_filters):
        bad = data.loc[
            ~enough_visible_filters,
            [
                "catalog_row",
                "invalid_reason",
                "catalog_visible_filter_count",
                "catalog_available_bands",
                *[
                    column for column in DETECTION_FLAG_COLUMNS
                    if column in data.columns
                ],
                *BLEND_COLUMNS,
            ],
        ].head(20)

        print(
            "[warning] Hay eventos con menos filtros visibles que el "
            f"mínimo configurado ({BLENDING_MINIMUM_VISIBLE_FILTERS}) "
            f"usando band_availability={BAND_AVAILABILITY_MODE}. "
            "Estas filas se marcarán como inválidas y no se simularán. "
            "Primeros casos:"
        )
        print(bad.to_string(index=False))

    if BLENDING_STRICT and np.any(~source_fraction_positive_in_available):
        bad = data.loc[
            ~source_fraction_positive_in_available,
            [
                "catalog_row",
                "invalid_reason",
                "catalog_available_bands",
                *BLEND_COLUMNS,
            ],
        ].head(20)

        print(
            "[warning] Hay bandas catalogadas como visibles pero con "
            "source fraction <= 0. Para bandas activas se requiere "
            "0 < F_source/F_total <= 1. Estas filas se marcarán como "
            "inválidas y no se simularán. Primeros casos:"
        )
        print(bad.to_string(index=False))

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


def _build_tasks_single_realization_legacy(prepared_catalog):
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
            # Use the absolute catalog row as global_i so merged chunks have
            # unique event identifiers. prepared_index remains the local row
            # inside this chunk's prepared catalog.
            "global_i": int(row["catalog_row"]),
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
# NOISE_REALIZATION_RUNNER_PATCH_V2_RUNTIME
#
# Hidden-Parallax-only Monte Carlo over photometric noise realizations.
#
# The shared functions_roman_rubin.py source file is NOT modified.
#
# Scientific design
# -----------------
# For each physical catalog event:
#
#   - the original simulation_seed is kept fixed;
#   - geometry/cadence/source/blending/etc. remain fixed;
#   - only the photometric noise seed changes;
#   - every noisy dataset is fit by H0 and H1 through the existing
#     sim_fit_multi_fits machinery;
#   - H0-truth uses piEN=piEE=0;
#   - H1-truth uses the catalog piEN/piEE;
#   - truth_parallax remains enabled in BOTH cases so that the truth-model
#     code path is identical and the only physical difference is pi_E.
#
# paired_noise=True uses common random numbers:
# H0 and H1 with the same realization_id receive the same standard-normal
# random stream.
# ============================================================================


_RUNTIME_PHOTOMETRY_NOISE_SEED = None

_ORIGINAL_APPLY_ROMAN_RUBIN_PHOTOMETRY = None

_NOISE_RUNTIME_PATCH_INSTALLED = False



# ============================================================================
# H1_EMBEDDED_H0_INITIALIZATION_PATCH_V1
#
# Hidden-Parallax-only, opt-in initialization policy for calibrated LRT tests.
#
# If CONFIG["fit"]["h1_initialization"] == "embedded_H0", the alternative
# H1 fit is initialized from the best-fit H0 physical parameters with
# piEN=piEE=0.
#
# This makes the H1 initialization a function of the observed dataset rather
# than of the truth case used by the Monte-Carlo generator.
#
# Shared functions_roman_rubin.py and fit_lc.py are NOT modified.
# ============================================================================


_ORIGINAL_RUN_MULTIPLE_NAMED_FITS_EMBEDDED_H0 = None
_H1_EMBEDDED_H0_PATCH_INSTALLED = False


def _h1_initialization_mode():
    fit_cfg = CONFIG.get("fit", {})

    if not isinstance(fit_cfg, dict):
        return ""

    return str(
        fit_cfg.get(
            "h1_initialization",
            "",
        )
    ).strip().lower()


def install_h1_embedded_h0_runtime_patch():
    global _ORIGINAL_RUN_MULTIPLE_NAMED_FITS_EMBEDDED_H0
    global _H1_EMBEDDED_H0_PATCH_INSTALLED

    mode = _h1_initialization_mode()

    if mode in {"", "none", "legacy", "truth"}:
        return

    if mode != "embedded_h0":
        raise ValueError(
            "fit.h1_initialization no reconocido: "
            f"{mode!r}. "
            "Opciones: legacy/none o embedded_H0."
        )

    if _H1_EMBEDDED_H0_PATCH_INSTALLED:
        return

    if not hasattr(
        frr,
        "run_multiple_named_fits",
    ):
        raise RuntimeError(
            "functions_roman_rubin no expone "
            "run_multiple_named_fits."
        )

    original = frr.run_multiple_named_fits

    def run_multiple_named_fits_embedded_h0(
        *args,
        **kwargs,
    ):
        fit_specs = kwargs.get(
            "fit_specs",
            None,
        )

        existing_results = kwargs.get(
            "existing_results",
            None,
        )

        # Keep generic/legacy calls untouched.
        if (
            not isinstance(fit_specs, dict)
            or not isinstance(existing_results, dict)
        ):
            return original(
                *args,
                **kwargs,
            )

        lrt_cfg = (
            LRT_CONFIG
            if isinstance(LRT_CONFIG, dict)
            else {}
        )

        null_key = str(
            lrt_cfg.get(
                "null",
                "H0",
            )
        )

        alternative_key = str(
            lrt_cfg.get(
                "alternative",
                "H1",
            )
        )

        if alternative_key not in fit_specs:
            return original(
                *args,
                **kwargs,
            )

        if null_key not in existing_results:
            raise RuntimeError(
                "embedded_H0 requested, but the null fit "
                f"{null_key!r} is not available in existing_results. "
                "The null model must be the primary fit."
            )

        h0_entry = existing_results[
            null_key
        ]

        if not isinstance(h0_entry, dict):
            raise RuntimeError(
                "embedded_H0: invalid H0 result entry."
            )

        if str(
            h0_entry.get(
                "status",
                "",
            )
        ) != "fitted":
            raise RuntimeError(
                "embedded_H0: H0 is not fitted successfully."
            )

        fit_rr = h0_entry.get(
            "fit_rr",
            None,
        )

        best_model = h0_entry.get(
            "best_model",
            None,
        )

        if fit_rr is None or best_model is None:
            raise RuntimeError(
                "embedded_H0: missing H0 fit_rr or best_model."
            )

        fit_results = getattr(
            fit_rr,
            "fit_results",
            None,
        )

        if not isinstance(fit_results, dict):
            raise RuntimeError(
                "embedded_H0: H0 fit_results is unavailable."
            )

        h0_order = fit_results.get(
            "initial_guess_parameter_order",
            None,
        )

        if not isinstance(
            h0_order,
            (list, tuple),
        ):
            raise RuntimeError(
                "embedded_H0: H0 physical parameter order "
                "is unavailable."
            )

        best = np.asarray(
            best_model,
            dtype=float,
        ).reshape(-1)

        if len(best) < len(h0_order):
            raise RuntimeError(
                "embedded_H0: H0 best_model is shorter "
                "than its physical parameter order."
            )

        h1_guess = {
            str(name): float(best[k])
            for k, name in enumerate(h0_order)
        }

        # Embed the null solution in the parallax model.
        h1_guess["piEN"] = 0.0
        h1_guess["piEE"] = 0.0

        fit_specs_use = fit_specs.copy()

        alternative_spec = dict(
            fit_specs_use[
                alternative_key
            ]
        )

        if not bool(
            alternative_spec.get(
                "parallax",
                False,
            )
        ):
            raise RuntimeError(
                "embedded_H0 requested but the alternative "
                f"{alternative_key!r} is not a parallax fit."
            )

        alternative_spec[
            "initial_guess"
        ] = h1_guess

        fit_specs_use[
            alternative_key
        ] = alternative_spec

        kwargs_use = dict(
            kwargs
        )

        kwargs_use[
            "fit_specs"
        ] = fit_specs_use

        print(
            "[embedded_H0] "
            f"{alternative_key} initial guess from {null_key}: "
            f"{h1_guess}",
            flush=True,
        )

        return original(
            *args,
            **kwargs_use,
        )

    _ORIGINAL_RUN_MULTIPLE_NAMED_FITS_EMBEDDED_H0 = (
        original
    )

    frr.run_multiple_named_fits = (
        run_multiple_named_fits_embedded_h0
    )

    _H1_EMBEDDED_H0_PATCH_INSTALLED = True

    print(
        "[embedded_H0] runtime patch installed",
        flush=True,
    )


# Install at module import time. With fork workers the patched module state is
# inherited; with spawn/import workers the same configuration installs it again.
install_h1_embedded_h0_runtime_patch()


# ============================================================================
# H1_FULL_EMBEDDED_H0_FLUX_PATCH_V1
#
# Optional extension of embedded_H0 initialization.
#
# pyLIMA normally recomputes telescope flux guesses from the physical model.
# For an exact nested H0 -> H1 initialization, we instead reuse the flux
# parameters from the fitted H0 solution.
#
# Enabled only with:
#
#     fit.h1_embed_fluxes_from_H0 = true
#
# Shared microlensing code and installed pyLIMA files are NOT modified.
# ============================================================================


_H1_EMBEDDED_H0_FLUX_CONTEXT = None

_ORIGINAL_TRF_INITIAL_GUESS_EMBEDDED_H0 = None
_ORIGINAL_RUN_MULTIPLE_NAMED_FITS_FULL_EMBEDDED_H0 = None

_H1_FULL_EMBEDDED_H0_FLUX_PATCH_INSTALLED = False


def _h1_embed_fluxes_from_h0_enabled():

    fit_cfg = CONFIG.get("fit", {})

    if not isinstance(fit_cfg, dict):
        return False

    return bool(
        fit_cfg.get(
            "h1_embed_fluxes_from_H0",
            False,
        )
    )


def install_h1_full_embedded_h0_flux_runtime_patch():

    global _H1_EMBEDDED_H0_FLUX_CONTEXT
    global _ORIGINAL_TRF_INITIAL_GUESS_EMBEDDED_H0
    global _ORIGINAL_RUN_MULTIPLE_NAMED_FITS_FULL_EMBEDDED_H0
    global _H1_FULL_EMBEDDED_H0_FLUX_PATCH_INSTALLED

    if not _h1_embed_fluxes_from_h0_enabled():
        return

    if _h1_initialization_mode() != "embedded_h0":
        raise ValueError(
            "fit.h1_embed_fluxes_from_H0=true requires "
            "fit.h1_initialization='embedded_H0'."
        )

    if _H1_FULL_EMBEDDED_H0_FLUX_PATCH_INSTALLED:
        return

    from pyLIMA.fits import TRF_fit

    # ------------------------------------------------------------------
    # Patch TRF.initial_guess so that, when the runtime H0-flux context
    # is active, pyLIMA does NOT recompute telescope flux guesses.
    # ------------------------------------------------------------------

    original_initial_guess = TRF_fit.TRFfit.initial_guess

    def initial_guess_with_embedded_h0_fluxes(self):

        global _H1_EMBEDDED_H0_FLUX_CONTEXT

        context = _H1_EMBEDDED_H0_FLUX_CONTEXT

        if context is not None:

            fluxes = [
                float(x)
                for x in context
            ]

            n_model = len(
                self.model_parameters_guess
            )

            fit_keys = list(
                self.fit_parameters.keys()
            )

            trailing_keys = fit_keys[
                n_model:
            ]

            if len(fluxes) != len(trailing_keys):
                raise RuntimeError(
                    "embedded_H0_full: H0 flux-vector length "
                    "does not match H1 telescope-flux parameter count: "
                    f"{len(fluxes)} != {len(trailing_keys)}; "
                    f"trailing H1 parameters={trailing_keys}"
                )

            allowed_flux_tokens = (
                "fsource_",
                "ftotal_",
                "fblend_",
                "gblend_",
            )

            non_flux = [
                key
                for key in trailing_keys
                if not key.startswith(
                    allowed_flux_tokens
                )
            ]

            if non_flux:
                raise RuntimeError(
                    "embedded_H0_full expected only telescope flux "
                    "parameters after physical parameters, but found: "
                    f"{non_flux}"
                )

            # Validate that the H0 fitted fluxes lie inside the H1 bounds.
            for key, value in zip(
                trailing_keys,
                fluxes,
            ):

                lo, hi = self.fit_parameters[
                    key
                ][1]

                lo = float(lo)
                hi = float(hi)

                # Tiny numerical deviations at an active bound are clipped
                # only at machine-precision scale.
                scale = max(
                    1.0,
                    abs(lo),
                    abs(hi),
                    abs(value),
                )

                atol = 1e-10 * scale

                if value < lo - atol or value > hi + atol:
                    raise RuntimeError(
                        "embedded_H0_full: fitted H0 flux is outside "
                        f"H1 bounds for {key}: "
                        f"value={value}, bounds=[{lo}, {hi}]"
                    )

                if value < lo:
                    value = lo

                if value > hi:
                    value = hi

            self.telescopes_fluxes_parameters_guess = fluxes

            print(
                "[embedded_H0_full] using H0 fitted telescope fluxes "
                f"order={trailing_keys}, values={fluxes}",
                flush=True,
            )

        full_guess = original_initial_guess(
            self
        )

        if context is not None:

            print(
                "[embedded_H0_full] exact full TRF initial guess "
                f"order={list(self.fit_parameters.keys())}, "
                f"values={full_guess}",
                flush=True,
            )

        return full_guess

    TRF_fit.TRFfit.initial_guess = (
        initial_guess_with_embedded_h0_fluxes
    )

    _ORIGINAL_TRF_INITIAL_GUESS_EMBEDDED_H0 = (
        original_initial_guess
    )

    # ------------------------------------------------------------------
    # The embedded-H0 physical wrapper is already installed.
    # Wrap it once more so that the H0 fitted fluxes are exposed only
    # while the alternative H1 fit is being executed.
    # ------------------------------------------------------------------

    original_multi = frr.run_multiple_named_fits

    def run_multiple_named_fits_full_embedded_h0(
        *args,
        **kwargs,
    ):

        global _H1_EMBEDDED_H0_FLUX_CONTEXT

        fit_specs = kwargs.get(
            "fit_specs",
            None,
        )

        existing_results = kwargs.get(
            "existing_results",
            None,
        )

        if (
            not isinstance(fit_specs, dict)
            or not isinstance(existing_results, dict)
        ):
            return original_multi(
                *args,
                **kwargs,
            )

        lrt_cfg = (
            LRT_CONFIG
            if isinstance(LRT_CONFIG, dict)
            else {}
        )

        null_key = str(
            lrt_cfg.get(
                "null",
                "H0",
            )
        )

        alternative_key = str(
            lrt_cfg.get(
                "alternative",
                "H1",
            )
        )

        if (
            null_key not in existing_results
            or alternative_key not in fit_specs
        ):
            return original_multi(
                *args,
                **kwargs,
            )

        # The current calibrated-LRT implementation expects H0 to be
        # already fitted and only H1 to remain.
        missing_fit_keys = [
            key
            for key in fit_specs
            if key not in existing_results
        ]

        if missing_fit_keys != [
            alternative_key
        ]:
            raise RuntimeError(
                "embedded_H0_full currently requires that the only "
                "remaining fit is the LRT alternative. "
                f"remaining={missing_fit_keys}"
            )

        h0_entry = existing_results[
            null_key
        ]

        fit_rr = h0_entry.get(
            "fit_rr",
            None,
        )

        best_model = h0_entry.get(
            "best_model",
            None,
        )

        if fit_rr is None or best_model is None:
            raise RuntimeError(
                "embedded_H0_full: H0 fit result unavailable."
            )

        fit_results = getattr(
            fit_rr,
            "fit_results",
            None,
        )

        if not isinstance(
            fit_results,
            dict,
        ):
            raise RuntimeError(
                "embedded_H0_full: H0 fit_results unavailable."
            )

        h0_order = fit_results.get(
            "initial_guess_parameter_order",
            None,
        )

        if not isinstance(
            h0_order,
            (list, tuple),
        ):
            raise RuntimeError(
                "embedded_H0_full: H0 physical parameter order "
                "unavailable."
            )

        best = np.asarray(
            best_model,
            dtype=float,
        ).reshape(-1)

        n_phys_h0 = len(
            h0_order
        )

        if len(best) <= n_phys_h0:
            raise RuntimeError(
                "embedded_H0_full: H0 best_model has no "
                "telescope-flux parameters."
            )

        fluxes_h0 = [
            float(x)
            for x in best[
                n_phys_h0:
            ]
        ]

        if not np.all(
            np.isfinite(fluxes_h0)
        ):
            raise RuntimeError(
                "embedded_H0_full: non-finite H0 fitted fluxes."
            )

        print(
            "[embedded_H0_full] H0 fitted flux vector "
            f"values={fluxes_h0}",
            flush=True,
        )

        if _H1_EMBEDDED_H0_FLUX_CONTEXT is not None:
            raise RuntimeError(
                "embedded_H0_full: stale runtime flux context."
            )

        _H1_EMBEDDED_H0_FLUX_CONTEXT = (
            fluxes_h0
        )

        try:

            return original_multi(
                *args,
                **kwargs,
            )

        finally:

            _H1_EMBEDDED_H0_FLUX_CONTEXT = None

    frr.run_multiple_named_fits = (
        run_multiple_named_fits_full_embedded_h0
    )

    _ORIGINAL_RUN_MULTIPLE_NAMED_FITS_FULL_EMBEDDED_H0 = (
        original_multi
    )

    _H1_FULL_EMBEDDED_H0_FLUX_PATCH_INSTALLED = True

    print(
        "[embedded_H0_full] runtime flux patch installed",
        flush=True,
    )


install_h1_full_embedded_h0_flux_runtime_patch()


# ============================================================================
# H1_PARALLAX_GRID_MULTISTART_PATCH_V1
#
# Hidden-Parallax-only, opt-in deterministic multistart for the H1 parallax fit.
#
# Design:
#
#   - H0 is fitted once as usual.
#   - All H1 starts use the H0 best-fit t0,u0,tE,rho.
#   - A deterministic grid is explored in (piEN, piEE).
#   - Grid points use pyLIMA's automatic telescope-flux initialization.
#   - Optionally, an additional (0,0) candidate uses the EXACT fitted H0
#     telescope fluxes ("center_full").
#   - The H1 entry exposed to the normal LRT machinery is the candidate with
#     minimum chi2.
#
# The same procedure is applied under H0-truth and H1-truth.
#
# Shared functions_roman_rubin.py, fit_lc.py and installed pyLIMA files are
# NOT modified.
# ============================================================================


_H1_PARALLAX_GRID_MULTISTART_PATCH_INSTALLED = False
_ORIGINAL_RUN_MULTIPLE_NAMED_FITS_H1_MULTISTART = None


def _h1_multistart_config():

    fit_cfg = CONFIG.get("fit", {})

    if not isinstance(fit_cfg, dict):
        return {}

    cfg = fit_cfg.get(
        "h1_multistart",
        {},
    )

    if not isinstance(cfg, dict):
        return {}

    return cfg


def _h1_multistart_enabled():

    return bool(
        _h1_multistart_config().get(
            "enabled",
            False,
        )
    )


def _h1_multistart_bound_half_width(
    bound_spec,
    parameter_name,
):

    if isinstance(
        bound_spec,
        (list, tuple),
    ):
        if len(bound_spec) != 2:
            raise ValueError(
                f"{parameter_name} multistart bound must have length 2."
            )

        lo = float(bound_spec[0])
        hi = float(bound_spec[1])

    elif isinstance(
        bound_spec,
        dict,
    ):

        if "half_width" in bound_spec:

            center = float(
                bound_spec.get(
                    "center",
                    0.0,
                )
            )

            half_width = float(
                bound_spec["half_width"]
            )

            if abs(center) > 1.0e-12:
                raise ValueError(
                    f"{parameter_name} multistart requires bounds centered "
                    f"on zero; center={center}."
                )

            if not np.isfinite(
                half_width
            ) or half_width <= 0:
                raise ValueError(
                    f"Invalid half_width for {parameter_name}: "
                    f"{half_width}"
                )

            return half_width

        if "bounds" in bound_spec:
            bounds = bound_spec["bounds"]
            if len(bounds) != 2:
                raise ValueError(
                    f"Invalid bounds for {parameter_name}: {bounds}"
                )
            lo = float(bounds[0])
            hi = float(bounds[1])

        elif (
            "lower" in bound_spec
            and "upper" in bound_spec
        ):
            lo = float(bound_spec["lower"])
            hi = float(bound_spec["upper"])

        else:
            raise ValueError(
                f"Cannot infer symmetric multistart bounds for "
                f"{parameter_name}: {bound_spec}"
            )

    else:
        raise TypeError(
            f"Unsupported bound specification for {parameter_name}: "
            f"{bound_spec!r}"
        )

    if (
        not np.isfinite(lo)
        or not np.isfinite(hi)
        or lo >= hi
    ):
        raise ValueError(
            f"Invalid bounds for {parameter_name}: [{lo}, {hi}]"
        )

    if not (
        lo < 0.0 < hi
    ):
        raise ValueError(
            f"{parameter_name} multistart bounds must contain zero: "
            f"[{lo}, {hi}]"
        )

    scale = max(
        1.0,
        abs(lo),
        abs(hi),
    )

    if abs(
        abs(lo) - abs(hi)
    ) > 1.0e-8 * scale:
        raise ValueError(
            f"{parameter_name} multistart requires symmetric zero-centered "
            f"bounds; got [{lo}, {hi}]"
        )

    return 0.5 * (
        hi - lo
    )


def _h1_multistart_float_or_none(value):

    try:
        value = float(value)
    except Exception:
        return None

    if not np.isfinite(value):
        return None

    return value


def _h1_multistart_entry_chi2(entry):

    if not isinstance(
        entry,
        dict,
    ):
        return np.nan

    fit_rr = entry.get(
        "fit_rr",
        None,
    )

    fit_results = getattr(
        fit_rr,
        "fit_results",
        None,
    )

    if isinstance(
        fit_results,
        dict,
    ):

        for key in (
            "chi2",
            "chi2_photometry",
        ):
            value = _h1_multistart_float_or_none(
                fit_results.get(
                    key,
                    None,
                )
            )

            if value is not None:
                return value

    likelihood_stats = entry.get(
        "likelihood_stats",
        None,
    )

    if isinstance(
        likelihood_stats,
        dict,
    ):
        value = _h1_multistart_float_or_none(
            likelihood_stats.get(
                "chi2",
                None,
            )
        )

        if value is not None:
            return value

    return np.nan


def _h1_multistart_optimizer_record(
    entry,
):

    record = {}

    fit_rr = None

    if isinstance(
        entry,
        dict,
    ):
        fit_rr = entry.get(
            "fit_rr",
            None,
        )

    fit_results = getattr(
        fit_rr,
        "fit_results",
        None,
    )

    if not isinstance(
        fit_results,
        dict,
    ):
        return record

    for key in (
        "optimizer_status",
        "optimizer_success",
        "optimizer_message",
        "optimizer_nfev",
        "optimizer_njev",
        "optimizer_optimality",
        "optimizer_n_active_bounds",
        "optimizer_active_mask",
    ):

        value = fit_results.get(
            key,
            None,
        )

        if isinstance(
            value,
            np.ndarray,
        ):
            value = value.tolist()

        elif isinstance(
            value,
            np.generic,
        ):
            value = value.item()

        record[key] = value

    return record


def install_h1_parallax_grid_multistart_runtime_patch():

    global _H1_PARALLAX_GRID_MULTISTART_PATCH_INSTALLED
    global _ORIGINAL_RUN_MULTIPLE_NAMED_FITS_H1_MULTISTART
    global _H1_EMBEDDED_H0_FLUX_CONTEXT

    if not _h1_multistart_enabled():
        return

    if _H1_PARALLAX_GRID_MULTISTART_PATCH_INSTALLED:
        return

    if _h1_initialization_mode() != "embedded_h0":
        raise ValueError(
            "H1 multistart currently requires "
            "fit.h1_initialization='embedded_H0'."
        )

    if not _h1_embed_fluxes_from_h0_enabled():
        raise ValueError(
            "H1 multistart currently requires "
            "fit.h1_embed_fluxes_from_H0=true so that the exact "
            "nested center_full candidate is available."
        )

    if _ORIGINAL_RUN_MULTIPLE_NAMED_FITS_EMBEDDED_H0 is None:
        raise RuntimeError(
            "Cannot install H1 multistart: original "
            "run_multiple_named_fits handle is unavailable."
        )

    # Current binding includes the already validated embedded-H0 wrappers.
    # Generic calls continue through it.
    current_multi = frr.run_multiple_named_fits

    # For individual grid candidates we intentionally call the ORIGINAL shared
    # function.  Otherwise the embedded-H0 wrapper would overwrite every
    # non-zero piE grid start back to piEN=piEE=0.
    base_multi = (
        _ORIGINAL_RUN_MULTIPLE_NAMED_FITS_EMBEDDED_H0
    )

    def run_multiple_named_fits_h1_multistart(
        *args,
        **kwargs,
    ):

        global _H1_EMBEDDED_H0_FLUX_CONTEXT

        fit_specs = kwargs.get(
            "fit_specs",
            None,
        )

        existing_results = kwargs.get(
            "existing_results",
            None,
        )

        if (
            not isinstance(fit_specs, dict)
            or not isinstance(existing_results, dict)
        ):
            return current_multi(
                *args,
                **kwargs,
            )

        lrt_cfg = (
            LRT_CONFIG
            if isinstance(LRT_CONFIG, dict)
            else {}
        )

        null_key = str(
            lrt_cfg.get(
                "null",
                "H0",
            )
        )

        alternative_key = str(
            lrt_cfg.get(
                "alternative",
                "H1",
            )
        )

        if (
            null_key not in existing_results
            or alternative_key not in fit_specs
        ):
            return current_multi(
                *args,
                **kwargs,
            )

        missing_fit_keys = [
            key
            for key in fit_specs
            if key not in existing_results
        ]

        if missing_fit_keys != [
            alternative_key
        ]:
            raise RuntimeError(
                "H1 multistart requires the null H0 to be the primary "
                "already-fitted model and H1 to be the only remaining fit. "
                f"remaining={missing_fit_keys}"
            )

        h0_entry = existing_results[
            null_key
        ]

        if str(
            h0_entry.get(
                "status",
                "",
            )
        ) != "fitted":
            raise RuntimeError(
                "H1 multistart: H0 is not fitted."
            )

        h0_fit_rr = h0_entry.get(
            "fit_rr",
            None,
        )

        h0_best_model = h0_entry.get(
            "best_model",
            None,
        )

        if (
            h0_fit_rr is None
            or h0_best_model is None
        ):
            raise RuntimeError(
                "H1 multistart: missing H0 fit object/best_model."
            )

        h0_fit_results = getattr(
            h0_fit_rr,
            "fit_results",
            None,
        )

        if not isinstance(
            h0_fit_results,
            dict,
        ):
            raise RuntimeError(
                "H1 multistart: H0 fit_results unavailable."
            )

        h0_order = h0_fit_results.get(
            "initial_guess_parameter_order",
            None,
        )

        if not isinstance(
            h0_order,
            (list, tuple),
        ):
            raise RuntimeError(
                "H1 multistart: H0 physical parameter order unavailable."
            )

        h0_best = np.asarray(
            h0_best_model,
            dtype=float,
        ).reshape(-1)

        n_h0_phys = len(
            h0_order
        )

        if _bounded_flux_profile_enabled():

            if len(
                h0_best
            ) != n_h0_phys:
                raise RuntimeError(
                    "H1 multistart/profile: expected H0 best_model "
                    "to contain physical parameters only. "
                    f"len(best_model)={len(h0_best)}, "
                    f"n_physical={n_h0_phys}"
                )

        elif len(
            h0_best
        ) <= n_h0_phys:
            raise RuntimeError(
                "H1 multistart: H0 best_model has no fitted fluxes."
            )

        h0_physical_guess = {
            str(name): float(
                h0_best[k]
            )
            for k, name in enumerate(
                h0_order
            )
        }

        for required in (
            "t0",
            "u0",
            "tE",
            "rho",
        ):
            if required not in h0_physical_guess:
                raise RuntimeError(
                    "H1 multistart expected FSPL H0 parameters "
                    f"including {required!r}. "
                    f"Available={list(h0_physical_guess)}"
                )

        # BOUNDED_FLUX_PROFILE_FIX_V2
        # In reduced/profile mode H0 best_model contains only
        # the physical nonlinear parameters. No explicit flux
        # vector exists or is needed.
        if _bounded_flux_profile_enabled():

            h0_fluxes = []

        else:

            h0_fluxes = [
                float(x)
                for x in h0_best[
                    n_h0_phys:
                ]
            ]

            if not np.all(
                np.isfinite(
                    h0_fluxes
                )
            ):
                raise RuntimeError(
                    "H1 multistart: non-finite H0 fitted fluxes."
                )

        h0_chi2 = _h1_multistart_entry_chi2(
            h0_entry
        )

        if not np.isfinite(
            h0_chi2
        ):
            raise RuntimeError(
                "H1 multistart: H0 chi2 unavailable."
            )

        alternative_spec = dict(
            fit_specs[
                alternative_key
            ]
        )

        if not bool(
            alternative_spec.get(
                "parallax",
                False,
            )
        ):
            raise RuntimeError(
                "H1 multistart requires a parallax alternative."
            )

        alt_bounds = alternative_spec.get(
            "bounds",
            None,
        )

        if not isinstance(
            alt_bounds,
            dict,
        ):
            raise RuntimeError(
                "H1 multistart requires explicit H1 bounds."
            )

        if (
            "piEN" not in alt_bounds
            or "piEE" not in alt_bounds
        ):
            raise RuntimeError(
                "H1 multistart requires explicit piEN/piEE bounds."
            )

        width_n = _h1_multistart_bound_half_width(
            alt_bounds["piEN"],
            "piEN",
        )

        width_e = _h1_multistart_bound_half_width(
            alt_bounds["piEE"],
            "piEE",
        )

        ms_cfg = _h1_multistart_config()

        fractions = ms_cfg.get(
            "piE_grid_fractions",
            [-0.5, 0.0, 0.5],
        )

        if not isinstance(
            fractions,
            (list, tuple),
        ):
            raise TypeError(
                "fit.h1_multistart.piE_grid_fractions must be a list."
            )

        fractions = [
            float(x)
            for x in fractions
        ]

        if len(
            fractions
        ) == 0:
            raise ValueError(
                "H1 multistart grid cannot be empty."
            )

        if not np.all(
            np.isfinite(
                fractions
            )
        ):
            raise ValueError(
                "Non-finite H1 multistart grid fraction."
            )

        for fraction in fractions:
            if abs(
                fraction
            ) > 1.0:
                raise ValueError(
                    "H1 multistart grid fractions must satisfy |f|<=1."
                )

        include_auto_center = bool(
            ms_cfg.get(
                "include_auto_center",
                True,
            )
        )

        include_exact_center = bool(
            ms_cfg.get(
                "include_exact_H0_center",
                True,
            )
        )

        # ------------------------------------------------------------
        # Build deterministic candidate list.
        # ------------------------------------------------------------

        candidates = []

        if include_exact_center:

            candidates.append(
                {
                    "id": "center_full",
                    "piEN": 0.0,
                    "piEE": 0.0,
                    "flux_mode": "H0_exact",
                }
            )

        for frac_n in fractions:

            for frac_e in fractions:

                if (
                    frac_n == 0.0
                    and frac_e == 0.0
                    and not include_auto_center
                ):
                    continue

                candidates.append(
                    {
                        "id": (
                            "grid_"
                            f"N{frac_n:+.3f}_"
                            f"E{frac_e:+.3f}_auto"
                        )
                        .replace(
                            "+",
                            "p",
                        )
                        .replace(
                            "-",
                            "m",
                        )
                        .replace(
                            ".",
                            "p",
                        ),
                        "piEN": float(
                            frac_n
                            * width_n
                        ),
                        "piEE": float(
                            frac_e
                            * width_e
                        ),
                        "flux_mode": "auto",
                    }
                )

        if len(
            candidates
        ) == 0:
            raise RuntimeError(
                "H1 multistart produced zero candidates."
            )

        path_to_save_fit = kwargs.get(
            "path_to_save_fit",
            None,
        )

        if path_to_save_fit is None:
            raise RuntimeError(
                "H1 multistart requires keyword path_to_save_fit."
            )

        base_fit_path = Path(
            path_to_save_fit
        )

        multistart_root = (
            base_fit_path
            / "_H1_multistart"
        )

        multistart_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        candidate_records = []

        # H1_TOPK_DIAGNOSTIC_POLISH_PATCH_V1
        # Keep successful coarse entries so selected ranks can be
        # independently polished for basin/globality diagnostics.
        successful_candidate_entries = []

        best_entry = None
        best_chi2 = np.inf
        best_candidate = None
        best_candidate_dir = None

        print(
            "[H1_multistart] "
            f"N_candidates={len(candidates)}, "
            f"piEN_half_width={width_n}, "
            f"piEE_half_width={width_e}, "
            f"fractions={fractions}",
            flush=True,
        )

        # ------------------------------------------------------------
        # Run all H1 candidates on the SAME lc_to_fit.
        # ------------------------------------------------------------

        for candidate_index, candidate in enumerate(
            candidates
        ):

            start_guess = dict(
                h0_physical_guess
            )

            start_guess[
                "piEN"
            ] = float(
                candidate["piEN"]
            )

            start_guess[
                "piEE"
            ] = float(
                candidate["piEE"]
            )

            spec = dict(
                alternative_spec
            )

            spec[
                "initial_guess"
            ] = start_guess

            fit_specs_one = {
                alternative_key: spec
            }

            candidate_dir = (
                multistart_root
                / (
                    f"{candidate_index:02d}_"
                    f"{candidate['id']}"
                )
            )

            candidate_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            kwargs_one = dict(
                kwargs
            )

            kwargs_one[
                "fit_specs"
            ] = fit_specs_one

            kwargs_one[
                "existing_results"
            ] = existing_results

            kwargs_one[
                "path_to_save_fit"
            ] = str(
                candidate_dir
            )

            if _H1_EMBEDDED_H0_FLUX_CONTEXT is not None:
                raise RuntimeError(
                    "H1 multistart found stale H0-flux context."
                )

            if candidate[
                "flux_mode"
            ] == "H0_exact":

                _H1_EMBEDDED_H0_FLUX_CONTEXT = list(
                    h0_fluxes
                )

            print(
                "[H1_multistart] START "
                f"{candidate_index:02d} "
                f"id={candidate['id']} "
                f"piEN={candidate['piEN']:.16g} "
                f"piEE={candidate['piEE']:.16g} "
                f"flux_mode={candidate['flux_mode']}",
                flush=True,
            )

            try:

                candidate_result = base_multi(
                    *args,
                    **kwargs_one,
                )

            finally:

                _H1_EMBEDDED_H0_FLUX_CONTEXT = None

            entry = candidate_result.get(
                alternative_key,
                None,
            )

            status = None

            if isinstance(
                entry,
                dict,
            ):
                status = entry.get(
                    "status",
                    None,
                )

            chi2 = _h1_multistart_entry_chi2(
                entry
            )

            optimizer_record = (
                _h1_multistart_optimizer_record(
                    entry
                )
            )

            record = {
                "index": int(
                    candidate_index
                ),
                "id": str(
                    candidate["id"]
                ),
                "piEN": float(
                    candidate["piEN"]
                ),
                "piEE": float(
                    candidate["piEE"]
                ),
                "flux_mode": str(
                    candidate["flux_mode"]
                ),
                "status": (
                    None
                    if status is None
                    else str(status)
                ),
                "chi2": (
                    None
                    if not np.isfinite(
                        chi2
                    )
                    else float(
                        chi2
                    )
                ),
            }

            record.update(
                optimizer_record
            )

            # H1_MULTISTART_SAVE_FINAL_PIE_V1
            # Save the final fitted parallax, not only the grid start.
            if isinstance(entry, dict):

                candidate_fit_rr = entry.get(
                    "fit_rr",
                    None,
                )

                candidate_fit_results = getattr(
                    candidate_fit_rr,
                    "fit_results",
                    None,
                )

                if isinstance(
                    candidate_fit_results,
                    dict,
                ):

                    candidate_best = candidate_fit_results.get(
                        "best_model",
                        None,
                    )

                    candidate_order = candidate_fit_results.get(
                        "initial_guess_parameter_order",
                        None,
                    )

                    if (
                        candidate_best is not None
                        and isinstance(
                            candidate_order,
                            (list, tuple),
                        )
                    ):

                        candidate_best = np.asarray(
                            candidate_best,
                            dtype=float,
                        ).reshape(-1)

                        for parallax_name in (
                            "piEN",
                            "piEE",
                        ):

                            if parallax_name in candidate_order:

                                parallax_index = list(
                                    candidate_order
                                ).index(
                                    parallax_name
                                )

                                if parallax_index < len(
                                    candidate_best
                                ):

                                    record[
                                        "best_" + parallax_name
                                    ] = float(
                                        candidate_best[
                                            parallax_index
                                        ]
                                    )

            candidate_records.append(
                record
            )

            if (
                status == "fitted"
                and np.isfinite(chi2)
            ):
                successful_candidate_entries.append({
                    "index": int(candidate_index),
                    "chi2": float(chi2),
                    "candidate": dict(candidate),
                    "entry": entry,
                })

            print(
                "[H1_multistart] END "
                f"{candidate_index:02d} "
                f"id={candidate['id']} "
                f"status={status} "
                f"chi2={chi2} "
                f"optimality={optimizer_record.get('optimizer_optimality')}",
                flush=True,
            )

            if (
                status == "fitted"
                and np.isfinite(
                    chi2
                )
                and chi2 < best_chi2
            ):

                best_chi2 = float(
                    chi2
                )

                best_entry = entry
                best_candidate = dict(
                    candidate
                )
                best_candidate[
                    "index"
                ] = int(
                    candidate_index
                )

                best_candidate_dir = (
                    candidate_dir
                )

        if (
            best_entry is None
            or best_candidate is None
        ):
            raise RuntimeError(
                "H1 multistart: no successful finite H1 candidate."
            )

        # ============================================================
        # H1_GLOBAL_DE_DIAGNOSTIC_PATCH_V1
        #
        # Optional diagnostic global search:
        #
        #   pyLIMA Differential Evolution (6D physical H1)
        #       -> exact bounded flux profiling at every evaluation
        #       -> strict TRF polish from DE best point
        #
        # Diagnostic only: this DOES NOT replace the normal H1 result.
        # ============================================================

        de_cfg = (
            ms_cfg.get(
                "diagnostic_global_de",
                {},
            )
            or {}
        )

        de_enabled = bool(
            de_cfg.get(
                "enabled",
                False,
            )
        )

        if de_enabled:

            if not _bounded_flux_profile_enabled():
                raise RuntimeError(
                    "Global DE diagnostic requires "
                    "HIDDEN_PARALLAX_BOUNDED_PROFILE=1."
                )

            from pyLIMA.fits import DE_fit as _DE_fit_local

            de_population_size = int(
                de_cfg.get(
                    "population_size",
                    10,
                )
            )

            de_max_iteration = int(
                de_cfg.get(
                    "max_iteration",
                    1500,
                )
            )

            # H1_GLOBAL_DE_STRICT_STOPPING_FIX_V4
            #
            # pyLIMA 1.9.8 hardcodes scipy DE with atol=1,
            # which is far too loose for this likelihood problem.
            de_atol = float(
                de_cfg.get(
                    "atol",
                    1.0e-4,
                )
            )

            de_tol = float(
                de_cfg.get(
                    "tol",
                    0.0,
                )
            )

            if (
                not np.isfinite(de_atol)
                or de_atol < 0.0
            ):
                raise ValueError(
                    f"Invalid DE atol={de_atol}"
                )

            if (
                not np.isfinite(de_tol)
                or de_tol < 0.0
            ):
                raise ValueError(
                    f"Invalid DE tol={de_tol}"
                )

            de_strategy = str(
                de_cfg.get(
                    "strategy",
                    "rand1bin",
                )
            )

            de_seeds = de_cfg.get(
                "seeds",
                [20260903],
            )

            if not isinstance(
                de_seeds,
                (list, tuple),
            ):
                de_seeds = [
                    de_seeds
                ]

            de_seeds = [
                int(x)
                for x in de_seeds
            ]

            if len(de_seeds) == 0:
                raise ValueError(
                    "diagnostic_global_de.seeds cannot be empty."
                )

            if de_population_size <= 0:
                raise ValueError(
                    "diagnostic_global_de.population_size "
                    "must be > 0."
                )

            if de_max_iteration <= 0:
                raise ValueError(
                    "diagnostic_global_de.max_iteration "
                    "must be > 0."
                )

            de_polish_options = {
                "xtol": 1.0e-10,
                "ftol": 1.0e-10,
                "gtol": 1.0e-8,
                "max_nfev": 50000,
                "x_scale": "jac",
            }

            user_de_polish_options = (
                de_cfg.get(
                    "trf_polish_optimizer_options",
                    None,
                )
            )

            if user_de_polish_options is not None:

                if not isinstance(
                    user_de_polish_options,
                    dict,
                ):
                    raise TypeError(
                        "diagnostic_global_de."
                        "trf_polish_optimizer_options "
                        "must be a dict."
                    )

                de_polish_options.update(
                    user_de_polish_options
                )

            reference_fit_rr = (
                best_entry.get(
                    "fit_rr",
                    None,
                )
            )

            if reference_fit_rr is None:
                raise RuntimeError(
                    "Global DE diagnostic could not recover "
                    "reference H1 fit object."
                )

            de_model = getattr(
                reference_fit_rr,
                "model",
                None,
            )

            if de_model is None:
                raise RuntimeError(
                    "Global DE diagnostic could not recover "
                    "H1 pyLIMA model."
                )

            de_root = (
                base_fit_path
                / "_H1_global_DE"
            )

            de_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            print(
                "[H1_global_DE] "
                f"enabled=True "
                f"population_size={de_population_size} "
                f"max_iteration={de_max_iteration} "
                f"strategy={de_strategy} "
                f"seeds={de_seeds}",
                flush=True,
            )

            for de_seed in de_seeds:

                seed_dir = (
                    de_root
                    / f"seed_{de_seed}"
                )

                seed_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # -----------------------------------------------
                # Build a fresh DE fitter.
                # polyfit removes telescope fluxes from the
                # nonlinear vector. Our runtime derive-flux patch
                # replaces pyLIMA's unbounded polyfit with the
                # exact bounded profile.
                # -----------------------------------------------

                de_fit = _DE_fit_local.DEfit(
                    de_model,
                    telescopes_fluxes_method="polyfit",
                    loss_function="chi2",
                    DE_population_size=de_population_size,
                    max_iteration=de_max_iteration,
                    display_progress=False,
                    strategy=de_strategy,
                )

                # H1_GLOBAL_DE_MATCH_H1_BOUNDS_FIX_V3
                #
                # DEfit initializes with pyLIMA's standard model bounds.
                # For this diagnostic we MUST use exactly the same resolved
                # physical H1 domain as the TRF fits used by the LRT.
                #
                # reference_fit_rr is the already-created bounded-profile
                # H1 TRF fitter. Its fit_parameters therefore already contain
                # the final numerical bounds after fit_lc.apply_fit_bounds()
                # and apply_custom_bounds().
                #
                reference_fit_parameters = getattr(
                    reference_fit_rr,
                    "fit_parameters",
                    None,
                )

                if not isinstance(
                    reference_fit_parameters,
                    dict,
                ):
                    raise RuntimeError(
                        "Global DE could not recover reference H1 "
                        "fit_parameters."
                    )

                de_order = list(
                    de_fit.fit_parameters.keys()
                )

                reference_order = list(
                    reference_fit_parameters.keys()
                )

                if de_order != reference_order:
                    raise RuntimeError(
                        "Global DE/reference H1 parameter-order mismatch: "
                        f"DE={de_order}, reference={reference_order}"
                    )

                de_resolved_bounds = {}

                for parameter_name in de_order:

                    reference_interval = (
                        reference_fit_parameters[
                            parameter_name
                        ][1]
                    )

                    lo = float(
                        reference_interval[0]
                    )

                    hi = float(
                        reference_interval[1]
                    )

                    if (
                        not np.isfinite(lo)
                        or not np.isfinite(hi)
                        or lo >= hi
                    ):
                        raise RuntimeError(
                            "Invalid reference H1 bounds for "
                            f"{parameter_name}: {reference_interval}"
                        )

                    # Exact same numerical interval used by H1 TRF.
                    de_fit.fit_parameters[
                        parameter_name
                    ][1] = [
                        lo,
                        hi,
                    ]

                    # Keep prior metadata consistent too. For chi2 the
                    # priors do not enter the objective, but there is no
                    # reason to leave contradictory physical bounds here.
                    if (
                        isinstance(
                            de_fit.priors_parameters,
                            dict,
                        )
                        and parameter_name
                        in de_fit.priors_parameters
                    ):
                        de_fit.priors_parameters[
                            parameter_name
                        ][1] = [
                            lo,
                            hi,
                        ]

                    de_resolved_bounds[
                        parameter_name
                    ] = [
                        lo,
                        hi,
                    ]

                print(
                    "[H1_global_DE] MATCHED_H1_BOUNDS "
                    f"{de_resolved_bounds}",
                    flush=True,
                )

                required_de = {
                    "t0",
                    "u0",
                    "tE",
                    "rho",
                    "piEN",
                    "piEE",
                }

                if (
                    len(de_order) != 6
                    or set(de_order) != required_de
                ):
                    raise RuntimeError(
                        "Global DE expected exactly the 6 "
                        "physical H1 parameters, got "
                        f"{de_order}"
                    )

                print(
                    "[H1_global_DE] START "
                    f"seed={de_seed} "
                    f"order={de_order}",
                    flush=True,
                )

                # scipy differential_evolution in this pyLIMA
                # version does not expose an RNG argument through
                # DEfit. Make the diagnostic deterministic while
                # restoring NumPy global state afterwards.
                np_state = np.random.get_state()

                # pyLIMA DEfit.fit() hardcodes:
                #
                #     tol=0.0
                #     atol=1
                #
                # Temporarily wrap scipy DE so the diagnostic can use
                # a scientifically meaningful stopping criterion.
                original_scipy_de = (
                    _DE_fit_local.scipy.optimize.differential_evolution
                )

                def strict_scipy_de(*de_args, **de_kwargs):

                    de_kwargs["atol"] = float(
                        de_atol
                    )

                    de_kwargs["tol"] = float(
                        de_tol
                    )

                    return original_scipy_de(
                        *de_args,
                        **de_kwargs,
                    )

                _DE_fit_local.scipy.optimize.differential_evolution = (
                    strict_scipy_de
                )

                try:

                    np.random.seed(
                        int(de_seed)
                    )

                    de_fit.fit()

                finally:

                    _DE_fit_local.scipy.optimize.differential_evolution = (
                        original_scipy_de
                    )

                    np.random.set_state(
                        np_state
                    )

                de_results = getattr(
                    de_fit,
                    "fit_results",
                    None,
                )

                if not isinstance(
                    de_results,
                    dict,
                ):
                    raise RuntimeError(
                        "Global DE returned no fit_results."
                    )

                # H1_GLOBAL_DE_DIAGNOSTIC_FIX_V2
                #
                # In pyLIMA 1.9.8 DEfit.fit_results["best_model"]
                # is reconstructed from trials_parameters. With profiled
                # telescope fluxes, that record may contain the physical
                # DE coordinates plus derived flux parameters.
                #
                # The authoritative nonlinear optimizer vector is instead
                # scipy.optimize.differential_evolution result.x.
                #
                de_fit_object = de_results.get(
                    "fit_object",
                    None,
                )

                if de_fit_object is None:
                    raise RuntimeError(
                        "Global DE returned no fit_object."
                    )

                de_x = getattr(
                    de_fit_object,
                    "x",
                    None,
                )

                if (
                    de_x is None
                    and isinstance(
                        de_fit_object,
                        dict,
                    )
                ):
                    de_x = de_fit_object.get(
                        "x",
                        None,
                    )

                if de_x is None:
                    raise RuntimeError(
                        "Global DE fit_object contains no optimizer x."
                    )

                de_best = np.asarray(
                    de_x,
                    dtype=float,
                ).reshape(-1)

                if len(de_best) != len(de_order):
                    raise RuntimeError(
                        "Global DE optimizer x length "
                        f"is {len(de_best)}, "
                        f"expected {len(de_order)} "
                        f"for order={de_order}."
                    )

                if len(de_best) != 6:
                    raise RuntimeError(
                        "Global DE expected 6 physical H1 "
                        f"parameters but optimizer x has {len(de_best)}."
                    )

                raw_best_model_value = de_results.get(
                    "best_model",
                    None,
                )

                if raw_best_model_value is None:
                    de_best_raw = np.asarray(
                        [],
                        dtype=float,
                    )
                else:
                    de_best_raw = np.asarray(
                        raw_best_model_value,
                        dtype=float,
                    ).reshape(-1)

                de_chi2 = float(
                    de_results.get(
                        "chi2",
                        np.nan,
                    )
                )

                de_time = float(
                    de_results.get(
                        "fit_time",
                        np.nan,
                    )
                )

                if not np.isfinite(
                    de_chi2
                ):
                    raise RuntimeError(
                        "Global DE returned non-finite chi2."
                    )

                de_guess = {
                    str(name): float(de_best[k])
                    for k, name in enumerate(
                        de_order
                    )
                }

                compact_de = {
                    "seed": int(de_seed),
                    "parameter_order": list(
                        de_order
                    ),
                    # Physical coordinates actually optimized by DE.
                    "best_model": np.asarray(
                        de_best,
                        dtype=float,
                    ),
                    "optimizer_x": np.asarray(
                        de_best,
                        dtype=float,
                    ),

                    # Keep pyLIMA's reconstructed record for diagnosis.
                    "pyLIMA_best_model_raw": np.asarray(
                        de_best_raw,
                        dtype=float,
                    ),
                    "pyLIMA_best_model_raw_length": int(
                        len(de_best_raw)
                    ),

                    "chi2": float(
                        de_chi2
                    ),
                    "fit_time": float(
                        de_time
                    ),
                    "population_size": int(
                        de_population_size
                    ),
                    "max_iteration": int(
                        de_max_iteration
                    ),
                    "atol": float(
                        de_atol
                    ),
                    "tol": float(
                        de_tol
                    ),
                    "nit": int(
                        de_fit_object.get(
                            "nit",
                            -1,
                        )
                    ),
                    "nfev": int(
                        de_fit_object.get(
                            "nfev",
                            -1,
                        )
                    ),
                    "success": bool(
                        de_fit_object.get(
                            "success",
                            False,
                        )
                    ),
                    "message": str(
                        de_fit_object.get(
                            "message",
                            "",
                        )
                    ),
                    "population_energy_std": float(
                        np.std(
                            np.asarray(
                                de_fit_object.get(
                                    "population_energies",
                                    [],
                                ),
                                dtype=float,
                            )
                        )
                    ),
                    "strategy": str(
                        de_strategy
                    ),
                    "fit_parameters": {
                        str(k): v
                        for k, v in
                        de_fit.fit_parameters.items()
                    },
                    "resolved_H1_bounds": {
                        str(k): list(v)
                        for k, v in
                        de_resolved_bounds.items()
                    },
                }

                np.save(
                    seed_dir / "de_raw.npy",
                    compact_de,
                    allow_pickle=True,
                )

                print(
                    "[H1_global_DE] RAW_END "
                    f"seed={de_seed} "
                    f"chi2={de_chi2:.16g} "
                    f"fit_time={de_time}",
                    flush=True,
                )

                # -----------------------------------------------
                # Strict TRF polish from DE global solution.
                # -----------------------------------------------

                de_polish_spec = dict(
                    alternative_spec
                )

                de_polish_spec[
                    "initial_guess"
                ] = de_guess

                de_polish_dir = (
                    seed_dir
                    / "trf_polish"
                )

                de_polish_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                kwargs_de_polish = dict(
                    kwargs
                )

                kwargs_de_polish[
                    "fit_specs"
                ] = {
                    alternative_key:
                        de_polish_spec
                }

                kwargs_de_polish[
                    "existing_results"
                ] = existing_results

                kwargs_de_polish[
                    "path_to_save_fit"
                ] = str(
                    de_polish_dir
                )

                kwargs_de_polish[
                    "optimizer_options"
                ] = dict(
                    de_polish_options
                )

                if (
                    _H1_EMBEDDED_H0_FLUX_CONTEXT
                    is not None
                ):
                    raise RuntimeError(
                        "Global DE diagnostic found stale "
                        "embedded-flux context."
                    )

                print(
                    "[H1_global_DE] TRF_START "
                    f"seed={de_seed} "
                    f"DE_chi2={de_chi2:.16g}",
                    flush=True,
                )

                try:

                    de_polish_results = base_multi(
                        *args,
                        **kwargs_de_polish,
                    )

                finally:

                    _H1_EMBEDDED_H0_FLUX_CONTEXT = None

                de_polish_entry = (
                    de_polish_results.get(
                        alternative_key,
                        None,
                    )
                )

                de_polish_chi2 = (
                    _h1_multistart_entry_chi2(
                        de_polish_entry
                    )
                )

                de_polish_optimizer = (
                    _h1_multistart_optimizer_record(
                        de_polish_entry
                    )
                )

                print(
                    "[H1_global_DE] TRF_END "
                    f"seed={de_seed} "
                    f"chi2={de_polish_chi2} "
                    f"improvement="
                    f"{de_chi2 - de_polish_chi2 if np.isfinite(de_polish_chi2) else np.nan} "
                    f"nfev="
                    f"{de_polish_optimizer.get('optimizer_nfev')} "
                    f"optimality="
                    f"{de_polish_optimizer.get('optimizer_optimality')}",
                    flush=True,
                )

        # ============================================================
        # H1_TOPK_DIAGNOSTIC_POLISH_PATCH_V1
        #
        # Optional diagnostic only:
        #
        #   coarse candidates
        #       -> sort by chi2
        #       -> polish the best K independently
        #
        # These fits are written below _H1_multistart but DO NOT
        # replace the normal H1 final result.  This lets us test
        # basin/globality without changing production behavior.
        # ============================================================

        diagnostic_top_k = int(
            ms_cfg.get(
                "diagnostic_polish_top_k",
                0,
            )
        )

        if diagnostic_top_k < 0:
            raise ValueError(
                "h1_multistart.diagnostic_polish_top_k "
                "must be >= 0."
            )

        if diagnostic_top_k > 0:

            if not _bounded_flux_profile_enabled():
                raise RuntimeError(
                    "Top-K diagnostic polish is currently intended "
                    "for bounded-profile mode only."
                )

            ranked_entries = sorted(
                successful_candidate_entries,
                key=lambda item: item["chi2"],
            )

            diagnostic_top_k = min(
                diagnostic_top_k,
                len(ranked_entries),
            )

            topk_options = {
                "xtol": 1.0e-10,
                "ftol": 1.0e-10,
                "gtol": 1.0e-8,
                "max_nfev": 50000,
                "x_scale": "jac",
            }

            user_topk_options = ms_cfg.get(
                "diagnostic_topk_polish_optimizer_options",
                None,
            )

            if user_topk_options is not None:

                if not isinstance(
                    user_topk_options,
                    dict,
                ):
                    raise TypeError(
                        "diagnostic_topk_polish_optimizer_options "
                        "must be a dict."
                    )

                topk_options.update(
                    user_topk_options
                )

            print(
                "[H1_topk_polish] "
                f"enabled=True "
                f"K={diagnostic_top_k} "
                f"optimizer_options={topk_options}",
                flush=True,
            )

            for rank_index, ranked in enumerate(
                ranked_entries[:diagnostic_top_k],
                start=1,
            ):

                source_entry = ranked["entry"]
                source_candidate = ranked["candidate"]

                source_fit_rr = source_entry.get(
                    "fit_rr",
                    None,
                )

                source_fit_results = getattr(
                    source_fit_rr,
                    "fit_results",
                    None,
                )

                if not isinstance(
                    source_fit_results,
                    dict,
                ):
                    raise RuntimeError(
                        "Top-K polish: source fit_results unavailable."
                    )

                source_best = source_entry.get(
                    "best_model",
                    None,
                )

                if source_best is None:
                    source_best = source_fit_results.get(
                        "best_model",
                        None,
                    )

                source_order = source_fit_results.get(
                    "initial_guess_parameter_order",
                    None,
                )

                if (
                    source_best is None
                    or not isinstance(
                        source_order,
                        (list, tuple),
                    )
                ):
                    raise RuntimeError(
                        "Top-K polish: source best_model/order "
                        "unavailable."
                    )

                source_best = np.asarray(
                    source_best,
                    dtype=float,
                ).reshape(-1)

                n_phys_topk = len(
                    source_order
                )

                if len(source_best) != n_phys_topk:
                    raise RuntimeError(
                        "Top-K profile polish expected physical-only "
                        f"best_model length {n_phys_topk}, "
                        f"got {len(source_best)}."
                    )

                topk_initial_guess = {
                    str(name): float(source_best[k])
                    for k, name in enumerate(source_order)
                }

                for required in (
                    "t0",
                    "u0",
                    "tE",
                    "rho",
                    "piEN",
                    "piEE",
                ):
                    if required not in topk_initial_guess:
                        raise RuntimeError(
                            "Top-K polish missing physical parameter "
                            f"{required!r}; order={source_order}"
                        )

                topk_spec = dict(
                    alternative_spec
                )

                topk_spec[
                    "initial_guess"
                ] = topk_initial_guess

                topk_dir = (
                    multistart_root
                    / (
                        f"polish_topk_rank{rank_index:02d}_"
                        f"{source_candidate['id']}"
                    )
                )

                topk_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                kwargs_topk = dict(
                    kwargs
                )

                kwargs_topk[
                    "fit_specs"
                ] = {
                    alternative_key: topk_spec
                }

                kwargs_topk[
                    "existing_results"
                ] = existing_results

                kwargs_topk[
                    "path_to_save_fit"
                ] = str(
                    topk_dir
                )

                kwargs_topk[
                    "optimizer_options"
                ] = dict(
                    topk_options
                )

                if _H1_EMBEDDED_H0_FLUX_CONTEXT is not None:
                    raise RuntimeError(
                        "Top-K polish found stale flux context."
                    )

                print(
                    "[H1_topk_polish] START "
                    f"rank={rank_index} "
                    f"source={source_candidate['id']} "
                    f"coarse_chi2={ranked['chi2']:.16g} "
                    f"piEN0={topk_initial_guess['piEN']:.16g} "
                    f"piEE0={topk_initial_guess['piEE']:.16g}",
                    flush=True,
                )

                try:

                    topk_results = base_multi(
                        *args,
                        **kwargs_topk,
                    )

                finally:

                    _H1_EMBEDDED_H0_FLUX_CONTEXT = None

                topk_entry = topk_results.get(
                    alternative_key,
                    None,
                )

                topk_chi2 = _h1_multistart_entry_chi2(
                    topk_entry
                )

                topk_optimizer = (
                    _h1_multistart_optimizer_record(
                        topk_entry
                    )
                )

                print(
                    "[H1_topk_polish] END "
                    f"rank={rank_index} "
                    f"source={source_candidate['id']} "
                    f"chi2={topk_chi2} "
                    f"improvement="
                    f"{ranked['chi2'] - topk_chi2 if np.isfinite(topk_chi2) else np.nan} "
                    f"nfev={topk_optimizer.get('optimizer_nfev')} "
                    f"optimality="
                    f"{topk_optimizer.get('optimizer_optimality')}",
                    flush=True,
                )

        # ============================================================
        # H1_MULTISTART_WINNER_POLISH_V1
        #
        # Re-run TRF from the COMPLETE best multistart solution:
        # physical H1 parameters + fitted telescope fluxes.
        #
        # This is a numerical refinement only.  The final H1 result is
        # min(grid winner, polished winner), so polishing cannot degrade H1.
        # ============================================================

        polish_record = {
            "enabled": False,
            "attempted": False,
            "accepted": False,
        }

        # H1_ADAPTIVE_PROFILE_POLISH_PATCH_V1
        #
        # polish_winner accepts:
        #
        #   False       -> never polish
        #   True        -> always polish
        #   "adaptive"  -> polish only when the coarse winner has
        #                  suspicious optimizer optimality.
        #
        # Missing/non-finite optimality is treated conservatively:
        # the winner is polished.
        polish_setting = ms_cfg.get(
            "polish_winner",
            False,
        )

        if isinstance(
            polish_setting,
            str,
        ):

            polish_mode = (
                polish_setting
                .strip()
                .lower()
            )

            if polish_mode in (
                "adaptive",
            ):
                polish_mode = "adaptive"

            elif polish_mode in (
                "always",
                "true",
                "yes",
                "on",
            ):
                polish_mode = "always"

            elif polish_mode in (
                "never",
                "false",
                "no",
                "off",
            ):
                polish_mode = "never"

            else:
                raise ValueError(
                    "h1_multistart.polish_winner must be "
                    "False, True, or 'adaptive'; "
                    f"got {polish_setting!r}."
                )

        else:

            polish_mode = (
                "always"
                if bool(polish_setting)
                else "never"
            )

        polish_threshold = float(
            ms_cfg.get(
                "polish_optimality_threshold",
                0.05,
            )
        )

        if (
            not np.isfinite(
                polish_threshold
            )
            or polish_threshold < 0.0
        ):
            raise ValueError(
                "h1_multistart.polish_optimality_threshold "
                "must be finite and >= 0."
            )

        coarse_winner_optimizer = (
            _h1_multistart_optimizer_record(
                best_entry
            )
        )

        coarse_opt_raw = (
            coarse_winner_optimizer.get(
                "optimizer_optimality",
                None,
            )
        )

        try:
            coarse_optimality = float(
                coarse_opt_raw
            )
        except (
            TypeError,
            ValueError,
        ):
            coarse_optimality = np.nan

        if polish_mode == "always":

            polish_enabled = True
            polish_reason = "always"

        elif polish_mode == "never":

            polish_enabled = False
            polish_reason = "disabled"

        else:

            if not np.isfinite(
                coarse_optimality
            ):

                polish_enabled = True
                polish_reason = (
                    "nonfinite_coarse_optimality"
                )

            elif (
                coarse_optimality
                > polish_threshold
            ):

                polish_enabled = True
                polish_reason = (
                    "coarse_optimality_above_threshold"
                )

            else:

                polish_enabled = False
                polish_reason = (
                    "coarse_optimality_below_threshold"
                )

        polish_record.update({
            "policy_mode": str(
                polish_mode
            ),
            "optimality_threshold": (
                float(polish_threshold)
                if polish_mode == "adaptive"
                else None
            ),
            "coarse_winner_optimality": (
                None
                if not np.isfinite(
                    coarse_optimality
                )
                else float(
                    coarse_optimality
                )
            ),
            "triggered": bool(
                polish_enabled
            ),
            "trigger_reason": str(
                polish_reason
            ),
        })

        if (
            polish_mode == "adaptive"
            and not polish_enabled
        ):

            print(
                "[H1_polish] SKIP "
                f"mode=adaptive "
                f"coarse_optimality={coarse_optimality} "
                f"threshold={polish_threshold} "
                f"reason={polish_reason}",
                flush=True,
            )

        if polish_enabled:

            polish_record["enabled"] = True
            polish_record["attempted"] = True

            winner_fit_rr = best_entry.get(
                "fit_rr",
                None,
            )

            winner_fit_results = getattr(
                winner_fit_rr,
                "fit_results",
                None,
            )

            winner_best = best_entry.get(
                "best_model",
                None,
            )

            if (
                not isinstance(winner_fit_results, dict)
                or winner_best is None
            ):
                raise RuntimeError(
                    "H1 polish: winning fit result/best_model unavailable."
                )

            winner_order = winner_fit_results.get(
                "initial_guess_parameter_order",
                None,
            )

            if not isinstance(
                winner_order,
                (list, tuple),
            ):
                raise RuntimeError(
                    "H1 polish: physical parameter order unavailable."
                )

            winner_best = np.asarray(
                winner_best,
                dtype=float,
            ).reshape(-1)

            n_phys = len(
                winner_order
            )

            # H1_PROFILE_POLISH_6D_PATCH_V1
            #
            # Legacy explicit-flux mode:
            #     best_model = physical parameters + telescope fluxes
            #
            # Bounded-profile mode:
            #     best_model = physical parameters only
            #
            # In profile mode the telescope fluxes must NOT be injected
            # into the polish. They are re-profiled at every residual
            # evaluation by the bounded-flux runtime patch.
            profile_polish = _bounded_flux_profile_enabled()

            if profile_polish:

                if len(winner_best) != n_phys:
                    raise RuntimeError(
                        "H1 profile polish: expected physical-only "
                        f"best_model length {n_phys}, got "
                        f"{len(winner_best)}."
                    )

            else:

                if len(winner_best) <= n_phys:
                    raise RuntimeError(
                        "H1 polish: winning best_model has no "
                        "flux parameters."
                    )

            polish_physical_guess = {
                str(name): float(winner_best[k])
                for k, name in enumerate(winner_order)
            }

            for required in (
                "t0",
                "u0",
                "tE",
                "rho",
                "piEN",
                "piEE",
            ):
                if required not in polish_physical_guess:
                    raise RuntimeError(
                        "H1 polish missing physical parameter "
                        f"{required!r}; order={winner_order}"
                    )

            if profile_polish:

                polish_fluxes = []

            else:

                polish_fluxes = [
                    float(x)
                    for x in winner_best[n_phys:]
                ]

                if not np.all(
                    np.isfinite(polish_fluxes)
                ):
                    raise RuntimeError(
                        "H1 polish: non-finite winner flux vector."
                    )

            polish_options_default = {
                "xtol": 1.0e-12,
                "ftol": 1.0e-12,
                "gtol": 1.0e-8,
                "max_nfev": 50000,
                "x_scale": "jac",
            }

            polish_options = dict(
                polish_options_default
            )

            user_polish_options = ms_cfg.get(
                "polish_optimizer_options",
                None,
            )

            if user_polish_options is not None:

                if not isinstance(
                    user_polish_options,
                    dict,
                ):
                    raise TypeError(
                        "h1_multistart.polish_optimizer_options "
                        "must be a dict."
                    )

                polish_options.update(
                    user_polish_options
                )

            polish_spec = dict(
                alternative_spec
            )

            polish_spec[
                "initial_guess"
            ] = polish_physical_guess

            polish_dir = (
                multistart_root
                / "polish_winner"
            )

            polish_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            kwargs_polish = dict(
                kwargs
            )

            kwargs_polish[
                "fit_specs"
            ] = {
                alternative_key: polish_spec
            }

            kwargs_polish[
                "existing_results"
            ] = existing_results

            kwargs_polish[
                "path_to_save_fit"
            ] = str(
                polish_dir
            )

            kwargs_polish[
                "optimizer_options"
            ] = polish_options

            if _H1_EMBEDDED_H0_FLUX_CONTEXT is not None:
                raise RuntimeError(
                    "H1 polish found stale flux context."
                )

            if profile_polish:

                # Fluxes remain implicit/profiled.
                _H1_EMBEDDED_H0_FLUX_CONTEXT = None

            else:

                _H1_EMBEDDED_H0_FLUX_CONTEXT = list(
                    polish_fluxes
                )

            chi2_before_polish = float(
                best_chi2
            )

            print(
                "[H1_polish] START "
                f"from={best_candidate['id']} "
                f"chi2_before={chi2_before_polish:.16g} "
                f"piEN0={polish_physical_guess['piEN']:.16g} "
                f"piEE0={polish_physical_guess['piEE']:.16g} "
                f"profile_mode={profile_polish} "
                f"optimizer_options={polish_options}",
                flush=True,
            )

            try:

                polish_results = base_multi(
                    *args,
                    **kwargs_polish,
                )

            finally:

                _H1_EMBEDDED_H0_FLUX_CONTEXT = None

            polish_entry = polish_results.get(
                alternative_key,
                None,
            )

            polish_status = None

            if isinstance(
                polish_entry,
                dict,
            ):
                polish_status = polish_entry.get(
                    "status",
                    None,
                )

            polish_chi2 = _h1_multistart_entry_chi2(
                polish_entry
            )

            polish_optimizer = (
                _h1_multistart_optimizer_record(
                    polish_entry
                )
            )

            polish_record.update({
                "source_candidate_id": str(
                    best_candidate["id"]
                ),
                "chi2_before": float(
                    chi2_before_polish
                ),
                "chi2_after": (
                    None
                    if not np.isfinite(polish_chi2)
                    else float(polish_chi2)
                ),
                "improvement": (
                    None
                    if not np.isfinite(polish_chi2)
                    else float(
                        chi2_before_polish
                        - polish_chi2
                    )
                ),
                "status": (
                    None
                    if polish_status is None
                    else str(polish_status)
                ),
                "start_piEN": float(
                    polish_physical_guess["piEN"]
                ),
                "start_piEE": float(
                    polish_physical_guess["piEE"]
                ),
                "optimizer_options": dict(
                    polish_options
                ),
            })

            polish_record.update(
                polish_optimizer
            )

            # Save the final polished physical parameters.
            if isinstance(
                polish_entry,
                dict,
            ):

                polish_best = polish_entry.get(
                    "best_model",
                    None,
                )

                polish_fit_rr = polish_entry.get(
                    "fit_rr",
                    None,
                )

                polish_fit_results = getattr(
                    polish_fit_rr,
                    "fit_results",
                    None,
                )

                polish_order = None

                if isinstance(
                    polish_fit_results,
                    dict,
                ):
                    polish_order = polish_fit_results.get(
                        "initial_guess_parameter_order",
                        None,
                    )

                if (
                    polish_best is not None
                    and isinstance(
                        polish_order,
                        (list, tuple),
                    )
                ):

                    polish_best = np.asarray(
                        polish_best,
                        dtype=float,
                    ).reshape(-1)

                    for parallax_name in (
                        "piEN",
                        "piEE",
                    ):

                        if parallax_name in polish_order:

                            j = list(
                                polish_order
                            ).index(
                                parallax_name
                            )

                            if j < len(
                                polish_best
                            ):
                                polish_record[
                                    "final_" + parallax_name
                                ] = float(
                                    polish_best[j]
                                )

            # Accept only a strictly better finite fitted solution.
            if (
                polish_status == "fitted"
                and np.isfinite(polish_chi2)
                and polish_chi2 < best_chi2
            ):

                best_chi2 = float(
                    polish_chi2
                )

                best_entry = polish_entry
                best_candidate_dir = polish_dir

                polish_record[
                    "accepted"
                ] = True

            print(
                "[H1_polish] END "
                f"status={polish_status} "
                f"chi2_before={chi2_before_polish:.16g} "
                f"chi2_after={polish_chi2} "
                f"improvement="
                f"{polish_record.get('improvement')} "
                f"accepted={polish_record['accepted']} "
                f"optimality="
                f"{polish_optimizer.get('optimizer_optimality')}",
                flush=True,
            )

        nested_tolerance = float(
            ms_cfg.get(
                "nested_chi2_tolerance",
                1.0e-6,
            )
        )

        if (
            best_chi2
            > h0_chi2
            + nested_tolerance
        ):
            raise RuntimeError(
                "H1 multistart failed nested-model sanity check: "
                f"best_H1_chi2={best_chi2}, "
                f"H0_chi2={h0_chi2}, "
                f"tolerance={nested_tolerance}"
            )

        multistart_metadata = {
            "version": "H1_PARALLAX_GRID_MULTISTART_PATCH_V1",
            "n_candidates": int(
                len(candidates)
            ),
            "grid_fractions": [
                float(x)
                for x in fractions
            ],
            "piEN_half_width": float(
                width_n
            ),
            "piEE_half_width": float(
                width_e
            ),
            "include_auto_center": bool(
                include_auto_center
            ),
            "include_exact_H0_center": bool(
                include_exact_center
            ),
            "h0_chi2": float(
                h0_chi2
            ),
            "selected_index": int(
                best_candidate["index"]
            ),
            "selected_id": str(
                best_candidate["id"]
            ),
            "selected_piEN": float(
                best_candidate["piEN"]
            ),
            "selected_piEE": float(
                best_candidate["piEE"]
            ),
            "selected_flux_mode": str(
                best_candidate["flux_mode"]
            ),
            "selected_chi2": float(
                best_chi2
            ),
            "lrt_T_selected": float(
                h0_chi2
                - best_chi2
            ),
            "candidates": candidate_records,
            "polish": polish_record,
        }

        # ------------------------------------------------------------
        # Add metadata to the in-memory winning fit.
        # ------------------------------------------------------------

        best_entry = dict(
            best_entry
        )

        best_entry[
            "h1_multistart"
        ] = multistart_metadata

        best_entry[
            "h1_multistart_n"
        ] = int(
            len(candidates)
        )

        best_entry[
            "h1_multistart_selected"
        ] = str(
            best_candidate["id"]
        )

        best_entry[
            "h1_multistart_selected_piEN"
        ] = float(
            best_candidate["piEN"]
        )

        best_entry[
            "h1_multistart_selected_piEE"
        ] = float(
            best_candidate["piEE"]
        )

        best_entry[
            "h1_multistart_selected_chi2"
        ] = float(
            best_chi2
        )

        winner_fit_rr = best_entry.get(
            "fit_rr",
            None,
        )

        winner_fit_results = getattr(
            winner_fit_rr,
            "fit_results",
            None,
        )

        if isinstance(
            winner_fit_results,
            dict,
        ):

            winner_fit_results[
                "h1_multistart_n"
            ] = int(
                len(candidates)
            )

            winner_fit_results[
                "h1_multistart_selected"
            ] = str(
                best_candidate["id"]
            )

            winner_fit_results[
                "h1_multistart_selected_piEN"
            ] = float(
                best_candidate["piEN"]
            )

            winner_fit_results[
                "h1_multistart_selected_piEE"
            ] = float(
                best_candidate["piEE"]
            )

            winner_fit_results[
                "h1_multistart_selected_chi2"
            ] = float(
                best_chi2
            )

            winner_fit_results[
                "h1_multistart_candidates"
            ] = candidate_records

        # ------------------------------------------------------------
        # Persist a human-readable diagnostic sidecar.
        # ------------------------------------------------------------

        import json as _json

        diagnostic_path = (
            base_fit_path
            / "H1_multistart_diagnostics.json"
        )

        diagnostic_path.write_text(
            _json.dumps(
                multistart_metadata,
                indent=2,
            )
            + "\n"
        )

        # ------------------------------------------------------------
        # Preserve the traditional H1 .npy location.
        #
        # Candidate fits live under _H1_multistart/<candidate>/.
        # Copy the selected candidate's saved .npy file(s) back to the
        # normal event fit directory so existing analysis code still works.
        # ------------------------------------------------------------

        import shutil as _shutil

        copied_files = []

        for src in sorted(
            best_candidate_dir.glob(
                "*.npy"
            )
        ):

            dst = (
                base_fit_path
                / src.name
            )

            _shutil.copy2(
                src,
                dst,
            )

            copied_files.append(
                dst
            )

            # Add compact multistart metadata to the standard winning npy.
            try:

                saved = np.load(
                    dst,
                    allow_pickle=True,
                ).item()

                saved[
                    "h1_multistart_n"
                ] = int(
                    len(candidates)
                )

                saved[
                    "h1_multistart_selected"
                ] = str(
                    best_candidate["id"]
                )

                saved[
                    "h1_multistart_selected_piEN"
                ] = float(
                    best_candidate["piEN"]
                )

                saved[
                    "h1_multistart_selected_piEE"
                ] = float(
                    best_candidate["piEE"]
                )

                saved[
                    "h1_multistart_selected_chi2"
                ] = float(
                    best_chi2
                )

                saved[
                    "h1_multistart_candidates"
                ] = candidate_records

                np.save(
                    dst,
                    saved,
                )

            except Exception as error:

                print(
                    "[H1_multistart] WARNING: could not append metadata "
                    f"to {dst}: {repr(error)}",
                    flush=True,
                )

        print(
            "[H1_multistart] SELECTED "
            f"id={best_candidate['id']} "
            f"index={best_candidate['index']} "
            f"piEN={best_candidate['piEN']:.16g} "
            f"piEE={best_candidate['piEE']:.16g} "
            f"flux_mode={best_candidate['flux_mode']} "
            f"chi2={best_chi2:.16g} "
            f"H0_chi2={h0_chi2:.16g} "
            f"T={h0_chi2-best_chi2:.16g}",
            flush=True,
        )

        print(
            "[H1_multistart] standard winner files copied: "
            f"{[str(p) for p in copied_files]}",
            flush=True,
        )

        results = dict(
            existing_results
        )

        results[
            alternative_key
        ] = best_entry

        return results

    _ORIGINAL_RUN_MULTIPLE_NAMED_FITS_H1_MULTISTART = (
        current_multi
    )

    frr.run_multiple_named_fits = (
        run_multiple_named_fits_h1_multistart
    )

    _H1_PARALLAX_GRID_MULTISTART_PATCH_INSTALLED = True

    print(
        "[H1_multistart] runtime grid patch installed",
        flush=True,
    )


install_h1_parallax_grid_multistart_runtime_patch()

def _noise_realizations_config():
    """
    Read and validate the optional top-level noise_realizations section.

    If absent or disabled, legacy one-task-per-event behavior is preserved.
    """

    raw = CONFIG.get(
        "noise_realizations",
        {},
    )

    if raw is None:
        raw = {}

    if not isinstance(
        raw,
        dict,
    ):
        raise TypeError(
            "noise_realizations must be a JSON object."
        )

    enabled = bool(
        raw.get(
            "enabled",
            False,
        )
    )

    if not enabled:
        return {
            "enabled": False,
            "n_realizations": 1,
            "base_seed": 0,
            "truth_cases": [],
            "paired_noise": True,
        }

    # Initial implementation is deliberately Rubin-only because the
    # controlled random draw is apply_roman_rubin_photometry().
    if USE_ROMAN:
        raise RuntimeError(
            "The current noise-realization implementation is "
            "intentionally Rubin-only. "
            "Set observing.use_roman=false for this experiment."
        )

    n_realizations = int(
        raw.get(
            "n_realizations",
            1,
        )
    )

    if n_realizations <= 0:
        raise ValueError(
            "noise_realizations.n_realizations must be > 0."
        )

    base_seed = int(
        raw.get(
            "base_seed",
            20260901,
        )
    )

    raw_truth_cases = raw.get(
        "truth_cases",
        [
            "H0",
            "H1",
        ],
    )

    if not isinstance(
        raw_truth_cases,
        (
            list,
            tuple,
        ),
    ):
        raise TypeError(
            "noise_realizations.truth_cases must be a list."
        )

    truth_cases = []

    for value in raw_truth_cases:

        value = str(
            value
        ).strip().upper()

        if value not in {
            "H0",
            "H1",
        }:
            raise ValueError(
                "noise_realizations.truth_cases accepts "
                "only H0/H1; "
                f"got {value!r}."
            )

        if value not in truth_cases:
            truth_cases.append(
                value
            )

    if not truth_cases:
        raise ValueError(
            "noise_realizations.truth_cases cannot be empty."
        )

    return {
        "enabled": True,
        "n_realizations":
            n_realizations,
        "base_seed":
            base_seed,
        "truth_cases":
            truth_cases,
        "paired_noise":
            bool(
                raw.get(
                    "paired_noise",
                    True,
                )
            ),
    }


def _noise_realization_seed(
    base_seed,
    catalog_row,
    realization_id,
    truth_case,
    paired_noise,
):
    """
    Construct a deterministic photometric-noise seed.

    This seed depends on physical-event identity and realization number,
    but NOT on worker/chunk/SLURM ordering.

    If paired_noise=True, truth_case is deliberately excluded so that
    H0 and H1 receive common random numbers.
    """

    entropy = [
        int(base_seed),
        int(catalog_row),
        int(realization_id),
    ]

    if not paired_noise:
        entropy.append(
            0
            if str(
                truth_case
            ).upper() == "H0"
            else 1
        )

    seed_sequence = (
        np.random.SeedSequence(
            entropy
        )
    )

    return int(
        seed_sequence.generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )


def _noise_realization_tasks_per_event():
    """
    Number of logical tasks generated per physical event.
    """

    config = (
        _noise_realizations_config()
    )

    if not config["enabled"]:
        return 1

    return (
        int(
            config[
                "n_realizations"
            ]
        )
        * len(
            config[
                "truth_cases"
            ]
        )
    )


def build_tasks(
    prepared_catalog,
):
    """
    Wrapper around the original one-task-per-event builder.

    Legacy mode
    -----------
    If noise_realizations.enabled is false/missing:
        exactly the old task list is returned.

    Noise-MC mode
    -------------
    Each physical event is expanded into

        n_realizations x truth_cases

    logical datasets.

    Important:
        simulation_seed is NOT changed between realizations.
        noise_seed is the quantity that changes.

    H0 truth:
        piEN = 0
        piEE = 0

    H1 truth:
        piEN/piEE = catalog values

    truth_parallax=True for BOTH H0 and H1 so the same truth-generator
    code path is used. H0 is simply the nested point pi_E=(0,0).
    """

    base_tasks = (
        _build_tasks_single_realization_legacy(
            prepared_catalog
        )
    )

    config = (
        _noise_realizations_config()
    )

    if not config["enabled"]:
        return base_tasks

    expanded = []

    for base_task in base_tasks:

        catalog_row = int(
            base_task[
                "catalog_row"
            ]
        )

        catalog_piEN = float(
            base_task[
                "piEN"
            ]
        )

        catalog_piEE = float(
            base_task[
                "piEE"
            ]
        )

        catalog_piE = float(
            base_task.get(
                "piE",
                np.hypot(
                    catalog_piEN,
                    catalog_piEE,
                ),
            )
        )

        # This is intentionally identical across every noise realization
        # of the same physical event.
        simulation_seed = int(
            base_task[
                "simulation_seed"
            ]
        )

        for realization_id in range(
            config[
                "n_realizations"
            ]
        ):

            for truth_case in config[
                "truth_cases"
            ]:

                noise_seed = (
                    _noise_realization_seed(
                        base_seed=
                            config[
                                "base_seed"
                            ],
                        catalog_row=
                            catalog_row,
                        realization_id=
                            realization_id,
                        truth_case=
                            truth_case,
                        paired_noise=
                            config[
                                "paired_noise"
                            ],
                    )
                )

                if truth_case == "H0":

                    effective_piEN = 0.0
                    effective_piEE = 0.0
                    effective_piE = 0.0

                else:

                    effective_piEN = (
                        catalog_piEN
                    )

                    effective_piEE = (
                        catalog_piEE
                    )

                    effective_piE = float(
                        np.hypot(
                            effective_piEN,
                            effective_piEE,
                        )
                    )

                task = dict(
                    base_task
                )

                realization_key = (
                    f"event_{catalog_row:07d}_"
                    f"{truth_case.lower()}_"
                    f"r{realization_id:06d}"
                )

                task.update(
                    {
                        # ------------------------------
                        # Monte-Carlo identity
                        # ------------------------------
                        "truth_case":
                            truth_case,

                        # Keep parallax machinery active in both cases.
                        "truth_parallax":
                            True,

                        "noise_realization_id":
                            int(
                                realization_id
                            ),

                        "noise_seed":
                            int(
                                noise_seed
                            ),

                        "noise_base_seed":
                            int(
                                config[
                                    "base_seed"
                                ]
                            ),

                        "paired_noise":
                            bool(
                                config[
                                    "paired_noise"
                                ]
                            ),

                        "realization_key":
                            realization_key,

                        # ------------------------------
                        # Original catalog parallax
                        # ------------------------------
                        "catalog_piE":
                            catalog_piE,

                        "catalog_piEN":
                            catalog_piEN,

                        "catalog_piEE":
                            catalog_piEE,

                        # ------------------------------
                        # Effective truth simulated
                        # ------------------------------
                        "piE":
                            effective_piE,

                        "piEN":
                            effective_piEN,

                        "piEE":
                            effective_piEE,

                        "effective_truth_piE":
                            effective_piE,

                        "effective_truth_piEN":
                            effective_piEN,

                        "effective_truth_piEE":
                            effective_piEE,

                        # Explicitly preserve physical-event seed.
                        "simulation_seed":
                            simulation_seed,
                    }
                )

                expanded.append(
                    task
                )

    return expanded


def _apply_roman_rubin_photometry_noise_realization(
    *args,
    **kwargs,
):
    """
    Call the ORIGINAL Rubin photometry routine with an optional
    temporary NumPy RNG seed.

    The old NumPy RNG state is restored immediately after photometry.

    Therefore noise_seed affects only this photometric-noise block,
    rather than changing the random state of the full simulation/fit.
    """

    if (
        _ORIGINAL_APPLY_ROMAN_RUBIN_PHOTOMETRY
        is None
    ):
        raise RuntimeError(
            "Noise runtime patch was called before storing "
            "the original apply_roman_rubin_photometry function."
        )

    if (
        _RUNTIME_PHOTOMETRY_NOISE_SEED
        is None
    ):
        return (
            _ORIGINAL_APPLY_ROMAN_RUBIN_PHOTOMETRY(
                *args,
                **kwargs,
            )
        )

    seed = int(
        _RUNTIME_PHOTOMETRY_NOISE_SEED
    )

    if (
        seed < 0
        or seed > 2**32 - 1
    ):
        raise ValueError(
            f"Invalid photometry noise seed: {seed}"
        )

    previous_state = (
        np.random.get_state()
    )

    try:

        np.random.seed(
            seed
        )

        return (
            _ORIGINAL_APPLY_ROMAN_RUBIN_PHOTOMETRY(
                *args,
                **kwargs,
            )
        )

    finally:

        np.random.set_state(
            previous_state
        )


def install_noise_realization_runtime_patch():
    """
    Patch the imported functions_roman_rubin module in this worker only.

    No source file is overwritten.
    """

    global _ORIGINAL_APPLY_ROMAN_RUBIN_PHOTOMETRY
    global _NOISE_RUNTIME_PATCH_INSTALLED

    if _NOISE_RUNTIME_PATCH_INSTALLED:
        return

    if not hasattr(
        frr,
        "apply_roman_rubin_photometry",
    ):
        raise RuntimeError(
            "functions_roman_rubin does not expose "
            "apply_roman_rubin_photometry."
        )

    _ORIGINAL_APPLY_ROMAN_RUBIN_PHOTOMETRY = (
        frr.apply_roman_rubin_photometry
    )

    frr.apply_roman_rubin_photometry = (
        _apply_roman_rubin_photometry_noise_realization
    )

    _NOISE_RUNTIME_PATCH_INSTALLED = True


def set_runtime_noise_realization_context(
    task,
):
    """
    Select the photometric-noise realization for the current event.

    Legacy tasks have no noise_seed and therefore reproduce old behavior.
    """

    global _RUNTIME_PHOTOMETRY_NOISE_SEED

    seed = task.get(
        "noise_seed",
        None,
    )

    if seed in (
        None,
        "",
    ):
        _RUNTIME_PHOTOMETRY_NOISE_SEED = None

    else:
        _RUNTIME_PHOTOMETRY_NOISE_SEED = int(
            seed
        )



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


def validate_t0_first_maf_timestamp(base_row, context=""):
    """
    Enforce the only allowed t0 convention for this runner.

    Sedighe's catalog t0 is relative to the first OpSim/MAF timestamp
    read for the selected field/source:

        t0_jd = t0_reference_jd + t0_catalog_days

    Any caller that tries to use the provisional/fixed-origin t0 must fail.
    """

    prefix = f"[{context}] " if context else ""

    origin = str(
        base_row.get("t0_origin", "")
    )

    if origin != "first_maf_timestamp":
        raise RuntimeError(
            prefix
            + "t0_jd todavía no fue resuelto desde el primer timestamp "
            "OpSim/MAF del campo. Llamá primero a "
            "apply_t0_from_first_maf_timestamp(base_row, config). "
            f"t0_origin actual={origin!r}."
        )

    t0_reference_jd = float(base_row.get("t0_reference_jd", np.nan))
    t0_catalog_days = float(base_row.get("t0_catalog_days", np.nan))
    t0_jd = float(base_row.get("t0_jd", np.nan))

    if not np.isfinite(t0_reference_jd):
        raise RuntimeError(
            prefix + "t0_reference_jd no es finito."
        )

    if not np.isfinite(t0_catalog_days):
        raise RuntimeError(
            prefix + "t0_catalog_days no es finito."
        )

    if t0_catalog_days < 0.0:
        raise RuntimeError(
            prefix
            + "t0_catalog_days es negativo. En este runner debe estar "
            "medido como días desde el primer timestamp OpSim/MAF. "
            f"Valor={t0_catalog_days}."
        )

    if not np.isfinite(t0_jd):
        raise RuntimeError(
            prefix + "t0_jd no es finito."
        )

    expected = t0_reference_jd + t0_catalog_days

    if not np.isclose(t0_jd, expected, rtol=0.0, atol=1.0e-7):
        raise RuntimeError(
            prefix
            + "Convención temporal inválida para t0. Debe cumplirse "
            "t0_jd = t0_reference_jd + t0_catalog_days. "
            f"t0_jd={t0_jd:.9f}, "
            f"t0_reference_jd={t0_reference_jd:.9f}, "
            f"t0_catalog_days={t0_catalog_days:.9f}, "
            f"expected={expected:.9f}."
        )

    return True


def first_timestamps_from_pylima_event(pyLIMA_model):
    """
    Return first/last JD among all non-empty telescope lightcurves,
    globally and per retained telescope, after event construction and
    photometric filtering.
    """

    out = {
        "event_first_jd_after_filters": np.nan,
        "event_last_jd_after_filters": np.nan,
        "event_first_band_after_filters": "",
        "event_n_points_after_filters": 0,
    }

    first_candidates = []
    last_candidates = []
    total_points = 0

    for telescope in pyLIMA_model.event.telescopes:
        if telescope.lightcurve is None:
            continue
        if len(telescope.lightcurve) == 0:
            continue
        if "time" not in telescope.lightcurve.colnames:
            continue

        band = str(telescope.name)

        time = np.asarray(
            getattr(
                telescope.lightcurve["time"],
                "value",
                telescope.lightcurve["time"],
            ),
            dtype=float,
        )

        time = time[np.isfinite(time)]

        if len(time) == 0:
            continue

        t_min = float(np.min(time))
        t_max = float(np.max(time))
        n_band = int(len(time))

        out[f"event_first_jd_after_filters_{band}"] = t_min
        out[f"event_last_jd_after_filters_{band}"] = t_max
        out[f"event_n_points_after_filters_{band}"] = n_band

        first_candidates.append((t_min, band))
        last_candidates.append(t_max)
        total_points += n_band

    if len(first_candidates) == 0:
        raise RuntimeError(
            "No hay timestamps finitos en el evento pyLIMA simulado."
        )

    first_candidates = sorted(first_candidates, key=lambda x: x[0])

    out["event_first_jd_after_filters"] = float(first_candidates[0][0])
    out["event_first_band_after_filters"] = str(first_candidates[0][1])
    out["event_last_jd_after_filters"] = float(np.max(last_candidates))
    out["event_n_points_after_filters"] = int(total_points)

    return out


def first_timestamp_from_pylima_event(pyLIMA_model):
    """
    Backward-compatible scalar helper.
    """

    return float(
        first_timestamps_from_pylima_event(pyLIMA_model)[
            "event_first_jd_after_filters"
        ]
    )


def validate_simulated_event_uses_t0_reference(
    result,
    base_row,
    strict_equality=False,
    atol_days=1.0e-7,
):
    """
    Verify the t0 convention and record how the retained pyLIMA event starts.

    Strict, fatal check:
        t0_jd = t0_reference_jd + t0_catalog_days

    Non-fatal diagnostic:
        the first retained point after band availability and photometric
        filtering can be later than t0_reference_jd.
    """

    validate_t0_first_maf_timestamp(
        base_row,
        context="validate_simulated_event_uses_t0_reference",
    )

    diagnostic = {
        "t0_reference_check_status": "not_checked",
        "t0_reference_jd": float(base_row.get("t0_reference_jd", np.nan)),
        "t0_catalog_days": float(base_row.get("t0_catalog_days", np.nan)),
        "t0_jd": float(base_row.get("t0_jd", np.nan)),
        "event_first_jd_after_filters": np.nan,
        "event_first_minus_t0_reference_days": np.nan,
        "t0_reference_check_message": "",
    }

    if not isinstance(result, dict):
        diagnostic["t0_reference_check_status"] = "non_dict_result"
        return diagnostic

    pyLIMA_model = result.get("pyLIMAmodel_true", None)

    if pyLIMA_model is None:
        diagnostic["t0_reference_check_status"] = "no_true_model"
        return diagnostic

    event_time_info = first_timestamps_from_pylima_event(
        pyLIMA_model
    )

    diagnostic.update(event_time_info)

    reference_jd = float(base_row["t0_reference_jd"])
    event_first_jd = float(event_time_info["event_first_jd_after_filters"])
    delta_days = float(event_first_jd - reference_jd)

    diagnostic["event_first_minus_t0_reference_days"] = delta_days

    if np.isclose(
        event_first_jd,
        reference_jd,
        rtol=0.0,
        atol=atol_days,
    ):
        diagnostic["t0_reference_check_status"] = "ok_exact"
        return diagnostic

    if event_first_jd < reference_jd - atol_days:
        message = (
            "El primer timestamp retenido en pyLIMA es anterior al "
            "timestamp MAF usado como origen de t0. Esto sí es inconsistente. "
            f"event_first_jd_after_filters={event_first_jd:.9f}, "
            f"t0_reference_jd={reference_jd:.9f}, "
            f"delta_days={delta_days:.9f}."
        )

        diagnostic["t0_reference_check_status"] = (
            "error_event_starts_before_reference"
        )
        diagnostic["t0_reference_check_message"] = message

        raise RuntimeError(message)

    message = (
        "El primer timestamp retenido en pyLIMA es posterior al timestamp MAF "
        "usado como origen de t0. Esto puede ocurrir si el filtro fotométrico "
        "m5/5sigma descarta los primeros puntos visibles. "
        f"event_first_jd_after_filters={event_first_jd:.9f}, "
        f"t0_reference_jd={reference_jd:.9f}, "
        f"delta_days={delta_days:.9f}, "
        f"first_band={event_time_info.get('event_first_band_after_filters', '')}."
    )

    diagnostic["t0_reference_check_status"] = "warning_event_starts_later"
    diagnostic["t0_reference_check_message"] = message

    if strict_equality:
        raise RuntimeError(message)

    print("[warning]", message)

    return diagnostic


def build_event_fit_window(base_row):
    """
    Return the absolute-JD interval applied only to the fit arrays.

    For half_width_tE = 3.5:
        [t0 - 3.5*tE, t0 + 3.5*tE]

    If the option is disabled, return None and use the complete light curve.
    """

    validate_t0_first_maf_timestamp(
        base_row,
        context="build_event_fit_window",
    )

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

    validate_t0_first_maf_timestamp(
        base_row,
        context="fixed_param_samplers",
    )

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
    validate_t0_first_maf_timestamp(
        base_row,
        context="task_metadata",
    )

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
        "t0_reference_raw_first_maf_jd": float(
            base_row.get("t0_reference_raw_first_maf_jd", np.nan)
        ),
        "t0_reference_first_filter": str(
            base_row.get("t0_reference_first_filter", "")
        ),
        "t0_reference_visible_bands": str(
            base_row.get("t0_reference_visible_bands", "")
        ),
        "t0_reference_band_availability_mode": str(
            base_row.get("t0_reference_band_availability_mode", "")
        ),
        "t0_reference_n_obs_raw": float(
            base_row.get("t0_reference_n_obs_raw", np.nan)
        ),
        "t0_reference_n_obs_visible_bands": float(
            base_row.get("t0_reference_n_obs_visible_bands", np.nan)
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
        "DetectionFlag_u": float(base_row.get("DetectionFlag_u", np.nan)),
        "DetectionFlag_g": float(base_row.get("DetectionFlag_g", np.nan)),
        "DetectionFlag_r": float(base_row.get("DetectionFlag_r", np.nan)),
        "DetectionFlag_i": float(base_row.get("DetectionFlag_i", np.nan)),
        "DetectionFlag_z": float(base_row.get("DetectionFlag_z", np.nan)),
        "DetectionFlag_y": float(base_row.get("DetectionFlag_y", np.nan)),
        "catalog_visible_filter_count": int(
            base_row["catalog_visible_filter_count"]
        ),
        "catalog_available_bands": str(
            base_row.get("catalog_available_bands", "")
        ),
        "catalog_band_availability_mode": BAND_AVAILABILITY_MODE,
        "catalog_zero_fraction_bands": ",".join(
            band
            for band, column in zip(
                CATALOG_BANDS,
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
        "apply_detection_criteria": APPLY_DETECTION_CRITERIA,
        "apply_photometric_filter": APPLY_PHOTOMETRIC_FILTER,
        "fit_bounds": json.dumps(FIT_BOUNDS_NOPIE, default=str),
        "fit_initial_guess": (
            FIT_INITIAL_GUESS
            if isinstance(FIT_INITIAL_GUESS, str)
            else (
                "random"
                if FIT_INITIAL_GUESS is None
                else json.dumps(
                    FIT_INITIAL_GUESS,
                    default=str,
                )
            )
        ),
        "fit_optimizer_options": (
            "pyLIMA_default"
            if FIT_OPTIMIZER_OPTIONS is None
            else json.dumps(
                FIT_OPTIMIZER_OPTIONS,
                sort_keys=True,
                default=str,
            )
        ),
        "run_multiple_fits": bool(RUN_MULTIPLE_FITS),
        "primary_fit": PRIMARY_FIT if PRIMARY_FIT is not None else "",
        "fit_specs": json.dumps(FIT_SPECS, default=str),
        "lrt_config": json.dumps(LRT_CONFIG, default=str),
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


def _get_attr_or_key(obj, key, default=None):
    """
    Access key/attribute from dict-like objects or normal Python objects.
    """

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    try:
        return obj[key]
    except Exception:
        pass

    return getattr(obj, key, default)


def _extract_fit_results_object(result):
    """
    Return the pyLIMA fit_results object/dict from the sim_fit output.

    In the current pipeline the most common structure is
        result["fit_rr"].fit_results
    but this function also accepts a few direct aliases for compatibility.
    """

    if not isinstance(result, dict):
        return None

    for key in [
        "fit_results",
        "fitter_results",
        "fit_result",
        "pyLIMAfit_results",
        "fit_output",
    ]:
        value = result.get(key, None)
        if value is not None:
            return value

    for key in ["fit_rr", "fit_roman", "fitter", "fit"]:
        value = result.get(key, None)
        fit_results = _get_attr_or_key(value, "fit_results", None)
        if fit_results is not None:
            return fit_results

    return None


def _extract_fit_model_object(result):
    """
    Return the fitted pyLIMA model object, if available.
    """

    if not isinstance(result, dict):
        return None

    for key in [
        "pyLIMAmodel_rr",
        "pyLIMAmodel_fit",
        "fit_model",
        "pyLIMAmodel",
    ]:
        value = result.get(key, None)
        if value is not None:
            return value

    return None


def _parameter_names_from_model_dictionnary(model_obj):
    """
    Infer parameter names in vector/covariance order from pyLIMA's
    model_dictionnary.
    """

    if model_obj is None:
        return None

    model_dict = getattr(model_obj, "model_dictionnary", None)

    if model_dict is None:
        return None

    if not isinstance(model_dict, dict):
        return None

    pairs = []

    for name, value in model_dict.items():
        index = None

        if isinstance(value, (int, np.integer)):
            index = int(value)
        elif isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
            try:
                index = int(value[0])
            except Exception:
                index = None
        else:
            try:
                index = int(value)
            except Exception:
                index = None

        if index is not None:
            pairs.append((index, str(name)))

    if len(pairs) == 0:
        return None

    pairs = sorted(pairs, key=lambda item: item[0])
    return [name for _, name in pairs]


def _parameter_names_from_fit_results(fit_results):
    """
    Try to infer fitted-parameter names directly from fit_results.
    """

    if fit_results is None:
        return None

    for key in [
        "fit_parameters",
        "model_parameters",
        "parameters",
        "fit_parameter_names",
        "parameter_names",
    ]:
        value = _get_attr_or_key(fit_results, key, None)

        if value is None:
            continue

        if isinstance(value, dict):
            return [str(name) for name in value.keys()]

        if isinstance(value, (list, tuple, np.ndarray)):
            return [str(name) for name in value]

    return None


def _extract_fit_parameter_names(result, fit_results):
    """
    Determine the parameter order used by best_model and covariance_matrix.

    Preferred source is pyLIMA_model.model_dictionnary.  If unavailable,
    fall back to fit_results metadata.  If both are unavailable, use the
    known nonlinear order for an FSPL fit without parallax.
    """

    model_obj = _extract_fit_model_object(result)
    names = _parameter_names_from_model_dictionnary(model_obj)

    if names is not None:
        return names, "pyLIMAmodel_rr.model_dictionnary"

    names = _parameter_names_from_fit_results(fit_results)

    if names is not None:
        return names, "fit_results"

    return ["t0", "u0", "tE", "rho"], "fallback_FSPL_no_parallax"


def _extract_best_model_vector(fit_results):
    """
    Return fit_results['best_model'] as a float vector, if available.
    """

    best_model = _get_attr_or_key(fit_results, "best_model", None)

    if best_model is None:
        return None

    try:
        return np.asarray(best_model, dtype=float)
    except Exception:
        return None


def _extract_covariance_matrix(fit_results):
    """
    Return fit_results['covariance_matrix'] as a float array, if available.
    """

    covariance = _get_attr_or_key(fit_results, "covariance_matrix", None)

    if covariance is None:
        return None

    try:
        return np.asarray(covariance, dtype=float)
    except Exception:
        return None


def extract_covariance_uncertainties(result, base_row=None):
    """
    Extract covariance-based uncertainties from pyLIMA fit_results.

    For a fitted parameter p_i, the 1-sigma uncertainty is
        sigma_i = sqrt(C_ii),
    where C is fit_results['covariance_matrix'].

    The main quantity used for the finite-source analysis is
        sigma_rho / rho_fit.
    The true-denominator version sigma_rho / rho_true is also stored for
    diagnostic purposes.
    """

    out = {
        "fit_results_available": False,
        "covariance_matrix_available": False,
        "covariance_matrix_shape": "",
        "covariance_parameter_order_source": "",
        "covariance_parameter_names": "",
        "rho_fit_from_best_model": np.nan,
        "rho_err_cov": np.nan,
        "sigma_rho_over_rho_fit": np.nan,
        "sigma_rho_over_rho_true": np.nan,
        "rho_covariance_index": np.nan,
        "rho_covariance_variance": np.nan,
        "t0_fit_from_best_model": np.nan,
        "u0_fit_from_best_model": np.nan,
        "tE_fit_from_best_model": np.nan,
        "t0_err_cov": np.nan,
        "u0_err_cov": np.nan,
        "tE_err_cov": np.nan,
    }

    fit_results = _extract_fit_results_object(result)

    if fit_results is None:
        return out

    out["fit_results_available"] = True

    covariance = _extract_covariance_matrix(fit_results)

    if covariance is None:
        return out

    out["covariance_matrix_available"] = True
    out["covariance_matrix_shape"] = str(tuple(covariance.shape))

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        return out

    names, names_source = _extract_fit_parameter_names(
        result,
        fit_results,
    )

    names = [str(name) for name in names]

    out["covariance_parameter_order_source"] = str(names_source)
    out["covariance_parameter_names"] = ",".join(names)

    best_model = _extract_best_model_vector(fit_results)

    def _index_of(parameter_name):
        if parameter_name in names:
            return names.index(parameter_name)
        return None

    for parameter_name in ["t0", "u0", "tE", "rho"]:
        index = _index_of(parameter_name)

        if index is None:
            continue

        if index < 0 or index >= covariance.shape[0]:
            continue

        variance = float(covariance[index, index])

        if not np.isfinite(variance) or variance < 0.0:
            continue

        sigma = float(np.sqrt(variance))
        out[f"{parameter_name}_err_cov"] = sigma

        if best_model is not None and index < len(best_model):
            value = float(best_model[index])
            out[f"{parameter_name}_fit_from_best_model"] = value

        if parameter_name == "rho":
            out["rho_covariance_index"] = int(index)
            out["rho_covariance_variance"] = variance
            out["rho_err_cov"] = sigma

            rho_fit = out["rho_fit_from_best_model"]

            if np.isfinite(rho_fit) and rho_fit > 0.0:
                out["sigma_rho_over_rho_fit"] = sigma / rho_fit

            if base_row is not None:
                try:
                    rho_true = float(base_row.get("rho", np.nan))
                except Exception:
                    rho_true = np.nan

                if np.isfinite(rho_true) and rho_true > 0.0:
                    out["sigma_rho_over_rho_true"] = sigma / rho_true

    return out


# ============================================================================
# Worker
# ============================================================================
def apply_t0_from_first_maf_timestamp(base_row, config):
    """
    Resolve t0_jd using the allowed convention:

        t0_jd = first relevant OpSim/MAF timestamp + t0_catalog_days

    Relevant means:
        - same coordinate / Rubin pointing mode as the simulation;
        - only bands available for this catalog event.

    With sedighe.band_availability='detection_flag', the reference timestamp
    is computed using only filters with DetectionFlag_band == 1.

    The photometric m5/5-sigma filter is not used to define this reference,
    because the simulated magnification is needed before that filter exists.
    """

    import set_telescopes_pyLIMA as stp

    base_row = base_row.copy()

    if not bool(config.get("use_rubin", True)):
        raise RuntimeError(
            "t0_origin=first_maf_timestamp requiere use_rubin=True."
        )

    if "t0_catalog_days" not in base_row.index:
        raise RuntimeError(
            "Falta t0_catalog_days; no puedo construir t0_jd desde el "
            "primer timestamp OpSim/MAF."
        )

    t0_catalog_days = float(base_row["t0_catalog_days"])

    if not np.isfinite(t0_catalog_days):
        raise RuntimeError(
            f"t0_catalog_days no finito: {t0_catalog_days}."
        )

    if t0_catalog_days < 0.0:
        raise RuntimeError(
            "t0_catalog_days es negativo. Para esta convención debe ser "
            "un tiempo positivo medido desde el primer timestamp OpSim/MAF "
            f"del campo. Valor={t0_catalog_days}."
        )

    source_ra = float(base_row["ra"])
    source_dec = float(base_row["dec"])

    visible_bands = _visible_bands_from_row(
        base_row,
        mode=BAND_AVAILABILITY_MODE,
    )

    if len(visible_bands) == 0:
        raise RuntimeError(
            "No hay bandas visibles según "
            f"band_availability={BAND_AVAILABILITY_MODE}."
        )

    tel_kwargs = {
        "path_ephemerides": config["path_ephemerides"],
        "time_window": None,
        "use_roman": bool(config.get("use_roman", False)),
        "use_rubin": bool(config.get("use_rubin", True)),
        "Ra": source_ra,
        "Dec": source_dec,
        "rubin_pointing_mode": config.get("rubin_pointing_mode", "source"),
        "rubin_cache_cell_deg": config.get("rubin_cache_cell_deg", None),
    }

    tel_parameters = inspect.signature(stp.tel_roman_rubin).parameters

    optional_rubin_path_kwargs = {
        "opsim_db_path": config.get("rubin_opsim_db_path", None),
        "rubin_sim_data_dir": config.get("rubin_sim_data_dir", None),
        "rubin_throughputs_dir": config.get("rubin_throughputs_dir", None),
    }

    for key, value in optional_rubin_path_kwargs.items():
        if key in tel_parameters and value not in (None, ""):
            tel_kwargs[key] = value

    _, dataSlice, _ = stp.tel_roman_rubin(**tel_kwargs)

    if dataSlice is None or len(dataSlice) == 0:
        raise RuntimeError(
            "No hay observaciones Rubin/MAF para calcular el primer timestamp."
        )

    if "observationStartMJD" not in dataSlice.dtype.names:
        raise RuntimeError(
            "dataSlice no contiene la columna observationStartMJD; "
            "no puedo definir el origen local de t0."
        )

    filter_column = _find_dataslice_filter_column(dataSlice)

    mjd = np.asarray(
        dataSlice["observationStartMJD"],
        dtype=float,
    )

    jd = mjd + 2400000.5

    filters = np.asarray(
        [
            _canonical_maf_filter(value)
            for value in dataSlice[filter_column]
        ],
        dtype=str,
    )

    finite_mask = np.isfinite(jd)
    finite = jd[finite_mask]

    if len(finite) == 0:
        raise RuntimeError(
            "dataSlice no contiene observationStartMJD finitos."
        )

    raw_first_timestamp_jd = float(np.min(finite))

    visible_mask = (
        finite_mask
        & np.isin(filters, visible_bands)
    )

    visible_jd = jd[visible_mask]
    visible_filters = filters[visible_mask]

    if len(visible_jd) == 0:
        raise RuntimeError(
            "No hay observaciones MAF en las bandas visibles del catálogo. "
            f"visible_bands={visible_bands}, "
            f"available filters in dataSlice={sorted(set(filters.tolist()))}."
        )

    first_index = int(np.argmin(visible_jd))
    first_timestamp_jd = float(visible_jd[first_index])
    first_filter = str(visible_filters[first_index])

    old_t0_jd = float(base_row.get("t0_jd", np.nan))

    legacy_t0_jd = float(
        base_row.get("t0_jd_original_global_zero", np.nan)
    )

    if not np.isfinite(legacy_t0_jd) and np.isfinite(old_t0_jd):
        legacy_t0_jd = old_t0_jd

    new_t0_jd = first_timestamp_jd + t0_catalog_days

    base_row["t0_jd_original_global_zero"] = legacy_t0_jd
    base_row["t0_reference_raw_first_maf_jd"] = raw_first_timestamp_jd
    base_row["t0_reference_jd"] = first_timestamp_jd
    base_row["t0_reference_mjd"] = first_timestamp_jd - 2400000.5
    base_row["t0_reference_source"] = (
        "min(dataSlice.observationStartMJD in catalog-visible bands)"
    )
    base_row["t0_reference_band_availability_mode"] = BAND_AVAILABILITY_MODE
    base_row["t0_reference_visible_bands"] = ",".join(visible_bands)
    base_row["t0_reference_first_filter"] = first_filter
    base_row["t0_reference_n_obs_raw"] = int(len(finite))
    base_row["t0_reference_n_obs_visible_bands"] = int(len(visible_jd))
    base_row["t0_reference_n_obs"] = int(len(visible_jd))
    base_row["t0_origin"] = "first_maf_timestamp"
    base_row["t0_jd"] = new_t0_jd

    validate_t0_first_maf_timestamp(
        base_row,
        context="apply_t0_from_first_maf_timestamp",
    )

    print(
        "t0 convention applied: "
        f"t0_jd={new_t0_jd:.8f} = "
        f"first_visible_maf_jd={first_timestamp_jd:.8f} + "
        f"t0_catalog_days={t0_catalog_days:.8f}. "
        f"visible_bands={visible_bands}, first_filter={first_filter}, "
        f"raw_first_maf_jd={raw_first_timestamp_jd:.8f}"
    )

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
    event_tag = task.get(
        "realization_key",
        f"event_{global_i:07d}",
    )

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
        "needs_refit": False,
        "timeout_scope": "post_detectability_logical_task",
        "timeout_seconds": POST_DETECTABILITY_TIMEOUT_S,
        "timeout_elapsed_s": np.nan,
        "log_file": str(log_file),
        "model_dir": str(model_dir),
        "fit_dir": str(fit_dir),
        "results_dir": str(results_dir),
    }

    set_runtime_event_context(base_row)
    set_runtime_noise_realization_context(task)

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
                    "truth_parallax": bool(
                        task.get(
                            "truth_parallax",
                            config.get(
                                "truth_parallax",
                                True,
                            ),
                        )
                    ),
                    "rubin_pointing_mode": config["rubin_pointing_mode"],
                    "rubin_cache_cell_deg": config["rubin_cache_cell_deg"],
                    "return_data": True,
                    # Critical: the simulation is not temporally cropped.
                    "time_window": None,
                }

                sim_fit_parameters = inspect.signature(sim_fit).parameters

                if "initial_guess" in sim_fit_parameters:

                    sim_fit_kwargs["initial_guess"] = config.get(
                        "initial_guess",
                        None,
                    )

                elif config.get("initial_guess", None) is not None:

                    raise RuntimeError(
                        "El config define fit.initial_guess, "
                        "pero la función seleccionada "
                        f"{getattr(sim_fit, '__name__', sim_fit)!r} "
                        "no acepta initial_guess."
                    )

                if "optimizer_options" in sim_fit_parameters:

                    sim_fit_kwargs["optimizer_options"] = config.get(
                        "optimizer_options",
                        None,
                    )

                elif config.get("optimizer_options", None) is not None:

                    raise RuntimeError(
                        "El config define fit.optimizer_options, "
                        "pero la función seleccionada "
                        f"{getattr(sim_fit, '__name__', sim_fit)!r} "
                        "no acepta optimizer_options."
                    )

                optional_sim_fit_path_kwargs = {
                    "opsim_db_path": config.get("rubin_opsim_db_path", None),
                    "rubin_sim_data_dir": config.get("rubin_sim_data_dir", None),
                    "rubin_throughputs_dir": config.get("rubin_throughputs_dir", None),
                }

                for key, value in optional_sim_fit_path_kwargs.items():
                    if key in sim_fit_parameters and value not in (None, ""):
                        sim_fit_kwargs[key] = value

                if config.get("run_multiple_fits", False):
                    if "fit_specs" in sim_fit_parameters:
                        sim_fit_kwargs["fit_specs"] = config.get("fit_specs", None)
                    else:
                        raise RuntimeError(
                            "fit.run_multiple_fits=true, pero la función seleccionada "
                            "no acepta fit_specs. Verificá que el runner esté usando "
                            "functions_roman_rubin.sim_fit_multi_fits."
                        )

                    if "primary_fit" in sim_fit_parameters:
                        sim_fit_kwargs["primary_fit"] = config.get("primary_fit", None)

                    if "lrt_config" in sim_fit_parameters:
                        sim_fit_kwargs["lrt_config"] = config.get("lrt_config", None)

                    if "save_multi_fit_summary" in sim_fit_parameters:
                        sim_fit_kwargs["save_multi_fit_summary"] = True

                print("rubin_sim_data_dir =", config.get("rubin_sim_data_dir", ""))
                print("rubin_throughputs_dir =", config.get("rubin_throughputs_dir", ""))
                print("rubin_opsim_db_path =", config.get("rubin_opsim_db_path", ""))
                print("run_multiple_fits =", config.get("run_multiple_fits", False))
                print("initial_guess =", config.get("initial_guess", None))
                print(
                    "optimizer_options =",
                    config.get("optimizer_options", None),
                )
                print("primary_fit =", config.get("primary_fit", None))
                print("fit_specs keys =", list(config.get("fit_specs", {}).keys()) if isinstance(config.get("fit_specs", None), dict) else config.get("fit_specs", None))
                print("lrt_config =", config.get("lrt_config", None))

                if "apply_detection_criteria" in sim_fit_parameters:
                    sim_fit_kwargs["apply_detection_criteria"] = config[
                        "apply_detection_criteria"
                    ]

                if "apply_photometric_filter" in sim_fit_parameters:
                    sim_fit_kwargs["apply_photometric_filter"] = config[
                        "apply_photometric_filter"
                    ]
                else:
                    raise RuntimeError(
                        "El config solicita controlar apply_photometric_filter, "
                        "pero sim_fit no acepta ese argumento. Actualizá "
                        "functions_roman_rubin.sim_fit antes de correr."
                    )

                print("apply_detection_criteria =", sim_fit_kwargs.get("apply_detection_criteria"))
                print("apply_photometric_filter =", sim_fit_kwargs.get("apply_photometric_filter"))

                result = sim_fit(
                    simulation_seed,
                    config["system_type"],
                    **sim_fit_kwargs,
                )

                t0_reference_diagnostic = validate_simulated_event_uses_t0_reference(
                    result,
                    base_row,
                    strict_equality=False,
                )

                summary.update(t0_reference_diagnostic)

                covariance_uncertainty_info = extract_covariance_uncertainties(
                    result,
                    base_row=base_row,
                )

                summary.update(covariance_uncertainty_info)

                print("covariance uncertainty info")
                print(json.dumps(covariance_uncertainty_info, indent=2, default=str))

                if isinstance(result, dict):
                    sim_status = str(result.get("status", ""))
                    summary["sim_fit_status"] = sim_status

                    if sim_status == "fitted":
                        summary["status"] = "ok"
                    elif sim_status == "rejected":
                        summary["status"] = "rejected_pipeline"
                    else:
                        summary["status"] = "returned_dict"

                    # Multi-fit / likelihood-ratio diagnostics, when present.
                    summary["multi_fit_status"] = str(
                        result.get("multi_fit_status", "")
                    )
                    summary["primary_fit_key"] = str(
                        result.get("primary_fit_key", "")
                    )
                    summary["multi_fit_summary_path"] = str(
                        result.get("multi_fit_summary_path", "")
                    )

                    pipeline_timings = result.get(
                        "pipeline_timings",
                        {},
                    )

                    if isinstance(pipeline_timings, dict):
                        for key, value in pipeline_timings.items():

                            if isinstance(value, np.generic):
                                value = value.item()

                            summary[str(key)] = value

                    lrt_results = result.get("lrt_results", None)
                    if isinstance(lrt_results, dict):
                        for key, value in lrt_results.items():
                            try:
                                if isinstance(value, (np.generic,)):
                                    value = value.item()
                                summary[f"lrt_{key}"] = value
                            except Exception:
                                summary[f"lrt_{key}"] = repr(value)

                    multi_record = result.get("multi_fit_summary_record", None)
                    if isinstance(multi_record, dict):
                        for key, value in multi_record.items():
                            # Keep the run_summary compact but include the
                            # quantities needed for hypothesis testing.
                            if (
                                key.startswith("H0_")
                                or key.startswith("H1_")
                                or key.startswith("true_generator_")
                                or key in {
                                    "LRT",
                                    "LRT_from_nll",
                                    "p_value_LRT",
                                    "delta_k",
                                    "delta_chi2_H0_minus_H1",
                                    "oracle_LRT_true_vs_H0",
                                    "delta_chi2_H0_minus_true_generator",
                                }
                            ):
                                try:
                                    if isinstance(value, (np.generic,)):
                                        value = value.item()
                                    summary[f"multi_{key}"] = value
                                except Exception:
                                    summary[f"multi_{key}"] = repr(value)
                else:
                    summary["status"] = "ok"
                    summary["sim_fit_status"] = type(result).__name__

                # PREFIT_DETECTABILITY summary
                detectability_payload = dict(
                    _RUNTIME_LAST_DETECTABILITY
                )
                summary.update(
                    detectability_payload
                )
                if detectability_payload.get("detectability_enabled", False):
                    if detectability_payload.get("detectability_audit_only", False):
                        if detectability_payload.get("detectability_pass", False):
                            summary["status"] = "detectability_pass"
                        else:
                            summary["status"] = "detectability_fail"
                    elif not detectability_payload.get("detectability_pass", False):
                        summary["status"] = "rejected_detectability"

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
                    **covariance_uncertainty_info,
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

    except PostDetectabilityTimeout as error:
        summary["status"] = "fit_timeout"
        summary["sim_fit_status"] = "fit_timeout"
        summary["needs_refit"] = True
        summary["timeout_seconds"] = float(
            error.timeout_s
        )
        summary["timeout_elapsed_s"] = float(
            error.elapsed_s
        )

        summary.update(
            dict(_RUNTIME_LAST_DETECTABILITY)
        )

        fit_counts = dict(
            _RUNTIME_LAST_FIT_COUNTS
        )

        summary["fit_n_points_total"] = int(
            sum(fit_counts.values())
        )

        for band in [
            "W149", "u", "g", "r", "i", "z", "y"
        ]:
            summary[f"fit_n_points_{band}"] = int(
                fit_counts.get(band, 0)
            )

        with open(log_file, "a") as log:
            log.write("\n" + "=" * 80 + "\n")
            log.write("FIT TIMEOUT\n")
            log.write("=" * 80 + "\n")
            log.write(str(error) + "\n")
            log.write(
                "This realization is reproducible "
                "from catalog_row + seeds + frozen config.\n"
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

    if "status" in summary.columns:
        laggards = summary.loc[
            summary["status"].eq("fit_timeout")
        ].copy()
    else:
        laggards = summary.iloc[0:0].copy()

    laggards.to_csv(
        DIRS["logs"] / "laggards.csv",
        index=False,
    )

    laggards.to_parquet(
        DIRS["logs"] / "laggards.parquet",
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
    print("t0 origin policy:       first OpSim/MAF timestamp per field")
    print(f"T0_ZERO_JD legacy:      {T0_ZERO_JD} (diagnostic only; not used for t0)")
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
    print(f"Detection criteria:    {APPLY_DETECTION_CRITERIA}")
    print(f"Photometric filter:    {APPLY_PHOTOMETRIC_FILTER}")
    print(f"Fit bounds:            {FIT_BOUNDS_NOPIE}")
    print(
        "Initial guess:         "
        f"{FIT_INITIAL_GUESS if FIT_INITIAL_GUESS is not None else 'random'}"
    )
    print(
        "TRF options:           "
        f"{FIT_OPTIMIZER_OPTIONS if FIT_OPTIMIZER_OPTIONS is not None else 'pyLIMA defaults'}"
    )
    print(f"Truth parallax:        {TRUTH_PARALLAX}")
    print(f"Run multiple fits:     {RUN_MULTIPLE_FITS}")
    if RUN_MULTIPLE_FITS:
        print(f"Primary fit:           {PRIMARY_FIT}")
        print(f"Fit specs keys:        {list(FIT_SPECS.keys()) if isinstance(FIT_SPECS, dict) else FIT_SPECS}")
        print(f"LRT config:            {LRT_CONFIG}")
    if FIT_WINDOW_ENABLED:
        print(
            "Fit-only window:      "
            f"t0 +/- {FIT_WINDOW_HALF_WIDTH_TE:g} tE "
            f"(minimum {FIT_WINDOW_MINIMUM_TOTAL_POINTS} total points)"
        )
    else:
        print("Fit-only window:      disabled")
    print(f"Run name base:          {RUN_NAME_BASE}")
    print(f"Chunk output label:     {CHUNK_OUTPUT_LABEL if CHUNK_OUTPUT_LABEL else '(none)'}")
    print(f"Run directory:          {RUN_DIR}")
    print(f"Rubin sim data dir:     {RUBIN_SIM_DATA_DIR}")
    print(f"Rubin throughputs dir:  {RUBIN_THROUGHPUTS_DIR}")
    print(f"Rubin OpSim DB:         {RUBIN_OPSIM_DB_PATH}")
    print(f"Band availability:      {BAND_AVAILABILITY_MODE}")
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
        "--catalog-row-start",
        type=int,
        default=None,
        help=(
            "First raw LSSTMONTS row to read, zero-indexed and inclusive. "
            "Use this for SLURM/job-array chunks."
        ),
    )

    parser.add_argument(
        "--catalog-row-stop",
        type=int,
        default=None,
        help=(
            "Last raw LSSTMONTS row to read, zero-indexed and exclusive. "
            "Use this for SLURM/job-array chunks."
        ),
    )

    parser.add_argument(
        "--chunk-id",
        default=None,
        help=(
            "Optional chunk identifier used only for output naming and "
            "metadata. With --chunk-id N, outputs go under chunk_NNNNNN."
        ),
    )

    parser.add_argument(
        "--run-name-suffix",
        default=None,
        help=(
            "Optional output subdirectory label under the config run_name. "
            "This is useful for manual chunking without changing the config."
        ),
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Override the config and only prepare the catalog/tasks."
        ),
    )

    parser.add_argument(
        "--logical-task-start",
        type=int,
        default=None,
        help=(
            "Optional start index in the expanded logical task table. "
            "Used by the noise-realization SLURM array."
        ),
    )

    parser.add_argument(
        "--logical-task-stop",
        type=int,
        default=None,
        help=(
            "Optional exclusive stop index in the expanded logical task "
            "table. Used by the noise-realization SLURM array."
        ),
    )

    return parser


# ============================================================================
# Main
# ============================================================================


# ============================================================
# BOUNDED_FLUX_PROFILE_RUNTIME_PATCH_V1
#
# Experimental Hidden-Parallax-only variable projection.
#
# Activate with:
#
#   export HIDDEN_PARALLAX_BOUNDED_PROFILE=1
#
# When active:
#
#   TRF H0: 4 nonlinear parameters
#   TRF H1: 6 nonlinear parameters
#
# Telescope flux nuisance parameters are solved exactly, for each
# model evaluation and each telescope, under the same pyLIMA bounds:
#
#   0 <= fsource <= max(flux)
#   0 <= ftotal  <= max(flux)
#
# The model is
#
#   F = fsource * (A - 1) + ftotal
#
# This patch does NOT modify installed pyLIMA or shared fit_lc.py.
# ============================================================


def _bounded_flux_profile_enabled():

    import os as _os

    value = str(
        _os.environ.get(
            "HIDDEN_PARALLAX_BOUNDED_PROFILE",
            "0",
        )
    ).strip().lower()

    return value in {
        "1",
        "true",
        "yes",
        "on",
    }


_BOUNDED_FLUX_PROFILE_PATCH_INSTALLED = False
_ORIGINAL_TRF_INIT_BOUNDED_PROFILE = None
_ORIGINAL_DERIVE_TELESCOPE_FLUX_BOUNDED_PROFILE = None


def _bounded_profile_ftotal_2d(
    magnification,
    flux,
    err_flux,
):
    """
    Exact weighted box-constrained least-squares solution for

        flux = fsource * (A - 1) + ftotal

    with pyLIMA's current bounds

        0 <= fsource <= max(flux)
        0 <= ftotal  <= max(flux)

    Because there are only two linear variables, solve the convex
    quadratic exactly by checking:

        - unconstrained interior solution,
        - fsource = lower,
        - fsource = upper,
        - ftotal  = lower,
        - ftotal  = upper.

    Edge minimizers are clipped to their allowed interval, so corners
    are automatically included.
    """

    import numpy as _np

    A = _np.asarray(
        magnification,
        dtype=float,
    ).reshape(-1)

    y = _np.asarray(
        flux,
        dtype=float,
    ).reshape(-1)

    sigma = _np.asarray(
        err_flux,
        dtype=float,
    ).reshape(-1)

    if not (
        len(A)
        == len(y)
        == len(sigma)
    ):
        raise RuntimeError(
            "bounded profile: incompatible array lengths."
        )

    good = (
        _np.isfinite(A)
        & _np.isfinite(y)
        & _np.isfinite(sigma)
        & (sigma > 0.0)
    )

    if not _np.all(good):
        raise RuntimeError(
            "bounded profile: non-finite photometric data."
        )

    x = A - 1.0

    w = 1.0 / (
        sigma * sigma
    )

    fmax = float(
        _np.max(y)
    )

    if (
        not _np.isfinite(fmax)
        or fmax < 0.0
    ):
        raise RuntimeError(
            "bounded profile: invalid max(flux)."
        )

    # Weighted sufficient statistics.
    sw = float(
        _np.sum(w)
    )

    sx = float(
        _np.sum(w * x)
    )

    sxx = float(
        _np.sum(w * x * x)
    )

    sy = float(
        _np.sum(w * y)
    )

    sxy = float(
        _np.sum(w * x * y)
    )

    if sw <= 0.0:
        raise RuntimeError(
            "bounded profile: non-positive total weight."
        )

    candidates = []

    def _clip(value):

        return float(
            _np.clip(
                value,
                0.0,
                fmax,
            )
        )

    def _add(fs, ft, source):

        fs = _clip(fs)
        ft = _clip(ft)

        residual = (
            y
            - fs * x
            - ft
        ) / sigma

        chi2 = float(
            _np.dot(
                residual,
                residual,
            )
        )

        candidates.append(
            (
                chi2,
                fs,
                ft,
                source,
            )
        )

    # --------------------------------------------------------
    # Unconstrained weighted least-squares interior
    # --------------------------------------------------------

    determinant = (
        sxx * sw
        - sx * sx
    )

    determinant_scale = max(
        abs(
            sxx * sw
        ),
        abs(
            sx * sx
        ),
        1.0,
    )

    if abs(
        determinant
    ) > (
        1.0e-14
        * determinant_scale
    ):

        fs_free = (
            sxy * sw
            - sy * sx
        ) / determinant

        ft_free = (
            sxx * sy
            - sx * sxy
        ) / determinant

        if (
            0.0 <= fs_free <= fmax
            and
            0.0 <= ft_free <= fmax
        ):
            _add(
                fs_free,
                ft_free,
                "interior",
            )

    # --------------------------------------------------------
    # Edge fsource = 0
    # --------------------------------------------------------

    fs = 0.0

    ft = (
        sy - fs * sx
    ) / sw

    _add(
        fs,
        ft,
        "fsource_lower",
    )

    # --------------------------------------------------------
    # Edge fsource = fmax
    # --------------------------------------------------------

    fs = fmax

    ft = (
        sy - fs * sx
    ) / sw

    _add(
        fs,
        ft,
        "fsource_upper",
    )

    # --------------------------------------------------------
    # Edges ftotal = 0 / fmax
    # --------------------------------------------------------

    if sxx > 0.0:

        ft = 0.0

        fs = (
            sxy - ft * sx
        ) / sxx

        _add(
            fs,
            ft,
            "ftotal_lower",
        )

        ft = fmax

        fs = (
            sxy - ft * sx
        ) / sxx

        _add(
            fs,
            ft,
            "ftotal_upper",
        )

    # Degenerate A ~= constant case.
    else:

        _add(
            0.0,
            sy / sw,
            "degenerate",
        )

    if not candidates:
        raise RuntimeError(
            "bounded profile: no finite candidate."
        )

    chi2, fs, ft, source = min(
        candidates,
        key=lambda item: item[0],
    )

    fb = float(
        ft - fs
    )

    bound_tol = (
        1.0e-10
        * max(
            1.0,
            fmax,
        )
    )

    if abs(
        fs
    ) <= bound_tol:
        fs_active = "lower"

    elif abs(
        fs - fmax
    ) <= bound_tol:
        fs_active = "upper"

    else:
        fs_active = "free"

    if abs(
        ft
    ) <= bound_tol:
        ft_active = "lower"

    elif abs(
        ft - fmax
    ) <= bound_tol:
        ft_active = "upper"

    else:
        ft_active = "free"

    return {
        "fsource": float(fs),
        "ftotal": float(ft),
        "fblend": float(fb),
        "chi2": float(chi2),
        "fmax": float(fmax),
        "solution_type": str(source),
        "fsource_active": fs_active,
        "ftotal_active": ft_active,
    }


def _install_bounded_flux_profile_runtime_patch():

    global _BOUNDED_FLUX_PROFILE_PATCH_INSTALLED
    global _ORIGINAL_TRF_INIT_BOUNDED_PROFILE
    global _ORIGINAL_DERIVE_TELESCOPE_FLUX_BOUNDED_PROFILE

    if _BOUNDED_FLUX_PROFILE_PATCH_INSTALLED:
        return

    if not _bounded_flux_profile_enabled():
        return

    # BOUNDED_FLUX_PROFILE_FIX_V2
    from pyLIMA.models import ML_model as _ML_model
    from pyLIMA.fits import TRF_fit as _TRF_fit

    # --------------------------------------------------------
    # Force TRF constructor into pyLIMA's "fluxes not fitted"
    # path. This is essential: fit_parameters is constructed
    # during __init__, so changing the method later is too late.
    # --------------------------------------------------------

    _ORIGINAL_TRF_INIT_BOUNDED_PROFILE = (
        _TRF_fit.TRFfit.__init__
    )

    def _trf_init_bounded_profile(
        self,
        model,
        telescopes_fluxes_method="fit",
        loss_function="chi2",
    ):

        _ORIGINAL_TRF_INIT_BOUNDED_PROFILE(
            self,
            model,
            telescopes_fluxes_method="polyfit",
            loss_function=loss_function,
        )

        # The analytical pyLIMA residual Jacobian includes explicit
        # telescope-flux columns and is therefore not the Jacobian
        # of the reduced profiled problem.
        self.model.Jacobian_flag = "Numerical"

    _TRF_fit.TRFfit.__init__ = (
        _trf_init_bounded_profile
    )

    # --------------------------------------------------------
    # Replace only the implicit-flux branch.
    #
    # If explicit flux parameters are present, preserve original
    # pyLIMA behavior exactly.
    # --------------------------------------------------------

    _ORIGINAL_DERIVE_TELESCOPE_FLUX_BOUNDED_PROFILE = (
        _ML_model.MLmodel.derive_telescope_flux
    )

    def _derive_telescope_flux_bounded_profile(
        self,
        telescope,
        pyLIMA_parameters,
        magnification,
    ):

        key_fs = (
            "fsource_"
            + telescope.name
        )

        explicit_flux = False

        try:
            value = pyLIMA_parameters[
                key_fs
            ]

            explicit_flux = (
                value is not None
            )

        except (
            TypeError,
            KeyError,
            IndexError,
        ):
            explicit_flux = False

        if explicit_flux:

            return (
                _ORIGINAL_DERIVE_TELESCOPE_FLUX_BOUNDED_PROFILE(
                    self,
                    telescope,
                    pyLIMA_parameters,
                    magnification,
                )
            )

        if (
            self.blend_flux_parameter
            != "ftotal"
        ):

            return (
                _ORIGINAL_DERIVE_TELESCOPE_FLUX_BOUNDED_PROFILE(
                    self,
                    telescope,
                    pyLIMA_parameters,
                    magnification,
                )
            )

        lightcurve = (
            telescope.lightcurve
        )

        flux = lightcurve[
            "flux"
        ].value

        err_flux = lightcurve[
            "err_flux"
        ].value

        result = (
            _bounded_profile_ftotal_2d(
                magnification,
                flux,
                err_flux,
            )
        )

        fs = result[
            "fsource"
        ]

        ft = result[
            "ftotal"
        ]

        fb = result[
            "fblend"
        ]

        pyLIMA_parameters[
            key_fs
        ] = fs

        pyLIMA_parameters[
            "fblend_"
            + telescope.name
        ] = fb

        pyLIMA_parameters[
            "ftotal_"
            + telescope.name
        ] = ft

        if fs != 0.0:

            gblend = (
                fb / fs
            )

        elif fb == 0.0:

            gblend = 0.0

        else:

            gblend = float(
                "inf"
            )

        pyLIMA_parameters[
            "gblend_"
            + telescope.name
        ] = gblend

        # Keep the most recent solution for diagnostics.
        if not hasattr(
            self,
            "_bounded_profile_last_fluxes",
        ):

            self._bounded_profile_last_fluxes = {}

        self._bounded_profile_last_fluxes[
            telescope.name
        ] = dict(
            result
        )

        return None

    _ML_model.MLmodel.derive_telescope_flux = (
        _derive_telescope_flux_bounded_profile
    )

    _BOUNDED_FLUX_PROFILE_PATCH_INSTALLED = True

    print(
        "[bounded_flux_profile] "
        "runtime patch installed; "
        "TRF telescope fluxes are profiled with exact box constraints; "
        "Jacobian=Numerical",
        flush=True,
    )


_install_bounded_flux_profile_runtime_patch()


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

    catalog_row_start, catalog_row_stop = resolve_catalog_row_window(
        first_config_value(
            args.catalog_row_start,
            cfg("input", "catalog_row_start", None),
            cfg("catalog", "row_start", None),
            default=0,
        ),
        first_config_value(
            args.catalog_row_stop,
            cfg("input", "catalog_row_stop", None),
            cfg("catalog", "row_stop", None),
            default=None,
        ),
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

    if "apply_photometric_filter" not in sim_fit_parameters:
        raise RuntimeError(
            "El config controla simulation.apply_photometric_filter, "
            "pero tu función sim_fit todavía no acepta ese argumento. "
            "Propagalo por sim_fit -> simulate_event_for_fit -> sim_event."
        )


    if (
        FIT_INITIAL_GUESS is not None
        and "initial_guess" not in sim_fit_parameters
    ):
        raise RuntimeError(
            "fit.initial_guess está definido, pero "
            f"{getattr(sim_fit, '__name__', sim_fit)!r} "
            "no acepta initial_guess."
        )


    if (
        FIT_OPTIMIZER_OPTIONS is not None
        and "optimizer_options" not in sim_fit_parameters
    ):
        raise RuntimeError(
            "fit.optimizer_options está definido, pero "
            f"{getattr(sim_fit, '__name__', sim_fit)!r} "
            "no acepta optimizer_options."
        )

    log_step("[main] Loading raw catalog ...")
    raw_catalog = load_raw_catalog(
        COLUMNS_FILE,
        DATA_FILE,
        nrows=read_nrows,
        catalog_row_start=catalog_row_start,
        catalog_row_stop=catalog_row_stop,
    )

    log_step("[main] Preparing catalog ...")
    prepared_catalog, invalid_catalog = (
        prepare_catalog(
            raw_catalog,
            max_base_events=max_base_events,
            catalog_row_offset=catalog_row_start,
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


    # ------------------------------------------------------------
    # Optional slicing AFTER noise-realization expansion.
    #
    # This lets each SLURM array element load the same physical catalog
    # window but execute a disjoint interval of logical realizations.
    # ------------------------------------------------------------

    logical_task_total = len(
        tasks
    )

    logical_task_start = (
        0
        if args.logical_task_start is None
        else int(
            args.logical_task_start
        )
    )

    logical_task_stop = (
        logical_task_total
        if args.logical_task_stop is None
        else min(
            int(
                args.logical_task_stop
            ),
            logical_task_total,
        )
    )

    if logical_task_start < 0:
        raise ValueError(
            "--logical-task-start must be >= 0."
        )

    if (
        logical_task_stop
        < logical_task_start
    ):
        raise ValueError(
            "--logical-task-stop must be >= "
            "--logical-task-start."
        )

    tasks = tasks[
        logical_task_start:
        logical_task_stop
    ]

    print(
        "[logical tasks] "
        f"total_expanded={logical_task_total}, "
        f"selected=[{logical_task_start}, "
        f"{logical_task_stop}), "
        f"n_selected={len(tasks)}",
        flush=True,
    )

    if len(tasks) == 0:
        print(
            "[logical tasks] Empty logical slice. "
            "Nothing to run.",
            flush=True,
        )
        return


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
        "MICROLENSING_ROOT": str(MICROLENSING_ROOT),
        "ULENSING_DEGENERATE_MODELS_ROOT": str(
            ULENSING_DEGENERATE_MODELS_ROOT
        ),
        "OUTPUT_ROOT_FROM_CONFIG": str(OUTPUT_ROOT_FROM_CONFIG),
        "RUN_NAME_BASE": str(RUN_NAME_BASE),
        "CHUNK_OUTPUT_LABEL": str(CHUNK_OUTPUT_LABEL),
        "RUN_DIR": str(RUN_DIR),
        "COLUMNS_FILE": str(COLUMNS_FILE),
        "DATA_FILE": str(DATA_FILE),
        "ROMAN_RUBIN_DIR": str(
            ROMAN_RUBIN_DIR
        ),
        "RUBIN_SIM_DATA_DIR": str(
            RUBIN_SIM_DATA_DIR
        ),
        "RUBIN_THROUGHPUTS_DIR": str(
            RUBIN_THROUGHPUTS_DIR
        ),
        "RUBIN_OPSIM_DB_PATH": str(
            RUBIN_OPSIM_DB_PATH
        ),
        "PATH_EPHEMERIDES": str(
            PATH_EPHEMERIDES
        ),
        "TASKS_PER_EVENT": _noise_realization_tasks_per_event(),
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
        "T0_ORIGIN_POLICY": T0_ORIGIN_POLICY,
        "T0_DEFINITION": (
            "t0_jd = first OpSim/MAF timestamp in catalog-visible "
            "bands for selected field + t0_catalog_days"
        ),
        "T0_ZERO_JD_LEGACY_DIAGNOSTIC_ONLY": T0_ZERO_JD,
        "MAX_BASE_EVENTS": max_base_events,
        "READ_NROWS": read_nrows,
        "CATALOG_ROW_START": catalog_row_start,
        "CATALOG_ROW_STOP": catalog_row_stop,
        "N_BASE_EVENTS_SELECTED":
            len(prepared_catalog),
        "N_TASKS": len(tasks),
        "N_WORKERS": workers,
        "POST_DETECTABILITY_TIMEOUT_S":
            POST_DETECTABILITY_TIMEOUT_S,
        "RANDOM_SEED": RANDOM_SEED,
        "SYSTEM_TYPE": SYSTEM_TYPE,
        "MODEL": MODEL,
        "FIT_MODEL": FIT_MODEL,
        "FIT_PARALLAX": FIT_PARALLAX,
        "TRUTH_PARALLAX": TRUTH_PARALLAX,
        "RUN_MULTIPLE_FITS": RUN_MULTIPLE_FITS,
        "PRIMARY_FIT": PRIMARY_FIT,
        "FIT_SPECS": FIT_SPECS,
        "LRT_CONFIG": LRT_CONFIG,
        "APPLY_DETECTION_CRITERIA":
            APPLY_DETECTION_CRITERIA,
        "APPLY_PHOTOMETRIC_FILTER":
            APPLY_PHOTOMETRIC_FILTER,
        "ALGO": ALGO,
        "USE_ROMAN": USE_ROMAN,
        "USE_RUBIN": USE_RUBIN,
        "RUBIN_POINTING_MODE":
            RUBIN_POINTING_MODE,
        "RUBIN_CACHE_CELL_DEG":
            RUBIN_CACHE_CELL_DEG,
        "FIT_BOUNDS_NOPIE":
            FIT_BOUNDS_NOPIE,
        "FIT_INITIAL_GUESS":
            FIT_INITIAL_GUESS,
        "FIT_OPTIMIZER_OPTIONS":
            FIT_OPTIMIZER_OPTIONS,
        "SIMULATION_TIME_RANGE": "complete MAF cadence",
        "FIT_WINDOW_ENABLED": FIT_WINDOW_ENABLED,
        "FIT_WINDOW_HALF_WIDTH_TE": FIT_WINDOW_HALF_WIDTH_TE,
        "FIT_WINDOW_MINIMUM_TOTAL_POINTS":
            FIT_WINDOW_MINIMUM_TOTAL_POINTS,
        "FIT_WINDOW_DEFINITION":
            "[t0-k*tE, t0+k*tE] in absolute JD, fit only",
        "BLENDING_ASSUMPTION":
            BLENDING_ASSUMPTION,
        "BAND_AVAILABILITY_MODE":
            BAND_AVAILABILITY_MODE,
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
        "truth_parallax": TRUTH_PARALLAX,
        "run_multiple_fits": RUN_MULTIPLE_FITS,
        "fit_specs": FIT_SPECS,
        "primary_fit": PRIMARY_FIT,
        "lrt_config": LRT_CONFIG,
        "algo": ALGO,
        "use_roman": USE_ROMAN,
        "use_rubin": USE_RUBIN,
        "rubin_sim_data_dir": str(
            RUBIN_SIM_DATA_DIR
        ),
        "rubin_throughputs_dir": str(
            RUBIN_THROUGHPUTS_DIR
        ),
        "rubin_opsim_db_path": str(
            RUBIN_OPSIM_DB_PATH
        ),
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
        "initial_guess":
            FIT_INITIAL_GUESS,
        "optimizer_options":
            FIT_OPTIMIZER_OPTIONS,
        "rubin_pointing_mode":
            RUBIN_POINTING_MODE,
        "rubin_cache_cell_deg":
            RUBIN_CACHE_CELL_DEG,
        "apply_detection_criteria":
            APPLY_DETECTION_CRITERIA,
        "apply_photometric_filter":
            APPLY_PHOTOMETRIC_FILTER,
        "band_availability_mode":
            BAND_AVAILABILITY_MODE,
        "fit_window_enabled": FIT_WINDOW_ENABLED,
        "fit_window_half_width_tE": FIT_WINDOW_HALF_WIDTH_TE,
        "fit_window_minimum_total_points":
            FIT_WINDOW_MINIMUM_TOTAL_POINTS,
        "catalog_row_start": catalog_row_start,
        "catalog_row_stop": catalog_row_stop,
        "chunk_output_label": CHUNK_OUTPUT_LABEL,
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

    # ------------------------------------------------------------------------
    # Serial/debug mode.
    #
    # With workers=1 we avoid ProcessPool/fork completely.  This is useful on
    # CHE to distinguish a real slow pyLIMA fit from multiprocessing/cache/SQLite
    # interactions in rubin_sim/MAF.
    # ------------------------------------------------------------------------
    if workers == 1:
        print("[main] Running in SERIAL mode because workers=1", flush=True)

        init_worker(
            prepared_catalog,
            worker_config,
        )

        for completed, task in enumerate(
            tasks,
            start=1,
        ):
            try:
                row = run_single_event(
                    task,
                )

            except Exception as error:
                row = {
                    **task,
                    "status": "serial_failed",
                    "error": str(error),
                }

            summary_rows.append(row)

            _summary_wall0 = time.perf_counter()
            _summary_cpu0 = time.process_time()

            save_summary(summary_rows)

            row["timing_run_summary_write_wall_s"] = float(
                time.perf_counter() - _summary_wall0
            )

            row["timing_run_summary_write_cpu_s"] = float(
                time.process_time() - _summary_cpu0
            )

            row["timing_run_summary_rows_written"] = int(
                len(summary_rows)
            )

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
                f"{status_text}",
                flush=True,
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
        return

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

            _summary_wall0 = time.perf_counter()
            _summary_cpu0 = time.process_time()

            save_summary(summary_rows)

            row["timing_run_summary_write_wall_s"] = float(
                time.perf_counter() - _summary_wall0
            )

            row["timing_run_summary_write_cpu_s"] = float(
                time.process_time() - _summary_cpu0
            )

            row["timing_run_summary_rows_written"] = int(
                len(summary_rows)
            )

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
