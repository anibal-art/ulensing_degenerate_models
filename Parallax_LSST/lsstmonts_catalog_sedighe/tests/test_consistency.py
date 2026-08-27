#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Consistency test for the LSSTMONTS hidden-parallax runner.

This version is config-driven:
- repository paths come from config["paths"]
- Rubin sim data, OpSim DB, and throughputs come from config["rubin"]
- no assumption is made that the repository lives under HOME

Typical usage on CHE:
    python tests/test_consistency.py --config configs/config_lsstmonts_baseline_v5p3p5.json --n-test 5

Optional:
    python test_consistency.py --config config.json --global-indices 119842,290522,391145
"""

# ============================================================
# Imports
# ============================================================

import os
import sys
import json
import yaml
import sqlite3
import random
import shutil
import inspect
import argparse
import importlib.util
import traceback
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd


# ============================================================
# Thread control
# ============================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


# ============================================================
# CLI
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
HOME = Path.home()


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Consistency test for run_lsstmonts_catalog_hidden_parallax.py"
    )

    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs" / "config_lsstmonts_baseline_v5p3p5.json"),
        help="Path to JSON/YAML config file.",
    )

    parser.add_argument(
        "--n-test",
        type=int,
        default=5,
        help="Number of prepared valid events to test if --global-indices is not used.",
    )

    parser.add_argument(
        "--global-indices",
        default=None,
        help="Comma-separated global_i list, e.g. 119842,290522,391145.",
    )

    parser.add_argument(
        "--out-root",
        default=None,
        help="Optional output directory for this test. Overrides config-derived test output.",
    )

    parser.add_argument(
        "--no-reset-out",
        action="store_true",
        help="Do not delete previous test output directory.",
    )

    parser.add_argument(
        "--apply-photometric-filter",
        action="store_true",
        help="Apply m5/saturation photometric filtering in sim_fit. Default: disabled for consistency testing.",
    )

    parser.add_argument(
        "--apply-detection-criteria",
        action="store_true",
        help="Apply internal detection criteria. Default: disabled for consistency testing.",
    )

    return parser.parse_args()


ARGS = parse_cli()
CONFIG_PATH = Path(ARGS.config).expanduser().resolve()
CONFIG_DIR = CONFIG_PATH.parent

N_TEST = int(ARGS.n_test)
GLOBAL_INDICES = None
if ARGS.global_indices not in (None, ""):
    GLOBAL_INDICES = [
        int(x.strip())
        for x in str(ARGS.global_indices).split(",")
        if x.strip()
    ]

RESET_OUT = not bool(ARGS.no_reset_out)
APPLY_PHOTOMETRIC_FILTER = bool(ARGS.apply_photometric_filter)
APPLY_DETECTION_CRITERIA = bool(ARGS.apply_detection_criteria)

TIME_WINDOW = None
FIT_TIME_WINDOW = None
TRUTH_PARALLAX = True


# ============================================================
# Config helpers
# ============================================================


def load_config_file(path):
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"No existe CONFIG_PATH: {path}")

    suffix = path.suffix.lower()

    with open(path, "r", encoding="utf-8") as f:
        if suffix == ".json":
            config = json.load(f)
        elif suffix in {".yaml", ".yml"}:
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Extensión de config no soportada: {suffix}")

    if config is None:
        config = {}

    if not isinstance(config, dict):
        raise TypeError("La raíz del config debe ser un diccionario.")

    return config


CONFIG_ORIGINAL = load_config_file(CONFIG_PATH)


def section(config, name):
    value = config.get(name, {})

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(f"La sección {name!r} debe ser un diccionario.")

    return value


PATHS_CFG = section(CONFIG_ORIGINAL, "paths")
RUBIN_CFG = section(CONFIG_ORIGINAL, "rubin")
INPUT_CFG = section(CONFIG_ORIGINAL, "input")


def deep_copy_jsonable(obj):
    return json.loads(json.dumps(obj))


def expand_config_vars(value, variables=None):
    if value is None:
        return None

    if variables is None:
        variables = {}

    s = str(value)

    env = dict(os.environ)
    env["HOME"] = str(HOME)

    for key, val in variables.items():
        if val not in (None, ""):
            env[str(key)] = str(val)

    # Explicit replacement of ${VAR}.
    for key, val in env.items():
        s = s.replace("${" + key + "}", str(val))

    # Fallback for $VAR and ~.
    s = os.path.expandvars(os.path.expanduser(s))

    return s


def resolve_config_path(value, variables=None, base_dir=None):
    if value is None:
        return None

    s = expand_config_vars(value, variables=variables)
    path = Path(s)

    if not path.is_absolute():
        if base_dir is None:
            base_dir = CONFIG_DIR
        path = Path(base_dir) / path

    return path.resolve()


def first_nonempty(*values, default=None):
    for value in values:
        if value not in (None, ""):
            return value
    return default


# ============================================================
# Resolve machine-dependent paths from config["paths"]
# ============================================================


def find_parent_named(start, name):
    start = Path(start).resolve()
    for candidate in [start] + list(start.parents):
        if candidate.name == name:
            return candidate
    return None


DEFAULT_PARALLAX_LSST_BASE = find_parent_named(SCRIPT_DIR, "Parallax_LSST")
if DEFAULT_PARALLAX_LSST_BASE is None:
    DEFAULT_PARALLAX_LSST_BASE = CONFIG_DIR.parent

DEFAULT_ULENSING_ROOT = DEFAULT_PARALLAX_LSST_BASE.parent

MICROLENSING_ROOT = resolve_config_path(
    first_nonempty(
        PATHS_CFG.get("microlensing_root", None),
        default="${HOME}/microlensing",
    ),
    variables={},
    base_dir=CONFIG_DIR,
)

ULENSING_DEGENERATE_MODELS_ROOT = resolve_config_path(
    first_nonempty(
        PATHS_CFG.get("ulensing_degenerate_models_root", None),
        default=str(DEFAULT_ULENSING_ROOT),
    ),
    variables={
        "MICROLENSING_ROOT": MICROLENSING_ROOT,
    },
    base_dir=CONFIG_DIR,
)

PARALLAX_LSST_BASE = resolve_config_path(
    first_nonempty(
        PATHS_CFG.get("parallax_lsst_base", None),
        PATHS_CFG.get("project_base", None),
        default="${ULENSING_DEGENERATE_MODELS_ROOT}/Parallax_LSST",
    ),
    variables={
        "MICROLENSING_ROOT": MICROLENSING_ROOT,
        "ULENSING_DEGENERATE_MODELS_ROOT": ULENSING_DEGENERATE_MODELS_ROOT,
    },
    base_dir=CONFIG_DIR,
)

ROMAN_RUBIN_DIR = resolve_config_path(
    first_nonempty(
        PATHS_CFG.get("roman_rubin_dir", None),
        default="${MICROLENSING_ROOT}/simulation_Rubin/roman_rubin",
    ),
    variables={
        "MICROLENSING_ROOT": MICROLENSING_ROOT,
        "ULENSING_DEGENERATE_MODELS_ROOT": ULENSING_DEGENERATE_MODELS_ROOT,
        "PARALLAX_LSST_BASE": PARALLAX_LSST_BASE,
    },
    base_dir=CONFIG_DIR,
)

OUTPUT_ROOT = resolve_config_path(
    first_nonempty(
        PATHS_CFG.get("output_root", None),
        CONFIG_ORIGINAL.get("path_storage", None),
        default=str(CONFIG_DIR / "test_outputs"),
    ),
    variables={
        "MICROLENSING_ROOT": MICROLENSING_ROOT,
        "ULENSING_DEGENERATE_MODELS_ROOT": ULENSING_DEGENERATE_MODELS_ROOT,
        "PARALLAX_LSST_BASE": PARALLAX_LSST_BASE,
        "ROMAN_RUBIN_DIR": ROMAN_RUBIN_DIR,
    },
    base_dir=CONFIG_DIR,
)

RUNNER_PATH = resolve_config_path(
    first_nonempty(
        PATHS_CFG.get("runner_path", None),
        default="${PARALLAX_LSST_BASE}/lsstmonts_catalog_sedighe/run_lsstmonts_catalog_hidden_parallax.py",
    ),
    variables={
        "MICROLENSING_ROOT": MICROLENSING_ROOT,
        "ULENSING_DEGENERATE_MODELS_ROOT": ULENSING_DEGENERATE_MODELS_ROOT,
        "PARALLAX_LSST_BASE": PARALLAX_LSST_BASE,
        "ROMAN_RUBIN_DIR": ROMAN_RUBIN_DIR,
        "OUTPUT_ROOT": OUTPUT_ROOT,
    },
    base_dir=CONFIG_DIR,
)

if ARGS.out_root not in (None, ""):
    OUT_ROOT = Path(ARGS.out_root).expanduser().resolve()
else:
    OUT_ROOT = OUTPUT_ROOT / "test_consistency" / CONFIG_PATH.stem


# ============================================================
# Resolve Rubin paths from config["rubin"]
# ============================================================


def resolve_rubin_paths():
    if "sim_data_dir" not in RUBIN_CFG:
        raise KeyError(
            "Falta rubin.sim_data_dir en el config. "
            "El test no lo infiere desde HOME."
        )

    common_vars = {
        "MICROLENSING_ROOT": MICROLENSING_ROOT,
        "ULENSING_DEGENERATE_MODELS_ROOT": ULENSING_DEGENERATE_MODELS_ROOT,
        "PARALLAX_LSST_BASE": PARALLAX_LSST_BASE,
        "ROMAN_RUBIN_DIR": ROMAN_RUBIN_DIR,
        "OUTPUT_ROOT": OUTPUT_ROOT,
    }

    rubin_sim_data_dir = resolve_config_path(
        RUBIN_CFG["sim_data_dir"],
        variables=common_vars,
        base_dir=CONFIG_DIR,
    )

    rubin_vars = {
        **common_vars,
        "RUBIN_SIM_DATA_DIR": rubin_sim_data_dir,
        "SIMS_DATA_DIR": rubin_sim_data_dir,
    }

    if "opsim_db_path" not in RUBIN_CFG:
        raise KeyError("Falta rubin.opsim_db_path en el config.")

    rubin_opsim_db_path = resolve_config_path(
        RUBIN_CFG["opsim_db_path"],
        variables=rubin_vars,
        base_dir=CONFIG_DIR,
    )

    if "throughputs_dir" not in RUBIN_CFG:
        raise KeyError("Falta rubin.throughputs_dir en el config.")

    rubin_throughputs_dir = resolve_config_path(
        RUBIN_CFG["throughputs_dir"],
        variables={
            **rubin_vars,
            "RUBIN_OPSIM_DB_PATH": rubin_opsim_db_path,
            "RUBIN_OPSIM_DB": rubin_opsim_db_path,
        },
        base_dir=CONFIG_DIR,
    )

    return rubin_sim_data_dir, rubin_throughputs_dir, rubin_opsim_db_path


RUBIN_SIM_DATA_DIR, RUBIN_THROUGHPUTS_DIR, RUBIN_OPSIM_DB_PATH = resolve_rubin_paths()
EXPECTED_OPSIM_DB = RUBIN_OPSIM_DB_PATH
EXPECTED_OPSIM_BASENAME = EXPECTED_OPSIM_DB.name


# ============================================================
# Export environment before importing runner or set_telescopes
# ============================================================

os.environ["MICROLENSING_ROOT"] = str(MICROLENSING_ROOT)
os.environ["ULENSING_DEGENERATE_MODELS_ROOT"] = str(ULENSING_DEGENERATE_MODELS_ROOT)
os.environ["PARALLAX_LSST_BASE"] = str(PARALLAX_LSST_BASE)
os.environ["ROMAN_RUBIN_DIR"] = str(ROMAN_RUBIN_DIR)
os.environ["OUTPUT_ROOT"] = str(OUTPUT_ROOT)

os.environ["RUBIN_SIM_DATA_DIR"] = str(RUBIN_SIM_DATA_DIR)
os.environ["SIMS_DATA_DIR"] = str(RUBIN_SIM_DATA_DIR)
os.environ["RUBIN_THROUGHPUTS_DIR"] = str(RUBIN_THROUGHPUTS_DIR)
os.environ["RUBIN_OPSIM_DB_PATH"] = str(RUBIN_OPSIM_DB_PATH)
os.environ["RUBIN_OPSIM_DB"] = str(RUBIN_OPSIM_DB_PATH)

if str(ROMAN_RUBIN_DIR) not in sys.path:
    sys.path.insert(0, str(ROMAN_RUBIN_DIR))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================
# Build runtime config with resolved paths
# ============================================================


def write_runtime_config():
    config_runtime = deep_copy_jsonable(CONFIG_ORIGINAL)

    if "paths" not in config_runtime or config_runtime["paths"] is None:
        config_runtime["paths"] = {}

    if "input" not in config_runtime or config_runtime["input"] is None:
        config_runtime["input"] = {}

    if "rubin" not in config_runtime or config_runtime["rubin"] is None:
        config_runtime["rubin"] = {}

    config_runtime["paths"].update(
        {
            "microlensing_root": str(MICROLENSING_ROOT),
            "ulensing_degenerate_models_root": str(ULENSING_DEGENERATE_MODELS_ROOT),
            "parallax_lsst_base": str(PARALLAX_LSST_BASE),
            "roman_rubin_dir": str(ROMAN_RUBIN_DIR),
            "output_root": str(OUTPUT_ROOT),
            "runner_path": str(RUNNER_PATH),
        }
    )

    config_runtime["rubin"].update(
        {
            "sim_data_dir": str(RUBIN_SIM_DATA_DIR),
            "throughputs_dir": str(RUBIN_THROUGHPUTS_DIR),
            "opsim_db_path": str(RUBIN_OPSIM_DB_PATH),
        }
    )

    path_vars = {
        "MICROLENSING_ROOT": MICROLENSING_ROOT,
        "ULENSING_DEGENERATE_MODELS_ROOT": ULENSING_DEGENERATE_MODELS_ROOT,
        "PARALLAX_LSST_BASE": PARALLAX_LSST_BASE,
        "ROMAN_RUBIN_DIR": ROMAN_RUBIN_DIR,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "RUBIN_SIM_DATA_DIR": RUBIN_SIM_DATA_DIR,
        "SIMS_DATA_DIR": RUBIN_SIM_DATA_DIR,
        "RUBIN_THROUGHPUTS_DIR": RUBIN_THROUGHPUTS_DIR,
        "RUBIN_OPSIM_DB_PATH": RUBIN_OPSIM_DB_PATH,
        "RUBIN_OPSIM_DB": RUBIN_OPSIM_DB_PATH,
    }

    if "path_storage" in config_runtime:
        config_runtime["path_storage"] = str(
            resolve_config_path(
                config_runtime["path_storage"],
                variables=path_vars,
                base_dir=CONFIG_DIR,
            )
        )

    for key in ["columns_file", "data_file"]:
        if key in config_runtime["input"]:
            config_runtime["input"][key] = str(
                resolve_config_path(
                    config_runtime["input"][key],
                    variables=path_vars,
                    base_dir=CONFIG_DIR,
                )
            )

    runtime_config_path = CONFIG_DIR / f".runtime_{CONFIG_PATH.stem}_test_resolved.json"

    with open(runtime_config_path, "w", encoding="utf-8") as f:
        json.dump(config_runtime, f, indent=2)

    return runtime_config_path, config_runtime


RUNTIME_CONFIG_PATH, CONFIG_RUNTIME = write_runtime_config()


# ============================================================
# Initial checks
# ============================================================

print("=" * 80)
print("PATHS")
print("=" * 80)
print("HOME                         =", HOME)
print("SCRIPT_PATH                  =", SCRIPT_PATH)
print("CONFIG_PATH                  =", CONFIG_PATH)
print("RUNTIME_CONFIG_PATH          =", RUNTIME_CONFIG_PATH)
print("MICROLENSING_ROOT            =", MICROLENSING_ROOT)
print("ULENSING_DEGENERATE_MODELS   =", ULENSING_DEGENERATE_MODELS_ROOT)
print("PARALLAX_LSST_BASE           =", PARALLAX_LSST_BASE)
print("ROMAN_RUBIN_DIR              =", ROMAN_RUBIN_DIR)
print("RUNNER_PATH                  =", RUNNER_PATH)
print("OUTPUT_ROOT                  =", OUTPUT_ROOT)
print("OUT_ROOT                     =", OUT_ROOT)
print("RUBIN_SIM_DATA_DIR           =", RUBIN_SIM_DATA_DIR)
print("RUBIN_THROUGHPUTS_DIR        =", RUBIN_THROUGHPUTS_DIR)
print("RUBIN_OPSIM_DB_PATH          =", RUBIN_OPSIM_DB_PATH)

assert CONFIG_PATH.exists(), f"No existe CONFIG_PATH: {CONFIG_PATH}"
assert RUNNER_PATH.exists(), f"No existe RUNNER_PATH: {RUNNER_PATH}"
assert ROMAN_RUBIN_DIR.exists(), f"No existe ROMAN_RUBIN_DIR: {ROMAN_RUBIN_DIR}"
assert (ROMAN_RUBIN_DIR / "functions_roman_rubin.py").exists(), (
    f"No encontré functions_roman_rubin.py en ROMAN_RUBIN_DIR: {ROMAN_RUBIN_DIR}"
)
assert RUBIN_SIM_DATA_DIR.exists(), f"No existe RUBIN_SIM_DATA_DIR: {RUBIN_SIM_DATA_DIR}"
assert RUBIN_THROUGHPUTS_DIR.exists(), f"No existe RUBIN_THROUGHPUTS_DIR: {RUBIN_THROUGHPUTS_DIR}"
assert RUBIN_OPSIM_DB_PATH.exists(), f"No existe RUBIN_OPSIM_DB_PATH: {RUBIN_OPSIM_DB_PATH}"

for band in "ugrizy":
    plain = RUBIN_THROUGHPUTS_DIR / f"total_{band}.dat"
    gz = RUBIN_THROUGHPUTS_DIR / f"total_{band}.dat.gz"
    assert plain.exists() or gz.exists(), f"Falta throughput para {band}: {plain}"

if RESET_OUT and OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

for sub in ["models", "fits", "results", "logs", "summary"]:
    (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)


# ============================================================
# OpSim inspection
# ============================================================


def quote_sql_name(name):
    return '"' + str(name).replace('"', '""') + '"'


def inspect_opsim_db(db_path):
    db_path = Path(db_path).resolve()
    con = sqlite3.connect(str(db_path))

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';",
        con,
    )

    obs_table = None
    obs_cols = None

    for table in tables["name"]:
        cols = pd.read_sql(f"PRAGMA table_info({quote_sql_name(table)});", con)
        colnames = set(cols["name"].astype(str))

        if "observationStartMJD" in colnames:
            obs_table = table
            obs_cols = cols
            break

    if obs_table is None:
        con.close()
        raise RuntimeError("No encontré tabla con observationStartMJD en OpSim.")

    colnames = set(obs_cols["name"].astype(str))
    filter_col = None

    for candidate in ["filter", "band", "filtername", "filterName", "filter_name"]:
        if candidate in colnames:
            filter_col = candidate
            break

    q_table = quote_sql_name(obs_table)

    summary = pd.read_sql(
        f"""
        SELECT
            COUNT(*) AS n_obs,
            MIN(observationStartMJD) AS min_mjd,
            MAX(observationStartMJD) AS max_mjd
        FROM {q_table}
        """,
        con,
    )

    by_filter = None
    if filter_col is not None:
        q_filter = quote_sql_name(filter_col)
        by_filter = pd.read_sql(
            f"""
            SELECT
                {q_filter} AS filter,
                COUNT(*) AS n_obs,
                MIN(observationStartMJD) AS min_mjd,
                MAX(observationStartMJD) AS max_mjd
            FROM {q_table}
            GROUP BY {q_filter}
            ORDER BY {q_filter}
            """,
            con,
        )

    con.close()

    return {
        "db_path": db_path,
        "tables": tables,
        "obs_table": obs_table,
        "obs_cols": obs_cols,
        "filter_col": filter_col,
        "summary": summary,
        "by_filter": by_filter,
    }


opsim_info = inspect_opsim_db(EXPECTED_OPSIM_DB)

print("\n" + "=" * 80)
print("OPSIM INFO")
print("=" * 80)
print("obs_table  =", opsim_info["obs_table"])
print("filter_col =", opsim_info["filter_col"])
print(opsim_info["summary"].to_string(index=False))
if opsim_info["by_filter"] is not None:
    print(opsim_info["by_filter"].to_string(index=False))


# ============================================================
# Reproducible RNG
# ============================================================

@contextmanager
def deterministic_rng(seed):
    seed = int(seed)

    original_default_rng = np.random.default_rng
    master_rng = original_default_rng(seed)

    def seeded_default_rng(arg=None):
        if arg is None:
            child_seed = int(master_rng.integers(0, 2**32 - 1))
            return original_default_rng(child_seed)
        return original_default_rng(arg)

    np.random.seed(seed)
    random.seed(seed)
    np.random.default_rng = seeded_default_rng

    try:
        yield
    finally:
        np.random.default_rng = original_default_rng


# ============================================================
# Clean runner import
# ============================================================


def import_runner_clean():
    for module_name in [
        "runner_sedighe_unified",
        "fit_lc",
        "functions_roman_rubin",
        "set_telescopes_pyLIMA",
        "set_model_pyLIMA",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    old_argv = sys.argv[:]

    try:
        sys.argv = [
            str(RUNNER_PATH),
            "--config",
            str(RUNTIME_CONFIG_PATH),
        ]

        module_name = "runner_sedighe_unified"
        spec = importlib.util.spec_from_file_location(module_name, str(RUNNER_PATH))
        runner = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = runner
        spec.loader.exec_module(runner)

    finally:
        sys.argv = old_argv

    return runner


runner = import_runner_clean()

print("\n" + "=" * 80)
print("RUNNER IMPORTADO")
print("=" * 80)
print("runner file        =", getattr(runner, "__file__", ""))
print("runner.CONFIG_PATH =", getattr(runner, "CONFIG_PATH", None))


# ============================================================
# Configure set_telescopes_pyLIMA explicitly
# ============================================================

import set_telescopes_pyLIMA as stp

if hasattr(stp, "configure_rubin_paths"):
    stp.configure_rubin_paths(
        rubin_sim_data_dir=RUBIN_SIM_DATA_DIR,
        rubin_throughputs_dir=RUBIN_THROUGHPUTS_DIR,
        rubin_opsim_db_path=RUBIN_OPSIM_DB_PATH,
        reset_caches=True,
        validate=True,
    )
else:
    print(
        "[warning] set_telescopes_pyLIMA no tiene configure_rubin_paths. "
        "Se usan variables de entorno."
    )


# ============================================================
# Internal path diagnostics
# ============================================================


def module_path_strings(module):
    rows = []

    for name, value in vars(module).items():
        if isinstance(value, (str, Path)):
            s = str(value)
            low = s.lower()

            if (
                ".db" in low
                or "opsim" in low
                or "baseline" in low
                or "rubin_sim_data" in low
                or "throughputs" in low
                or "microlensing" in low
                or "ulensing" in low
            ):
                p = None
                exists = ""

                try:
                    candidate = Path(os.path.expandvars(os.path.expanduser(s)))
                    if candidate.is_absolute():
                        p = candidate.resolve()
                        exists = p.exists()
                except Exception:
                    pass

                rows.append(
                    {
                        "module": module.__name__,
                        "name": name,
                        "value": s,
                        "resolved": str(p) if p is not None else "",
                        "exists": exists,
                    }
                )

    return rows


modules_to_check = [runner, stp]
for module_name in ["fit_lc", "functions_roman_rubin", "set_model_pyLIMA"]:
    if module_name in sys.modules:
        modules_to_check.append(sys.modules[module_name])

internal_paths = []
for module in modules_to_check:
    internal_paths.extend(module_path_strings(module))

internal_paths_df = pd.DataFrame(internal_paths)

print("\n" + "=" * 80)
print("PATHS INTERNOS RELACIONADOS")
print("=" * 80)
if len(internal_paths_df):
    print(internal_paths_df.to_string(index=False))
else:
    print("No encontré paths internos relevantes.")

if len(internal_paths_df):
    bad_paths = internal_paths_df[
        internal_paths_df["value"].astype(str).str.contains(
            "/home/anibal/rubin_sim_data",
            regex=False,
        )
    ]
    assert len(bad_paths) == 0, (
        "Todavía hay path hardcodeado a /home/anibal/rubin_sim_data:\n"
        + bad_paths.to_string(index=False)
    )


# ============================================================
# Seeded fit wrapper
# ============================================================


def install_seeded_fit_wrapper(runner):
    import fit_lc
    import functions_roman_rubin as frr

    state = {"seed": None}
    original_fit_rubin_roman = fit_lc.fit_rubin_roman

    def seeded_fit_rubin_roman(*args, **kwargs):
        if kwargs.get("random_state", None) is None:
            if state["seed"] is not None:
                kwargs["random_state"] = int(state["seed"])
        return original_fit_rubin_roman(*args, **kwargs)

    fit_lc.fit_rubin_roman = seeded_fit_rubin_roman
    frr.fit_rubin_roman = seeded_fit_rubin_roman

    if hasattr(frr, "run_all_fits"):
        frr.run_all_fits.__globals__["fit_rubin_roman"] = seeded_fit_rubin_roman

    if hasattr(runner, "run_all_fits"):
        runner.run_all_fits.__globals__["fit_rubin_roman"] = seeded_fit_rubin_roman

    def set_fit_seed(seed):
        state["seed"] = int(seed)

    return set_fit_seed


set_fit_seed = install_seeded_fit_wrapper(runner)


# ============================================================
# Load catalog and build tasks
# ============================================================


def load_catalog_and_tasks(runner, n_test, global_indices=None):
    read_nrows = None

    if hasattr(runner, "READ_NROWS_CONFIG"):
        read_nrows = runner.READ_NROWS_CONFIG
        if hasattr(runner, "parse_optional_positive_int"):
            read_nrows = runner.parse_optional_positive_int(read_nrows, "input.read_nrows")

    raw_catalog = runner.load_raw_catalog(
        runner.COLUMNS_FILE,
        runner.DATA_FILE,
        nrows=read_nrows,
    )

    if global_indices is None:
        max_base_events = int(n_test)
    else:
        max_base_events = int(max(global_indices)) + 1

    prepared_catalog, invalid_catalog = runner.prepare_catalog(
        raw_catalog,
        max_base_events=max_base_events,
    )

    tasks = runner.build_tasks(prepared_catalog)

    if global_indices is not None:
        allowed = set(int(x) for x in global_indices)
        tasks = [t for t in tasks if int(t["global_i"]) in allowed]
    else:
        tasks = tasks[: int(n_test)]

    return raw_catalog, prepared_catalog, invalid_catalog, tasks


raw_catalog, prepared_catalog, invalid_catalog, tasks = load_catalog_and_tasks(
    runner,
    n_test=N_TEST,
    global_indices=GLOBAL_INDICES,
)

print("\n" + "=" * 80)
print("CATÁLOGO")
print("=" * 80)
print("raw_catalog       =", len(raw_catalog))
print("prepared_catalog  =", len(prepared_catalog))
print("invalid_catalog   =", len(invalid_catalog))
print("tasks test        =", len(tasks))
print("global_i test     =", [int(t["global_i"]) for t in tasks])


# ============================================================
# Helpers to run one event
# ============================================================


def notebook_like_config_from_runner(runner):
    return {
        "path_ephemerides": str(runner.PATH_EPHEMERIDES),
        "use_roman": bool(runner.USE_ROMAN),
        "use_rubin": bool(runner.USE_RUBIN),
        "rubin_pointing_mode": runner.RUBIN_POINTING_MODE,
        "rubin_cache_cell_deg": runner.RUBIN_CACHE_CELL_DEG,
        "opsim_db_path": str(RUBIN_OPSIM_DB_PATH),
        "rubin_sim_data_dir": str(RUBIN_SIM_DATA_DIR),
        "rubin_throughputs_dir": str(RUBIN_THROUGHPUTS_DIR),
    }


def prepare_task_inputs(runner, prepared_catalog, task):
    global_i = int(task["global_i"])
    prepared_index = int(task["prepared_index"])

    base_row = prepared_catalog.iloc[prepared_index].copy()

    if not hasattr(runner, "apply_t0_from_first_maf_timestamp"):
        raise AttributeError(
            "El runner no tiene apply_t0_from_first_maf_timestamp. "
            "Este test requiere la versión con t0 referido al primer timestamp MAF."
        )

    t0_config = notebook_like_config_from_runner(runner)

    base_row = runner.apply_t0_from_first_maf_timestamp(base_row, t0_config)

    if hasattr(runner, "validate_t0_first_maf_timestamp"):
        runner.validate_t0_first_maf_timestamp(base_row, context=f"global_i={global_i}")

    pair_catalog = runner.build_single_row_pair_catalog(base_row, task)
    param_samplers = runner.fixed_param_samplers(base_row, task)
    seed = int(task["simulation_seed"])

    return {
        "global_i": global_i,
        "prepared_index": prepared_index,
        "base_row": base_row,
        "pair_catalog": pair_catalog,
        "param_samplers": param_samplers,
        "seed": seed,
    }


def make_event_dirs(out_root, task):
    global_i = int(task["global_i"])
    field_name = str(task.get("field_name", "unknown_field"))
    event_tag = f"event_{global_i:07d}"

    dirs = {
        "models": out_root / "models" / field_name / event_tag,
        "fits": out_root / "fits" / field_name / event_tag,
        "results": out_root / "results" / field_name / event_tag,
        "logs": out_root / "logs" / field_name,
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def build_sim_fit_kwargs(runner, task_inputs, dirs):
    seed = int(task_inputs["seed"])
    fit_parallax = bool(runner.FIT_PARALLAX)

    if fit_parallax:
        fit_bounds = getattr(runner, "FIT_BOUNDS", None)
    else:
        fit_bounds = getattr(
            runner,
            "FIT_BOUNDS_NOPIE",
            getattr(runner, "FIT_BOUNDS", None),
        )

    kwargs = dict(
        i=seed,
        system_type=runner.SYSTEM_TYPE,
        model=runner.MODEL,
        algo=runner.ALGO,
        path_TRILEGAL_set=None,
        path_GENULENS_set=None,
        path_to_save_model=str(dirs["models"]),
        path_to_save_fit=str(dirs["fits"]),
        path_to_save_results=str(dirs["results"]),
        path_ephemerides=str(runner.PATH_EPHEMERIDES),
        time_window=TIME_WINDOW,
        param_samplers=task_inputs["param_samplers"],
        custom_system=None,
        catalog_mode="astrodatalab_pairs",
        pair_catalog=task_inputs["pair_catalog"],
        path_pair_catalog=None,
        use_roman=bool(runner.USE_ROMAN),
        use_rubin=bool(runner.USE_RUBIN),
        truth_parallax=bool(TRUTH_PARALLAX),
        fit_time_window=FIT_TIME_WINDOW,
        return_data=True,
        fit_model=runner.FIT_MODEL,
        fit_parallax=fit_parallax,
        fit_defaults=None,
        fit_bounds=fit_bounds,
        rubin_pointing_mode=runner.RUBIN_POINTING_MODE,
        rubin_cache_cell_deg=runner.RUBIN_CACHE_CELL_DEG,
        apply_detection_criteria=bool(APPLY_DETECTION_CRITERIA),
    )

    sig = inspect.signature(runner.sim_fit).parameters

    if "apply_photometric_filter" in sig:
        kwargs["apply_photometric_filter"] = bool(APPLY_PHOTOMETRIC_FILTER)

    if "opsim_db_path" in sig:
        kwargs["opsim_db_path"] = str(RUBIN_OPSIM_DB_PATH)

    if "rubin_sim_data_dir" in sig:
        kwargs["rubin_sim_data_dir"] = str(RUBIN_SIM_DATA_DIR)

    if "rubin_throughputs_dir" in sig:
        kwargs["rubin_throughputs_dir"] = str(RUBIN_THROUGHPUTS_DIR)

    return kwargs


def run_one_event_for_test(runner, prepared_catalog, task):
    task_inputs = prepare_task_inputs(runner, prepared_catalog, task)
    seed = int(task_inputs["seed"])
    set_fit_seed(seed)

    dirs = make_event_dirs(OUT_ROOT, task)

    if hasattr(runner, "install_runtime_patches"):
        runner.install_runtime_patches()

    if hasattr(runner, "set_runtime_event_context"):
        runner.set_runtime_event_context(task_inputs["base_row"])

    kwargs = build_sim_fit_kwargs(runner, task_inputs, dirs)

    try:
        with deterministic_rng(seed):
            simfit_result = runner.sim_fit(**kwargs)
    finally:
        if hasattr(runner, "clear_runtime_event_context"):
            runner.clear_runtime_event_context()

    return simfit_result, task_inputs, dirs


# ============================================================
# Event/OpSim verification helpers
# ============================================================


def _to_numpy(x):
    if hasattr(x, "value"):
        x = x.value
    return np.asarray(x, dtype=float)


def canonical_filter_name(value):
    value = str(value).strip().lower()

    if len(value) >= 4 and value.startswith("b'") and value.endswith("'"):
        value = value[2:-1]

    if len(value) >= 4 and value.startswith('b"') and value.endswith('"'):
        value = value[2:-1]

    value = value.split("_")[0]

    if value in {"u", "g", "r", "i", "z", "y"}:
        return value

    if len(value) > 0 and value[0] in {"u", "g", "r", "i", "z", "y"}:
        return value[0]

    return value


def telescope_band(tel):
    band = getattr(tel, "camera_filter", None)

    if band is None:
        band = getattr(tel, "filter", None)

    if band is None:
        band = getattr(tel, "name", "")

    return canonical_filter_name(band)


def extract_event_times(simfit_result):
    model_obj = simfit_result["pyLIMAmodel_true"]
    rows = []

    for tel in model_obj.event.telescopes:
        if tel.lightcurve is None or len(tel.lightcurve) == 0:
            continue

        t_jd = _to_numpy(tel.lightcurve["time"])
        if len(t_jd) == 0:
            continue

        band = telescope_band(tel)
        is_roman = str(tel.name) == "W149" or band == "w149"

        rows.append(
            {
                "tel_name": str(tel.name),
                "band": band,
                "is_roman": bool(is_roman),
                "n": int(len(t_jd)),
                "tmin_jd": float(np.nanmin(t_jd)),
                "tmax_jd": float(np.nanmax(t_jd)),
                "tmin_mjd": float(np.nanmin(t_jd) - 2400000.5),
                "tmax_mjd": float(np.nanmax(t_jd) - 2400000.5),
            }
        )

    return pd.DataFrame(rows)


def sample_rubin_mjds_from_event(simfit_result, max_per_band=10):
    model_obj = simfit_result["pyLIMAmodel_true"]
    samples = []

    for tel in model_obj.event.telescopes:
        if tel.lightcurve is None or len(tel.lightcurve) == 0:
            continue

        band = telescope_band(tel)

        if str(tel.name) == "W149" or band == "w149":
            continue

        t_jd = _to_numpy(tel.lightcurve["time"])
        t_mjd = t_jd - 2400000.5

        if len(t_mjd) == 0:
            continue

        if len(t_mjd) <= max_per_band:
            selected = np.arange(len(t_mjd))
        else:
            selected = np.linspace(0, len(t_mjd) - 1, max_per_band, dtype=int)

        for idx in selected:
            samples.append({"band": band, "mjd": float(t_mjd[idx])})

    return pd.DataFrame(samples)


def check_sample_mjds_exist_in_opsim(samples_df, opsim_db, opsim_info, tol_days=1.0e-7):
    if len(samples_df) == 0:
        return pd.DataFrame()

    obs_table = opsim_info["obs_table"]
    filter_col = opsim_info["filter_col"]
    q_table = quote_sql_name(obs_table)
    con = sqlite3.connect(str(opsim_db))
    rows = []

    for _, sample in samples_df.iterrows():
        mjd = float(sample["mjd"])
        band = str(sample["band"])

        if filter_col is None:
            query = f"""
            SELECT observationStartMJD
            FROM {q_table}
            WHERE observationStartMJD BETWEEN ? AND ?
            LIMIT 10
            """
            match = pd.read_sql(query, con, params=(mjd - tol_days, mjd + tol_days))
            filter_matched = np.nan
            filters_found = ""

        else:
            q_filter = quote_sql_name(filter_col)
            query = f"""
            SELECT observationStartMJD, {q_filter} AS filter
            FROM {q_table}
            WHERE observationStartMJD BETWEEN ? AND ?
            LIMIT 20
            """
            match = pd.read_sql(query, con, params=(mjd - tol_days, mjd + tol_days))

            if len(match):
                filters_found_list = [
                    canonical_filter_name(v)
                    for v in match["filter"].astype(str).tolist()
                ]
                filter_matched = band in filters_found_list
                filters_found = ",".join(sorted(set(filters_found_list)))
            else:
                filter_matched = False
                filters_found = ""

        rows.append(
            {
                "band": band,
                "mjd": mjd,
                "matched_time": bool(len(match) > 0),
                "matched_filter": filter_matched,
                "filters_found": filters_found,
                "n_matches": int(len(match)),
            }
        )

    con.close()
    return pd.DataFrame(rows)


def get_event_param(simfit_result, name):
    ep = simfit_result.get("event_params", {})

    if isinstance(ep, dict) and name in ep:
        return ep[name]

    if hasattr(ep, name):
        return getattr(ep, name)

    raise KeyError(name)


def base_row_scalar(base_row, name, default=np.nan):
    try:
        if name in base_row.index:
            return base_row[name]
    except Exception:
        pass
    return default


def verify_event_against_opsim(simfit_result, task_inputs, opsim_info):
    times_df = extract_event_times(simfit_result)

    if len(times_df) == 0:
        raise RuntimeError("El evento no tiene telescopios con lightcurve.")

    rubin_df = times_df[~times_df["is_roman"]].copy()

    if len(rubin_df) == 0:
        raise RuntimeError("El evento no tiene puntos Rubin para verificar OpSim.")

    opsim_summary = opsim_info["summary"].iloc[0]
    opsim_min_mjd = float(opsim_summary["min_mjd"])
    opsim_max_mjd = float(opsim_summary["max_mjd"])

    rubin_min_mjd = float(rubin_df["tmin_mjd"].min())
    rubin_max_mjd = float(rubin_df["tmax_mjd"].max())

    assert rubin_min_mjd >= opsim_min_mjd - 1.0e-6, (
        "Hay timestamps Rubin antes del inicio global de la OpSim."
    )

    assert rubin_max_mjd <= opsim_max_mjd + 1.0e-6, (
        "Hay timestamps Rubin después del fin global de la OpSim."
    )

    samples_df = sample_rubin_mjds_from_event(simfit_result, max_per_band=10)

    match_df = check_sample_mjds_exist_in_opsim(
        samples_df,
        EXPECTED_OPSIM_DB,
        opsim_info,
        tol_days=1.0e-7,
    )

    if len(match_df) > 0:
        assert match_df["matched_time"].all(), (
            "Algunos timestamps Rubin simulados no aparecen en la OpSim esperada."
        )

        finite_filter = match_df["matched_filter"].dropna()
        if len(finite_filter) > 0:
            assert finite_filter.astype(bool).all(), (
                "Algunos timestamps coinciden con la OpSim, pero no con el filtro esperado."
            )

    base_row = task_inputs["base_row"]

    t0_used = float(get_event_param(simfit_result, "t0"))
    tE_used = float(get_event_param(simfit_result, "tE"))

    field_start_visible_jd = float(rubin_df["tmin_jd"].min())
    field_end_visible_jd = float(rubin_df["tmax_jd"].max())

    t0_reference_jd = float(base_row_scalar(base_row, "t0_reference_jd", np.nan))
    t0_catalog_days = float(base_row_scalar(base_row, "t0_catalog_days", np.nan))
    t0_jd_from_base_row = float(base_row_scalar(base_row, "t0_jd", np.nan))

    assert np.isfinite(t0_reference_jd), "t0_reference_jd no es finito."
    assert np.isfinite(t0_catalog_days), "t0_catalog_days no es finito."
    assert np.isfinite(t0_jd_from_base_row), "base_row['t0_jd'] no es finito."

    expected_t0_jd = t0_reference_jd + t0_catalog_days
    diff_t0_expected = t0_used - expected_t0_jd

    assert abs(diff_t0_expected) < 1.0e-6, (
        "No se cumple t0 = t0_reference_jd + t0_catalog_days.\n"
        f"t0_used={t0_used}\n"
        f"t0_reference_jd={t0_reference_jd}\n"
        f"t0_catalog_days={t0_catalog_days}\n"
        f"diff={diff_t0_expected}"
    )

    assert abs(t0_used - t0_jd_from_base_row) < 1.0e-6, (
        "event_params['t0'] no coincide con base_row['t0_jd']."
    )

    summary = {
        "global_i": int(task_inputs["global_i"]),
        "prepared_index": int(task_inputs["prepared_index"]),
        "seed": int(task_inputs["seed"]),
        "status": str(simfit_result.get("status", "")),
        "microlensing_root": str(MICROLENSING_ROOT),
        "ulensing_degenerate_models_root": str(ULENSING_DEGENERATE_MODELS_ROOT),
        "roman_rubin_dir": str(ROMAN_RUBIN_DIR),
        "rubin_sim_data_dir": str(RUBIN_SIM_DATA_DIR),
        "rubin_throughputs_dir": str(RUBIN_THROUGHPUTS_DIR),
        "opsim_db": str(EXPECTED_OPSIM_DB),
        "opsim_basename": EXPECTED_OPSIM_DB.name,
        "opsim_obs_table": str(opsim_info["obs_table"]),
        "n_telescopes": int(len(times_df)),
        "n_rubin_bands": int(len(rubin_df)),
        "n_rubin_points": int(rubin_df["n"].sum()),
        "n_roman_points": int(
            times_df.loc[times_df["is_roman"], "n"].sum()
            if np.any(times_df["is_roman"])
            else 0
        ),
        "rubin_first_mjd": rubin_min_mjd,
        "rubin_last_mjd": rubin_max_mjd,
        "rubin_first_jd": field_start_visible_jd,
        "rubin_last_jd": field_end_visible_jd,
        "opsim_global_min_mjd": opsim_min_mjd,
        "opsim_global_max_mjd": opsim_max_mjd,
        "t0_used_jd": t0_used,
        "tE_used_days": tE_used,
        "t0_minus_first_retained_rubin_days": t0_used - field_start_visible_jd,
        "t0_reference_jd": t0_reference_jd,
        "t0_catalog_days": t0_catalog_days,
        "expected_t0_jd": expected_t0_jd,
        "t0_used_minus_expected_days": diff_t0_expected,
        "t0_origin": str(base_row_scalar(base_row, "t0_origin", "")),
        "t0_reference_source": str(base_row_scalar(base_row, "t0_reference_source", "")),
        "t0_reference_visible_bands": str(
            base_row_scalar(base_row, "t0_reference_visible_bands", "")
        ),
        "t0_reference_first_filter": str(
            base_row_scalar(base_row, "t0_reference_first_filter", "")
        ),
        "sampled_opsim_times": int(len(match_df)),
        "sampled_opsim_times_matched": int(
            match_df["matched_time"].sum() if len(match_df) else 0
        ),
        "sampled_opsim_filters_matched": int(
            match_df["matched_filter"].fillna(False).sum()
            if len(match_df) and "matched_filter" in match_df
            else 0
        ),
    }

    try:
        info = getattr(stp, "LAST_DATASLICE_INFO", {})
        summary["stp_last_opsim_db_path"] = str(info.get("opsim_db_path", ""))
        summary["stp_last_opsim_cache_tag"] = str(info.get("opsim_cache_tag", ""))
        summary["stp_last_maf_n_obs"] = info.get("n_obs", np.nan)
        summary["stp_last_maf_ra"] = info.get("maf_Ra", info.get("Ra", np.nan))
        summary["stp_last_maf_dec"] = info.get("maf_Dec", info.get("Dec", np.nan))
    except Exception:
        pass

    return summary, times_df, match_df


# ============================================================
# Run small sample
# ============================================================

all_summaries = []
all_times = []
all_matches = []
failures = []

for task in tasks:
    global_i = int(task["global_i"])

    print("\n" + "=" * 100)
    print(f"RUN global_i = {global_i}")
    print("=" * 100)

    try:
        simfit_result, task_inputs, dirs = run_one_event_for_test(
            runner,
            prepared_catalog,
            task,
        )

        summary, times_df, match_df = verify_event_against_opsim(
            simfit_result,
            task_inputs,
            opsim_info,
        )

        summary["model_dir"] = str(dirs["models"])
        summary["fit_dir"] = str(dirs["fits"])
        summary["results_dir"] = str(dirs["results"])

        all_summaries.append(summary)

        times_df.insert(0, "global_i", global_i)
        all_times.append(times_df)

        match_df.insert(0, "global_i", global_i)
        all_matches.append(match_df)

        print("OK")
        print(
            "t0 - first retained Rubin = "
            f"{summary['t0_minus_first_retained_rubin_days']:.8f} days"
        )
        print(
            "t0_used - expected = "
            f"{summary['t0_used_minus_expected_days']:.3e} days"
        )
        print("Rubin points =", summary["n_rubin_points"])

    except Exception as error:
        print("FAILED:", repr(error))
        traceback.print_exc()

        failures.append(
            {
                "global_i": global_i,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )


summary_df = pd.DataFrame(all_summaries)
times_all_df = pd.concat(all_times, ignore_index=True) if len(all_times) else pd.DataFrame()
matches_all_df = pd.concat(all_matches, ignore_index=True) if len(all_matches) else pd.DataFrame()
failures_df = pd.DataFrame(failures)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
if len(summary_df):
    print(summary_df.to_string(index=False))
else:
    print("summary_df vacío")

print("\n" + "=" * 80)
print("TIMES PER BAND")
print("=" * 80)
if len(times_all_df):
    print(times_all_df.to_string(index=False))
else:
    print("times_all_df vacío")

print("\n" + "=" * 80)
print("OPSIM SAMPLE MATCHES")
print("=" * 80)
if len(matches_all_df):
    print(matches_all_df.to_string(index=False))
else:
    print("matches_all_df vacío")

if len(failures_df):
    print("\n" + "=" * 80)
    print("FAILURES")
    print("=" * 80)
    print(failures_df.to_string(index=False))


# ============================================================
# Save outputs
# ============================================================

summary_dir = OUT_ROOT / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)

summary_csv = summary_dir / "opsim_check_summary.csv"
times_csv = summary_dir / "opsim_check_times_by_band.csv"
matches_csv = summary_dir / "opsim_check_sample_matches.csv"
failures_csv = summary_dir / "opsim_check_failures.csv"

summary_df.to_csv(summary_csv, index=False)
times_all_df.to_csv(times_csv, index=False)
matches_all_df.to_csv(matches_csv, index=False)
failures_df.to_csv(failures_csv, index=False)

try:
    summary_df.to_parquet(summary_dir / "opsim_check_summary.parquet", index=False)
    times_all_df.to_parquet(summary_dir / "opsim_check_times_by_band.parquet", index=False)
    matches_all_df.to_parquet(summary_dir / "opsim_check_sample_matches.parquet", index=False)
    failures_df.to_parquet(summary_dir / "opsim_check_failures.parquet", index=False)
except Exception as error:
    print("No pude guardar parquet, pero los CSV están guardados:", repr(error))

print("\n" + "=" * 80)
print("ARCHIVOS GUARDADOS")
print("=" * 80)
print(summary_csv)
print(times_csv)
print(matches_csv)
print(failures_csv)


# ============================================================
# Final asserts
# ============================================================

assert len(summary_df) > 0, "No se pudo correr ningún evento correctamente."

assert summary_df["opsim_basename"].eq(EXPECTED_OPSIM_BASENAME).all(), (
    "Algún evento no quedó asociado a la OpSim esperada."
)

assert summary_df["opsim_db"].eq(str(EXPECTED_OPSIM_DB)).all(), (
    "Algún evento reporta una OpSim distinta a la esperada."
)

assert summary_df["roman_rubin_dir"].eq(str(ROMAN_RUBIN_DIR)).all(), (
    "Algún evento reporta un ROMAN_RUBIN_DIR distinto al config."
)

assert np.all(summary_df["n_rubin_points"] > 0), "Algún evento no tiene puntos Rubin."

assert np.all(
    np.abs(summary_df["t0_used_minus_expected_days"].astype(float)) < 1.0e-6
), "Algún evento viola t0 = t0_reference_jd + t0_catalog_days."

assert np.all(
    summary_df["sampled_opsim_times_matched"] == summary_df["sampled_opsim_times"]
), "Algún timestamp Rubin sampleado no aparece en la OpSim esperada."

if "sampled_opsim_filters_matched" in summary_df.columns:
    assert np.all(
        summary_df["sampled_opsim_filters_matched"] == summary_df["sampled_opsim_times"]
    ), "Algún timestamp Rubin sampleado aparece en la OpSim, pero con filtro distinto."

if "stp_last_opsim_db_path" in summary_df.columns:
    non_empty = summary_df["stp_last_opsim_db_path"].astype(str) != ""
    if np.any(non_empty):
        assert summary_df.loc[non_empty, "stp_last_opsim_db_path"].eq(
            str(EXPECTED_OPSIM_DB)
        ).all(), "set_telescopes_pyLIMA reporta una OpSim distinta."

print("\n" + "=" * 80)
print("OK FINAL")
print("=" * 80)
print("La muestra chica usa paths, OpSim y throughputs definidos en el config.")
print("Config original:", CONFIG_PATH)
print("Config runtime: ", RUNTIME_CONFIG_PATH)
print("Runner:         ", RUNNER_PATH)
print("Roman/Rubin:    ", ROMAN_RUBIN_DIR)
print("OpSim:          ", EXPECTED_OPSIM_DB)
print("Throughputs:    ", RUBIN_THROUGHPUTS_DIR)
print("Output:         ", OUT_ROOT)
