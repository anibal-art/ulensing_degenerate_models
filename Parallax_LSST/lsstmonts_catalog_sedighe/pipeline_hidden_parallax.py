# ============================================================
# pipeline_hidden_parallax.py
# Pipeline modular para:
# Sedighe catalog -> sim_fit reproducible -> resumen -> plots opcionales
#
# Plot final opcional:
#   bandas alineadas con flujos TRUE + ZP por banda + residuales + inset trayectoria
# ============================================================

import os
import gc
import sys
import time
import random
import shutil
import inspect
import traceback
import importlib.util
import io
import json

from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager, redirect_stdout
from typing import Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cycler

from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ============================================================
# Configuración
# ============================================================

@dataclass
class SedighePipelineConfig:
    runner_path: Path
    config_path: Path
    roman_rubin_dir: Path
    out_dir: Path

    # Roots resueltos desde configuration file
    microlensing_root: Optional[Path] = None
    ulensing_degenerate_models_root: Optional[Path] = None
    output_root: Optional[Path] = None

    # Paths Rubin resueltos desde configuration file
    rubin_sim_data_dir: Optional[Path] = None
    rubin_throughputs_dir: Optional[Path] = None
    rubin_opsim_db_path: Optional[Path] = None

    # Dataset
    max_base_events: Optional[int] = None
    global_indices: Optional[list[int]] = None

    # Simulación
    truth_parallax: bool = True
    time_window: Optional[Any] = None

    # Fit
    fit_time_window: Optional[Any] = None
    fit_defaults: Optional[Any] = None
    fit_bounds: Optional[Any] = None
    return_data: bool = True

    # Filtros
    apply_photometric_filter: bool = True
    apply_detection_criteria: bool = True

    # Roman/Rubin
    use_roman: Optional[bool] = None
    use_rubin: Optional[bool] = None
    rubin_pointing_mode: Optional[str] = None
    rubin_cache_cell_deg: Optional[float] = None

    # Output
    reset_output: bool = False
    append_summary: bool = True
    summary_name: str = "run_summary.parquet"

    # Plots
    make_plots: bool = False
    plot_indices: Optional[list[int]] = None

    # Opciones de plot:
    #   "aligned_inset" = gráfico final recomendado.
    #   "quick_pylima" = gráfico simple con pyLIMA_plots.
    plot_style: str = "aligned_inset"
    reference_band: str = "g"
    allow_reference_band_fallback: bool = True

    plot_n_dense: int = 10000
    plot_n_tE: float = 10.0
    plot_inset_n_tE: float = 4.0
    plot_inset_loc: str = "upper left"
    plot_show_true_no_parallax_reference: bool = False

    # Trayectoria del fit en el inset:
    # - "same_true_window": evalúa true y fit en la ventana temporal del evento true.
    # - "own_fit_tE_window": evalúa el fit alrededor de t0_fit ± N tE_fit.
    plot_fit_trajectory_time_mode: str = "own_fit_tE_window"

    plot_residuals: bool = True
    save_plot_dpi: int = 180

    # Debug
    verbose: bool = True

    @classmethod
    def from_config_file(
        cls,
        config_path,
        runner_path=None,
        roman_rubin_dir=None,
        out_dir=None,
        **overrides,
    ):
        """
        Construye SedighePipelineConfig leyendo el JSON/YAML principal usado
        por el runner.

        La versión corregida resuelve también la sección paths:

            paths.microlensing_root
            paths.ulensing_degenerate_models_root
            paths.output_root

        y la sección rubin:

            rubin.sim_data_dir
            rubin.opsim_db_path
            rubin.throughputs_dir

        De esta forma la PC y el cluster se controlan cambiando solo el
        configuration file, sin reconstruir paths a partir de HOME.
        """

        config_path = Path(config_path).expanduser().resolve()
        raw_config = _load_pipeline_config_file(config_path)

        fit_cfg = _section(raw_config, "fit")
        simulation_cfg = _section(raw_config, "simulation")
        observing_cfg = _section(raw_config, "observing")
        truth_cfg = _section(raw_config, "truth")
        hidden_cfg = _section(raw_config, "hidden_parallax")
        output_cfg = _section(raw_config, "output")
        selection_cfg = _section(raw_config, "selection")

        path_info = _resolve_pipeline_paths_from_config(
            raw_config=raw_config,
            config_path=config_path,
            runner_path=runner_path,
            roman_rubin_dir=roman_rubin_dir,
            out_dir=out_dir,
        )

        rubin_info = _resolve_rubin_paths_from_config(
            raw_config=raw_config,
            config_path=config_path,
            path_info=path_info,
        )

        fit_time_window = _fit_time_window_from_config(fit_cfg)

        kwargs = dict(
            runner_path=Path(path_info["runner_path"]),
            config_path=Path(config_path),
            roman_rubin_dir=Path(path_info["roman_rubin_dir"]),
            out_dir=Path(path_info["out_dir"]),

            microlensing_root=Path(path_info["microlensing_root"]),
            ulensing_degenerate_models_root=Path(
                path_info["ulensing_degenerate_models_root"]
            ),
            output_root=Path(path_info["output_root"]),

            rubin_sim_data_dir=Path(rubin_info["rubin_sim_data_dir"]),
            rubin_throughputs_dir=Path(rubin_info["rubin_throughputs_dir"]),
            rubin_opsim_db_path=Path(rubin_info["rubin_opsim_db_path"]),

            max_base_events=_parse_all_or_int(
                _first_non_none(
                    raw_config.get("Nevents", None),
                    selection_cfg.get("max_base_events", None),
                    default=None,
                )
            ),

            truth_parallax=bool(
                truth_cfg.get("parallax", True)
            ),
            time_window=simulation_cfg.get("time_window", None),

            fit_time_window=fit_time_window,
            fit_defaults=fit_cfg.get("defaults", None),
            fit_bounds=fit_cfg.get("bounds", None),
            return_data=bool(
                hidden_cfg.get("return_data", True)
            ),

            apply_photometric_filter=bool(
                _first_non_none(
                    simulation_cfg.get("apply_photometric_filter", None),
                    selection_cfg.get("apply_photometric_filter", None),
                    default=True,
                )
            ),
            apply_detection_criteria=bool(
                _first_non_none(
                    simulation_cfg.get("apply_detection_criteria", None),
                    selection_cfg.get("apply_detection_criteria", None),
                    default=False,
                )
            ),

            use_roman=observing_cfg.get("use_roman", None),
            use_rubin=observing_cfg.get("use_rubin", None),
            rubin_pointing_mode=simulation_cfg.get(
                "rubin_pointing_mode",
                None,
            ),
            rubin_cache_cell_deg=simulation_cfg.get(
                "rubin_cache_cell_deg",
                None,
            ),

            reset_output=bool(
                hidden_cfg.get("reset_output", False)
            ),
            append_summary=bool(
                hidden_cfg.get("append_summary", True)
            ),
            summary_name=hidden_cfg.get(
                "summary_name",
                "run_summary.parquet",
            ),

            make_plots=bool(
                hidden_cfg.get("make_plots", False)
            ),
            plot_indices=hidden_cfg.get("plot_indices", None),
            plot_style=hidden_cfg.get(
                "plot_style",
                "aligned_inset",
            ),
            reference_band=hidden_cfg.get(
                "reference_band",
                "g",
            ),
            allow_reference_band_fallback=bool(
                hidden_cfg.get("allow_reference_band_fallback", True)
            ),
            plot_n_dense=int(
                hidden_cfg.get("plot_n_dense", 10000)
            ),
            plot_n_tE=float(
                hidden_cfg.get("plot_n_tE", 10.0)
            ),
            plot_inset_n_tE=float(
                hidden_cfg.get("plot_inset_n_tE", 4.0)
            ),
            plot_inset_loc=hidden_cfg.get(
                "plot_inset_loc",
                "upper left",
            ),
            plot_show_true_no_parallax_reference=bool(
                hidden_cfg.get("plot_show_true_no_parallax_reference", False)
            ),
            plot_fit_trajectory_time_mode=hidden_cfg.get(
                "plot_fit_trajectory_time_mode",
                "own_fit_tE_window",
            ),
            plot_residuals=bool(
                hidden_cfg.get("plot_residuals", True)
            ),
            save_plot_dpi=int(
                hidden_cfg.get("save_plot_dpi", 180)
            ),

            verbose=bool(
                hidden_cfg.get("verbose", True)
            ),
        )

        # Los overrides siguen ganando sobre el configuration file.
        kwargs.update(overrides)

        return cls(**kwargs)


@dataclass
class PipelinePaths:
    root: Path
    models: Path
    fits: Path
    results: Path
    plots: Path
    logs: Path
    summary: Path


# ============================================================
# Constantes fotométricas para plotting
# ============================================================

ZP_PYLIMA = 27.4

ZP_RUBIN = {
    "u": 27.03,
    "g": 28.38,
    "r": 28.16,
    "i": 27.85,
    "z": 27.46,
    "y": 26.68,
    "W149": 27.615,
}

BAND_PREFERENCE = ["g", "r", "i", "z", "y", "u", "W149"]


def _load_pipeline_config_file(path):
    """
    Lee el mismo JSON/YAML usado por el runner.
    """

    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"No existe el config: {path}")

    suffix = path.suffix.lower()

    with open(path, "r", encoding="utf-8") as file:
        if suffix == ".json":
            data = json.load(file)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as error:
                raise ImportError(
                    "Para leer YAML instalá PyYAML o usá JSON."
                ) from error
            data = yaml.safe_load(file)
        else:
            raise ValueError(
                f"Extensión de config no soportada: {suffix!r}"
            )

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise TypeError(
            "La raíz del configuration file debe ser un diccionario."
        )

    return data


def _section(config_dict, name):
    value = config_dict.get(name, {})

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(
            f"La sección {name!r} del configuration file debe ser un diccionario."
        )

    return value


def _first_non_none(*values, default=None):
    for value in values:
        if value is not None and value != "":
            return value

    return default


def _parse_all_or_int(value):
    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"all", "none", ""}:
            return None
        return int(stripped)

    return int(value)


def _expand_config_vars(value, variables=None):
    """
    Expande ~, variables de entorno y placeholders ${VAR} usando variables
    explícitas además de os.environ.
    """

    if value is None:
        return None

    if variables is None:
        variables = {}

    text = os.path.expanduser(str(value))

    env = dict(os.environ)
    env["HOME"] = str(Path.home())

    for key, val in variables.items():
        if val not in (None, ""):
            env[str(key)] = str(val)

    for key, val in env.items():
        text = text.replace("${" + str(key) + "}", str(val))

    return os.path.expandvars(text)


def _resolve_config_path(value, variables=None, base_dir=None):
    """
    Resuelve paths del configuration file.

    Los paths relativos se interpretan relativos al directorio del config.
    """

    if value is None:
        return None

    text = _expand_config_vars(
        value,
        variables=variables,
    )

    path = Path(text)

    if not path.is_absolute():
        if base_dir is None:
            base_dir = Path.cwd()
        path = Path(base_dir) / path

    return path.resolve()


def _resolve_pipeline_paths_from_config(
    raw_config,
    config_path,
    runner_path=None,
    roman_rubin_dir=None,
    out_dir=None,
):
    """
    Resuelve paths machine-dependent del pipeline desde raw_config["paths"].
    """

    config_path = Path(config_path).expanduser().resolve()
    config_dir = config_path.parent

    paths_cfg = _section(raw_config, "paths")
    hidden_cfg = _section(raw_config, "hidden_parallax")
    output_cfg = _section(raw_config, "output")

    base_vars = {
        "HOME": Path.home(),
    }

    microlensing_root = _resolve_config_path(
        _first_non_none(
            paths_cfg.get("microlensing_root", None),
            os.environ.get("MICROLENSING_ROOT", ""),
            default="${HOME}/microlensing",
        ),
        variables=base_vars,
        base_dir=config_dir,
    )

    vars_after_microlensing = {
        **base_vars,
        "MICROLENSING_ROOT": microlensing_root,
    }

    ulensing_root = _resolve_config_path(
        _first_non_none(
            paths_cfg.get("ulensing_degenerate_models_root", None),
            paths_cfg.get("ulensing_root", None),
            os.environ.get("ULENSING_DEGENERATE_MODELS_ROOT", ""),
            default="${HOME}/ulensing_degenerate_models",
        ),
        variables=vars_after_microlensing,
        base_dir=config_dir,
    )

    vars_after_ulensing = {
        **vars_after_microlensing,
        "ULENSING_DEGENERATE_MODELS_ROOT": ulensing_root,
    }

    parallax_lsst_root = _resolve_config_path(
        _first_non_none(
            paths_cfg.get("parallax_lsst_root", None),
            paths_cfg.get("project_base", None),
            os.environ.get("PARALLAX_LSST_BASE", ""),
            default=str(ulensing_root / "Parallax_LSST"),
        ),
        variables=vars_after_ulensing,
        base_dir=config_dir,
    )

    vars_after_parallax = {
        **vars_after_ulensing,
        "PARALLAX_LSST_BASE": parallax_lsst_root,
    }

    if runner_path is None:
        runner_path_value = _first_non_none(
            paths_cfg.get("runner_path", None),
            hidden_cfg.get("runner_path", None),
            default=str(
                parallax_lsst_root
                / "lsstmonts_catalog_sedighe"
                / "run_lsstmonts_catalog_hidden_parallax.py"
            ),
        )
    else:
        runner_path_value = runner_path

    runner_path = _resolve_config_path(
        runner_path_value,
        variables=vars_after_parallax,
        base_dir=config_dir,
    )

    if roman_rubin_dir is None:
        roman_rubin_value = _first_non_none(
            paths_cfg.get("roman_rubin_dir", None),
            os.environ.get("ROMAN_RUBIN_DIR", ""),
            default=str(
                microlensing_root
                / "simulation_Rubin"
                / "roman_rubin"
            ),
        )
    else:
        roman_rubin_value = roman_rubin_dir

    roman_rubin_dir = _resolve_config_path(
        roman_rubin_value,
        variables=vars_after_parallax,
        base_dir=config_dir,
    )

    vars_after_roman = {
        **vars_after_parallax,
        "ROMAN_RUBIN_DIR": roman_rubin_dir,
    }

    output_root = _resolve_config_path(
        _first_non_none(
            paths_cfg.get("output_root", None),
            os.environ.get("OUTPUT_ROOT", ""),
            default="${HOME}/hidden_parallax",
        ),
        variables=vars_after_roman,
        base_dir=config_dir,
    )

    vars_after_output = {
        **vars_after_roman,
        "OUTPUT_ROOT": output_root,
    }

    run_name = str(
        _first_non_none(
            raw_config.get("run_name", None),
            output_cfg.get("run_name", None),
            default="hidden_parallax_run",
        )
    )

    if out_dir is None:
        out_dir_value = _first_non_none(
            hidden_cfg.get("out_dir", None),
            output_cfg.get("pipeline_dir", None),
            paths_cfg.get("pipeline_output_dir", None),
            default=str(output_root / run_name),
        )
    else:
        out_dir_value = out_dir

    out_dir = _resolve_config_path(
        out_dir_value,
        variables=vars_after_output,
        base_dir=config_dir,
    )

    for key, val in {
        "MICROLENSING_ROOT": microlensing_root,
        "ULENSING_DEGENERATE_MODELS_ROOT": ulensing_root,
        "PARALLAX_LSST_BASE": parallax_lsst_root,
        "ROMAN_RUBIN_DIR": roman_rubin_dir,
        "OUTPUT_ROOT": output_root,
    }.items():
        os.environ[key] = str(val)

    return {
        "microlensing_root": microlensing_root,
        "ulensing_degenerate_models_root": ulensing_root,
        "parallax_lsst_root": parallax_lsst_root,
        "runner_path": runner_path,
        "roman_rubin_dir": roman_rubin_dir,
        "output_root": output_root,
        "out_dir": out_dir,
    }


def _resolve_rubin_paths_from_config(raw_config, config_path, path_info=None):
    """
    Resuelve rubin.sim_data_dir / opsim_db_path / throughputs_dir desde config.
    """

    config_path = Path(config_path).expanduser().resolve()
    config_dir = config_path.parent

    rubin_cfg = _section(raw_config, "rubin")
    paths_cfg = _section(raw_config, "paths")

    variables = {
        "HOME": Path.home(),
    }

    if path_info is not None:
        variables.update({
            "MICROLENSING_ROOT": path_info.get("microlensing_root"),
            "ULENSING_DEGENERATE_MODELS_ROOT": path_info.get(
                "ulensing_degenerate_models_root"
            ),
            "PARALLAX_LSST_BASE": path_info.get("parallax_lsst_root"),
            "ROMAN_RUBIN_DIR": path_info.get("roman_rubin_dir"),
            "OUTPUT_ROOT": path_info.get("output_root"),
        })

    sim_data_value = _first_non_none(
        rubin_cfg.get("sim_data_dir", None),
        paths_cfg.get("rubin_sim_data_dir", None),
        os.environ.get("RUBIN_SIM_DATA_DIR", ""),
        os.environ.get("SIMS_DATA_DIR", ""),
        default="${HOME}/rubin_sim_data",
    )

    rubin_sim_data_dir = _resolve_config_path(
        sim_data_value,
        variables=variables,
        base_dir=config_dir,
    )

    variables.update({
        "RUBIN_SIM_DATA_DIR": rubin_sim_data_dir,
        "SIMS_DATA_DIR": rubin_sim_data_dir,
    })

    throughputs_value = _first_non_none(
        rubin_cfg.get("throughputs_dir", None),
        paths_cfg.get("rubin_throughputs_dir", None),
        os.environ.get("RUBIN_THROUGHPUTS_DIR", ""),
        default=str(rubin_sim_data_dir / "throughputs" / "baseline"),
    )

    rubin_throughputs_dir = _resolve_config_path(
        throughputs_value,
        variables=variables,
        base_dir=config_dir,
    )

    variables["RUBIN_THROUGHPUTS_DIR"] = rubin_throughputs_dir

    opsim_value = _first_non_none(
        rubin_cfg.get("opsim_db_path", None),
        _section(raw_config, "simulation").get("opsim_db_path", None),
        paths_cfg.get("opsim_db_path", None),
        os.environ.get("RUBIN_OPSIM_DB_PATH", ""),
        os.environ.get("RUBIN_OPSIM_DB", ""),
        default=None,
    )

    if opsim_value is None:
        rubin_opsim_db_path = None
    else:
        rubin_opsim_db_path = _resolve_config_path(
            opsim_value,
            variables=variables,
            base_dir=config_dir,
        )

    for key, val in {
        "RUBIN_SIM_DATA_DIR": rubin_sim_data_dir,
        "SIMS_DATA_DIR": rubin_sim_data_dir,
        "RUBIN_THROUGHPUTS_DIR": rubin_throughputs_dir,
        "RUBIN_OPSIM_DB_PATH": rubin_opsim_db_path,
        "RUBIN_OPSIM_DB": rubin_opsim_db_path,
    }.items():
        if val is not None:
            os.environ[key] = str(val)

    return {
        "rubin_sim_data_dir": rubin_sim_data_dir,
        "rubin_throughputs_dir": rubin_throughputs_dir,
        "rubin_opsim_db_path": rubin_opsim_db_path,
    }


def _fit_time_window_from_config(fit_cfg):
    """
    Para el pipeline de inspección dejamos que el runner aplique su patch
    de fit-window. Este valor se pasa a sim_fit solo si el config lo pide
    explícitamente con una lista/tupla o con enabled=True.

    En el config discutido fit.time_window.enabled=False, así que devuelve None.
    """

    tw = fit_cfg.get("time_window", None)

    if tw is None:
        return None

    if isinstance(tw, (list, tuple)):
        return tw

    if isinstance(tw, dict):
        if not bool(tw.get("enabled", False)):
            return None

        # Mantener explícito: el runner ya interpreta este bloque para su
        # cropping interno. sim_fit espera una ventana ya resuelta, por eso
        # aquí devolvemos None salvo que se haya pasado una ventana absoluta.
        if "window" in tw:
            return tw["window"]

        return None

    return tw


# ============================================================
# Utilidades generales
# ============================================================

@contextmanager
def deterministic_rng(seed):
    """
    Context manager para reproducibilidad local.
    """

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


def _safe_scalar(value):
    """
    Convierte valores tipo lista/array/np scalar en escalares guardables.
    """

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return np.nan
        if len(value) == 1:
            return _safe_scalar(value[0])
        return repr(value)

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.nan
        if value.size == 1:
            return _safe_scalar(value.ravel()[0])
        return repr(value.tolist())

    if isinstance(value, pd.Series):
        if len(value) == 0:
            return np.nan
        if len(value) == 1:
            return _safe_scalar(value.iloc[0])
        return repr(value.tolist())

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if value is None:
        return np.nan

    if isinstance(value, (str, int, float, bool)):
        return value

    return repr(value)


def _has(params, name):
    if isinstance(params, dict):
        return name in params

    if hasattr(params, name):
        return True

    try:
        params[name]
        return True
    except Exception:
        return False


def _val(params, name, default=np.nan):
    if isinstance(params, dict):
        return params.get(name, default)

    if hasattr(params, name):
        return getattr(params, name)

    try:
        return params[name]
    except Exception:
        return default


def _to_numpy(x):
    return np.asarray(
        x.value if hasattr(x, "value") else x,
        dtype=float,
    )


def _arr(x):
    return np.asarray(
        getattr(x, "value", x),
        dtype=float,
    )


def _band_zp(band):
    band = str(band)

    if band in ZP_RUBIN:
        return float(ZP_RUBIN[band])

    return float(ZP_PYLIMA)


def _mag_to_flux_band(mag, band):
    zp = _band_zp(band)

    return 10.0 ** (
        (zp - np.asarray(mag, dtype=float)) / 2.5
    )


def _flux_to_mag_band(flux, band):
    zp = _band_zp(band)

    flux = np.asarray(flux, dtype=float)

    mag = np.full(
        flux.shape,
        np.nan,
        dtype=float,
    )

    ok = np.isfinite(flux) & (flux > 0.0)

    mag[ok] = zp - 2.5 * np.log10(flux[ok])

    return mag


def _magerr_to_fluxerr(mag, err_mag, band):
    flux = _mag_to_flux_band(
        mag,
        band,
    )

    return (
        np.log(10.0)
        / 2.5
        * flux
        * np.asarray(err_mag, dtype=float)
    )


def _fluxerr_to_magerr(flux, err_flux):
    flux = np.asarray(flux, dtype=float)
    err_flux = np.asarray(err_flux, dtype=float)

    err_mag = np.full(
        flux.shape,
        np.nan,
        dtype=float,
    )

    ok = (
        np.isfinite(flux)
        & np.isfinite(err_flux)
        & (flux > 0.0)
        & (err_flux >= 0.0)
    )

    err_mag[ok] = (
        2.5
        / np.log(10.0)
        * err_flux[ok]
        / flux[ok]
    )

    return err_mag


def make_output_dirs(config: SedighePipelineConfig) -> PipelinePaths:
    """
    Crea carpetas de salida.

    Para dataset completo, por defecto NO borra el output anterior.
    Si querés empezar de cero, usá reset_output=True.
    """

    root = Path(config.out_dir)

    if config.reset_output and root.exists():
        shutil.rmtree(root)

    paths = PipelinePaths(
        root=root,
        models=root / "models",
        fits=root / "fits",
        results=root / "results",
        plots=root / "plots",
        logs=root / "logs",
        summary=root / config.summary_name,
    )

    for path in [
        paths.root,
        paths.models,
        paths.fits,
        paths.results,
        paths.plots,
        paths.logs,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return paths


def append_record_to_parquet(record: dict, path: Path, unique_cols=("global_i", "simulation_seed")):
    """
    Guarda un registro de resumen por evento.

    Si el parquet ya existe, concatena y elimina duplicados por global_i + simulation_seed.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean_record = {
        key: _safe_scalar(value)
        for key, value in record.items()
    }

    df_new = pd.DataFrame([clean_record])

    if path.exists():
        df_old = pd.read_parquet(path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)

        cols = [c for c in unique_cols if c in df_out.columns]
        if cols:
            df_out = df_out.drop_duplicates(subset=cols, keep="last")

    else:
        df_out = df_new

    df_out.to_parquet(path, index=False)

    return df_out


# ============================================================
# Importar runner y preparar módulos
# ============================================================

def import_runner(config: SedighePipelineConfig):
    """
    Importa el runner asegurando que fit_lc y functions_roman_rubin
    se lean nuevamente desde disco.

    También exporta los paths resueltos desde el configuration file para que
    el runner y set_telescopes_pyLIMA no reconstruyan nada a partir de HOME.
    """

    path_exports = {
        "ROMAN_RUBIN_DIR": config.roman_rubin_dir,
        "MICROLENSING_ROOT": config.microlensing_root,
        "ULENSING_DEGENERATE_MODELS_ROOT": config.ulensing_degenerate_models_root,
        "OUTPUT_ROOT": config.output_root,
        "RUBIN_SIM_DATA_DIR": config.rubin_sim_data_dir,
        "SIMS_DATA_DIR": config.rubin_sim_data_dir,
        "RUBIN_THROUGHPUTS_DIR": config.rubin_throughputs_dir,
        "RUBIN_OPSIM_DB_PATH": config.rubin_opsim_db_path,
        "RUBIN_OPSIM_DB": config.rubin_opsim_db_path,
    }

    for key, value in path_exports.items():
        if value is not None:
            os.environ[key] = str(value)

    if str(config.roman_rubin_dir) not in sys.path:
        sys.path.insert(0, str(config.roman_rubin_dir))

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
            str(config.runner_path),
            "--config",
            str(config.config_path),
        ]

        module_name = "runner_sedighe_unified"

        spec = importlib.util.spec_from_file_location(
            module_name,
            config.runner_path,
        )

        runner = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = runner
        spec.loader.exec_module(runner)

    finally:
        sys.argv = old_argv

    return runner


def install_seeded_fit_patch(runner):
    """
    Instala una sola vez un wrapper para fit_rubin_roman.

    A diferencia del wrapper de debug, este NO se envuelve recursivamente
    cada vez que cambiamos de evento. Cambia la seed con set_fit_seed(seed).
    """

    import fit_lc
    import functions_roman_rubin as frr

    state = getattr(
        fit_lc.fit_rubin_roman,
        "_seeded_fit_state",
        None,
    )

    if state is None:
        original_fit_rubin_roman = fit_lc.fit_rubin_roman
        state = {"seed": None}

        def seeded_fit_rubin_roman(*args, **kwargs):
            if kwargs.get("random_state", None) is None:
                if state["seed"] is not None:
                    kwargs["random_state"] = int(state["seed"])

            return original_fit_rubin_roman(*args, **kwargs)

        seeded_fit_rubin_roman._seeded_fit_state = state

    else:
        seeded_fit_rubin_roman = fit_lc.fit_rubin_roman

    fit_lc.fit_rubin_roman = seeded_fit_rubin_roman
    frr.fit_rubin_roman = seeded_fit_rubin_roman

    if hasattr(frr, "run_all_fits"):
        frr.run_all_fits.__globals__["fit_rubin_roman"] = seeded_fit_rubin_roman

    if hasattr(runner, "run_all_fits"):
        runner.run_all_fits.__globals__["fit_rubin_roman"] = seeded_fit_rubin_roman

    def set_fit_seed(seed):
        state["seed"] = int(seed)

    return set_fit_seed


# ============================================================
# Cargar catálogo y tareas
# ============================================================

def load_catalog_and_tasks(runner, config: SedighePipelineConfig):
    """
    Carga catálogo crudo, prepara catálogo y construye tasks.

    Fix para plotting:
    si config.global_indices contiene eventos específicos, no leemos
    desde la fila 0 hasta max(global_i). Leemos solamente la ventana
    cruda mínima que contiene esos global_i:

        catalog_row_start = min(global_indices)
        catalog_row_stop  = max(global_indices) + 1

    y pasamos catalog_row_offset a prepare_catalog para que los global_i
    preparados sigan siendo los índices globales originales.
    """

    read_nrows = None

    if hasattr(runner, "READ_NROWS_CONFIG"):
        read_nrows = runner.READ_NROWS_CONFIG

        if hasattr(runner, "parse_optional_positive_int"):
            read_nrows = runner.parse_optional_positive_int(
                read_nrows,
                name="input.read_nrows",
            )
        else:
            read_nrows = _parse_all_or_int(read_nrows)

    # ------------------------------------------------------------
    # Determinar ventana de catálogo.
    # Para el uso normal sin global_indices, conserva comportamiento viejo.
    # Para plotting de eventos seleccionados, lee solo esas filas.
    # ------------------------------------------------------------

    catalog_row_start = 0
    catalog_row_stop = None

    global_indices = getattr(config, "global_indices", None)

    if global_indices is not None and len(global_indices) > 0:
        global_indices = [int(x) for x in global_indices]

        catalog_row_start = int(min(global_indices))
        catalog_row_stop = int(max(global_indices)) + 1

        window_nrows = catalog_row_stop - catalog_row_start

        if read_nrows is None:
            read_nrows = window_nrows
        else:
            read_nrows = min(int(read_nrows), int(window_nrows))

        if getattr(config, "verbose", True):
            print(
                "[pipeline] Using catalog row window from global_indices:",
                f"[{catalog_row_start}, {catalog_row_stop})",
                flush=True,
            )

    raw_catalog = runner.load_raw_catalog(
        runner.COLUMNS_FILE,
        runner.DATA_FILE,
        nrows=read_nrows,
        catalog_row_start=catalog_row_start,
        catalog_row_stop=catalog_row_stop,
    )

    # ------------------------------------------------------------
    # Preparar catálogo conservando global_i correcto.
    # Si el runner acepta catalog_row_offset, se lo pasamos.
    # ------------------------------------------------------------

    prepare_kwargs = {
        "max_base_events": config.max_base_events,
    }

    try:
        import inspect as _inspect
        prepare_params = _inspect.signature(runner.prepare_catalog).parameters

        if "catalog_row_offset" in prepare_params:
            prepare_kwargs["catalog_row_offset"] = int(catalog_row_start)

    except Exception:
        pass

    prepared_catalog, invalid_catalog = runner.prepare_catalog(
        raw_catalog,
        **prepare_kwargs,
    )

    tasks = runner.build_tasks(
        prepared_catalog,
    )

    if config.global_indices is not None:
        allowed = set(int(x) for x in config.global_indices)

        tasks = [
            task for task in tasks
            if int(task["global_i"]) in allowed
        ]

    return raw_catalog, prepared_catalog, invalid_catalog, tasks


def notebook_like_config_from_runner(runner, config: SedighePipelineConfig):
    """
    Config mínima que necesita apply_t0_from_first_maf_timestamp.
    """

    return {
        "path_ephemerides": str(runner.PATH_EPHEMERIDES),
        "use_roman": runner.USE_ROMAN if config.use_roman is None else config.use_roman,
        "use_rubin": runner.USE_RUBIN if config.use_rubin is None else config.use_rubin,
        "rubin_pointing_mode": (
            runner.RUBIN_POINTING_MODE
            if config.rubin_pointing_mode is None
            else config.rubin_pointing_mode
        ),
        "rubin_cache_cell_deg": (
            runner.RUBIN_CACHE_CELL_DEG
            if config.rubin_cache_cell_deg is None
            else config.rubin_cache_cell_deg
        ),
        "opsim_db_path": (
            str(config.rubin_opsim_db_path)
            if config.rubin_opsim_db_path is not None
            else str(getattr(runner, "RUBIN_OPSIM_DB_PATH", ""))
        ),
        "rubin_sim_data_dir": (
            str(config.rubin_sim_data_dir)
            if config.rubin_sim_data_dir is not None
            else str(getattr(runner, "RUBIN_SIM_DATA_DIR", ""))
        ),
        "rubin_throughputs_dir": (
            str(config.rubin_throughputs_dir)
            if config.rubin_throughputs_dir is not None
            else str(getattr(runner, "RUBIN_THROUGHPUTS_DIR", ""))
        ),
    }


def prepare_single_task_inputs(
    runner,
    prepared_catalog,
    task,
    config: SedighePipelineConfig,
):
    """
    Para un task:
    - toma base_row,
    - resuelve t0 desde primer timestamp MAF,
    - construye pair_catalog,
    - construye param_samplers,
    - extrae seed.
    """

    global_i = int(task["global_i"])
    prepared_index = int(task["prepared_index"])

    base_row = prepared_catalog.iloc[prepared_index].copy()

    t0_config = notebook_like_config_from_runner(
        runner,
        config,
    )

    base_row = runner.apply_t0_from_first_maf_timestamp(
        base_row,
        t0_config,
    )

    runner.validate_t0_first_maf_timestamp(
        base_row,
        context=f"global_i={global_i}",
    )

    pair_catalog = runner.build_single_row_pair_catalog(
        base_row,
        task,
    )

    param_samplers = runner.fixed_param_samplers(
        base_row,
        task,
    )

    seed = int(task["simulation_seed"])

    return {
        "global_i": global_i,
        "prepared_index": prepared_index,
        "base_row": base_row,
        "pair_catalog": pair_catalog,
        "param_samplers": param_samplers,
        "seed": seed,
    }


# ============================================================
# Construcción de kwargs para sim_fit
# ============================================================

def build_sim_fit_kwargs(
    runner,
    paths: PipelinePaths,
    task_inputs: dict,
    config: SedighePipelineConfig,
):
    """
    Centraliza todos los kwargs de sim_fit.
    """

    seed = int(task_inputs["seed"])
    pair_catalog = task_inputs["pair_catalog"]
    param_samplers = task_inputs["param_samplers"]

    use_roman = runner.USE_ROMAN if config.use_roman is None else config.use_roman
    use_rubin = runner.USE_RUBIN if config.use_rubin is None else config.use_rubin

    rubin_pointing_mode = (
        runner.RUBIN_POINTING_MODE
        if config.rubin_pointing_mode is None
        else config.rubin_pointing_mode
    )

    rubin_cache_cell_deg = (
        runner.RUBIN_CACHE_CELL_DEG
        if config.rubin_cache_cell_deg is None
        else config.rubin_cache_cell_deg
    )

    fit_parallax = runner.FIT_PARALLAX

    if config.fit_bounds is not None:
        fit_bounds = config.fit_bounds
    else:
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

        path_to_save_model=str(paths.models),
        path_to_save_fit=str(paths.fits),
        path_to_save_results=str(paths.results),
        path_ephemerides=str(runner.PATH_EPHEMERIDES),

        time_window=config.time_window,
        param_samplers=param_samplers,

        custom_system=None,
        catalog_mode="astrodatalab_pairs",
        pair_catalog=pair_catalog,
        path_pair_catalog=None,

        use_roman=use_roman,
        use_rubin=use_rubin,

        truth_parallax=bool(config.truth_parallax),

        fit_time_window=config.fit_time_window,
        return_data=bool(config.return_data),

        fit_model=runner.FIT_MODEL,
        fit_parallax=fit_parallax,
        fit_defaults=config.fit_defaults,
        fit_bounds=fit_bounds,

        rubin_pointing_mode=rubin_pointing_mode,
        rubin_cache_cell_deg=rubin_cache_cell_deg,

        apply_detection_criteria=bool(config.apply_detection_criteria),
    )

    sig = inspect.signature(runner.sim_fit).parameters

    if "apply_photometric_filter" in sig:
        kwargs["apply_photometric_filter"] = bool(config.apply_photometric_filter)

    else:
        raise RuntimeError(
            "runner.sim_fit no acepta apply_photometric_filter. "
            "Hay que actualizar functions_roman_rubin/sim_fit."
        )

    # Pasar paths Rubin solo si la versión de sim_fit los acepta.
    if "opsim_db_path" in sig and config.rubin_opsim_db_path is not None:
        kwargs["opsim_db_path"] = str(config.rubin_opsim_db_path)

    if "rubin_sim_data_dir" in sig and config.rubin_sim_data_dir is not None:
        kwargs["rubin_sim_data_dir"] = str(config.rubin_sim_data_dir)

    if "rubin_throughputs_dir" in sig and config.rubin_throughputs_dir is not None:
        kwargs["rubin_throughputs_dir"] = str(config.rubin_throughputs_dir)

    return kwargs


# ============================================================
# Extraer y resumir resultados
# ============================================================

def vector_from_pylima_parameters(model_obj, pyLIMA_parameters):
    """
    Reconstruye el vector en el orden de model_dictionnary a partir de los
    parámetros YA convertidos por pyLIMA durante la simulación.
    """

    values = []
    missing = []

    for name, _ in sorted(
        model_obj.model_dictionnary.items(),
        key=lambda x: x[1],
    ):
        try:
            value = pyLIMA_parameters[name]
        except Exception:
            if hasattr(pyLIMA_parameters, name):
                value = getattr(pyLIMA_parameters, name)
            else:
                missing.append(name)
                continue

        values.append(float(value))

    if missing:
        raise KeyError(
            "No pude reconstruir el vector true. "
            f"Faltan parámetros en pyLIMA_parameters_true: {missing}"
        )

    return np.asarray(values, dtype=float)


def count_points_by_telescope(model_obj):
    """
    Devuelve conteo de puntos por telescopio.
    """

    counts = {}

    if model_obj is None:
        return counts

    for tel in model_obj.event.telescopes:
        if tel.lightcurve is None:
            n = 0
        else:
            try:
                n = len(tel.lightcurve)
            except Exception:
                n = 0

        counts[str(tel.name)] = int(n)

    return counts


def summarize_simfit_result(
    simfit_result,
    task_inputs: dict,
    config: SedighePipelineConfig,
    elapsed_sec: float,
):
    """
    Convierte el resultado de sim_fit en una fila de resumen.
    """

    record = {
        "global_i": int(task_inputs["global_i"]),
        "prepared_index": int(task_inputs["prepared_index"]),
        "simulation_seed": int(task_inputs["seed"]),
        "status": "unknown",
        "elapsed_sec": float(elapsed_sec),
        "apply_photometric_filter": bool(config.apply_photometric_filter),
        "apply_detection_criteria": bool(config.apply_detection_criteria),
        "truth_parallax": bool(config.truth_parallax),
        "time_window": repr(config.time_window),
        "fit_time_window": repr(config.fit_time_window),
    }

    if not isinstance(simfit_result, dict):
        record["status"] = "non_dict_result"
        record["result_type"] = type(simfit_result).__name__
        return record

    record["status"] = _safe_scalar(simfit_result.get("status", "unknown"))
    record["sim_model"] = _safe_scalar(simfit_result.get("model", np.nan))
    record["fit_model"] = _safe_scalar(simfit_result.get("fit_model", np.nan))
    record["fit_parallax"] = _safe_scalar(simfit_result.get("fit_parallax", np.nan))
    record["use_roman"] = _safe_scalar(simfit_result.get("use_roman", np.nan))
    record["use_rubin"] = _safe_scalar(simfit_result.get("use_rubin", np.nan))

    event_params = simfit_result.get("event_params", {})

    if isinstance(event_params, dict):
        for key, value in event_params.items():
            record[f"true_{key}"] = _safe_scalar(value)

    true_model_obj = simfit_result.get("pyLIMAmodel_true", None)
    fit_model_obj = simfit_result.get("pyLIMAmodel_rr", None)

    true_counts = count_points_by_telescope(true_model_obj)
    fit_counts = count_points_by_telescope(fit_model_obj)

    for band, n in true_counts.items():
        record[f"n_true_{band}"] = n

    for band, n in fit_counts.items():
        record[f"n_fit_{band}"] = n

    record["n_true_total"] = int(sum(true_counts.values()))
    record["n_fit_total"] = int(sum(fit_counts.values()))

    try:
        fit_rr = simfit_result.get("fit_rr", None)

        if fit_rr is not None and fit_model_obj is not None:
            best_model = np.asarray(
                fit_rr.fit_results["best_model"],
                dtype=float,
            )

            fit_pyparams = fit_model_obj.compute_pyLIMA_parameters(
                best_model,
            )

            for key in [
                "t0",
                "u0",
                "tE",
                "rho",
                "piEN",
                "piEE",
                "chi2",
            ]:
                if _has(fit_pyparams, key):
                    record[f"fit_{key}"] = _safe_scalar(
                        _val(fit_pyparams, key)
                    )

            if _has(fit_pyparams, "piEN") and _has(fit_pyparams, "piEE"):
                piEN = float(_val(fit_pyparams, "piEN"))
                piEE = float(_val(fit_pyparams, "piEE"))
                record["fit_piE"] = float(np.hypot(piEN, piEE))

            if "chi2" in fit_rr.fit_results:
                record["fit_chi2_raw"] = _safe_scalar(
                    fit_rr.fit_results["chi2"]
                )

    except Exception as error:
        record["fit_summary_error"] = repr(error)

    try:
        if true_model_obj is not None:
            record["event_ra_true_model"] = float(true_model_obj.event.ra)
            record["event_dec_true_model"] = float(true_model_obj.event.dec)
            record["true_parallax_model"] = repr(true_model_obj.parallax_model)

        if fit_model_obj is not None:
            record["event_ra_fit_model"] = float(fit_model_obj.event.ra)
            record["event_dec_fit_model"] = float(fit_model_obj.event.dec)
            record["fit_parallax_model"] = repr(fit_model_obj.parallax_model)

    except Exception as error:
        record["geometry_summary_error"] = repr(error)

    return record


# ============================================================
# Plot helpers compartidos
# ============================================================

def _n_points_lightcurve(tel):
    if tel.lightcurve is None:
        return 0

    try:
        return len(tel.lightcurve)
    except Exception:
        return 0


@contextmanager
def _temporarily_disable_empty_lightcurves(model_obj):
    """
    pyLIMA_plots falla si tel.lightcurve existe pero tiene len=0.
    Para plotear, ponemos esas curvas en None temporalmente.
    """

    backups = []

    if model_obj is None:
        yield model_obj
        return

    for tel in model_obj.event.telescopes:
        if tel.lightcurve is not None and _n_points_lightcurve(tel) == 0:
            backups.append((tel, tel.lightcurve))
            tel.lightcurve = None

    try:
        yield model_obj

    finally:
        for tel, original_lightcurve in backups:
            tel.lightcurve = original_lightcurve


def extract_plot_objects(simfit_result):
    """
    Extrae objetos necesarios para graficar desde un simfit_result con return_data=True.
    """

    if not isinstance(simfit_result, dict):
        raise TypeError("simfit_result debe ser un dict. Usá return_data=True.")

    if simfit_result.get("status") != "fitted":
        raise ValueError(
            f"Solo puedo graficar status='fitted'. "
            f"status={simfit_result.get('status')!r}"
        )

    true_model_obj = simfit_result["pyLIMAmodel_true"]
    true_pyLIMA_parameters = simfit_result["pyLIMA_parameters_true"]
    true_event_params = simfit_result["event_params"]

    true_model_parameters = vector_from_pylima_parameters(
        true_model_obj,
        true_pyLIMA_parameters,
    )

    fit_rr = simfit_result["fit_rr"]
    fit_model_obj = simfit_result["pyLIMAmodel_rr"]

    fit_model_parameters = np.asarray(
        fit_rr.fit_results["best_model"],
        dtype=float,
    )

    return {
        "true_model_obj": true_model_obj,
        "true_model_parameters": true_model_parameters,
        "fit_model_obj": fit_model_obj,
        "fit_model_parameters": fit_model_parameters,
        "true_event_params": true_event_params,
    }


def _get_flux_pair(pyparams, band):
    fs = float(_val(pyparams, f"fsource_{band}"))

    if _has(pyparams, f"fblend_{band}"):
        fb = float(_val(pyparams, f"fblend_{band}"))

    elif _has(pyparams, f"ftotal_{band}"):
        fb = float(
            _val(pyparams, f"ftotal_{band}")
            - _val(pyparams, f"fsource_{band}")
        )

    else:
        raise KeyError(
            f"No encontré fblend_{band} ni ftotal_{band}."
        )

    return fs, fb


def _vector_from_pyparams(model_obj, pyparams):
    """
    Reconstruye el vector de parámetros en el orden real esperado por pyLIMA.
    Se usa para la curva densa, no para la trayectoria estilo widget.
    """

    values = []
    missing = []

    for name, _ in sorted(
        model_obj.model_dictionnary.items(),
        key=lambda x: x[1],
    ):
        if _has(pyparams, name):
            values.append(float(_val(pyparams, name)))
            continue

        if name.startswith("fblend_"):
            band = name.split("_")[-1]

            if (
                _has(pyparams, f"ftotal_{band}")
                and _has(pyparams, f"fsource_{band}")
            ):
                values.append(
                    float(
                        _val(pyparams, f"ftotal_{band}")
                        - _val(pyparams, f"fsource_{band}")
                    )
                )
                continue

        if name.startswith("ftotal_"):
            band = name.split("_")[-1]

            if (
                _has(pyparams, f"fblend_{band}")
                and _has(pyparams, f"fsource_{band}")
            ):
                values.append(
                    float(
                        _val(pyparams, f"fsource_{band}")
                        + _val(pyparams, f"fblend_{band}")
                    )
                )
                continue

        missing.append(name)

    if missing:
        raise KeyError(
            f"No pude reconstruir el vector. Faltan: {missing}"
        )

    return np.asarray(values, dtype=float)


def _active_bands(model_obj):
    bands = []

    if model_obj is None:
        return bands

    for tel in model_obj.event.telescopes:
        if tel.lightcurve is None or len(tel.lightcurve) == 0:
            continue

        bands.append(str(tel.name))

    return bands


def _choose_reference_band(requested_band, active_bands, allow_fallback=True):
    requested_band = str(requested_band)

    if requested_band in active_bands:
        return requested_band

    if not allow_fallback:
        raise ValueError(
            f"reference_band={requested_band!r} no está activa. "
            f"Bandas activas: {active_bands}"
        )

    for band in BAND_PREFERENCE:
        if band in active_bands:
            print(
                f"WARNING: reference_band={requested_band!r} no está activa; "
                f"uso reference_band={band!r}."
            )
            return band

    if active_bands:
        band = str(active_bands[0])
        print(
            f"WARNING: reference_band={requested_band!r} no está activa; "
            f"uso reference_band={band!r}."
        )
        return band

    raise ValueError("No hay bandas activas para graficar.")


def _derive_fit_fluxes(fit_model_obj, fit_pyparams):
    """
    Completa fblend si pyLIMA lo deriva internamente.
    Estos flujos NO se usan para alinear los datos.
    """

    for tel in fit_model_obj.event.telescopes:
        if tel.lightcurve is None or len(tel.lightcurve) == 0:
            continue

        A = fit_model_obj.model_magnification(
            tel,
            fit_pyparams,
        )

        fit_model_obj.derive_telescope_flux(
            tel,
            fit_pyparams,
            A,
        )

    return fit_pyparams


def check_fit_fluxes_are_physical(
    fit_model_obj,
    fit_model_parameters,
):
    """
    Diagnóstico de flujos del fit. No frena el pipeline.
    """

    fit_pyparams = fit_model_obj.compute_pyLIMA_parameters(
        fit_model_parameters,
    )

    fit_pyparams = _derive_fit_fluxes(
        fit_model_obj,
        fit_pyparams,
    )

    bad = []

    print("=" * 80)
    print("FIT flux diagnostics")
    print("=" * 80)

    for tel in fit_model_obj.event.telescopes:
        if tel.lightcurve is None or len(tel.lightcurve) == 0:
            continue

        band = tel.name
        fs, fb = _get_flux_pair(fit_pyparams, band)

        ft = fs + fb
        frac = fs / ft if ft != 0.0 else np.nan

        print(
            f"{band:3s}  "
            f"fsource={fs: .6e}  "
            f"ftotal={ft: .6e}  "
            f"fblend={fb: .6e}  "
            f"fsource/ftotal={frac: .6e}"
        )

        if (
            not np.isfinite(fs)
            or not np.isfinite(fb)
            or not np.isfinite(ft)
            or fs <= 0.0
            or ft <= 0.0
            or fb < 0.0
            or fs > ft
        ):
            bad.append(
                {
                    "band": band,
                    "fsource": fs,
                    "fblend": fb,
                    "ftotal": ft,
                    "fsource_over_ftotal": frac,
                }
            )

    if bad:
        print("=" * 80)
        print("WARNING: el fit tiene flujos no físicos.")
        print("El plot usará flujos TRUE para alinear las bandas.")
        print("=" * 80)

    return pd.DataFrame(bad)


def _is_no_parallax(parallax_model):
    if parallax_model is None:
        return True

    return str(parallax_model[0]).lower() == "none"


def _force_parallax_reference_to_t0(
    parallax_model,
    t0_reference,
):
    """
    Fuerza que el epoch de referencia del paralaje sea t0_reference.
    Si no hay paralaje, devuelve ['None', 0.0].
    """

    if _is_no_parallax(parallax_model):
        return ["None", 0.0]

    return [
        parallax_model[0],
        float(t0_reference),
    ]


def _parallax_to_label(parallax_model):
    if _is_no_parallax(parallax_model):
        return "No parallax"

    return f"{parallax_model[0]} parallax"


def _make_dense_model_for_lightcurve(
    t_dense,
    band,
    ra,
    dec,
    model_name,
    parallax_model,
    pyparams,
):
    """
    Construye un modelo pyLIMA denso para calcular A(t).
    Se usa para la curva de luz, no para el inset.
    """

    from pyLIMA import event, telescopes
    from set_model_pyLIMA import build_pyLIMA_model

    lc_dense = np.column_stack(
        [
            t_dense,
            np.ones_like(t_dense) * 20.0,
            np.ones_like(t_dense) * 0.01,
        ]
    )

    e_dense = event.Event(
        ra=float(ra),
        dec=float(dec),
    )

    e_dense.name = f"dense_{model_name}_{band}"

    tel_dense = telescopes.Telescope(
        name=band,
        camera_filter=band,
        lightcurve=lc_dense.astype(float),
        lightcurve_names=["time", "mag", "err_mag"],
        lightcurve_units=["JD", "mag", "mag"],
        location="Earth",
    )

    tel_dense.ld_gamma = 0.0
    e_dense.telescopes.append(tel_dense)
    e_dense.check_event()

    use_parallax = not _is_no_parallax(parallax_model)
    t0_parallax = float(parallax_model[1])

    dense_model = build_pyLIMA_model(
        pyLIMA_event=e_dense,
        model=model_name,
        use_parallax=use_parallax,
        t0_parallax=t0_parallax,
        origin=None,
        random_origin=False,
        blend_flux_parameter="ftotal",
    )

    dense_parameters = _vector_from_pyparams(
        dense_model,
        pyparams,
    )

    dense_pyparams = dense_model.compute_pyLIMA_parameters(
        dense_parameters,
    )

    A_dense = dense_model.model_magnification(
        dense_model.event.telescopes[0],
        dense_pyparams,
    )

    return dense_model, dense_parameters, dense_pyparams, A_dense



# ============================================================
# Caja de parámetros true/fit para figuras aligned_inset
# ============================================================

def _safe_float(value, default=np.nan):
    """
    Conversión robusta a float para textos de parámetros.
    """
    try:
        if value is None:
            return default

        if isinstance(value, str):
            if value.strip() == "" or value.strip().lower() in {"nan", "none"}:
                return default

        out = float(value)

        return out

    except Exception:
        return default


def _fmt_float(value, digits=4, sci=False, signed=False):
    """
    Formato compacto para la caja de parámetros.
    """
    value = _safe_float(value)

    if not np.isfinite(value):
        return r"--"

    sign = "+" if signed else ""

    if sci:
        return rf"{value:{sign}.{digits}e}"

    return rf"{value:{sign}.{digits}f}"


def _fmt_days(value, digits=3):
    value = _safe_float(value)

    if not np.isfinite(value):
        return r"--"

    return rf"{value:.{digits}f}"


def _parameter_box_get_chi2_from_fit_results(fit_rr):
    """
    Extrae chi2 del objeto fit_rr si está disponible.
    """
    if fit_rr is None:
        return np.nan

    try:
        fit_results = getattr(fit_rr, "fit_results", {})
        if isinstance(fit_results, dict):
            for key in ["chi2", "chichi"]:
                if key in fit_results:
                    return _safe_float(fit_results[key])
    except Exception:
        pass

    return np.nan


def _parameter_box_best_dof(fit_model_obj, fit_model_parameters):
    """
    Estima dof = N_fit - N_params para mostrar chi2_red en la figura.
    """
    try:
        n_data = int(sum(count_points_by_telescope(fit_model_obj).values()))
        n_params = int(len(fit_model_parameters))
        dof = n_data - n_params

        if dof > 0:
            return dof

    except Exception:
        pass

    return np.nan


def _parameter_box_get_fit_results_dict(fit_rr):
    """
    Devuelve fit_rr.fit_results si existe y es un diccionario.
    """
    if fit_rr is None:
        return {}

    try:
        fit_results = getattr(fit_rr, "fit_results", {})
        if isinstance(fit_results, dict):
            return fit_results
    except Exception:
        pass

    return {}


def _parameter_box_parameter_names(fit_model_obj, fit_model_parameters=None):
    """
    Nombres de parámetros en el orden del vector best_model/covariance.
    Se usa para ubicar rho en la matriz de covarianza.
    """
    try:
        model_dict = getattr(fit_model_obj, "model_dictionnary", {})

        if isinstance(model_dict, dict) and len(model_dict) > 0:
            ordered = sorted(model_dict.items(), key=lambda item: item[1])
            names = [str(name) for name, _ in ordered]

            if fit_model_parameters is not None:
                n = len(fit_model_parameters)
                names = names[:n]

            return names

    except Exception:
        pass

    return []


def _parameter_box_extract_rho_sigma(
    fit_rr,
    fit_model_obj,
    fit_model_parameters,
    fit_pyparams=None,
):
    """
    Extrae sigma_rho del fit, preferentemente desde covariance_matrix.

    Orden de búsqueda:
      1. rho_err / sigma_rho en fit_results, si existe.
      2. sqrt(covariance_matrix[rho_index, rho_index]), usando
         fit_model_obj.model_dictionnary para ubicar rho.
      3. rho_err en fit_pyparams, si existiera.

    Devuelve un diccionario con sigma_rho y fuente diagnóstica.
    """
    out = {
        "rho_err": np.nan,
        "sigma_rho": np.nan,
        "rho_err_source": "",
        "rho_covariance_index": np.nan,
        "rho_covariance_variance": np.nan,
        "covariance_matrix_shape": "",
        "covariance_parameter_names": "",
    }

    fit_results = _parameter_box_get_fit_results_dict(fit_rr)

    # 1. Si el fitter ya guardó rho_err o sigma_rho.
    for key in [
        "rho_err",
        "sigma_rho",
        "rho_sigma",
        "err_rho",
        "rho_err_cov",
    ]:
        if key in fit_results:
            val = _safe_float(fit_results.get(key))
            if np.isfinite(val) and val >= 0.0:
                out["rho_err"] = val
                out["sigma_rho"] = val
                out["rho_err_source"] = f"fit_results[{key}]"
                return out

    # 2. Matriz de covarianza.
    covariance = None
    covariance_key = None

    for key in [
        "covariance_matrix",
        "covariance",
        "covariance_mat",
        "covariance_best_model",
    ]:
        if key in fit_results:
            covariance = fit_results.get(key)
            covariance_key = key
            break

    if covariance is not None:
        try:
            cov = np.asarray(covariance, dtype=float)
            out["covariance_matrix_shape"] = str(cov.shape)

            names = _parameter_box_parameter_names(
                fit_model_obj=fit_model_obj,
                fit_model_parameters=fit_model_parameters,
            )
            out["covariance_parameter_names"] = ",".join(names)

            rho_index = None
            if "rho" in names:
                rho_index = names.index("rho")
            else:
                # Para FSPL sin paralaje el orden mínimo suele ser t0,u0,tE,rho.
                if cov.ndim == 2 and cov.shape[0] > 3:
                    rho_index = 3

            if (
                cov.ndim == 2
                and cov.shape[0] == cov.shape[1]
                and rho_index is not None
                and 0 <= int(rho_index) < cov.shape[0]
            ):
                var = float(cov[int(rho_index), int(rho_index)])
                out["rho_covariance_index"] = int(rho_index)
                out["rho_covariance_variance"] = var

                if np.isfinite(var) and var >= 0.0:
                    sigma = float(np.sqrt(var))
                    out["rho_err"] = sigma
                    out["sigma_rho"] = sigma
                    out["rho_err_source"] = f"sqrt({covariance_key}[rho,rho])"
                    return out

        except Exception:
            pass

    # 3. Fallback improbable: pyparams con rho_err.
    if fit_pyparams is not None:
        for key in ["rho_err", "sigma_rho", "rho_sigma", "err_rho"]:
            if _has(fit_pyparams, key):
                val = _safe_float(_val(fit_pyparams, key))
                if np.isfinite(val) and val >= 0.0:
                    out["rho_err"] = val
                    out["sigma_rho"] = val
                    out["rho_err_source"] = f"fit_pyparams.{key}"
                    return out

    return out


def _extract_parameter_box_fit_diagnostics(
    simfit_result,
    fit_model_obj,
    fit_model_parameters,
    fit_pyparams=None,
):
    """
    Extrae diagnósticos del ajuste para mostrar en la caja de parámetros.
    No frena el plot si algo no está disponible.
    """

    diagnostics = {
        "chi2_fit": np.nan,
        "dof_fit": np.nan,
        "chi2_red": np.nan,
        "chi2_true": np.nan,
        "delta_chi2_true": np.nan,
        "rho_err": np.nan,
        "sigma_rho": np.nan,
        "sigma_rho_over_rho_fit": np.nan,
        "sigma_rho_over_rho_true": np.nan,
        "rho_err_source": "",
        "rho_covariance_index": np.nan,
        "rho_covariance_variance": np.nan,
        "covariance_matrix_shape": "",
        "covariance_parameter_names": "",
    }

    if not isinstance(simfit_result, dict):
        return diagnostics

    # Optional hard override injected by plot_aligned_inset_from_result().
    # This is the safest route because the runner has already saved the
    # formal pyLIMA quantities in results/fit_rr/fit_rr_*.parquet before
    # the figure is generated.  In particular, chi2_true is usually there,
    # not inside fit_pyparams.
    override = simfit_result.get("plot_fit_diagnostics", None)
    if isinstance(override, dict):
        key_map = {
            "chichi": "chi2_fit",
            "chi2": "chi2_fit",
            "chi2_fit": "chi2_fit",
            "dof": "dof_fit",
            "dof_fit": "dof_fit",
            "chi2_red": "chi2_red",
            "chi2_true": "chi2_true",
            "true_chi2": "chi2_true",
            "delta_chi2_true": "delta_chi2_true",
            "delta_chi2": "delta_chi2_true",
            "rho_err": "rho_err",
            "sigma_rho": "sigma_rho",
        }

        for in_key, out_key in key_map.items():
            if in_key in override:
                value = _safe_float(override.get(in_key))
                if np.isfinite(value):
                    diagnostics[out_key] = value

        if np.isfinite(diagnostics["rho_err"]):
            diagnostics["sigma_rho"] = diagnostics["rho_err"]
            diagnostics["rho_err_source"] = "fit_rr parquet"

        # If override contains enough information, complete derived values.
        if (
            np.isfinite(diagnostics["chi2_fit"])
            and np.isfinite(diagnostics["dof_fit"])
            and diagnostics["dof_fit"] > 0
            and not np.isfinite(diagnostics["chi2_red"])
        ):
            diagnostics["chi2_red"] = diagnostics["chi2_fit"] / diagnostics["dof_fit"]

        if (
            np.isfinite(diagnostics["chi2_fit"])
            and np.isfinite(diagnostics["chi2_true"])
            and not np.isfinite(diagnostics["delta_chi2_true"])
        ):
            diagnostics["delta_chi2_true"] = (
                diagnostics["chi2_fit"] - diagnostics["chi2_true"]
            )

        if (
            np.isfinite(diagnostics["chi2_fit"])
            and np.isfinite(diagnostics["delta_chi2_true"])
            and not np.isfinite(diagnostics["chi2_true"])
        ):
            diagnostics["chi2_true"] = (
                diagnostics["chi2_fit"] - diagnostics["delta_chi2_true"]
            )

    fit_rr = simfit_result.get("fit_rr", None)
    event_params = simfit_result.get("event_params", {})

    rho_sigma_info = _parameter_box_extract_rho_sigma(
        fit_rr=fit_rr,
        fit_model_obj=fit_model_obj,
        fit_model_parameters=fit_model_parameters,
        fit_pyparams=fit_pyparams,
    )
    diagnostics.update(rho_sigma_info)

    rho_fit_for_sigma = np.nan
    if fit_pyparams is not None and _has(fit_pyparams, "rho"):
        rho_fit_for_sigma = _safe_float(_val(fit_pyparams, "rho"))

    rho_true_for_sigma = np.nan
    event_params = simfit_result.get("event_params", {})
    if isinstance(event_params, dict) and "rho" in event_params:
        rho_true_for_sigma = _safe_float(event_params.get("rho"))

    if np.isfinite(diagnostics["sigma_rho"]):
        if np.isfinite(rho_fit_for_sigma) and rho_fit_for_sigma != 0.0:
            diagnostics["sigma_rho_over_rho_fit"] = (
                diagnostics["sigma_rho"] / abs(rho_fit_for_sigma)
            )

        if np.isfinite(rho_true_for_sigma) and rho_true_for_sigma != 0.0:
            diagnostics["sigma_rho_over_rho_true"] = (
                diagnostics["sigma_rho"] / abs(rho_true_for_sigma)
            )

    chi2_fit = np.nan

    if fit_pyparams is not None and _has(fit_pyparams, "chi2"):
        chi2_fit = _safe_float(_val(fit_pyparams, "chi2"))

    if not np.isfinite(chi2_fit):
        chi2_fit = _parameter_box_get_chi2_from_fit_results(fit_rr)

    diagnostics["chi2_fit"] = chi2_fit

    # dof normalmente no está en fit_results; lo estimamos.
    dof_fit = _parameter_box_best_dof(
        fit_model_obj=fit_model_obj,
        fit_model_parameters=fit_model_parameters,
    )

    diagnostics["dof_fit"] = dof_fit

    if np.isfinite(chi2_fit) and np.isfinite(dof_fit) and dof_fit > 0:
        diagnostics["chi2_red"] = chi2_fit / dof_fit

    # chi2_true / delta_chi2_true pueden existir a distintos niveles,
    # dependiendo de la versión del runner.
    for key in ["chi2_true", "true_chi2", "chi2_true_rr"]:
        if key in simfit_result:
            diagnostics["chi2_true"] = _safe_float(simfit_result.get(key))
            break

    for key in ["delta_chi2_true", "delta_chi2", "Delta_chi2"]:
        if key in simfit_result:
            diagnostics["delta_chi2_true"] = _safe_float(simfit_result.get(key))
            break

    # Intento desde event_params. En este pipeline chi2_true suele vivir
    # en simfit_result["event_params"], porque viene de la curva simulada
    # FSPL+parallax evaluada por el runner.
    if isinstance(event_params, dict):
        if not np.isfinite(diagnostics["chi2_true"]):
            for key in ["chi2_true", "true_chi2", "chi2_true_rr"]:
                if key in event_params:
                    diagnostics["chi2_true"] = _safe_float(event_params.get(key))
                    break

        if not np.isfinite(diagnostics["delta_chi2_true"]):
            for key in ["delta_chi2_true", "delta_chi2", "Delta_chi2"]:
                if key in event_params:
                    diagnostics["delta_chi2_true"] = _safe_float(event_params.get(key))
                    break

    # Intento adicional desde fit_results.
    try:
        fit_results = getattr(fit_rr, "fit_results", {})
        if isinstance(fit_results, dict):
            if not np.isfinite(diagnostics["delta_chi2_true"]):
                for key in ["delta_chi2_true", "delta_chi2", "Delta_chi2"]:
                    if key in fit_results:
                        diagnostics["delta_chi2_true"] = _safe_float(fit_results[key])
                        break

            if not np.isfinite(diagnostics["chi2_true"]):
                for key in ["chi2_true", "true_chi2", "chi2_true_rr"]:
                    if key in fit_results:
                        diagnostics["chi2_true"] = _safe_float(fit_results[key])
                        break

    except Exception:
        pass

    # Si tengo chi2_fit y delta, infiero chi2_true.
    if (
        np.isfinite(diagnostics["chi2_fit"])
        and np.isfinite(diagnostics["delta_chi2_true"])
        and not np.isfinite(diagnostics["chi2_true"])
    ):
        diagnostics["chi2_true"] = (
            diagnostics["chi2_fit"]
            - diagnostics["delta_chi2_true"]
        )

    # Si tengo chi2_fit y chi2_true, infiero delta.
    if (
        np.isfinite(diagnostics["chi2_fit"])
        and np.isfinite(diagnostics["chi2_true"])
        and not np.isfinite(diagnostics["delta_chi2_true"])
    ):
        diagnostics["delta_chi2_true"] = (
            diagnostics["chi2_fit"]
            - diagnostics["chi2_true"]
        )

    return diagnostics


def _make_true_fit_parameter_box_text(
    true_pyparams,
    fit_pyparams,
    fit_diagnostics=None,
):
    """
    Construye el texto de parámetros true y fit para aligned_inset.
    """

    if fit_diagnostics is None:
        fit_diagnostics = {}

    t0_true = _safe_float(_val(true_pyparams, "t0"))
    u0_true = _safe_float(_val(true_pyparams, "u0"))
    tE_true = _safe_float(_val(true_pyparams, "tE"))
    rho_true = _safe_float(_val(true_pyparams, "rho"))

    piEN_true = _safe_float(_val(true_pyparams, "piEN"))
    piEE_true = _safe_float(_val(true_pyparams, "piEE"))

    if np.isfinite(piEN_true) and np.isfinite(piEE_true):
        piE_true = float(np.hypot(piEN_true, piEE_true))
    else:
        piE_true = np.nan

    t0_fit = _safe_float(_val(fit_pyparams, "t0"))
    u0_fit = _safe_float(_val(fit_pyparams, "u0"))
    tE_fit = _safe_float(_val(fit_pyparams, "tE"))
    rho_fit = _safe_float(_val(fit_pyparams, "rho"))

    chi2_red = _safe_float(fit_diagnostics.get("chi2_red", np.nan))
    chi2_fit = _safe_float(fit_diagnostics.get("chi2_fit", np.nan))
    chi2_true = _safe_float(fit_diagnostics.get("chi2_true", np.nan))
    dof_fit = _safe_float(fit_diagnostics.get("dof_fit", np.nan))

    # Usamos exclusivamente el chi2 formal de pyLIMA / runner, no el chi2
    # recalculado en la escala alineada de la figura. La escala alineada usa
    # flujos TRUE de referencia y no corresponde al likelihood del fit.
    delta_chi2_display = _safe_float(
        fit_diagnostics.get("delta_chi2_true", np.nan)
    )

    if (
        not np.isfinite(delta_chi2_display)
        and np.isfinite(chi2_fit)
        and np.isfinite(chi2_true)
    ):
        delta_chi2_display = chi2_fit - chi2_true

    sigma_rho = _safe_float(fit_diagnostics.get("sigma_rho", np.nan))
    sigma_rho_over_rho_fit = _safe_float(
        fit_diagnostics.get("sigma_rho_over_rho_fit", np.nan)
    )
    sigma_rho_over_rho_true = _safe_float(
        fit_diagnostics.get("sigma_rho_over_rho_true", np.nan)
    )

    t0_offset = t0_fit - t0_true if np.isfinite(t0_fit) and np.isfinite(t0_true) else np.nan
    tE_ratio = tE_fit / tE_true if np.isfinite(tE_fit) and np.isfinite(tE_true) and tE_true != 0 else np.nan
    rho_ratio = rho_fit / rho_true if np.isfinite(rho_fit) and np.isfinite(rho_true) and rho_true > 0 else np.nan

    text = (
        r"$\bf{True:\ FSPL+parallax}$" "\n"
        rf"$t_0={_fmt_days(t0_true, 3)}$" "\n"
        rf"$u_0={_fmt_float(u0_true, 4)}$" "\n"
        rf"$t_E={_fmt_float(tE_true, 3)}\,{{\rm d}}$" "\n"
        rf"$\rho={_fmt_float(rho_true, 3, sci=True)}$" "\n"
        rf"$\pi_{{E,N}}={_fmt_float(piEN_true, 4)}$" "\n"
        rf"$\pi_{{E,E}}={_fmt_float(piEE_true, 4)}$" "\n"
        rf"$\pi_E={_fmt_float(piE_true, 4)}$" "\n"
        "\n"
        r"$\bf{Fit:\ FSPL,\ no\ parallax}$" "\n"
        rf"$t_0={_fmt_days(t0_fit, 3)}$" "\n"
        rf"$u_0={_fmt_float(u0_fit, 4)}$" "\n"
        rf"$t_E={_fmt_float(tE_fit, 3)}\,{{\rm d}}$" "\n"
        rf"$\rho={_fmt_float(rho_fit, 3, sci=True)}$" "\n"
        rf"$\sigma_{{\rho}}={_fmt_float(sigma_rho, 3, sci=True)}$" "\n"
        rf"$\sigma_{{\rho}}/\rho_{{\rm fit}}={_fmt_float(sigma_rho_over_rho_fit, 3)}$" "\n"
        rf"$\sigma_{{\rho}}/\rho_{{\rm true}}={_fmt_float(sigma_rho_over_rho_true, 3)}$" "\n"
        "\n"
        r"$\bf{Diagnostics}$" "\n"
        rf"$\chi^2_{{\rm red}}={_fmt_float(chi2_red, 3)}$" "\n"
        rf"$\chi^2_{{\rm fit}}={_fmt_float(chi2_fit, 2)}$" "\n"
        rf"$\chi^2_{{\rm true}}={_fmt_float(chi2_true, 2)}$" "\n"
        rf"${{\rm dof}}={_fmt_float(dof_fit, 0)}$" "\n"
        rf"$\Delta\chi^2={_fmt_float(delta_chi2_display, 3)}$" "\n"
        rf"$\Delta t_0={_fmt_float(t0_offset, 3, signed=True)}\,{{\rm d}}$" "\n"
        rf"$t_{{E,\rm fit}}/t_{{E,\rm true}}={_fmt_float(tE_ratio, 3)}$" "\n"
        rf"$\rho_{{\rm fit}}/\rho_{{\rm true}}={_fmt_float(rho_ratio, 3)}$"
    )

    return text


def _add_true_fit_parameter_box(
    fig,
    true_pyparams,
    fit_pyparams,
    fit_diagnostics=None,
    x=0.765,
    y=0.50,
    fontsize=7.5,
):
    """
    Agrega caja de parámetros en el margen derecho de la figura.
    """

    text = _make_true_fit_parameter_box_text(
        true_pyparams=true_pyparams,
        fit_pyparams=fit_pyparams,
        fit_diagnostics=fit_diagnostics,
    )

    fig.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=fontsize,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="0.55",
            alpha=0.94,
        ),
    )


# ============================================================
# Trayectoria estilo widget
# ============================================================

def _build_sim_event_widget_style(
    t,
    ra,
    dec,
    mag0=19.0,
    emag=1e-6,
    filt="g",
):
    from pyLIMA import event, telescopes

    ev = event.Event()
    ev.name = "FSPL widget-style trajectory"
    ev.ra = float(ra)
    ev.dec = float(dec)

    lc = np.c_[
        t,
        np.full_like(t, mag0),
        np.full_like(t, emag),
    ]

    tel = telescopes.Telescope(
        name="Simulation",
        camera_filter=filt,
        lightcurve=lc.astype(float),
        lightcurve_names=["time", "mag", "err_mag"],
        lightcurve_units=["JD", "mag", "mag"],
        location="Earth",
    )

    tel.ld_gamma = 0.0
    ev.telescopes.append(tel)

    return ev


def _make_pylima_parameters_widget_style(model, values):
    vector = []

    for parameter_name in model.model_dictionnary.keys():
        vector.append(values.get(parameter_name, None))

    return model.compute_pyLIMA_parameters(vector)


def _compute_fspl_trajectory_widget_style(
    t,
    ra,
    dec,
    values,
    parallax,
    filt="g",
):
    from pyLIMA.models import FSPL_model

    ev = _build_sim_event_widget_style(
        t=t,
        ra=ra,
        dec=dec,
        filt=filt,
    )

    tel = ev.telescopes[0]

    if str(parallax[0]).lower() == "none":
        model = FSPL_model.FSPLmodel(
            ev,
            parallax=parallax,
        )
    else:
        with redirect_stdout(io.StringIO()):
            model = FSPL_model.FSPLmodel(
                ev,
                parallax=parallax,
            )

    params = _make_pylima_parameters_widget_style(
        model,
        values,
    )

    traj = model.sources_trajectory(
        tel,
        params,
        data_type="photometry",
    )

    x = _arr(traj[0])
    y = _arr(traj[1])

    return {
        "x": x,
        "y": y,
        "t": np.asarray(t, dtype=float),
        "model": model,
        "params": params,
        "event": ev,
        "telescope": tel,
        "parallax": parallax,
        "values": values,
    }


def _build_trajectory_specs_widget_style(
    true_pyparams,
    fit_pyparams,
    fit_model_obj,
    t_traj,
    ra,
    dec,
    reference_band="g",
    show_true_no_parallax_reference=False,
    fit_trajectory_time_mode="own_fit_tE_window",
):
    """
    Devuelve trajectory_specs y traj_data usando el código estilo widget.

    Modos para la trayectoria del fit:

    - same_true_window:
        Evalúa el fit en la misma grilla temporal del evento TRUE.
        Esto compara la predicción del fit en los mismos tiempos observados.

    - own_fit_tE_window:
        Evalúa el fit alrededor de t0_fit con la misma ventana en unidades de tE
        que se usó para el TRUE. Esto es útil para visualizar la geometría propia
        del modelo ajustado, aunque ya no sean los mismos tiempos observados.
    """

    t_traj = np.asarray(t_traj, dtype=float)

    t0_true = float(_val(true_pyparams, "t0"))
    u0_true = float(_val(true_pyparams, "u0"))
    tE_true = float(_val(true_pyparams, "tE"))
    rho_true = float(_val(true_pyparams, "rho"))

    if not np.isfinite(tE_true) or tE_true == 0.0:
        raise RuntimeError(f"tE_true inválido para graficar trayectoria: {tE_true}")

    tE_true_abs = abs(tE_true)

    piEN_true = float(_val(true_pyparams, "piEN")) if _has(true_pyparams, "piEN") else 0.0
    piEE_true = float(_val(true_pyparams, "piEE")) if _has(true_pyparams, "piEE") else 0.0

    true_values_parallax = {
        "t0": t0_true,
        "u0": u0_true,
        "tE": tE_true_abs,
        "rho": rho_true,
        "piEN": piEN_true,
        "piEE": piEE_true,
    }

    true_parallax = _compute_fspl_trajectory_widget_style(
        t=t_traj,
        ra=ra,
        dec=dec,
        values=true_values_parallax,
        parallax=["Full", t0_true],
        filt=reference_band,
    )

    trajectory_specs = [
        {
            "x": true_parallax["x"],
            "y": true_parallax["y"],
            "label": "True trajectory",
            "color": "k",
            "ls": "-",
            "lw": 1.9,
            "draw_rho": True,
        }
    ]

    true_no_parallax = None

    if show_true_no_parallax_reference:
        true_values_no_parallax = {
            "t0": t0_true,
            "u0": u0_true,
            "tE": tE_true_abs,
            "rho": rho_true,
        }

        true_no_parallax = _compute_fspl_trajectory_widget_style(
            t=t_traj,
            ra=ra,
            dec=dec,
            values=true_values_no_parallax,
            parallax=["None", t0_true],
            filt=reference_band,
        )

        trajectory_specs.append(
            {
                "x": true_no_parallax["x"],
                "y": true_no_parallax["y"],
                "label": "True no-parallax trajectory",
                "color": "C0",
                "ls": ":",
                "lw": 1.8,
                "draw_rho": False,
            }
        )

    fit_traj = None
    fit_values = None
    t_traj_fit = None

    if fit_pyparams is not None:
        fit_t0 = float(_val(fit_pyparams, "t0"))
        fit_u0 = float(_val(fit_pyparams, "u0"))
        fit_tE = float(_val(fit_pyparams, "tE"))

        if not np.isfinite(fit_tE) or fit_tE == 0.0:
            raise RuntimeError(f"tE_fit inválido para graficar trayectoria: {fit_tE}")

        fit_tE_abs = abs(fit_tE)

        fit_rho = (
            float(_val(fit_pyparams, "rho"))
            if _has(fit_pyparams, "rho")
            else rho_true
        )

        fit_parallax_model = getattr(
            fit_model_obj,
            "parallax_model",
            ["None", fit_t0],
        )

        if str(fit_parallax_model[0]).lower() == "none":
            fit_values = {
                "t0": fit_t0,
                "u0": fit_u0,
                "tE": fit_tE_abs,
                "rho": fit_rho,
            }

            fit_parallax_for_plot = ["None", fit_t0]

        else:
            fit_values = {
                "t0": fit_t0,
                "u0": fit_u0,
                "tE": fit_tE_abs,
                "rho": fit_rho,
                "piEN": float(_val(fit_pyparams, "piEN")),
                "piEE": float(_val(fit_pyparams, "piEE")),
            }

            fit_parallax_for_plot = [
                fit_parallax_model[0],
                fit_t0,
            ]

        # --------------------------------------------------------
        # Grilla temporal para la trayectoria del fit
        # --------------------------------------------------------

        if fit_trajectory_time_mode == "same_true_window":
            t_traj_fit = t_traj

        elif fit_trajectory_time_mode == "own_fit_tE_window":
            # Número de tE usado para el TRUE, deducido de la grilla t_traj.
            half_window_true_days = 0.5 * (np.nanmax(t_traj) - np.nanmin(t_traj))
            inset_n_tE_equiv = half_window_true_days / tE_true_abs

            t_traj_fit = np.linspace(
                fit_t0 - inset_n_tE_equiv * fit_tE_abs,
                fit_t0 + inset_n_tE_equiv * fit_tE_abs,
                len(t_traj),
            )

        else:
            raise ValueError(
                "fit_trajectory_time_mode debe ser "
                "'same_true_window' o 'own_fit_tE_window'. "
                f"Valor recibido: {fit_trajectory_time_mode!r}"
            )

        fit_traj = _compute_fspl_trajectory_widget_style(
            t=t_traj_fit,
            ra=ra,
            dec=dec,
            values=fit_values,
            parallax=fit_parallax_for_plot,
            filt=reference_band,
        )

        trajectory_specs.append(
            {
                "x": fit_traj["x"],
                "y": fit_traj["y"],
                "label": "Fit trajectory",
                "color": "purple",
                "ls": "--",
                "lw": 1.9,
                "draw_rho": False,
            }
        )

    traj_data = {
        "t": t_traj,  # compatibilidad hacia atrás
        "t_true": t_traj,
        "t_fit": t_traj_fit,
        "true_parallax": true_parallax,
        "true_no_parallax": true_no_parallax,
        "fit": fit_traj,
        "true_values": true_values_parallax,
        "fit_values": fit_values,
        "fit_trajectory_time_mode": fit_trajectory_time_mode,
    }

    print("=" * 80)
    print("WIDGET-STYLE TRAJECTORY CHECK")
    print("=" * 80)
    print("RA, Dec =", ra, dec)
    print("TRUE parallax used =", ["Full", t0_true])
    print("TRUE t0, u0, tE, rho =", t0_true, u0_true, tE_true_abs, rho_true)
    print("TRUE piEN, piEE =", piEN_true, piEE_true)
    print(
        "TRUE trajectory time window =",
        float(np.nanmin(t_traj)),
        float(np.nanmax(t_traj)),
    )
    print(
        "TRUE parallax min u =",
        np.nanmin(np.hypot(true_parallax["x"], true_parallax["y"])),
    )

    if true_no_parallax is not None:
        print(
            "TRUE no-parallax min u =",
            np.nanmin(np.hypot(true_no_parallax["x"], true_no_parallax["y"])),
        )

    if fit_traj is not None:
        print("FIT trajectory time mode =", fit_trajectory_time_mode)
        print("FIT parallax used =", fit_traj["parallax"])
        print("FIT values =", fit_values)
        print(
            "FIT trajectory time window =",
            float(np.nanmin(t_traj_fit)),
            float(np.nanmax(t_traj_fit)),
        )
        print(
            "FIT min u on plotted fit-time grid =",
            np.nanmin(np.hypot(fit_traj["x"], fit_traj["y"])),
        )

    print("=" * 80)

    return trajectory_specs, traj_data


def _add_direction_arrows(ax, x, y, color, n_arrows=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 4:
        return

    idxs = np.linspace(
        1,
        len(x) - 2,
        n_arrows,
        dtype=int,
    )

    for idx in idxs:
        dx = x[idx + 1] - x[idx - 1]
        dy = y[idx + 1] - y[idx - 1]

        norm = np.hypot(dx, dy)

        if not np.isfinite(norm) or norm == 0:
            continue

        scale = 0.08

        ax.annotate(
            "",
            xy=(
                x[idx] + scale * dx / norm,
                y[idx] + scale * dy / norm,
            ),
            xytext=(x[idx], y[idx]),
            arrowprops=dict(
                arrowstyle="->",
                mutation_scale=11,
                color=color,
                lw=1.1,
            ),
        )


def _add_trajectory_inset(
    ax,
    trajectory_specs,
    rho_true=None,
    loc="upper left",
    width="35%",
    height="45%",
):
    """
    Agrega inset usando trajectory_specs ya calculado estilo widget.
    La leyenda del inset solo deja theta_E y Lens.
    """

    axins = inset_axes(
        ax,
        width=width,
        height=height,
        loc=loc,
        borderpad=1.2,
    )

    phi = np.linspace(0.0, 2.0 * np.pi, 400)

    axins.plot(
        np.cos(phi),
        np.sin(phi),
        color="0.5",
        ls=":",
        lw=1.0,
        alpha=0.8,
        label=r"$\theta_E$",
    )

    axins.scatter(
        [0.0],
        [0.0],
        marker="+",
        s=90,
        color="k",
        zorder=10,
        label="Lens",
    )

    all_x = [np.array([-1.0, 1.0])]
    all_y = [np.array([-1.0, 1.0])]

    for spec in trajectory_specs:
        x = np.asarray(spec["x"], dtype=float)
        y = np.asarray(spec["y"], dtype=float)

        color = spec.get("color", None)
        ls = spec.get("ls", "-")
        lw = spec.get("lw", 1.8)

        axins.plot(
            x,
            y,
            color=color,
            ls=ls,
            lw=lw,
            label="_nolegend_",
        )

        _add_direction_arrows(
            axins,
            x,
            y,
            color=color,
            n_arrows=3,
        )

        u = np.hypot(x, y)

        if np.any(np.isfinite(u)):
            i_min = int(np.nanargmin(u))

            axins.scatter(
                [x[i_min]],
                [y[i_min]],
                s=24,
                color=color,
                zorder=8,
                label="_nolegend_",
            )

            if spec.get("draw_rho", False):
                if rho_true is not None and np.isfinite(rho_true) and rho_true > 0:
                    source_circle = Circle(
                        (x[i_min], y[i_min]),
                        rho_true,
                        fill=False,
                        color=color,
                        lw=1.0,
                        alpha=0.9,
                    )

                    axins.add_patch(source_circle)

        all_x.append(x[np.isfinite(x)])
        all_y.append(y[np.isfinite(y)])

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
    y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    half = 0.58 * max(
        x_max - x_min,
        y_max - y_min,
        2.0,
    )

    axins.set_xlim(cx - half, cx + half)
    axins.set_ylim(cy - half, cy + half)

    axins.axhline(0, color="k", lw=0.5, alpha=0.25)
    axins.axvline(0, color="k", lw=0.5, alpha=0.25)

    axins.set_aspect("equal", adjustable="box")
    axins.set_xlabel(r"$u_x$", fontsize=8)
    axins.set_ylabel(r"$u_y$", fontsize=8)
    axins.tick_params(labelsize=7)
    axins.grid(alpha=0.20)
    axins.legend(fontsize=7, loc="best", framealpha=0.85)

    return axins


def plot_event_aligned_true_fit_with_widget_style_trajectory(
    true_model_obj,
    true_model_parameters,
    fit_model_obj,
    fit_model_parameters,
    true_params=None,
    reference_band="g",
    allow_reference_band_fallback=True,
    n_dense=10000,
    n_tE=10.0,
    inset_n_tE=4.0,
    true_model_name="FSPL",
    fit_model_name="FSPL",
    skip_bad_aligned_points=True,
    inset_loc="upper left",
    force_true_parallax_reference_to_t0=True,
    show_true_no_parallax_reference=False,
    fit_trajectory_time_mode="own_fit_tE_window",
    fit_diagnostics=None,
    show_parameter_box=True,
):
    """
    Curva de luz + residuales + inset de trayectoria.

    Curva de luz:
        - datos alineados en escala de flujo TRUE de reference_band;
        - conversión mag/flux con zero-point por banda;
        - flujos del fit NO se usan para alinear datos.

    Trayectoria:
        - usa FSPL_model.FSPLmodel(...).sources_trajectory(...), estilo widget.
    """

    true_pyparams = true_model_obj.compute_pyLIMA_parameters(
        true_model_parameters,
    )

    fit_pyparams = fit_model_obj.compute_pyLIMA_parameters(
        fit_model_parameters,
    )

    fit_pyparams = _derive_fit_fluxes(
        fit_model_obj,
        fit_pyparams,
    )

    t0 = float(_val(true_pyparams, "t0"))
    tE = float(_val(true_pyparams, "tE"))
    fit_t0 = float(_val(fit_pyparams, "t0"))
    fit_tE = float(_val(fit_pyparams, "tE"))

    print("t0 usado para graficar =", t0)
    print("tE usado para graficar =", tE)
    print("t0 fit                 =", fit_t0)
    print("tE fit                 =", fit_tE)

    if true_params is not None:
        print("t0 en true_params      =", float(_val(true_params, "t0")))
        print("tE en true_params      =", float(_val(true_params, "tE")))

    ra = float(true_model_obj.event.ra)
    dec = float(true_model_obj.event.dec)

    active_bands = _active_bands(fit_model_obj)

    reference_band = _choose_reference_band(
        requested_band=reference_band,
        active_bands=active_bands,
        allow_fallback=allow_reference_band_fallback,
    )

    if force_true_parallax_reference_to_t0:
        true_parallax_model_for_plot = _force_parallax_reference_to_t0(
            true_model_obj.parallax_model,
            t0,
        )
    else:
        true_parallax_model_for_plot = true_model_obj.parallax_model

    if _is_no_parallax(fit_model_obj.parallax_model):
        fit_parallax_model_for_plot = ["None", 0.0]
    else:
        fit_parallax_model_for_plot = _force_parallax_reference_to_t0(
            fit_model_obj.parallax_model,
            fit_t0,
        )

    print("=" * 80)
    print("PARALLAX REFERENCE CHECK")
    print("=" * 80)
    print("true original parallax_model =", true_model_obj.parallax_model)
    print("true plot parallax_model     =", true_parallax_model_for_plot)
    print("fit original parallax_model  =", fit_model_obj.parallax_model)
    print("fit plot parallax_model      =", fit_parallax_model_for_plot)
    print("=" * 80)

    fsource_ref, fblend_ref = _get_flux_pair(
        true_pyparams,
        reference_band,
    )

    if fsource_ref <= 0.0 or not np.isfinite(fsource_ref):
        raise RuntimeError(
            f"fsource TRUE inválido en {reference_band}: {fsource_ref}"
        )

    t_dense = np.linspace(
        t0 - n_tE * tE,
        t0 + n_tE * tE,
        int(n_dense),
    )

    true_dense_model, _, _, A_true_dense = _make_dense_model_for_lightcurve(
        t_dense=t_dense,
        band=reference_band,
        ra=ra,
        dec=dec,
        model_name=true_model_name,
        parallax_model=true_parallax_model_for_plot,
        pyparams=true_pyparams,
    )

    fit_dense_model, _, _, A_fit_dense = _make_dense_model_for_lightcurve(
        t_dense=t_dense,
        band=reference_band,
        ra=ra,
        dec=dec,
        model_name=fit_model_name,
        parallax_model=fit_parallax_model_for_plot,
        pyparams=fit_pyparams,
    )

    print("=" * 80)
    print("DENSE MODEL CHECK")
    print("=" * 80)
    print("true dense class          =", type(true_dense_model))
    print("true dense parallax_model =", true_dense_model.parallax_model)
    print("fit dense class           =", type(fit_dense_model))
    print("fit dense parallax_model  =", fit_dense_model.parallax_model)
    print("=" * 80)

    m_true_dense = _flux_to_mag_band(
        fsource_ref * A_true_dense + fblend_ref,
        reference_band,
    )

    m_fit_dense = _flux_to_mag_band(
        fsource_ref * A_fit_dense + fblend_ref,
        reference_band,
    )

    rows = []
    skipped = []

    for tel in fit_model_obj.event.telescopes:
        if tel.lightcurve is None or len(tel.lightcurve) == 0:
            continue

        band = str(tel.name)

        time_arr = _to_numpy(tel.lightcurve["time"])
        mag = _to_numpy(tel.lightcurve["mag"])
        err_mag = _to_numpy(tel.lightcurve["err_mag"])

        A_fit_tel = fit_model_obj.model_magnification(
            tel,
            fit_pyparams,
        )

        # ------------------------------------------------------------
        # True model evaluated on a fresh temporary pyLIMA telescope
        # with the same times as the plotted data.
        #
        # Do NOT call:
        #     true_model_obj.model_magnification(tel, true_pyparams)
        # here, because tel belongs to fit_model_obj.  For parallax
        # models this can trigger KeyError('photometry') since the
        # telescope does not carry the true model's parallax cache.
        # ------------------------------------------------------------
        _, _, _, A_true_tel = _make_dense_model_for_lightcurve(
            t_dense=time_arr,
            band=band,
            ra=ra,
            dec=dec,
            model_name=true_model_name,
            parallax_model=true_parallax_model_for_plot,
            pyparams=true_pyparams,
        )

        fsource_tel, fblend_tel = _get_flux_pair(
            true_pyparams,
            band,
        )

        if fsource_tel <= 0.0 or not np.isfinite(fsource_tel):
            skipped.append(
                {
                    "band": band,
                    "reason": "invalid_true_fsource",
                    "fsource_true": fsource_tel,
                }
            )
            continue

        flux_obs = _mag_to_flux_band(mag, band)
        err_flux_obs = _magerr_to_fluxerr(mag, err_mag, band)

        A_obs_equiv = (flux_obs - fblend_tel) / fsource_tel
        err_A_obs_equiv = err_flux_obs / fsource_tel

        flux_aligned = fsource_ref * A_obs_equiv + fblend_ref
        err_flux_aligned = fsource_ref * err_A_obs_equiv

        mag_aligned = _flux_to_mag_band(flux_aligned, reference_band)
        err_mag_aligned = _fluxerr_to_magerr(flux_aligned, err_flux_aligned)

        flux_fit_aligned = fsource_ref * A_fit_tel + fblend_ref
        mag_fit_aligned = _flux_to_mag_band(flux_fit_aligned, reference_band)

        flux_true_aligned = fsource_ref * A_true_tel + fblend_ref
        mag_true_aligned = _flux_to_mag_band(flux_true_aligned, reference_band)

        residual = mag_aligned - mag_fit_aligned
        true_residual = mag_aligned - mag_true_aligned

        for k in range(len(time_arr)):
            row = {
                "band": band,
                "time": float(time_arr[k]),
                "t_minus_t0": float(time_arr[k] - t0),
                "mag": float(mag[k]),
                "mag_aligned": float(mag_aligned[k]),
                "err_mag": float(err_mag_aligned[k]),
                "err_mag_original": float(err_mag[k]),
                "fit_mag_aligned": float(mag_fit_aligned[k]),
                "true_mag_aligned": float(mag_true_aligned[k]),
                "residual": float(residual[k]),
                "true_residual": float(true_residual[k]),
                "A_obs_equiv": float(A_obs_equiv[k]),
                "A_fit": float(A_fit_tel[k]),
                "A_true": float(A_true_tel[k]),
                "fsource_true_band": float(fsource_tel),
                "fblend_true_band": float(fblend_tel),
            }

            bad_point = (
                not np.isfinite(row["mag_aligned"])
                or not np.isfinite(row["residual"])
                or not np.isfinite(row["err_mag"])
                or row["mag_aligned"] < 0.0
                or row["mag_aligned"] > 40.0
            )

            if bad_point and skip_bad_aligned_points:
                skipped.append({**row, "reason": "bad_aligned_point"})
                continue

            rows.append(row)

    aligned_data = pd.DataFrame(rows)
    skipped_data = pd.DataFrame(skipped)

    # ============================================================
    # No recalculamos Delta chi2 desde la figura alineada.
    #
    # La escala alineada usa flujos TRUE de la banda de referencia y sirve
    # solo para visualización. El criterio formal debe venir del chi2 de
    # pyLIMA / runner: Delta chi2 = chi2_fit - chi2_true.
    # Ese valor se extrae en _extract_parameter_box_fit_diagnostics().
    # ============================================================

    if fit_diagnostics is None:
        fit_diagnostics = {}
    else:
        fit_diagnostics = dict(fit_diagnostics)

    if len(skipped_data) > 0:
        print("=" * 80)
        print(f"Skipped aligned points: {len(skipped_data)}")
        print("=" * 80)
        print(skipped_data.head(20))

    if len(aligned_data) == 0:
        raise RuntimeError("No quedó ningún punto válido para graficar.")

    print("=" * 80)
    print("ALIGNED BASELINE CHECK")
    print("=" * 80)
    print("reference_band =", reference_band)
    print("ZP reference   =", _band_zp(reference_band))
    print("fsource_ref    =", fsource_ref)
    print("fblend_ref     =", fblend_ref)
    print(
        "baseline aligned mag =",
        float(
            _flux_to_mag_band(
                fsource_ref + fblend_ref,
                reference_band,
            )
        ),
    )

    for band, group in aligned_data.groupby("band"):
        med_A = np.nanmedian(group["A_obs_equiv"])
        med_mag = np.nanmedian(group["mag_aligned"])
        print(
            f"{band:3s}  "
            f"median A_obs_equiv={med_A:.4f}  "
            f"median mag_aligned={med_mag:.4f}"
        )

    print("=" * 80)

    t_traj = np.linspace(
        t0 - inset_n_tE * tE,
        t0 + inset_n_tE * tE,
        int(n_dense),
    )

    trajectory_specs, traj_data = _build_trajectory_specs_widget_style(
        true_pyparams=true_pyparams,
        fit_pyparams=fit_pyparams,
        fit_model_obj=fit_model_obj,
        t_traj=t_traj,
        ra=ra,
        dec=dec,
        reference_band=reference_band,
        show_true_no_parallax_reference=show_true_no_parallax_reference,
        fit_trajectory_time_mode=fit_trajectory_time_mode,
    )

    rho_true = float(_val(true_pyparams, "rho")) if _has(true_pyparams, "rho") else None

    fig_width = 13.5 if show_parameter_box else 10.0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_width, 7),
        dpi=130,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax, ax_res = axes

    true_curve_label = f"{true_model_name} + {_parallax_to_label(true_parallax_model_for_plot)}"
    fit_curve_label = f"{fit_model_name} fit + {_parallax_to_label(fit_parallax_model_for_plot)}"

    ax.plot(
        t_dense,
        m_true_dense,
        "k-",
        lw=2.0,
        label=true_curve_label,
    )

    ax.plot(
        t_dense,
        m_fit_dense,
        "--",
        color="purple",
        lw=2.0,
        label=fit_curve_label,
    )

    marker_map = {
        "u": "o",
        "g": "s",
        "r": "d",
        "i": "^",
        "z": "*",
        "y": "v",
        "W149": "x",
    }

    for band, group in aligned_data.groupby("band"):
        group = group.sort_values("time")

        ax.errorbar(
            group["time"],
            group["mag_aligned"],
            yerr=group["err_mag"],
            fmt=marker_map.get(band, "."),
            ms=4,
            linestyle="none",
            alpha=0.75,
            label=band,
        )

        ax_res.errorbar(
            group["time"],
            group["residual"],
            yerr=group["err_mag"],
            fmt=marker_map.get(band, "."),
            ms=4,
            linestyle="none",
            alpha=0.75,
        )

    ax.axvline(
        t0,
        ls="--",
        color="0.5",
        label=r"$t_{0,\mathrm{true}}$",
    )

    ax.axvline(
        fit_t0,
        ls=":",
        color="purple",
        alpha=0.7,
        label=r"$t_{0,\mathrm{fit}}$",
    )

    ax.axvspan(
        t0 - tE,
        t0 + tE,
        alpha=0.15,
        label=r"$t_{0,\mathrm{true}}\pm t_E$",
    )

    ax_res.axhline(0.0, color="k", lw=1, alpha=0.5)

    ax.set_xlim(t0 - n_tE * tE, t0 + n_tE * tE)
    ax.invert_yaxis()
    ax_res.invert_yaxis()

    ax.set_ylabel(
        rf"Aligned magnitude ({reference_band}-band; true-flux scale)"
    )

    ax_res.set_ylabel("Data - fit [mag]")
    ax_res.set_xlabel("JD")

    ax.grid(alpha=0.25)
    ax_res.grid(alpha=0.25)

    ax_traj = _add_trajectory_inset(
        ax=ax,
        trajectory_specs=trajectory_specs,
        rho_true=rho_true,
        loc=inset_loc,
        width="35%",
        height="45%",
    )

    handles, labels = ax.get_legend_handles_labels()
    seen = {}

    for h, lab in zip(handles, labels):
        if lab and not lab.startswith("_") and lab not in seen:
            seen[lab] = h

    ax.legend(
        seen.values(),
        seen.keys(),
        shadow=True,
        fontsize="large",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=3,
        numpoints=1,
    )

    if show_parameter_box:
        # Leave free space at the right for the parameter box.
        fig.tight_layout(rect=[0.0, 0.0, 0.74, 1.0])

        _add_true_fit_parameter_box(
            fig=fig,
            true_pyparams=true_pyparams,
            fit_pyparams=fit_pyparams,
            fit_diagnostics=fit_diagnostics,
            x=0.765,
            y=0.50,
            fontsize=8.2,
        )
    else:
        fig.tight_layout()

    dense = {
        "t_dense": t_dense,
        "m_true_dense": m_true_dense,
        "m_fit_dense": m_fit_dense,
        "A_true_dense": A_true_dense,
        "A_fit_dense": A_fit_dense,
        "t_traj": t_traj,  # compatibilidad hacia atrás
        "t_traj_true": traj_data["t_true"],
        "t_traj_fit": traj_data["t_fit"],
        "fit_trajectory_time_mode": fit_trajectory_time_mode,
        "traj_data": traj_data,
        "true_traj_x": traj_data["true_parallax"]["x"],
        "true_traj_y": traj_data["true_parallax"]["y"],
        "fit_traj_x": None if traj_data["fit"] is None else traj_data["fit"]["x"],
        "fit_traj_y": None if traj_data["fit"] is None else traj_data["fit"]["y"],
        "true_no_par_traj_x": (
            None if traj_data["true_no_parallax"] is None
            else traj_data["true_no_parallax"]["x"]
        ),
        "true_no_par_traj_y": (
            None if traj_data["true_no_parallax"] is None
            else traj_data["true_no_parallax"]["y"]
        ),
        "t0": t0,
        "tE": tE,
        "fit_t0": fit_t0,
        "fit_tE": fit_tE,
        "rho_true": rho_true,
        "true_parallax_model_for_plot": true_parallax_model_for_plot,
        "fit_parallax_model_for_plot": fit_parallax_model_for_plot,
        # Formal pyLIMA/runner chi2 diagnostics.
        # Do not compute Delta chi2 from the aligned plotting scale.
        "chi2_fit_pylima": fit_diagnostics.get("chi2_fit", np.nan),
        "chi2_true_pylima": fit_diagnostics.get("chi2_true", np.nan),
        "delta_chi2_pylima": fit_diagnostics.get("delta_chi2_true", np.nan),
        "dof_fit_pylima": fit_diagnostics.get("dof_fit", np.nan),
        "chi2_red_pylima": fit_diagnostics.get("chi2_red", np.nan),
        "fsource_ref_true": fsource_ref,
        "fblend_ref_true": fblend_ref,
        "reference_band": reference_band,
        "skipped_data": skipped_data,
        "ax_traj": ax_traj,
    }

    return fig, axes, aligned_data, dense



def _read_fit_rr_diagnostics_for_plot(save_path, simfit_result=None):
    """
    Read formal pyLIMA/runner diagnostics from the fit_rr parquet saved by
    sim_fit/save_extracted_results immediately before plotting.

    This avoids recomputing chi2 from the aligned plotting scale.  The plotted
    aligned magnitudes use true fluxes for display, whereas pyLIMA's fit chi2
    uses the original fitted flux parameters.  Therefore the formal Delta chi2
    for the figure must come from fit_rr:

        delta_chi2_true = chichi - chi2_true
    """
    out = {}

    if save_path is None:
        return out

    save_path = Path(save_path)

    # In this pipeline save_path is usually:
    #   <out_dir>/plots/global_...png
    # and the parquet lives in:
    #   <out_dir>/results/fit_rr/fit_rr_manual_<seed>.parquet
    try:
        out_dir = save_path.parent.parent
        fit_rr_dir = out_dir / "results" / "fit_rr"
    except Exception:
        return out

    if not fit_rr_dir.exists():
        return out

    seeds = []

    if isinstance(simfit_result, dict):
        for key in ["i", "simulation_seed", "seed"]:
            try:
                value = simfit_result.get(key)
                if value is not None and np.isfinite(float(value)):
                    seeds.append(int(value))
            except Exception:
                pass

        event_params = simfit_result.get("event_params", {})
        if isinstance(event_params, dict):
            for key in ["Source", "source", "simulation_seed", "seed"]:
                try:
                    value = event_params.get(key)
                    if value is not None and np.isfinite(float(value)):
                        seeds.append(int(value))
                except Exception:
                    pass

    files = []

    for seed in dict.fromkeys(seeds):
        files.extend(sorted(fit_rr_dir.glob(f"fit_rr_*_{seed}.parquet")))

    if len(files) == 0:
        files = sorted(fit_rr_dir.glob("fit_rr_*.parquet"))

    if len(files) == 0:
        return out

    # Prefer the newest file if there is any ambiguity.
    files = sorted(files, key=lambda p: p.stat().st_mtime)
    fit_file = files[-1]

    try:
        fit_df = pd.read_parquet(fit_file)
        if len(fit_df) == 0:
            return out

        row = fit_df.iloc[0].to_dict()

    except Exception:
        return out

    def get_first(keys):
        for key in keys:
            if key in row:
                value = _safe_float(row.get(key))
                if np.isfinite(value):
                    return value
        return np.nan

    chi2_fit = get_first(["chichi", "chi2", "chi2_fit"])
    chi2_true = get_first(["chi2_true", "true_chi2", "chi2_true_rr"])
    delta_chi2 = get_first(["delta_chi2_true", "delta_chi2", "Delta_chi2"])
    dof = get_first(["dof", "dof_fit"])
    rho_err = get_first(["rho_err", "rho_err_cov", "sigma_rho"])

    if not np.isfinite(delta_chi2) and np.isfinite(chi2_fit) and np.isfinite(chi2_true):
        delta_chi2 = chi2_fit - chi2_true

    chi2_red = np.nan
    if np.isfinite(chi2_fit) and np.isfinite(dof) and dof > 0:
        chi2_red = chi2_fit / dof

    out.update(
        {
            "fit_rr_file_for_plot": str(fit_file),
            "chi2_fit": chi2_fit,
            "chichi": chi2_fit,
            "chi2_true": chi2_true,
            "delta_chi2_true": delta_chi2,
            "dof_fit": dof,
            "dof": dof,
            "chi2_red": chi2_red,
            "rho_err": rho_err,
            "sigma_rho": rho_err,
        }
    )

    print("=" * 80)
    print("PARAMETER BOX FIT_RR DIAGNOSTICS")
    print("=" * 80)
    print("fit_rr_file       =", fit_file)
    print("chi2_fit/chichi   =", chi2_fit)
    print("chi2_true         =", chi2_true)
    print("delta_chi2_true   =", delta_chi2)
    print("dof               =", dof)
    print("chi2_red          =", chi2_red)
    print("rho_err           =", rho_err)
    print("=" * 80)

    return out

def plot_aligned_inset_from_result(
    simfit_result,
    save_path=None,
    global_i=None,
    true_model_name="FSPL",
    fit_model_name="FSPL",
    reference_band="g",
    allow_reference_band_fallback=True,
    n_dense=10000,
    n_tE=10.0,
    inset_n_tE=4.0,
    inset_loc="upper left",
    show_true_no_parallax_reference=False,
    fit_trajectory_time_mode="own_fit_tE_window",
    save_dpi=180,
):
    """
    Plot final para inspección:
    bandas alineadas, modelo true/fit, residuales e inset de trayectoria.
    """

    obj = extract_plot_objects(simfit_result)

    true_model_obj = obj["true_model_obj"]
    true_model_parameters = obj["true_model_parameters"]
    fit_model_obj = obj["fit_model_obj"]
    fit_model_parameters = obj["fit_model_parameters"]
    true_event_params = obj["true_event_params"]

    # Compute fit pyLIMA parameters here only to extract diagnostics for the
    # figure parameter box. The plotting function recomputes them internally.
    try:
        fit_pyparams_for_box = fit_model_obj.compute_pyLIMA_parameters(
            fit_model_parameters,
        )
    except Exception:
        fit_pyparams_for_box = None

    # Read the formal chi2 diagnostics from the fit_rr parquet saved by the
    # runner.  This is the value we want in the figure; do not recompute chi2
    # from the aligned display magnitudes.
    parquet_diagnostics = _read_fit_rr_diagnostics_for_plot(
        save_path=save_path,
        simfit_result=simfit_result,
    )

    if isinstance(parquet_diagnostics, dict) and len(parquet_diagnostics) > 0:
        simfit_result["plot_fit_diagnostics"] = parquet_diagnostics

    fit_diagnostics = _extract_parameter_box_fit_diagnostics(
        simfit_result=simfit_result,
        fit_model_obj=fit_model_obj,
        fit_model_parameters=fit_model_parameters,
        fit_pyparams=fit_pyparams_for_box,
    )

    if global_i is None:
        global_i = simfit_result.get("global_i", simfit_result.get("i", np.nan))

    fig, axes, aligned_data, dense = plot_event_aligned_true_fit_with_widget_style_trajectory(
        true_model_obj=true_model_obj,
        true_model_parameters=true_model_parameters,
        fit_model_obj=fit_model_obj,
        fit_model_parameters=fit_model_parameters,
        true_params=true_event_params,
        reference_band=reference_band,
        allow_reference_band_fallback=allow_reference_band_fallback,
        n_dense=n_dense,
        n_tE=n_tE,
        inset_n_tE=inset_n_tE,
        true_model_name=true_model_name,
        fit_model_name=fit_model_name,
        inset_loc=inset_loc,
        force_true_parallax_reference_to_t0=True,
        show_true_no_parallax_reference=show_true_no_parallax_reference,
        fit_trajectory_time_mode=fit_trajectory_time_mode,
        fit_diagnostics=fit_diagnostics,
        show_parameter_box=True,
    )

    tE = float(true_event_params["tE"])
    u0 = float(true_event_params["u0"])
    rho = float(true_event_params.get("rho", np.nan))

    axes[0].set_title(
        f"global_i={global_i}\n"
        rf"$t_E={tE:.2f}\,\mathrm{{d}}$, "
        rf"$u_0={u0:.3g}$, "
        rf"$\rho={rho:.3g}$",
        fontsize=10,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")

    return fig, axes, aligned_data, dense


def plot_pylima_true_fit_from_result(
    simfit_result,
    save_path=None,
    global_i=None,
    true_label=None,
    fit_label=None,
    xlim_factor=3.5,
    plot_residuals=True,
):
    """
    Plot simple y robusto true vs fit.
    Usa pyLIMA_plots, evitando telescopios vacíos.
    """

    try:
        import pyLIMA_plots
    except ModuleNotFoundError:
        from pyLIMA.outputs import pyLIMA_plots

    obj = extract_plot_objects(simfit_result)

    true_model_obj = obj["true_model_obj"]
    true_model_parameters = obj["true_model_parameters"]
    fit_model_obj = obj["fit_model_obj"]
    fit_model_parameters = obj["fit_model_parameters"]
    true_params = obj["true_event_params"]

    if global_i is None:
        global_i = simfit_result.get("global_i", simfit_result.get("i", np.nan))

    if true_label is None:
        true_label = f"{simfit_result.get('model', 'model')} + parallax (True)"

    if fit_label is None:
        fit_label = (
            f"{simfit_result.get('fit_model', 'fit')} "
            f"(fit - {'parallax' if simfit_result.get('fit_parallax', False) else 'No parallax'})"
        )

    def plot_model(ax, model_obj, pars, color, ls, label):
        n0 = len(ax.lines)

        with _temporarily_disable_empty_lightcurves(model_obj):
            try:
                pyLIMA_plots.plot_photometric_models(
                    figure_axe=ax,
                    microlensing_model=model_obj,
                    model_parameters=pars,
                    MARKERS_COLORS=cycler.cycler(color=[color] * 10),
                    bokeh_plot=None,
                    plot_unit="Mag",
                )
            except TypeError:
                pyLIMA_plots.plot_photometric_models(
                    figure_axe=ax,
                    microlensing_model=model_obj,
                    model_parameters=pars,
                    bokeh_plot=None,
                    plot_unit="Mag",
                )

        for k, line in enumerate(ax.lines[n0:]):
            line.set_color(color)
            line.set_linestyle(ls)
            line.set_linewidth(1.8)
            line.set_label(label if k == 0 else "_nolegend_")

    t0 = float(true_params["t0"])
    tE = float(true_params["tE"])

    if plot_residuals:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 7),
            dpi=130,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax_lc, ax_res = axes
    else:
        fig, ax_lc = plt.subplots(figsize=(10, 5), dpi=130)
        axes = [ax_lc]
        ax_res = None

    plot_model(ax_lc, true_model_obj, true_model_parameters, "0.55", "-", true_label)
    plot_model(ax_lc, fit_model_obj, fit_model_parameters, "purple", "--", fit_label)

    with _temporarily_disable_empty_lightcurves(fit_model_obj):
        pyLIMA_plots.plot_aligned_data(
            figure_axe=ax_lc,
            microlensing_model=fit_model_obj,
            model_parameters=fit_model_parameters,
            bokeh_plot=None,
            plot_unit="Mag",
        )

    if plot_residuals:
        with _temporarily_disable_empty_lightcurves(fit_model_obj):
            try:
                pyLIMA_plots.plot_residuals(
                    ax_res,
                    fit_model_obj,
                    fit_model_parameters,
                    bokeh_plot=None,
                    plot_unit="Mag",
                )
            except TypeError:
                pyLIMA_plots.plot_residuals(
                    figure_axe=ax_res,
                    microlensing_model=fit_model_obj,
                    model_parameters=fit_model_parameters,
                    bokeh_plot=None,
                    plot_unit="Mag",
                )

        ax_res.axhline(0, color="k", alpha=0.4, lw=1)
        ax_res.invert_yaxis()
        ax_res.set_ylabel(r"$\Delta$ mag")
        ax_res.set_xlabel(r"$t - t_0$ [days]")

    x_min = t0 - xlim_factor * tE
    x_max = t0 + xlim_factor * tE
    xticks = np.linspace(x_min, x_max, 7)

    ax_lc.set_xlim(x_min, x_max)

    for ax in axes:
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x - t0:.1f}" for x in xticks])

    ax_lc.invert_yaxis()
    ax_lc.set_ylabel("Aligned mag")

    ax_lc.set_title(
        f"global_i={global_i}\n"
        f"tE={tE:.2f} d, "
        f"u0={float(true_params['u0']):.3g}, "
        f"rho={float(true_params.get('rho', np.nan)):.3g}",
        fontsize=10,
    )

    handles, labels = ax_lc.get_legend_handles_labels()
    keep = {}

    for h, lab in zip(handles, labels):
        if lab and not lab.startswith("_") and lab not in keep:
            keep[lab] = h

    ax_lc.legend(
        keep.values(),
        keep.keys(),
        fontsize=8,
        ncol=3,
        loc="best",
        framealpha=0.9,
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes


# ============================================================
# Ejecutar un evento
# ============================================================

def run_single_task(
    runner,
    prepared_catalog,
    task,
    config: SedighePipelineConfig,
    paths: PipelinePaths,
    set_fit_seed,
    make_plot: Optional[bool] = None,
):
    """
    Ejecuta un solo evento del catálogo.
    Devuelve:
        record, simfit_result
    """

    t_start = time.time()

    task_inputs = prepare_single_task_inputs(
        runner=runner,
        prepared_catalog=prepared_catalog,
        task=task,
        config=config,
    )

    global_i = int(task_inputs["global_i"])
    seed = int(task_inputs["seed"])
    base_row = task_inputs["base_row"]

    if make_plot is None:
        make_plot = bool(config.make_plots)

    if config.verbose:
        print("=" * 80)
        print(f"START global_i={global_i}, seed={seed}")
        print("=" * 80)
        print(f"t0_jd = {float(base_row['t0_jd']):.8f}")
        print(f"t0_origin = {base_row['t0_origin']}")
        print(f"apply_photometric_filter = {config.apply_photometric_filter}")
        print(f"apply_detection_criteria = {config.apply_detection_criteria}")
        print(f"plot_style = {config.plot_style}")
        print("=" * 80)

    set_fit_seed(seed)

    kwargs = build_sim_fit_kwargs(
        runner=runner,
        paths=paths,
        task_inputs=task_inputs,
        config=config,
    )

    simfit_result = None

    if hasattr(runner, "set_runtime_event_context"):
        runner.set_runtime_event_context(base_row)

    try:
        with deterministic_rng(seed):
            simfit_result = runner.sim_fit(**kwargs)

        elapsed = time.time() - t_start

        record = summarize_simfit_result(
            simfit_result=simfit_result,
            task_inputs=task_inputs,
            config=config,
            elapsed_sec=elapsed,
        )

        record["error"] = ""

        do_plot = make_plot

        if config.plot_indices is not None:
            do_plot = global_i in set(int(x) for x in config.plot_indices)

        if do_plot and isinstance(simfit_result, dict) and simfit_result.get("status") == "fitted":
            if config.plot_style == "aligned_inset":
                plot_path = paths.plots / f"global_{global_i:07d}_seed_{seed}_aligned_inset.png"

                fig, axes, aligned_data, dense = plot_aligned_inset_from_result(
                    simfit_result=simfit_result,
                    save_path=plot_path,
                    global_i=global_i,
                    true_model_name=runner.MODEL,
                    fit_model_name=runner.FIT_MODEL,
                    reference_band=config.reference_band,
                    allow_reference_band_fallback=config.allow_reference_band_fallback,
                    n_dense=config.plot_n_dense,
                    n_tE=config.plot_n_tE,
                    inset_n_tE=config.plot_inset_n_tE,
                    inset_loc=config.plot_inset_loc,
                    show_true_no_parallax_reference=config.plot_show_true_no_parallax_reference,
                    fit_trajectory_time_mode=config.plot_fit_trajectory_time_mode,
                    save_dpi=config.save_plot_dpi,
                )

                plt.close(fig)

                record["plot_style"] = "aligned_inset"
                record["plot_path"] = str(plot_path)
                record["n_aligned_points"] = int(len(aligned_data))
                record["plot_reference_band"] = dense.get("reference_band", config.reference_band)
                record["plot_fit_trajectory_time_mode"] = dense.get("fit_trajectory_time_mode", config.plot_fit_trajectory_time_mode)
                record["A_true_min"] = float(np.nanmin(dense["A_true_dense"]))
                record["A_true_max"] = float(np.nanmax(dense["A_true_dense"]))
                record["A_fit_min"] = float(np.nanmin(dense["A_fit_dense"]))
                record["A_fit_max"] = float(np.nanmax(dense["A_fit_dense"]))

            elif config.plot_style == "quick_pylima":
                plot_path = paths.plots / f"global_{global_i:07d}_seed_{seed}_quick_pylima.png"

                fig, axes = plot_pylima_true_fit_from_result(
                    simfit_result,
                    save_path=plot_path,
                    global_i=global_i,
                    xlim_factor=config.plot_n_tE,
                    plot_residuals=config.plot_residuals,
                )

                plt.close(fig)

                record["plot_style"] = "quick_pylima"
                record["plot_path"] = str(plot_path)
                record["n_aligned_points"] = np.nan

            else:
                raise ValueError(
                    f"plot_style desconocido: {config.plot_style!r}. "
                    "Usá 'aligned_inset' o 'quick_pylima'."
                )

        else:
            record["plot_path"] = ""

    except Exception as error:
        elapsed = time.time() - t_start

        record = {
            "global_i": global_i,
            "prepared_index": int(task_inputs["prepared_index"]),
            "simulation_seed": seed,
            "status": "error",
            "elapsed_sec": float(elapsed),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }

        error_path = paths.logs / f"global_{global_i:07d}_seed_{seed}_error.txt"

        with open(error_path, "w") as f:
            f.write(record["traceback"])

        record["error_log_path"] = str(error_path)

        if config.verbose:
            print("=" * 80)
            print(f"ERROR global_i={global_i}, seed={seed}")
            print(repr(error))
            print(f"Traceback guardado en: {error_path}")
            print("=" * 80)

    finally:
        if hasattr(runner, "clear_runtime_event_context"):
            runner.clear_runtime_event_context()

    if config.append_summary:
        append_record_to_parquet(
            record,
            paths.summary,
        )

    gc.collect()

    return record, simfit_result


# ============================================================
# Ejecutar dataset
# ============================================================

def run_dataset(config: SedighePipelineConfig):
    """
    Corre el pipeline completo sobre todos los tasks seleccionados.
    """

    paths = make_output_dirs(config)

    runner = import_runner(config)

    if hasattr(runner, "install_runtime_patches"):
        runner.install_runtime_patches()

    set_fit_seed = install_seeded_fit_patch(runner)

    raw_catalog, prepared_catalog, invalid_catalog, tasks = load_catalog_and_tasks(
        runner,
        config,
    )

    print("=" * 80)
    print("DATASET READY")
    print("=" * 80)
    print("raw_catalog shape      =", getattr(raw_catalog, "shape", None))
    print("prepared_catalog shape =", getattr(prepared_catalog, "shape", None))
    print("invalid_catalog shape  =", getattr(invalid_catalog, "shape", None))
    print("n tasks                =", len(tasks))
    print("out_dir                =", paths.root)
    print("summary                =", paths.summary)
    print("=" * 80)

    records = []

    for n, task in enumerate(tasks, start=1):
        global_i = int(task["global_i"])

        print(f"\n[{n}/{len(tasks)}] global_i={global_i}")

        record, _ = run_single_task(
            runner=runner,
            prepared_catalog=prepared_catalog,
            task=task,
            config=config,
            paths=paths,
            set_fit_seed=set_fit_seed,
        )

        records.append(record)

        print(
            f"[{n}/{len(tasks)}] global_i={global_i} "
            f"status={record.get('status')} "
            f"elapsed={record.get('elapsed_sec', np.nan):.2f}s"
        )

    df_summary = pd.DataFrame(records)

    final_summary_path = paths.root / "run_summary_current_session.parquet"
    df_summary.to_parquet(final_summary_path, index=False)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("summary incremental =", paths.summary)
    print("summary session     =", final_summary_path)
    print("=" * 80)

    return {
        "runner": runner,
        "paths": paths,
        "raw_catalog": raw_catalog,
        "prepared_catalog": prepared_catalog,
        "invalid_catalog": invalid_catalog,
        "tasks": tasks,
        "summary": df_summary,
    }


# ============================================================
# Ejecutar un solo evento por global_i
# ============================================================

def run_one_by_global_i(
    config: SedighePipelineConfig,
    global_i: int,
    make_plot: bool = True,
):
    """
    Útil para debugging o visualizar un evento.
    """

    config.global_indices = [int(global_i)]
    config.max_base_events = max(
        int(global_i) + 1,
        config.max_base_events or 0,
    )
    config.make_plots = make_plot

    paths = make_output_dirs(config)

    runner = import_runner(config)

    if hasattr(runner, "install_runtime_patches"):
        runner.install_runtime_patches()

    set_fit_seed = install_seeded_fit_patch(runner)

    raw_catalog, prepared_catalog, invalid_catalog, tasks = load_catalog_and_tasks(
        runner,
        config,
    )

    if len(tasks) != 1:
        raise RuntimeError(
            f"Esperaba un task para global_i={global_i}, "
            f"pero encontré {len(tasks)}."
        )

    record, simfit_result = run_single_task(
        runner=runner,
        prepared_catalog=prepared_catalog,
        task=tasks[0],
        config=config,
        paths=paths,
        set_fit_seed=set_fit_seed,
        make_plot=make_plot,
    )

    return {
        "runner": runner,
        "paths": paths,
        "raw_catalog": raw_catalog,
        "prepared_catalog": prepared_catalog,
        "invalid_catalog": invalid_catalog,
        "task": tasks[0],
        "record": record,
        "simfit_result": simfit_result,
    }
