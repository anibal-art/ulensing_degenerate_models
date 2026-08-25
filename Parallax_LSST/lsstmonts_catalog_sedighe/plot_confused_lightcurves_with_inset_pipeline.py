#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-run selected confused events through pipeline_hidden_parallax.py and save
light-curve plots with aligned bands, residuals and trajectory inset.

This script does NOT require previously saved point-by-point light curves.
It re-runs only the selected events and uses simfit_result in memory.

Default selected-events file:
    runs/<run_name>/figures/confused_fspl_noparallax/
        representative_confused_events.csv

Output:
    runs/<run_name>/figures/confused_lightcurves_with_inset_pipeline/plots/
"""

from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np
import pandas as pd


# ============================================================
# Default paths
# ============================================================

DEFAULT_PROJECT_DIR = Path(
    "/home/anibal/ulensing_degenerate_models/Parallax_LSST/"
    "lsstmonts_catalog_sedighe"
)

DEFAULT_RUNNER = DEFAULT_PROJECT_DIR / "run_lsstmonts_catalog_hidden_parallax.py"

DEFAULT_PIPELINE = DEFAULT_PROJECT_DIR / "pipeline_hidden_parallax.py"

DEFAULT_CONFIG = DEFAULT_PROJECT_DIR / "config_lsstmonts_baseline_v5p3p5.json"

DEFAULT_ROMAN_RUBIN_DIR = Path(
    "/home/anibal/microlensing/simulation_Rubin/roman_rubin"
)


# ============================================================
# Helpers
# ============================================================

def load_json(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path


def infer_run_dir_from_config(config_path):
    cfg = load_json(config_path)

    run_name = cfg["run_name"]

    path_storage = cfg.get(
        "path_storage",
        "/home/anibal/ulensing_degenerate_models/Parallax_LSST/runs",
    )

    return Path(path_storage) / run_name


def import_pipeline_module(pipeline_path):
    pipeline_path = Path(pipeline_path).expanduser().resolve()

    if not pipeline_path.exists():
        raise FileNotFoundError(f"No existe pipeline_hidden_parallax.py: {pipeline_path}")

    module_name = "pipeline_hidden_parallax_runtime"

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name,
        pipeline_path,
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def parse_global_i_list(values):
    if values is None:
        return None

    out = []

    for value in values:
        for piece in str(value).replace(",", " ").split():
            if piece.strip():
                out.append(int(piece))

    return out


def load_selected_events(events_csv, n_events):
    events_csv = Path(events_csv)

    if not events_csv.exists():
        raise FileNotFoundError(
            f"No existe la tabla de eventos seleccionados:\n{events_csv}\n"
            "Pasá --global-i manualmente o generá primero representative_confused_events.csv."
        )

    df = pd.read_csv(events_csv)

    if "global_i" not in df.columns:
        raise KeyError(
            f"{events_csv} no tiene columna global_i."
        )

    df = df.copy()

    if "selection_score" in df.columns:
        df = df.sort_values("selection_score")

    if n_events is not None:
        df = df.head(int(n_events)).copy()

    global_indices = (
        df["global_i"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )

    if len(global_indices) == 0:
        raise RuntimeError("No encontré global_i válidos en la tabla.")

    return df, global_indices


def make_temp_plot_config(
    original_config_path,
    temp_config_path,
    global_indices,
    selected_table=None,
    read_buffer=2000,
    reference_band="g",
    plot_n_tE=10.0,
    plot_inset_n_tE=4.0,
    plot_inset_loc="upper left",
    show_true_no_parallax_reference=False,
    fit_trajectory_time_mode="own_fit_tE_window",
):
    """
    Create a temporary config so the runner does not read the full catalog.

    We do not overwrite the science config.
    """

    raw = load_json(original_config_path)

    global_indices = [int(x) for x in global_indices]

    max_global_i = max(global_indices)

    max_catalog_row = max_global_i

    if selected_table is not None and "catalog_row" in selected_table.columns:
        valid_catalog_rows = (
            selected_table["catalog_row"]
            .dropna()
            .astype(int)
            .to_numpy()
        )

        if len(valid_catalog_rows) > 0:
            max_catalog_row = max(max_catalog_row, int(np.max(valid_catalog_rows)))

    # Read enough rows to include selected catalog rows.
    read_nrows = int(max_catalog_row + 1 + read_buffer)

    raw["Nevents"] = read_nrows

    raw.setdefault("input", {})
    raw["input"]["read_nrows"] = read_nrows

    raw.setdefault("selection", {})
    raw["selection"]["max_base_events"] = read_nrows
    raw["selection"]["apply_detection_criteria"] = False

    raw.setdefault("simulation", {})
    raw["simulation"]["apply_detection_criteria"] = False
    raw["simulation"]["apply_photometric_filter"] = True

    raw.setdefault("hidden_parallax", {})
    raw["hidden_parallax"].update(
        {
            "return_data": True,
            "make_plots": True,
            "plot_indices": global_indices,
            "plot_style": "aligned_inset",
            "reference_band": reference_band,
            "allow_reference_band_fallback": True,
            "plot_n_dense": 10000,
            "plot_n_tE": float(plot_n_tE),
            "plot_inset_n_tE": float(plot_inset_n_tE),
            "plot_inset_loc": plot_inset_loc,
            "plot_show_true_no_parallax_reference": bool(show_true_no_parallax_reference),
            "plot_fit_trajectory_time_mode": fit_trajectory_time_mode,
            "plot_residuals": True,
            "save_plot_dpi": 220,
            "append_summary": True,
            "summary_name": "plot_summary.parquet",
            "verbose": True,
        }
    )

    save_json(raw, temp_config_path)

    return temp_config_path, read_nrows


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline",
        default=str(DEFAULT_PIPELINE),
        help="Path to pipeline_hidden_parallax.py.",
    )

    parser.add_argument(
        "--runner",
        default=str(DEFAULT_RUNNER),
        help="Path to run_lsstmonts_catalog_hidden_parallax.py.",
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Main JSON config used by the runner.",
    )

    parser.add_argument(
        "--roman-rubin-dir",
        default=str(DEFAULT_ROMAN_RUBIN_DIR),
    )

    parser.add_argument(
        "--events-csv",
        default=None,
        help=(
            "CSV with representative confused events. "
            "If omitted, uses the default confused_fspl_noparallax table."
        ),
    )

    parser.add_argument(
        "--global-i",
        nargs="*",
        default=None,
        help="Manual list of global_i values, e.g. --global-i 12 35 91.",
    )

    parser.add_argument(
        "--n-events",
        type=int,
        default=6,
        help="Number of selected events to plot from the CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root for this plotting rerun.",
    )

    parser.add_argument(
        "--reference-band",
        default="g",
    )

    parser.add_argument(
        "--plot-n-tE",
        type=float,
        default=10.0,
        help="Half-width of main plot in units of true tE.",
    )

    parser.add_argument(
        "--plot-inset-n-tE",
        type=float,
        default=4.0,
        help="Half-width of trajectory inset in units of true tE.",
    )

    parser.add_argument(
        "--plot-inset-loc",
        default="upper left",
        choices=[
            "upper left",
            "upper right",
            "lower left",
            "lower right",
            "center left",
            "center right",
        ],
    )

    parser.add_argument(
        "--show-true-no-parallax-reference",
        action="store_true",
        help="Also draw the true no-parallax trajectory as reference in the inset.",
    )

    parser.add_argument(
        "--fit-trajectory-time-mode",
        default="own_fit_tE_window",
        choices=["own_fit_tE_window", "same_true_window"],
    )

    parser.add_argument(
        "--read-buffer",
        type=int,
        default=2000,
        help="Extra raw catalog rows to read beyond max selected index.",
    )

    parser.add_argument(
        "--reset-output",
        action="store_true",
    )

    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    run_dir = infer_run_dir_from_config(config_path)

    if args.output_dir is None:
        output_dir = run_dir / "figures" / "confused_lightcurves_with_inset_pipeline"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    manual_global_i = parse_global_i_list(args.global_i)

    selected_table = None

    if manual_global_i is not None and len(manual_global_i) > 0:
        global_indices = manual_global_i
    else:
        if args.events_csv is None:
            events_csv = (
                run_dir
                / "figures"
                / "confused_fspl_noparallax"
                / "representative_confused_events.csv"
            )
        else:
            events_csv = Path(args.events_csv).expanduser().resolve()

        selected_table, global_indices = load_selected_events(
            events_csv,
            n_events=args.n_events,
        )

    global_indices = [int(x) for x in global_indices]

    print("=" * 80)
    print("Selected events")
    print("=" * 80)
    print("global_i =", global_indices)
    print("n events =", len(global_indices))
    print("output_dir =", output_dir)
    print("=" * 80)

    temp_config_path = output_dir / "plot_run_config.json"

    temp_config_path, read_nrows = make_temp_plot_config(
        original_config_path=config_path,
        temp_config_path=temp_config_path,
        global_indices=global_indices,
        selected_table=selected_table,
        read_buffer=args.read_buffer,
        reference_band=args.reference_band,
        plot_n_tE=args.plot_n_tE,
        plot_inset_n_tE=args.plot_inset_n_tE,
        plot_inset_loc=args.plot_inset_loc,
        show_true_no_parallax_reference=args.show_true_no_parallax_reference,
        fit_trajectory_time_mode=args.fit_trajectory_time_mode,
    )

    print("Temporary plot config:", temp_config_path)
    print("Temporary input.read_nrows:", read_nrows)

    pipeline = import_pipeline_module(args.pipeline)

    cfg = pipeline.SedighePipelineConfig.from_config_file(
        runner_path=Path(args.runner).expanduser().resolve(),
        config_path=temp_config_path,
        roman_rubin_dir=Path(args.roman_rubin_dir).expanduser().resolve(),
        out_dir=output_dir,
        global_indices=global_indices,
        max_base_events=read_nrows,
        return_data=True,
        make_plots=True,
        plot_indices=global_indices,
        plot_style="aligned_inset",
        reference_band=args.reference_band,
        allow_reference_band_fallback=True,
        plot_n_tE=args.plot_n_tE,
        plot_inset_n_tE=args.plot_inset_n_tE,
        plot_inset_loc=args.plot_inset_loc,
        plot_show_true_no_parallax_reference=args.show_true_no_parallax_reference,
        plot_fit_trajectory_time_mode=args.fit_trajectory_time_mode,
        plot_residuals=True,
        apply_photometric_filter=True,
        apply_detection_criteria=False,
        reset_output=bool(args.reset_output),
        append_summary=True,
        summary_name="plot_summary.parquet",
        verbose=True,
    )

    result = pipeline.run_dataset(cfg)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("Plots directory:")
    print(output_dir / "plots")
    print()
    print("Generated PNG files:")
    for file in sorted((output_dir / "plots").glob("*.png")):
        print(file)
    print("=" * 80)

    return result


if __name__ == "__main__":
    main()
