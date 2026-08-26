#!/usr/bin/env python3
import json
from pathlib import Path

BASE = Path(
    "/share/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models/Parallax_LSST"
)

HERE = BASE / "lsstmonts_catalog_sedighe"

SRC = HERE / "config_lsstmonts_baseline_v5p3p5_cluster_che_multifit_LRT.json"
DST = HERE / "config_lsstmonts_baseline_v5p3p5_cluster_che_multifit_LRT_ABSOLUTE_PLOT.json"

RUNNER = HERE / "run_lsstmonts_catalog_hidden_parallax.py"
PIPELINE = HERE / "pipeline_hidden_parallax.py"
PLOT_SCRIPT = HERE / "plot_confused_lightcurves_with_inset_pipeline.py"

assert SRC.exists(), SRC
assert RUNNER.exists(), RUNNER
assert PIPELINE.exists(), PIPELINE
assert PLOT_SCRIPT.exists(), PLOT_SCRIPT


def replace_obj(obj):
    if isinstance(obj, dict):
        return {k: replace_obj(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [replace_obj(v) for v in obj]

    if isinstance(obj, str):
        s = obj

        s = s.replace(
            "${PARALLAX_LSST_BASE}",
            str(BASE),
        )

        s = s.replace(
            "$PARALLAX_LSST_BASE",
            str(BASE),
        )

        s = s.replace(
            "/home/anibal/ulensing_degenerate_models/Parallax_LSST",
            str(BASE),
        )

        s = s.replace(
            "/home/anibal/ulensing_degenerate_models",
            str(BASE.parent),
        )

        s = s.replace(
            "/home/anibal/microlensing",
            "/home/anibalvarela/microlensing",
        )

        s = s.replace(
            "/home/anibal-pc/ulensing_degenerate_models",
            str(BASE.parent),
        )

        s = s.replace(
            "/home/anibal-pc/microlensing",
            "/home/anibalvarela/microlensing",
        )

        s = s.replace(
            "/home/anibal-pc/rubin_sim_data",
            "/share/storage3/rubin/microlensing/romanrubin/rubin_sim_data",
        )

        return s

    return obj


with open(SRC) as f:
    cfg = json.load(f)

cfg = replace_obj(cfg)

cfg.setdefault("paths", {})
cfg["paths"]["runner_path"] = str(RUNNER)
cfg["paths"]["pipeline_path"] = str(PIPELINE)
cfg["paths"]["plot_script"] = str(PLOT_SCRIPT)
cfg["paths"]["ulensing_degenerate_models_root"] = str(BASE.parent)
cfg["paths"]["microlensing_root"] = "/home/anibalvarela/microlensing"
cfg["paths"]["roman_rubin_dir"] = "/home/anibalvarela/microlensing/simulation_Rubin/roman_rubin"
cfg["paths"]["output_root"] = "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax"

cfg.setdefault("runner", {})
cfg["runner"]["path"] = str(RUNNER)
cfg["runner"]["runner_path"] = str(RUNNER)

cfg.setdefault("pipeline", {})
cfg["pipeline"]["path"] = str(PIPELINE)
cfg["pipeline"]["pipeline_path"] = str(PIPELINE)

cfg.setdefault("rubin", {})
cfg["rubin"]["sim_data_dir"] = "/share/storage3/rubin/microlensing/romanrubin/rubin_sim_data"
cfg["rubin"]["throughputs_dir"] = "/share/storage3/rubin/microlensing/romanrubin/rubin_sim_data/throughputs/baseline"

opsim_candidates = [
    "/share/storage3/rubin/microlensing/romanrubin/rubin_sim_data/sim_baseline/sim_baseline_2026_07_23/sim_baseline/baseline_v5.3.5_10yrs.db",
    "/export/storage3/rubin/microlensing/romanrubin/rubin_sim_data/sim_baseline/sim_baseline_2026_07_23/sim_baseline/baseline_v5.3.5_10yrs.db",
]

for p in opsim_candidates:
    if Path(p).exists():
        cfg["rubin"]["opsim_db_path"] = p
        break

with open(DST, "w") as f:
    json.dump(cfg, f, indent=2)

print("Wrote:", DST)
print("runner_path:", cfg["paths"]["runner_path"])
print("pipeline_path:", cfg["paths"]["pipeline_path"])
print("plot_script:", cfg["paths"]["plot_script"])
print("opsim_db_path:", cfg.get("rubin", {}).get("opsim_db_path"))
