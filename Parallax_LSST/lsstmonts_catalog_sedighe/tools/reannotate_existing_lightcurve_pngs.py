#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import importlib.util

RUN_TAG = "prod_multifit_20260826T041803Z"

BASE_ANALYSIS = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/partial_analysis"
)

SCRIPT = Path(
    "/share/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models/Parallax_LSST/"
    "lsstmonts_catalog_sedighe/analysis/select_and_plot_delta_chi2_lt12.py"
)

spec = importlib.util.spec_from_file_location("selplot", SCRIPT)
selplot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selplot)

input_csv = sorted(
    BASE_ANALYSIS.glob(f"{RUN_TAG}_*/confused_delta_chi2_0_12_strict/representative_delta_chi2_events.csv")
)[-1]

outdir = input_csv.parent / "lightcurves_with_inset_pipeline"

events = pd.read_csv(input_csv)
events = selplot.ensure_columns(events)

print("input_csv =", input_csv)
print("outdir    =", outdir)

for _, row in events.iterrows():
    row = row.to_dict()

    global_i = int(row["global_i"])
    seed = int(row["simulation_seed"])

    base_pngs = sorted(
        p for p in outdir.rglob(f"global_{global_i:07d}_seed_{seed}_aligned_inset.png")
        if not p.name.endswith("_annotated.png")
    )

    if len(base_pngs) == 0:
        print("[missing]", global_i, seed)
        continue

    for base_png in base_pngs:
        out_png = base_png.with_name(
            base_png.stem + "_annotated_big_external.png"
        )

        selplot.annotate_png(
            input_png=base_png,
            output_png=out_png,
            row=row,
        )

        print("[ok]", out_png)
