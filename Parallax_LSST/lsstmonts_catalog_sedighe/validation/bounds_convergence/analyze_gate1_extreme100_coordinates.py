#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


MODES = [
    "physical",
    "log_te",
    "log_rho",
    "log_te_rho",
]

TOL = 0.1

ROOT = Path(
    "~/Downloads/hidden_parallax/"
    "production_validation/gate1_coordinates"
).expanduser()

REPO_RESULTS = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results"
)

MANIFEST = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/data/"
    "extreme100_refit_manifest.csv"
)

REPO_RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)

manifest = pd.read_csv(MANIFEST)

rows = (
    pd.to_numeric(
        manifest["catalog_row"],
        errors="raise",
    )
    .astype(int)
    .drop_duplicates()
    .tolist()
)

records = []

for row in rows:

    record = {
        "catalog_row": row,
    }

    for mode in MODES:

        p = (
            ROOT
            / mode
            / "refits"
            / "bounds_convergence"
            / "te500000_h0only"
            / "extreme100"
            / str(row)
            / "all_refits.csv"
        )

        record[f"exists_{mode}"] = p.exists()

        if not p.exists():
            record[f"n_success_{mode}"] = 0
            record[f"chi2_{mode}"] = np.nan
            record[f"best_label_{mode}"] = None
            record[f"wall_{mode}"] = np.nan
            continue

        df = pd.read_csv(p)

        ok = df[
            (df["hypothesis"] == "H0")
            & (df["status"] == "success")
        ].copy()

        ok["chi2"] = pd.to_numeric(
            ok["chi2"],
            errors="coerce",
        )

        ok = ok[
            np.isfinite(ok["chi2"])
        ]

        record[f"n_success_{mode}"] = len(ok)

        if len(ok):

            best = ok.loc[
                ok["chi2"].idxmin()
            ]

            record[f"chi2_{mode}"] = float(
                best["chi2"]
            )

            record[f"best_label_{mode}"] = str(
                best["label"]
            )

        else:

            record[f"chi2_{mode}"] = np.nan
            record[f"best_label_{mode}"] = None

        wall_file = (
            ROOT
            / mode
            / f"wall_{row}.txt"
        )

        try:
            record[f"wall_{mode}"] = float(
                wall_file.read_text().strip()
            )
        except Exception:
            record[f"wall_{mode}"] = np.nan

    records.append(record)


out = pd.DataFrame(records)

chi_cols = [
    f"chi2_{mode}"
    for mode in MODES
]

out["chi2_envelope"] = (
    out[chi_cols]
    .min(
        axis=1,
        skipna=True,
    )
)

for mode in MODES:

    out[f"delta_{mode}"] = (
        out[f"chi2_{mode}"]
        - out["chi2_envelope"]
    )

out["winner"] = (
    out[chi_cols]
    .idxmin(axis=1)
    .str.replace(
        "chi2_",
        "",
        regex=False,
    )
)

per_event_path = (
    REPO_RESULTS
    / "gate1_extreme100_coordinate_per_event.csv"
)

out.to_csv(
    per_event_path,
    index=False,
)


summary = []

for mode in MODES:

    delta = pd.to_numeric(
        out[f"delta_{mode}"],
        errors="coerce",
    )

    wall = pd.to_numeric(
        out[f"wall_{mode}"],
        errors="coerce",
    )

    success = pd.to_numeric(
        out[f"n_success_{mode}"],
        errors="coerce",
    )

    finite = delta[
        np.isfinite(delta)
    ]

    summary.append(
        {
            "mode": mode,
            "n_events": len(out),
            "n_complete_13_starts":
                int((success >= 13).sum()),
            "n_missing_chi2":
                int(delta.isna().sum()),
            "n_delta_gt_0p1":
                int((delta > TOL).sum()),
            "n_delta_gt_1":
                int((delta > 1.0).sum()),
            "n_delta_gt_10":
                int((delta > 10.0).sum()),
            "max_delta":
                float(finite.max())
                if len(finite)
                else np.nan,
            "p50_delta":
                float(finite.quantile(0.50))
                if len(finite)
                else np.nan,
            "p90_delta":
                float(finite.quantile(0.90))
                if len(finite)
                else np.nan,
            "p95_delta":
                float(finite.quantile(0.95))
                if len(finite)
                else np.nan,
            "p99_delta":
                float(finite.quantile(0.99))
                if len(finite)
                else np.nan,
            "n_wins":
                int((out["winner"] == mode).sum()),
            "median_wall_s":
                float(wall.median()),
            "total_wall_s":
                float(wall.sum()),
        }
    )

summary = pd.DataFrame(summary)

summary_path = (
    REPO_RESULTS
    / "gate1_extreme100_coordinate_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

print()
print("GATE 1 SUMMARY")
print("=" * 120)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print()
print("Largest coordinate disagreements")
print("=" * 120)

delta_cols = [
    f"delta_{mode}"
    for mode in MODES
]

out["max_coordinate_delta"] = (
    out[delta_cols].max(axis=1)
)

cols = (
    ["catalog_row", "winner", "chi2_envelope"]
    + delta_cols
)

print(
    out.sort_values(
        "max_coordinate_delta",
        ascending=False,
    )[cols]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.8g}",
    )
)

print()
print("saved:")
print(per_event_path)
print(summary_path)
