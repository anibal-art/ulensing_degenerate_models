#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd

PILOT_DIR = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/"
    "detectability_catalogs/"
    "prefitDetectability_pilot_20260906T201607Z"
)

INPUT = PILOT_DIR / "detectable_events.parquet"
OUTPUT = PILOT_DIR / "lrt_validation_sample.parquet"
ROWS = PILOT_DIR / "lrt_validation_catalog_rows.txt"

N_SELECT = 36

df = pd.read_parquet(INPUT).copy()

if "detectability_pass" in df:
    df = df[df["detectability_pass"].fillna(False)].copy()

if "catalog_row" not in df:
    raise RuntimeError("catalog_row is missing")

print("Detectable candidates:", len(df))


def first_existing(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


features = {}

# ---------------------------------------------------------
# Observational / detectability coordinates.
# ---------------------------------------------------------

for out_name, column in [
    ("n_peak", "detectability_n_peak"),
    ("nearest_peak", "detectability_nearest_dt_over_tE"),
    ("delta_chi2_per_point", "detectability_delta_chi2_per_point"),
    ("n_nsigma", "detectability_n_nsigma"),
    ("max_snr", "detectability_max_snr"),
]:
    if column in df:
        features[out_name] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ---------------------------------------------------------
# Physical parameters, using aliases because the summary
# evolved during development.
# ---------------------------------------------------------

aliases = {
    "tE": [
        "tE",
        "true_tE",
        "truth_tE",
        "catalog_tE",
        "pyLIMA_tE",
    ],
    "u0": [
        "u0",
        "true_u0",
        "truth_u0",
        "catalog_u0",
        "pyLIMA_u0",
    ],
    "rho": [
        "rho",
        "true_rho",
        "truth_rho",
        "catalog_rho",
        "pyLIMA_rho",
    ],
    "piEN": [
        "effective_truth_piEN",
        "catalog_piEN",
        "piEN",
        "true_piEN",
        "truth_piEN",
    ],
    "piEE": [
        "effective_truth_piEE",
        "catalog_piEE",
        "piEE",
        "true_piEE",
        "truth_piEE",
    ],
}

resolved = {}

for name, candidates in aliases.items():
    c = first_existing(candidates)
    if c is not None:
        resolved[name] = c

if "tE" in resolved:
    features["log_tE"] = np.log10(
        np.abs(
            pd.to_numeric(df[resolved["tE"]], errors="coerce")
        ) + 1e-12
    )

if "u0" in resolved:
    features["abs_u0"] = np.abs(
        pd.to_numeric(df[resolved["u0"]], errors="coerce")
    )

if "rho" in resolved:
    features["log_rho"] = np.log10(
        np.abs(
            pd.to_numeric(df[resolved["rho"]], errors="coerce")
        ) + 1e-15
    )

if "piEN" in resolved and "piEE" in resolved:
    pen = pd.to_numeric(
        df[resolved["piEN"]],
        errors="coerce",
    )
    pee = pd.to_numeric(
        df[resolved["piEE"]],
        errors="coerce",
    )

    piE = np.sqrt(
        pen**2 + pee**2
    )

    features["log_piE"] = np.log10(
        piE + 1e-12
    )


print("Physical aliases:")
for k, v in resolved.items():
    print(f"  {k:5s} <- {v}")

print("Selection features:")
for k in features:
    print(" ", k)


# ---------------------------------------------------------
# Rank transform.
#
# This prevents a large-S/N coordinate from numerically
# dominating tE, coverage, piE, etc.
# ---------------------------------------------------------

X_columns = []
feature_names = []

for name, values in features.items():

    values = pd.Series(
        values,
        index=df.index,
        dtype=float,
    )

    if values.notna().sum() < 10:
        continue

    med = values.median()

    values = values.fillna(
        med
    )

    rank = values.rank(
        pct=True,
        method="average",
    )

    X_columns.append(
        rank.to_numpy(dtype=float)
    )

    feature_names.append(
        name
    )


if len(X_columns) < 3:
    raise RuntimeError(
        "Too few usable dimensions for representative selection."
    )

X = np.column_stack(
    X_columns
)


# ---------------------------------------------------------
# Start with extrema along every dimension.
# ---------------------------------------------------------

selected = []

for j in range(X.shape[1]):

    for idx in [
        int(np.argmin(X[:, j])),
        int(np.argmax(X[:, j])),
    ]:
        if idx not in selected:
            selected.append(idx)

        if len(selected) >= N_SELECT:
            break

    if len(selected) >= N_SELECT:
        break


# ---------------------------------------------------------
# Greedy maximin:
# repeatedly select the event farthest from the already
# selected set in rank-space.
# ---------------------------------------------------------

while len(selected) < min(N_SELECT, len(df)):

    selected_X = X[selected]

    d2 = (
        (
            X[:, None, :]
            - selected_X[None, :, :]
        )**2
    ).sum(axis=2)

    min_d2 = d2.min(axis=1)

    min_d2[selected] = -np.inf

    idx = int(
        np.argmax(min_d2)
    )

    selected.append(
        idx
    )


sample = (
    df.iloc[selected]
    .copy()
    .reset_index(drop=True)
)

sample.insert(
    0,
    "validation_index",
    np.arange(len(sample)),
)

sample.to_parquet(
    OUTPUT,
    index=False,
)

with ROWS.open("w") as f:
    for row in sample["catalog_row"].astype(int):
        f.write(f"{row}\n")


print()
print("=" * 80)
print("LRT VALIDATION SAMPLE")
print("=" * 80)
print("N selected:", len(sample))
print("features  :", feature_names)
print()
print("catalog rows:")
print(
    " ".join(
        str(x)
        for x
        in sample["catalog_row"].astype(int)
    )
)
print()
print("Saved:")
print(" ", OUTPUT)
print(" ", ROWS)
