#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

FULLROOT = (
    BASE
    / "results"
    / "h1_turnover_chi2_layers"
    / "full_population_v1"
)

CANON = FULLROOT / "h1_chi2_layers_full_canonical_333653.parquet"
MAN_BASE = FULLROOT / "h1_materialized_all.csv"
MAN_REPAIR = FULLROOT / "h1_repair_materialized_453.csv"

EXTREME = (
    BASE.parent
    / "bounds_convergence"
    / "data"
    / "extreme100_selection.csv"
)

OUT = HERE / "h1_h0_robustness_sample_frozen.csv"

SEED = 20260927
RNG = np.random.default_rng(SEED)

TCRIT = 13.15982011480088


# ----------------------------------------------------------------------
# Canonical H1 population
# ----------------------------------------------------------------------

cols = [
    "catalog_row",
    "tE_true",
    "X_pi",
    "T_fit",
    "chi2_h0_fit",
    "chi2_h1_fit",
]

df = pd.read_parquet(CANON)

missing = [c for c in cols if c not in df.columns]
if missing:
    raise RuntimeError(
        f"Canonical parquet missing columns: {missing}\n"
        f"Available: {list(df.columns)}"
    )

df = df[cols].copy()
df["catalog_row"] = df["catalog_row"].astype(np.int64)

if len(df) != 333653:
    raise RuntimeError(f"Expected 333653 rows, got {len(df)}")


# ----------------------------------------------------------------------
# H5 path map: 333200 main + 453 repairs
# ----------------------------------------------------------------------

paths = []

for p in [MAN_BASE, MAN_REPAIR]:
    x = pd.read_csv(p)

    required = {"catalog_row", "h5_path"}
    if not required.issubset(x.columns):
        raise RuntimeError(
            f"{p} lacks {required - set(x.columns)}"
        )

    paths.append(
        x[["catalog_row", "h5_path"]].copy()
    )

pm = pd.concat(paths, ignore_index=True)
pm["catalog_row"] = pm["catalog_row"].astype(np.int64)

if pm["catalog_row"].duplicated().any():
    dup = pm.loc[
        pm["catalog_row"].duplicated(False),
        "catalog_row"
    ]
    raise RuntimeError(
        f"Duplicate path rows: {dup.tolist()[:20]}"
    )

if len(pm) != 333653:
    raise RuntimeError(
        f"Expected 333653 path rows, got {len(pm)}"
    )

df = df.merge(
    pm,
    on="catalog_row",
    how="left",
    validate="one_to_one",
)

if df["h5_path"].isna().any():
    raise RuntimeError("Some canonical events lack h5_path.")


# ----------------------------------------------------------------------
# Exclude the development extreme100
# ----------------------------------------------------------------------

ext = pd.read_csv(EXTREME)

id_col = None
for c in ["catalog_row", "row", "global_i"]:
    if c in ext.columns:
        id_col = c
        break

if id_col is None:
    raise RuntimeError(
        f"Cannot identify event ID in {EXTREME}; "
        f"columns={list(ext.columns)}"
    )

extreme_ids = set(
    ext[id_col].dropna().astype(np.int64)
)

df["excluded_extreme100"] = df["catalog_row"].isin(extreme_ids)

pool = df[
    ~df["excluded_extreme100"]
].copy()


# ----------------------------------------------------------------------
# A. Uniform random component
# ----------------------------------------------------------------------

random_ids = set(
    pool.sample(
        n=min(1500, len(pool)),
        random_state=SEED,
    )["catalog_row"].astype(int)
)


# ----------------------------------------------------------------------
# B. Long/high-Xpi stratified component
# ----------------------------------------------------------------------

long = pool[
    (pool["tE_true"] >= 100.0)
    & (pool["X_pi"] >= 0.1)
].copy()

long["te_cell"] = pd.cut(
    long["tE_true"],
    bins=[100, 200, 365, np.inf],
    right=False,
    labels=["100-200", "200-365", ">=365"],
)

long["xpi_cell"] = pd.cut(
    long["X_pi"],
    bins=[0.1, 0.2, 0.5, 1.0, np.inf],
    right=False,
    labels=["0.1-0.2", "0.2-0.5", "0.5-1", ">=1"],
)

long_ids = set()

for (_, _), g in long.groupby(
    ["te_cell", "xpi_cell"],
    observed=True,
):
    n = min(75, len(g))
    if n:
        chosen = g.sample(
            n=n,
            random_state=int(
                RNG.integers(0, 2**31 - 1)
            ),
        )
        long_ids.update(
            chosen["catalog_row"].astype(int)
        )


# ----------------------------------------------------------------------
# C. Threshold-sensitive component
# ----------------------------------------------------------------------

near = pool[
    (pool["T_fit"] >= 8.0)
    & (pool["T_fit"] <= 25.0)
].copy()

near_ids = set(
    near.sample(
        n=min(1000, len(near)),
        random_state=SEED + 1,
    )["catalog_row"].astype(int)
)


# ----------------------------------------------------------------------
# Union
# ----------------------------------------------------------------------

all_ids = random_ids | long_ids | near_ids

sample = pool[
    pool["catalog_row"].isin(all_ids)
].copy()

sample["in_random"] = sample["catalog_row"].isin(random_ids)
sample["in_long_high"] = sample["catalog_row"].isin(long_ids)
sample["in_near_threshold"] = sample["catalog_row"].isin(near_ids)

sample["detected_prod"] = sample["T_fit"] > TCRIT

sample = sample.sort_values(
    "catalog_row"
).reset_index(drop=True)


# ----------------------------------------------------------------------
# Integrity
# ----------------------------------------------------------------------

if sample["catalog_row"].duplicated().any():
    raise RuntimeError("Duplicate sample IDs.")

if sample["excluded_extreme100"].any():
    raise RuntimeError("extreme100 leaked into validation sample.")

print("=" * 90)
print("H1 -> H0 OPTIMIZER ROBUSTNESS SAMPLE")
print("=" * 90)

print("canonical N          =", len(df))
print("extreme100 excluded  =", len(extreme_ids))
print("random component     =", len(random_ids))
print("long/high-Xpi        =", len(long_ids))
print("near threshold       =", len(near_ids))
print("union N              =", len(sample))

print()
print("Production detections in sample =",
      int(sample["detected_prod"].sum()))

print()
print("Component overlaps:")
print(
    sample[
        ["in_random", "in_long_high", "in_near_threshold"]
    ]
    .value_counts()
    .sort_index()
)

print()
print("Long/high-Xpi cells:")
print(
    long[
        long["catalog_row"].isin(long_ids)
    ]
    .groupby(
        ["te_cell", "xpi_cell"],
        observed=True,
    )
    .size()
)

sample.to_csv(
    OUT,
    index=False,
)

print()
print("saved =", OUT)
