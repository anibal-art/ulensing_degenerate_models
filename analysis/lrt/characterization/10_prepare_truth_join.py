#!/usr/bin/env python3
"""Join the classified H1 table to the original generating catalog.

This is the only characterization script that needs the raw generating catalog.
No fitting or simulation is performed. The join key is catalog_row. If the raw
catalog has no explicit catalog_row column, row position is used, which matches
the production definition of catalog_row for the original catalog file.

Example:
    python 10_prepare_truth_join.py --catalog /path/to/original_catalog.parquet

Supported input formats: parquet, csv, csv.gz.
"""

from pathlib import Path
import sys
import argparse
import json
import numpy as np
import pandas as pd

LRT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LRT_ROOT))
import lrt_common as C

ALIASES = {
    "true_t0": ["t0", "t0_jd", "t0_true"],
    "true_u0": ["u0", "u0_true"],
    "true_tE": ["tE", "tE_days", "tE_catalog_days", "tE_true"],
    "true_rho": ["rho", "rho_true", "rho_catalog"],
    "true_piEN": ["piEN", "piE_N", "piEN_true"],
    "true_piEE": ["piEE", "piE_E", "piEE_true"],
}

OPTIONAL_ALIASES = {
    "true_ML": ["ML", "M_L", "lens_mass", "mass_lens"],
    "true_DL": ["DL", "D_L", "lens_distance"],
    "true_DS": ["DS", "D_S", "source_distance"],
    "true_mu_rel": ["mu_rel", "mu_rel_masyr", "mu_rel_true"],
}


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported catalog format: {path}")


def resolve_alias(columns, aliases):
    for name in aliases:
        if name in columns:
            return name
    norm = {C.normalize_name(c): c for c in columns}
    for name in aliases:
        key = C.normalize_name(name)
        if key in norm:
            return norm[key]
    return None


parser = argparse.ArgumentParser()
parser.add_argument("--catalog", required=True, type=Path)
args = parser.parse_args()

catalog_path = args.catalog.expanduser().resolve()
C.require_file(catalog_path, "raw H1 generating catalog")
C.require_file(C.CLASSIFIED_H1_PATH, "classified H1 table; run validation_tests/09 first")

h1 = pd.read_parquet(C.CLASSIFIED_H1_PATH)
cat = read_table(catalog_path)

if "catalog_row" not in cat.columns:
    cat = cat.copy()
    cat.insert(0, "catalog_row", np.arange(len(cat), dtype=np.int64))
    row_definition = "row_position_in_raw_catalog"
else:
    row_definition = "explicit_catalog_row_column"

selected = {"catalog_row": "catalog_row"}
for target, aliases in ALIASES.items():
    source = resolve_alias(cat.columns, aliases)
    if source is None:
        related = [c for c in cat.columns if any(a.lower().replace("_", "") in c.lower().replace("_", "") for a in aliases)]
        raise KeyError(f"Could not resolve {target}. Aliases={aliases}. Related columns={related}")
    selected[target] = source

for target, aliases in OPTIONAL_ALIASES.items():
    source = resolve_alias(cat.columns, aliases)
    if source is not None:
        selected[target] = source

truth = pd.DataFrame({target: cat[source] for target, source in selected.items()})
truth["catalog_row"] = pd.to_numeric(truth["catalog_row"], errors="raise").astype(np.int64)
if truth["catalog_row"].duplicated().any():
    raise RuntimeError("Duplicate catalog_row values in truth catalog")

joined = h1.merge(truth, on="catalog_row", how="left", validate="one_to_one", indicator=True)
missing = int((joined["_merge"] != "both").sum())
if missing:
    bad = joined.loc[joined["_merge"] != "both", "catalog_row"].head(20).tolist()
    raise RuntimeError(f"Missing truth rows for {missing} H1 events. First={bad}")
joined = joined.drop(columns="_merge")
joined["true_piE_abs"] = np.hypot(
    pd.to_numeric(joined["true_piEN"], errors="coerce"),
    pd.to_numeric(joined["true_piEE"], errors="coerce"),
)
joined.to_parquet(C.TRUTH_JOIN_PATH, index=False, compression="zstd")

meta = {
    "catalog": str(catalog_path),
    "catalog_row_definition": row_definition,
    "n_catalog_rows": int(len(cat)),
    "n_h1_rows": int(len(h1)),
    "n_joined_rows": int(len(joined)),
    "column_mapping": selected,
}
C.save_json(meta, C.TRUTH_JOIN_PATH.with_suffix(".metadata.json"))
print(json.dumps(meta, indent=2))
print("Saved:", C.TRUTH_JOIN_PATH)
