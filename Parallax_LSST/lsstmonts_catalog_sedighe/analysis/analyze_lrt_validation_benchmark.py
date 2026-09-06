#!/usr/bin/env python

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument(
    "--study-dir",
    required=True,
)

args = parser.parse_args()

study_dir = Path(
    args.study_dir
).resolve()


# ======================================================================
# Manifest
# ======================================================================

manifest = {}

for line in (
    study_dir
    / "manifest.env"
).read_text().splitlines():

    line = line.strip()

    if not line or "=" not in line:
        continue

    k, v = line.split(
        "=",
        1,
    )

    manifest[k] = (
        v.strip()
        .strip("'")
        .strip('"')
    )


output_root = Path(
    manifest["OUTPUT_ROOT"]
)

study_tag = manifest[
    "STUDY_TAG"
]

fast_run_name = manifest[
    "FAST_RUN_NAME"
]

global_run_name = manifest[
    "GLOBAL_RUN_NAME"
]


# ======================================================================
# Load normal summaries
# ======================================================================

def load_summaries(
    method,
    run_name,
):

    root = (
        output_root
        / "runs"
        / run_name
    )

    dirs = sorted(
        root.glob(
            f"{study_tag}_{method}_v*_row*"
        )
    )

    frames = []

    for d in dirs:

        p = (
            d
            / "logs"
            / "run_summary.parquet"
        )

        if not p.exists():
            continue

        x = pd.read_parquet(
            p
        )

        x["_run_dir"] = str(
            d
        )

        frames.append(
            x
        )

    if not frames:
        raise RuntimeError(
            f"No summaries for {method}"
        )

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


fast = load_summaries(
    "FAST",
    fast_run_name,
)

normal_global = load_summaries(
    "GLOBAL",
    global_run_name,
)


# ======================================================================
# Dataset key parser from event directory
# ======================================================================

EVENT_RE = re.compile(
    r"event_(\d+)_(h0|h1)_r(\d+)$",
    re.IGNORECASE,
)


def event_key_from_path(
    path,
):

    for parent in [
        path,
        *path.parents,
    ]:

        m = EVENT_RE.match(
            parent.name
        )

        if m:

            return {
                "catalog_row":
                    int(
                        m.group(1)
                    ),

                "truth_case":
                    m.group(2).upper(),

                "noise_realization_id":
                    int(
                        m.group(3)
                    ),
            }

    return None


# ======================================================================
# Generic pyLIMA npy chi2 extraction
# ======================================================================

def load_npy_object(
    path,
):

    obj = np.load(
        path,
        allow_pickle=True,
    )

    try:
        if (
            isinstance(
                obj,
                np.ndarray,
            )
            and obj.shape == ()
        ):
            obj = obj.item()
    except Exception:
        pass

    return obj


def extract_chi2(
    path,
):

    try:
        obj = load_npy_object(
            path
        )
    except Exception:
        return np.nan

    if isinstance(
        obj,
        dict,
    ):

        # Standard pyLIMA saved result.
        for key in [
            "chi2",
            "nll_chi2",
        ]:

            if key in obj:

                try:
                    value = float(
                        np.asarray(
                            obj[key]
                        ).reshape(-1)[0]
                    )

                    if np.isfinite(
                        value
                    ):
                        if (
                            key
                            == "nll_chi2"
                        ):
                            return (
                                2.0
                                * value
                            )

                        return value

                except Exception:
                    pass

    return np.nan


# ======================================================================
# Read DE diagnostics directly from filesystem
# ======================================================================

global_root = (
    output_root
    / "runs"
    / global_run_name
)


records = []


de_files = sorted(
    global_root.rglob(
        "de_raw.npy"
    )
)


for de_path in de_files:

    key = event_key_from_path(
        de_path
    )

    if key is None:
        continue

    try:
        de = load_npy_object(
            de_path
        )
    except Exception:
        continue

    if not isinstance(
        de,
        dict,
    ):
        continue

    try:
        raw_chi2 = float(
            de.get(
                "chi2",
                np.nan,
            )
        )
    except Exception:
        raw_chi2 = np.nan


    seed_dir = de_path.parent


    m_seed = re.match(
        r"seed_(\d+)",
        seed_dir.name,
    )

    de_seed = (
        int(
            m_seed.group(1)
        )
        if m_seed
        else np.nan
    )


    # --------------------------------------------------------------
    # Find TRF polish produced from THIS DE seed.
    # --------------------------------------------------------------

    polish_root = (
        seed_dir
        / "trf_polish"
    )


    polish_files = []

    if polish_root.exists():

        polish_files = [
            p
            for p
            in polish_root.rglob(
                "*.npy"
            )
            if p.name
            != "de_raw.npy"
        ]


    polish_candidates = []

    for p in polish_files:

        chi2 = extract_chi2(
            p
        )

        if np.isfinite(
            chi2
        ):

            polish_candidates.append(
                (
                    float(
                        chi2
                    ),
                    str(
                        p
                    ),
                )
            )


    if polish_candidates:

        polish_candidates.sort(
            key=lambda x: x[0]
        )

        trf_chi2 = (
            polish_candidates[0][0]
        )

        trf_file = (
            polish_candidates[0][1]
        )

    else:

        trf_chi2 = np.nan
        trf_file = ""


    # Prefer polished solution.
    if np.isfinite(
        trf_chi2
    ):

        final_chi2 = float(
            trf_chi2
        )

        final_source = (
            "DE_TRF"
        )

    else:

        final_chi2 = float(
            raw_chi2
        )

        final_source = (
            "DE_RAW"
        )


    records.append(
        {
            **key,

            "de_seed":
                de_seed,

            "de_raw_chi2":
                raw_chi2,

            "de_trf_chi2":
                trf_chi2,

            "de_final_chi2":
                final_chi2,

            "de_final_source":
                final_source,

            "de_raw_file":
                str(
                    de_path
                ),

            "de_trf_file":
                trf_file,
        }
    )


de_df = pd.DataFrame(
    records
)


print("=" * 100)
print("DE FILE INVENTORY")
print("=" * 100)

print(
    "de_raw.npy files found =",
    len(
        de_files
    ),
)

print(
    "parsed DE records      =",
    len(
        de_df
    ),
)


if de_df.empty:

    raise RuntimeError(
        "No usable DE diagnostics found."
    )


print(
    "records with TRF polish =",
    int(
        de_df[
            "de_trf_chi2"
        ]
        .notna()
        .sum()
    ),
)


# ======================================================================
# Best DE seed per DATASET
# ======================================================================

keys = [
    "catalog_row",
    "truth_case",
    "noise_realization_id",
]


idx = (
    de_df
    .groupby(
        keys,
        dropna=False,
    )[
        "de_final_chi2"
    ]
    .idxmin()
)


best_de = (
    de_df
    .loc[
        idx
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


# ======================================================================
# FAST reference
# ======================================================================

required_fast = (
    keys
    + [
        "lrt_chi2_H0",
        "lrt_chi2_H1",
    ]
)


for c in required_fast:

    if c not in fast.columns:
        raise RuntimeError(
            f"FAST summary missing {c}"
        )


fast_small = fast[
    required_fast
].copy()


fast_small[
    "fast_chi2_H0"
] = pd.to_numeric(
    fast_small[
        "lrt_chi2_H0"
    ],
    errors="coerce",
)


fast_small[
    "fast_chi2_H1"
] = pd.to_numeric(
    fast_small[
        "lrt_chi2_H1"
    ],
    errors="coerce",
)


fast_small[
    "fast_T"
] = (
    fast_small[
        "fast_chi2_H0"
    ]
    - fast_small[
        "fast_chi2_H1"
    ]
)


# ======================================================================
# Merge
# ======================================================================

comparison = (
    fast_small
    .merge(
        best_de,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
)


comparison[
    "global_T"
] = (
    comparison[
        "fast_chi2_H0"
    ]
    - comparison[
        "de_final_chi2"
    ]
)


comparison[
    "delta_H1_fast_minus_DE"
] = (
    comparison[
        "fast_chi2_H1"
    ]
    - comparison[
        "de_final_chi2"
    ]
)


comparison[
    "de_improves_FAST"
] = (
    comparison[
        "delta_H1_fast_minus_DE"
    ]
    > 1e-6
)


comparison[
    "fast_better_than_DE"
] = (
    comparison[
        "delta_H1_fast_minus_DE"
    ]
    < -1e-6
)


comparison[
    "fast_nested_violation"
] = (
    comparison[
        "fast_T"
    ]
    < -1e-6
)


comparison[
    "global_nested_violation"
] = (
    comparison[
        "global_T"
    ]
    < -1e-6
)


# ======================================================================
# Save
# ======================================================================

de_df.to_parquet(
    study_dir
    / "DE_all_seeds.parquet",
    index=False,
)


best_de.to_parquet(
    study_dir
    / "DE_best_per_dataset.parquet",
    index=False,
)


comparison.to_parquet(
    study_dir
    / "FAST_vs_DETRF.parquet",
    index=False,
)


comparison.to_csv(
    study_dir
    / "FAST_vs_DETRF.csv",
    index=False,
)


# ======================================================================
# Report
# ======================================================================

d = pd.to_numeric(
    comparison[
        "delta_H1_fast_minus_DE"
    ],
    errors="coerce",
).dropna()


print()
print("=" * 100)
print("REAL FAST vs DE->TRF COMPARISON")
print("=" * 100)

print(
    "Matched datasets =",
    len(
        comparison
    ),
)

print()

print(
    "FAST nested violations =",
    int(
        comparison[
            "fast_nested_violation"
        ].sum()
    ),
)

print(
    "DE nested violations   =",
    int(
        comparison[
            "global_nested_violation"
        ].sum()
    ),
)

print()

print(
    "delta = chi2_FAST - chi2_DETRF"
)

for q in [
    0.50,
    0.90,
    0.95,
    0.99,
]:

    print(
        f"q{int(q*100):02d} =",
        float(
            d.quantile(
                q
            )
        ),
    )


print(
    "max =",
    float(
        d.max()
    ),
)


print()

for threshold in [
    0.01,
    0.1,
    1.0,
    5.0,
]:

    print(
        f"N(delta > {threshold}) =",
        int(
            (
                d
                > threshold
            ).sum()
        ),
    )


print()

print(
    "N DE improves FAST =",
    int(
        comparison[
            "de_improves_FAST"
        ].sum()
    ),
)

print(
    "N FAST better than DE =",
    int(
        comparison[
            "fast_better_than_DE"
        ].sum()
    ),
)


print()
print("Worst FAST misses:")
print()


show = [
    "catalog_row",
    "truth_case",
    "noise_realization_id",
    "de_seed",

    "fast_chi2_H0",
    "fast_chi2_H1",

    "de_raw_chi2",
    "de_trf_chi2",
    "de_final_chi2",

    "fast_T",
    "global_T",

    "delta_H1_fast_minus_DE",
]


worst = (
    comparison
    .sort_values(
        "delta_H1_fast_minus_DE",
        ascending=False,
    )
    .head(
        20
    )
)


print(
    worst[
        show
    ].to_string(
        index=False
    )
)


report = (
    study_dir
    / "FAST_vs_DETRF_report.txt"
)


with report.open(
    "w"
) as f:

    f.write(
        "FAST vs DE->TRF\n"
    )

    f.write(
        "=" * 100
        + "\n"
    )

    f.write(
        f"Matched datasets = {len(comparison)}\n"
    )

    f.write(
        f"median delta = {d.median()}\n"
    )

    f.write(
        f"q90 delta = {d.quantile(0.90)}\n"
    )

    f.write(
        f"q95 delta = {d.quantile(0.95)}\n"
    )

    f.write(
        f"q99 delta = {d.quantile(0.99)}\n"
    )

    f.write(
        f"max delta = {d.max()}\n"
    )

    f.write(
        "\n"
    )

    for threshold in [
        0.01,
        0.1,
        1.0,
        5.0,
    ]:

        f.write(
            f"N(delta > {threshold}) = "
            f"{int((d > threshold).sum())}\n"
        )

    f.write(
        "\n"
    )

    f.write(
        worst[
            show
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n"
    )


print()
print("Saved:")
print(
    " ",
    report
)
