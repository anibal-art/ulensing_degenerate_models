#!/usr/bin/env bash

set -euo pipefail

# ============================================================================
# LRT validation benchmark
#
# Purpose:
#   Use real detectable events from the validated 5000-row pilot to decide
#   the numerical H0/H1 fitting policy before launching the full catalog.
#
# FAST:
#   bounded flux profile
#   H0 TRF
#   H1 embedded-H0
#   H1 3x3 piE multistart
#   adaptive TRF polish
#
# GLOBAL:
#   same FAST procedure
#   + strict Differential Evolution
#   + TRF polish
#   + two DE seeds
#
# Each physical event:
#   2 truth cases x 2 paired noise realizations = 4 datasets
#
# Detectability remains ACTIVE:
#   simulate -> Asimov detectability -> FAIL stop / PASS fit H0+H1
# ============================================================================


PROJECT_DIR="/export/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe"

OUTPUT_ROOT="/export/storage3/rubin/microlensing/romanrubin/hidden_parallax"

PYTHON_BIN="/home/anibalvarela/.conda/envs/pyLIMA_test/bin/python"

RUNNER="${PROJECT_DIR}/run_lsstmonts_catalog_hidden_parallax.py"

# IMPORTANT:
# Explicitly use the validated pilot, NOT LATEST.
PILOT_TAG="prefitDetectability_pilot_20260906T201607Z"

PILOT_DIR="${OUTPUT_ROOT}/detectability_catalogs/${PILOT_TAG}"

PILOT_MANIFEST="${PILOT_DIR}/manifest.env"

DETECTABLE_PARQUET="${PILOT_DIR}/detectable_events.parquet"

N_SELECT="${N_SELECT:-36}"

N_REALIZATIONS="${N_REALIZATIONS:-2}"

WORKERS="${WORKERS:-2}"

FAST_MAX_CONCURRENT="${FAST_MAX_CONCURRENT:-12}"

GLOBAL_MAX_CONCURRENT="${GLOBAL_MAX_CONCURRENT:-6}"

PARTITION="${PARTITION:-cosmoobs}"

FAST_TIME="${FAST_TIME:-06:00:00}"

GLOBAL_TIME="${GLOBAL_TIME:-24:00:00}"

FAST_MEM="${FAST_MEM:-20G}"

GLOBAL_MEM="${GLOBAL_MEM:-28G}"


# ============================================================================
# Preflight
# ============================================================================

echo "======================================================================"
echo "LRT VALIDATION BENCHMARK SETUP"
echo "======================================================================"

for f in \
    "$RUNNER" \
    "$PILOT_MANIFEST" \
    "$DETECTABLE_PARQUET"
do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing file:"
        echo "  $f"
        exit 1
    fi
done


if ! grep -q "PREFIT_DETECTABILITY_RUNTIME_PATCH_V1" "$RUNNER"; then
    echo "ERROR:"
    echo "PREFIT_DETECTABILITY_RUNTIME_PATCH_V1 not found in runner."
    exit 1
fi


if grep -q "BOUNDED_FLUX_PROFILE" "$RUNNER"; then
    echo "[preflight] bounded flux profiling patch found"
else
    echo "WARNING:"
    echo "Could not find BOUNDED_FLUX_PROFILE marker in runner."
fi


if grep -q "GLOBAL_DE" "$RUNNER"; then
    echo "[preflight] GLOBAL DE runtime code found"
else
    echo "ERROR:"
    echo "Could not find GLOBAL_DE code in runner."
    exit 1
fi


# ============================================================================
# Read source config from the pilot manifest.
# ============================================================================

SOURCE_CONFIG=$(
    awk -F= '
        $1=="CFG_PATH" {
            gsub(/\047/, "", $2);
            gsub(/"/, "", $2);
            print $2
        }
    ' "$PILOT_MANIFEST"
)

if [[ -z "$SOURCE_CONFIG" || ! -f "$SOURCE_CONFIG" ]]; then
    echo "ERROR: could not resolve source config from:"
    echo "  $PILOT_MANIFEST"
    echo "Resolved:"
    echo "  $SOURCE_CONFIG"
    exit 1
fi


echo
echo "Validated pilot:"
echo "  $PILOT_DIR"

echo
echo "Frozen source config:"
echo "  $SOURCE_CONFIG"


# ============================================================================
# Study directories
# ============================================================================

STAMP=$(date -u +%Y%m%dT%H%M%SZ)

STUDY_TAG="lrt_validation_detectable_${STAMP}"

STUDY_DIR="${OUTPUT_ROOT}/lrt_validation/${STUDY_TAG}"

FROZEN_DIR="${OUTPUT_ROOT}/production_configs/${STUDY_TAG}"

mkdir -p \
    "$STUDY_DIR" \
    "$FROZEN_DIR" \
    "${PROJECT_DIR}/slurm_logs"


FAST_CONFIG="${FROZEN_DIR}/FAST.json"

GLOBAL_CONFIG="${FROZEN_DIR}/GLOBAL.json"

SAMPLE_PARQUET="${STUDY_DIR}/lrt_validation_sample.parquet"

ROWS_FILE="${STUDY_DIR}/lrt_validation_catalog_rows.txt"

SELECTION_REPORT="${STUDY_DIR}/selection_report.txt"

MANIFEST="${STUDY_DIR}/manifest.env"


FAST_RUN_NAME="LSSTMONTS_${STUDY_TAG}_FAST"

GLOBAL_RUN_NAME="LSSTMONTS_${STUDY_TAG}_GLOBAL"


# ============================================================================
# Select events + generate configs automatically
# ============================================================================

"$PYTHON_BIN" - \
    "$DETECTABLE_PARQUET" \
    "$SOURCE_CONFIG" \
    "$FAST_CONFIG" \
    "$GLOBAL_CONFIG" \
    "$SAMPLE_PARQUET" \
    "$ROWS_FILE" \
    "$SELECTION_REPORT" \
    "$FAST_RUN_NAME" \
    "$GLOBAL_RUN_NAME" \
    "$N_SELECT" \
    "$N_REALIZATIONS" \
<<'PY'
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


(
    detectable_file,
    source_config_file,
    fast_config_file,
    global_config_file,
    sample_file,
    rows_file,
    report_file,
    fast_run_name,
    global_run_name,
    n_select,
    n_realizations,
) = sys.argv[1:]


detectable_file = Path(detectable_file)
source_config_file = Path(source_config_file)
fast_config_file = Path(fast_config_file)
global_config_file = Path(global_config_file)
sample_file = Path(sample_file)
rows_file = Path(rows_file)
report_file = Path(report_file)

n_select = int(n_select)
n_realizations = int(n_realizations)


# ============================================================================
# Load validated PASS events
# ============================================================================

df = pd.read_parquet(
    detectable_file
).copy()


if "detectability_pass" in df.columns:
    df = df[
        df["detectability_pass"]
        .fillna(False)
        .astype(bool)
    ].copy()


if len(df) == 0:
    raise RuntimeError(
        "No detectable events found."
    )


if "catalog_row" not in df.columns:
    raise RuntimeError(
        "catalog_row is missing from detectable_events.parquet"
    )


# ============================================================================
# Resolve representative selection coordinates
# ============================================================================

def first_existing(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


features = {}
resolved = {}


# Observational information
observational = {
    "n_peak":
        "detectability_n_peak",

    "nearest_peak":
        "detectability_nearest_dt_over_tE",

    "delta_chi2_per_point":
        "detectability_delta_chi2_per_point",

    "n_nsigma":
        "detectability_n_nsigma",

    "max_snr":
        "detectability_max_snr",
}


for name, col in observational.items():
    if col in df.columns:
        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        if values.notna().sum() >= 10:
            features[name] = values


# Physical aliases because summary column naming changed during development.
aliases = {
    "tE": [
        "effective_truth_tE",
        "true_tE",
        "truth_tE",
        "catalog_tE",
        "tE",
        "pyLIMA_tE",
    ],

    "u0": [
        "effective_truth_u0",
        "true_u0",
        "truth_u0",
        "catalog_u0",
        "u0",
        "pyLIMA_u0",
    ],

    "rho": [
        "effective_truth_rho",
        "true_rho",
        "truth_rho",
        "catalog_rho",
        "rho",
        "pyLIMA_rho",
    ],

    "piEN": [
        "effective_truth_piEN",
        "catalog_piEN",
        "true_piEN",
        "truth_piEN",
        "piEN",
    ],

    "piEE": [
        "effective_truth_piEE",
        "catalog_piEE",
        "true_piEE",
        "truth_piEE",
        "piEE",
    ],
}


for name, candidates in aliases.items():
    c = first_existing(candidates)

    if c is not None:
        resolved[name] = c


if "tE" in resolved:
    x = pd.to_numeric(
        df[resolved["tE"]],
        errors="coerce",
    )

    features["log_tE"] = np.log10(
        np.abs(x) + 1e-12
    )


if "u0" in resolved:
    x = pd.to_numeric(
        df[resolved["u0"]],
        errors="coerce",
    )

    features["abs_u0"] = np.abs(x)


if "rho" in resolved:
    x = pd.to_numeric(
        df[resolved["rho"]],
        errors="coerce",
    )

    features["log_rho"] = np.log10(
        np.abs(x) + 1e-15
    )


if (
    "piEN" in resolved
    and "piEE" in resolved
):
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


# ============================================================================
# Rank-space representation
#
# Each dimension contributes similarly regardless of physical scale.
# ============================================================================

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

    values = values.fillna(
        values.median()
    )

    rank = values.rank(
        pct=True,
        method="average",
    )

    X_columns.append(
        rank.to_numpy(
            dtype=float
        )
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


# ============================================================================
# Deterministic maximin selection
# ============================================================================

selected = []


# Start with extrema of each dimension.
for j in range(
    X.shape[1]
):

    extrema = [
        int(
            np.argmin(
                X[:, j]
            )
        ),
        int(
            np.argmax(
                X[:, j]
            )
        ),
    ]

    for idx in extrema:

        if idx not in selected:
            selected.append(
                idx
            )

        if len(selected) >= n_select:
            break

    if len(selected) >= n_select:
        break


# Greedy farthest-point sampling.
while len(selected) < min(
    n_select,
    len(df),
):

    SX = X[
        selected
    ]

    d2 = (
        (
            X[:, None, :]
            - SX[None, :, :]
        )**2
    ).sum(
        axis=2
    )

    min_d2 = d2.min(
        axis=1
    )

    min_d2[
        selected
    ] = -np.inf

    next_idx = int(
        np.argmax(
            min_d2
        )
    )

    selected.append(
        next_idx
    )


sample = (
    df.iloc[
        selected
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


sample.insert(
    0,
    "validation_index",
    np.arange(
        len(sample)
    ),
)


sample.to_parquet(
    sample_file,
    index=False,
)


with rows_file.open(
    "w"
) as f:

    for row in sample[
        "catalog_row"
    ].astype(int):

        f.write(
            f"{row}\n"
        )


# ============================================================================
# Config generation
# ============================================================================

source_cfg = json.loads(
    source_config_file.read_text()
)


def prepare_common(
    source,
    run_name,
):

    cfg = copy.deepcopy(
        source
    )

    cfg[
        "run_name"
    ] = run_name


    # ----------------------------------------------------------------
    # Noise Monte Carlo
    # ----------------------------------------------------------------

    noise = cfg.setdefault(
        "noise_realizations",
        {},
    )

    noise[
        "enabled"
    ] = True

    noise[
        "n_realizations"
    ] = n_realizations

    noise[
        "truth_cases"
    ] = [
        "H0",
        "H1",
    ]

    # Fixed between FAST and GLOBAL.
    noise[
        "base_seed"
    ] = 20260906

    noise[
        "paired_noise"
    ] = True


    # ----------------------------------------------------------------
    # Detection gate:
    # simulate -> detect -> fit only on PASS
    # ----------------------------------------------------------------

    simulation = cfg.setdefault(
        "simulation",
        {},
    )

    simulation[
        "apply_detection_criteria"
    ] = False

    simulation[
        "apply_photometric_filter"
    ] = True


    selection = cfg.setdefault(
        "selection",
        {},
    )

    detect = selection.setdefault(
        "prefit_detectability",
        {},
    )

    detect.update(
        {
            "enabled": True,

            # IMPORTANT:
            # PASS continues to H0/H1.
            "audit_only": False,

            # Do not select on parallax signal.
            "reference_model":
                "no_parallax",

            "bands":
                "rubin",

            "peak_window_tE":
                1.0,

            "min_total_points":
                10,

            "min_bands":
                3,

            "min_peak_points":
                5,

            "min_left_peak_points":
                1,

            "min_right_peak_points":
                1,

            "nsigma":
                3.0,

            "min_nsigma_points":
                6,

            "min_delta_chi2_per_point":
                2.0,

            "max_nearest_peak_distance_tE":
                None,
        }
    )


    # ----------------------------------------------------------------
    # H0/H1 nested LRT
    # ----------------------------------------------------------------

    if (
        "fit" not in cfg
        or not isinstance(
            cfg["fit"],
            dict,
        )
    ):
        raise RuntimeError(
            "Source config has no fit dict."
        )


    fit = cfg[
        "fit"
    ]


    if (
        "fits" not in fit
        or not isinstance(
            fit["fits"],
            dict,
        )
    ):
        raise RuntimeError(
            "Source config has no fit.fits dict."
        )


    fits = fit[
        "fits"
    ]


    for key in [
        "H0",
        "H1",
    ]:
        if key not in fits:
            raise RuntimeError(
                f"Missing fit.fits.{key}"
            )


    fit[
        "run_multiple_fits"
    ] = True

    fit[
        "primary_fit"
    ] = "H0"

    fit[
        "lrt"
    ] = {
        "null":
            "H0",

        "alternative":
            "H1",

        "delta_k":
            2,
    }


    fits[
        "H0"
    ][
        "model"
    ] = "FSPL"

    fits[
        "H0"
    ][
        "parallax"
    ] = False


    fits[
        "H1"
    ][
        "model"
    ] = "FSPL"

    fits[
        "H1"
    ][
        "parallax"
    ] = True


    # Keep the validated nested bounds from the frozen source config.
    # This benchmark is meant to TEST them, not silently redefine them.


    # ----------------------------------------------------------------
    # Shared TRF settings
    # ----------------------------------------------------------------

    fit[
        "initial_guess"
    ] = "truth"


    optimizer = fit.setdefault(
        "optimizer_options",
        {},
    )

    optimizer.update(
        {
            "xtol":
                1e-8,

            "ftol":
                1e-8,

            "gtol":
                1e-8,

            "max_nfev":
                50000,

            "x_scale":
                "jac",
        }
    )


    # ----------------------------------------------------------------
    # Embedded H0 -> H1
    # ----------------------------------------------------------------

    fit[
        "h1_initialization"
    ] = "embedded_H0"

    fit[
        "h1_embed_fluxes_from_H0"
    ] = True


    # ----------------------------------------------------------------
    # H1 multistart
    # ----------------------------------------------------------------

    ms = fit.setdefault(
        "h1_multistart",
        {},
    )

    ms[
        "enabled"
    ] = True

    ms[
        "piE_grid_fractions"
    ] = [
        -0.5,
        0.0,
        0.5,
    ]

    ms[
        "include_auto_center"
    ] = True

    ms[
        "include_exact_H0_center"
    ] = True

    ms[
        "nested_chi2_tolerance"
    ] = 1e-6


    # Adaptive local polish
    ms[
        "polish_winner"
    ] = "adaptive"

    ms[
        "polish_optimality_threshold"
    ] = 0.05

    ms[
        "polish_optimizer_options"
    ] = {
        "xtol":
            1e-10,

        "ftol":
            1e-10,

        "gtol":
            1e-8,

        "max_nfev":
            50000,

        "x_scale":
            "jac",
    }


    # We already established top-k local polish was not enough.
    ms[
        "diagnostic_polish_top_k"
    ] = 0


    # ----------------------------------------------------------------
    # Production noise reduction
    # ----------------------------------------------------------------

    hp = cfg.setdefault(
        "hidden_parallax",
        {},
    )

    hp[
        "make_plots"
    ] = False

    hp[
        "append_summary"
    ] = True

    hp[
        "summary_name"
    ] = "run_summary.parquet"


    return cfg


# ============================================================================
# FAST
# ============================================================================

fast_cfg = prepare_common(
    source_cfg,
    fast_run_name,
)


fast_ms = fast_cfg[
    "fit"
][
    "h1_multistart"
]


fast_de = fast_ms.setdefault(
    "diagnostic_global_de",
    {},
)

fast_de[
    "enabled"
] = False


# ============================================================================
# GLOBAL
# ============================================================================

global_cfg = prepare_common(
    source_cfg,
    global_run_name,
)


global_ms = global_cfg[
    "fit"
][
    "h1_multistart"
]


global_de = global_ms.setdefault(
    "diagnostic_global_de",
    {},
)


global_de.update(
    {
        "enabled":
            True,

        "population_size":
            10,

        # Strict test used to recover the real deep basins.
        "max_iteration":
            1500,

        "atol":
            1e-4,

        "tol":
            0.0,

        "strategy":
            "rand1bin",

        "seeds": [
            20260903,
            20260904,
        ],

        "trf_polish_optimizer_options": {
            "xtol":
                1e-10,

            "ftol":
                1e-10,

            "gtol":
                1e-8,

            "max_nfev":
                50000,

            "x_scale":
                "jac",
        },
    }
)


# ============================================================================
# Save frozen configs
# ============================================================================

fast_config_file.write_text(
    json.dumps(
        fast_cfg,
        indent=2,
    )
    + "\n"
)


global_config_file.write_text(
    json.dumps(
        global_cfg,
        indent=2,
    )
    + "\n"
)


# ============================================================================
# Selection report
# ============================================================================

with report_file.open(
    "w"
) as f:

    print(
        "LRT validation sample",
        file=f,
    )

    print(
        "=" * 80,
        file=f,
    )

    print(
        f"Detectable candidates: {len(df)}",
        file=f,
    )

    print(
        f"N selected: {len(sample)}",
        file=f,
    )

    print(
        "",
        file=f,
    )

    print(
        "Selection features:",
        file=f,
    )

    for name in feature_names:
        print(
            f"  {name}",
            file=f,
        )

    print(
        "",
        file=f,
    )

    print(
        "Resolved physical columns:",
        file=f,
    )

    for k, v in resolved.items():
        print(
            f"  {k:5s} <- {v}",
            file=f,
        )

    print(
        "",
        file=f,
    )

    print(
        "catalog_row:",
        file=f,
    )

    for row in sample[
        "catalog_row"
    ].astype(int):

        print(
            f"  {row}",
            file=f,
        )


print("=" * 80)
print("VALIDATION SAMPLE")
print("=" * 80)

print(
    "Detectable candidates:",
    len(df),
)

print(
    "Selected:",
    len(sample),
)

print(
    "Features:",
    feature_names,
)

print(
    "Rows:",
    " ".join(
        str(x)
        for x
        in sample[
            "catalog_row"
        ].astype(int)
    ),
)

print()
print("FAST config:")
print(
    fast_config_file
)

print()
print("GLOBAL config:")
print(
    global_config_file
)
PY


# ============================================================================
# Freeze hashes
# ============================================================================

sha256sum \
    "$SOURCE_CONFIG" \
    "$FAST_CONFIG" \
    "$GLOBAL_CONFIG" \
    > "${STUDY_DIR}/SHA256SUMS.txt"


# ============================================================================
# Validate JSONs
# ============================================================================

"$PYTHON_BIN" - \
    "$FAST_CONFIG" \
    "$GLOBAL_CONFIG" \
<<'PY'
import json
import sys

for path in sys.argv[1:]:

    with open(path) as f:
        cfg = json.load(f)

    fit = cfg["fit"]
    detect = cfg["selection"]["prefit_detectability"]
    noise = cfg["noise_realizations"]

    assert fit["run_multiple_fits"] is True
    assert fit["primary_fit"] == "H0"

    assert fit["fits"]["H0"]["parallax"] is False
    assert fit["fits"]["H1"]["parallax"] is True

    assert fit["lrt"]["null"] == "H0"
    assert fit["lrt"]["alternative"] == "H1"
    assert fit["lrt"]["delta_k"] == 2

    assert detect["enabled"] is True
    assert detect["audit_only"] is False
    assert detect["reference_model"] == "no_parallax"

    assert noise["enabled"] is True
    assert noise["paired_noise"] is True
    assert noise["truth_cases"] == ["H0", "H1"]

    print(
        "[config OK]",
        path,
    )
PY


# ============================================================================
# Manifest
# ============================================================================

cat > "$MANIFEST" <<EOF
STUDY_TAG='${STUDY_TAG}'
STUDY_DIR='${STUDY_DIR}'
PROJECT_DIR='${PROJECT_DIR}'
OUTPUT_ROOT='${OUTPUT_ROOT}'
PILOT_DIR='${PILOT_DIR}'
SOURCE_CONFIG='${SOURCE_CONFIG}'
FAST_CONFIG='${FAST_CONFIG}'
GLOBAL_CONFIG='${GLOBAL_CONFIG}'
FAST_RUN_NAME='${FAST_RUN_NAME}'
GLOBAL_RUN_NAME='${GLOBAL_RUN_NAME}'
SAMPLE_PARQUET='${SAMPLE_PARQUET}'
ROWS_FILE='${ROWS_FILE}'
N_SELECT='${N_SELECT}'
N_REALIZATIONS='${N_REALIZATIONS}'
WORKERS='${WORKERS}'
EOF


echo "$STUDY_TAG" \
    > "${OUTPUT_ROOT}/lrt_validation/LATEST"


# ============================================================================
# Generic SLURM worker: one physical row per array element
# ============================================================================

SLURM_RUNNER="${PROJECT_DIR}/validation/lrt/run_lrt_validation_one_row.slurm"

cat > "$SLURM_RUNNER" <<'SLURM'
#!/usr/bin/env bash

set -euo pipefail


: "${PROJECT_DIR:?}"
: "${OUTPUT_ROOT:?}"
: "${CFG_PATH:?}"
: "${RUN_NAME:?}"
: "${ROWS_FILE:?}"
: "${METHOD:?}"
: "${STUDY_TAG:?}"
: "${WORKERS:?}"


PYTHON_BIN="/home/anibalvarela/.conda/envs/pyLIMA_test/bin/python"


INDEX="${SLURM_ARRAY_TASK_ID}"

LINE=$(( INDEX + 1 ))


CATALOG_ROW=$(
    sed -n "${LINE}p" \
        "$ROWS_FILE"
)


if [[ -z "$CATALOG_ROW" ]]; then
    echo "ERROR: no catalog row for array index ${INDEX}"
    exit 2
fi


STOP=$(( CATALOG_ROW + 1 ))


SUFFIX="${STUDY_TAG}_${METHOD}_v$(printf '%03d' "$INDEX")_row${CATALOG_ROW}"


echo "======================================================================"
echo "LRT VALIDATION"
echo "======================================================================"
echo "method       = ${METHOD}"
echo "array index  = ${INDEX}"
echo "catalog row  = ${CATALOG_ROW}"
echo "config       = ${CFG_PATH}"
echo "run name     = ${RUN_NAME}"
echo "suffix       = ${SUFFIX}"
echo "workers      = ${WORKERS}"
echo "======================================================================"


# Exact bounded profiling likelihood.
export HIDDEN_PARALLAX_BOUNDED_PROFILE=1


cd "$PROJECT_DIR"


"$PYTHON_BIN" \
    run_lsstmonts_catalog_hidden_parallax.py \
    --config "$CFG_PATH" \
    --catalog-row-start "$CATALOG_ROW" \
    --catalog-row-stop "$STOP" \
    --workers "$WORKERS" \
    --run-name-suffix "$SUFFIX"


SUMMARY="${OUTPUT_ROOT}/runs/${RUN_NAME}/${SUFFIX}/logs/run_summary.parquet"


if [[ ! -f "$SUMMARY" ]]; then
    echo "ERROR: expected summary missing:"
    echo "  $SUMMARY"
    exit 3
fi


echo
echo "SUMMARY:"
echo "  $SUMMARY"

echo
echo "DONE"
SLURM


chmod +x \
    "$SLURM_RUNNER"


# ============================================================================
# Analysis script
# ============================================================================

ANALYZER="${PROJECT_DIR}/analysis/analyze_lrt_validation_benchmark.py"

cat > "$ANALYZER" <<'PY'
#!/usr/bin/env python

import argparse
import json
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


manifest_file = (
    study_dir
    / "manifest.env"
)


if not manifest_file.exists():
    raise FileNotFoundError(
        manifest_file
    )


# ============================================================================
# Parse manifest
# ============================================================================

manifest = {}


for raw in manifest_file.read_text().splitlines():

    raw = raw.strip()

    if not raw or "=" not in raw:
        continue

    key, value = raw.split(
        "=",
        1,
    )

    manifest[
        key.strip()
    ] = (
        value
        .strip()
        .strip("'")
        .strip('"')
    )


output_root = Path(
    manifest[
        "OUTPUT_ROOT"
    ]
)


fast_run_name = manifest[
    "FAST_RUN_NAME"
]


global_run_name = manifest[
    "GLOBAL_RUN_NAME"
]


study_tag = manifest[
    "STUDY_TAG"
]


# ============================================================================
# Read summaries, searching only inside the two dedicated run_name folders.
# ============================================================================

def load_method(
    method,
    run_name,
):

    root = (
        output_root
        / "runs"
        / run_name
    )


    pattern = (
        f"{study_tag}_{method}_v*_row*"
    )


    run_dirs = sorted(
        root.glob(
            pattern
        )
    )


    frames = []


    for run_dir in run_dirs:

        p = (
            run_dir
            / "logs"
            / "run_summary.parquet"
        )

        if not p.exists():
            continue

        d = pd.read_parquet(
            p
        )

        d[
            "_run_dir"
        ] = str(
            run_dir
        )

        frames.append(
            d
        )


    if not frames:
        raise RuntimeError(
            f"No summaries found for {method} under {root}"
        )


    out = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


    return out


fast = load_method(
    "FAST",
    fast_run_name,
)


glob = load_method(
    "GLOBAL",
    global_run_name,
)


# ============================================================================
# Helpers
# ============================================================================

def choose_column(
    frame,
    candidates,
    label,
):

    for c in candidates:
        if c in frame.columns:
            return c

    possible = [
        c
        for c in frame.columns
        if label.lower()
        in c.lower()
    ]

    raise RuntimeError(
        f"Could not find {label}. "
        f"Candidate-looking columns: {possible}"
    )


def numeric(
    s,
):
    return pd.to_numeric(
        s,
        errors="coerce",
    )


# ============================================================================
# Keys
# ============================================================================

keys = [
    "catalog_row",
    "truth_case",
    "noise_realization_id",
]


for key in keys:

    if key not in fast.columns:
        raise RuntimeError(
            f"FAST missing key {key}"
        )

    if key not in glob.columns:
        raise RuntimeError(
            f"GLOBAL missing key {key}"
        )


# ============================================================================
# Main LRT chi2 columns
# ============================================================================

h0_candidates = [
    "lrt_chi2_H0",
    "multi_chi2_H0",
    "multi_H0_chi2",
    "chi2_H0",
]


h1_candidates = [
    "lrt_chi2_H1",
    "multi_chi2_H1",
    "multi_H1_chi2",
    "chi2_H1",
]


fast_h0_col = choose_column(
    fast,
    h0_candidates,
    "H0 chi2",
)


fast_h1_col = choose_column(
    fast,
    h1_candidates,
    "H1 chi2",
)


global_h0_col = choose_column(
    glob,
    h0_candidates,
    "H0 chi2",
)


global_h1_col = choose_column(
    glob,
    h1_candidates,
    "H1 chi2",
)


# ============================================================================
# GLOBAL DE candidate columns
#
# Use the minimum among the normal H1 solution and any explicit
# GLOBAL-DE H1 chi2 diagnostic that the runner saved.
# ============================================================================

global_de_chi2_cols = []


for c in glob.columns:

    low = c.lower()

    if (
        "global" in low
        and "de" in low
        and "chi2" in low
        and "delta" not in low
        and "h0" not in low
    ):
        global_de_chi2_cols.append(
            c
        )


global_h1_matrix = [
    numeric(
        glob[
            global_h1_col
        ]
    ).to_numpy()
]


for c in global_de_chi2_cols:

    arr = numeric(
        glob[c]
    ).to_numpy()

    global_h1_matrix.append(
        arr
    )


global_h1_ref = np.nanmin(
    np.vstack(
        global_h1_matrix
    ),
    axis=0,
)


# ============================================================================
# Compact frames
# ============================================================================

fast_small = fast[
    keys
].copy()


fast_small[
    "fast_chi2_H0"
] = numeric(
    fast[
        fast_h0_col
    ]
)


fast_small[
    "fast_chi2_H1"
] = numeric(
    fast[
        fast_h1_col
    ]
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


glob_small = glob[
    keys
].copy()


glob_small[
    "global_chi2_H0"
] = numeric(
    glob[
        global_h0_col
    ]
)


glob_small[
    "global_main_chi2_H1"
] = numeric(
    glob[
        global_h1_col
    ]
)


glob_small[
    "global_ref_chi2_H1"
] = global_h1_ref


glob_small[
    "global_T"
] = (
    glob_small[
        "global_chi2_H0"
    ]
    - glob_small[
        "global_ref_chi2_H1"
    ]
)


# ============================================================================
# Merge identical datasets
# ============================================================================

comparison = fast_small.merge(
    glob_small,
    on=keys,
    how="outer",
    validate="one_to_one",
    indicator=True,
)


comparison[
    "delta_H0_fast_minus_global"
] = (
    comparison[
        "fast_chi2_H0"
    ]
    - comparison[
        "global_chi2_H0"
    ]
)


comparison[
    "delta_H1_fast_minus_global"
] = (
    comparison[
        "fast_chi2_H1"
    ]
    - comparison[
        "global_ref_chi2_H1"
    ]
)


comparison[
    "delta_T_global_minus_fast"
] = (
    comparison[
        "global_T"
    ]
    - comparison[
        "fast_T"
    ]
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


# ============================================================================
# Save
# ============================================================================

comparison_path = (
    study_dir
    / "FAST_vs_GLOBAL.parquet"
)


comparison_csv = (
    study_dir
    / "FAST_vs_GLOBAL.csv"
)


comparison.to_parquet(
    comparison_path,
    index=False,
)


comparison.to_csv(
    comparison_csv,
    index=False,
)


# ============================================================================
# Summary
# ============================================================================

matched = comparison[
    comparison[
        "_merge"
    ] == "both"
].copy()


d = numeric(
    matched[
        "delta_H1_fast_minus_global"
    ]
)


summary = {
    "n_fast_rows":
        int(
            len(fast)
        ),

    "n_global_rows":
        int(
            len(glob)
        ),

    "n_matched":
        int(
            len(matched)
        ),

    "fast_h0_column":
        fast_h0_col,

    "fast_h1_column":
        fast_h1_col,

    "global_h0_column":
        global_h0_col,

    "global_main_h1_column":
        global_h1_col,

    "global_de_chi2_columns":
        global_de_chi2_cols,

    "n_fast_nested_violations":
        int(
            matched[
                "fast_nested_violation"
            ].sum()
        ),

    "n_global_nested_violations":
        int(
            matched[
                "global_nested_violation"
            ].sum()
        ),

    "delta_H1_fast_minus_global": {
        "median":
            float(
                d.median()
            ),

        "q90":
            float(
                d.quantile(
                    0.90
                )
            ),

        "q95":
            float(
                d.quantile(
                    0.95
                )
            ),

        "q99":
            float(
                d.quantile(
                    0.99
                )
            ),

        "max":
            float(
                d.max()
            ),

        "n_gt_0p01":
            int(
                (
                    d > 0.01
                ).sum()
            ),

        "n_gt_0p1":
            int(
                (
                    d > 0.1
                ).sum()
            ),

        "n_gt_1":
            int(
                (
                    d > 1.0
                ).sum()
            ),

        "n_gt_5":
            int(
                (
                    d > 5.0
                ).sum()
            ),
    },
}


summary_path = (
    study_dir
    / "comparison_summary.json"
)


summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)


# ============================================================================
# Console / text report
# ============================================================================

lines = []


def emit(*items):
    text = " ".join(
        str(x)
        for x in items
    )

    print(
        text
    )

    lines.append(
        text
    )


emit(
    "=" * 100
)

emit(
    "LRT VALIDATION: FAST vs GLOBAL"
)

emit(
    "=" * 100
)

emit(
    "FAST rows   =",
    len(fast),
)

emit(
    "GLOBAL rows =",
    len(glob),
)

emit(
    "Matched     =",
    len(matched),
)

emit()

emit(
    "FAST H0 column:",
    fast_h0_col,
)

emit(
    "FAST H1 column:",
    fast_h1_col,
)

emit(
    "GLOBAL H0 column:",
    global_h0_col,
)

emit(
    "GLOBAL main H1 column:",
    global_h1_col,
)

emit(
    "GLOBAL-DE chi2 columns:",
    global_de_chi2_cols,
)

emit()

emit(
    "Nested violations FAST   =",
    int(
        matched[
            "fast_nested_violation"
        ].sum()
    ),
)

emit(
    "Nested violations GLOBAL =",
    int(
        matched[
            "global_nested_violation"
        ].sum()
    ),
)


emit()

emit(
    "delta_H1 = chi2_FAST - chi2_GLOBAL"
)

emit(
    "median =",
    d.median(),
)

emit(
    "q90    =",
    d.quantile(
        0.90
    ),
)

emit(
    "q95    =",
    d.quantile(
        0.95
    ),
)

emit(
    "q99    =",
    d.quantile(
        0.99
    ),
)

emit(
    "max    =",
    d.max(),
)

emit()

emit(
    "N(delta_H1 > 0.01) =",
    int(
        (
            d > 0.01
        ).sum()
    ),
)

emit(
    "N(delta_H1 > 0.1)  =",
    int(
        (
            d > 0.1
        ).sum()
    ),
)

emit(
    "N(delta_H1 > 1)    =",
    int(
        (
            d > 1
        ).sum()
    ),
)

emit(
    "N(delta_H1 > 5)    =",
    int(
        (
            d > 5
        ).sum()
    ),
)


emit()

emit(
    "Worst 15 FAST misses:"
)


worst = (
    matched
    .sort_values(
        "delta_H1_fast_minus_global",
        ascending=False,
    )
    .head(
        15
    )
)


show_cols = [
    "catalog_row",
    "truth_case",
    "noise_realization_id",
    "fast_chi2_H0",
    "fast_chi2_H1",
    "global_ref_chi2_H1",
    "fast_T",
    "global_T",
    "delta_H1_fast_minus_global",
]


table_text = worst[
    show_cols
].to_string(
    index=False
)


print(
    table_text
)

lines.append(
    table_text
)


report = (
    study_dir
    / "comparison_report.txt"
)


report.write_text(
    "\n".join(
        lines
    )
    + "\n"
)


print()
print("Saved:")
print(
    " ",
    comparison_path,
)

print(
    " ",
    comparison_csv,
)

print(
    " ",
    summary_path,
)

print(
    " ",
    report,
)
PY


chmod +x \
    "$ANALYZER"


"$PYTHON_BIN" -m py_compile \
    "$ANALYZER"


# ============================================================================
# SLURM analysis wrapper
# ============================================================================

ANALYSIS_SLURM="${PROJECT_DIR}/validation/lrt/run_lrt_validation_analysis.slurm"

cat > "$ANALYSIS_SLURM" <<'SLURM'
#!/usr/bin/env bash

set -euo pipefail

: "${PROJECT_DIR:?}"
: "${STUDY_DIR:?}"


PYTHON_BIN="/home/anibalvarela/.conda/envs/pyLIMA_test/bin/python"


cd "$PROJECT_DIR"


"$PYTHON_BIN" \
    analysis/analyze_lrt_validation_benchmark.py \
    --study-dir "$STUDY_DIR"
SLURM


chmod +x \
    "$ANALYSIS_SLURM"


# ============================================================================
# Submit FAST
# ============================================================================

ARRAY_MAX=$(( N_SELECT - 1 ))


FAST_JOB=$(
    sbatch \
        --parsable \
        --partition="$PARTITION" \
        --array="0-${ARRAY_MAX}%${FAST_MAX_CONCURRENT}" \
        --cpus-per-task="$WORKERS" \
        --mem="$FAST_MEM" \
        --time="$FAST_TIME" \
        --chdir="$PROJECT_DIR" \
        --output="${PROJECT_DIR}/slurm_logs/${STUDY_TAG}_FAST_%A_%a.out" \
        --error="${PROJECT_DIR}/slurm_logs/${STUDY_TAG}_FAST_%A_%a.err" \
        --export=ALL,PROJECT_DIR="${PROJECT_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}",CFG_PATH="${FAST_CONFIG}",RUN_NAME="${FAST_RUN_NAME}",ROWS_FILE="${ROWS_FILE}",METHOD="FAST",STUDY_TAG="${STUDY_TAG}",WORKERS="${WORKERS}" \
        "$SLURM_RUNNER"
)


FAST_JOB="${FAST_JOB%%;*}"


# ============================================================================
# Submit GLOBAL
# ============================================================================

GLOBAL_JOB=$(
    sbatch \
        --parsable \
        --partition="$PARTITION" \
        --array="0-${ARRAY_MAX}%${GLOBAL_MAX_CONCURRENT}" \
        --cpus-per-task="$WORKERS" \
        --mem="$GLOBAL_MEM" \
        --time="$GLOBAL_TIME" \
        --chdir="$PROJECT_DIR" \
        --output="${PROJECT_DIR}/slurm_logs/${STUDY_TAG}_GLOBAL_%A_%a.out" \
        --error="${PROJECT_DIR}/slurm_logs/${STUDY_TAG}_GLOBAL_%A_%a.err" \
        --export=ALL,PROJECT_DIR="${PROJECT_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}",CFG_PATH="${GLOBAL_CONFIG}",RUN_NAME="${GLOBAL_RUN_NAME}",ROWS_FILE="${ROWS_FILE}",METHOD="GLOBAL",STUDY_TAG="${STUDY_TAG}",WORKERS="${WORKERS}" \
        "$SLURM_RUNNER"
)


GLOBAL_JOB="${GLOBAL_JOB%%;*}"


# ============================================================================
# Submit analysis automatically when both arrays finish successfully
# ============================================================================

ANALYSIS_JOB=$(
    sbatch \
        --parsable \
        --partition="$PARTITION" \
        --dependency="afterok:${FAST_JOB}:${GLOBAL_JOB}" \
        --cpus-per-task=1 \
        --mem=6G \
        --time=01:00:00 \
        --chdir="$PROJECT_DIR" \
        --output="${PROJECT_DIR}/slurm_logs/${STUDY_TAG}_ANALYSIS_%j.out" \
        --error="${PROJECT_DIR}/slurm_logs/${STUDY_TAG}_ANALYSIS_%j.err" \
        --export=ALL,PROJECT_DIR="${PROJECT_DIR}",STUDY_DIR="${STUDY_DIR}" \
        "$ANALYSIS_SLURM"
)


ANALYSIS_JOB="${ANALYSIS_JOB%%;*}"


# ============================================================================
# Save job IDs
# ============================================================================

cat >> "$MANIFEST" <<EOF
FAST_JOB='${FAST_JOB}'
GLOBAL_JOB='${GLOBAL_JOB}'
ANALYSIS_JOB='${ANALYSIS_JOB}'
EOF


echo "$FAST_JOB" \
    > "${STUDY_DIR}/FAST_JOB.txt"

echo "$GLOBAL_JOB" \
    > "${STUDY_DIR}/GLOBAL_JOB.txt"

echo "$ANALYSIS_JOB" \
    > "${STUDY_DIR}/ANALYSIS_JOB.txt"


# ============================================================================
# Report
# ============================================================================

echo
echo "======================================================================"
echo "LRT VALIDATION SUBMITTED"
echo "======================================================================"
echo
echo "Study:"
echo "  ${STUDY_DIR}"
echo
echo "Selected physical events:"
echo "  ${N_SELECT}"
echo
echo "Logical datasets per event:"
echo "  2 truth cases x ${N_REALIZATIONS} noise realizations"
echo
echo "FAST job:"
echo "  ${FAST_JOB}"
echo
echo "GLOBAL job:"
echo "  ${GLOBAL_JOB}"
echo
echo "Dependent analysis job:"
echo "  ${ANALYSIS_JOB}"
echo
echo "FAST config:"
echo "  ${FAST_CONFIG}"
echo
echo "GLOBAL config:"
echo "  ${GLOBAL_CONFIG}"
echo
echo "Rows:"
echo "  ${ROWS_FILE}"
echo
echo "Monitor:"
echo
echo "  squeue -j ${FAST_JOB},${GLOBAL_JOB},${ANALYSIS_JOB}"
echo
echo "Accounting:"
echo
echo "  sacct -j ${FAST_JOB},${GLOBAL_JOB},${ANALYSIS_JOB} --format=JobID,State,Elapsed,TotalCPU,MaxRSS,ExitCode"
echo
echo "When analysis has completed:"
echo
echo "  cat ${STUDY_DIR}/comparison_report.txt"
echo
echo "======================================================================"
