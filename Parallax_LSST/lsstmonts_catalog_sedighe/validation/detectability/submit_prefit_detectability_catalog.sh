#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-pilot}"

if [[ "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
    echo "Usage:"
    echo "  $0 pilot"
    echo "  $0 full"
    exit 2
fi

PROJECT_DIR="/export/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe"

OUTPUT_ROOT="/export/storage3/rubin/microlensing/romanrubin/hidden_parallax"

BASE_CONFIG="${PROJECT_DIR}/configs/validation/detectability/config_lsstmonts_prefit_detectability_audit.json"

SLURM_SCRIPT="${PROJECT_DIR}/validation/detectability/run_prefit_detectability_chunk.slurm"

DATA_FILE="/export/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models/Parallax_LSST/data_sedighe/LSSTMONTS.dat"

PYTHON_BIN="/home/anibalvarela/.conda/envs/pyLIMA_test/bin/python"

PARTITION="${PARTITION:-cosmoobs}"
WORKERS="${WORKERS:-4}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

if [[ "${MODE}" == "pilot" ]]; then
    REQUESTED_ROWS="${PILOT_ROWS:-5000}"
    CHUNK_SIZE="${CHUNK_SIZE:-500}"
    TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
else
    REQUESTED_ROWS="all"
    CHUNK_SIZE="${CHUNK_SIZE:-5000}"
    TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
fi

for f in \
    "${BASE_CONFIG}" \
    "${SLURM_SCRIPT}" \
    "${DATA_FILE}"
do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: missing file:"
        echo "  ${f}"
        exit 1
    fi
done

# Exact number of non-empty catalog rows.
N_TOTAL=$(
"${PYTHON_BIN}" - "${DATA_FILE}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])

n = 0
with path.open("r", errors="replace") as f:
    for line in f:
        if line.strip():
            n += 1

print(n)
PY
)

if [[ "${MODE}" == "pilot" ]]; then
    if (( REQUESTED_ROWS < N_TOTAL )); then
        N_ROWS="${REQUESTED_ROWS}"
    else
        N_ROWS="${N_TOTAL}"
    fi
else
    N_ROWS="${N_TOTAL}"
fi

N_CHUNKS=$(( (N_ROWS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
ARRAY_MAX=$(( N_CHUNKS - 1 ))

STAMP=$(date -u +%Y%m%dT%H%M%SZ)

RUN_TAG="prefitDetectability_${MODE}_${STAMP}"

RUN_NAME="LSSTMONTS_prefit_detectability_${MODE}_${STAMP}"

STUDY_DIR="${OUTPUT_ROOT}/detectability_catalogs/${RUN_TAG}"

FROZEN_DIR="${OUTPUT_ROOT}/production_configs/${RUN_TAG}"

mkdir -p \
    "${STUDY_DIR}" \
    "${FROZEN_DIR}" \
    "${PROJECT_DIR}/slurm_logs"

CFG_PATH="${FROZEN_DIR}/config.json"

# ------------------------------------------------------------------
# Freeze config and explicitly enforce AUDIT ONLY.
# ------------------------------------------------------------------

"${PYTHON_BIN}" - \
    "${BASE_CONFIG}" \
    "${CFG_PATH}" \
    "${RUN_NAME}" \
<<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
run_name = sys.argv[3]

cfg = json.loads(src.read_text())

cfg["run_name"] = run_name

noise = cfg.setdefault(
    "noise_realizations",
    {}
)

noise["enabled"] = False
noise["n_realizations"] = 1

simulation = cfg.setdefault(
    "simulation",
    {}
)

simulation["apply_detection_criteria"] = False
simulation["apply_photometric_filter"] = True

selection = cfg.setdefault(
    "selection",
    {}
)

det = selection.setdefault(
    "prefit_detectability",
    {}
)

det["enabled"] = True
det["audit_only"] = True

# Selection independent of the parallax signal tested later.
det["reference_model"] = "no_parallax"

# Explicit frozen criterion.
det["bands"] = "rubin"
det["peak_window_tE"] = 1.0

det["min_total_points"] = 10
det["min_bands"] = 3

det["min_peak_points"] = 5
det["min_left_peak_points"] = 1
det["min_right_peak_points"] = 1

det["nsigma"] = 3.0
det["min_nsigma_points"] = 6

det["min_delta_chi2_per_point"] = 2.0

# Record nearest point, but don't impose another cut yet.
det["max_nearest_peak_distance_tE"] = None

dst.write_text(
    json.dumps(cfg, indent=2) + "\n"
)

print("Frozen config:", dst)
print(json.dumps(det, indent=2))
PY

sha256sum "${CFG_PATH}" > "${CFG_PATH}.SHA256"

# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------

cat > "${STUDY_DIR}/manifest.env" <<EOF
MODE='${MODE}'
RUN_TAG='${RUN_TAG}'
RUN_NAME='${RUN_NAME}'
CFG_PATH='${CFG_PATH}'
PROJECT_DIR='${PROJECT_DIR}'
OUTPUT_ROOT='${OUTPUT_ROOT}'
DATA_FILE='${DATA_FILE}'
N_TOTAL='${N_TOTAL}'
N_ROWS='${N_ROWS}'
CHUNK_SIZE='${CHUNK_SIZE}'
N_CHUNKS='${N_CHUNKS}'
WORKERS='${WORKERS}'
EOF

echo "${RUN_TAG}" \
    > "${OUTPUT_ROOT}/detectability_catalogs/LATEST"

echo
echo "======================================================================"
echo "PREFIT DETECTABILITY CATALOG"
echo "======================================================================"
echo "mode             = ${MODE}"
echo "catalog total    = ${N_TOTAL}"
echo "rows to audit    = ${N_ROWS}"
echo "chunk size       = ${CHUNK_SIZE}"
echo "n chunks         = ${N_CHUNKS}"
echo "workers/task     = ${WORKERS}"
echo "max concurrent   = ${MAX_CONCURRENT}"
echo "partition        = ${PARTITION}"
echo "run name         = ${RUN_NAME}"
echo "run tag          = ${RUN_TAG}"
echo "study dir        = ${STUDY_DIR}"
echo "======================================================================"
echo

JOB_ID=$(
    sbatch \
        --parsable \
        --partition="${PARTITION}" \
        --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" \
        --cpus-per-task="${WORKERS}" \
        --mem=24G \
        --time="${TIME_LIMIT}" \
        --chdir="${PROJECT_DIR}" \
        --output="${PROJECT_DIR}/slurm_logs/detectability_${RUN_TAG}_%A_%a.out" \
        --error="${PROJECT_DIR}/slurm_logs/detectability_${RUN_TAG}_%A_%a.err" \
        --export=ALL,PROJECT_DIR="${PROJECT_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}",CFG_PATH="${CFG_PATH}",RUN_NAME="${RUN_NAME}",RUN_TAG="${RUN_TAG}",N_ROWS="${N_ROWS}",CHUNK_SIZE="${CHUNK_SIZE}",WORKERS="${WORKERS}" \
        "${SLURM_SCRIPT}"
)

JOB_ID="${JOB_ID%%;*}"

echo "${JOB_ID}" > "${STUDY_DIR}/job_id.txt"

echo "Submitted batch job ${JOB_ID}"
echo
echo "Monitor:"
echo "  squeue -j ${JOB_ID}"
echo
echo "Accounting:"
echo "  sacct -j ${JOB_ID} --format=JobID,State,Elapsed,TotalCPU,MaxRSS,ExitCode"
