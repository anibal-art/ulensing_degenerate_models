#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# ============================================================
# Submit LSSTMONTS production as a SLURM job array.
# Designed for 5 free machines, 40 cores each.
# ============================================================

RUN_DIR="/export/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe"
OUTPUT_ROOT="/export/storage3/rubin/microlensing/romanrubin/hidden_parallax"

RUNNER_SOURCE="${PROJECT_DIR}/run_lsstmonts_catalog_hidden_parallax.py"
CONFIG_SOURCE="${PROJECT_DIR}/configs/config_lsstmonts_baseline_v5p3p5_cluster_che_multifit_LRT.json"
SLURM_SCRIPT="${SCRIPT_DIR}/run_lsstmonts_production_array.slurm"

# Full LSSTMONTS catalog has 966000 rows.
ROW_START_GLOBAL="${ROW_START_GLOBAL:-0}"
ROW_STOP_GLOBAL="${ROW_STOP_GLOBAL:-966000}"

# Chunk size in raw catalog rows. Chunks are non-overlapping [START, STOP).
CHUNK_SIZE="${CHUNK_SIZE:-5000}"

# Workers inside each SLURM job. Keep conservative at first.
# With MAX_CONCURRENT=5 and WORKERS=10, total simultaneous event fits ~= 50.
WORKERS="${WORKERS:-10}"

# Use exactly 5 machines at a time.
MAX_CONCURRENT="${MAX_CONCURRENT:-5}"

# If 1, existing partial output directories for a chunk are moved aside and rerun.
# Keep 0 for production safety.
FORCE_RERUN="${FORCE_RERUN:-0}"

# A common tag shared by all chunks in this production launch.
# Override RUN_TAG to resume/submit a known batch label intentionally.
RUN_TAG="${RUN_TAG:-prod_multifit_$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! -f "${RUNNER_SOURCE}" ]]; then
  echo "ERROR: runner not found: ${RUNNER_SOURCE}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_SOURCE}" ]]; then
  echo "ERROR: config not found: ${CONFIG_SOURCE}" >&2
  exit 1
fi

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "ERROR: slurm script not found: ${SLURM_SCRIPT}" >&2
  exit 1
fi

if (( ROW_STOP_GLOBAL <= ROW_START_GLOBAL )); then
  echo "ERROR: ROW_STOP_GLOBAL must be larger than ROW_START_GLOBAL." >&2
  exit 1
fi

N_ROWS=$(( ROW_STOP_GLOBAL - ROW_START_GLOBAL ))
N_CHUNKS=$(( (N_ROWS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
ARRAY_MAX=$(( N_CHUNKS - 1 ))

mkdir -p "${RUN_DIR}/slurm_logs"

# Freeze config for reproducibility. Jobs read the frozen copy.
FROZEN_DIR="${OUTPUT_ROOT}/production_configs/${RUN_TAG}"
mkdir -p "${FROZEN_DIR}"

CFG_PATH="${FROZEN_DIR}/config.json"
RUNNER_PATH="${RUNNER_SOURCE}"

cp "${CONFIG_SOURCE}" "${CFG_PATH}"
sha256sum "${CFG_PATH}" > "${CFG_PATH}.SHA256"

# Save production manifest.
MANIFEST="${FROZEN_DIR}/manifest.txt"
{
  echo "run_tag=${RUN_TAG}"
  echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "submit_host=$(hostname)"
  echo "run_dir=${RUN_DIR}"
  echo "runner_source=${RUNNER_SOURCE}"
  echo "runner_path=${RUNNER_PATH}"
  echo "config_source=${CONFIG_SOURCE}"
  echo "cfg_path=${CFG_PATH}"
  echo "row_start_global=${ROW_START_GLOBAL}"
  echo "row_stop_global=${ROW_STOP_GLOBAL}"
  echo "chunk_size=${CHUNK_SIZE}"
  echo "n_rows=${N_ROWS}"
  echo "n_chunks=${N_CHUNKS}"
  echo "array_max=${ARRAY_MAX}"
  echo "workers=${WORKERS}"
  echo "max_concurrent=${MAX_CONCURRENT}"
  echo "force_rerun=${FORCE_RERUN}"
  echo "config_sha256=$(awk '{print $1}' "${CFG_PATH}.SHA256")"
} > "${MANIFEST}"

cat <<INFO
============================================================
Submitting LSSTMONTS production array
============================================================
RUN_TAG          = ${RUN_TAG}
RUN_DIR          = ${RUN_DIR}
RUNNER_PATH      = ${RUNNER_PATH}
CFG_PATH         = ${CFG_PATH}
CONFIG SHA256    = $(awk '{print $1}' "${CFG_PATH}.SHA256")
ROW_START_GLOBAL = ${ROW_START_GLOBAL}
ROW_STOP_GLOBAL  = ${ROW_STOP_GLOBAL}
CHUNK_SIZE       = ${CHUNK_SIZE}
N_ROWS           = ${N_ROWS}
N_CHUNKS         = ${N_CHUNKS}
ARRAY            = 0-${ARRAY_MAX}%${MAX_CONCURRENT}
WORKERS/job      = ${WORKERS}
MAX_CONCURRENT   = ${MAX_CONCURRENT}
FORCE_RERUN      = ${FORCE_RERUN}
MANIFEST         = ${MANIFEST}
============================================================
INFO

cd "${RUN_DIR}"

sbatch \
  --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" \
  --export=ALL,CFG_PATH="${CFG_PATH}",RUNNER_PATH="${RUNNER_PATH}",RUN_TAG="${RUN_TAG}",ROW_START_GLOBAL="${ROW_START_GLOBAL}",ROW_STOP_GLOBAL="${ROW_STOP_GLOBAL}",CHUNK_SIZE="${CHUNK_SIZE}",WORKERS="${WORKERS}",FORCE_RERUN="${FORCE_RERUN}" \
  "${SLURM_SCRIPT}"

cat <<NEXT

Submitted.

Useful commands:
  squeue -u \$USER
  tail -f ${RUN_DIR}/slurm_logs/lsstmonts_prod_<ARRAYJOB>_<TASK>.out
  cat ${MANIFEST}

After completion, merge with:
  conda activate pyLIMA_test
  python ${SCRIPT_DIR}/merge_lsstmonts_production.py \\
    --run-tag ${RUN_TAG}

NEXT
