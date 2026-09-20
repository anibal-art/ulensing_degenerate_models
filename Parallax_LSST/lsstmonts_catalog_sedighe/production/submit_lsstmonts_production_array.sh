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

# Frozen final two-fit production entry points.
RUNNER_SOURCE="${PROJECT_DIR}/production/run_lrt_two_fit_chunk.py"
CONFIG_SOURCE="${CONFIG_SOURCE:-${PROJECT_DIR}/configs/production/LRT_TWO_FIT_V2.json}"
SLURM_SCRIPT="${SCRIPT_DIR}/run_lsstmonts_production_array.slurm"


# ============================================================
# CHE production environment required by LRT_TWO_FIT_V2.json
# ============================================================

MICROLENSING_ROOT="${MICROLENSING_ROOT:-/home/anibalvarela/microlensing}"

ULENSING_DEGENERATE_MODELS_ROOT="${ULENSING_DEGENERATE_MODELS_ROOT:-/export/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models}"
PARALLAX_LSST_BASE="${PARALLAX_LSST_BASE:-${ULENSING_DEGENERATE_MODELS_ROOT}/Parallax_LSST}"
ROMAN_RUBIN_DIR="${ROMAN_RUBIN_DIR:-${MICROLENSING_ROOT}/simulation_Rubin/roman_rubin}"
RUBIN_SIM_DATA_DIR="${RUBIN_SIM_DATA_DIR:-/share/storage3/rubin/microlensing/romanrubin/rubin_sim_data}"

# Full LSSTMONTS catalog has 966000 rows.
ROW_START_GLOBAL="${ROW_START_GLOBAL:-0}"
ROW_STOP_GLOBAL="${ROW_STOP_GLOBAL:-966000}"

# Chunk size in raw catalog rows. Chunks are non-overlapping [START, STOP).
CHUNK_SIZE="${CHUNK_SIZE:-5000}"

# Use exactly 5 machines at a time.
MAX_CONCURRENT="${MAX_CONCURRENT:-5}"


# Resources requested per SLURM array task.
# Command-line sbatch options below override the fixed #SBATCH defaults
# in run_lsstmonts_production_array.slurm.
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"

# Leave empty to keep the memory value defined in the .slurm file.
# Examples: 4G, 20G, 80G.
MEM_PER_TASK="${MEM_PER_TASK:-}"

# Optional SLURM dependency, e.g. afterok:123456.
DEPENDENCY="${DEPENDENCY:-}"

# A common tag shared by all chunks in this production launch.
# Override RUN_TAG to resume/submit a known batch label intentionally.
RUN_TAG="${RUN_TAG:-lrt_two_fit_v2_$(date -u +%Y%m%dT%H%M%SZ)}"

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

# ============================================================
# Frozen repository revision
# ============================================================

EXPECTED_REPO_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"

if ! git -C "${PROJECT_DIR}" diff --quiet --; then
  echo "ERROR: tracked unstaged changes exist in the production checkout." >&2
  echo "Commit or revert them before submitting production." >&2
  exit 1
fi

if ! git -C "${PROJECT_DIR}" diff --cached --quiet --; then
  echo "ERROR: tracked staged changes exist in the production checkout." >&2
  echo "Commit or unstage them before submitting production." >&2
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

SOURCE_CONFIG_SHA256="$(sha256sum "${CONFIG_SOURCE}" | awk '{print $1}')"

if [[ -f "${CFG_PATH}" ]]; then
  FROZEN_CONFIG_SHA256="$(sha256sum "${CFG_PATH}" | awk '{print $1}')"

  echo "Existing frozen config found for RUN_TAG=${RUN_TAG}"
  echo "SOURCE SHA256 = ${SOURCE_CONFIG_SHA256}"
  echo "FROZEN SHA256 = ${FROZEN_CONFIG_SHA256}"

  if [[ "${SOURCE_CONFIG_SHA256}" != "${FROZEN_CONFIG_SHA256}" ]]; then
    echo "ERROR: RUN_TAG already exists with a different frozen config." >&2
    echo "Refusing to change configuration while resuming a campaign." >&2
    exit 1
  fi

  echo "Frozen config matches source; reusing it."
else
  cp "${CONFIG_SOURCE}" "${CFG_PATH}"
  sha256sum "${CFG_PATH}" > "${CFG_PATH}.SHA256"
  echo "Created frozen production config: ${CFG_PATH}"
fi

if [[ ! -f "${CFG_PATH}.SHA256" ]]; then
  sha256sum "${CFG_PATH}" > "${CFG_PATH}.SHA256"
fi


# ============================================================
# Immutable campaign identity
# ============================================================
#
# A reused RUN_TAG must refer to exactly the same scientific
# campaign geometry and repository revision.
# ============================================================

CAMPAIGN_SPEC="${FROZEN_DIR}/campaign_spec.txt"
CAMPAIGN_SPEC_CANDIDATE="$(mktemp "${FROZEN_DIR}/campaign_spec.XXXXXX")"

{
  echo "repo_commit=${EXPECTED_REPO_COMMIT}"
  echo "config_sha256=$(awk '{print $1}' "${CFG_PATH}.SHA256")"
  echo "row_start_global=${ROW_START_GLOBAL}"
  echo "row_stop_global=${ROW_STOP_GLOBAL}"
  echo "chunk_size=${CHUNK_SIZE}"
  echo "microlensing_root=${MICROLENSING_ROOT}"
  echo "roman_rubin_dir=${ROMAN_RUBIN_DIR}"
  echo "ulensing_degenerate_models_root=${ULENSING_DEGENERATE_MODELS_ROOT}"
  echo "parallax_lsst_base=${PARALLAX_LSST_BASE}"
  echo "rubin_sim_data_dir=${RUBIN_SIM_DATA_DIR}"
  echo "output_root=${OUTPUT_ROOT}"
} > "${CAMPAIGN_SPEC_CANDIDATE}"

if [[ -f "${CAMPAIGN_SPEC}" ]]; then
  if ! cmp -s "${CAMPAIGN_SPEC}" "${CAMPAIGN_SPEC_CANDIDATE}"; then
    echo "ERROR: RUN_TAG already exists with a different campaign identity." >&2
    echo "Existing vs requested campaign:" >&2
    diff -u "${CAMPAIGN_SPEC}" "${CAMPAIGN_SPEC_CANDIDATE}" >&2 || true
    rm -f "${CAMPAIGN_SPEC_CANDIDATE}"
    exit 1
  fi

  rm -f "${CAMPAIGN_SPEC_CANDIDATE}"
  echo "Campaign identity matches existing RUN_TAG."
else
  mv "${CAMPAIGN_SPEC_CANDIDATE}" "${CAMPAIGN_SPEC}"
  echo "Created campaign identity: ${CAMPAIGN_SPEC}"
fi

# Save production manifest.
MANIFEST="${FROZEN_DIR}/manifest.txt"
{
  echo "run_tag=${RUN_TAG}"
  echo "repo_commit=${EXPECTED_REPO_COMMIT}"
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
  echo "max_concurrent=${MAX_CONCURRENT}"
  echo "cpus_per_task=${CPUS_PER_TASK}"
  echo "mem_per_task=${MEM_PER_TASK:-slurm_default}"
  echo "dependency=${DEPENDENCY:-none}"
  echo "config_sha256=$(awk '{print $1}' "${CFG_PATH}.SHA256")"
} > "${MANIFEST}"

SUBMISSION_LOG="${FROZEN_DIR}/submissions.log"

{
  echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "submit_host=$(hostname)"
  echo "run_tag=${RUN_TAG}"
  echo "repo_commit=${EXPECTED_REPO_COMMIT}"
  echo "row_start_global=${ROW_START_GLOBAL}"
  echo "row_stop_global=${ROW_STOP_GLOBAL}"
  echo "chunk_size=${CHUNK_SIZE}"
  echo "array=0-${ARRAY_MAX}%${MAX_CONCURRENT}"
  echo "config_sha256=$(awk '{print $1}' "${CFG_PATH}.SHA256")"
  echo "---"
} >> "${SUBMISSION_LOG}"

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
MAX_CONCURRENT   = ${MAX_CONCURRENT}
CPUS/task        = ${CPUS_PER_TASK}
MEM/task         = ${MEM_PER_TASK:-SLURM default}
DEPENDENCY       = ${DEPENDENCY:-none}
MANIFEST         = ${MANIFEST}
============================================================
INFO

cd "${RUN_DIR}"

SBATCH_ARGS=(
  --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}"
  --cpus-per-task="${CPUS_PER_TASK}"
)

if [[ -n "${MEM_PER_TASK}" ]]; then
  SBATCH_ARGS+=(--mem="${MEM_PER_TASK}")
fi

if [[ -n "${DEPENDENCY}" ]]; then
  SBATCH_ARGS+=(--dependency="${DEPENDENCY}")
fi

SBATCH_ARGS+=(
  --export=ALL,CFG_PATH="${CFG_PATH}",RUNNER_PATH="${RUNNER_PATH}",RUN_TAG="${RUN_TAG}",ROW_START_GLOBAL="${ROW_START_GLOBAL}",ROW_STOP_GLOBAL="${ROW_STOP_GLOBAL}",CHUNK_SIZE="${CHUNK_SIZE}",EXPECTED_REPO_COMMIT="${EXPECTED_REPO_COMMIT}",MICROLENSING_ROOT="${MICROLENSING_ROOT}",ULENSING_DEGENERATE_MODELS_ROOT="${ULENSING_DEGENERATE_MODELS_ROOT}",PARALLAX_LSST_BASE="${PARALLAX_LSST_BASE}",ROMAN_RUBIN_DIR="${ROMAN_RUBIN_DIR}",RUBIN_SIM_DATA_DIR="${RUBIN_SIM_DATA_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}"
)

sbatch "${SBATCH_ARGS[@]}" "${SLURM_SCRIPT}"

cat <<NEXT

Submitted.

Useful commands:
  squeue -u \$USER
  tail -f ${RUN_DIR}/slurm_logs/lsstmonts_prod_<ARRAYJOB>_<TASK>.out
  cat ${MANIFEST}

Per-chunk outputs are written in the resumable two-fit format.
Do not use the historical production merger for this campaign.

NEXT
