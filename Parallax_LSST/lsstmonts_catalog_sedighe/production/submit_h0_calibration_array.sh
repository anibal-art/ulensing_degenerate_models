#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"

PROJECT_DIR="$(
  dirname "${SCRIPT_DIR}"
)"

RUN_DIR="${PROJECT_DIR}"

OUTPUT_ROOT="/export/storage3/rubin/microlensing/romanrubin/hidden_parallax"

RUNNER_SOURCE="${PROJECT_DIR}/production/run_h0_calibration_chunk.py"

CONFIG_SOURCE="${CONFIG_SOURCE:-${PROJECT_DIR}/configs/production/LRT_TWO_FIT_V2.json}"

SLURM_SCRIPT="${PROJECT_DIR}/production/run_h0_calibration_array.slurm"


N_CANDIDATES="${N_CANDIDATES:-40000}"

RANK_START_GLOBAL="${RANK_START_GLOBAL:-0}"
RANK_STOP_GLOBAL="${RANK_STOP_GLOBAL:-${N_CANDIDATES}}"

SHARD_SIZE="${SHARD_SIZE:-1250}"

MAX_CONCURRENT="${MAX_CONCURRENT:-16}"

SEED="${SEED:-707070}"

TARGET_N="${TARGET_N:-10000}"

PARTITION="${PARTITION:-milliways-int}"

CPUS_PER_TASK="${CPUS_PER_TASK:-1}"

MEM_PER_TASK="${MEM_PER_TASK:-4G}"

RUN_TAG="${RUN_TAG:-lrt_two_fit_v2_h0_calibration_$(date -u +%Y%m%dT%H%M%SZ)}"


if [[ ! -f "${RUNNER_SOURCE}" ]]; then
  echo "ERROR: runner not found: ${RUNNER_SOURCE}" >&2
  exit 1
fi

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "ERROR: slurm wrapper not found: ${SLURM_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_SOURCE}" ]]; then
  echo "ERROR: config not found: ${CONFIG_SOURCE}" >&2
  exit 1
fi


EXPECTED_REPO_COMMIT="$(
  git -C "${PROJECT_DIR}" rev-parse HEAD
)"


if ! git -C "${PROJECT_DIR}" diff --quiet --; then
  echo "ERROR: tracked unstaged changes exist." >&2
  exit 1
fi

if ! git -C "${PROJECT_DIR}" diff --cached --quiet --; then
  echo "ERROR: tracked staged changes exist." >&2
  exit 1
fi


if (( N_CANDIDATES <= 0 )); then
  echo "ERROR: N_CANDIDATES must be positive." >&2
  exit 1
fi

if (( RANK_START_GLOBAL < 0 )); then
  echo "ERROR: RANK_START_GLOBAL must be >= 0." >&2
  exit 1
fi

if (( RANK_STOP_GLOBAL > N_CANDIDATES )); then
  echo "ERROR: RANK_STOP_GLOBAL exceeds N_CANDIDATES." >&2
  exit 1
fi

if (( RANK_STOP_GLOBAL <= RANK_START_GLOBAL )); then
  echo "ERROR: invalid rank interval." >&2
  exit 1
fi


N_RANKS=$(( RANK_STOP_GLOBAL - RANK_START_GLOBAL ))

N_CHUNKS=$(
  (
    N_RANKS
    + SHARD_SIZE
    - 1
  )
  / SHARD_SIZE
)

ARRAY_MAX=$(( N_CHUNKS - 1 ))


mkdir -p "${RUN_DIR}/slurm_logs"


# ============================================================
# Frozen config
# ============================================================

FROZEN_DIR="${OUTPUT_ROOT}/production_configs/${RUN_TAG}"

mkdir -p "${FROZEN_DIR}"

CFG_PATH="${FROZEN_DIR}/config.json"

SOURCE_SHA="$(
  sha256sum "${CONFIG_SOURCE}" \
    | awk '{print $1}'
)"


if [[ -f "${CFG_PATH}" ]]; then

  FROZEN_SHA="$(
    sha256sum "${CFG_PATH}" \
      | awk '{print $1}'
  )"

  if [[ "${SOURCE_SHA}" != "${FROZEN_SHA}" ]]; then
    echo "ERROR: RUN_TAG has a different frozen config." >&2
    exit 1
  fi

else

  cp \
    "${CONFIG_SOURCE}" \
    "${CFG_PATH}"

fi


sha256sum \
  "${CFG_PATH}" \
  > "${CFG_PATH}.SHA256"


# ============================================================
# Immutable campaign identity
# ============================================================

CAMPAIGN_SPEC="${FROZEN_DIR}/campaign_spec.txt"

CANDIDATE_SPEC="$(
  mktemp "${FROZEN_DIR}/campaign_spec.XXXXXX"
)"


{
  echo "campaign_type=h0_calibration_parallel"
  echo "repo_commit=${EXPECTED_REPO_COMMIT}"
  echo "config_sha256=$(awk '{print $1}' "${CFG_PATH}.SHA256")"
  echo "seed=${SEED}"
  echo "n_candidates=${N_CANDIDATES}"
  echo "target_n=${TARGET_N}"
  echo "rank_start_global=${RANK_START_GLOBAL}"
  echo "rank_stop_global=${RANK_STOP_GLOBAL}"
  echo "shard_size=${SHARD_SIZE}"
  echo "output_root=${OUTPUT_ROOT}"
} > "${CANDIDATE_SPEC}"


if [[ -f "${CAMPAIGN_SPEC}" ]]; then

  if ! cmp -s \
    "${CAMPAIGN_SPEC}" \
    "${CANDIDATE_SPEC}"
  then
    echo "ERROR: RUN_TAG campaign identity mismatch." >&2

    diff -u \
      "${CAMPAIGN_SPEC}" \
      "${CANDIDATE_SPEC}" \
      >&2 || true

    rm -f "${CANDIDATE_SPEC}"

    exit 1
  fi

  rm -f "${CANDIDATE_SPEC}"

else

  mv \
    "${CANDIDATE_SPEC}" \
    "${CAMPAIGN_SPEC}"

fi


MANIFEST="${FROZEN_DIR}/manifest.txt"

{
  echo "run_tag=${RUN_TAG}"
  echo "campaign_type=h0_calibration_parallel"
  echo "repo_commit=${EXPECTED_REPO_COMMIT}"
  echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "runner_source=${RUNNER_SOURCE}"
  echo "slurm_script=${SLURM_SCRIPT}"
  echo "config_source=${CONFIG_SOURCE}"
  echo "cfg_path=${CFG_PATH}"
  echo "seed=${SEED}"
  echo "n_candidates=${N_CANDIDATES}"
  echo "target_n=${TARGET_N}"
  echo "rank_start_global=${RANK_START_GLOBAL}"
  echo "rank_stop_global=${RANK_STOP_GLOBAL}"
  echo "shard_size=${SHARD_SIZE}"
  echo "n_chunks=${N_CHUNKS}"
  echo "max_concurrent=${MAX_CONCURRENT}"
  echo "partition=${PARTITION}"
  echo "cpus_per_task=${CPUS_PER_TASK}"
  echo "mem_per_task=${MEM_PER_TASK}"
  echo "config_sha256=$(awk '{print $1}' "${CFG_PATH}.SHA256")"
} > "${MANIFEST}"


echo "============================================================"
echo "Submitting parallel H0 calibration"
echo "============================================================"
echo "RUN_TAG           = ${RUN_TAG}"
echo "REPO COMMIT       = ${EXPECTED_REPO_COMMIT}"
echo "SEED              = ${SEED}"
echo "N_CANDIDATES      = ${N_CANDIDATES}"
echo "TARGET_N          = ${TARGET_N}"
echo "RANK RANGE        = [${RANK_START_GLOBAL}, ${RANK_STOP_GLOBAL})"
echo "SHARD_SIZE        = ${SHARD_SIZE}"
echo "N_CHUNKS          = ${N_CHUNKS}"
echo "ARRAY             = 0-${ARRAY_MAX}%${MAX_CONCURRENT}"
echo "PARTITION         = ${PARTITION}"
echo "CPUS/task         = ${CPUS_PER_TASK}"
echo "MEM/task          = ${MEM_PER_TASK}"
echo "CFG_PATH          = ${CFG_PATH}"
echo "============================================================"


cd "${RUN_DIR}"


sbatch \
  --partition="${PARTITION}" \
  --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM_PER_TASK}" \
  --export=ALL,CFG_PATH="${CFG_PATH}",RUNNER_PATH="${RUNNER_SOURCE}",RUN_TAG="${RUN_TAG}",N_CANDIDATES="${N_CANDIDATES}",RANK_START_GLOBAL="${RANK_START_GLOBAL}",RANK_STOP_GLOBAL="${RANK_STOP_GLOBAL}",SHARD_SIZE="${SHARD_SIZE}",SEED="${SEED}",EXPECTED_REPO_COMMIT="${EXPECTED_REPO_COMMIT}" \
  "${SLURM_SCRIPT}"


echo
echo "Submitted."
echo
echo "Manifest:"
echo "  ${MANIFEST}"
