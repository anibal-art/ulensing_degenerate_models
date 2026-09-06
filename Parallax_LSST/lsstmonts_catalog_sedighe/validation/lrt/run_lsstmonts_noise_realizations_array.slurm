#!/bin/bash
#SBATCH --partition=cosmoobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --signal=TERM@90
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=lsst_noise_mc
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -Eeuo pipefail


CFG_PATH="${CFG_PATH:?CFG_PATH is required}"
RUNNER_PATH="${RUNNER_PATH:?RUNNER_PATH is required}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"

ROW_START_GLOBAL="${ROW_START_GLOBAL:?ROW_START_GLOBAL is required}"
ROW_STOP_GLOBAL="${ROW_STOP_GLOBAL:?ROW_STOP_GLOBAL is required}"

LOGICAL_CHUNK_SIZE="${LOGICAL_CHUNK_SIZE:?LOGICAL_CHUNK_SIZE is required}"
N_LOGICAL_UPPER="${N_LOGICAL_UPPER:?N_LOGICAL_UPPER is required}"

WORKERS="${WORKERS:?WORKERS is required}"
FORCE_RERUN="${FORCE_RERUN:-0}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"


# ============================================================================
# Frozen config integrity
# ============================================================================

if [[ ! -f "${CFG_PATH}" ]]; then
    echo "ERROR: config does not exist: ${CFG_PATH}" >&2
    exit 1
fi


if [[ -f "${CFG_PATH}.SHA256" ]]; then

    EXPECTED="$(
        awk \
            '{print $1}' \
            "${CFG_PATH}.SHA256"
    )"

    ACTUAL="$(
        sha256sum \
            "${CFG_PATH}" \
        | awk \
            '{print $1}'
    )"

    if [[ "${EXPECTED}" != "${ACTUAL}" ]]; then
        echo "ERROR: frozen config checksum mismatch." >&2
        exit 1
    fi

fi


# ============================================================================
# Conda
# ============================================================================

set +u

if [[ -f "${HOME}/.bashrc" ]]; then
    source "${HOME}/.bashrc" || true
fi

set -u


if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available in the batch environment." >&2
    exit 1
fi


CONDA_BASE="$(
    conda info --base
)"

# shellcheck disable=SC1091
source \
    "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate pyLIMA_test


# ============================================================================
# Numerical threads
# ============================================================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export PYTHONUNBUFFERED=1

export SIMFIT_RETRIES="${SIMFIT_RETRIES:-1}"
export SIMFIT_MP_CONTEXT="${SIMFIT_MP_CONTEXT:-spawn}"
export SIMFIT_SKIP_MODE="${SIMFIT_SKIP_MODE:-none}"
export SIMFIT_PROGRESS_EVERY="${SIMFIT_PROGRESS_EVERY:-100}"

export SLURM_CPU_BIND=cores

export TMPDIR="${SLURM_TMPDIR:-/tmp}"

export MPLCONFIGDIR="${TMPDIR}/mplconfig_${SLURM_JOB_ID:-manual}_${TASK_ID}"

mkdir -p \
    "${MPLCONFIGDIR}" \
    slurm_logs


# ============================================================================
# Logical task interval
# ============================================================================

LSTART=$(( TASK_ID * LOGICAL_CHUNK_SIZE ))

LSTOP=$(( LSTART + LOGICAL_CHUNK_SIZE ))


if (( LSTOP > N_LOGICAL_UPPER )); then
    LSTOP="${N_LOGICAL_UPPER}"
fi


if (( LSTART >= LSTOP )); then
    echo "Empty upper-bound logical chunk."
    exit 0
fi


SUFFIX="${RUN_TAG}_logical_${LSTART}_${LSTOP}_w${WORKERS}"


# ============================================================================
# Resolve output tree
# ============================================================================

mapfile -t CFG_INFO < <(
python - "${CFG_PATH}" <<'PYCFG'
import json
import sys

cfg = json.load(
    open(sys.argv[1])
)

paths = (
    cfg.get(
        "paths",
        {},
    )
    or {}
)

output = (
    cfg.get(
        "output",
        {},
    )
    or {}
)

output_root = (
    paths.get(
        "output_root"
    )
    or output.get(
        "root_dir"
    )
    or cfg.get(
        "path_storage"
    )
)

run_name = (
    cfg.get(
        "run_name"
    )
    or output.get(
        "run_name"
    )
)

if not output_root or not run_name:
    raise SystemExit(
        "Could not resolve output_root/run_name"
    )

print(output_root)
print(run_name)
PYCFG
)


OUTPUT_ROOT="${CFG_INFO[0]}"
RUN_NAME_BASE="${CFG_INFO[1]}"

RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_NAME_BASE}/${SUFFIX}"

LOCK_BASE="${OUTPUT_ROOT}/slurm_locks/${RUN_TAG}"
DONE_BASE="${OUTPUT_ROOT}/slurm_done/${RUN_TAG}"

LOCK_DIR="${LOCK_BASE}/logical_${LSTART}_${LSTOP}.lock"

DONE_FILE="${DONE_BASE}/logical_${LSTART}_${LSTOP}.DONE"


mkdir -p \
    "${LOCK_BASE}" \
    "${DONE_BASE}"


if [[ -f "${DONE_FILE}" ]]; then
    echo "DONE marker exists: ${DONE_FILE}"
    exit 0
fi


if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "ERROR: lock exists: ${LOCK_DIR}" >&2
    exit 1
fi


cleanup_lock() {
    rmdir \
        "${LOCK_DIR}" \
        2>/dev/null \
        || true
}

trap cleanup_lock EXIT


if [[ -d "${RUN_DIR}" && "${FORCE_RERUN}" != "1" ]]; then

    echo "ERROR: output run directory already exists:" >&2
    echo "  ${RUN_DIR}" >&2

    exit 1

fi


if [[ -d "${RUN_DIR}" && "${FORCE_RERUN}" == "1" ]]; then

    TS="$(
        date -u +%Y%m%dT%H%M%SZ
    )"

    mv \
        "${RUN_DIR}" \
        "${RUN_DIR}.partial_${TS}"

fi


cat <<INFO
========================================================================
Noise-realization SLURM task
========================================================================

HOST               = $(hostname)

JOB_ID             = ${SLURM_JOB_ID:-NA}
ARRAY_TASK         = ${TASK_ID}

physical rows      = [${ROW_START_GLOBAL}, ${ROW_STOP_GLOBAL})
logical tasks      = [${LSTART}, ${LSTOP})

WORKERS            = ${WORKERS}

RUN_DIR            = ${RUN_DIR}
CFG_PATH           = ${CFG_PATH}

========================================================================
INFO


# ============================================================================
# Run
# ============================================================================

srun \
    --cpu-bind=cores \
    -u \
    python \
    -u \
    "${RUNNER_PATH}" \
    --config "${CFG_PATH}" \
    --catalog-row-start "${ROW_START_GLOBAL}" \
    --catalog-row-stop "${ROW_STOP_GLOBAL}" \
    --logical-task-start "${LSTART}" \
    --logical-task-stop "${LSTOP}" \
    --workers "${WORKERS}" \
    --run-name-suffix "${SUFFIX}"


# ============================================================================
# Done marker
# ============================================================================

{
    echo "status=done"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    echo "hostname=$(hostname)"

    echo "slurm_job_id=${SLURM_JOB_ID:-NA}"
    echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-NA}"
    echo "slurm_array_task_id=${TASK_ID}"

    echo "run_tag=${RUN_TAG}"

    echo "row_start=${ROW_START_GLOBAL}"
    echo "row_stop=${ROW_STOP_GLOBAL}"

    echo "logical_start=${LSTART}"
    echo "logical_stop=${LSTOP}"

    echo "workers=${WORKERS}"

    echo "run_dir=${RUN_DIR}"

} > "${DONE_FILE}"


echo "DONE: ${DONE_FILE}"
