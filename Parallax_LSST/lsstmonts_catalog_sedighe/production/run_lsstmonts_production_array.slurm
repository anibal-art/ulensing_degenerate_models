#!/bin/bash
#SBATCH --partition=cosmoobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
# #SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@90
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=lsstmonts_prod
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

# ============================================================
# SLURM job-array chunk runner
# One array task = one non-overlapping catalog row interval [START, STOP)
# ============================================================

echo "============================================================"
echo "SLURM job info"
echo "============================================================"
echo "HOST       : $(hostname)"
echo "CWD        : $(pwd)"
echo "JOB_ID     : ${SLURM_JOB_ID:-NA}"
echo "ARRAY_JOB  : ${SLURM_ARRAY_JOB_ID:-NA}"
echo "ARRAY_TASK : ${SLURM_ARRAY_TASK_ID:-NA}"
echo "CFG_PATH   : ${CFG_PATH:-UNSET}"
echo "RUNNER_PATH: ${RUNNER_PATH:-UNSET}"
echo "RUN_TAG    : ${RUN_TAG:-UNSET}"
echo "============================================================"

if [[ -z "${CFG_PATH:-}" ]]; then
  echo "ERROR: CFG_PATH is not defined." >&2
  exit 1
fi

if [[ -z "${RUNNER_PATH:-}" ]]; then
  echo "ERROR: RUNNER_PATH is not defined." >&2
  exit 1
fi

if [[ -z "${RUN_TAG:-}" ]]; then
  echo "ERROR: RUN_TAG is not defined." >&2
  exit 1
fi

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "ERROR: config file does not exist: ${CFG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${RUNNER_PATH}" ]]; then
  echo "ERROR: runner file does not exist: ${RUNNER_PATH}" >&2
  exit 1
fi

# ------------------------------------------------------------
# Verify frozen config checksum
# ------------------------------------------------------------

if [[ -f "${CFG_PATH}.SHA256" ]]; then
  EXPECTED="$(awk '{print $1}' "${CFG_PATH}.SHA256")"
  ACTUAL="$(sha256sum "${CFG_PATH}" | awk '{print $1}')"

  echo "EXPECTED SHA256: ${EXPECTED}"
  echo "ACTUAL SHA256  : ${ACTUAL}"

  if [[ "${EXPECTED}" != "${ACTUAL}" ]]; then
    echo "ERROR: checksum mismatch for ${CFG_PATH}" >&2
    exit 1
  fi
else
  echo "WARNING: checksum file not found: ${CFG_PATH}.SHA256"
fi

# ------------------------------------------------------------
# Conda environment
# ------------------------------------------------------------
# The cluster bashrc can reference unset variables. Since this
# script uses `set -u`, temporarily disable nounset while
# initializing conda.

set +u

if [[ -f "${HOME}/.bashrc" ]]; then
  source "${HOME}/.bashrc"
fi

# Robust conda activation for batch shells.
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  # Fallbacks in case conda is not in PATH before shell init.
  for CONDA_SH in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/anaconda3/etc/profile.d/conda.sh" \
    "/opt/miniconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "${CONDA_SH}" ]]; then
      # shellcheck disable=SC1090
      source "${CONDA_SH}"
      break
    fi
  done
fi

conda activate pyLIMA_test

set -u

# ------------------------------------------------------------
# Numerical-library thread control
# ------------------------------------------------------------

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

# Extra defensive multiprocessing settings.
export SIMFIT_RETRIES="${SIMFIT_RETRIES:-1}"
export SIMFIT_MP_CONTEXT="${SIMFIT_MP_CONTEXT:-spawn}"
export SIMFIT_SKIP_MODE="${SIMFIT_SKIP_MODE:-none}"
export SIMFIT_PROGRESS_EVERY="${SIMFIT_PROGRESS_EVERY:-100}"

# ------------------------------------------------------------
# CPU binding and local scratch
# ------------------------------------------------------------

export SLURM_CPU_BIND=cores
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export MPLCONFIGDIR="${TMPDIR}/mplconfig_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p slurm_logs

# ------------------------------------------------------------
# Chunk parameters
# ------------------------------------------------------------

ROW_START_GLOBAL="${ROW_START_GLOBAL:?ROW_START_GLOBAL is required}"
ROW_STOP_GLOBAL="${ROW_STOP_GLOBAL:?ROW_STOP_GLOBAL is required}"
CHUNK_SIZE="${CHUNK_SIZE:?CHUNK_SIZE is required}"
WORKERS="${WORKERS:?WORKERS is required}"
FORCE_RERUN="${FORCE_RERUN:-0}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

START=$(( ROW_START_GLOBAL + TASK_ID * CHUNK_SIZE ))
STOP=$(( START + CHUNK_SIZE ))

if (( STOP > ROW_STOP_GLOBAL )); then
  STOP=${ROW_STOP_GLOBAL}
fi

if (( START >= STOP )); then
  echo "Nothing to do for TASK_ID=${TASK_ID}: START=${START}, STOP=${STOP}"
  exit 0
fi

SUFFIX="${RUN_TAG}_rows_${START}_${STOP}_w${WORKERS}"

# ------------------------------------------------------------
# Read output root and run-name from config
# ------------------------------------------------------------

mapfile -t CFG_INFO < <(python - "${CFG_PATH}" <<'PY'
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
cfg = json.loads(cfg_path.read_text())

paths = cfg.get("paths", {})
output_root = paths.get("output_root")
run_name = cfg.get("run_name")

if not output_root:
    raise SystemExit("Missing paths.output_root in config")
if not run_name:
    raise SystemExit("Missing run_name in config")

print(output_root)
print(run_name)
PY
)

OUTPUT_ROOT="${CFG_INFO[0]}"
RUN_NAME_BASE="${CFG_INFO[1]}"

RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_NAME_BASE}/${SUFFIX}"
LOCK_BASE="${OUTPUT_ROOT}/slurm_locks/${RUN_TAG}"
DONE_BASE="${OUTPUT_ROOT}/slurm_done/${RUN_TAG}"
LOCK_DIR="${LOCK_BASE}/rows_${START}_${STOP}.lock"
DONE_FILE="${DONE_BASE}/rows_${START}_${STOP}.DONE"

mkdir -p "${LOCK_BASE}" "${DONE_BASE}"

# ------------------------------------------------------------
# Collision protection
# ------------------------------------------------------------

if [[ -f "${DONE_FILE}" ]]; then
  echo "DONE marker exists. Skipping chunk: ${DONE_FILE}"
  exit 0
fi

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "ERROR: lock already exists: ${LOCK_DIR}" >&2
  echo "Another job may be running this same chunk. If stale, inspect and remove manually." >&2
  exit 1
fi

cleanup_lock() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup_lock EXIT

if [[ -d "${RUN_DIR}" && "${FORCE_RERUN}" != "1" ]]; then
  echo "ERROR: run directory already exists but DONE marker is absent:" >&2
  echo "  ${RUN_DIR}" >&2
  echo "This prevents accidental overwrite/append of a partial chunk." >&2
  echo "Inspect it. To rerun intentionally, set FORCE_RERUN=1 in submit script." >&2
  exit 1
fi

if [[ -d "${RUN_DIR}" && "${FORCE_RERUN}" == "1" ]]; then
  TS="$(date -u +%Y%m%dT%H%M%SZ)"
  BACKUP="${RUN_DIR}.partial_${TS}"
  echo "FORCE_RERUN=1: moving existing run dir to: ${BACKUP}"
  mv "${RUN_DIR}" "${BACKUP}"
fi

# ------------------------------------------------------------
# Audit
# ------------------------------------------------------------

echo "============================================================"
echo "Resource audit"
echo "============================================================"
lscpu | egrep 'Model name|Socket|Core|Thread|CPU\(s\)|MHz' || true
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"
echo "SLURM_CPU_BIND=${SLURM_CPU_BIND:-NA}"
echo "TMPDIR=${TMPDIR}"
echo "MPLCONFIGDIR=${MPLCONFIGDIR}"
echo "SIMFIT_RETRIES=${SIMFIT_RETRIES}"
echo "SIMFIT_MP_CONTEXT=${SIMFIT_MP_CONTEXT}"
echo "SIMFIT_SKIP_MODE=${SIMFIT_SKIP_MODE}"
echo "SIMFIT_PROGRESS_EVERY=${SIMFIT_PROGRESS_EVERY}"
echo "ROW_START_GLOBAL=${ROW_START_GLOBAL}"
echo "ROW_STOP_GLOBAL=${ROW_STOP_GLOBAL}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "TASK_ID=${TASK_ID}"
echo "START=${START}"
echo "STOP=${STOP}"
echo "WORKERS=${WORKERS}"
echo "SUFFIX=${SUFFIX}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RUN_NAME_BASE=${RUN_NAME_BASE}"
echo "RUN_DIR=${RUN_DIR}"
echo "DONE_FILE=${DONE_FILE}"
echo "============================================================"

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

srun --cpu-bind=cores -u python -u "${RUNNER_PATH}" \
  --config "${CFG_PATH}" \
  --catalog-row-start "${START}" \
  --catalog-row-stop "${STOP}" \
  --workers "${WORKERS}" \
  --run-name-suffix "${SUFFIX}"

# ------------------------------------------------------------
# Mark done only after successful completion
# ------------------------------------------------------------

{
  echo "status=done"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-NA}"
  echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-NA}"
  echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-NA}"
  echo "run_tag=${RUN_TAG}"
  echo "start=${START}"
  echo "stop=${STOP}"
  echo "workers=${WORKERS}"
  echo "run_dir=${RUN_DIR}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${DONE_FILE}"

echo "DONE: ${DONE_FILE}"
