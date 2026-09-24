#!/bin/bash
#SBATCH --partition=milliways-int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@90
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=h0_cal
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail


if [[ -z "${CFG_PATH:-}" ]]; then
  echo "ERROR: CFG_PATH not defined." >&2
  exit 1
fi

if [[ -z "${RUNNER_PATH:-}" ]]; then
  echo "ERROR: RUNNER_PATH not defined." >&2
  exit 1
fi

if [[ -z "${RUN_TAG:-}" ]]; then
  echo "ERROR: RUN_TAG not defined." >&2
  exit 1
fi

if [[ -z "${EXPECTED_REPO_COMMIT:-}" ]]; then
  echo "ERROR: EXPECTED_REPO_COMMIT not defined." >&2
  exit 1
fi


JOB_PROJECT_DIR="$(
  cd "$(dirname "${RUNNER_PATH}")/.."
  pwd
)"


# ============================================================
# Resolve repository HEAD without requiring git on compute node
# ============================================================

ACTUAL_REPO_COMMIT="$(
python - "${JOB_PROJECT_DIR}" <<'PY'
import sys
from pathlib import Path

start = Path(sys.argv[1]).resolve()

repo_root = None
git_marker = None

for candidate in (start, *start.parents):
    marker = candidate / ".git"

    if marker.exists():
        repo_root = candidate
        git_marker = marker
        break

if git_marker is None:
    raise SystemExit(
        "ERROR: repository .git metadata not found"
    )

if git_marker.is_dir():
    git_dir = git_marker
else:
    text = git_marker.read_text().strip()

    if not text.startswith("gitdir:"):
        raise SystemExit(
            "ERROR: unsupported .git file"
        )

    git_dir = Path(
        text[len("gitdir:"):].strip()
    )

    if not git_dir.is_absolute():
        git_dir = (
            repo_root
            / git_dir
        ).resolve()

head = (
    git_dir
    / "HEAD"
).read_text().strip()

if head.startswith("ref:"):
    ref_name = head[4:].strip()
    ref_path = (
        git_dir
        / ref_name
    )

    if ref_path.is_file():
        commit = (
            ref_path
            .read_text()
            .strip()
        )
    else:
        packed = (
            git_dir
            / "packed-refs"
        )

        commit = None

        for line in (
            packed
            .read_text()
            .splitlines()
        ):
            line = line.strip()

            if (
                not line
                or line.startswith("#")
                or line.startswith("^")
            ):
                continue

            sha, name = line.split(
                " ",
                1,
            )

            if name == ref_name:
                commit = sha
                break

        if commit is None:
            raise SystemExit(
                "ERROR: git ref not found"
            )
else:
    commit = head

print(commit)
PY
)"


echo "EXPECTED REPO COMMIT: ${EXPECTED_REPO_COMMIT}"
echo "ACTUAL REPO COMMIT  : ${ACTUAL_REPO_COMMIT}"

if [[ "${ACTUAL_REPO_COMMIT}" != "${EXPECTED_REPO_COMMIT}" ]]; then
  echo "ERROR: repository commit changed after submission." >&2
  exit 1
fi


# ============================================================
# Verify frozen config
# ============================================================

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "ERROR: config does not exist: ${CFG_PATH}" >&2
  exit 1
fi

if [[ -f "${CFG_PATH}.SHA256" ]]; then
  EXPECTED="$(
    awk '{print $1}' "${CFG_PATH}.SHA256"
  )"

  ACTUAL="$(
    sha256sum "${CFG_PATH}" \
      | awk '{print $1}'
  )"

  if [[ "${EXPECTED}" != "${ACTUAL}" ]]; then
    echo "ERROR: frozen config checksum mismatch." >&2
    exit 1
  fi
fi


# ============================================================
# Environment
# ============================================================

set +u

if [[ -f "${HOME}/.bashrc" ]]; then
  source "${HOME}/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  for CONDA_SH in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/anaconda3/etc/profile.d/conda.sh" \
    "/opt/miniconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "${CONDA_SH}" ]]; then
      source "${CONDA_SH}"
      break
    fi
  done
fi

conda activate pyLIMA_test

set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export SLURM_CPU_BIND=cores

export TMPDIR="${SLURM_TMPDIR:-/tmp}"

export MPLCONFIGDIR="${TMPDIR}/mplconfig_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p "${MPLCONFIGDIR}"
mkdir -p slurm_logs


# ============================================================
# Candidate-rank shard
# ============================================================

RANK_START_GLOBAL="${RANK_START_GLOBAL:?}"
RANK_STOP_GLOBAL="${RANK_STOP_GLOBAL:?}"
SHARD_SIZE="${SHARD_SIZE:?}"
N_CANDIDATES="${N_CANDIDATES:?}"
SEED="${SEED:?}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

START=$(( RANK_START_GLOBAL + TASK_ID * SHARD_SIZE ))
STOP=$(( START + SHARD_SIZE ))

if (( STOP > RANK_STOP_GLOBAL )); then
  STOP=${RANK_STOP_GLOBAL}
fi

if (( START >= STOP )); then
  echo "Nothing to do."
  exit 0
fi

CHUNK_NAME="ranks_${START}_${STOP}"


# ============================================================
# Output geometry
# ============================================================

mapfile -t CFG_INFO < <(
python - "${CFG_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path

cfg = json.loads(
    Path(sys.argv[1]).read_text()
)

paths = cfg.get(
    "paths",
    {},
)

output_root = os.path.expandvars(
    paths.get(
        "output_root",
        "",
    )
)

run_name = cfg.get(
    "run_name"
)

if not output_root:
    raise SystemExit(
        "Missing paths.output_root"
    )

if not run_name:
    raise SystemExit(
        "Missing run_name"
    )

print(output_root)
print(run_name)
PY
)

OUTPUT_ROOT="${CFG_INFO[0]}"
RUN_NAME_BASE="${CFG_INFO[1]}"

RUN_BASE="${OUTPUT_ROOT}/runs/${RUN_NAME_BASE}_H0_CALIBRATION/${RUN_TAG}"
RUN_DIR="${RUN_BASE}/${CHUNK_NAME}"

LOCK_BASE="${OUTPUT_ROOT}/slurm_locks/${RUN_TAG}"
DONE_BASE="${OUTPUT_ROOT}/slurm_done/${RUN_TAG}"

LOCK_DIR="${LOCK_BASE}/${CHUNK_NAME}.lock"
DONE_FILE="${DONE_BASE}/${CHUNK_NAME}.DONE"

mkdir -p \
  "${RUN_BASE}" \
  "${LOCK_BASE}" \
  "${DONE_BASE}"


# ============================================================
# Collision protection
# ============================================================

if [[ -f "${DONE_FILE}" ]]; then
  if \
    grep -Fxq "status=done" "${DONE_FILE}" \
    && grep -Fxq "run_tag=${RUN_TAG}" "${DONE_FILE}" \
    && grep -Fxq "rank_start=${START}" "${DONE_FILE}" \
    && grep -Fxq "rank_stop=${STOP}" "${DONE_FILE}"
  then
    echo "Validated DONE marker. Skipping:"
    echo "${DONE_FILE}"
    exit 0
  fi

  echo "ERROR: inconsistent DONE marker: ${DONE_FILE}" >&2
  exit 1
fi


if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "ERROR: lock exists: ${LOCK_DIR}" >&2
  exit 1
fi


cleanup_lock() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}

trap cleanup_lock EXIT


mkdir -p "${RUN_DIR}"


echo "============================================================"
echo "H0 calibration shard"
echo "============================================================"
echo "HOST              = $(hostname)"
echo "JOB_ID            = ${SLURM_JOB_ID:-NA}"
echo "ARRAY_TASK        = ${SLURM_ARRAY_TASK_ID:-NA}"
echo "RUN_TAG           = ${RUN_TAG}"
echo "SEED              = ${SEED}"
echo "N_CANDIDATES      = ${N_CANDIDATES}"
echo "RANK_START        = ${START}"
echo "RANK_STOP         = ${STOP}"
echo "RUN_DIR           = ${RUN_DIR}"
echo "============================================================"


srun --cpu-bind=cores -u python -u "${RUNNER_PATH}" \
  --config "${CFG_PATH}" \
  --rank-start "${START}" \
  --rank-stop "${STOP}" \
  --n-candidates "${N_CANDIDATES}" \
  --seed "${SEED}" \
  --out-dir "${RUN_DIR}"


DONE_TMP="${DONE_FILE}.tmp.${SLURM_JOB_ID:-manual}.${SLURM_ARRAY_TASK_ID:-0}"

{
  echo "status=done"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-NA}"
  echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-NA}"
  echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-NA}"
  echo "run_tag=${RUN_TAG}"
  echo "rank_start=${START}"
  echo "rank_stop=${STOP}"
  echo "run_dir=${RUN_DIR}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${DONE_TMP}"

mv "${DONE_TMP}" "${DONE_FILE}"

echo "DONE: ${DONE_FILE}"
