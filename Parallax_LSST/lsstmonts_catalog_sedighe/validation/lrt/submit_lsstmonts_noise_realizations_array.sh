#!/bin/bash
set -Eeuo pipefail

# ============================================================================
# Submit Hidden-Parallax photometric-noise realizations.
#
# The physical catalog window stays fixed.
# The Python runner expands it into:
#
#     event x realization x truth_case
#
# and the SLURM array slices the expanded logical task table.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNNER_SOURCE="${PROJECT_DIR}/run_lsstmonts_catalog_hidden_parallax.py"

DEFAULT_CONFIG="${PROJECT_DIR}/configs/config_lsstmonts_baseline_v5p3p5_cluster_che_multifit_LRT_truth_init_T8_xscale_jac_noiseMC_smoke5.json"

CONFIG_SOURCE="${CONFIG_SOURCE:-${DEFAULT_CONFIG}}"

SLURM_SCRIPT="${PROJECT_DIR}/validation/lrt/run_lsstmonts_noise_realizations_array.slurm"

# Physical catalog rows.
ROW_START_GLOBAL="${ROW_START_GLOBAL:-0}"
ROW_STOP_GLOBAL="${ROW_STOP_GLOBAL:-1}"

# Number of logical datasets handled by one SLURM job.
LOGICAL_CHUNK_SIZE="${LOGICAL_CHUNK_SIZE:-100}"

# CHE architecture.
WORKERS="${WORKERS:-2}"
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
MEM_PER_TASK="${MEM_PER_TASK:-8G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

FORCE_RERUN="${FORCE_RERUN:-0}"

RUN_TAG="${RUN_TAG:-noiseMC_$(date -u +%Y%m%dT%H%M%SZ)}"


for path in \
    "${RUNNER_SOURCE}" \
    "${CONFIG_SOURCE}" \
    "${SLURM_SCRIPT}"
do

    if [[ ! -f "${path}" ]]; then
        echo "ERROR: missing file: ${path}" >&2
        exit 1
    fi

done


if (( ROW_STOP_GLOBAL <= ROW_START_GLOBAL )); then
    echo "ERROR: ROW_STOP_GLOBAL must be > ROW_START_GLOBAL." >&2
    exit 1
fi


if (( LOGICAL_CHUNK_SIZE <= 0 )); then
    echo "ERROR: LOGICAL_CHUNK_SIZE must be > 0." >&2
    exit 1
fi


# ============================================================================
# Read MC dimensions from config
# ============================================================================

mapfile -t MC_INFO < <(
python - "${CONFIG_SOURCE}" <<'PYCFG'
import json
import sys

cfg = json.load(
    open(sys.argv[1])
)

mc = cfg.get(
    "noise_realizations",
    {},
)

if not mc.get(
    "enabled",
    False,
):
    raise SystemExit(
        "noise_realizations.enabled must be true"
    )

n = int(
    mc[
        "n_realizations"
    ]
)

truth_cases = mc.get(
    "truth_cases",
    [],
)

if n <= 0:
    raise SystemExit(
        "n_realizations must be > 0"
    )

if not truth_cases:
    raise SystemExit(
        "truth_cases cannot be empty"
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

if not output_root:
    raise SystemExit(
        "Could not resolve output_root from config"
    )

if not run_name:
    raise SystemExit(
        "Could not resolve run_name from config"
    )

print(n)
print(len(truth_cases))
print(output_root)
print(run_name)
PYCFG
)


N_REALIZATIONS="${MC_INFO[0]}"
N_TRUTH_CASES="${MC_INFO[1]}"
OUTPUT_ROOT="${MC_INFO[2]}"
RUN_NAME_BASE="${MC_INFO[3]}"


# ============================================================================
# Correct Bash integer arithmetic
# ============================================================================

N_PHYSICAL_ROWS=$(( ROW_STOP_GLOBAL - ROW_START_GLOBAL ))

N_LOGICAL_UPPER=$(( N_PHYSICAL_ROWS * N_REALIZATIONS * N_TRUTH_CASES ))

N_CHUNKS=$(( (N_LOGICAL_UPPER + LOGICAL_CHUNK_SIZE - 1) / LOGICAL_CHUNK_SIZE ))

ARRAY_MAX=$(( N_CHUNKS - 1 ))


if (( N_LOGICAL_UPPER <= 0 || N_CHUNKS <= 0 )); then

    echo "ERROR: empty logical experiment." >&2
    exit 1

fi


# ============================================================================
# Freeze configuration
# ============================================================================

FROZEN_DIR="${OUTPUT_ROOT}/production_configs/${RUN_TAG}"

mkdir -p "${FROZEN_DIR}"

CFG_PATH="${FROZEN_DIR}/config.json"

cp \
    "${CONFIG_SOURCE}" \
    "${CFG_PATH}"

sha256sum \
    "${CFG_PATH}" \
    > "${CFG_PATH}.SHA256"


MANIFEST="${FROZEN_DIR}/manifest.txt"

{
    echo "experiment=noise_realizations"
    echo "run_tag=${RUN_TAG}"
    echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "submit_host=$(hostname)"
    echo "config_source=${CONFIG_SOURCE}"
    echo "cfg_path=${CFG_PATH}"

    echo "row_start_global=${ROW_START_GLOBAL}"
    echo "row_stop_global=${ROW_STOP_GLOBAL}"
    echo "n_physical_rows=${N_PHYSICAL_ROWS}"

    echo "n_realizations=${N_REALIZATIONS}"
    echo "n_truth_cases=${N_TRUTH_CASES}"
    echo "n_logical_upper=${N_LOGICAL_UPPER}"

    echo "logical_chunk_size=${LOGICAL_CHUNK_SIZE}"
    echo "n_chunks=${N_CHUNKS}"

    echo "workers=${WORKERS}"
    echo "max_concurrent=${MAX_CONCURRENT}"
    echo "cpus_per_task=${CPUS_PER_TASK}"
    echo "mem_per_task=${MEM_PER_TASK}"
    echo "time_limit=${TIME_LIMIT}"

    echo "force_rerun=${FORCE_RERUN}"

    # BOUNDED_PROFILE_SUBMIT_MANIFEST_V1
    echo "hidden_parallax_bounded_profile=${HIDDEN_PARALLAX_BOUNDED_PROFILE:-0}"
} > "${MANIFEST}"


mkdir -p \
    "${PROJECT_DIR}/slurm_logs"


cat <<INFO
========================================================================
Submitting Hidden-Parallax noise-realization experiment
========================================================================

RUN_TAG             = ${RUN_TAG}

CONFIG_SOURCE       = ${CONFIG_SOURCE}
CFG_PATH            = ${CFG_PATH}
RUN_NAME            = ${RUN_NAME_BASE}

physical rows       = [${ROW_START_GLOBAL}, ${ROW_STOP_GLOBAL})
N physical rows     = ${N_PHYSICAL_ROWS}

N realizations      = ${N_REALIZATIONS}
N truth cases       = ${N_TRUTH_CASES}

logical upper       = ${N_LOGICAL_UPPER}
logical chunk       = ${LOGICAL_CHUNK_SIZE}

array               = 0-${ARRAY_MAX}%${MAX_CONCURRENT}

workers/job         = ${WORKERS}
cpus/task           = ${CPUS_PER_TASK}
memory/task         = ${MEM_PER_TASK}
time                = ${TIME_LIMIT}

MANIFEST            = ${MANIFEST}

========================================================================
INFO


cd "${PROJECT_DIR}"


sbatch \
    --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM_PER_TASK}" \
    --time="${TIME_LIMIT}" \
    --export=ALL,HIDDEN_PARALLAX_BOUNDED_PROFILE="${HIDDEN_PARALLAX_BOUNDED_PROFILE:-0}",CFG_PATH="${CFG_PATH}",RUNNER_PATH="${RUNNER_SOURCE}",RUN_TAG="${RUN_TAG}",ROW_START_GLOBAL="${ROW_START_GLOBAL}",ROW_STOP_GLOBAL="${ROW_STOP_GLOBAL}",LOGICAL_CHUNK_SIZE="${LOGICAL_CHUNK_SIZE}",N_LOGICAL_UPPER="${N_LOGICAL_UPPER}",WORKERS="${WORKERS}",FORCE_RERUN="${FORCE_RERUN}" \
    "${SLURM_SCRIPT}"
