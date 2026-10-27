#!/usr/bin/env bash

set -euo pipefail

: "${PROJECT_DIR:?}"
: "${OUTPUT_ROOT:?}"
: "${CFG_PATH:?}"
: "${RUN_NAME:?}"
: "${RUN_TAG:?}"
: "${N_ROWS:?}"
: "${CHUNK_SIZE:?}"
: "${WORKERS:?}"

PYTHON_BIN="/home/anibalvarela/.conda/envs/pyLIMA_test/bin/python"

TASK_ID="${SLURM_ARRAY_TASK_ID}"

START=$(( TASK_ID * CHUNK_SIZE ))
STOP=$(( START + CHUNK_SIZE ))

if (( STOP > N_ROWS )); then
    STOP="${N_ROWS}"
fi

if (( START >= N_ROWS )); then
    echo "Nothing to do: START=${START} >= N_ROWS=${N_ROWS}"
    exit 0
fi

SUFFIX="${RUN_TAG}_chunk_${START}_${STOP}_w${WORKERS}"

echo "============================================================"
echo "PREFIT DETECTABILITY"
echo "============================================================"
echo "task       = ${TASK_ID}"
echo "rows       = [${START}, ${STOP})"
echo "workers    = ${WORKERS}"
echo "config     = ${CFG_PATH}"
echo "run_name   = ${RUN_NAME}"
echo "suffix     = ${SUFFIX}"
echo "============================================================"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" run_lsstmonts_catalog_hidden_parallax.py \
    --config "${CFG_PATH}" \
    --catalog-row-start "${START}" \
    --catalog-row-stop "${STOP}" \
    --workers "${WORKERS}" \
    --run-name-suffix "${SUFFIX}"

SUMMARY="${OUTPUT_ROOT}/runs/${RUN_NAME}/${SUFFIX}/logs/run_summary.parquet"

if [[ ! -f "${SUMMARY}" ]]; then
    echo "ERROR: summary missing:"
    echo "${SUMMARY}"
    exit 2
fi

echo
echo "SUMMARY=${SUMMARY}"
echo "DONE"
