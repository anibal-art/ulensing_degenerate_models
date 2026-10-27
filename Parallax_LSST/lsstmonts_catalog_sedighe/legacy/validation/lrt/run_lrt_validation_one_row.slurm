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
