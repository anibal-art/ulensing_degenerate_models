#!/usr/bin/env bash

set -euo pipefail

: "${PROJECT_DIR:?}"
: "${STUDY_DIR:?}"


PYTHON_BIN="/home/anibalvarela/.conda/envs/pyLIMA_test/bin/python"


cd "$PROJECT_DIR"


"$PYTHON_BIN" \
    analysis/analyze_lrt_validation_benchmark.py \
    --study-dir "$STUDY_DIR"
