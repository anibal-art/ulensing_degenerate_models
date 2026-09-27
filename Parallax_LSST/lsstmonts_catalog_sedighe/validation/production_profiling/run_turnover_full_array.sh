#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

WORK="/export/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe"

cd "$WORK"

/home/anibalvarela/.conda/envs/pyLIMA_test/bin/python \
  validation/production_profiling/turnover_chi2_worker_full.py \
  --sample "$MANIFEST" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --chunk-size "$CHUNK" \
  --out-dir "$PARTS"
