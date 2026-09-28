#!/bin/bash
#SBATCH --job-name=h0robust
#SBATCH --partition=milliways-int
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/export/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/validation/production_profiling/optimizer_robustness/slurm/h0robust_%A_%a.out
#SBATCH --error=/export/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/validation/production_profiling/optimizer_robustness/slurm/h0robust_%A_%a.err

set -euo pipefail

REPO="/export/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe"
ROOT="$REPO/validation/production_profiling/optimizer_robustness"

PY="/home/anibalvarela/.conda/envs/pyLIMA_test/bin/python"

MAN="$ROOT/h1_h0_robustness_sample_frozen.csv"

N=3336
CHUNK=50

TASK="${SLURM_ARRAY_TASK_ID}"

START=$(( TASK * CHUNK ))
STOP=$(( START + CHUNK ))

if [ "$STOP" -gt "$N" ]; then
    STOP="$N"
fi

if [ "$START" -ge "$N" ]; then
    echo "Nothing to do: START=$START N=$N"
    exit 0
fi

cd "$REPO"

echo "============================================================"
echo "H0 OPTIMIZER ROBUSTNESS"
echo "============================================================"
echo "job_id       = ${SLURM_JOB_ID}"
echo "array_task   = ${SLURM_ARRAY_TASK_ID}"
echo "host         = $(hostname)"
echo "start        = ${START}"
echo "stop         = ${STOP}"
echo "python       = ${PY}"
echo "date_start   = $(date -Is)"
echo "git_head     = $(git rev-parse HEAD)"
echo "============================================================"

"$PY" \
    "$ROOT/run_h0_robustness_worker.py" \
    --manifest "$MAN" \
    --start "$START" \
    --stop "$STOP" \
    --out-root "$ROOT/results/full3336_events" \
    --summary-out "$ROOT/results/full3336_parts/part_${START}_${STOP}.parquet"

echo "date_end = $(date -Is)"
echo "DONE"
