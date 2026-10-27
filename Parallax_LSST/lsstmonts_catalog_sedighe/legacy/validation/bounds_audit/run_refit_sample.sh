#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

MANIFEST="${MANIFEST:-$SCRIPT_DIR/data/refit_manifest.csv}"

WORK_ROOT="${HIDDEN_PARALLAX_BOUNDS_AUDIT_ROOT:-$SCRIPT_DIR/work}"

JOBS="${JOBS:-4}"

FORCE="${FORCE:-0}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found:"
    echo "$MANIFEST"
    exit 1
fi

mkdir -p \
    "$WORK_ROOT"

export HIDDEN_PARALLAX_BOUNDS_AUDIT_ROOT="$WORK_ROOT"
export HIDDEN_PARALLAX_BOUNDED_PROFILE=1

ROWS_FILE="$WORK_ROOT/catalog_rows.txt"

python - "$MANIFEST" > "$ROWS_FILE" <<'PY'
import sys
import pandas as pd

p = sys.argv[1]

df = pd.read_csv(
    p
)

if "catalog_row" not in df:
    raise RuntimeError(
        "manifest has no catalog_row column"
    )

for row in df["catalog_row"].astype(int):
    print(row)
PY

echo
echo "============================================================"
echo "BOUNDS AUDIT"
echo "============================================================"
echo "manifest  = $MANIFEST"
echo "work_root = $WORK_ROOT"
echo "jobs      = $JOBS"
echo "events    = $(wc -l < "$ROWS_FILE")"
echo

if [[ "$FORCE" == "1" ]]; then

    cat "$ROWS_FILE" | \
    xargs \
        -P "$JOBS" \
        -I{} \
        python \
        "$SCRIPT_DIR/run_bounds_audit_refit.py" \
        --catalog-row {} \
        --force

else

    cat "$ROWS_FILE" | \
    xargs \
        -P "$JOBS" \
        -I{} \
        python \
        "$SCRIPT_DIR/run_bounds_audit_refit.py" \
        --catalog-row {}

fi
