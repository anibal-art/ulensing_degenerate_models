#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

PROJECT_DIR="$(
    cd "${SCRIPT_DIR}/../.."
    pwd
)"

WRAPPER="${PROJECT_DIR}/validation/bounds_audit/run_bounds_audit_refit.py"

MANIFEST="${PROJECT_DIR}/validation/bounds_audit/data/refit_manifest.csv"

CROSSSEEDS="${PROJECT_DIR}/validation/bounds_convergence/data/h0_crossseeds_34.csv"

export HIDDEN_PARALLAX_BOUNDS_AUDIT_ROOT="${HIDDEN_PARALLAX_BOUNDS_AUDIT_ROOT:-${HOME}/Downloads/hidden_parallax/hidden_parallax_refit_test}"

export ROMAN_RUBIN_DIR="${ROMAN_RUBIN_DIR:-${HOME}/microlensing/simulation_Rubin/roman_rubin}"

export HIDDEN_PARALLAX_H0_CROSSSEED_MANIFEST="${CROSSSEEDS}"

if [[ ! -f "${CROSSSEEDS}" ]]; then
    echo "Missing cross-seed manifest:"
    echo "  ${CROSSSEEDS}"
    exit 1
fi

if [[ -n "${ROW_ONLY:-}" ]]; then

    ROWS="${ROW_ONLY}"

else

    ROWS="$(
        python - "${MANIFEST}" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])

print(
    " ".join(
        map(
            str,
            df["catalog_row"].astype(int)
        )
    )
)
PY
    )"

fi

EXTRA_ARGS=()

if [[ "${FORCE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--force)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi

echo "PROJECT_DIR      = ${PROJECT_DIR}"
echo "WORK_ROOT        = ${HIDDEN_PARALLAX_BOUNDS_AUDIT_ROOT}"
echo "ROMAN_RUBIN_DIR  = ${ROMAN_RUBIN_DIR}"
echo "CROSSSEEDS       = ${CROSSSEEDS}"
echo "ROWS             = ${ROWS}"

for ROW in ${ROWS}
do

    echo
    echo "======================================================================"
    echo "reference5000 catalog_row=${ROW}"
    echo "======================================================================"

    python "${WRAPPER}" \
        --catalog-row "${ROW}" \
        --bounds-profile reference5000 \
        "${EXTRA_ARGS[@]}"

done
