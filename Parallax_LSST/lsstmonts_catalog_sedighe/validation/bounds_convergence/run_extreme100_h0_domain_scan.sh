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

MANIFEST="${PROJECT_DIR}/validation/bounds_convergence/data/extreme100_refit_manifest.csv"

export HIDDEN_PARALLAX_BOUNDS_AUDIT_ROOT="${
    HIDDEN_PARALLAX_BOUNDS_AUDIT_ROOT:-
    ${HOME}/Downloads/hidden_parallax/hidden_parallax_refit_test
}"

export ROMAN_RUBIN_DIR="${
    ROMAN_RUBIN_DIR:-
    ${HOME}/microlensing/simulation_Rubin/roman_rubin
}"

# Optional validation cross-seeds.
if [[ -n "${CROSSSEED_MANIFEST:-}" ]]; then
    export HIDDEN_PARALLAX_H0_CROSSSEED_MANIFEST="${CROSSSEED_MANIFEST}"
    echo "Cross-seeds = ${HIDDEN_PARALLAX_H0_CROSSSEED_MANIFEST}"
else
    unset HIDDEN_PARALLAX_H0_CROSSSEED_MANIFEST || true
    echo "Cross-seeds = NONE"
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

PROFILES="${PROFILES:-reference5000 control20000}"

EXTRA_ARGS=()

if [[ "${FORCE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--force)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi

echo
echo "PROJECT_DIR = ${PROJECT_DIR}"
echo "MANIFEST    = ${MANIFEST}"
echo "WORK_ROOT   = ${HIDDEN_PARALLAX_BOUNDS_AUDIT_ROOT}"
echo "PROFILES    = ${PROFILES}"

NROWS="$(
    wc -w <<< "${ROWS}"
)"

echo "N rows      = ${NROWS}"
echo

for ROW in ${ROWS}
do

    for PROFILE in ${PROFILES}
    do

        echo
        echo "======================================================================"
        echo "catalog_row=${ROW} profile=${PROFILE}"
        echo "======================================================================"

        python "${WRAPPER}" \
            --manifest "${MANIFEST}" \
            --catalog-row "${ROW}" \
            --bounds-profile "${PROFILE}" \
            --fit-scope h0 \
            "${EXTRA_ARGS[@]}"

    done

done
