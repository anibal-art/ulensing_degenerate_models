#!/usr/bin/env bash
set -euo pipefail

PROJECT="Parallax_LSST/lsstmonts_catalog_sedighe"
BC="$PROJECT/validation/bounds_convergence"
WRAPPER="$PROJECT/validation/bounds_audit/run_bounds_audit_refit.py"
MANIFEST="$BC/data/extreme100_refit_manifest.csv"

SOURCE_ROOT="$HOME/Downloads/hidden_parallax/hidden_parallax_refit_test"

ROOT_BASE="$HOME/Downloads/hidden_parallax/production_validation/gate1B_final18"

export ROMAN_RUBIN_DIR="$HOME/microlensing/simulation_Rubin/roman_rubin"
export HIDDEN_PARALLAX_TRF_X_SCALE=pylima
export HIDDEN_PARALLAX_T0_MARGIN_FACTOR=0.25
export HIDDEN_PARALLAX_H0_START_PLAN=gate1_final18

unset HIDDEN_PARALLAX_H0_CROSSSEED_MANIFEST

declare -A EXPECTED
EXPECTED[physical]=3
EXPECTED[log_te]=6
EXPECTED[log_rho]=8
EXPECTED[log_te_rho]=1

mapfile -t ROWS < <(
python - "$MANIFEST" <<'PY'
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])

for row in (
    pd.to_numeric(
        df["catalog_row"],
        errors="raise",
    )
    .astype(int)
    .drop_duplicates()
):
    print(row)
PY
)

echo "N events = ${#ROWS[@]}"

for MODE in physical log_te log_rho log_te_rho
do
    export HIDDEN_PARALLAX_TRF_COORDS="$MODE"

    ROOT="$ROOT_BASE/$MODE"

    mkdir -p "$ROOT"
    ln -sfn "$SOURCE_ROOT/artifacts" "$ROOT/artifacts"

    for ROW in "${ROWS[@]}"
    do
        OUT="$ROOT/refits/bounds_convergence/te500000_h0only/extreme100/$ROW/all_refits.csv"

        COMPLETE=0

        if [[ -f "$OUT" ]]; then
            COMPLETE=$(
            python - "$OUT" "${EXPECTED[$MODE]}" <<'PY'
import pandas as pd
import sys

p = sys.argv[1]
expected = int(sys.argv[2])

try:
    df = pd.read_csv(p)

    ok = df[
        (df["hypothesis"] == "H0")
        & (df["status"] == "success")
    ]

    print(
        int(
            len(ok) == expected
        )
    )
except Exception:
    print(0)
PY
            )
        fi

        if [[ "$COMPLETE" == "1" ]]; then
            echo "SKIP MODE=$MODE ROW=$ROW"
            continue
        fi

        echo
        echo "============================================================"
        echo "MODE=$MODE ROW=$ROW"
        echo "============================================================"

        /usr/bin/time \
            -f "%e" \
            -o "$ROOT/wall_${ROW}.txt" \
            python "$WRAPPER" \
                --manifest "$MANIFEST" \
                --catalog-row "$ROW" \
                --bounds-profile te500000 \
                --fit-scope h0 \
                --work-root "$ROOT" \
                --force \
                > "$ROOT/run_${ROW}.log" \
                2>&1

        grep -E \
            '\[gate1_final18\]|new_h0_chi2|best_h0' \
            "$ROOT/run_${ROW}.log" \
            | tail -n 5 || true
    done
done
