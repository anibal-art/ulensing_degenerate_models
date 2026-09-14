#!/usr/bin/env bash
set -euo pipefail

PROJECT="Parallax_LSST/lsstmonts_catalog_sedighe"
BC="$PROJECT/validation/bounds_convergence"
WRAPPER="$PROJECT/validation/bounds_audit/run_bounds_audit_refit.py"
MANIFEST="$BC/data/extreme100_refit_manifest.csv"

SOURCE_ROOT="$HOME/Downloads/hidden_parallax/hidden_parallax_refit_test"
BENCH_ROOT="$HOME/Downloads/hidden_parallax/production_validation/gate1_coordinates"

export ROMAN_RUBIN_DIR="$HOME/microlensing/simulation_Rubin/roman_rubin"
export HIDDEN_PARALLAX_TRF_X_SCALE=pylima

unset HIDDEN_PARALLAX_H0_CROSSSEED_MANIFEST

mkdir -p "$BENCH_ROOT"

mapfile -t ROWS < <(
python - "$MANIFEST" <<'PY'
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])

rows = (
    pd.to_numeric(df["catalog_row"], errors="raise")
    .astype(int)
    .drop_duplicates()
    .tolist()
)

for row in rows:
    print(row)
PY
)

echo "N extreme100 rows = ${#ROWS[@]}"

for MODE in physical log_te log_rho log_te_rho
do
    ROOT="$BENCH_ROOT/$MODE"

    mkdir -p "$ROOT"
    ln -sfn "$SOURCE_ROOT/artifacts" "$ROOT/artifacts"

    export HIDDEN_PARALLAX_TRF_COORDS="$MODE"

    echo
    echo "################################################################"
    echo "MODE=$MODE"
    echo "################################################################"

    MODE_START=$(date +%s)

    for ROW in "${ROWS[@]}"
    do
        OUT="$ROOT/refits/bounds_convergence/te500000_h0only/extreme100/$ROW/all_refits.csv"

        # --------------------------------------------------------
        # Resume support: skip only if the expected 13 H0 fits
        # are already present and successful.
        # --------------------------------------------------------

        if [[ -f "$OUT" ]]; then
            COMPLETE=$(
            python - "$OUT" <<'PY'
import pandas as pd
import sys

try:
    df = pd.read_csv(sys.argv[1])

    ok = df[
        (df["hypothesis"] == "H0")
        & (df["status"] == "success")
    ]

    print(int(len(ok) >= 13))
except Exception:
    print(0)
PY
            )

            if [[ "$COMPLETE" == "1" ]]; then
                echo "SKIP MODE=$MODE ROW=$ROW already complete"
                continue
            fi
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
            "\[trf_x_scale\]|\[trf_coords\]" \
            "$ROOT/run_${ROW}.log" \
            | tail -n 2 || true

        grep -E \
            "new_h0_chi2|best_h0" \
            "$ROOT/run_${ROW}.log" \
            | tail -n 2 || true
    done

    MODE_END=$(date +%s)

    echo "$((MODE_END - MODE_START))" \
        > "$ROOT/wall_mode_total_seconds.txt"

done
