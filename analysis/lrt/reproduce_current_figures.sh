#!/usr/bin/env bash

# =============================================================================
# Reproduce the current LRT analysis figures
# =============================================================================
#
# Purpose
# -------
# Re-run the figure-producing scripts that belong to the CURRENT
# hidden-parallax LRT analysis.
#
# This script:
#   - uses the frozen H0/H1 production outputs already present in analysis/lrt;
#   - regenerates current characterization and validation figures;
#   - regenerates derived tables produced by those scripts;
#   - may re-run Monte-Carlo propagation or statistical cross-validation;
#   - DOES NOT re-simulate microlensing events;
#   - DOES NOT re-fit the production H0/H1 light curves.
#
# Explicit exclusions
# -------------------
# Legacy scripts requiring the obsolete truth-join product:
#
#   11_detection_efficiency_1d.py
#   12_detection_efficiency_maps.py
#   13_parallax_recovery.py
#   14_covariance_coverage.py
#
# These depend on:
#
#   analysis/lrt/data/h1_lrt_truth_joined_20260921.parquet
#
# and have been superseded by the later characterization pipeline based on
# 18_positive_truth/h1_positive_truth_characterization.parquet.
#
# Also excluded:
#
#   30_adjusted_detection_vs_finite_source_strength.py
#
# because that analysis was scientifically superseded/rejected.
#
# paper_style.py is a plotting helper and is not itself a figure script.
#
# Usage
# -----
#
#   bash analysis/lrt/reproduce_current_figures.sh
#
# Logs are written under:
#
#   analysis/lrt/_figure_reproduction_logs/
#
# The script continues after individual failures and reports them at the end.
# =============================================================================


set -u
set -o pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

ROOT="$(
    cd "$SCRIPT_DIR/../.."
    pwd
)"

cd "$ROOT" || {
    echo "ERROR: could not enter repository root: $ROOT"
    exit 1
}

export MPLBACKEND=Agg


# =============================================================================
# Configuration
# =============================================================================

OUTDIR="analysis/lrt/_figure_reproduction_logs"

LIST="$OUTDIR/current_figure_scripts.txt"
FAILS="$OUTDIR/failed_scripts.txt"
SUMMARY="$OUTDIR/reproduction_summary.txt"

mkdir -p "$OUTDIR"

: > "$FAILS"
: > "$SUMMARY"


# -----------------------------------------------------------------------------
# Scripts intentionally excluded from the current reproducible figure set.
# -----------------------------------------------------------------------------

EXCLUDE_RE='(__init__|common|chi2_reference|paper_style|10_prepare_truth_join|11_detection_efficiency_1d|12_detection_efficiency_maps|13_parallax_recovery|14_covariance_coverage|30_adjusted_detection_vs_finite_source_strength)\.py$'


# =============================================================================
# Discover active figure-producing scripts
# =============================================================================

grep -RIl \
    --include='*.py' \
    -E 'savefig|plt\.subplots|plt\.figure|figure\(' \
    analysis/lrt/characterization \
    analysis/lrt/validation_tests \
    analysis/lrt/plotting \
    | grep -vE "$EXCLUDE_RE" \
    | sort -V \
    > "$LIST"


N="$(
    wc -l < "$LIST"
)"

echo "============================================================================="
echo "CURRENT LRT FIGURE REPRODUCTION"
echo "============================================================================="
echo "Repository : $ROOT"
echo "Scripts    : $N"
echo "Logs       : $OUTDIR"
echo
echo "Scripts to execute:"
cat "$LIST"
echo


# =============================================================================
# Run
# =============================================================================

i=0
n_ok=0
n_fail=0

while IFS= read -r script; do

    i=$((i + 1))

    safe_name="$(
        echo "$script" \
        | sed 's|/|__|g; s|\.py$||'
    )"

    log="$OUTDIR/${safe_name}.log"

    echo
    echo "============================================================================="
    echo "[$i/$N] $script"
    echo "============================================================================="

    python -u "$script" 2>&1 | tee "$log"

    status=${PIPESTATUS[0]}

    if [ "$status" -eq 0 ]; then

        echo "[OK] $script"

        n_ok=$((n_ok + 1))

    else

        echo "[FAIL: exit $status] $script"

        echo "$script" >> "$FAILS"

        n_fail=$((n_fail + 1))

    fi

done < "$LIST"


# =============================================================================
# Summary
# =============================================================================

{
    echo "CURRENT LRT FIGURE REPRODUCTION"
    echo "================================"
    echo
    echo "Repository : $ROOT"
    echo "Scripts    : $N"
    echo "Successful : $n_ok"
    echo "Failed     : $n_fail"
    echo

    if [ -s "$FAILS" ]; then
        echo "FAILED SCRIPTS"
        echo "--------------"
        cat "$FAILS"
    else
        echo "All current figure scripts completed successfully."
    fi
} | tee "$SUMMARY"


echo
echo "Script list:"
echo "  $LIST"

echo
echo "Summary:"
echo "  $SUMMARY"

echo
echo "Logs:"
echo "  $OUTDIR"

echo
echo "Most recently written figures:"
find analysis/lrt/figures \
    -type f \
    \( -name '*.png' -o -name '*.pdf' \) \
    -printf '%TY-%Tm-%Td %TH:%TM %p\n' \
    | sort \
    | tail -40
