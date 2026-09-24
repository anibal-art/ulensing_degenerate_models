#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash tools/compile_latex_clean.sh path/to/file.tex" >&2
    exit 2
fi

TEX="$1"
TEX_DIR="$(cd "$(dirname "$TEX")" && pwd)"
TEX_NAME="$(basename "$TEX")"
BASE="${TEX_NAME%.tex}"
LOG="$TEX_DIR/$BASE.log"

cd "$TEX_DIR"

latexmk -C "$TEX_NAME"

latexmk \
    -pdf \
    -interaction=nonstopmode \
    -halt-on-error \
    "$TEX_NAME"

echo
echo "===== LaTeX layout / file warnings ====="
if [[ -f "$LOG" ]]; then
    grep -nE \
        'Overfull|Underfull|LaTeX Warning|pdfTeX warning|File .* not found' \
        "$LOG" || true
else
    echo "No log file found: $LOG"
fi
