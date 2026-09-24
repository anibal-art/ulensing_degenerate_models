#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

python "$SCRIPT_DIR/01_full_distribution.py"

python "$SCRIPT_DIR/02_low_delta_chi2.py"

python "$SCRIPT_DIR/03_survival_function.py"
