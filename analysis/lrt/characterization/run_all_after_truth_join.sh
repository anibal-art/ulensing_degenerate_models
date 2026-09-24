#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$HERE/11_detection_efficiency_1d.py"
python "$HERE/12_detection_efficiency_maps.py"
python "$HERE/13_parallax_recovery.py"
python "$HERE/14_covariance_coverage.py"
