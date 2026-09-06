#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"

PROJECT_DIR="/export/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe"
OUTPUT_ROOT="/export/storage3/rubin/microlensing/romanrubin/hidden_parallax"

RUNNER_PATH="${PROJECT_DIR}/run_lsstmonts_catalog_hidden_parallax.py"
SLURM_SCRIPT="${PROJECT_DIR}/validation/lrt/run_lsstmonts_noise_realizations_array.slurm"

# ============================================================================
# IMPORTANT:
# Start EXACTLY from the validated B2 config.
# ============================================================================

B2_CONFIG="${OUTPUT_ROOT}/production_configs/r50_boundary_sensitivity_20260905T212556Z_B2_expand4/config.json"

B2_STUDY_DIR="${OUTPUT_ROOT}/boundary_sensitivity/r50_boundary_sensitivity_20260905T212556Z"

STUDY_ROOT="${OUTPUT_ROOT}/u0_sensitivity"
LATEST_FILE="${STUDY_ROOT}/LATEST_R50_U0_STUDY"

mkdir -p "${STUDY_ROOT}"


# ============================================================================
# Preconditions
# ============================================================================

if [[ "${CONDA_DEFAULT_ENV:-}" != "pyLIMA_test" ]]; then
    echo "ERROR: activate pyLIMA_test first"
    echo
    echo "    conda activate pyLIMA_test"
    exit 1
fi

for f in \
    "${RUNNER_PATH}" \
    "${SLURM_SCRIPT}" \
    "${B2_CONFIG}"
do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing file:"
        echo "    $f"
        exit 1
    fi
done


# ============================================================================
# Verify that the source really is B2
# ============================================================================

python - "${B2_CONFIG}" <<'PY'
import json
import sys

cfg = json.load(open(sys.argv[1]))

h0 = cfg["fit"]["fits"]["H0"]["bounds"]
h1 = cfg["fit"]["fits"]["H1"]["bounds"]

u0 = h1["u0"]

assert u0["type"] == "center_width"
assert abs(float(u0["half_width"]) - 4.0) < 1e-12

assert abs(float(h1["t0"]["half_width"]) - 120.0) < 1e-12
assert abs(float(h1["rho"]["lower"]) - 1e-11) < 1e-20
assert abs(float(h1["piEN"]["half_width"]) - 1.9973) < 1e-12
assert abs(float(h1["piEE"]["half_width"]) - 8.804384) < 1e-12

de = cfg["fit"]["h1_multistart"]["diagnostic_global_de"]

assert de["enabled"] is True
assert int(de["max_iteration"]) == 1500
assert abs(float(de["atol"]) - 1e-4) < 1e-15
assert float(de["tol"]) == 0.0
assert de["seeds"] == [20260903, 20260904]

print("B2 SOURCE CONFIG VERIFIED")
print("H1 u0 half-width =", u0["half_width"])
print("H1 t0 half-width =", h1["t0"]["half_width"])
print("H1 rho lower     =", h1["rho"]["lower"])
print("H1 piEN halfwidth=", h1["piEN"]["half_width"])
print("H1 piEE halfwidth=", h1["piEE"]["half_width"])
print("DE max_iteration =", de["max_iteration"])
print("DE atol          =", de["atol"])
print("DE seeds         =", de["seeds"])
PY


# ============================================================================
# SUBMIT
# ============================================================================

if [[ "${MODE}" == "submit" ]]; then

    STUDY_TAG="r50_u0_sensitivity_$(date -u +%Y%m%dT%H%M%SZ)"
    STUDY_DIR="${STUDY_ROOT}/${STUDY_TAG}"

    mkdir -p "${STUDY_DIR}" "${PROJECT_DIR}/slurm_logs"

    echo "${STUDY_TAG}" > "${LATEST_FILE}"

    MANIFEST="${STUDY_DIR}/manifest.tsv"

    printf \
        "scenario\tjob_id\tu0_half_width\trun_name\trun_tag\tconfig_path\n" \
        > "${MANIFEST}"

    # Only new cases. B2_u4 already exists.
    SCENARIOS=(
        "B3_u8|8.0"
        "B4_u16|16.0"
    )

    echo
    echo "======================================================================"
    echo "R50 — U0-ONLY SENSITIVITY"
    echo "======================================================================"
    echo
    echo "Control already available:"
    echo "    B2_u4"
    echo
    echo "New study:"
    echo "    ${STUDY_TAG}"
    echo

    for SPEC in "${SCENARIOS[@]}"; do

        IFS='|' read -r SCENARIO U0_HALF_WIDTH <<< "${SPEC}"

        RUN_NAME="LSSTMONTS_xi_baseline_v5p3p5_hiddenParallax_truthInit_truthFSPLparallax_multiFit_LRT_smallConservativeBounds_T8_xscaleJac_noiseMC_nestedBounds_H1u0Sens_r50_${SCENARIO}"

        RUN_TAG="${STUDY_TAG}_${SCENARIO}"

        FROZEN_DIR="${OUTPUT_ROOT}/production_configs/${RUN_TAG}"
        mkdir -p "${FROZEN_DIR}"

        CFG_PATH="${FROZEN_DIR}/config.json"

        # ====================================================================
        # Clone B2 and change ONLY H1 u0 half-width.
        # ====================================================================

        python - \
            "${B2_CONFIG}" \
            "${CFG_PATH}" \
            "${RUN_NAME}" \
            "${SCENARIO}" \
            "${U0_HALF_WIDTH}" \
        <<'PY'
import copy
import json
import sys
from pathlib import Path

src, dst, run_name, scenario, u0_hw = sys.argv[1:]

u0_hw = float(u0_hw)

cfg = json.loads(Path(src).read_text())

# Save exact structures before modification for strict QC.
h0_before = copy.deepcopy(
    cfg["fit"]["fits"]["H0"]
)

h1_bounds_before = copy.deepcopy(
    cfg["fit"]["fits"]["H1"]["bounds"]
)

de_before = copy.deepcopy(
    cfg["fit"]["h1_multistart"]["diagnostic_global_de"]
)

# Unique run location.
cfg["run_name"] = run_name

# THE ONLY SCIENTIFIC CHANGE.
cfg["fit"]["fits"]["H1"]["bounds"]["u0"]["half_width"] = u0_hw

# Metadata only.
cfg["u0_sensitivity_study"] = {
    "scenario": scenario,
    "logical_task": 101,
    "truth_case": "H1",
    "noise_realization_id": 50,
    "source_domain": "B2_expand4",
    "source_u0_half_width": 4.0,
    "new_u0_half_width": u0_hw,
}

# -------------------------------------------------------------------------
# Strict QC:
# H0 unchanged.
# DE unchanged.
# Every H1 bound except u0 unchanged.
# -------------------------------------------------------------------------

assert cfg["fit"]["fits"]["H0"] == h0_before

assert (
    cfg["fit"]["h1_multistart"]["diagnostic_global_de"]
    == de_before
)

h1_after = cfg["fit"]["fits"]["H1"]["bounds"]

for parameter, original in h1_bounds_before.items():

    if parameter == "u0":
        continue

    assert h1_after[parameter] == original, (
        f"Unexpected change in H1 bound {parameter}"
    )

assert h1_after["u0"]["type"] == h1_bounds_before["u0"]["type"]

assert abs(
    float(h1_after["u0"]["half_width"]) - u0_hw
) < 1e-12

Path(dst).write_text(
    json.dumps(cfg, indent=2) + "\n"
)

print()
print("CREATED", scenario)
print("u0 half-width:", u0_hw)
print("H1 bounds:")

for p, b in h1_after.items():
    print(f"  {p}: {b}")

print()
print("DE settings unchanged:")
print(de_before)
PY

        sha256sum "${CFG_PATH}" > "${CFG_PATH}.SHA256"

        cd "${PROJECT_DIR}"

        JOB_ID=$(
            sbatch \
                --parsable \
                --chdir="${PROJECT_DIR}" \
                --array=101 \
                --cpus-per-task=1 \
                --mem=32G \
                --time=06:00:00 \
                --output="${PROJECT_DIR}/slurm_logs/%x_%A_%a.out" \
                --error="${PROJECT_DIR}/slurm_logs/%x_%A_%a.err" \
                --export=ALL,HIDDEN_PARALLAX_BOUNDED_PROFILE=1,CFG_PATH="${CFG_PATH}",RUNNER_PATH="${RUNNER_PATH}",RUN_TAG="${RUN_TAG}",ROW_START_GLOBAL=0,ROW_STOP_GLOBAL=1,LOGICAL_CHUNK_SIZE=1,N_LOGICAL_UPPER=200,WORKERS=1,FORCE_RERUN=0 \
                "${SLURM_SCRIPT}"
        )

        JOB_ID="${JOB_ID%%;*}"

        printf \
            "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${SCENARIO}" \
            "${JOB_ID}" \
            "${U0_HALF_WIDTH}" \
            "${RUN_NAME}" \
            "${RUN_TAG}" \
            "${CFG_PATH}" \
            >> "${MANIFEST}"

        echo
        echo "Submitted:"
        echo "    ${SCENARIO}"
        echo "    u0 half-width = ${U0_HALF_WIDTH}"
        echo "    job           = ${JOB_ID}"

    done

    echo
    echo "======================================================================"
    echo "SUBMITTED"
    echo "======================================================================"
    echo

    column -t -s $'\t' "${MANIFEST}" || cat "${MANIFEST}"

    echo
    echo "After both jobs complete:"
    echo
    echo "    ./validation/optimizer/r50_u0_sensitivity.sh analyze"
    echo

    exit 0
fi


# ============================================================================
# ANALYZE
# ============================================================================

if [[ "${MODE}" == "analyze" ]]; then

    if [[ ! -f "${LATEST_FILE}" ]]; then
        echo "ERROR: no u0 sensitivity study found"
        exit 1
    fi

    STUDY_TAG=$(cat "${LATEST_FILE}")
    STUDY_DIR="${STUDY_ROOT}/${STUDY_TAG}"
    MANIFEST="${STUDY_DIR}/manifest.tsv"

    RESULTS_CSV="${STUDY_DIR}/u0_sensitivity_results.csv"
    PARAMS_CSV="${STUDY_DIR}/u0_sensitivity_parameters.csv"

    python - \
        "${MANIFEST}" \
        "${OUTPUT_ROOT}" \
        "${B2_STUDY_DIR}" \
        "${RESULTS_CSV}" \
        "${PARAMS_CSV}" \
    <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd


manifest_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
b2_study_dir = Path(sys.argv[3])

results_csv = Path(sys.argv[4])
params_csv = Path(sys.argv[5])


def load_npy(path):
    return np.load(
        path,
        allow_pickle=True
    ).item()


summary_rows = []
parameter_rows = []


# =========================================================================
# First import the already validated B2 result.
# =========================================================================

old_results = pd.read_csv(
    b2_study_dir / "boundary_sensitivity_results.csv"
)

old_params = pd.read_csv(
    b2_study_dir / "boundary_sensitivity_parameters.csv"
)

b2 = old_results[
    old_results["scenario"] == "B2_expand4"
].iloc[0]

summary_rows.append({
    "scenario": "B2_u4",
    "u0_half_width": 4.0,
    "status": b2["status"],
    "chi2_H0": float(b2["chi2_H0"]),
    "chi2_DE_TRF": float(b2["chi2_DE_TRF"]),
    "T_H0_minus_H1": float(b2["T_H0_minus_H1"]),
    "best_seed": int(b2["best_seed"]),
    "DE_nit": int(b2["DE_nit"]),
    "DE_nfev": int(b2["DE_nfev"]),
    "minimum_edge_fraction": float(
        b2["minimum_edge_fraction"]
    ),
})

for _, r in old_params[
    old_params["scenario"] == "B2_expand4"
].iterrows():

    parameter_rows.append({
        "scenario": "B2_u4",
        "u0_half_width": 4.0,
        "parameter": r["parameter"],
        "value": float(r["value"]),
        "lower": float(r["lower"]),
        "upper": float(r["upper"]),
        "left_fraction": float(r["left_fraction"]),
        "right_fraction": float(r["right_fraction"]),
        "edge_fraction": float(r["edge_fraction"]),
    })


# =========================================================================
# Analyze B3 and B4.
# =========================================================================

manifest = pd.read_csv(
    manifest_path,
    sep="\t"
)


for _, m in manifest.iterrows():

    scenario = str(m["scenario"])
    u0_hw = float(m["u0_half_width"])

    run_name = str(m["run_name"])
    run_tag = str(m["run_tag"])

    run_base = (
        output_root
        / "runs"
        / run_name
    )

    matches = list(
        run_base.glob(
            f"{run_tag}_logical_101_102_w1"
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"{scenario}: expected one run dir, found {len(matches)}"
        )

    run_dir = matches[0]

    run_summary = pd.read_parquet(
        run_dir / "logs/run_summary.parquet"
    )

    if str(run_summary.iloc[0]["status"]) != "ok":
        raise RuntimeError(
            f"{scenario}: event status is not ok"
        )

    event_dirs = list(
        (run_dir / "fits").glob("*/*")
    )

    if len(event_dirs) != 1:
        raise RuntimeError(
            f"{scenario}: expected one event dir"
        )

    event_dir = event_dirs[0]

    h0_files = list(
        event_dir.glob(
            "*TRF_FSPL_NoParallax.npy"
        )
    )

    if len(h0_files) != 1:
        raise RuntimeError(
            f"{scenario}: H0 file missing"
        )

    h0 = load_npy(
        h0_files[0]
    )

    chi2_h0 = float(
        h0["chi2"]
    )

    de_files = sorted(
        event_dir.glob(
            "_H1_global_DE/seed_*/de_raw.npy"
        )
    )

    candidates = []

    for de_file in de_files:

        de = load_npy(
            de_file
        )

        polish_files = list(
            (
                de_file.parent
                / "trf_polish"
            ).glob(
                "*TRF_FSPL_Parallax.npy"
            )
        )

        if len(polish_files) != 1:
            continue

        polish = load_npy(
            polish_files[0]
        )

        candidates.append(
            (
                float(polish["chi2"]),
                de,
                polish,
            )
        )

    if not candidates:
        raise RuntimeError(
            f"{scenario}: no DE+TRF candidates"
        )

    candidates.sort(
        key=lambda x: x[0]
    )

    chi2_h1, de, polish = candidates[0]

    order = list(
        de["parameter_order"]
    )

    bounds = de[
        "resolved_H1_bounds"
    ]

    best = np.asarray(
        polish["best_model"],
        dtype=float
    ).reshape(-1)

    best = best[
        :len(order)
    ]

    min_edge = np.inf

    for name, value in zip(
        order,
        best
    ):

        lo, hi = bounds[name]

        lo = float(lo)
        hi = float(hi)

        width = hi - lo

        left = (
            float(value) - lo
        ) / width

        right = (
            hi - float(value)
        ) / width

        edge = min(
            left,
            right
        )

        min_edge = min(
            min_edge,
            edge
        )

        parameter_rows.append({
            "scenario": scenario,
            "u0_half_width": u0_hw,
            "parameter": name,
            "value": float(value),
            "lower": lo,
            "upper": hi,
            "left_fraction": left,
            "right_fraction": right,
            "edge_fraction": edge,
        })

    summary_rows.append({
        "scenario": scenario,
        "u0_half_width": u0_hw,
        "status": "ok",
        "chi2_H0": chi2_h0,
        "chi2_DE_TRF": chi2_h1,
        "T_H0_minus_H1": (
            chi2_h0 - chi2_h1
        ),
        "best_seed": int(
            de["seed"]
        ),
        "DE_nit": int(
            de.get("nit", -1)
        ),
        "DE_nfev": int(
            de.get("nfev", -1)
        ),
        "minimum_edge_fraction": float(
            min_edge
        ),
    })


results = pd.DataFrame(
    summary_rows
).sort_values(
    "u0_half_width"
)

parameters = pd.DataFrame(
    parameter_rows
).sort_values(
    ["u0_half_width", "parameter"]
)


results.to_csv(
    results_csv,
    index=False
)

parameters.to_csv(
    params_csv,
    index=False
)


print()
print("=" * 110)
print("R50 — U0-ONLY SENSITIVITY")
print("=" * 110)

print(
    results[
        [
            "scenario",
            "u0_half_width",
            "chi2_H0",
            "chi2_DE_TRF",
            "T_H0_minus_H1",
            "best_seed",
            "DE_nit",
            "DE_nfev",
            "minimum_edge_fraction",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 110)
print("U0 SOLUTION")
print("=" * 110)

u0 = parameters[
    parameters["parameter"] == "u0"
][
    [
        "scenario",
        "u0_half_width",
        "value",
        "lower",
        "upper",
        "left_fraction",
        "right_fraction",
        "edge_fraction",
    ]
]

print(
    u0.to_string(
        index=False
    )
)


print()
print("=" * 110)
print("ALL PARAMETER EDGE FRACTIONS")
print("=" * 110)

edge_table = parameters.pivot(
    index="scenario",
    columns="parameter",
    values="edge_fraction"
)

print(
    edge_table.to_string()
)


print()
print("=" * 110)
print("DELTA CHI2 RELATIVE TO B2")
print("=" * 110)

b2_chi = float(
    results.loc[
        results["scenario"] == "B2_u4",
        "chi2_DE_TRF"
    ].iloc[0]
)

for _, row in results.iterrows():

    delta = (
        float(row["chi2_DE_TRF"])
        - b2_chi
    )

    print(
        f"{row['scenario']:10s}: "
        f"{delta:+.9f}"
    )


print()
print("H0 chi2 spread =",
      results["chi2_H0"].max()
      - results["chi2_H0"].min())

print()
print("Saved:")
print(" ", results_csv)
print(" ", params_csv)
PY

    exit 0
fi


echo
echo "Usage:"
echo
echo "  conda activate pyLIMA_test"
echo
echo "  $0 submit"
echo
echo "  $0 analyze"
echo

exit 2
