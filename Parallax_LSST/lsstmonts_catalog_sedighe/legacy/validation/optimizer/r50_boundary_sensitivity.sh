#!/usr/bin/env bash

set -euo pipefail


# =============================================================================
# r50 H1 boundary-sensitivity experiment
#
# Usage:
#
#   conda activate pyLIMA_test
#
#   ./validation/optimizer/r50_boundary_sensitivity.sh submit
#
# After all jobs finish:
#
#   ./validation/optimizer/r50_boundary_sensitivity.sh analyze
#
# Scientific design:
#
#   same event/noise realization:
#       H1 truth, realization r50, logical task 101
#
#   same:
#       data
#       noise seed
#       H0
#       bounded flux profiling
#       DE strategy
#       DE seeds
#       TRF polish
#
#   only H1 fitting domain changes.
# =============================================================================


MODE="${1:-}"


PROJECT_ROOT="/export/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models"

PROJECT_DIR="${PROJECT_ROOT}/Parallax_LSST/lsstmonts_catalog_sedighe"

OUTPUT_ROOT="/export/storage3/rubin/microlensing/romanrubin/hidden_parallax"


RUNNER_PATH="${PROJECT_DIR}/run_lsstmonts_catalog_hidden_parallax.py"

SLURM_SCRIPT="${PROJECT_DIR}/validation/lrt/run_lsstmonts_noise_realizations_array.slurm"


# -------------------------------------------------------------------------
# Base config.
#
# We explicitly overwrite the relevant DE settings below, so the study
# does not depend on what max_iteration/atol happened to be in this file.
# -------------------------------------------------------------------------

BASE_CONFIG="${PROJECT_DIR}/configs/config_lsstmonts_baseline_v5p3p5_cluster_che_multifit_LRT_T8_xscale_jac_noiseMC_nestedBounds_H1globalDE1500x2_R100.json"


# -------------------------------------------------------------------------
# Persistent study bookkeeping
# -------------------------------------------------------------------------

STUDY_ROOT="${OUTPUT_ROOT}/boundary_sensitivity"

LATEST_FILE="${STUDY_ROOT}/LATEST_R50_BOUNDARY_STUDY"


mkdir -p "${STUDY_ROOT}"


# =============================================================================
# Basic checks
# =============================================================================

if [[ "${CONDA_DEFAULT_ENV:-}" != "pyLIMA_test" ]]; then

    echo "ERROR: activate pyLIMA_test first:"
    echo
    echo "    conda activate pyLIMA_test"
    echo

    exit 1

fi


if [[ ! -f "${RUNNER_PATH}" ]]; then
    echo "ERROR: runner not found:"
    echo "${RUNNER_PATH}"
    exit 1
fi


if [[ ! -f "${SLURM_SCRIPT}" ]]; then
    echo "ERROR: SLURM script not found:"
    echo "${SLURM_SCRIPT}"
    exit 1
fi


if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "ERROR: base config not found:"
    echo "${BASE_CONFIG}"
    exit 1
fi


# Required patches for this experiment.

for MARKER in \
    "BOUNDED_FLUX_PROFILE_RUNTIME_PATCH_V1" \
    "H1_GLOBAL_DE_DIAGNOSTIC_PATCH_V1" \
    "H1_GLOBAL_DE_DIAGNOSTIC_FIX_V2" \
    "H1_GLOBAL_DE_MATCH_H1_BOUNDS_FIX_V3" \
    "H1_GLOBAL_DE_STRICT_STOPPING_FIX_V4"
do

    if ! grep -q "${MARKER}" "${RUNNER_PATH}"; then

        echo "ERROR: required runner marker missing:"
        echo "    ${MARKER}"

        exit 1

    fi

done


python -m py_compile "${RUNNER_PATH}"


# =============================================================================
# SUBMIT
# =============================================================================

if [[ "${MODE}" == "submit" ]]; then


    STUDY_TAG="r50_boundary_sensitivity_$(date -u +%Y%m%dT%H%M%SZ)"

    STUDY_DIR="${STUDY_ROOT}/${STUDY_TAG}"

    mkdir -p \
        "${STUDY_DIR}" \
        "${PROJECT_DIR}/slurm_logs"


    echo "${STUDY_TAG}" > "${LATEST_FILE}"


    MANIFEST="${STUDY_DIR}/manifest.tsv"


    printf \
        "scenario\tjob_id\trun_name\trun_tag\tconfig_path\n" \
        > "${MANIFEST}"


    # =====================================================================
    # Scenario definitions
    #
    # Columns:
    #
    #   scenario
    #   t0 half-width [days]
    #   u0 half-width
    #   rho lower bound
    #   piEE half-width
    #
    # piEN is intentionally unchanged because the current best solution
    # is not especially close to its boundary.
    #
    # tE is intentionally unchanged for the same reason.
    # =====================================================================

    SCENARIOS=(
        "B0_current|30.0|1.0|1e-7|2.201096"
        "B1_expand2|60.0|2.0|1e-9|4.402192"
        "B2_expand4|120.0|4.0|1e-11|8.804384"
    )


    echo
    echo "======================================================================"
    echo "R50 BOUNDARY SENSITIVITY"
    echo "======================================================================"
    echo
    echo "Study tag:"
    echo "    ${STUDY_TAG}"
    echo
    echo "Manifest:"
    echo "    ${MANIFEST}"
    echo


    for SPEC in "${SCENARIOS[@]}"; do


        IFS='|' read -r \
            SCENARIO \
            T0_HALF_WIDTH \
            U0_HALF_WIDTH \
            RHO_LOWER \
            PIEE_HALF_WIDTH \
            <<< "${SPEC}"


        RUN_NAME="LSSTMONTS_xi_baseline_v5p3p5_hiddenParallax_truthInit_truthFSPLparallax_multiFit_LRT_smallConservativeBounds_T8_xscaleJac_noiseMC_nestedBounds_H1boundarySens_r50_${SCENARIO}"

        RUN_TAG="${STUDY_TAG}_${SCENARIO}"


        FROZEN_DIR="${OUTPUT_ROOT}/production_configs/${RUN_TAG}"

        mkdir -p "${FROZEN_DIR}"


        CFG_PATH="${FROZEN_DIR}/config.json"


        # -----------------------------------------------------------------
        # Generate scenario config.
        # -----------------------------------------------------------------

        python - \
            "${BASE_CONFIG}" \
            "${CFG_PATH}" \
            "${RUN_NAME}" \
            "${SCENARIO}" \
            "${T0_HALF_WIDTH}" \
            "${U0_HALF_WIDTH}" \
            "${RHO_LOWER}" \
            "${PIEE_HALF_WIDTH}" \
        <<'PY'
import json
import sys
from pathlib import Path


(
    src_name,
    dst_name,
    run_name,
    scenario,
    t0_half_width,
    u0_half_width,
    rho_lower,
    piee_half_width,
) = sys.argv[1:]


src = Path(src_name)
dst = Path(dst_name)


cfg = json.loads(
    src.read_text()
)


t0_half_width = float(t0_half_width)
u0_half_width = float(u0_half_width)
rho_lower = float(rho_lower)
piee_half_width = float(piee_half_width)


# -------------------------------------------------------------------------
# Unique output location.
# -------------------------------------------------------------------------

cfg["run_name"] = run_name


# -------------------------------------------------------------------------
# R50_BOUNDARY_FIT_SPECS_LOCATION_FIX_V1
#
# Depending on the config generation, fit_specs may live either at the
# top level or inside cfg["fit"]. Resolve it once and use the actual object.
# -------------------------------------------------------------------------

# R50_BOUNDARY_FIT_FITS_LOCATION_FIX_V2
#
# Config variants seen in this repository:
#
#   cfg["fit_specs"]
#   cfg["fit"]["fit_specs"]
#   cfg["fit"]["fits"]
#
# The current production config uses cfg["fit"]["fits"].

fit_section = cfg.get(
    "fit",
    {},
)

fit_specs = cfg.get(
    "fit_specs",
    None,
)

if fit_specs is None:
    fit_specs = fit_section.get(
        "fit_specs",
        None,
    )

if fit_specs is None:
    fit_specs = fit_section.get(
        "fits",
        None,
    )

if not isinstance(fit_specs, dict):
    raise RuntimeError(
        "Could not locate H0/H1 fit definitions. "
        f"Top-level keys={list(cfg.keys())}; "
        f"fit keys={list(fit_section.keys())}"
    )

if "H0" not in fit_specs or "H1" not in fit_specs:
    raise RuntimeError(
        f"fit_specs must contain H0 and H1; got {list(fit_specs.keys())}"
    )


# -------------------------------------------------------------------------
# H0 MUST REMAIN UNCHANGED.
# -------------------------------------------------------------------------

h0_before = json.dumps(
    fit_specs["H0"],
    sort_keys=True,
)


# -------------------------------------------------------------------------
# Modify ONLY the H1 bounds under study.
# -------------------------------------------------------------------------

h1_bounds = fit_specs["H1"]["bounds"]


h1_bounds["t0"] = {
    "type": "center_width",
    "half_width": t0_half_width,
}


h1_bounds["u0"] = {
    "type": "center_width",
    "half_width": u0_half_width,
}


# Preserve the original relative rho upper-side policy, changing only
# the absolute lower floor that the current solution is hitting.
#
# Current truth rho ~= 4.987e-4 and frac=1 gives upper ~= 9.974e-4.

rho_cfg = dict(
    h1_bounds["rho"]
)

rho_cfg["lower"] = rho_lower

# Avoid a min_width larger than the new floor influencing the lower side.
rho_cfg["min_width"] = min(
    float(rho_cfg.get("min_width", rho_lower)),
    rho_lower,
)

h1_bounds["rho"] = rho_cfg


h1_bounds["piEE"] = {
    "type": "center_width",
    "center": 0.0,
    "half_width": piee_half_width,
}


# piEN deliberately unchanged.
# tE deliberately unchanged.


# -------------------------------------------------------------------------
# Global-search settings.
#
# One local center fit is sufficient to construct H1/model/bounds.
# The actual global diagnostic is DE -> TRF.
# -------------------------------------------------------------------------

ms = cfg["fit"]["h1_multistart"]


ms["enabled"] = True

ms["piE_grid_fractions"] = [
    0.0,
]

ms["include_auto_center"] = True
ms["include_exact_H0_center"] = False

ms["polish_winner"] = False
ms["diagnostic_polish_top_k"] = 0


ms["diagnostic_global_de"] = {
    "enabled": True,

    # 6 nonlinear dimensions.
    # scipy/Sobol will generally use 64 individuals.
    "population_size": 10,

    "max_iteration": 1500,

    # Strict convergence, unlike pyLIMA's original atol=1.
    "atol": 1.0e-4,
    "tol": 0.0,

    "strategy": "rand1bin",

    # Same algorithmic seeds in every boundary scenario.
    "seeds": [
        20260903,
        20260904,
    ],

    "trf_polish_optimizer_options": {
        "xtol": 1.0e-10,
        "ftol": 1.0e-10,
        "gtol": 1.0e-8,
        "max_nfev": 50000,
        "x_scale": "jac",
    },
}


# -------------------------------------------------------------------------
# QC: H0 must truly be unchanged.
# -------------------------------------------------------------------------

h0_after = json.dumps(
    fit_specs["H0"],
    sort_keys=True,
)

if h0_after != h0_before:
    raise RuntimeError(
        "Boundary-sensitivity generator modified H0."
    )


# Add explicit study metadata.

cfg["boundary_sensitivity_study"] = {
    "scenario": scenario,
    "logical_task": 101,
    "truth_case": "H1",
    "noise_realization_id": 50,

    "changed_H1_bounds": {
        "t0_half_width": t0_half_width,
        "u0_half_width": u0_half_width,
        "rho_lower": rho_lower,
        "piEE_half_width": piee_half_width,
    },

    "unchanged_H1_bounds": [
        "tE",
        "piEN",
    ],
}


dst.write_text(
    json.dumps(
        cfg,
        indent=2,
    )
    + "\n"
)


print(
    f"[config] {scenario}: "
    f"t0_hw={t0_half_width} "
    f"u0_hw={u0_half_width} "
    f"rho_lower={rho_lower} "
    f"piEE_hw={piee_half_width}"
)

print(
    "[config] DE =",
    ms["diagnostic_global_de"],
)
PY


        sha256sum \
            "${CFG_PATH}" \
            > "${CFG_PATH}.SHA256"


        # -----------------------------------------------------------------
        # Submit r50 H1 only.
        #
        # logical mapping:
        #   H0 r50 = 100
        #   H1 r50 = 101
        # -----------------------------------------------------------------

        export HIDDEN_PARALLAX_BOUNDED_PROFILE=1


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


        # sbatch --parsable may return e.g. 12345;cluster
        JOB_ID="${JOB_ID%%;*}"


        printf \
            "%s\t%s\t%s\t%s\t%s\n" \
            "${SCENARIO}" \
            "${JOB_ID}" \
            "${RUN_NAME}" \
            "${RUN_TAG}" \
            "${CFG_PATH}" \
            >> "${MANIFEST}"


        echo
        echo "Submitted:"
        echo "    scenario = ${SCENARIO}"
        echo "    job      = ${JOB_ID}"
        echo "    config   = ${CFG_PATH}"


    done


    echo
    echo "======================================================================"
    echo "SUBMITTED ALL SCENARIOS"
    echo "======================================================================"
    echo

    column -t -s $'\t' "${MANIFEST}" || cat "${MANIFEST}"

    echo
    echo "Monitor:"
    echo
    echo "    while read -r s j rest; do sacct -j \"\$j\" --format=JobID,State,Elapsed,TotalCPU,MaxRSS,ExitCode; done < <(tail -n +2 \"${MANIFEST}\")"
    echo
    echo "After completion:"
    echo
    echo "    ./validation/optimizer/r50_boundary_sensitivity.sh analyze"
    echo

    exit 0

fi


# =============================================================================
# ANALYZE
# =============================================================================

if [[ "${MODE}" == "analyze" ]]; then


    if [[ ! -f "${LATEST_FILE}" ]]; then

        echo "ERROR: no previous boundary study found:"
        echo "${LATEST_FILE}"

        exit 1

    fi


    STUDY_TAG="$(
        cat "${LATEST_FILE}"
    )"


    STUDY_DIR="${STUDY_ROOT}/${STUDY_TAG}"

    MANIFEST="${STUDY_DIR}/manifest.tsv"


    if [[ ! -f "${MANIFEST}" ]]; then

        echo "ERROR: manifest missing:"
        echo "${MANIFEST}"

        exit 1

    fi


    RESULTS_CSV="${STUDY_DIR}/boundary_sensitivity_results.csv"

    PARAMETERS_CSV="${STUDY_DIR}/boundary_sensitivity_parameters.csv"


    python - \
        "${MANIFEST}" \
        "${OUTPUT_ROOT}" \
        "${RESULTS_CSV}" \
        "${PARAMETERS_CSV}" \
    <<'PY'
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


manifest_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])

results_csv = Path(sys.argv[3])
parameters_csv = Path(sys.argv[4])


manifest = pd.read_csv(
    manifest_path,
    sep="\t",
)


summary_rows = []
parameter_rows = []


def load_npy(path):
    return np.load(
        path,
        allow_pickle=True,
    ).item()


for _, m in manifest.iterrows():

    scenario = str(
        m["scenario"]
    )

    run_name = str(
        m["run_name"]
    )

    run_tag = str(
        m["run_tag"]
    )


    base = (
        output_root
        / "runs"
        / run_name
    )


    candidates = sorted(
        base.glob(
            f"{run_tag}_logical_101_102_w1"
        )
    )


    if len(candidates) != 1:

        summary_rows.append(
            {
                "scenario": scenario,
                "status": "RUN_DIR_NOT_FOUND",
                "run_dir": "",
            }
        )

        continue


    run_dir = candidates[0]


    summary_file = (
        run_dir
        / "logs"
        / "run_summary.parquet"
    )


    if not summary_file.exists():

        summary_rows.append(
            {
                "scenario": scenario,
                "status": "SUMMARY_NOT_FOUND",
                "run_dir": str(run_dir),
            }
        )

        continue


    run_summary = pd.read_parquet(
        summary_file
    )


    event_dirs = list(
        (run_dir / "fits").glob("*/*")
    )


    if len(event_dirs) != 1:

        summary_rows.append(
            {
                "scenario": scenario,
                "status": "EVENT_DIR_ERROR",
                "run_dir": str(run_dir),
            }
        )

        continue


    event_dir = event_dirs[0]


    h0_files = list(
        event_dir.glob(
            "*TRF_FSPL_NoParallax.npy"
        )
    )


    h1_normal_files = list(
        event_dir.glob(
            "*TRF_FSPL_Parallax.npy"
        )
    )


    if len(h0_files) != 1:

        summary_rows.append(
            {
                "scenario": scenario,
                "status": "H0_MISSING",
                "run_dir": str(run_dir),
            }
        )

        continue


    h0 = load_npy(
        h0_files[0]
    )

    chi2_h0 = float(
        h0["chi2"]
    )


    chi2_h1_normal = np.nan

    if len(h1_normal_files) == 1:

        normal_h1 = load_npy(
            h1_normal_files[0]
        )

        chi2_h1_normal = float(
            normal_h1["chi2"]
        )


    de_files = sorted(
        event_dir.glob(
            "_H1_global_DE/seed_*/de_raw.npy"
        )
    )


    seed_records = []


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


        chi2_polish = float(
            polish["chi2"]
        )


        seed = int(
            de["seed"]
        )


        order = list(
            de["parameter_order"]
        )

        bounds = de[
            "resolved_H1_bounds"
        ]


        best = np.asarray(
            polish["best_model"],
            dtype=float,
        ).reshape(-1)


        best = best[
            :len(order)
        ]


        seed_records.append(
            {
                "seed": seed,
                "chi2_de": float(
                    de["chi2"]
                ),
                "chi2_polish": chi2_polish,
                "nit": de.get(
                    "nit",
                    np.nan,
                ),
                "nfev": de.get(
                    "nfev",
                    np.nan,
                ),
                "fit_time": de.get(
                    "fit_time",
                    np.nan,
                ),
                "success": de.get(
                    "success",
                    np.nan,
                ),
                "message": de.get(
                    "message",
                    "",
                ),
                "order": order,
                "bounds": bounds,
                "best": best,
            }
        )


    if len(seed_records) == 0:

        summary_rows.append(
            {
                "scenario": scenario,
                "status": "NO_DE_POLISH",
                "run_dir": str(run_dir),
                "chi2_H0": chi2_h0,
                "chi2_H1_normal": chi2_h1_normal,
            }
        )

        continue


    winner = min(
        seed_records,
        key=lambda x: x[
            "chi2_polish"
        ],
    )


    chi2_best = float(
        winner[
            "chi2_polish"
        ]
    )


    T = (
        chi2_h0
        - chi2_best
    )


    min_edge = np.inf


    for name, value in zip(
        winner["order"],
        winner["best"],
    ):

        lo, hi = winner[
            "bounds"
        ][name]

        lo = float(lo)
        hi = float(hi)

        width = (
            hi
            - lo
        )


        left_fraction = (
            value
            - lo
        ) / width


        right_fraction = (
            hi
            - value
        ) / width


        edge_fraction = min(
            left_fraction,
            right_fraction,
        )


        min_edge = min(
            min_edge,
            edge_fraction,
        )


        parameter_rows.append(
            {
                "scenario": scenario,
                "seed": winner["seed"],
                "parameter": name,
                "value": float(value),
                "lower": lo,
                "upper": hi,
                "left_fraction": float(
                    left_fraction
                ),
                "right_fraction": float(
                    right_fraction
                ),
                "edge_fraction": float(
                    edge_fraction
                ),
            }
        )


    summary_rows.append(
        {
            "scenario": scenario,
            "status": str(
                run_summary.iloc[0][
                    "status"
                ]
            ),
            "run_dir": str(
                run_dir
            ),

            "chi2_H0": chi2_h0,

            "chi2_H1_normal": (
                chi2_h1_normal
            ),

            "best_seed": int(
                winner["seed"]
            ),

            "chi2_DE": float(
                winner["chi2_de"]
            ),

            "chi2_DE_TRF": (
                chi2_best
            ),

            "T_H0_minus_H1": float(
                T
            ),

            "DE_nit": winner[
                "nit"
            ],

            "DE_nfev": winner[
                "nfev"
            ],

            "DE_fit_time_s": winner[
                "fit_time"
            ],

            "minimum_edge_fraction": float(
                min_edge
            ),
        }
    )


results = pd.DataFrame(
    summary_rows
)

parameters = pd.DataFrame(
    parameter_rows
)


results.to_csv(
    results_csv,
    index=False,
)

parameters.to_csv(
    parameters_csv,
    index=False,
)


print()
print("=" * 110)
print("BOUNDARY SENSITIVITY — SUMMARY")
print("=" * 110)

display_columns = [
    "scenario",
    "status",
    "chi2_H0",
    "chi2_DE_TRF",
    "T_H0_minus_H1",
    "best_seed",
    "DE_nit",
    "DE_nfev",
    "minimum_edge_fraction",
]

available = [
    c
    for c in display_columns
    if c in results.columns
]

print(
    results[
        available
    ].to_string(
        index=False
    )
)


if not parameters.empty:

    print()
    print("=" * 110)
    print("WINNING PARAMETERS / BOUNDARY DISTANCES")
    print("=" * 110)

    table = parameters.pivot(
        index="scenario",
        columns="parameter",
        values="edge_fraction",
    )

    print()
    print("edge_fraction:")
    print(
        table.to_string()
    )


# -------------------------------------------------------------------------
# Useful stability diagnostics.
# -------------------------------------------------------------------------

if (
    "chi2_H0"
    in results.columns
    and results[
        "chi2_H0"
    ].notna().any()
):

    h0_values = results[
        "chi2_H0"
    ].dropna()

    print()
    print(
        "H0 chi2 spread across scenarios =",
        float(
            h0_values.max()
            - h0_values.min()
        ),
    )


if (
    "chi2_DE_TRF"
    in results.columns
):

    fitted = results.dropna(
        subset=[
            "chi2_DE_TRF"
        ]
    )

    if len(fitted) >= 2:

        first = float(
            fitted.iloc[0][
                "chi2_DE_TRF"
            ]
        )

        print()
        print(
            "Delta chi2 relative to first scenario:"
        )

        for _, row in fitted.iterrows():

            print(
                f"  {row['scenario']:15s}: "
                f"{float(row['chi2_DE_TRF']) - first:+.9f}"
            )


print()
print("Saved:")
print(" ", results_csv)
print(" ", parameters_csv)
PY


    echo
    echo "Analysis complete:"
    echo "    ${RESULTS_CSV}"
    echo "    ${PARAMETERS_CSV}"
    echo

    exit 0

fi


# =============================================================================
# Usage
# =============================================================================

echo "Usage:"
echo
echo "    conda activate pyLIMA_test"
echo
echo "    $0 submit"
echo
echo "or:"
echo
echo "    $0 analyze"
echo

exit 2
