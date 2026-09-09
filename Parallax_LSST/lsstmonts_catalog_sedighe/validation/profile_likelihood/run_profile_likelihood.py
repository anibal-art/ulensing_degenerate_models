from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import copy
import gc
import json
import sys
import traceback

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "~/Downloads/hidden_parallax/profile_likelihood_case_study"
).expanduser().resolve()

RR = Path(
    "/home/anibal-pc/microlensing/"
    "simulation_Rubin/roman_rubin"
)

HP = Path(
    "/home/anibal-pc/ulensing_degenerate_models/"
    "Parallax_LSST/lsstmonts_catalog_sedighe"
)

CONFIG_PATH = (
    HP
    / "configs"
    / "config_lsstmonts_baseline_v5p3p5_cluster_che_multifit_LRT_H1ONLY.json"
)

EPHEMERIDES = (
    RR
    / "ephemerides"
    / "Roman_positions.npy"
)

OUT = ROOT / "profile_results"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(RR))
sys.path.insert(0, str(HP))

import fit_lc
import pyLIMA


# ============================================================
# CONFIGURATION
# ============================================================

CASES = [
    63218,
    72168,
]

# s = 0   -> H1 winner
# s = 1   -> truth piE
#
# Extending on both sides helps reveal the geometry of the valley.
S_GRID = np.round(
    np.arange(
        -0.25,
        1.250001,
        0.05,
    ),
    10,
)

# Effectively fix piEN/piEE while satisfying scipy lb < ub.
PIE_FIXED_HALF_WIDTH = 1.0e-7

SANITY_CHI2_TOL = 0.05

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

H1_BOUNDS_BASE = copy.deepcopy(
    CONFIG["fit"]["fits"]["H1"]["bounds"]
)

OPTIMIZER_OPTIONS = copy.deepcopy(
    CONFIG["fit"]["optimizer_options"]
)


# ============================================================
# BASIC HELPERS
# ============================================================

def load_npy(path):
    obj = np.load(
        path,
        allow_pickle=True,
    )

    if (
        isinstance(obj, np.ndarray)
        and obj.shape == ()
    ):
        obj = obj.item()

    return obj


def empty_lc():
    return np.empty(
        (0, 3),
        dtype=float,
    )


def load_h5_lightcurves(path):
    """
    Reconstruct the exact noisy photometry stored in the production
    artifact.

    Each returned light curve has columns:
        time, mag, err_mag

    The production photometric mask is respected.
    """

    curves = {
        "W149": empty_lc(),
        "u": empty_lc(),
        "g": empty_lc(),
        "r": empty_lc(),
        "i": empty_lc(),
        "z": empty_lc(),
        "y": empty_lc(),
    }

    with h5py.File(
        path,
        "r",
    ) as f:

        for band in [
            "W149",
            "u",
            "g",
            "r",
            "i",
            "z",
            "y",
        ]:

            if band not in f:
                continue

            group = f[band]

            time = np.asarray(
                group["time"],
                dtype=float,
            )

            mag = np.asarray(
                group["mag"],
                dtype=float,
            )

            err_mag = np.asarray(
                group["err_mag"],
                dtype=float,
            )

            if "photometry_keep" in group:
                keep = np.asarray(
                    group["photometry_keep"],
                    dtype=bool,
                )
            else:
                keep = np.ones(
                    len(time),
                    dtype=bool,
                )

            valid = (
                keep
                & np.isfinite(time)
                & np.isfinite(mag)
                & np.isfinite(err_mag)
                & (err_mag > 0)
            )

            curves[band] = np.column_stack(
                [
                    time[valid],
                    mag[valid],
                    err_mag[valid],
                ]
            )

    return curves


def nuisance_from_best_model(best):
    return {
        "t0": float(best[0]),
        "u0": float(best[1]),
        "tE": float(best[2]),
        "rho": float(best[3]),
    }


def make_initial_guess(
    nuisance,
    piEN,
    piEE,
):
    return {
        "t0": float(nuisance["t0"]),
        "u0": float(nuisance["u0"]),
        "tE": float(nuisance["tE"]),
        "rho": float(nuisance["rho"]),
        "piEN": float(piEN),
        "piEE": float(piEE),
    }


# ============================================================
# LOAD CASE
# ============================================================

def load_case(catalog_row):

    case_dir = ROOT / str(catalog_row)

    h5_files = list(
        case_dir.rglob("Event_*.h5")
    )

    h1_files = [
        p
        for p in case_dir.rglob(
            "*TRF_FSPL_Parallax.npy"
        )
        if "_H1_multistart" not in str(p)
    ]

    true_files = list(
        case_dir.rglob(
            "true_rr_manual_*.parquet"
        )
    )

    multi_files = list(
        case_dir.rglob(
            "multi_fit_manual_*.parquet"
        )
    )

    if len(h5_files) != 1:
        raise RuntimeError(
            f"{catalog_row}: expected one H5, "
            f"found {len(h5_files)}"
        )

    if len(h1_files) != 1:
        raise RuntimeError(
            f"{catalog_row}: expected one final H1 NPY, "
            f"found {len(h1_files)}"
        )

    if len(true_files) != 1:
        raise RuntimeError(
            f"{catalog_row}: expected one truth parquet, "
            f"found {len(true_files)}"
        )

    if len(multi_files) != 1:
        raise RuntimeError(
            f"{catalog_row}: expected one multi-fit parquet, "
            f"found {len(multi_files)}"
        )

    final = load_npy(
        h1_files[0]
    )

    truth_df = pd.read_parquet(
        true_files[0]
    )

    multi_df = pd.read_parquet(
        multi_files[0]
    )

    truth = truth_df.iloc[0]
    multi = multi_df.iloc[0]

    best = np.asarray(
        final["best_model"],
        dtype=float,
    )

    cov = np.asarray(
        final["covariance_matrix"],
        dtype=float,
    )

    curves = load_h5_lightcurves(
        h5_files[0]
    )

    n_lc = sum(
        len(curves[b])
        for b in [
            "W149",
            "u",
            "g",
            "r",
            "i",
            "z",
            "y",
        ]
    )

    stored_n = int(
        multi["H1_n_data"]
    )

    if n_lc != stored_n:
        raise RuntimeError(
            f"{catalog_row}: H5 gives {n_lc} fit points "
            f"but production H1_n_data={stored_n}."
        )

    true_params = final.get(
        "true_params",
        None,
    )

    if true_params is None:
        raise RuntimeError(
            f"{catalog_row}: final NPY has no true_params."
        )

    return {
        "catalog_row": catalog_row,
        "case_dir": case_dir,
        "h5": h5_files[0],
        "final_path": h1_files[0],
        "final": final,
        "truth": truth,
        "multi": multi,
        "best": best,
        "cov": cov,
        "curves": curves,
        "true_params": true_params,
        "source": int(truth["Source"]),
        "stored_chi2": float(final["chi2"]),
        "chi2_red": float(multi["H1_chi2_red"]),
        "n_data": stored_n,
        "rango": int(final.get("rango", 1)),
        "event_ra": float(
            final.get(
                "event_ra",
                truth["ra"],
            )
        ),
        "event_dec": float(
            final.get(
                "event_dec",
                truth["dec"],
            )
        ),
    }


# ============================================================
# PROFILE GEOMETRY
# ============================================================

def piE_at_s(meta, s):

    best = meta["best"]
    truth = meta["truth"]

    best_piEN = float(best[4])
    best_piEE = float(best[5])

    true_piEN = float(
        truth["piEN"]
    )

    true_piEE = float(
        truth["piEE"]
    )

    piEN = (
        best_piEN
        + s
        * (
            true_piEN
            - best_piEN
        )
    )

    piEE = (
        best_piEE
        + s
        * (
            true_piEE
            - best_piEE
        )
    )

    return piEN, piEE


def build_profile_bounds(
    piEN,
    piEE,
):

    bounds = copy.deepcopy(
        H1_BOUNDS_BASE
    )

    bounds["piEN"] = {
        "type": "center_width",
        "center": float(piEN),
        "half_width":
            float(PIE_FIXED_HALF_WIDTH),
    }

    bounds["piEE"] = {
        "type": "center_width",
        "center": float(piEE),
        "half_width":
            float(PIE_FIXED_HALF_WIDTH),
    }

    return bounds


# ============================================================
# ONE PROFILE FIT
# ============================================================

def run_profile_fit(
    meta,
    s,
    branch,
    nuisance_start,
):

    piEN, piEE = piE_at_s(
        meta,
        s,
    )

    bounds = build_profile_bounds(
        piEN,
        piEE,
    )

    initial_guess = make_initial_guess(
        nuisance_start,
        piEN,
        piEE,
    )

    tag = (
        f"s_{s:+.2f}"
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )

    fit_dir = (
        OUT
        / str(meta["catalog_row"])
        / "fits"
        / branch
        / tag
    )

    fit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        fit_dir
        / "stdout_stderr.log"
    )

    print(
        f"  {branch:14s}"
        f" s={s:+.2f}"
        f" piE=({piEN:+.6f},{piEE:+.6f})",
        flush=True,
    )

    try:

        with open(
            log_path,
            "w",
        ) as log:

            with redirect_stdout(log), redirect_stderr(log):

                fit, event_fit, model_fit = (
                    fit_lc.fit_rubin_roman(
                        Source=meta["source"],
                        event_params=meta["true_params"],
                        path_save=str(fit_dir),
                        path_ephemerides=str(
                            EPHEMERIDES
                        ),
                        model="FSPL",
                        algo="TRF",
                        Origin=None,
                        rango=meta["rango"],

                        wfirst_lc=
                            meta["curves"]["W149"],

                        lsst_u=
                            meta["curves"]["u"],

                        lsst_g=
                            meta["curves"]["g"],

                        lsst_r=
                            meta["curves"]["r"],

                        lsst_i=
                            meta["curves"]["i"],

                        lsst_z=
                            meta["curves"]["z"],

                        lsst_y=
                            meta["curves"]["y"],

                        fit_model="FSPL",
                        fit_parallax=True,

                        fit_defaults=None,
                        fit_bounds=bounds,

                        random_state=123456,

                        initial_guess=
                            initial_guess,

                        optimizer_options=
                            OPTIMIZER_OPTIONS,

                        event_ra=
                            meta["event_ra"],

                        event_dec=
                            meta["event_dec"],
                    )
                )

        results = fit.fit_results

        chi2_value = float(
            results["chi2"]
        )

        best = np.asarray(
            results["best_model"],
            dtype=float,
        )

        if len(best) < 6:
            raise RuntimeError(
                "Returned best_model has fewer than 6 parameters."
            )

        record = {
            "catalog_row":
                meta["catalog_row"],

            "branch":
                branch,

            "s":
                float(s),

            "piEN_fixed":
                float(piEN),

            "piEE_fixed":
                float(piEE),

            "chi2":
                chi2_value,

            "t0_fit":
                float(best[0]),

            "u0_fit":
                float(best[1]),

            "tE_fit":
                float(best[2]),

            "rho_fit":
                float(best[3]),

            "piEN_fit":
                float(best[4]),

            "piEE_fit":
                float(best[5]),

            "optimizer_success":
                results.get(
                    "optimizer_success",
                    np.nan,
                ),

            "optimizer_optimality":
                results.get(
                    "optimizer_optimality",
                    np.nan,
                ),

            "optimizer_nfev":
                results.get(
                    "optimizer_nfev",
                    np.nan,
                ),

            "optimizer_active_mask":
                results.get(
                    "optimizer_active_mask",
                    None,
                ),

            "log_path":
                str(log_path),
        }

        print(
            f"      chi2={chi2_value:.9f}"
            f"  tE={best[2]:.5f}"
            f"  rho={best[3]:.6g}",
            flush=True,
        )

        del fit
        del event_fit
        del model_fit
        gc.collect()

        return record

    except Exception:

        with open(
            log_path,
            "a",
        ) as log:

            log.write(
                "\n\nPROFILE FIT EXCEPTION\n"
            )

            traceback.print_exc(
                file=log,
            )

        print(
            "      ERROR - see",
            log_path,
            flush=True,
        )

        raise


# ============================================================
# DIRECTIONAL BRANCH
# ============================================================

def run_branch(
    meta,
    branch,
    anchor_s,
    anchor_nuisance,
):

    records = {}

    # --------------------------------------------------------
    # Anchor
    # --------------------------------------------------------

    anchor = run_profile_fit(
        meta=meta,
        s=anchor_s,
        branch=branch,
        nuisance_start=
            anchor_nuisance,
    )

    records[
        float(anchor_s)
    ] = anchor

    anchor_solution = {
        "t0": anchor["t0_fit"],
        "u0": anchor["u0_fit"],
        "tE": anchor["tE_fit"],
        "rho": anchor["rho_fit"],
    }

    # --------------------------------------------------------
    # Walk downwards
    # --------------------------------------------------------

    current = dict(
        anchor_solution
    )

    lower = sorted(
        [
            float(s)
            for s in S_GRID
            if s < anchor_s
        ],
        reverse=True,
    )

    for s in lower:

        result = run_profile_fit(
            meta=meta,
            s=s,
            branch=branch,
            nuisance_start=current,
        )

        records[s] = result

        current = {
            "t0": result["t0_fit"],
            "u0": result["u0_fit"],
            "tE": result["tE_fit"],
            "rho": result["rho_fit"],
        }

    # --------------------------------------------------------
    # Walk upwards independently from anchor
    # --------------------------------------------------------

    current = dict(
        anchor_solution
    )

    upper = sorted(
        [
            float(s)
            for s in S_GRID
            if s > anchor_s
        ]
    )

    for s in upper:

        result = run_profile_fit(
            meta=meta,
            s=s,
            branch=branch,
            nuisance_start=current,
        )

        records[s] = result

        current = {
            "t0": result["t0_fit"],
            "u0": result["u0_fit"],
            "tE": result["tE_fit"],
            "rho": result["rho_fit"],
        }

    return records


# ============================================================
# QUADRATIC PREDICTION
# ============================================================

def local_quadratic_d2(meta):

    Cpi = meta["cov"][
        np.ix_(
            [4, 5],
            [4, 5],
        )
    ]

    delta = np.array(
        [
            float(meta["truth"]["piEN"])
            - float(meta["best"][4]),

            float(meta["truth"]["piEE"])
            - float(meta["best"][5]),
        ],
        dtype=float,
    )

    D2 = float(
        delta
        @ np.linalg.inv(Cpi)
        @ delta
    )

    return D2, Cpi


# ============================================================
# COMBINE BRANCHES
# ============================================================

def combine_profiles(
    meta,
    winner_branch,
    truth_branch,
):

    D2, Cpi = local_quadratic_d2(
        meta
    )

    rows = []

    for s in S_GRID:

        s = float(s)

        a = winner_branch[s]
        b = truth_branch[s]

        if a["chi2"] <= b["chi2"]:
            chosen = a
        else:
            chosen = b

        rows.append(
            {
                "catalog_row":
                    meta["catalog_row"],

                "s":
                    s,

                "piEN":
                    chosen["piEN_fixed"],

                "piEE":
                    chosen["piEE_fixed"],

                "chi2_winner_branch":
                    a["chi2"],

                "chi2_truth_branch":
                    b["chi2"],

                "chi2_profile":
                    chosen["chi2"],

                "selected_branch":
                    chosen["branch"],

                "delta_chi2_profile_from_stored_H1":
                    (
                        chosen["chi2"]
                        - meta["stored_chi2"]
                    ),

                "delta_chi2_winner_branch":
                    (
                        a["chi2"]
                        - meta["stored_chi2"]
                    ),

                "delta_chi2_truth_branch":
                    (
                        b["chi2"]
                        - meta["stored_chi2"]
                    ),

                "delta_chi2_quadratic":
                    (
                        s**2
                        * D2
                        * meta["chi2_red"]
                    ),

                "t0_profile":
                    chosen["t0_fit"],

                "u0_profile":
                    chosen["u0_fit"],

                "tE_profile":
                    chosen["tE_fit"],

                "rho_profile":
                    chosen["rho_fit"],
            }
        )

    df = pd.DataFrame(
        rows
    )

    profile_min = float(
        df["chi2_profile"].min()
    )

    df[
        "delta_chi2_profile_from_profile_min"
    ] = (
        df["chi2_profile"]
        - profile_min
    )

    return df, D2, Cpi


# ============================================================
# PLOTS
# ============================================================

def make_plots(
    meta,
    df,
    D2,
):

    row = meta["catalog_row"]

    # --------------------------------------------------------
    # Full scale
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.plot(
        df["s"],
        df[
            "delta_chi2_winner_branch"
        ],
        linestyle="--",
        linewidth=1,
        label="winner-start branch",
    )

    ax.plot(
        df["s"],
        df[
            "delta_chi2_truth_branch"
        ],
        linestyle=":",
        linewidth=1,
        label="truth-start branch",
    )

    ax.plot(
        df["s"],
        df[
            "delta_chi2_profile_from_stored_H1"
        ],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="profile = min(branches)",
    )

    ax.plot(
        df["s"],
        df[
            "delta_chi2_quadratic"
        ],
        linewidth=1.5,
        label="local covariance quadratic",
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.axhline(
        0.0,
        linewidth=0.8,
    )

    ax.set_xlabel(
        r"$s$  (0 = H1 winner, 1 = truth)"
    )

    ax.set_ylabel(
        r"$\Delta\chi^2$ relative to stored H1 winner"
    )

    ax.set_title(
        f"Profile likelihood — catalog_row={row}\n"
        f"D2={D2:.2f}, "
        f"chi2_red={meta['chi2_red']:.4f}"
    )

    ax.grid(alpha=0.3)
    ax.legend(
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / str(row)
        / f"profile_full_{row}.png",
        dpi=220,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Zoom near the competitive region
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.plot(
        df["s"],
        df[
            "delta_chi2_winner_branch"
        ],
        linestyle="--",
        linewidth=1,
        label="winner-start branch",
    )

    ax.plot(
        df["s"],
        df[
            "delta_chi2_truth_branch"
        ],
        linestyle=":",
        linewidth=1,
        label="truth-start branch",
    )

    ax.plot(
        df["s"],
        df[
            "delta_chi2_profile_from_stored_H1"
        ],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="profile = min(branches)",
    )

    ax.plot(
        df["s"],
        df[
            "delta_chi2_quadratic"
        ],
        linewidth=1.5,
        label="local covariance quadratic",
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.axhline(
        0.0,
        linewidth=0.8,
    )

    low = min(
        -0.5,
        float(
            df[
                "delta_chi2_profile_from_stored_H1"
            ].min()
        ) - 0.2,
    )

    ax.set_ylim(
        low,
        25,
    )

    ax.set_xlabel(
        r"$s$  (0 = H1 winner, 1 = truth)"
    )

    ax.set_ylabel(
        r"$\Delta\chi^2$ relative to stored H1 winner"
    )

    ax.set_title(
        f"Profile likelihood zoom — catalog_row={row}"
    )

    ax.grid(alpha=0.3)
    ax.legend(
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / str(row)
        / f"profile_zoom_{row}.png",
        dpi=220,
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

print("=" * 100)
print("PROFILE LIKELIHOOD TEST")
print("=" * 100)

print("ROOT       =", ROOT)
print("RR         =", RR)
print("CONFIG     =", CONFIG_PATH)
print("EPHEMERIS  =", EPHEMERIDES)

print()
print("Python     =", sys.executable)
print("pyLIMA     =", pyLIMA.__file__)
print()

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        CONFIG_PATH
    )

if not EPHEMERIDES.exists():
    raise FileNotFoundError(
        EPHEMERIDES
    )

print(
    "TRF options =",
    OPTIMIZER_OPTIONS,
)

print(
    "s grid =",
    S_GRID,
)

print()


for catalog_row in CASES:

    print("\n" + "#" * 100)
    print(
        "CASE",
        catalog_row,
    )
    print("#" * 100)

    meta = load_case(
        catalog_row
    )

    print(
        "stored H1 chi2 =",
        meta["stored_chi2"],
    )

    print(
        "H1 chi2_red    =",
        meta["chi2_red"],
    )

    print(
        "Ndata          =",
        meta["n_data"],
    )

    print(
        "bands          =",
        {
            b: len(meta["curves"][b])
            for b in meta["curves"]
        },
    )

    print(
        "winner piE     =",
        meta["best"][4],
        meta["best"][5],
    )

    print(
        "truth piE      =",
        float(meta["truth"]["piEN"]),
        float(meta["truth"]["piEE"]),
    )

    D2, Cpi = local_quadratic_d2(
        meta
    )

    print(
        "local D2       =",
        D2,
    )

    print(
        "quad at truth  =",
        D2 * meta["chi2_red"],
    )

    # --------------------------------------------------------
    # Winner branch: anchored at s=0
    # --------------------------------------------------------

    print(
        "\nRUNNING WINNER-START BRANCH"
    )

    winner_start = nuisance_from_best_model(
        meta["best"]
    )

    winner_branch = run_branch(
        meta=meta,
        branch="winner_branch",
        anchor_s=0.0,
        anchor_nuisance=
            winner_start,
    )

    reproduced_chi2 = (
        winner_branch[0.0]["chi2"]
    )

    diff = (
        reproduced_chi2
        - meta["stored_chi2"]
    )

    print()
    print(
        "SANITY CHECK s=0"
    )

    print(
        "stored     =",
        meta["stored_chi2"],
    )

    print(
        "reproduced =",
        reproduced_chi2,
    )

    print(
        "difference =",
        diff,
    )

    if abs(diff) > SANITY_CHI2_TOL:

        raise RuntimeError(
            f"CASE {catalog_row}: s=0 does not reproduce "
            f"the production H1 within tolerance. "
            f"Difference={diff:.6g}. "
            "Do not interpret the profile yet."
        )

    # --------------------------------------------------------
    # Truth branch: anchored at s=1
    # --------------------------------------------------------

    print(
        "\nRUNNING TRUTH-START BRANCH"
    )

    truth_start = {
        "t0":
            float(meta["truth"]["t0"]),

        "u0":
            float(meta["truth"]["u0"]),

        "tE":
            float(meta["truth"]["tE"]),

        "rho":
            float(meta["truth"]["rho"]),
    }

    truth_branch = run_branch(
        meta=meta,
        branch="truth_branch",
        anchor_s=1.0,
        anchor_nuisance=
            truth_start,
    )

    # --------------------------------------------------------
    # Save raw branch results
    # --------------------------------------------------------

    case_out = (
        OUT
        / str(catalog_row)
    )

    case_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        winner_branch.values()
    ).sort_values(
        "s"
    ).to_csv(
        case_out
        / "winner_branch.csv",
        index=False,
    )

    pd.DataFrame(
        truth_branch.values()
    ).sort_values(
        "s"
    ).to_csv(
        case_out
        / "truth_branch.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Global profile from both branches
    # --------------------------------------------------------

    df, D2, Cpi = combine_profiles(
        meta,
        winner_branch,
        truth_branch,
    )

    df.to_csv(
        case_out
        / "profile_likelihood.csv",
        index=False,
    )

    make_plots(
        meta,
        df,
        D2,
    )

    # --------------------------------------------------------
    # Compact diagnostic printout
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        f"PROFILE SUMMARY — {catalog_row}"
    )
    print("=" * 100)

    cols = [
        "s",
        "chi2_winner_branch",
        "chi2_truth_branch",
        "chi2_profile",
        "delta_chi2_profile_from_stored_H1",
        "delta_chi2_quadratic",
        "selected_branch",
    ]

    show_s = [
        -0.25,
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
        1.25,
    ]

    print(
        df[
            df["s"].isin(show_s)
        ][cols].to_string(
            index=False
        )
    )

    truth_row = df[
        np.isclose(
            df["s"],
            1.0,
        )
    ].iloc[0]

    print()
    print(
        "At truth piE:"
    )

    print(
        "actual profiled Delta chi2 =",
        truth_row[
            "delta_chi2_profile_from_stored_H1"
        ],
    )

    print(
        "local quadratic prediction =",
        truth_row[
            "delta_chi2_quadratic"
        ],
    )

    if (
        truth_row[
            "delta_chi2_profile_from_stored_H1"
        ] > 0
    ):

        print(
            "quadratic/profile ratio =",
            truth_row[
                "delta_chi2_quadratic"
            ]
            /
            truth_row[
                "delta_chi2_profile_from_stored_H1"
            ],
        )

    print(
        "profile minimum chi2 =",
        df["chi2_profile"].min(),
    )

    print(
        "stored H1 chi2       =",
        meta["stored_chi2"],
    )


print("\n" + "=" * 100)
print("ALL DONE")
print("=" * 100)

print("Results:")
print(OUT)

for p in sorted(
    OUT.rglob(
        "profile_likelihood.csv"
    )
):
    print(p)

for p in sorted(
    OUT.rglob(
        "profile_*.png"
    )
):
    print(p)
