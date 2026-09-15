#!/usr/bin/env python3
import os as _os_path_setup
HERE = _os_path_setup.path.dirname(_os_path_setup.path.abspath(__file__))

import sys
import os
import copy
import time
from pathlib import Path

CORE_DIR = (
    "/home/anibal-pc/ulensing_degenerate_models/"
    "Parallax_LSST/lsstmonts_catalog_sedighe/validation/bounds_audit"
)
sys.path.insert(0, CORE_DIR)
SCRATCH = _os_path_setup.path.join(HERE, "results")
sys.path.insert(0, HERE)

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"

sys.argv = [
    "run_bounds_audit_refit_core.py",
    "--catalog-row", "71181",
    "--bounds-profile", "production_candidate",
    "--fit-scope", "h1",
    "--dry-run",
]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402
from fisher_engine import (  # noqa: E402
    numerical_jacobian, fisher_from_jacobian_physical, fisher_diagnostics,
)

assert core.BOUNDS_PROFILE == "production_candidate"
assert core.FIT_SCOPE == "h1"


def data_driven_t0_bounds(meta):
    all_times = []
    for band in ["u", "g", "r", "i", "z", "y"]:
        lc = meta["curves"][band]
        if len(lc):
            all_times.extend(np.asarray(lc[:, 0], dtype=float).tolist())
    all_times = np.asarray(all_times, dtype=float)
    return float(np.min(all_times)), float(np.max(all_times))


def build_h1_forward_fit(meta):
    t_min, t_max = data_driven_t0_bounds(meta)
    bounds = copy.deepcopy(core.H1_BOUNDS)
    bounds = core.apply_bounds_profile(bounds, h1=True, profile=core.BOUNDS_PROFILE)
    bounds["t0"] = {
        "type": "center_width",
        "center": 0.5 * (t_min + t_max),
        "half_width": 0.5 * (t_max - t_min),
    }
    lsst_lcs = {b: meta["curves"][b] for b in ["u", "g", "r", "i", "z", "y"]}
    event_ra_use, event_dec_use = fit_lc.resolve_event_coordinates(
        event_ra=meta["event_ra"], event_dec=meta["event_dec"],
    )
    e = fit_lc.create_fit_event(
        meta["Source"], str(core.EPHEMERIDES), meta["curves"]["W149"], lsst_lcs,
        ra=event_ra_use, dec=event_dec_use, roman_name="Roman",
    )
    fit_params = fit_lc.initial_params_for_fit_model(
        meta["true_params"], "FSPL", fit_parallax=True, fit_defaults=None,
    )
    pyLIMAmodel = fit_lc.build_fit_pyLIMA_model(
        e, "FSPL", fit_params, Origin=None, fit_parallax=True,
    )
    fit, pool_processes = fit_lc.build_fitter(pyLIMAmodel, "TRF")
    fit_lc.apply_fit_bounds(
        fit, fit_params, "FSPL", meta["rango"], fit_parallax=True, fit_bounds=bounds,
    )
    param_order = fit_lc.fit_parameter_order("FSPL", fit_parallax=True)
    return fit, param_order


ROOT4 = Path("~/Downloads/hidden_parallax/production_validation/gate2_controlled4_extreme100").expanduser()
ref = pd.read_csv(
    "/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/gate2_start_ablation_per_event.csv"
).set_index("catalog_row")

rows_100 = sorted(ref.index.tolist())
print("n events:", len(rows_100))

REL_STEP = 1e-5
records = []
t_start = time.time()

for i, row in enumerate(rows_100, 1):
    d4 = pd.read_csv(list(ROOT4.rglob(f"extreme100/{row}/all_refits.csv"))[0])
    d4 = d4[(d4["hypothesis"] == "H1") & (d4["status"] == "success")]
    r_truth = d4[d4["label"] == "truth"].iloc[0]

    t0v, u0v, tEv, rhov, piENv, piEEv = (
        float(r_truth["t0"]), float(r_truth["u0"]), float(r_truth["tE"]),
        float(r_truth["rho"]), float(r_truth["piEN"]), float(r_truth["piEE"]),
    )
    chi2_truth = float(r_truth["chi2"])
    chi2_ref = float(ref.loc[row, "chi2_oracle5"])

    record = {"sample": "extreme100", "catalog_row": int(row), "matched_tail_catalog_row": -1}
    meta = core.load_case(record)
    fit_obj, param_order = build_h1_forward_fit(meta)
    assert param_order == ["t0", "u0", "tE", "rho", "piEN", "piEE"], param_order

    theta = np.array([t0v, u0v, tEv, rhov, piENv, piEEv])
    scale = [tEv, 1.0, tEv, rhov, 1.0, 1.0]

    t0c = time.time()
    J, r0 = numerical_jacobian(fit_obj.objective_function, theta, REL_STEP, scale)
    dt_fisher = time.time() - t0c
    chi2_check = float(np.sum(r0 ** 2))

    F_y, F_phys = fisher_from_jacobian_physical(J, scale)
    diag = fisher_diagnostics(F_y)

    # Schur complement F_pi_given_eta (eta=0:4, pi=4:6)
    F = diag["F_y"]
    F_ee = F[:4, :4]
    F_ep = F[:4, 4:6]
    F_pe = F[4:6, :4]
    F_pp = F[4:6, 4:6]

    ev_ee, evec_ee = np.linalg.eigh(0.5 * (F_ee + F_ee.T))
    scale_ee = max(abs(ev_ee.max()), 1e-300)
    tol_ee = 1e-10 * scale_ee
    inv_ee = np.array([1.0 / e if e > tol_ee else 0.0 for e in ev_ee])
    F_ee_pinv = (evec_ee * inv_ee) @ evec_ee.T

    F_schur = F_pp - F_pe @ F_ee_pinv @ F_ep
    F_schur = 0.5 * (F_schur + F_schur.T)
    ev_s, evec_s = np.linalg.eigh(F_schur)
    order = np.argsort(ev_s)
    ev_s = ev_s[order]
    evec_s = evec_s[:, order]
    ev_s_clip = np.clip(ev_s, 0.0, None)
    min_eig_schur = float(ev_s_clip[0])
    max_eig_schur = float(ev_s_clip[1])
    cond_schur = float(max_eig_schur / min_eig_schur) if min_eig_schur > 0 else np.inf
    scale_s = max(abs(ev_s_clip.max()), 1e-300)
    full_rank_schur = bool(np.sum(ev_s_clip > 1e-10 * scale_s) == 2)
    logdet_schur = float(np.sum(np.log(ev_s_clip))) if full_rank_schur else np.nan

    # covariance of the (piEN,piEE) block, marginal (from full C_y, not schur)
    C = diag["C_y"]
    C_pp = C[4:6, 4:6]
    sig_major = float(np.sqrt(max(np.linalg.eigvalsh(C_pp))))
    sig_minor = float(np.sqrt(max(min(np.linalg.eigvalsh(C_pp)), 0.0)))
    corr_piE = diag["corr"][4, 5]

    weak_schur = evec_s[:, 0]  # weakest direction in (piEN,piEE) schur space
    weak_schur_angle_deg = float(np.degrees(np.arctan2(weak_schur[1], weak_schur[0])))

    # normalized backbone-parallax coupling: max |correlation| between
    # any eta-coordinate (t0/tE,u0,logtE,logrho) and any pi-coordinate
    # (piEN,piEE), from the full dimensionless correlation matrix.
    coupling = float(np.max(np.abs(diag["corr"][:4, 4:6])))

    out = {
        "catalog_row": int(row),
        "chi2_truth_H1": chi2_truth,
        "chi2_check": chi2_check,
        "chi2_controlled5_ref": chi2_ref,
        "unsafe_H1_truth": bool(chi2_truth - chi2_ref > 0.1),
        "t0": t0v, "u0": u0v, "tE": tEv, "rho": rhov, "piEN": piENv, "piEE": piEEv,
        "fisher_wall_time_s": dt_fisher,
        "min_eig_full": diag["min_eig"],
        "max_eig_full": diag["max_eig"],
        "cond_number_full": diag["cond_number"],
        "full_rank_full": diag["full_rank"],
        "has_negative_eig_full": diag["has_negative_eig"],
        "max_abs_corr_full": diag["max_abs_corr"],
        "min_eig_schur": min_eig_schur,
        "max_eig_schur": max_eig_schur,
        "cond_number_schur": cond_schur,
        "full_rank_schur": full_rank_schur,
        "logdet_schur": logdet_schur,
        "sigma_major_piE": sig_major,
        "sigma_minor_piE": sig_minor,
        "corr_piEN_piEE": corr_piE,
        "weak_schur_angle_deg": weak_schur_angle_deg,
        "backbone_parallax_coupling": coupling,
    }
    records.append(out)

    if i % 10 == 0 or i == len(rows_100):
        print(f"[{i}/{len(rows_100)}] row={row} elapsed={time.time()-t_start:.0f}s")

df_out = pd.DataFrame(records)
OUT_CSV = os.path.join(SCRATCH, "fisher_h1_results.csv")
df_out.to_csv(OUT_CSV, index=False)
print("DONE ->", OUT_CSV)
print("total wall time:", time.time() - t_start)
print("max chi2 check diff:", (df_out["chi2_truth_H1"] - df_out["chi2_check"]).abs().max())
