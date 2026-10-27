#!/usr/bin/env python3
import os as _os_path_setup
HERE = _os_path_setup.path.dirname(_os_path_setup.path.abspath(__file__))

import sys
import os
import copy
import time
import json

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
    "--fit-scope", "h0",
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
assert core.FIT_SCOPE == "h0"


def data_driven_t0_bounds(meta):
    all_times = []
    for band in ["u", "g", "r", "i", "z", "y"]:
        lc = meta["curves"][band]
        if len(lc):
            all_times.extend(np.asarray(lc[:, 0], dtype=float).tolist())
    all_times = np.asarray(all_times, dtype=float)
    return float(np.min(all_times)), float(np.max(all_times))


def build_h0_forward_fit(meta):
    t_min, t_max = data_driven_t0_bounds(meta)
    bounds = copy.deepcopy(core.H0_BOUNDS)
    bounds = core.apply_bounds_profile(bounds, h1=False, profile=core.BOUNDS_PROFILE)
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
        meta["true_params"], "FSPL", fit_parallax=False, fit_defaults=None,
    )
    pyLIMAmodel = fit_lc.build_fit_pyLIMA_model(
        e, "FSPL", fit_params, Origin=None, fit_parallax=False,
    )
    fit, pool_processes = fit_lc.build_fitter(pyLIMAmodel, "TRF")
    fit_lc.apply_fit_bounds(
        fit, fit_params, "FSPL", meta["rango"], fit_parallax=False, fit_bounds=bounds,
    )
    param_order = fit_lc.fit_parameter_order("FSPL", fit_parallax=False)
    return fit, param_order


# ------------------------------------------------------------------
# oracle52 data: step1 = physical/truth/1, step2 = physical/truth/0.1
# ------------------------------------------------------------------
oracle_df = pd.read_csv(
    "/home/anibal-pc/ulensing_degenerate_models/Parallax_LSST/"
    "lsstmonts_catalog_sedighe/validation/bounds_convergence/results/"
    "gate1_oracle_diagnostics.csv"
)
d52 = oracle_df[oracle_df["source"] == "gate1_oracle_52"].copy()


def canon(mode, sid):
    if "/" in sid:
        anchor, tag = sid.split("/", 1)
    else:
        if sid.startswith("old_H0_rho_"):
            anchor = "old_H0"; tag = sid[len("old_H0_rho_"):]
        else:
            anchor = "truth"; tag = sid[len("truth_rho_"):]
    if tag == "truth":
        tag = "truth_rho"
    if tag not in ("truth_rho", "rho_from_old_H0"):
        tag = f"{float(tag):.10g}"
    return f"{mode}/{anchor}/{tag}"


d52["canon"] = [canon(m, s) for m, s in zip(d52["mode"], d52["strategy_id"])]

piv_chi2 = d52.pivot_table(index="catalog_row", columns="canon", values="chi2", aggfunc="min")
piv_t0 = d52.pivot_table(index="catalog_row", columns="canon", values="t0", aggfunc="min")
piv_u0 = d52.pivot_table(index="catalog_row", columns="canon", values="u0", aggfunc="min")
piv_tE = d52.pivot_table(index="catalog_row", columns="canon", values="tE", aggfunc="min")
piv_rho = d52.pivot_table(index="catalog_row", columns="canon", values="rho", aggfunc="min")

oracle52_min = piv_chi2.min(axis=1)

STEP1 = "physical/truth/1"
STEP2 = "physical/truth/0.1"

rows_100 = sorted(piv_chi2.index.tolist())
print("n events:", len(rows_100))

REL_STEP = 1e-5

records = []
t_start = time.time()
n_trf_timing = []

for i, row in enumerate(rows_100, 1):
    record = {"sample": "extreme100", "catalog_row": int(row), "matched_tail_catalog_row": -1}
    meta = core.load_case(record)
    fit_obj, param_order = build_h0_forward_fit(meta)
    assert param_order == ["t0", "u0", "tE", "rho"], param_order

    out = {"catalog_row": int(row)}
    out["chi2_oracle52"] = float(oracle52_min.loc[row])

    for step_name, strat in [("1", STEP1), ("2", STEP2)]:
        t0v = float(piv_t0.loc[row, strat])
        u0v = float(piv_u0.loc[row, strat])
        tEv = float(piv_tE.loc[row, strat])
        rhov = float(piv_rho.loc[row, strat])
        chi2v = float(piv_chi2.loc[row, strat])

        theta = np.array([t0v, u0v, tEv, rhov])
        scale_for_step = [tEv, 1.0, tEv, rhov]

        t0c = time.time()
        J, r0 = numerical_jacobian(fit_obj.objective_function, theta, REL_STEP, scale_for_step)
        dt_fisher = time.time() - t0c

        chi2_check = float(np.sum(r0 ** 2))

        F_y, F_phys = fisher_from_jacobian_physical(J, scale_for_step)
        diag = fisher_diagnostics(F_y)

        out[f"chi2_step{step_name}"] = chi2v
        out[f"chi2_check_step{step_name}"] = chi2_check
        out[f"t0_step{step_name}"] = t0v
        out[f"u0_step{step_name}"] = u0v
        out[f"tE_step{step_name}"] = tEv
        out[f"rho_step{step_name}"] = rhov
        out[f"fisher_wall_time_s_step{step_name}"] = dt_fisher

        out[f"min_eig_step{step_name}"] = diag["min_eig"]
        out[f"max_eig_step{step_name}"] = diag["max_eig"]
        out[f"cond_number_step{step_name}"] = diag["cond_number"]
        out[f"numerical_rank_step{step_name}"] = diag["numerical_rank"]
        out[f"full_rank_step{step_name}"] = diag["full_rank"]
        out[f"has_negative_eig_step{step_name}"] = diag["has_negative_eig"]
        out[f"logdet_step{step_name}"] = diag["logdet"]
        out[f"max_abs_corr_step{step_name}"] = diag["max_abs_corr"]
        sig = diag["sigmas"]
        out[f"sigma_t0overtE_step{step_name}"] = sig[0]
        out[f"sigma_u0_step{step_name}"] = sig[1]
        out[f"sigma_logtE_step{step_name}"] = sig[2]
        out[f"sigma_logrho_step{step_name}"] = sig[3]
        wv = diag["weakest_eigvec"]
        out[f"weakvec_t0_step{step_name}"] = wv[0]
        out[f"weakvec_u0_step{step_name}"] = wv[1]
        out[f"weakvec_logtE_step{step_name}"] = wv[2]
        out[f"weakvec_logrho_step{step_name}"] = wv[3]

        # stash raw matrices for step2 concordance calc (not saved to csv)
        out[f"_F_y_step{step_name}"] = F_y
        out[f"_C_y_step{step_name}"] = diag["C_y"]
        out[f"_theta_phys_step{step_name}"] = theta

    records.append(out)

    if i % 10 == 0 or i == len(rows_100):
        print(f"[{i}/{len(rows_100)}] row={row} elapsed={time.time()-t_start:.0f}s")

# ------------------------------------------------------------------
# concordance metrics between step1 and step2
# ------------------------------------------------------------------
for out in records:
    F1 = out["_F_y_step1"]
    F2 = out["_F_y_step2"]
    C1 = out["_C_y_step1"]
    C2 = out["_C_y_step2"]
    th1_phys = out["_theta_phys_step1"]
    th2_phys = out["_theta_phys_step2"]

    tE_avg = 0.5 * (th1_phys[2] + th2_phys[2])
    rho_avg = 0.5 * (th1_phys[3] + th2_phys[3])
    D_common = np.diag([tE_avg, 1.0, tE_avg, rho_avg])

    # re-express F1, F2 (already in each solution's OWN dimensionless
    # coords) back to physical, then into the COMMON dimensionless
    # coords defined by (tE_avg, rho_avg), so C1+C2 is meaningful.
    D1 = np.diag([th1_phys[2], 1.0, th1_phys[2], th1_phys[3]])
    D2 = np.diag([th2_phys[2], 1.0, th2_phys[2], th2_phys[3]])
    D1_inv = np.diag(1.0 / np.diag(D1))
    D2_inv = np.diag(1.0 / np.diag(D2))
    F1_phys = D1_inv.T @ F1 @ D1_inv
    F2_phys = D2_inv.T @ F2 @ D2_inv
    F1_common = D_common.T @ F1_phys @ D_common
    F2_common = D_common.T @ F2_phys @ D_common

    # pseudoinverses via eigh for robustness
    def pinv_psd(F, tol=1e-10):
        ev, evec = np.linalg.eigh(0.5 * (F + F.T))
        scale = max(abs(ev.max()), 1e-300)
        inv = np.array([1.0 / e if e > tol * scale else 0.0 for e in ev])
        return (evec * inv) @ evec.T

    C1_common = pinv_psd(F1_common)
    C2_common = pinv_psd(F2_common)

    y1 = np.array([
        (th1_phys[0] - th1_phys[0]) / tE_avg,  # 0, kept for shape symmetry
        th1_phys[1],
        np.log(th1_phys[2]),
        np.log(th1_phys[3]),
    ])
    # actually we need y in the COMMON frame relative to a common
    # origin; use t0 difference directly scaled by tE_avg.
    dy = np.array([
        (th1_phys[0] - th2_phys[0]) / tE_avg,
        th1_phys[1] - th2_phys[1],
        np.log(th1_phys[2]) - np.log(th2_phys[2]),
        np.log(th1_phys[3]) - np.log(th2_phys[3]),
    ])

    C_sum = C1_common + C2_common
    C_sum_pinv = pinv_psd(C_sum, tol=1e-10)
    D12_sq = float(dy @ C_sum_pinv @ dy)

    # weakest eigenvector alignment (already stored components, recompute cos angle)
    w1 = np.array([out["weakvec_t0_step1"], out["weakvec_u0_step1"],
                   out["weakvec_logtE_step1"], out["weakvec_logrho_step1"]])
    w2 = np.array([out["weakvec_t0_step2"], out["weakvec_u0_step2"],
                   out["weakvec_logtE_step2"], out["weakvec_logrho_step2"]])
    cos_angle = float(np.abs(np.dot(w1, w2)) / (np.linalg.norm(w1) * np.linalg.norm(w2)))

    out["delta_chi2_12"] = abs(out["chi2_step1"] - out["chi2_step2"])
    out["D12_sq"] = D12_sq
    out["weak_eigvec_alignment"] = cos_angle
    out["sigma_ratio_min_1_2"] = out["min_eig_step2"] and (out["min_eig_step1"] / out["min_eig_step2"]) if out["min_eig_step2"] else np.nan

    for k in list(out.keys()):
        if k.startswith("_"):
            del out[k]

df_out = pd.DataFrame(records)
OUT_CSV = os.path.join(SCRATCH, "fisher_h0_results.csv")
df_out.to_csv(OUT_CSV, index=False)
print("DONE ->", OUT_CSV)
print("total wall time:", time.time() - t_start)
