"""
run_one_fit_full(meta, hypothesis, initial, label, core) -- exactly
replicates run_bounds_audit_refit_core.run_one_fit's bounds
construction (BOUNDS_PROFILE-resolved H0/H1 bounds + the same
data-driven, truth-independent t0 window) and fit_lc.fit_rubin_roman
call, but returns the FULL fit.fit_results dict (optimizer_nfev,
optimizer_njev, optimizer_status, optimizer_message, optimizer_cost,
optimizer_n_active_bounds, optimizer_active_mask, optimizer_success,
optimizer_optimality) instead of run_one_fit's trimmed record --
needed for the same-point-continuation / nested-rescue safeguard
audit trail (scipy status/message/nfev/njev must be retained for
every fit). core.run_one_fit itself is never modified.
"""
import copy
import numpy as np


def run_one_fit_full(core, fit_lc, meta, hypothesis, initial, label):
    h1 = hypothesis == "H1"

    bounds = copy.deepcopy(core.H1_BOUNDS if h1 else core.H0_BOUNDS)
    bounds = core.apply_bounds_profile(bounds, h1=h1, profile=core.BOUNDS_PROFILE)

    all_times = []
    for band in ["u", "g", "r", "i", "z", "y"]:
        lc = meta["curves"][band]
        if len(lc):
            all_times.extend(np.asarray(lc[:, 0], dtype=float).tolist())
    all_times = np.asarray(all_times, dtype=float)
    data_t_min, data_t_max = float(np.min(all_times)), float(np.max(all_times))

    import os
    t0_margin_factor = float(os.environ.get("HIDDEN_PARALLAX_T0_MARGIN_FACTOR", "0"))
    t0_margin = t0_margin_factor * (data_t_max - data_t_min)
    t_min, t_max = data_t_min - t0_margin, data_t_max + t0_margin
    bounds["t0"] = {"type": "center_width", "center": 0.5 * (t_min + t_max),
                     "half_width": 0.5 * (t_max - t_min)}

    fit, ev, model = fit_lc.fit_rubin_roman(
        Source=meta["Source"], event_params=meta["true_params"], path_save=f"/tmp/prof_discard_full_{label}",
        path_ephemerides=str(core.EPHEMERIDES), model="FSPL", algo="TRF", Origin=None, rango=meta["rango"],
        wfirst_lc=meta["curves"]["W149"], lsst_u=meta["curves"]["u"], lsst_g=meta["curves"]["g"],
        lsst_r=meta["curves"]["r"], lsst_i=meta["curves"]["i"], lsst_z=meta["curves"]["z"],
        lsst_y=meta["curves"]["y"], fit_model="FSPL", fit_parallax=h1, fit_defaults=None,
        fit_bounds=copy.deepcopy(bounds), random_state=20260909 + int(meta["row"]), initial_guess=initial,
        optimizer_options=copy.deepcopy(core.OPTIMIZER_OPTIONS), event_ra=meta["event_ra"],
        event_dec=meta["event_dec"],
    )

    res = fit.fit_results
    best = np.asarray(res["best_model"], dtype=float)

    record = dict(res)
    record.update({
        "label": label, "hypothesis": hypothesis, "status": "success", "chi2": float(res["chi2"]),
        "t0": float(best[0]), "u0": float(best[1]), "tE": float(best[2]), "rho": float(best[3]),
        "piEN": float(best[4]) if h1 else None, "piEE": float(best[5]) if h1 else None,
    })
    return record
