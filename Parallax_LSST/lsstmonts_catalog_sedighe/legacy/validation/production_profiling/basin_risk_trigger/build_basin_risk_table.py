#!/usr/bin/env python3
"""
Demonstration 1: build the frozen event-level development table for
basin-risk trigger design. See BASIN_RISK_TRIGGER_DEVELOPMENT.md for
every convention (label definitions, dof_H0, sigma_i, residual
features, missing-value policy, leakage audit).

Reuses, without modification: run_bounds_audit_refit_core (core.py),
fit_lc, morphology_seed.morphology_seed_from_curves,
morphology_extraction.extract_event_morphology,
final_policy.{n_photometry_points, chi2_dof_sanity_flag,
CHI2_DOF_SANITY_THRESHOLD}, numerical_safeguards.run_same_point_continuation,
run_one_fit_full. No fitter/morphology logic is duplicated or retuned.

The H0 nominal + (if triggered) continuation fit is RE-RUN in-memory
(deterministic: same event, same morphology-seed start, same bounds,
same optimizer options, same random_state -- bit-identical to the
already-frozen final_policy.py pipeline) so that the RAW fit_object
(residual vector, Jacobian) is available as real numpy arrays. The
already-saved final_policy_h1basinvalidation_100events.json only has
these serialized as truncated numpy-repr strings (via json.dump's
default=str), which is lossy for this analysis -- re-deriving them
in-memory is not a new fit, it reproduces the exact same H0 fit already
part of the frozen policy.
"""
import sys
import os
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PP = os.path.dirname(HERE)  # validation/production_profiling
BA = os.path.join(PP, "..", "bounds_audit")
BC = os.path.join(PP, "..", "bounds_convergence")
sys.path.insert(0, BA)
sys.path.insert(0, BC)
sys.path.insert(0, PP)

ap = argparse.ArgumentParser()
ap.add_argument("--rows-csv", default=os.path.join(PP, "results", "h1_basin_validation_sample_frozen.csv"))
ap.add_argument("--basin-gap-csv", default=os.path.join(PP, "results", "basin_gap_validation_100events.csv"))
ap.add_argument("--final-policy-json", default=os.path.join(PP, "results", "final_policy_h1basinvalidation_100events.json"))
ap.add_argument("--out", default=os.path.join(HERE, "basin_risk_development_table.csv"))
args = ap.parse_args()

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = ["run_bounds_audit_refit_core.py", "--catalog-row", "71181",
            "--bounds-profile", "production_candidate", "--fit-scope", "h0", "--dry-run"]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402
from load_new_event import load_new_case  # noqa: E402
from morphology_seed import morphology_seed_from_curves  # noqa: E402
from morphology_extraction import extract_event_morphology  # noqa: E402
from run_one_fit_full import run_one_fit_full  # noqa: E402
from final_policy import n_photometry_points, chi2_dof_sanity_flag, CHI2_DOF_SANITY_THRESHOLD  # noqa: E402
from numerical_safeguards import run_same_point_continuation  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]  # exact order add_rubin_telescopes uses (fit_lc.py LSST_BANDS)


def reproduce_final_h0(meta, seed):
    """Exactly final_policy.py's H0 side: nominal fit, then (if
    reduced_chi2 > 50) exactly one same-point continuation, kept if it
    does not worsen chi2. Returns the live (non-serialized) fit record
    dict, so fit_record['fit_object']['fun']/['jac'] are real ndarrays."""
    initial = {"t0": seed["t0"], "u0": seed["u0"], "tE": seed["tE"], "rho": seed["rho"]}
    nominal = run_one_fit_full(core, fit_lc, meta, "H0", initial, "nominal")
    n_points = n_photometry_points(meta["curves"])
    flagged, _ = chi2_dof_sanity_flag(nominal, n_points, 4)
    continuation_ran = False
    if flagged:
        continuation_ran = True
        cont = run_same_point_continuation(core, fit_lc, meta, "H0", nominal, label_suffix="chi2dof_continuation")
        if cont["fit_record"]["chi2"] <= nominal["chi2"]:
            final = cont["fit_record"]
        else:
            final = nominal
    else:
        final = nominal
    return final, continuation_ran, n_points


def segment_residuals_by_band(fun, jac, curves):
    """Splits the flat residual/Jacobian arrays into per-band arrays,
    using the verified telescope-addition order (LSST_BANDS, skipping
    empty bands; Roman/W149 always empty in this dataset, contributes
    zero rows) -- confirmed empirically: len(fun) == sum of per-band
    point counts exactly, for every event checked."""
    out = {}
    i = 0
    for b in LSST_BANDS:
        n_b = len(curves[b])
        if n_b == 0:
            continue
        t_b = np.asarray(curves[b][:, 0], dtype=float)
        out[b] = {
            "t": t_b,
            "resid": np.asarray(fun[i:i + n_b], dtype=float),
            "jac": np.asarray(jac[i:i + n_b], dtype=float) if jac is not None else None,
        }
        i += n_b
    assert i == len(fun), f"band segmentation mismatch: {i} != {len(fun)}"
    return out


def contiguous_width_resid(t_sorted, absr_sorted, peak_idx, q, r_max):
    """Exact analog of morphology_extraction's contiguous_width(q):
    walk outward from the peak index while |r| >= q*r_max; width is the
    distance from the peak to the LAST point that still satisfied the
    threshold (not the first failing point); a side that never fails
    before reaching the data boundary is CENSORED (unmeasurable), never
    extrapolated."""
    thresh = q * r_max
    n = len(absr_sorted)
    left = peak_idx
    while left > 0 and absr_sorted[left - 1] >= thresh:
        left -= 1
    right = peak_idx
    while right < n - 1 and absr_sorted[right + 1] >= thresh:
        right += 1
    left_censored = (left == 0)
    right_censored = (right == n - 1)
    left_w = None if left_censored else float(t_sorted[peak_idx] - t_sorted[left])
    right_w = None if right_censored else float(t_sorted[right] - t_sorted[peak_idx])
    return left_w, right_w, left_censored, right_censored


def residual_morphology(band_segments):
    """Pools (t, |resid|) across all bands, sorts by time, finds the
    peak |resid|, and computes T25/T50/T75-style widths at 25/50/75% of
    the peak -- exactly analogous to morphology_extraction's flux-excess
    width convention (same q values, same walk-from-peak-until-first-
    failure rule, same never-extrapolate censoring convention), applied
    to standardized H0 residuals instead of flux excess."""
    all_t, all_r = [], []
    for b, seg in band_segments.items():
        all_t.append(seg["t"])
        all_r.append(seg["resid"])
    all_t = np.concatenate(all_t)
    all_r = np.concatenate(all_r)
    order = np.argsort(all_t, kind="stable")
    t_sorted = all_t[order]
    absr_sorted = np.abs(all_r[order])

    peak_idx = int(np.argmax(absr_sorted))
    r_max = float(absr_sorted[peak_idx])
    t_peak = float(t_sorted[peak_idx])

    widths = {}
    for q, label in [(0.25, "T25"), (0.50, "T50"), (0.75, "T75")]:
        lw, rw, lc, rc = contiguous_width_resid(t_sorted, absr_sorted, peak_idx, q, r_max)
        widths[label] = {"left_width": lw, "right_width": rw, "left_censored": lc, "right_censored": rc}

    def total_width(entry):
        if entry["left_width"] is not None and entry["right_width"] is not None:
            return entry["left_width"] + entry["right_width"]
        return None

    T50_resid = total_width(widths["T50"])
    T75_resid = total_width(widths["T75"])
    T25_resid = total_width(widths["T25"])
    A_resid = None
    if widths["T50"]["left_width"] is not None and widths["T50"]["right_width"] is not None:
        lw, rw = widths["T50"]["left_width"], widths["T50"]["right_width"]
        if (lw + rw) > 0:
            A_resid = float((rw - lw) / (rw + lw))

    return {
        "R_max_resid": r_max, "t_peak_resid": t_peak,
        "T25_resid": T25_resid, "T50_resid": T50_resid, "T75_resid": T75_resid,
        "A_resid": A_resid,
    }


def per_band_dw(band_segments):
    """Classic Durbin-Watson statistic per band (sum of squared
    consecutive differences over sum of squares), time-ordered within
    band; combined across bands via a point-count-weighted average
    (weight = n_band-1, the number of consecutive pairs each band
    contributes). A band with < 2 points has an undefined per-band
    statistic and contributes nothing (not zero) to the combination.
    dw_stat is missing only if NO band has >=2 points."""
    weighted_sum = 0.0
    weight_total = 0.0
    any_valid = False
    for b, seg in band_segments.items():
        r = seg["resid"]
        n_b = len(r)
        if n_b < 2:
            continue
        diffs = np.diff(r)
        denom = np.sum(r ** 2)
        if denom <= 0 or not np.isfinite(denom):
            continue
        dw_b = float(np.sum(diffs ** 2) / denom)
        if not np.isfinite(dw_b):
            continue
        w = n_b - 1
        weighted_sum += dw_b * w
        weight_total += w
        any_valid = True
    if not any_valid or weight_total == 0:
        return None
    return weighted_sum / weight_total


def jacobian_condition(jac):
    if jac is None:
        return None
    try:
        s = np.linalg.svd(np.asarray(jac, dtype=float), compute_uv=False)
        s = s[np.isfinite(s)]
        if len(s) == 0 or s.min() <= 0:
            return None
        return float(np.log10(s.max() / s.min()))
    except Exception:
        return None


def bound_distance_features(final_h0, fit_bounds):
    """Normalized distance-to-nearest-bound, defined BEFORE looking at
    any dangerous label. LOG-scale normalization for tE/rho (bounds
    span 6-7 orders of magnitude: tE in [0.1,500000], rho in
    [1e-7,10]); LINEAR-scale normalization for t0/u0 (bounds span a
    single scale: t0 is a +-half_width window around the data range,
    u0 in [-10,10]). This is a diagnostic-only convention, independent
    of the optimizer's own internal x_scale='jac' trust-region scaling
    (which affects only the optimization step, not this feature)."""
    def lin_norm(x, lo, hi):
        if hi <= lo:
            return None
        return min(x - lo, hi - x) / (hi - lo)

    def log_norm(x, lo, hi):
        if x <= 0 or lo <= 0 or hi <= 0 or hi <= lo:
            return None
        lx, llo, lhi = np.log(x), np.log(lo), np.log(hi)
        return min(lx - llo, lhi - lx) / (lhi - llo)

    t0_bounds = fit_bounds["t0"]
    t0_lo = t0_bounds["center"] - t0_bounds["half_width"]
    t0_hi = t0_bounds["center"] + t0_bounds["half_width"]
    d_t0 = lin_norm(final_h0["t0"], t0_lo, t0_hi)
    d_u0 = lin_norm(final_h0["u0"], fit_bounds["u0"][0], fit_bounds["u0"][1])
    d_tE = log_norm(final_h0["tE"], fit_bounds["tE"][0], fit_bounds["tE"][1])
    d_rho = log_norm(final_h0["rho"], fit_bounds["rho"][0], fit_bounds["rho"][1])

    ds = [d for d in [d_t0, d_u0, d_tE, d_rho] if d is not None]
    d_bounds = min(ds) if ds else None

    active_mask = final_h0.get("optimizer_active_mask")
    if active_mask is None:
        h0_bound_active = None
    else:
        # fit_lc.py's optimizer_diagnostics_from_fit stores this as
        # repr(list), e.g. "[0, 0, 0, 0]" -- parse back to numbers.
        if isinstance(active_mask, str):
            import ast
            active_mask = ast.literal_eval(active_mask)
        arr = np.asarray(active_mask, dtype=float)
        h0_bound_active = bool(np.any(arr != 0))

    return d_bounds, h0_bound_active


# ---------------------------------------------------------------------
manifest = pd.read_csv(args.rows_csv)
basin_gap = pd.read_csv(args.basin_gap_csv).set_index("row")
fp_events = {e["catalog_row"]: e for e in json.load(open(args.final_policy_json))["events"]}

rows = []
for _, row_entry in manifest.iterrows():
    row = int(row_entry["catalog_row"])
    h5_path = row_entry["h5_path"]
    meta = load_new_case(h5_path, core)

    bg = basin_gap.loc[row]
    D_s, D_r = float(bg["d_simple"]), float(bg["d_ref"])
    flagged_existing = bool(bg["flagged"])

    dangerous = bool((D_r < 9) and (D_s >= 4) and (D_s > D_r))
    large_gap_abs = bool(abs(D_s - D_r) > 5)
    large_gap_up = bool((D_s - D_r) > 5)

    seed = morphology_seed_from_curves(meta["curves"])
    assert seed is not None, f"row {row}: morphology seed unexpectedly None (should not happen in this sample)"

    final_h0, continuation_ran, n_points = reproduce_final_h0(meta, seed)

    # cross-check reproduction fidelity against the already-frozen result
    stored_chi2 = fp_events[row]["chi2_h0"]
    reproduction_diff = abs(final_h0["chi2"] - stored_chi2)

    morph_raw = extract_event_morphology(meta["curves"])
    if morph_raw.get("n_points_T50") is not None:
        n_peak_points = morph_raw["n_points_T50"]
        n_peak_points_source = "n_points_T50"
    else:
        n_peak_points = morph_raw.get("n_points_T25")
        n_peak_points_source = "n_points_T25_fallback"

    fo = final_h0["fit_object"]
    fun = np.asarray(fo["fun"], dtype=float)
    jac = np.asarray(fo["jac"], dtype=float) if fo.get("jac") is not None else None
    band_segments = segment_residuals_by_band(fun, jac, meta["curves"])

    resid_morph = residual_morphology(band_segments)
    dw_stat = per_band_dw(band_segments)
    logkappa_J = jacobian_condition(jac)

    d_bounds, h0_bound_active = bound_distance_features(final_h0, final_h0["fit_bounds"])

    tE_fit, rho_fit = final_h0["tE"], final_h0["rho"]
    tE_seed, rho_seed = seed["tE"], seed["rho"]
    dlogtE = float(abs(np.log(tE_fit / tE_seed))) if (tE_fit > 0 and tE_seed > 0) else None
    dlogrho = float(abs(np.log(rho_fit / rho_seed))) if (rho_fit > 0 and rho_seed > 0) else None
    du0 = float(abs(final_h0["u0"] - seed["u0"]))
    dt0_over_tE = float(abs(final_h0["t0"] - seed["t0"]) / tE_fit) if tE_fit != 0 else None

    rows.append({
        "event_id": row,
        "D_s": D_s, "D_r": D_r,
        "dangerous": dangerous, "large_gap_abs": large_gap_abs, "large_gap_up": large_gap_up,
        "excluded_from_development": flagged_existing,

        "chi2nu_h0": final_h0["chi2"] / max(1, n_points - 4),
        "morph_mismatch": seed["mismatch"],
        "dlogtE": dlogtE, "dlogrho": dlogrho, "du0": du0, "dt0_over_tE": dt0_over_tE,
        "d_bounds": d_bounds,
        "R_max_resid": resid_morph["R_max_resid"],
        "T50_resid": resid_morph["T50_resid"], "T75_resid": resid_morph["T75_resid"],
        "T25_resid": resid_morph["T25_resid"], "A_resid": resid_morph["A_resid"],
        "dw_stat": dw_stat, "logkappa_J": logkappa_J,

        "n_points": n_points, "n_peak_points": n_peak_points, "n_peak_points_source": n_peak_points_source,
        "h0_optimizer_status": final_h0.get("optimizer_status"),
        "h0_nfev": final_h0.get("optimizer_nfev"),
        "h0_bound_active": h0_bound_active,
        "continuation_ran_h0": continuation_ran,

        "h0_optimizer_message": final_h0.get("optimizer_message"),
        "h0_optimality": final_h0.get("optimizer_optimality"),
        "h0_seed_t0": seed["t0"], "h0_seed_u0": seed["u0"], "h0_seed_tE": seed["tE"], "h0_seed_rho": seed["rho"],
        "h0_fit_t0": final_h0["t0"], "h0_fit_u0": final_h0["u0"], "h0_fit_tE": final_h0["tE"], "h0_fit_rho": final_h0["rho"],

        "reproduction_chi2_diff": reproduction_diff,

        "D_s2": None, "repaired": None,
    })
    print(f"row={row} D_s={D_s:.3f} D_r={D_r:.3f} dangerous={dangerous} excluded={flagged_existing} "
          f"chi2nu_h0={rows[-1]['chi2nu_h0']:.3f} repro_diff={reproduction_diff:.2e}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)
print(f"\nWrote {len(df)} rows -> {args.out}")
print("max reproduction_chi2_diff:", df["reproduction_chi2_diff"].max())
