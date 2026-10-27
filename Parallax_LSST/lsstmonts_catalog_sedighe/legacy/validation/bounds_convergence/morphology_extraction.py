"""
M0 -- deterministic light-curve morphology extraction (frozen definition).

This definition is frozen BEFORE any comparison against truth, oracle52,
or old_H0. Nothing here is tuned by looking at those results.

## Per-band excess-flux proxy

For each band with data, a robust baseline magnitude m_base is the
inverse-variance-weighted MEDIAN of all magnitudes in that band (not
mean -- robust to the brief brightened excursion itself, since a
microlensing event spends most of the observing baseline near its
unmagnified brightness whenever Tobs is not tiny compared to tE, which
is exactly the t0-bounds convention already used throughout this
project). This does not require locating the peak first.

The dimensionless excess-flux proxy is

    excess(t) = 10^(-0.4*(m(t) - m_base)) - 1

(excess=0 at baseline, positive during brightening). This is a SHAPE
proxy only -- it is not blend-corrected and does not require knowing
the blend fraction; only the *shape* of the bump is used downstream,
never its absolute amplitude in physical flux units.

Its uncertainty is propagated by linearizing the magnitude->flux-ratio
transform at each point:

    err_excess(t) = 0.4*ln(10)*(excess(t)+1)*err_mag(t)

## Smoothing

Per band, a heteroscedastic weighted smoothing spline
(`scipy.interpolate.UnivariateSpline`, weights `w = 1/err_excess`,
degree k=3) is fit on that band's own time support, with smoothing
parameter `s = N_band` -- the standard rule-of-thumb documented by
scipy for `w=1/sigma`-normalized weights (requires the weighted
residual sum of squares to be of order 1 per data point: "smooth just
enough to be statistically consistent with the measurement errors").
This is a fixed, principled, non-arbitrary choice -- it is not
retuned per event and is fixed identically for every band/event
before any oracle comparison. No Gaussian Process is used anywhere in
this module.

## Combining bands

Bands are combined ONLY after this per-band normalization -- fluxes
from different filters are never summed or averaged directly. The
combined signal on a common fine time grid is

    combined(t) = sum_b [ weight_b * spline_b(t) ] / sum_b [ weight_b ]

where the sum runs only over bands whose own time support covers t,
and `weight_b = 1 / median(err_excess_b)^2` is a single scalar per
band (not time-varying) -- a per-band SNR weight, not a per-point one.

## Peak and width extraction

`t_peak_morph` = argmax of `combined(t)` over the time range actually
supported by at least one band's data (no extrapolation).

For q in {0.25, 0.50, 0.75}, `Tq` is the width of the largest
CONTIGUOUS region containing `t_peak_morph` over which
`combined(t) >= q * combined(t_peak_morph)`. If that region's edge is
the edge of the data-supported range itself (i.e. the curve has not
dropped below the threshold by the time the data ends), that side is
censored and recorded explicitly as `not measurable` -- never
extrapolated or fabricated. `T50_left`/`T50_right` are measured
separately from `t_peak_morph`; `A50 = (T50_right - T50_left) /
(T50_right + T50_left)` is defined only when both sides of T50 are
measurable.

## Quality / SNR (M2)

- `peak_SN`: `combined(t_peak_morph)` divided by the pooled
  inverse-variance-weighted uncertainty of `excess` from all points
  (all bands) inside the measurable T50 window (or, if T50 is not
  measurable, inside a fixed fallback window of +/-5 days around
  the peak, flagged accordingly).
- `integrated_SN`: `sqrt(sum((excess_i/err_excess_i)^2))` over the
  same point set as `peak_SN`, pooled across bands -- a standard
  bump-detection statistic.
- `n_points_T25`, `n_points_T50`: raw point counts (all bands) inside
  those windows.
- `n_points_left`, `n_points_right`: raw point counts strictly before
  / after `t_peak_morph` inside the T50 window.
- `n_contributing_bands`: number of bands with at least one point
  inside the T50 window (or the fallback window if T50 unmeasurable).

No trained classifier is built from these quantities anywhere in this
module -- they are reported as diagnostics only.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

BANDS = ["u", "g", "r", "i", "z", "y"]
MIN_POINTS_FOR_SPLINE = 8
FALLBACK_WINDOW_DAYS = 5.0


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum = np.cumsum(weights)
    cutoff = 0.5 * cum[-1]
    idx = np.searchsorted(cum, cutoff)
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def build_band_excess(curve):
    """
    curve: (N,3) array [time, mag, err_mag] for one band, already
    filtered to finite/positive-error rows by the caller.
    Returns dict with time, excess, err_excess, m_base, or None if the
    band has too few points to be usable at all.
    """
    if curve is None or len(curve) < 3:
        return None

    t = np.asarray(curve[:, 0], dtype=float)
    m = np.asarray(curve[:, 1], dtype=float)
    err_m = np.asarray(curve[:, 2], dtype=float)

    good = np.isfinite(t) & np.isfinite(m) & np.isfinite(err_m) & (err_m > 0)
    t, m, err_m = t[good], m[good], err_m[good]
    if len(t) < 3:
        return None

    order = np.argsort(t)
    t, m, err_m = t[order], m[order], err_m[order]

    w_base = 1.0 / (err_m ** 2)
    m_base = weighted_median(m, w_base)

    excess = 10.0 ** (-0.4 * (m - m_base)) - 1.0
    err_excess = 0.4 * np.log(10.0) * (excess + 1.0) * err_m

    return {
        "t": t, "excess": excess, "err_excess": err_excess,
        "m_base": m_base, "n": len(t),
    }


def fit_band_spline(band_data):
    """Weighted smoothing spline for one band's excess-flux series.
    Returns a callable(t)->excess, or None if too few points."""
    if band_data is None or band_data["n"] < MIN_POINTS_FOR_SPLINE:
        return None

    t = band_data["t"]
    excess = band_data["excess"]
    err_excess = np.clip(band_data["err_excess"], 1e-6, None)
    w = 1.0 / err_excess

    # scipy requires strictly increasing x for UnivariateSpline; average
    # duplicate timestamps (rare, e.g. simultaneous multi-visit) with
    # inverse-variance weights before fitting.
    uniq_t, inv = np.unique(t, return_inverse=True)
    if len(uniq_t) < MIN_POINTS_FOR_SPLINE:
        return None
    if len(uniq_t) != len(t):
        excess_u = np.zeros(len(uniq_t))
        w_u = np.zeros(len(uniq_t))
        for k in range(len(uniq_t)):
            sel = inv == k
            wk = w[sel]
            excess_u[k] = float(np.sum(excess[sel] * wk) / np.sum(wk))
            w_u[k] = float(np.sum(wk))
        t, excess, w = uniq_t, excess_u, w_u
    else:
        pass

    s = float(len(t))
    try:
        spl = UnivariateSpline(t, excess, w=w, k=3, s=s)
    except Exception:
        return None

    return spl, float(t.min()), float(t.max())


def extract_event_morphology(curves):
    """
    curves: dict band -> (N,3) array [time, mag, err_mag], as returned
    by load_h5_lightcurves() (Rubin bands only; W149/Roman excluded --
    this is an LSST-photometry-only morphology signal).
    """
    band_info = {}
    band_splines = {}
    band_weights = {}

    for band in BANDS:
        bd = build_band_excess(curves.get(band))
        if bd is None:
            continue
        band_info[band] = bd

        fit = fit_band_spline(bd)
        if fit is None:
            continue
        spl, tmin, tmax = fit
        median_err = float(np.median(bd["err_excess"]))
        if not np.isfinite(median_err) or median_err <= 0:
            continue
        band_splines[band] = (spl, tmin, tmax)
        band_weights[band] = 1.0 / (median_err ** 2)

    result = {
        "n_bands_with_data": len(band_info),
        "n_bands_splined": len(band_splines),
        "measurable": False,
        "reason": None,
    }

    if len(band_splines) == 0:
        result["reason"] = "not_measurable: no band had enough points for a spline"
        return result

    t_lo = min(v[1] for v in band_splines.values())
    t_hi = max(v[2] for v in band_splines.values())
    grid = np.linspace(t_lo, t_hi, 4001)

    num = np.zeros_like(grid)
    den = 0.0
    coverage = np.zeros_like(grid, dtype=bool)
    for band, (spl, tmin, tmax) in band_splines.items():
        wgt = band_weights[band]
        in_range = (grid >= tmin) & (grid <= tmax)
        vals = np.zeros_like(grid)
        vals[in_range] = spl(grid[in_range])
        num += wgt * vals * in_range
        den_arr = wgt * in_range
        den = den + den_arr
        coverage |= in_range

    with np.errstate(invalid="ignore", divide="ignore"):
        combined = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    combined[~coverage] = np.nan

    if not np.any(coverage):
        result["reason"] = "not_measurable: no time overlap across band splines"
        return result

    valid = np.isfinite(combined)
    peak_idx_local = np.nanargmax(combined[valid])
    t_peak = grid[valid][peak_idx_local]
    peak_val = combined[valid][peak_idx_local]

    if not np.isfinite(peak_val) or peak_val <= 0:
        result["reason"] = "not_measurable: no positive peak excess found"
        return result

    def contiguous_width(q):
        thresh = q * peak_val
        peak_pos = int(np.argmin(np.abs(grid - t_peak)))
        above = np.where(np.isfinite(combined), combined >= thresh, False)

        if not above[peak_pos]:
            return None, None, None, None

        left = peak_pos
        while left > 0 and above[left - 1]:
            left -= 1
        right = peak_pos
        while right < len(grid) - 1 and above[right + 1]:
            right += 1

        left_censored = (left == 0)
        right_censored = (right == len(grid) - 1)

        t_left = grid[left]
        t_right = grid[right]
        return t_left, t_right, left_censored, right_censored

    widths = {}
    for q, label in [(0.25, "T25"), (0.50, "T50"), (0.75, "T75")]:
        t_left, t_right, lc, rc = contiguous_width(q)
        if t_left is None:
            widths[label] = {"value": None, "reason": "not_measurable: peak below threshold at grid resolution"}
            continue

        left_w = t_peak - t_left
        right_w = t_right - t_peak
        entry = {
            "value": None if (lc or rc) else float(t_right - t_left),
            "left_width": None if lc else float(left_w),
            "right_width": None if rc else float(right_w),
            "left_censored": bool(lc),
            "right_censored": bool(rc),
        }
        widths[label] = entry

    T50 = widths["T50"]
    A50 = None
    if T50["left_width"] is not None and T50["right_width"] is not None:
        lw, rw = T50["left_width"], T50["right_width"]
        if (lw + rw) > 0:
            A50 = float((rw - lw) / (rw + lw))

    # quality / SNR window: T50 if fully measurable, else fixed fallback
    if T50["value"] is not None:
        win_lo, win_hi = t_peak - T50["left_width"], t_peak + T50["right_width"]
        window_source = "T50"
    else:
        win_lo, win_hi = t_peak - FALLBACK_WINDOW_DAYS, t_peak + FALLBACK_WINDOW_DAYS
        window_source = "fallback_5d"

    all_excess, all_err, all_t, all_band = [], [], [], []
    for band, bd in band_info.items():
        sel = (bd["t"] >= win_lo) & (bd["t"] <= win_hi)
        all_excess.append(bd["excess"][sel])
        all_err.append(bd["err_excess"][sel])
        all_t.append(bd["t"][sel])
        all_band.extend([band] * int(sel.sum()))
    all_excess = np.concatenate(all_excess) if all_excess else np.array([])
    all_err = np.concatenate(all_err) if all_err else np.array([])
    all_t = np.concatenate(all_t) if all_t else np.array([])

    n_in_window = len(all_excess)
    if n_in_window > 0:
        w_ = 1.0 / np.clip(all_err, 1e-6, None) ** 2
        pooled_sigma = float(1.0 / np.sqrt(np.sum(w_)))
        peak_SN = float(peak_val / pooled_sigma) if pooled_sigma > 0 else np.nan
        integrated_SN = float(np.sqrt(np.sum((all_excess / np.clip(all_err, 1e-6, None)) ** 2)))
    else:
        peak_SN = np.nan
        integrated_SN = np.nan

    n_in_T25 = None
    if widths["T25"]["value"] is not None or (
        widths["T25"].get("left_width") is not None and widths["T25"].get("right_width") is not None
    ):
        t25_lo = t_peak - (widths["T25"].get("left_width") or 0)
        t25_hi = t_peak + (widths["T25"].get("right_width") or 0)
        n_in_T25 = int(np.sum((all_t >= t25_lo) & (all_t <= t25_hi))) if n_in_window else 0

    n_left = int(np.sum(all_t < t_peak)) if n_in_window else 0
    n_right = int(np.sum(all_t >= t_peak)) if n_in_window else 0
    n_bands_in_window = len(set(b for b, tt in zip(all_band, all_t)))

    result.update({
        "measurable": True,
        "reason": None,
        "t_peak_morph": float(t_peak),
        "peak_excess": float(peak_val),
        "T25": widths["T25"],
        "T50": widths["T50"],
        "T75": widths["T75"],
        "A50": A50,
        "window_source": window_source,
        "peak_SN": peak_SN,
        "integrated_SN": integrated_SN,
        "n_points_T25": n_in_T25,
        "n_points_T50": n_in_window if window_source == "T50" else None,
        "n_points_window": n_in_window,
        "n_points_left": n_left,
        "n_points_right": n_right,
        "n_contributing_bands": n_bands_in_window,
    })
    return result
