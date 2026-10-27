"""
Deterministic, truth-blind (t0,u0,tE,rho) seed derived only from
observable light-curve morphology.

morphology_seed_from_curves(curves) takes ONLY the light curves --
its signature has no truth/event_params argument, so no injected
truth parameter can enter it even by mistake. This is the seed used
for whichever model is NOT the one that generated the simulation
(H1 fit on an H0-generated event, or H0 fit on an H1-generated
event), per the simplified 2-fit production architecture.

Reuses the already-frozen morphology extraction (T25/T50/T75 excess-
flux widths, peak time, asymmetry) and the already-frozen FSPL
dimensionless (u0,rho) lookup table -- both built and validated in
validation/bounds_convergence/ -- unchanged. No new estimator,
smoothing, GP, or classifier is introduced.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BC = os.path.join(HERE, "..", "bounds_convergence")
LOOKUP_PATH = os.path.join(BC, "results", "fspl_dimensionless_lookup.csv")

from morphology_extraction import extract_event_morphology  # noqa: E402

_LOOKUP = None


def _lookup():
    global _LOOKUP
    if _LOOKUP is None:
        _LOOKUP = pd.read_csv(LOOKUP_PATH)
    return _LOOKUP


def _effective_width(entry):
    if entry is None:
        return None
    if not entry["left_censored"] and not entry["right_censored"] and entry["value"] is not None:
        return float(entry["value"])
    if not entry["left_censored"] and entry["left_width"] is not None and entry["right_censored"]:
        return 2.0 * float(entry["left_width"])
    if not entry["right_censored"] and entry["right_width"] is not None and entry["left_censored"]:
        return 2.0 * float(entry["right_width"])
    return None


def morphology_seed_from_curves(curves):
    """
    curves: the SAME dict of per-band arrays produced by
    run_bounds_audit_refit_core.load_h5_lightcurves() -- observable
    photometry only. No truth is accepted or used.

    Returns {"t0","u0","tE","rho","mismatch"} (a deterministic rank-1
    nearest-match against the FSPL dimensionless lookup table) or
    None if the event's morphology is not measurable (e.g. insufficient
    coverage around peak) -- callers must treat None as a genuine
    estimator/domain inconsistency, not silently substitute a fallback.
    """
    morph_raw = extract_event_morphology(curves)

    if not morph_raw.get("measurable"):
        return None

    e25 = _effective_width(morph_raw["T25"])
    e50 = _effective_width(morph_raw["T50"])
    e75 = _effective_width(morph_raw["T75"])

    if not (e25 and e50 and e75 and e50 > 0):
        return None

    lookup = _lookup()
    obs25, obs75 = e25 / e50, e75 / e50
    mismatch = np.sqrt(
        (lookup["T25_over_T50"].values - obs25) ** 2
        + (lookup["T75_over_T50"].values - obs75) ** 2
    )
    best_idx = int(np.argmin(mismatch))

    u0 = float(lookup["u0"].values[best_idx])
    rho = float(lookup["rho"].values[best_idx])
    tE = e50 / float(lookup["T50_over_tE"].values[best_idx])

    return {
        "t0": float(morph_raw["t_peak_morph"]),
        "u0": u0,
        "tE": tE,
        "rho": rho,
        "mismatch": float(mismatch[best_idx]),
    }
