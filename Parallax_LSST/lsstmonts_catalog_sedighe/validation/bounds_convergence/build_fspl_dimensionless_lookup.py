#!/usr/bin/env python3
"""
M3 -- FSPL dimensionless morphology lookup table (frozen before any
comparison against oracle/truth/old_H0).

For a grid in (|u0|, rho), compute the pure (noise-free) FSPL
magnification shape (pyLIMA's Yoo et al. 2004 finite-source
approximation, no limb darkening: gamma=0, a fixed, generic,
non-tuned choice) as a function of dimensionless time
tau=(t-t0)/tE, and extract the width ratios T25/T50 and T75/T50 (in
units of tE -- these ratios do not depend on tE or t0 by
construction).

Degeneracy check (frozen, verified once, not re-derived per event):
magnification_FSPL_Yoo(tau, u0, rho, gamma) is bit-identical under
u0 -> -u0 (confirmed: max|A(u0)-A(-u0)| = 0.0 over a test grid), so
H0 is EXACTLY degenerate under u0 -> -u0. The lookup therefore only
covers |u0| >= 0 and every morphology candidate downstream uses the
canonical convention u0_morph > 0 -- no mirrored duplicate starts.

Grid: |u0| log-spaced [1e-3, 3], rho log-spaced [1e-4, 2] -- chosen to
cover the physically-detectable-finite-source-effect regime, not
retuned against extreme100 results (built and frozen before any such
comparison). tau grid: [-15, 15], step 0.005 (6001 points), fixed and
generic; grid points whose T50 (in tau units) is not resolved by at
least 5 grid steps are flagged and excluded rather than silently
reported.
"""
import os
import numpy as np
import pandas as pd
from pyLIMA.magnification.magnification_FSPL import magnification_FSPL_Yoo

HERE = os.path.dirname(os.path.abspath(__file__))

N_U0 = 60
N_RHO = 60
U0_MIN, U0_MAX = 1.0e-3, 3.0
RHO_MIN, RHO_MAX = 1.0e-4, 2.0

TAU_MAX = 15.0
TAU_STEP = 0.005
GAMMA = 0.0
MIN_RESOLVED_STEPS = 5

tau = np.arange(0.0, TAU_MAX + TAU_STEP, TAU_STEP)  # positive half only (symmetry)

u0_grid = np.geomspace(U0_MIN, U0_MAX, N_U0)
rho_grid = np.geomspace(RHO_MIN, RHO_MAX, N_RHO)


def half_width_ratio(excess_half, q, peak):
    thresh = q * peak
    above = excess_half >= thresh
    if not above[0]:
        return None, "peak_below_threshold"
    if above.all():
        return None, "not_resolved_within_tau_max"
    idx = np.argmax(~above)  # first False
    if idx < MIN_RESOLVED_STEPS:
        return None, "grid_resolution_insufficient"
    # linear interpolation between idx-1 (above) and idx (below)
    e0, e1 = excess_half[idx - 1], excess_half[idx]
    t0_, t1_ = tau[idx - 1], tau[idx]
    if e0 == e1:
        t_cross = t0_
    else:
        frac = (thresh - e0) / (e1 - e0)
        t_cross = t0_ + frac * (t1_ - t0_)
    return float(t_cross), None


records = []
for u0 in u0_grid:
    for rho in rho_grid:
        A = magnification_FSPL_Yoo(tau, float(u0), float(rho), GAMMA)
        excess_half = A - 1.0
        peak = float(excess_half[0])
        if not np.isfinite(peak) or peak <= 0:
            continue

        hw25, r25 = half_width_ratio(excess_half, 0.25, peak)
        hw50, r50 = half_width_ratio(excess_half, 0.50, peak)
        hw75, r75 = half_width_ratio(excess_half, 0.75, peak)

        if hw25 is None or hw50 is None or hw75 is None or hw50 <= 0:
            continue

        records.append({
            "u0": float(u0),
            "rho": float(rho),
            "peak_excess_model": peak,
            "T50_over_tE": 2.0 * hw50,
            "T25_over_T50": hw25 / hw50,
            "T75_over_T50": hw75 / hw50,
        })

df = pd.DataFrame(records)
out_csv = os.path.join(HERE, "results", "fspl_dimensionless_lookup.csv")
df.to_csv(out_csv, index=False)
print("grid points requested:", N_U0 * N_RHO, "usable:", len(df))
print(df[["T25_over_T50", "T75_over_T50", "T50_over_tE"]].describe())
print("DONE ->", out_csv)
