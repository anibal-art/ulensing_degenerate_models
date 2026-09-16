# Phase A: geometry of the 4 residual cases (zero new fits)

All quantities below are read from already-stored results
(`basin_risk_development_table.csv`, `h1_projected_h0_results.csv`,
`reference_v2_independent.csv`, and the underlying per-mode/per-strategy
JSON files already produced by earlier orchestrator runs, e.g.
`results/pipeline_stage_json/{row}_h0_{mode}.json`,
`{row}_h1_controlled4.json`, `{row}_bridge.json`). No fit was rerun.
Full numeric table: `residual_geometry.csv`.

## Q1: Do 520062 and 324312 share a common H0 geometric failure mode?

| | 520062 | 324312 |
|---|---|---|
| independent-reference winner | `physical/old_H0/1.0`, chi2=156.16 | `physical/old_H0/1.0`, chi2=368.85 |
| candidate H0 (final, min(H0_1,H0_2)) | H0_1 wins, chi2=163.39 | H0_1 wins, chi2=373.98 |
| `dlogrho` (candidate vs reference) | **-1.63** (H0_1) / -2.23 (H0_2) | **+0.01** (H0_1) / -0.99 (H0_2) |
| `dlogtE` | +0.124 (H0_1) / +0.126 (H0_2) | -0.228 (H0_1) / -0.137 (H0_2) |
| `du0` | -0.55 (H0_1) / -0.57 (H0_2) | +0.10 (H0_1) / -0.04 (H0_2) |
| `sign(u0)` flip vs reference | No (both positive) | No (both positive) |
| `dt0_over_tE` | 0.065 | 0.25 |

**Using the actual winning H0 (whichever of H0_1/H0_2 is lower,
matching what feeds `D_s2`)**, there is **no clear common geometric
failure mode** across the two H0-limited residual events.

For 520062, the winning H0_1 solution differs substantially from the
independent reference in `rho` (`dlogrho=-1.63`), while `tE`, `u0`,
and `t0/tE` remain comparatively closer.

For 324312, however, the winning H0_1 solution has
`dlogrho=+0.01`, so its `rho` is essentially matched to the independent
reference. The approximately order-of-magnitude `rho` discrepancy
(`dlogrho=-0.99`) belongs to H0_2, which does not win and therefore is
not the solution entering `D_s2`.

Both residual events have no `u0` sign flip, but this alone is not a
sufficiently specific shared failure geometry.

**Conclusion: no clear common H0 geometric failure mode is identified.**
The two residual events do not justify introducing a third deterministic
H0 start from the present n=2 evidence.

## Q2: Do 443054 and 114675 share a common H1 geometric failure mode?

| | 443054 | 114675 |
|---|---|---|
| independent-reference H1 | `H0_NESTED_piE_0`, chi2=322.40 | `H0_NESTED_piE_0`, chi2=261.35 |
| candidate H1 (official settled) | chi2=325.48 | chi2=263.18 |
| `sign(u0)` flip vs reference | No (0.77 vs 1.43, both positive) | **Yes** (0.75 vs -0.26) |
| `dlogrho` | -2.16 (candidate rho 0.163 vs ref 1.431, ~8.8x smaller) | **-4.43** (candidate rho 0.026 vs ref 2.195, ~84x smaller) |
| `dlogtE` | +0.225 (61.4 vs 49.1, modest) | **+1.163** (143.0 vs 44.7, ~3.2x larger) |
| `piE_candidate` vs `piE_reference` | 0.142 vs 0.206 (candidate SMALLER) | 0.959 vs 0.460 (candidate LARGER) |
| `delta_piE` | -0.064 | **+0.499** |
| parallax-vector angle (deg) | 90.5 vs 100.8 (Δ=10.3°, similar direction) | 78.6 vs 74.9 (Δ=3.7°, similar direction) |

Both events involve a large `rho` discrepancy and both land in the
same qualitative region (the well-known FSPL `u0`-`rho`-`tE`-piE joint
degeneracy valley: a "large u0, large rho" branch vs a "small u0, small
rho" branch trading off against `tE` and `piE` to preserve the observed
light-curve shape). But the SPECIFIC displacement is not consistent
between the two: 443054 has no `u0` sign flip and a SMALLER candidate
`piE`; 114675 has a `u0` sign flip and a LARGER candidate `piE`. The
parallax-vector angle is reasonably close in both (`<=10.3°`), which is
the one genuinely common, consistent feature.

**Conclusion: weak/suggestive common pattern (shared degeneracy family,
similar parallax-vector direction), not a clear one** (opposite `piE`
magnitude direction, only one has a `u0` sign flip).

## Q3: Does either pair justify a new deterministic fitter mechanism now?

**No.** Per the predefined default, and given n=2 for each pair with
genuinely differing specifics (not a single uniform displacement), this
does not meet the "exceptionally obvious" bar needed to justify
designing a new mechanism from 2 examples. No third H0 start, no H1
rescue redesign, and no morphology change are proposed here. Both
directions the residual geometry pointed to -- `rho` sensitivity on the
H0 side, and the `u0`-`rho`-`piE` degeneracy valley on the H1 side --
are consistent with what is already scientifically expected for FSPL
fits, not a newly discovered, fixable defect.
