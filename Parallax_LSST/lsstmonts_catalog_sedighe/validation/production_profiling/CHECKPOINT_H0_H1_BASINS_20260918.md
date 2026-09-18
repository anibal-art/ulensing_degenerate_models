# H0-generated H1 basin validation checkpoint

Date: 2026-09-18

## Scope

This checkpoint summarizes the development-stage validation of the H1 optimizer for the hidden-parallax likelihood-ratio test.

The statistic is

D = chi2_H0 - chi2_H1,

where:

- H0 = FSPL without annual parallax:
  (t0, u0, tE, rho)
- H1 = FSPL with annual parallax:
  (t0, u0, tE, rho, piEN, piEE)

H0 is nested in H1 at

piEN = piEE = 0.

All fluxes are profiled analytically with the bounded WLS implementation used by the production-candidate fit.

This stage uses H0-generated events. Therefore, the best H1 solution is not a "true parallax" solution. It is the minimum of the alternative model for a particular null realization and is required for correct calibration of the LRT null distribution.

No detection threshold or alpha has been calibrated from this development sample.

---

## 1. Objective parity

The DE / profiled objective was evaluated directly at stored reference solutions.

For row 779307:

- stored H0 chi2 = 277.94852476493537
- objective at exact nested H0 = 277.9485247649354
- difference = 5.7e-14

- stored best-local H1 chi2 = 271.68907522695304
- objective at that point = 271.6890752269529
- difference = -1.1e-13

The later observable-space decomposition also reconstructed H0, H1-policy, and H1-oracle chi2 values point by point to approximately 1e-13 for all three pathological events.

Conclusion:

The optimizer difficulties are not caused by an objective-function mismatch, flux-profiling mismatch, or inconsistency between stored and reevaluated solutions.

---

## 2. Robust H1 reference on the frozen N=25 H0 sample

A validation-only 34-start H1 reference was constructed from the settled H0 morphology.

Starts varied:

- sign of u0
- piE = 0
- piE radii = 0.03, 0.3, 3, 30
- angles = 0, 90, 180, 270 deg

All 34 starts were finite for all 25 events.

For

G_H1 = chi2_H1_policy - chi2_H1_reference,

the distribution was approximately:

- min = 0.00079
- median = 0.148
- p90 = 3.79
- p95 = 4.37
- max = 4.88

Counts:

- G > 0.1: 14 / 25
- G > 0.5: 10 / 25
- G > 1: 7 / 25
- G > 5: 0 / 25

Conclusion:

The original single nested H1 start is not sufficient.

The 34-start reference is a best-known robust reference, not a mathematical proof of the global minimum.

---

## 3. Set-cover reduction does not generalize reliably

An exact set-cover optimization found compact subsets of the 34 starts that reproduce the development sample well.

However, leave-one-out tests showed poor generalization.

At tolerance 0.5:

- held-out max gap ≈ 4.27
- held-out failures > 0.5: 3 / 25
- held-out failures > 1: 3 / 25

At tolerance 0.1:

- held-out max gap ≈ 4.27
- held-out failures > 0.1: 4 / 25
- held-out failures > 0.5: 2 / 25
- held-out failures > 1: 2 / 25

Conclusion:

A fixed small subset learned from these 25 events should not be frozen as a production rule.

---

## 4. Local stability of three pathological events

Rows:

- 608288
- 134368
- 779307

### 608288

Best-known chi2:

372.661494102

The minimum is extremely locally stable.

Final parameters are approximately:

- u0 = 0.0771
- tE = 1060 d
- rho = 0.103
- piEN = -0.937
- piEE = 0.150

### 134368

Best-known chi2:

471.777247721

The good basin is locally stable but has a narrow entry region.

Final parameters are approximately:

- u0 = -0.165
- tE = 44.05 d
- rho = 0.147
- piEN = 0.064
- piEE = -0.134

### 779307

Best-known chi2:

271.689075227

A locally stable solution below the original 34-start reference was found.

Final parameters are approximately:

- u0 = -0.283
- tE = 216.77 d
- rho = 0.467
- piEN = 0.0397
- piEE = 0.1114

Conclusion:

Local minimum stability and ease of global basin discovery are different questions.

---

## 5. Seed chi2 is not predictive of final TRF basin

The initial chi2 of all 34 reference seeds was compared with the final chi2 after TRF.

The best final basin is frequently reached from a poor initial seed.

Examples include oracle-entry initial ranks near:

- 779307: 15 / 34
- 134368: 26 / 34
- 608288: 29 / 34

Other events have oracle-entry ranks as poor as 32 / 34 and 34 / 34.

The median initial rank of the oracle-producing seed is approximately 21 / 34.

Selecting only the best seeds according to pre-TRF chi2 therefore fails.

Conclusion:

The relevant basin-discovery quantity is not

f(x) = chi2(x),

but conceptually

B(x) = chi2[TRF(x)].

A seed can have a very poor initial chi2 and still lie inside the attraction region of the best local minimum.

---

## 6. Global DE tests

### Raw physical-domain DE

For row 779307, blind DE in the full physical 6D box converged to:

chi2 ≈ 425,

far worse than both the policy and best-known solution.

### Transformed-coordinate DE

Using:

- asinh t0 offset
- asinh u0
- log10 tE
- log10 rho
- asinh piEN
- asinh piEE

improved exploration substantially but still converged to:

chi2_DE+TRF ≈ 277.28,

versus:

chi2_target ≈ 271.69.

It entered a near-point-source basin with very small rho.

### 2D parallax DE with H0 morphology fixed

Optimizing piEN and piEE while holding the shared H0 morphology fixed also failed.

For row 779307:

chi2_DE+TRF ≈ 276.04.

Conclusion:

Choosing a global candidate according to its pre-TRF chi2 is not sufficient.

---

## 7. Sobol 6D + TRF

A global-local test was performed in which every Sobol proposal was passed directly to a full six-parameter TRF fit.

No preselection by initial chi2 was used.

At K = 32:

### 608288

The correct basin was already found at Sobol start 6.

Best gap to the known target:

approximately 1e-9.

The basin was rediscovered by several additional starts.

### 134368

No Sobol start improved the existing policy.

Final gap:

approximately 1.843.

### 779307

No Sobol start improved the existing policy.

Final gap:

approximately 4.320.

Conclusion:

A quasi-uniform 6D multistart can find some important basins, but the attraction volume of the difficult basins in 134368 and 779307 is too small for this to be a reliable production strategy at this budget.

---

## 8. Observational quality of the three pathological events

### 608288

True event:

- tE ≈ 28.54 d
- 405 observations
- 14 points within |tau| <= 1
- 6 within |tau| <= 0.25
- nearest point ≈ 0.008 tE from the peak
- 21 points in 0.5 < |tau| <= 2
- pre/post within tE = 3 / 11
- median model-signal significance within tE ≈ 61 sigma

Interpretation:

High S/N and observed peak, but temporally asymmetric and only moderately sampled around the event.

### 134368

True event:

- tE ≈ 47.52 d
- 539 observations
- 45 points within |tau| <= 1
- 17 within |tau| <= 0.25
- nearest point ≈ 0.004 tE
- 42 points in 0.5 < |tau| <= 2
- pre/post within tE = 28 / 17
- median model-signal significance within tE ≈ 172 sigma

Interpretation:

Excellent coverage and extremely high signal-to-noise.

Poor sampling cannot explain its H1 multimodality.

### 779307

True event:

- tE ≈ 190.61 d
- 289 observations
- 27 points within |tau| <= 1
- 0 points within |tau| <= 0.25
- only 2 points within |tau| <= 0.5
- nearest point ≈ 0.295 tE from the peak
- 38 points in 0.5 < |tau| <= 2
- pre/post within tE = 19 / 8
- median model-signal significance within tE ≈ 19 sigma

Interpretation:

The event is well detected, but its peak is poorly sampled.

This can substantially weaken morphological identifiability.

---

## 9. Observable-space decomposition

The stored H0, H1-policy, and H1-oracle chi2 values were reconstructed exactly as sums of per-observation contributions.

### 608288

- D_policy = 1.165
- D_oracle = 3.648
- oracle gain over policy = 2.482
- maximum H0-oracle model separation ≈ 0.90 sigma

Most of the oracle improvement occurs in:

0.5 < |tau| < 1,

with delta chi2 ≈ 2.60.

Band i contributes most of the improvement.

The oracle and H0 curves remain observationally very similar despite radically different physical parameters.

### 134368

- D_policy = 0.165
- D_oracle = 2.007
- oracle gain over policy = 1.843
- maximum H0-oracle model separation ≈ 0.50 sigma

The improvement is distributed through small sub-sigma changes, mainly near the peak.

Approximate phase contributions:

- |tau| <= 0.25: +1.02
- 0.25 < |tau| <= 0.5: +1.21
- baseline: -0.44

Band i contributes approximately +2.89, while z contributes approximately -1.18.

Interpretation:

This is a strong observable-space degeneracy even for an exceptionally well-observed event.

### 779307

- D_policy = 1.940
- D_oracle = 6.259
- oracle gain over policy = 4.320
- maximum H0-oracle separation ≈ 1.73 sigma
- maximum policy-oracle separation ≈ 1.97 sigma

The oracle is not better near the poorly sampled peak.

Approximate phase contributions:

- 0.25 < |tau| <= 0.5: -0.67
- 0.5 < |tau| <= 1: -1.40
- 1 < |tau| <= 2: +8.20

The improvement is strongly asymmetric:

- pre-peak contribution inside |tau| <= 2: ≈ 4.92
- post-peak contribution: ≈ 1.21

Bands r and z dominate.

Interpretation:

The missing peak information allows a substantially different H1 morphology to sacrifice the center while fitting one wing better.

---

## 10. Current interpretation

The H1 fitting problem is not adequately described as only a local-optimizer failure.

The evidence supports the combination of:

1. a strongly multimodal H1 objective surface;
2. very different physical parameter vectors mapping to nearly indistinguishable observed light curves;
3. narrow attraction regions for some important minima;
4. additional loss of identifiability when the peak is poorly sampled.

For H0-generated data this is especially important because the best H1 minimum represents noise accommodation under the alternative model, not recovery of a physical parallax signal.

Correct null calibration therefore requires a fitting architecture that reliably approximates the global H1 minimum.

---

## 11. Next diagnostic

Before selecting a production optimizer, map the profiled H1 surface

chi2_prof(piEN, piEE)
    =
min_{t0,u0,tE,rho}
chi2(t0,u0,tE,rho,piEN,piEE)

for the three pathological events.

The map should identify:

- separated islands;
- elongated valleys;
- sign degeneracies;
- narrow connecting regions;
- the locations of H0, the current policy minimum, and the best-known H1 minimum.

This is a diagnostic analysis. Known best-local solutions may be used as branch seeds to reveal the topology, but such oracle-assisted maps must not be interpreted as an independent validation of a production fitting rule.

---

## Validation status

Completed:

- objective parity
- H0-generated H1 basin audit
- robust 34-start reference
- set-cover analysis
- local stability
- seed-predictiveness analysis
- physical-domain DE test
- transformed-coordinate DE test
- piE-only DE test
- Sobol 6D + TRF test
- observational-quality audit
- observable-space decomposition

Not completed:

- profiled piE surface mapping
- final H1 production architecture
- throughput pilot
- alpha selection
- independent H0 calibration
- independent H1 holdout

The development sample must not be reused for final threshold calibration.
