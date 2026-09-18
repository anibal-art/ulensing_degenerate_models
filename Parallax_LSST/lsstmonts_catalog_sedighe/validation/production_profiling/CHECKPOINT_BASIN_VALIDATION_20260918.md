# Hidden-parallax basin-validation checkpoint

Date: 2026-09-18

## LRT

The statistic is

D = chi2_H0 - chi2_H1,

with

H0 = FSPL(t0, u0, tE, rho)

and

H1 = FSPL(t0, u0, tE, rho, piEN, piEE).

H0 is nested in H1 at piEN = piEE = 0.

This development sample is H0-generated. Therefore an H1 minimum is not
a recovered physical parallax signal; it is the minimum of the alternative
model for a particular null realization.

No alpha or detection threshold has been selected from this sample.

## H1 basin validation

The original production-candidate single nested H1 start is not sufficient.

On the frozen N=25 H0-generated development sample, a validation-only
34-start reference found non-negligible policy gaps:

- median gap ~0.15
- 10/25 events with gap > 0.5
- 7/25 events with gap > 1
- maximum gap ~4.88

A small fixed subset of starts did not generalize reliably in leave-one-out
tests.

Initial seed chi2 is not predictive of the basin reached after TRF. The
relevant object is effectively

B(x) = chi2[TRF(x)]

rather than the initial chi2(x).

Blind full-domain DE and transformed-coordinate DE did not reliably recover
the best-known H1 basin.

Full 6D Sobol + TRF recovered the difficult basin for row 608288, but failed
for rows 134368 and 779307 at 32 additional starts.

## Observable-space interpretation

Three difficult events were examined:

### 608288

High S/N, observed peak, moderate/asymmetric temporal coverage.

The best-known H1 solution differs radically in physical parameters from H0,
but the predicted observed fluxes differ by less than ~0.9 sigma per point.

### 134368

Excellent temporal coverage and extremely high S/N.

Despite this, H0 and the best-known H1 solution differ by less than ~0.5 sigma
per observation.

This shows that poor sampling is not required for strong H1 degeneracy.

### 779307

Good S/N but poor peak sampling:

- no points within |tau| <= 0.25
- only two points within |tau| <= 0.5

Most of the H1 improvement occurs in the wings, especially pre-peak.

## New H0 basin result

The profiled piE diagnostic revealed an additional morphology branch for
row 134368 at piE = 0.

Stored production-candidate H0:

chi2 = 473.78454688145683

New morphology:

t0  = 2463659.815943233
u0  = -0.14504007039311217
tE  = 46.635007713959425
rho = 0.10733028733213777

Direct pure-H0 evaluation gives:

chi2 = 473.3771052321031

The same parameters evaluated as nested H1 with

piEN = piEE = 0

give exactly the same chi2.

Therefore the stored H0 solution missed a lower basin by

Delta chi2 = 0.40744164935375693.

Using the current best-known H1 solution for row 134368,

chi2_H1 = 471.7771861565897,

the robust statistic becomes

D = 1.5999190755,

rather than approximately 2.0073 from the stored H0.

## Current interpretation

The fitting problem contains at least two distinct sources of multimodality:

1. multimodality in parallax space;
2. multimodality in the shared morphology
   (t0, u0, tE, rho), even at fixed piE.

Correct LRT calibration therefore requires sufficiently robust minimization
of BOTH H0 and H1.

## H0 basin audit on the frozen N=25 sample

A development-only H0 morphology audit was performed on all 25 frozen
H0-generated events.

For every event, the same pure-H0 objective was minimized from:

1. the stored H0 morphology;
2. its u0 mirror;
3. the shared (t0, u0, tE, rho) morphology of the H1 reference winner;
4. its u0 mirror.

The mirrors gave identical H0 minima, as expected from the u0 sign symmetry
of the no-parallax model. Therefore this audit effectively probes two
independent morphology starts.

The H0 basin gap was defined as

Delta_chi2_H0 =
    chi2_H0_policy - chi2_H0_audit.

Results:

- N = 25
- median gap = 0.0253
- p90 gap = 1.375
- p95 gap = 1.665
- maximum gap = 2.152

Counts:

- gap > 0.001: 20 / 25
- gap > 0.01: 15 / 25
- gap > 0.1: 11 / 25
- gap > 0.5: 6 / 25
- gap > 1: 4 / 25

The largest gaps include:

- row 635703: 2.152
- row 375851: 1.711
- row 846704: 1.479
- row 582916: 1.219
- row 722416: 0.913
- row 933999: 0.571

Two distinct failure modes are present.

### A. Incomplete local convergence

For some events, rerunning TRF from the stored H0 solution itself lowers
chi2 substantially.

The strongest example is row 635703:

stored H0:
chi2 = 277.584552

continued H0:
chi2 = 275.432992

improvement:
Delta_chi2 = 2.151561

The stored point evaluates to the stored chi2 exactly before the new
optimization, so this is not an objective mismatch.

It is evidence that the original fit was not fully settled under the same
objective.

### B. Alternative morphology basins

For many events, the best audited H0 solution is reached only from the
shared morphology discovered by the H1 reference fit.

Examples include rows:

- 375851
- 846704
- 582916
- 722416
- 933999
- 134368
- 779307

This demonstrates genuine morphology-basin structure in

(t0, u0, tE, rho)

even when piE is fixed to zero.

### Consequence for the LRT

An insufficiently minimized H0 biases

D = chi2_H0 - chi2_H1

upward.

For example, using the H1 reference solution:

- row 375851: D changes from 2.302 to 0.591
- row 635703: D changes from 3.554 to 1.403
- row 582916: D changes from 2.414 to 1.196
- row 722416: D changes from 1.670 to 0.757
- row 933999: D changes from 0.738 to 0.167

Conversely, an insufficiently minimized H1 biases D downward.

Therefore the relevant optimization diagnostic is the combined change

Delta_D =
    Delta_chi2_H1_basin
    - Delta_chi2_H0_basin.

Both hypotheses must be minimized robustly before the null distribution of D
can be calibrated.

### Special case: row 134368

The N=25 audit reaches

chi2_H0 = 473.379695

from the shared morphology of the original 34-start H1 reference winner.

A later, more refined diagnostic reaches

chi2_H0 = 473.3771052321031.

The latter remains the current best-known H0 value for this event.

This difference emphasizes that the N=25 audit is a diagnostic lower
baseline, not a proof of the global H0 minimum.

## Combined H0/H1 effect on the LRT statistic

The H0 and H1 basin corrections were combined event by event.

Define

G_H0 =
    chi2_H0_policy - chi2_H0_audit

and

G_H1 =
    chi2_H1_policy - chi2_H1_reference.

Then

D_audit =
    D_policy + G_H1 - G_H0,

so that

Delta_D =
    D_audit - D_policy
    = G_H1 - G_H0.

Positive Delta_D means that the H1 optimization failure dominated and the
original policy underestimated D.

Negative Delta_D means that the H0 optimization failure dominated and the
original policy overestimated D.

On the frozen N=25 development sample:

- H1 failure dominates: 15 / 25
- H0 failure dominates: 10 / 25
- balanced within 1e-6: 0 / 25

The signed Delta_D distribution is:

- minimum = -2.053
- median = 0.0034
- mean = 0.700
- p90 = 3.007
- maximum = 4.883

The near-zero signed median is misleading because the event-by-event
distortion is substantial.

For |Delta_D|:

- median = 0.351
- p90 = 3.007
- p95 = 3.912
- maximum = 4.883

Counts:

- |Delta_D| > 0.01: 20 / 25
- |Delta_D| > 0.1: 16 / 25
- |Delta_D| > 0.5: 12 / 25
- |Delta_D| > 1: 9 / 25
- |Delta_D| > 2: 7 / 25

Examples where the H1 failure strongly dominates include:

- row 28565:
  D_policy = 1.103
  D_audit = 5.986

- row 779307:
  D_policy = 1.940
  D_audit = 6.062

- row 806393:
  D_policy = 0.455
  D_audit = 3.522

Examples where the H0 failure dominates include:

- row 635703:
  D_policy = 3.456
  D_audit = 1.403

- row 375851:
  D_policy = 2.271
  D_audit = 0.591

- row 722416:
  D_policy = 1.392
  D_audit = 0.757

There are no nesting violations after the combined audit:

D_audit < -1e-6: 0 / 25.

This is a necessary consistency check but is not evidence that either
hypothesis has reached its mathematical global minimum.

The principal conclusion is that optimizer error does not introduce a simple
one-directional shift in D. H0 and H1 failures act in opposite directions,
and their relative importance varies substantially from event to event.

Therefore final null calibration requires the H0 and H1 optimization
architectures to be frozen jointly before the independent calibration sample
is generated.

## Revised next validation steps

Completed in this checkpoint:

1. quantified H1 basin failures on the frozen N=25 development sample;
2. identified and quantified H0 basin/convergence failures;
3. separated incomplete local H0 convergence from alternative morphology
   basins;
4. quantified the combined H0/H1 distortion of D event by event.

Next, before alpha calibration:

1. design and validate a cheap robust H0 production safeguard;
2. return to the profiled piE maps for representative difficult events;
3. finalize the H1 production architecture;
4. test the joint H0/H1 candidate policy on the frozen development sample;
5. run throughput validation;
6. freeze the optimizer;
7. calibrate the H0 null distribution on an independent sample;
8. validate power on an independent H1-generated holdout.

The frozen N=25 development sample must not be reused for final alpha
calibration.

The frozen N=25 development sample must not be reused for final alpha
calibration.
