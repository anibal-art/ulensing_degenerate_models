# Hidden-parallax bounds convergence validation

## Scientific purpose

The final hidden-parallax analysis must support three related questions:

1. Detectability:
   Can annual parallax be distinguished from an FSPL model without parallax?

2. Confusion:
   If parallax is not detected, can a no-parallax FSPL model provide an
   acceptable fit while returning biased values of t0, u0, tE, or rho?

3. Characterization:
   When the event is fitted, how accurately and precisely are the physical
   microlensing parameters recovered?

The likelihood-ratio statistic is

    Delta chi2 = chi2_H0,min - chi2_H1,min

with

    H0 = FSPL without parallax
    H1 = FSPL with annual parallax.

For this statistic to be meaningful, both minima must be robust against
arbitrary truth-dependent fit bounds and against local optimizer basins.

## Why this validation exists

The original production used truth-dependent bounds for the shared physical
parameters. The 34-event audit demonstrated that these bounds can artificially
increase chi2_H0 and therefore inflate the LRT.

Subsequent truth-independent tests compared the following domains:

candidate:
    u0  = [-2, 2]
    tE  = [0.1, 750] d
    rho = [1e-7, 2]
    piE components = [-10, 10]

stress:
    u0  = [-5, 5]
    tE  = [0.1, 2000] d
    rho = [1e-7, 5]
    piE components = [-20, 20]

historical audit:
    u0  = [-5, 5]
    tE  = [0.1, 20000] d
    rho = [1e-7, 10]

Pooling all already-known H0 solutions showed:

- candidate and stress differ because candidate excludes known minima in
  tE and rho for several events;
- stress and the historical audit agree for 33/34 events after pooling;
- the remaining event, catalog_row=83085, has a known H0 solution with
  tE ~ 2890.6 d, outside stress but well inside 5000 d.

Therefore the next validation reference domain is:

reference5000:
    t0  = full temporal range of the fitted Rubin photometry
    u0  = [-5, 5]
    tE  = [0.1, 5000] d
    rho = [1e-7, 5]
    piE components = [-20, 20] for this validation only

The piE limits above are NOT the final production H1 limits. Final H1 bounds
must later be checked against the full generating catalogue.

## Cross-seeding

Independent TRF runs can converge to different local minima even when one
domain contains the other. Therefore domain convergence cannot be assessed
only by comparing independent optimizations.

For the 34-event validation sample we construct diagnostic H0 cross-seeds from
the best already-known H0 solutions in:

- historical audit
- candidate
- stress

Only seeds lying inside reference5000 are retained.

These cross-seeds are a validation device, not a production initialization
strategy. They are allowed to use previously discovered solutions because the
question being asked here is:

    Does the proposed parameter domain contain the relevant likelihood minima?

The final production initialization must remain truth-independent.

## Acceptance criteria for reference5000

For each event:

1. The reference5000 H0 refit should reproduce or improve the best previously
   known H0 solution that lies inside reference5000.

2. We use a tolerance of

       Delta chi2 <= 0.1

   for the bounds-convergence diagnostic.

3. Pooling every known solution should not reveal a solution with lower chi2
   outside reference5000 by more than 0.1.

4. The winning solution should not be pinned to a physical bound.

If these criteria hold for the 34-event diagnostic sample, the next step is
to repeat the test on the pre-selected extreme-parameter sample.

## Important distinction

Bounds convergence and optimizer convergence are separate questions.

This validation determines whether the likelihood domain is sufficiently
broad. It does not establish the final production multistart strategy.

After bounds are frozen, production initialization and optimizer robustness
will be validated separately using truth-independent starts.

## Downstream science

Once bounds and optimization are frozen, the production must preserve enough
information for:

- empirical H0 LRT calibration;
- parallax detection power;
- hidden/confused-parallax classification;
- H0 bias and formal uncertainty in t0, u0, tE, rho;
- H1 bias and uncertainty in t0, u0, tE, rho, piEN, piEE;
- z-scores and empirical coverage;
- multimodality diagnostics from the multistart solutions;
- targeted profile likelihood analysis of problematic subsets.


## Result: 34-event reference5000 validation

The cross-seeded reference5000 validation was completed on the original
34-event tail/control diagnostic sample.

Results:

- N events: 34
- reference5000 refits worse than the best previously known feasible
  H0 minimum by Delta chi2 > 0.1: 0/34
- maximum refit-minus-preexisting Delta chi2:
  1.09139364e-11
- events with a known better H0 solution outside reference5000 by
  Delta chi2 > 0.1: 0/34
- maximum pooled reference5000-minus-historical-domain Delta chi2: 0
- pooled H0 winners at or near a physical reference5000 bound: 0/34

The previously problematic event catalog_row=83085 recovers the known
historical H0 solution with tE approximately 2890.63 d inside the
reference5000 domain.

Conclusion:

    reference5000 is accepted as the shared-parameter reference domain
    for the 34-event bounds validation sample.

This does not yet establish final production bounds. The next required
test is the independently selected 100-event extreme-parameter sample.

The H1 optimization/bounds problem remains a separate validation task.
Several reference5000 runs found H1 minima below the original production
H1 minimum, which matters for both the LRT and parameter characterization.
