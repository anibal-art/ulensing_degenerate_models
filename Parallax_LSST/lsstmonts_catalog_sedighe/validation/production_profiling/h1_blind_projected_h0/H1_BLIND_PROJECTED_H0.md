# H1-blind projected-H0 negative control

## Question

Could the successful H1-projected second H0 start be made independent
of the H1 truth initialization by replacing the official truth-started
H1 fit with a blind H1 fit initialized from deterministic morphology
and `piEN=piEE=0`?

This experiment is a negative control. Blind real-data operation is not
the scientific objective of the simulation study; truth initialization
is allowed for the model matching the generating truth.

## Experiment

For each H1-generated development event:

1. run H1 from morphology-derived `(t0,u0,tE,rho)` with
   `piEN=piEE=0`, all six nonlinear parameters free;
2. settle that H1 fit using the existing numerical safeguard;
3. project its settled shared parameters into a second H0 start;
4. compare the resulting LRT against the same frozen independent
   reference used for the H1-projected-H0 validation.

The official truth-started H1 fit is not replaced in the primary LRT.

## Result

Development sample:

- 93 clean H1-generated events;
- 15 dangerous events under the original one-H0-start candidate;
- truth-started H1 projection repaired 11/15 dangerous events;
- blind-H1 projection repaired only 2/15;
- 13 dangerous cases remained after the blind variant;
- the blind variant produced no evidence of a robust replacement for
  the truth-started H1 projection.

The blind H1 likelihood itself was substantially less stable than the
official truth-started H1 fit. For

    chi2_H1_blind - chi2_H1_official

the distribution had approximately:

- median: 3.64;
- p90: 1406.9;
- p95: 3415.9;
- maximum: 42516.4.

The blind variant also produced one newly missed development case
(row 114675).

## Interpretation

The successful H0_2 mechanism depends on obtaining a sufficiently good
H1 solution before projecting its shared parameters into H0.

A morphology-plus-zero-parallax H1 start does not reliably find that H1
basin. Its poor performance is therefore evidence against using it as
the production H1 initializer.

This does not constitute a methodological problem for the study:
the target is a simulation-based likelihood-ratio detectability
calculation, not a blind real-data search. The generating-model truth
may be used as an initialization aid while all fitted parameters remain
free.

## Decision

The H1-blind projected-H0 variant is rejected as a production policy.

Production retains:

- H1-generated:
  truth-started H1 + morphology H0 + H1-projected H0;
- H0-generated:
  truth-started H0 + nested H1 from the settled H0.

No further H1-blind development is required before the CHE pilot.
