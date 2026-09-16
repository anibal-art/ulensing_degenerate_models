# Production-candidate LRT fitting architecture

## Scientific objective

The purpose of this pipeline is to measure annual-parallax detectability
in simulated Rubin-only FSPL light curves using

    D = chi2_H0 - chi2_H1,

where:

- H0 is FSPL without annual parallax;
- H1 is FSPL with annual parallax;
- H0 is nested in H1 at piEN = piEE = 0.

This is a simulation-based likelihood-ratio study, not a blind real-data
search pipeline. Truth may therefore be used as an initialization aid
for the model that generated the simulated event. All nonlinear
parameters remain free during optimization.

Flux parameters are profiled through bounded weighted least squares.

## Frozen numerical domain

Production candidate:

- coordinates: physical;
- TRF x_scale: original pyLIMA scaling;
- t0 bounds: data-driven photometry domain;
- u0: [-10, 10];
- tE: [0.1, 500000];
- rho: [1e-7, 10];
- piEN: [-40, 40];
- piEE: [-40, 40].

The RuntimeWarning produced by pyLIMA when a nested H1 start contains
piEN = piEE = 0 is understood: the original scaling evaluates log10(0),
but the corresponding scale evaluates to 1 after the final +1.
The validated production-profiling calculations used this same pyLIMA
scaling, so it is retained for consistency.

## Nominal fitting matrix

### H1-generated event

Three nominal TRF calls:

1. H1 truth start:
   `(t0,u0,tE,rho,piEN,piEE)_true`.

2. H0 deterministic morphology start.

3. H0 projected start from the SETTLED H1 solution:
   `(t0,u0,tE,rho)_H1fit`.

The final H0 likelihood is

    chi2_H0 = min(
        chi2_H0_morphology,
        chi2_H0_H1projected
    ).

The final H1 likelihood is the settled H1 truth-started fit unless a
later nesting rescue improves it.

### H0-generated event

Two nominal TRF calls:

1. H0 truth start:
   `(t0,u0,tE,rho)_true`.

2. H1 exact nested start from the SETTLED H0 solution:
   `(t0,u0,tE,rho)_H0fit`,
   with `piEN = piEE = 0`.

All six H1 nonlinear parameters remain free.

## Conditional safeguards

Each nominal fit receives at most one same-point continuation when

    reduced chi2 > 50.

The continuation uses the settled point as its new initialization and
is kept only if chi2 does not worsen.

After H0 and H1 are settled, nesting is checked numerically. If

    chi2_H1 > chi2_H0 + 1e-6,

one H1 rescue is run from the winning H0 solution embedded at
`piEN = piEE = 0`.

Conditional continuations and the nesting rescue are not counted as
nominal starts.

## Expected nominal cost

| generated model | H0 starts | H1 starts | nominal TRF |
| --- | ---: | ---: | ---: |
| H1 | 2 | 1 | 3 |
| H0 | 1 | 1 | 2 |

## Validation basis

The original one-H0-start H1-generated policy produced basin-induced
inflation of the LRT statistic in the development sample.

A second H0 start projected from the settled H1 solution repaired
11 of 15 dangerous development cases against the frozen independent
reference.

A selective risk trigger did not provide sufficient recall, so the
second H0 start is unconditional for H1-generated events.

A blind H1 start from morphology plus zero parallax was tested as a
negative control. It repaired only 2 of the 15 dangerous cases and was
substantially less stable than the truth-started H1 fit. It is not part
of production.

Residual-geometry inspection did not establish a common additional H0
failure geometry from the two genuine H0-limited residual cases, so a
third deterministic H0 start is not justified by the current evidence.

Large legacy H0/H1 multistart panels and log-coordinate alternatives
remain validation/reference machinery and are not part of the scalable
production candidate.

## Local smoke validation

The final policy was smoke-tested on:

- H1-generated row 897838:
  3 nominal TRF, no continuation, no rescue,
  H0 morphology winner, final nesting satisfied.

- H0-generated row 813025:
  2 nominal TRF, no continuation, no rescue,
  H0 truth fit followed by nested H1 start,
  final nesting satisfied.

The H1-generated result reproduces the previously validated
H1-projected-H0 result. The H0-generated regression is exactly
reproducible under explicit `x_scale=pylima`.
