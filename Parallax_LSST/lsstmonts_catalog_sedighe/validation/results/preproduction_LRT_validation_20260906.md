# Final pre-production LRT validation

Date: 2026-09-06/07

## Integration test

- 50 physical catalogue events.
- 1 paired noise realization per truth case.
- 100 logical H0/H1 datasets.
- 18/50 physical events passed pre-fit detectability.
- 32/50 were rejected before fitting.
- 36 logical datasets completed the FAST fits.
- H0/H1 noise-seed pairing: 50/50.
- H0/H1 detectability-status pairing: 50/50.
- Negative LRT statistics: 0.

Preliminary H0 LRT distribution (18 realizations):
- median T = 2.163316
- q95 T = 7.685363
- max T = 8.131761

These 18 H0 realizations are a validation sample only and are NOT used to define the final T_alpha.

Preliminary H1 distribution:
- median T = 153.443427
- min T = 2.338039
- max T = 712448.121722

Timing for successful logical datasets:
- median total wall time = 91.85 s
- q95 total wall time = 153.75 s
- maximum total wall time = 248.03 s

The slowest cases were not dominated by the final H0/H1 TRF optimizer. Most of the excess wall time occurred in the broader additional-fit workflow.

## Timeout decision

A post-detectability timeout of 600 s per logical dataset is adopted for the first full FAST production pass.

Rationale:
- 600 s is about 3.9 times the observed q95.
- 600 s is about 2.4 times the observed maximum in the 50-event integration test.
- the timeout therefore targets pathological stragglers rather than normal FAST fits.
- the timeout starts only after pre-fit detectability passes.
- timeout events are marked fit_timeout / needs_refit and are never treated as scientific non-detections.
- timed-out realizations are written to laggards.parquet and must be refitted before final scientific conclusions.

## Optimizer strategy

FAST remains the default production optimizer.
Differential Evolution is not run for every event because the detectable-event benchmark showed only small FAST-to-DE corrections while DE was roughly an order of magnitude to tens of times slower.
DE+TRF will be reserved for realizations close enough to the empirically calibrated T_alpha that an optimizer correction could change classification.

## Statistical strategy

The full production run first builds the empirical H0 distribution over detectable events.
T_alpha will be calibrated from that population for the selected false-positive rate.
The same threshold will then be applied to H1 simulations to estimate statistical power.
