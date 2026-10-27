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

## Chunk artifact archival validation

Before full production, per-event output storage was changed to a chunk-level archive strategy.

Motivation:
- The 50-event integration produced 732 individual files.
- Linear extrapolation to 966000 physical events would imply of order 14 million files.
- Disk volume itself was acceptable, but filesystem metadata pressure from millions of files was not.
- Scientific inspection requires retaining simulated light curves, truth parameters, final H0/H1 fits, and optimizer diagnostics.

Policy adopted:
- run_summary.parquet and run_summary.csv remain directly accessible.
- laggards.parquet and laggards.csv remain directly accessible.
- frozen configuration and event-task tables remain directly accessible.
- fits/, models/, results/, and per-event logs are collected into one uncompressed TAR archive per chunk.
- the archive is verified with tar before source deletion.
- a SHA256 checksum and JSON manifest are written.
- original per-event trees are deleted only after successful archive verification.
- the DONE marker is written only after runner completion and archive verification.

Smoke test job 103112:
- 10 physical events / 20 logical H0-H1 datasets.
- 16 rejected_detectability and 4 ok.
- job completed successfully with exit code 0.
- archive size: 4730880 bytes.
- final chunk directory contained 14 files.
- archive SHA256 verification passed.

Scientific equivalence test:
- status, simulation_seed, noise_seed, and detectability_pass were identical to the previous integration run.
- 81 scientific numerical summary columns were compared.
- no numerical differences were found.

Recovery test:
- event 6 was recovered from the archive.
- H0 and H1 HDF5 files retained time, flux, magnitude, uncertainty, photometric masks, and model light curves.
- truth, fit_rr, multi_fit, final H0/H1 fit files, multistart diagnostics, and per-event logs were retained.
