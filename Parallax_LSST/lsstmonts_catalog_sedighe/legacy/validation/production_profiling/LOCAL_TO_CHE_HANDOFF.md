# Local -> CHE handoff

## Frozen fitting policy

Implementation:

- `lrt_fit_policy.py`
- `run_lrt_policy_batch.py`

Nominal optimizer cost:

- H1-generated: 3 TRF/event;
- H0-generated: 2 TRF/event.

Conditional extra calls:

- one same-point continuation per catastrophically bad fit;
- one nested H1 rescue when final numerical nesting fails.

Numerical setup:

- bounds profile: `production_candidate`;
- coordinates: `physical`;
- x_scale: `jac`;
- bounded flux profiling enabled;
- EPSILON_NUMERIC = 1e-6.

The pending modifications in
`run_lsstmonts_catalog_hidden_parallax.py` belong to earlier
bounds/multistart development and are intentionally not part of this
LRT-policy freeze unless reviewed separately.

## CHE sequence

1. Pull the committed local branch.

2. Activate the same pyLIMA/Rubin environment.

3. Run a tiny H0/H1 smoke test on materialized H5 events.

4. Run a modest pilot before any large production.

   Measure separately:

   - raw catalog rows inspected;
   - events passing the pre-fit selection;
   - fitted events;
   - wall time/event;
   - CPU time/event;
   - memory;
   - I/O;
   - nominal TRF/event;
   - continuation frequency;
   - nested-rescue frequency;
   - failed-event frequency.

5. Validate especially the new H0-generated branch:

   H0 truth fit -> settled H0 -> nested H1(piE=0).

6. Choose the false-positive level alpha.

7. Choose the required H0 calibration sample size from the desired
   precision in the relevant upper tail. Do not assume 100k a priori.

8. Generate the H0 calibration distribution and determine the empirical
   LRT threshold.

9. Evaluate the frozen threshold on an independent H1-generated holdout.

10. Only after these checks launch the main H1-generated production.

Always distinguish:

    N_raw_rows

from

    N_passing_fitted_events.

The final alpha, H0 calibration size, empirical threshold and main
production target are not frozen yet.
