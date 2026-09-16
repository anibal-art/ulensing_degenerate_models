# Final-policy production profiling (short representative benchmark)

## CORRECTED (2026-09-15): the earlier "ready within a week" claim was fitting-only

An earlier version of this document reported ~8082 events/hour and
concluded the 1M-event run comfortably fits in a week. **That number
measured fitting time only, on already-materialized H5 light curves. It
is not the end-to-end production cost and the "ready" conclusion drawn
from it is retracted below.**

## Fitting-only benchmark (kept for reference, NOT the production number)

Serial baseline (n=25, H1-generated profiling sample, 1 core), final
policy v2 (includes both adopted safeguards):

| | wall (s) | CPU (s) |
|---|---|---|
| mean | 0.450 | 0.366 |
| median | 0.376 | -- |
| p90 | 0.692 | -- |
| max | 0.879 | -- |

Average TRF/event = 2.28 (continuation fired 3/25=12%, rescue 4/25=16%).
Fitting-only throughput: `3600/0.450 ≈ 8000 events/hour` on 1 core. This
number is real and useful for capacity planning of the FITTING STAGE
ALONE, but is not, by itself, the answer to "does the 1M run fit in a
week" -- that depends on whether materialization is part of the
pipeline (see below).

## End-to-end benchmark: catalog row -> simulation -> detectability gate -> H5 -> fit -> output

Production (`run_lsstmonts_catalog_hidden_parallax.py`'s own `main()`,
and this session's `standalone_materialize.py`, which reuses the same
`sim_event`/detectability-gate machinery) simulates and gates EVERY
catalog row it processes, not just the ones that end up fitted -- H5s do
**not** already exist for the 1M Sedighe population; they are produced by
the same run. Measured on 72 real candidate draws
(`results/profiling_sample_candidate_log.csv`, deterministic seed 424242):

| | value |
|---|---|
| mean wall / candidate (simulate + detectability gate, any outcome) | **6.60s** |
| mean wall / candidate that passes (materialize + save H5) | 7.23s |
| mean wall / candidate that fails detectability | 6.26s |
| detectability pass rate | **34.72%** |
| mean fit wall / passing event (final policy v2) | 0.450s |

**Materialization dominates total cost by ~14.7x over fitting.** The
fitting-only throughput number is therefore not representative of the
true per-event production cost whenever simulation is part of the
pipeline.

### Two possible readings of "1M Sedighe population" -- both computed, ambiguity flagged

**(A) 1M raw catalog draws are processed** (most rejected by the
detectability gate, ~34.7% pass and get fitted -- the population-synthesis
reading, most consistent with how this project's driver treats "catalog
rows" throughout):

```
total = 1e6 * 6.60s (materialize/gate, all) + 0.3472e6 * 0.450s (fit, passers)
      = 6,600,000s + 156,240s = 6,756,240s = 1876.7 h = 78.2 days (1 core)
```

**(B) 1M DETECTABLE (fitted) events are required** (need
`1e6 / 0.3472 ≈ 2,880,184` raw draws to get there):

```
total = 2,880,184 * 6.60s + 1e6 * 0.450s = 19,009,214s + 450,000s
      = 19,459,214s = 5405.3 h = 225.2 days (1 core)
```

**Which reading applies is not decided here** -- it changes the answer
by ~2.9x and must be clarified before any final go/no-go on timeline.

### Parallel scaling, measured (not just extrapolated)

- **Fitting stage**, 4 workers (`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`,
  no oversubscription): 25 events in 5.87s vs 11.32s serial -- **1.93x
  speedup (~48% efficiency)**.
- **Materialization stage**, 2 workers, 10 fresh candidates each (20
  total, real live run, some contention from a concurrent unrelated
  background job so mildly conservative): 82.4s wall vs a 132.0s serial-
  equivalent estimate (20 x 6.60s) -- **1.60x speedup (~80% efficiency)**.
  Per-worker cost rose to 7.1-7.9s/candidate (vs the clean 6.60s serial
  baseline), consistent with real resource contention, not a pipeline
  defect.

### Throughput verdict at measured efficiencies, 16 cores

Using the (A) reading (raw-draw population) and a deliberately
conservative blended efficiency (48%, the WORSE of the two measured
efficiencies, applied to both stages):

```
effective_cores ~= 16 * 0.48 ~= 7.7
(A) wall days ~= 78.2 / 7.7 ~= 10.2 days   -- EXCEEDS the 1-week target
(B) wall days ~= 225.2 / 7.7 ~= 29.2 days  -- far exceeds
```

Using the materialization stage's own (better) measured efficiency
(80%, `effective_cores ~= 12.8`):

```
(A) wall days ~= 78.2 / 12.8 ~= 6.1 days   -- borderline, roughly at target
(B) wall days ~= 225.2 / 12.8 ~= 17.6 days -- still exceeds
```

**Verdict: the 1M-event run does NOT clearly fit in one week once
materialization is counted, under either reading, at any efficiency
actually measured in this session.** Reading (A) is close to the
boundary depending on which measured efficiency is used; reading (B) is
not close under any of them. This directly overturns the earlier
"ready" conclusion, which was based on fitting-only throughput.

## Memory / I/O

Materialization writes one H5 file per detected event -- no
shared-file/HDF5-locking risk observed at 2-4 concurrent workers, one
file per event by construction (`Event_{global_i}.h5`, unique per catalog
row). The fitting stage itself performs no I/O beyond reading these
already-written files. No swapping observed at 2-4 concurrent workers on
this 46GB-RAM machine; a full 16-worker memory/I/O check was not run in
this session (time-boxed).

## What is NOT resolved here

- A dedicated 8/16-worker sweep for either stage (only 2- and 4-worker
  spot checks were run, time-boxed).
- Clarifying which of readings (A)/(B) is the actual production target.
- Whether materialization parallel efficiency holds up at higher worker
  counts, or degrades further (e.g. from `rubin_sim`/OpSim database read
  contention) -- only a 2-worker check was run.

Recommendation: before committing to a 1M-event launch date, run a
proper 8/16-worker end-to-end (materialize+fit) throughput measurement on
a larger sample (several hundred events), and get an explicit answer on
reading (A) vs (B).
