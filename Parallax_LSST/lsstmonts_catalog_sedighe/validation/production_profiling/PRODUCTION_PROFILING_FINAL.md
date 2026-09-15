# Final-policy production profiling (short representative benchmark)

Not a re-run of the earlier full P1/P2 sweep (that was suspended in
`FEASIBILITY_AUDIT.md` for the abandoned `old_final`-based architecture
and is moot now). This is a short benchmark of the FROZEN final policy
(`final_policy.py`) only.

## Serial baseline (n=25, H1-generated profiling sample, 1 core)

`results/final_policy_h1generated_25events.json`:

| | wall (s) | CPU (s) |
|---|---|---|
| mean | 0.445 | 0.352 |
| median | 0.387 | 0.299 |
| p90 | 0.682 | 0.569 |
| max | 0.773 | 0.691 |

- Average TRF/event (including rescues): **2.16**.
- Rescue fraction: **4/25 = 16.0%**.
- Failure rate: **0/25** (no exceptions; all 25 events produced a final
  H0 and H1 result -- the 3 chi2-sanity-flagged events, see
  `BASIN_GAP_VALIDATION.md`, still completed and returned a result, they
  are flagged for review, not treated as pipeline failures).
- Serial throughput: `3600/0.445 = 8082 events/hour` on 1 core.

## 4-worker parallel check

25 events split into 4 roughly-even chunks, run as 4 concurrent
subprocesses (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=
NUMEXPR_NUM_THREADS=1` to avoid BLAS oversubscription on this 16-core
machine), timed end-to-end (`time.time()` around the whole 4-process
batch, includes process/import startup):

- Wall time for all 25 events: **5.87s** (vs 11.32s serial) ->
  **~1.93x speedup at 4 workers** (~48% parallel efficiency).
- Per-chunk "pure fit" time (excluding process startup) was actually
  *higher* per event than the serial baseline (~0.5-0.6s/event vs
  0.445s/event) -- consistent with real CPU/memory-bandwidth contention
  between 4 concurrent single-threaded fits on shared cache, not with
  BLAS oversubscription (which was explicitly disabled). Not
  investigated further -- a short representative check, not a full P1/P2
  sweep.
- Naive throughput at 4 workers: `25/5.87 * 3600 ≈ 15332 events/hour`.

## Memory / I/O

No HDF5 writes in the final policy at all (it only reads
already-materialized H5 light curves; there is no per-fit output file
beyond the batch-level JSON written once at the end of each process) --
the HDF5-collision risk documented for the earlier architecture's
materialization step does not apply to the fitting stage itself. Peak
memory not formally profiled in this short benchmark; 4 concurrent
processes on a 46GB-RAM/16-core machine showed no swapping or slowdown
attributable to memory pressure.

## Scaling verdict

Even at the conservative, unoptimized 4-worker measurement (~48%
efficiency), 16-core naive extrapolation (`8082 * 16 * 0.48 ≈ 62000
events/hour`, deliberately pessimistic) is **>10x** the
`1e6/(7*24) ≈ 5952 events/hour` 1-week target for the 1M-event Sedighe
population -- and the 1-core number ALONE (8082/hour) already exceeds the
target by ~1.36x. No further parallel-efficiency tuning is needed to
clear the production throughput bar; a full P1/P2 sweep at 8/16 workers
was not run here because it is not needed to answer the readiness
question ("do not redesign automatically if profiling is already
compatible with the production target").
