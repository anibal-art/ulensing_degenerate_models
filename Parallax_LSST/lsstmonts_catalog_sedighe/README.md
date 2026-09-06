# LSSTMONTS runner: assume catalog alpha is xi

This version makes the provisional assumption explicit:

\[
\alpha_{\rm catalog} \equiv \xi.
\]

Here \(\xi\) is the angle of the relative source-lens trajectory used in
Equations 5-6 of Sajadian & Sahu (2023). It is not treated as the geometrical
\(\alpha\) from Appendix Equation A3.

The local tangent-plane convention is:

\[
\mathbf n_1 = +l,\qquad \mathbf n_2 = +b,
\]

so the catalog parallax amplitude is decomposed as

\[
\pi_{E,n_1}=\pi_E\cos\xi,
\qquad
\pi_{E,n_2}=\pi_E\sin\xi.
\]

The runner then rotates that vector into the ICRS North/East components
required by pyLIMA. The same \(\xi\) is passed to the simulated trajectory as
`theta_rad` and `traj_angle`.

The catalog column remains named `alpha` for input compatibility. Output files
record:

- `alpha_catalog`
- `alpha_interpretation = xi`
- `xi_catalog`
- `xi_rad`
- `xi_deg`
- `piEN`
- `piEE`

## Preparation test

```bash
python run_lsstmonts_catalog_sedighe_alpha_as_xi.py \
    --config lsstmonts_catalog_sedighe_alpha_as_xi.yaml \
    --prepare-only
```

The preparation summary should contain:

```text
Catalog angle column:   alpha
Angle interpretation:   alpha == xi
Xi tangent-plane basis: galactic_n1n2
```

## Run the 25-event test

```bash
python run_lsstmonts_catalog_sedighe_alpha_as_xi.py \
    --config lsstmonts_catalog_sedighe_alpha_as_xi.yaml
```

The existing zero-blending-factor handling is retained: filters with
`f_s == 0` are omitted and at least three positive catalog filters are
required.

<!-- BEGIN HIDDEN_PARALLAX_LRT_STATUS -->

# Hidden-parallax LRT pipeline

## Scientific objective

The original scientific objective has not changed. Physical systems are taken from the Sedighe/LSSTMONTS catalogue and the primary physical population is simulated as FSPL microlensing with parallax.

The question is whether Rubin observations can distinguish that signal from an otherwise equivalent FSPL model without parallax.

The nested hypotheses are:

- `H0`: FSPL with `piEN = piEE = 0`
- `H1`: FSPL with parallax

The likelihood-ratio statistic is

`T = Delta chi2 = chi2(H0) - chi2(H1)`.

The auxiliary H0 truth population used in Monte Carlo calibration does not replace the original FSPL+parallax population. It is used only to determine the false-positive distribution of the LRT.

## Original workflow versus current workflow

Originally:

```text
Sedighe catalogue event
    -> simulate FSPL + parallax
    -> fit FSPL models to Rubin data
```

Currently:

```text
Sedighe catalogue event
    -> construct Rubin cadence
    -> noise-independent pre-fit detectability
       -> FAIL: save diagnostics and stop
       -> PASS: simulate noisy realization
    -> fit H0 and H1
    -> T = chi2(H0) - chi2(H1)
    -> empirical H0 calibration / H1 power
```

The physical model of interest is therefore unchanged. The additions are the detectability gate, explicit nested LRT, bounded flux profiling, robust optimization, and empirical calibration.

## Pre-fit detectability

The catalogue-level selection does not guarantee that the exact Rubin cadence assigned by the runner samples the microlensing event. A noise-independent detectability gate is therefore applied before the expensive fits.

The reference used for selection is FSPL without parallax, so selection does not depend on the parallax signal being tested.

Current requirements are:

- at least 10 usable Rubin observations;
- at least 3 Rubin bands;
- at least 5 observations within `|t-t0| <= tE`;
- at least one point before and after `t0` inside that interval;
- at least 6 expected points at 3 sigma relative to a constant model;
- Asimov `Delta chi2_const / Nobs >= 2`.

A 5000-event pilot gave:

```text
audited events      = 5000
detectable events   = 1508
not detectable      = 3492
detectable fraction = 0.3016
```

The catalogue row used in the earliest single-event noise experiments was later found to have no Rubin points inside `t0 +/- tE`. Its old empirical LRT threshold is therefore not used as the final calibration.

## Bounded flux profiling

For fixed physical microlensing parameters, source and blend fluxes are linear nuisance parameters. The current implementation profiles them at every physical trial point while preserving the original allowed flux bounds.

The bounded linear problem is solved with `scipy.optimize.lsq_linear` using BVLS.

For H1, the nonlinear optimizer therefore works only in:

```text
t0, u0, tE, rho, piEN, piEE
```

The runtime patch is enabled with:

```bash
export HIDDEN_PARALLAX_BOUNDED_PROFILE=1
```

Fixed-physical-parameter validation showed agreement with the original bounded likelihood at approximately `1e-3` in chi-square or better.

## H1 optimization

The current production candidate is the FAST optimizer:

```text
bounded flux profiling
    + embedded H0 start
    + 3x3 piE multistart
    + adaptive TRF polish
```

H0 is fitted with TRF. For H1, the best H0 physical solution is embedded with `piEN = piEE = 0`, a deterministic 3x3 grid in `(piEN, piEE)` supplies additional starts, and the best local solution is polished with tighter TRF tolerances.

## Differential Evolution validation

Differential Evolution was used as an expensive global-search reference, not as the default production optimizer.

The benchmark used:

```text
strategy        = rand1bin
population_size = 10
max_iteration   = 1500
tol             = 0
atol            = 1e-4
seeds           = 20260903, 20260904
```

Each DE result was subsequently polished with TRF.

For 75 matched FAST versus DE+TRF datasets:

```text
median(delta_H1) = 2.65e-10
q90              = 2.27e-4
q95              = 7.27e-3
q99              = 2.91e-1
max              = 9.32e-1

N(delta_H1 > 0.01) = 4
N(delta_H1 > 0.1)  = 1
N(delta_H1 > 1)    = 0
N(delta_H1 > 5)    = 0
```

where `delta_H1 = chi2_H1_FAST - chi2_H1_DE+TRF`.

FAST tasks required about 43 seconds to 2.5 minutes per physical event in this benchmark. Completed GLOBAL DE+TRF tasks ranged from about 9 to 53 minutes, with many around 25-35 minutes. DE for every event is therefore not computationally justified.

The refined H1 likelihood is always:

`chi2_H1_best = min(chi2_H1_FAST, chi2_H1_DE_seed1, chi2_H1_DE_seed2)`.

A worse DE result never replaces FAST.

## Current LRT strategy

The current adopted workflow is:

1. construct the Rubin event;
2. evaluate pre-fit detectability;
3. stop before fitting if the event is not detectable;
4. fit H0 with bounded-profile TRF;
5. fit H1 with FAST;
6. compute `T = chi2_H0 - chi2_H1`;
7. determine the empirical threshold `T_alpha` from H0 noise realizations;
8. refine only near-threshold cases with DE+TRF when necessary.

A provisional conservative refinement region is `|T_FAST - T_alpha| < 2`. This is deliberately wider than the largest FAST-to-DE correction observed so far (`0.932`) and must be reassessed after the full H0 calibration.

## Configuration files

The runner is controlled by JSON configuration files.

Frozen reference configurations from the optimizer benchmark are stored in:

```text
configs/validation/lrt_benchmark/FAST.json
configs/validation/lrt_benchmark/GLOBAL.json
```

Historical development configurations are stored in:

```text
configs/validation/lrt_history/
```

For a new experiment, copy the FAST reference rather than editing the frozen file:

```bash
cp configs/validation/lrt_benchmark/FAST.json configs/my_experiment.json
```

Typical setup:

```bash
conda activate pyLIMA_test

export PROJECT_DIR="/export/storage3/rubin/microlensing/romanrubin/ulensing_degenerate_models/Parallax_LSST/lsstmonts_catalog_sedighe"
export CFG_PATH="$PROJECT_DIR/configs/my_experiment.json"
export HIDDEN_PARALLAX_BOUNDED_PROFILE=1

bash "$PROJECT_DIR/production/submit_lsstmonts_production_array.sh"
```

Important configuration groups:

- `simulation`: event generation, surveys and photometric filtering.
- `noise_realizations`: Monte Carlo realization count, seeds, paired noise and H0/H1 truth cases.
- `selection.prefit_detectability`: noise-independent Rubin detectability gate.
- `fit.fits`: H0/H1 model definitions, bounds and optimizer options.
- `fit.h1_multistart`: embedded-H0 initialization, piE multistart and adaptive polish.
- `fit.h1_multistart.diagnostic_global_de`: expensive DE validation/fallback.

For detectability:

- `audit_only = true`: compute detectability diagnostics only.
- `audit_only = false`: detectable events continue to H0/H1 fitting.

The H1 parameter domain must include `piEN = piEE = 0` so that H0 is nested inside H1.

DE diagnostics are stored separately under:

```text
_H1_global_DE/
    seed_<seed>/
        de_raw.npy
        trf_polish/
```

They are not automatically written into the ordinary H1 LRT columns.

## LRT progress as of 2026-09-06

Completed:

- nested H0/H1 formulation;
- deterministic and paired noise realizations;
- H1 bounds including the H0 point;
- embedded-H0 initialization;
- bounded flux profiling;
- deterministic piE multistart;
- adaptive TRF polishing;
- strict DE diagnostic search;
- DE -> TRF optimizer benchmark;
- noise-independent pre-fit detectability;
- 5000-event detectability pilot;
- representative FAST-versus-DE validation.

Still required before quoting the final LRT threshold:

1. freeze the FAST production configuration;
2. generate a sufficiently large H0 Monte Carlo sample over detectable events;
3. determine empirical `T_alpha` for the chosen false-positive rate;
4. refine realizations close to `T_alpha` with DE+TRF where necessary;
5. recompute the final threshold;
6. apply the calibrated threshold to H1 simulations and measure power.

The scientific question remains the original one: can a Rubin light curve generated by an FSPL+parallax event be distinguished from the corresponding FSPL model without parallax?

<!-- END HIDDEN_PARALLAX_LRT_STATUS -->
