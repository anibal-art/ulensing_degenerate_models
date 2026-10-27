# Hidden-parallax fit-domain / bounds audit

## Purpose

This validation tests whether the truth-dependent physical bounds used in the original hidden-parallax production artificially increase the LRT.

The validation sample contains 17 covariance-tail events and 17 matched controls.

Sample definition:
- data/refit_manifest.csv
- data/tail_control_combined.csv

## Production parity

The refits reuse the exact photometric realizations from production. No events are resimulated.

The refits use the same flux treatment as production:
- telescopes_fluxes_method = polyfit
- exact bounded telescope-flux profile
- fsource >= 0
- ftotal >= 0
- fsource and ftotal bounded above by max observed flux
- Numerical Jacobian

The bounded flux profile is enabled with HIDDEN_PARALLAX_BOUNDED_PROFILE=1.

## Truth-independent physical fit domain

t0: full time range of the fitted photometry
u0: [-5, 5]
tE: [0.1, 20000] days
rho: [1e-7, 10]

The same nuisance-parameter domain is used for H0 and H1.

## Definitions

I0 = old_h0_chi2 - new_h0_chi2
I1 = old_h1_chi2 - new_h1_chi2
old_lrt = old_h0_chi2 - old_h1_chi2
new_lrt = new_h0_chi2 - new_h1_chi2
delta_lrt = new_lrt - old_lrt
fractional_lrt_reduction = 1 - new_lrt / old_lrt

## Main scripts

run_bounds_audit_refit_core.py
Snapshot of the validated local refit runner.

run_bounds_audit_refit.py
Portable launcher for the audit runner.

run_refit_sample.sh
Runs all events in the 17+17 manifest.

collect_refit_results.py
Collects the per-event summary files into one table.

make_bounds_audit_figures.py
Regenerates the bounds-audit figures from the consolidated table.

## Existing validated results

The completed 34-event refits are stored at:
~/Downloads/hidden_parallax/hidden_parallax_refit_test

To collect the existing results:

python validation/bounds_audit/collect_refit_results.py --work-root "$HOME/Downloads/hidden_parallax/hidden_parallax_refit_test"

Then regenerate the figures with:

python validation/bounds_audit/make_bounds_audit_figures.py

## Scientific interpretation

The 17+17 matched audit shows that widening the physical fit domain can strongly improve H0 in both covariance tails and matched controls, while H1 is generally much more stable.

Therefore the effect is not specific to the covariance-tail selection.

The H5 photometric simulations remain valid. The population LRT should be recomputed using truth-independent physical bounds.

## Reproducing the completed local audit

The validated 34-event work directory on the local machine is:

`$HOME/Downloads/hidden_parallax/hidden_parallax_refit_test`

Set:

```bash
export HIDDEN_PARALLAX_BOUNDS_AUDIT_ROOT="$HOME/Downloads/hidden_parallax/hidden_parallax_refit_test"
export ROMAN_RUBIN_DIR="$HOME/microlensing/simulation_Rubin/roman_rubin"
```

Collect the 34 completed fits with:

```bash
python validation/bounds_audit/collect_refit_results.py --work-root "$HIDDEN_PARALLAX_BOUNDS_AUDIT_ROOT"
```

Regenerate all bounds-audit figures with:

```bash
python validation/bounds_audit/make_bounds_audit_figures.py
```
