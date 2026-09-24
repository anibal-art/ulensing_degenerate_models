# LRT validation and event-characterization suite

This suite is analysis-only. It never simulates events and never runs an optimizer.
It reads the compact production Parquets already exported from CHE.

## Inputs already available

- `data/h0_lrt_results_20260921.parquet`
- `data/h1_lrt_results_20260920.parquet`

## Shared module

`lrt_common.py` centralizes paths, alpha levels, empirical thresholds, empirical
p-values, confidence intervals and robust column resolution.

## LRT validation tests

Run these in numerical order from `validation_tests/`:

0. `00_input_audit.py`: integrity and nesting-violation counts.
1. `01_h0_survival_vs_chi2.py`: empirical H0 survival versus chi-square(df=2).
2. `02_bootstrap_thresholds.py`: Monte Carlo uncertainty of calibrated critical values.
3. `03_crossvalidated_fpr.py`: out-of-sample false-positive validation.
4. `04_conditional_h0_stability.py`: FPR stability versus sampling/peak coverage.
5. `05_negative_lrt_diagnostics.py`: characterize Delta chi2 < 0 numerical failures.
6. `06_h0_false_positive_tail.py`: characterize actual H0 false positives.
7. `07_roc_curve.py`: complete H0/H1 ROC and low-FPR region.
8. `08_lrt_vs_wald_piE.py`: compare LRT with local covariance/Wald pi_E significance.
9. `09_classify_h1_empirical.py`: attach empirical p-values and calibrated classes to H1.

`run_all.sh` executes tests 00--09.

The primary alpha is centralized in `lrt_common.py` as `PRIMARY_ALPHA = 0.001`.
Changing it there changes the primary event classification consistently.

## Physical event characterization

The compact production export intentionally omitted generating truth. Before
physical characterization, create one joined table from the original generating
catalog:

```bash
python characterization/10_prepare_truth_join.py --catalog /path/to/original_H1_catalog.parquet
```

The join uses `catalog_row`. If the original raw catalog does not have an
explicit `catalog_row`, its row position is used, matching the production meaning
of `catalog_row`. The script records that choice and the resolved column mapping
in a metadata JSON and aborts if any H1 row cannot be matched.

Then run:

11. `11_detection_efficiency_1d.py`: efficiency versus truth and sampling variables.
12. `12_detection_efficiency_maps.py`: 2D efficiency maps in physical parameter space.
13. `13_parallax_recovery.py`: true versus fitted parallax-vector recovery.
14. `14_covariance_coverage.py`: 2D pi_E covariance coverage by LRT class.

or:

```bash
bash characterization/run_all_after_truth_join.sh
```

## Outputs

Tables are written below:

- `analysis/lrt/results/lrt_validation/`
- `analysis/lrt/results/characterization/`

Figures are written below:

- `analysis/lrt/figures/lrt_validation/`
- `analysis/lrt/figures/characterization/`

Production inputs are never modified.
