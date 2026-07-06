# Results

This folder is reserved for generated outputs from CARSBench experiments.

Generated results may include benchmark tables, QC summaries, validation plots, domain-comparison figures, and small example outputs.

Large generated datasets are not committed to the repository.

---

## Recommended result types

| Output type             | Example                                     |
| ----------------------- | ------------------------------------------- |
| QC summaries            | `qc/general_qc_spectrum.csv`                |
| Domain comparison plots | mean spectra, PCA plots, parameter boxplots |
| Validation figures      | example spectra, Raman targets, NRB checks  |
| Benchmark tables        | RMSE, MAE, spectral angle across domains    |
| Seed summaries          | mean ± standard deviation across seeds      |

---

## Suggested folder structure

```text id="itmyw4"
results/
  README.md

  benchmark/
    baseline_results.csv
    cross_domain_results.csv

  figures/
    example_spectrum.png
    domain_mean_spectra.png
    pca_domains.png
    parameter_boxplots.png

  seed_summaries/
    seed_42_results.csv
    seed_123_results.csv
    seed_777_results.csv
    summary_mean_std.csv
```

---

## What should be committed

Recommended to commit:

* small benchmark summary CSV files
* small QC summary CSV files
* small example figures used in the README or documentation
* scripts used to generate results
* documentation explaining how results were produced

Recommended not to commit:

* full generated datasets
* large `.npz` batch files
* large model checkpoints
* temporary experiment outputs
* duplicated generated files

---

## Reproducibility

When reporting results, include:

* Git commit hash
* dataset seed or list of seeds
* domains used for training
* domains used for testing
* samples per domain
* benchmark metric definitions
* generation command
* evaluation command

Example:

```text id="t4ukak"
Dataset version: carsbench_v1
Seeds: 42, 123, 777
Train domains: A_typical, B_high_res, G_biochemical_source
Test domains: C_low_res_noisy, D_calibration_shift, E_window_shift, F_nrb_family_shift, H_biochemical_target
Metrics: RMSE, MAE, spectral angle
```

---

## Notes

This folder is mainly for lightweight, human-readable results.

Full generated datasets should usually be stored outside the Git repository, for example on external storage, cloud storage, or an experiment-tracking system.
