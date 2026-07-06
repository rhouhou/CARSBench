# Quality-control outputs

This folder contains quality-control summaries for generated CARSBench spectra.

The purpose of these files is to check whether the simulated benchmark domains behave as expected before they are used for machine-learning experiments.

## Files

| File                             | Description                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| `general_qc_spectrum.csv`        | Domain-level QC summary for the final simulated CARS/BCARS spectra                   |
| `general_qc_clean_intensity.csv` | QC summary for the clean forward-model intensity before selected measurement effects |
| `general_qc_raman_target.csv`    | QC summary for the Raman-equivalent target signal                                    |
| `specific_domain_qc.csv`         | Domain-specific checks for expected shifts across benchmark domains                  |

## What the QC checks are used for

The QC workflow is intended to verify that:

* spectra are generated without numerical failures
* each domain has the expected number of samples
* signal ranges are reasonable
* no domain contains unexpected NaN or infinite values
* domain-specific shifts are visible in the generated metadata and spectra
* the Raman-equivalent targets are usable for retrieval benchmarks

## Expected domain behavior

| Domain                 | Expected behavior                                                  |
| ---------------------- | ------------------------------------------------------------------ |
| `A_typical`            | Reference domain with typical BCARS/CARS acquisition conditions    |
| `B_high_res`           | Sharper spectral features due to higher spectral resolution        |
| `C_low_res_noisy`      | More degraded spectra due to lower resolution and stronger noise   |
| `D_calibration_shift`  | Spectral-axis shift and/or warp compared with the reference domain |
| `E_window_shift`       | Different spectral-window coverage                                 |
| `F_nrb_family_shift`   | Different non-resonant-background shape family                     |
| `G_biochemical_source` | Source biochemical composition pattern                             |
| `H_biochemical_target` | Target biochemical composition pattern                             |

## Notes

These QC files are not final scientific results. They are sanity checks used to confirm that the simulated benchmark data are suitable for downstream visualization, retrieval, and domain-generalization experiments.
