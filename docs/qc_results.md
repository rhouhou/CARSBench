# QC results and interpretation

CARSBench includes quality-control scripts for checking whether generated spectra and benchmark domains behave as expected.

The QC outputs are intended as sanity checks before using generated datasets for Raman-retrieval or machine-learning experiments.

---

## QC output location

QC summaries are stored in the `qc/` folder.

Typical files include:

| File                             | Purpose                                             |
| -------------------------------- | --------------------------------------------------- |
| `general_qc_spectrum.csv`        | QC summary for final simulated CARS/BCARS spectra   |
| `general_qc_clean_intensity.csv` | QC summary for clean forward-model intensity        |
| `general_qc_raman_target.csv`    | QC summary for Raman-equivalent targets             |
| `specific_domain_qc.csv`         | Domain-specific checks for expected domain behavior |

---

## General QC

General QC checks whether generated arrays are numerically valid and statistically reasonable.

Typical checks include:

* number of samples per domain
* signal minimum and maximum
* signal mean and standard deviation
* NaN values
* infinite values
* domain-level differences in signal behavior

These checks help confirm that generated spectra are usable before running benchmark experiments.

---

## Domain-specific QC

Domain-specific QC checks whether each benchmark domain shows the expected type of shift.

| Domain                 | Expected QC behavior                                       |
| ---------------------- | ---------------------------------------------------------- |
| `A_typical`            | Serves as the reference domain                             |
| `B_high_res`           | Should show sharper spectral features than typical spectra |
| `C_low_res_noisy`      | Should show stronger degradation or noise                  |
| `D_calibration_shift`  | Should show spectral-axis shift or warp                    |
| `E_window_shift`       | Should show different spectral-window coverage             |
| `F_nrb_family_shift`   | Should show altered non-resonant-background behavior       |
| `G_biochemical_source` | Should reflect source biochemical composition              |
| `H_biochemical_target` | Should reflect shifted target biochemical composition      |

---

## How to run QC

Run general QC on the final simulated spectra:

```bash
python scripts/08_general_domain_qc.py \
  --data-root data/carsbench_v1 \
  --output-csv qc/general_qc_spectrum.csv \
  --value-key spectrum
```

Run general QC on clean intensity:

```bash
python scripts/08_general_domain_qc.py \
  --data-root data/carsbench_v1 \
  --output-csv qc/general_qc_clean_intensity.csv \
  --value-key clean_intensity
```

Run general QC on Raman targets:

```bash
python scripts/08_general_domain_qc.py \
  --data-root data/carsbench_v1 \
  --output-csv qc/general_qc_raman_target.csv \
  --value-key raman_target
```

Run domain-specific QC:

```bash
python scripts/09_specific_domain_qc.py \
  --data-root data/carsbench_v1 \
  --output-csv qc/specific_domain_qc.csv
```

---

## How to interpret QC results

QC results should be used to answer basic questions before benchmarking:

1. Did all domains generate successfully?
2. Are there any NaN or infinite values?
3. Are signal ranges reasonable?
4. Do domain shifts appear in the expected direction?
5. Are Raman targets available and numerically valid?
6. Are any domains behaving unexpectedly?

QC files are not final scientific results. They are checks that help decide whether the generated dataset is suitable for downstream analysis.

---

## Recommended workflow

A typical validation workflow is:

```text
Generate pilot dataset
Run smoke test
Run general QC
Run domain-specific QC
Inspect example spectra
Inspect domain comparison plots
Generate full benchmark dataset
Run QC again
Run baseline benchmark
```

This reduces the risk of using a faulty generated dataset in ML experiments.

---

## Notes

QC summaries are lightweight and may be committed to the repository.

Full generated datasets should usually not be committed because they can become large.
