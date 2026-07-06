# Reproducibility

CARSBench is designed to generate reproducible synthetic CARS/BCARS benchmark datasets.

Reproducibility is important because the benchmark is intended for comparing Raman-retrieval and machine-learning methods across controlled domain shifts.

---

## Recommended benchmark seeds

The recommended benchmark uses three independent seeds:

```text id="v0960t"
42
123
777
```

Using multiple seeds makes it possible to check whether model performance is stable across different random draws of spectra, biochemical mixtures, non-resonant backgrounds, noise, and measurement effects.

---

## Why multiple seeds are used

A single simulated dataset can give a misleading estimate of model performance.

Using multiple seeds helps measure:

* stability of benchmark results
* sensitivity to random spectral mixtures
* sensitivity to noise and background variation
* whether a model generalizes consistently across generated datasets

For reporting results, it is recommended to calculate the mean and standard deviation across seeds.

Example:

```text id="2kmmsh"
RMSE = 0.084 ± 0.006 across seeds 42, 123, and 777
```

---

## Recommended generation structure

Generated datasets should be organized by seed.

Example:

```text id="nd7d4o"
data/
  carsbench_v1/
    seed_42/
      A_typical/
      B_high_res/
      C_low_res_noisy/
      D_calibration_shift/
      E_window_shift/
      F_nrb_family_shift/
      G_biochemical_source/
      H_biochemical_target/

    seed_123/
      A_typical/
      B_high_res/
      C_low_res_noisy/
      D_calibration_shift/
      E_window_shift/
      F_nrb_family_shift/
      G_biochemical_source/
      H_biochemical_target/

    seed_777/
      A_typical/
      B_high_res/
      C_low_res_noisy/
      D_calibration_shift/
      E_window_shift/
      F_nrb_family_shift/
      G_biochemical_source/
      H_biochemical_target/
```

This structure makes it easier to compare the same domain across different seeds.

---

## Generating one seed

Example command for generating one full benchmark seed:

```bash id="23krni"
python scripts/01_generate_full_dataset.py \
  --output-root data/carsbench_v1/seed_42 \
  --samples-per-domain 5000 \
  --chunk-size 500 \
  --seed 42 \
  --include-latents
```

The same command can be repeated for the other seeds:

```bash id="g8vkeo"
python scripts/01_generate_full_dataset.py \
  --output-root data/carsbench_v1/seed_123 \
  --samples-per-domain 5000 \
  --chunk-size 500 \
  --seed 123 \
  --include-latents

python scripts/01_generate_full_dataset.py \
  --output-root data/carsbench_v1/seed_777 \
  --samples-per-domain 5000 \
  --chunk-size 500 \
  --seed 777 \
  --include-latents
```

---

## Generating all seeds

The repository also includes:

```text id="ea9zfy"
scripts/01_generate_all_seeds.py
```

This script can be used to run dataset generation for the recommended seeds.

Before running it, check the output path inside the script and make sure it points to the correct local or external storage location.

Then run:

```bash id="kbzney"
python scripts/01_generate_all_seeds.py
```

---

## Pilot datasets

Before generating the full benchmark, it is recommended to generate a smaller pilot dataset.

Example:

```bash id="fvy2y7"
python scripts/01_generate_full_dataset.py \
  --output-root data/carsbench_pilot/seed_42 \
  --samples-per-domain 500 \
  --chunk-size 250 \
  --seed 42 \
  --include-latents
```

Pilot datasets are useful for:

* checking that generation works
* creating quick visualizations
* running QC scripts
* debugging domain behavior
* avoiding large storage use during development

---

## Reproducible sample generation

For reproducible generation, always record:

| Item               | Example                  |
| ------------------ | ------------------------ |
| Code version       | Git commit hash          |
| Seed               | `42`                     |
| Domain             | `A_typical`              |
| Samples per domain | `5000`                   |
| Chunk size         | `500`                    |
| Latents saved      | `true`                   |
| Generation command | Full command used        |
| Date generated     | Optional but recommended |

The most reliable way to reproduce a dataset is to record both the seed and the exact Git commit used to generate it.

You can get the current Git commit with:

```bash id="5orqxv"
git rev-parse HEAD
```

---

## Metadata and manifest files

Each generated domain includes metadata files that support reproducibility.

Typical structure:

```text id="uv62o5"
A_typical/
  batches/
    batch_000.npz
    batch_001.npz
    ...
  metadata/
    metadata.jsonl
  manifest.json
```

The `manifest.json` file summarizes the generation settings.

The `metadata.jsonl` file stores per-sample metadata, such as sampled parameters, domain information, and simulation details.

Together, these files make it possible to inspect how each dataset was generated.

---

## Recommended reporting format

When reporting benchmark results, use seed-averaged results.

Example table:

| Method     | Train domains | Test domain       | Seed 42 | Seed 123 | Seed 777 |    Mean ± std |
| ---------- | ------------- | ----------------- | ------: | -------: | -------: | ------------: |
| Baseline A | `A_typical`   | `C_low_res_noisy` |   0.091 |    0.087 |    0.094 | 0.091 ± 0.004 |
| Baseline B | `A_typical`   | `C_low_res_noisy` |   0.078 |    0.081 |    0.076 | 0.078 ± 0.003 |

This makes the benchmark more reliable than reporting a single random seed.

---

## Recommended train/test setup

A common cross-domain benchmark setup is:

| Split      | Domains                                                                               |
| ---------- | ------------------------------------------------------------------------------------- |
| Train      | `A_typical`, `B_high_res`, `G_biochemical_source`                                     |
| Validation | `C_low_res_noisy`                                                                     |
| Test       | `D_calibration_shift`, `E_window_shift`, `F_nrb_family_shift`, `H_biochemical_target` |

A stricter biochemical transfer setup is:

| Split | Domains                |
| ----- | ---------------------- |
| Train | `G_biochemical_source` |
| Test  | `H_biochemical_target` |

The exact split should always be reported clearly.

---

## Reproducibility checklist

Before publishing results, check that you have recorded:

* dataset version
* Git commit hash
* seed or list of seeds
* domains used for training
* domains used for validation
* domains used for testing
* samples per domain
* chunk size
* whether latent arrays were saved
* metric definitions
* preprocessing steps
* model or baseline configuration

---

## Notes

Generated datasets are usually not committed to Git because they can be large.

Recommended practice:

* commit the code
* commit the documentation
* commit small QC summaries
* commit small example figures
* do not commit full generated datasets
* document exactly how the full datasets were generated

This makes the repository easier to clone while still keeping the benchmark reproducible.
