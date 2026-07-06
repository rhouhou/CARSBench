# CARSBench dataset format

CARSBench writes generated spectra in a chunked dataset format.

This design makes it possible to generate large synthetic BCARS/CARS benchmark datasets without storing everything in memory at once.

---

## Folder structure

A generated dataset is usually organized by seed and domain.

Example:

```text
data/
  carsbench_v1/
    seed_42/
      A_typical/
        batches/
          batch_000.npz
          batch_001.npz
          ...
        metadata/
          metadata.jsonl
        manifest.json

      B_high_res/
        batches/
          batch_000.npz
          batch_001.npz
          ...
        metadata/
          metadata.jsonl
        manifest.json

      ...

    seed_123/
      ...

    seed_777/
      ...
```

The recommended benchmark seeds are:

```text
42, 123, 777
```

---

## Main files

Each generated domain contains three main types of files.

| File or folder            | Description                                  |
| ------------------------- | -------------------------------------------- |
| `batches/`                | Folder containing chunked `.npz` batch files |
| `metadata/metadata.jsonl` | Per-sample simulation metadata               |
| `manifest.json`           | Summary of the generated domain dataset      |

---

## Batch files

The `batches/` folder contains files such as:

```text
batch_000.npz
batch_001.npz
batch_002.npz
```

Each `.npz` file stores a chunk of simulated spectra.

The chunk size is controlled by the `--chunk-size` argument during dataset generation.

Example:

```bash
python scripts/01_generate_full_dataset.py \
  --output-root data/carsbench_v1/seed_42 \
  --samples-per-domain 5000 \
  --chunk-size 500 \
  --seed 42 \
  --include-latents
```

With `5000` samples per domain and a chunk size of `500`, each domain will contain approximately ten batch files.

---

## Common `.npz` keys

Each `.npz` batch may contain the following arrays.

| Key               | Description                                                       |
| ----------------- | ----------------------------------------------------------------- |
| `axis`            | Wavenumber axis in cm⁻¹                                           |
| `spectrum`        | Final simulated CARS/BCARS spectrum                               |
| `raman_target`    | Raman-equivalent target signal used for retrieval benchmarks      |
| `clean_intensity` | Clean forward-model intensity before selected measurement effects |
| `envelope`        | Instrument or spectral envelope contribution                      |
| `baseline`        | Baseline contribution                                             |
| `metadata_json`   | Per-sample metadata encoded as JSON strings                       |

When latent variables are saved with `--include-latents`, additional arrays may be included.

Possible latent arrays include:

| Key           | Description                                   |
| ------------- | --------------------------------------------- |
| `chi_r_real`  | Real part of resonant susceptibility          |
| `chi_r_imag`  | Imaginary part of resonant susceptibility     |
| `chi_nr_real` | Real part of non-resonant susceptibility      |
| `chi_nr_imag` | Imaginary part of non-resonant susceptibility |

The exact available keys may depend on the generation options and simulator configuration.

---

## Main signal definitions

### `axis`

The `axis` array contains the wavenumber values.

It is usually expressed in cm⁻¹.

Example shape:

```text
(num_points,)
```

or, depending on storage format:

```text
(num_samples, num_points)
```

---

### `spectrum`

The `spectrum` array contains the final simulated CARS/BCARS intensity.

This is the main input signal for retrieval or machine-learning models.

Example shape:

```text
(num_samples, num_points)
```

---

### `raman_target`

The `raman_target` array contains the Raman-equivalent target signal.

In CARSBench, this is typically based on the imaginary part of the resonant susceptibility.

This signal is used as the target for supervised Raman-retrieval experiments.

Example shape:

```text
(num_samples, num_points)
```

---

### `clean_intensity`

The `clean_intensity` array contains the clean simulated forward intensity before selected measurement effects.

This can be useful for debugging, visualization, and quality control.

---

### `envelope`

The `envelope` array stores the instrument or spectral envelope applied during simulation.

This is useful for checking whether domain-specific envelope effects are behaving as expected.

---

### `baseline`

The `baseline` array stores the baseline contribution used in the simulated spectrum.

This is useful for validating baseline variability and identifying unwanted simulation artifacts.

---

## Metadata

The `metadata/metadata.jsonl` file stores one JSON object per simulated sample.

Each line corresponds to one sample.

Example format:

```json
{"sample_id": 0, "domain_name": "A_typical", "seed": 42, "parameters": {...}}
{"sample_id": 1, "domain_name": "A_typical", "seed": 42, "parameters": {...}}
{"sample_id": 2, "domain_name": "A_typical", "seed": 42, "parameters": {...}}
```

The metadata may include information such as:

| Metadata field         | Description                                       |
| ---------------------- | ------------------------------------------------- |
| `sample_id`            | Sample index                                      |
| `domain_name`          | Domain used to generate the sample                |
| `seed` or `child_seed` | Random seed information                           |
| `resolved_config`      | Final sampled simulation configuration            |
| `resonant_info`        | Resonant peak and biochemical mixture information |
| `nrb_info`             | Non-resonant-background parameters                |
| `instrument_info`      | Instrument resolution and envelope information    |
| `noise_info`           | Noise and detector parameters                     |
| `calibration_info`     | Spectral shift and warp parameters                |

The exact metadata fields may vary depending on the simulator version.

---

## Manifest file

Each domain folder contains a `manifest.json` file.

The manifest summarizes the generated dataset.

It may include:

| Field                | Description                                 |
| -------------------- | ------------------------------------------- |
| `domain_name`        | Name of the generated domain                |
| `samples_per_domain` | Number of samples generated for the domain  |
| `chunk_size`         | Number of samples per `.npz` batch          |
| `include_latents`    | Whether latent simulation arrays were saved |
| `master_seed`        | Main seed used for generation               |

The manifest is useful for quickly checking how a dataset was generated without opening every batch file.

---

## Reading batch files manually

A batch file can be read with NumPy.

```python
import numpy as np

batch = np.load("data/carsbench_v1/seed_42/A_typical/batches/batch_000.npz")

print(batch.files)

axis = batch["axis"]
spectrum = batch["spectrum"]
raman_target = batch["raman_target"]
```

---

## Reading metadata manually

The metadata file can be read line by line.

```python
import json

metadata_path = "data/carsbench_v1/seed_42/A_typical/metadata/metadata.jsonl"

with open(metadata_path, "r") as f:
    first_sample_metadata = json.loads(next(f))

print(first_sample_metadata)
```

---

## Recommended use in machine-learning workflows

A typical supervised Raman-retrieval task uses:

| Role                    | Array                                      |
| ----------------------- | ------------------------------------------ |
| Model input             | `spectrum`                                 |
| Model target            | `raman_target`                             |
| Optional auxiliary data | `axis`, `metadata_json`, `clean_intensity` |

Example:

```python
x = batch["spectrum"]
y = batch["raman_target"]
```

A model can then be trained to predict the Raman-equivalent target from the simulated CARS/BCARS spectrum.

---

## Recommended benchmark splits

CARSBench is designed for cross-domain evaluation.

Example split:

| Split      | Domains                                                                               |
| ---------- | ------------------------------------------------------------------------------------- |
| Train      | `A_typical`, `B_high_res`, `G_biochemical_source`                                     |
| Validation | `C_low_res_noisy`                                                                     |
| Test       | `D_calibration_shift`, `E_window_shift`, `F_nrb_family_shift`, `H_biochemical_target` |

Another common setup is source-to-target biochemical transfer:

| Split | Domains                |
| ----- | ---------------------- |
| Train | `G_biochemical_source` |
| Test  | `H_biochemical_target` |

---

## Storage notes

Generated datasets can become large, especially when using:

* many samples per domain
* multiple seeds
* high-resolution axes
* saved latent arrays

For this reason, generated datasets are usually not committed to Git.

Recommended practice:

* commit only scripts, configuration files, documentation, QC summaries, and small example figures
* keep full generated datasets in external storage
* document the generation command and seed used to create each dataset

---

## Notes

The dataset format is designed to be simple, transparent, and compatible with standard scientific Python tools.

The most important arrays for benchmark use are:

```text
spectrum
raman_target
axis
metadata_json
```

The remaining arrays are mainly useful for debugging, visualization, and quality control.
