# CARSBench

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/status-alpha-orange)
![Project Type](https://img.shields.io/badge/project-scientific%20ML%20benchmark-purple)
![CI](https://github.com/rhouhou/CARSBench/actions/workflows/ci.yml/badge.svg)

**CARSBench** is a simulation and benchmarking framework for broadband Coherent Anti-Stokes Raman Scattering (BCARS/CARS) spectroscopy.

It is designed to generate synthetic CARS/BCARS spectra with controlled domain shifts so that Raman-retrieval and machine-learning models can be tested for robustness across different acquisition, background, calibration, and biochemical conditions.

---

## Why this project matters

Machine-learning models for Raman retrieval from CARS/BCARS spectra can perform well on one simulated or experimental setting but fail when the acquisition conditions change.

CARSBench addresses this by creating controlled benchmark domains that vary factors such as:

* spectral resolution
* detector noise
* baseline drift
* spectral calibration
* spectral window
* non-resonant background shape
* biochemical composition

The goal is to support systematic domain-generalization experiments for spectroscopy-aware machine learning.

---

## What this repository demonstrates

This project demonstrates:

* scientific simulation design
* modular Python package engineering
* domain-shift benchmark construction
* reproducible synthetic dataset generation
* Raman-equivalent target generation
* quality-control workflows for simulated spectra
* baseline evaluation for Raman-retrieval benchmarking
* automated testing, formatting, linting, and CI

---

## Example output

![Example CARSBench simulated spectrum](docs/assets/example_spectrum.png)

---

## Key features

* Frequency-domain BCARS/CARS forward simulation
* Complex resonant and non-resonant susceptibility modeling
* Biochemical prototype-based Raman-like signal generation
* Eight benchmark domain presets
* Per-sample parameter variability
* Reproducible generation with fixed random seeds
* Chunked dataset writing for large synthetic datasets
* Metadata export for simulation parameters
* Quality-control and visualization scripts
* Simple benchmark metrics for Raman-retrieval evaluation
* Lightweight baseline benchmark script
* Basic unit tests and GitHub Actions CI

---

## Project status

CARSBench is currently an **alpha-stage research and portfolio project**.

| Component                              | Status      |
| -------------------------------------- | ----------- |
| Frequency-domain BCARS/CARS simulation | Implemented |
| Eight domain presets                   | Implemented |
| Raman-equivalent target generation     | Implemented |
| Chunked dataset writing                | Implemented |
| Multi-seed generation workflow         | Implemented |
| QC and validation scripts              | Implemented |
| Visualization scripts                  | Implemented |
| Baseline benchmark utilities           | Implemented |
| Basic API tests                        | Implemented |
| Domain generation tests                | Implemented |
| Reproducibility tests                  | Implemented |
| Dataset I/O tests                      | Implemented |
| Benchmark metric tests                 | Implemented |
| GitHub Actions CI                      | Implemented |
| Full ML training benchmark             | Planned     |
| Real experimental validation           | Planned     |

---

## Simulation pipeline

CARSBench separates the simulation process into four main stages.

### 1. Biochemical prototype library

Raman-like resonant peaks are generated from biochemical prototype components such as lipid, protein, nucleic-acid, and aromatic spectral patterns.

### 2. Clean Raman-like mixture generation

Random mixtures of prototype components create sample-to-sample biochemical variability.

### 3. CARS/BCARS forward model

The resonant susceptibility is combined with a non-resonant background to generate a CARS-like intensity signal.

### 4. Measurement and domain effects

Domain-specific acquisition effects are applied, including spectral resolution, noise, baseline drift, calibration shift, spectral-window shift, and NRB variation.

---

## Benchmark domains

CARSBench currently includes eight domains.

| Domain                 | Description                          | Main shift type   |
| ---------------------- | ------------------------------------ | ----------------- |
| `A_typical`            | Typical BCARS acquisition            | Reference domain  |
| `B_high_res`           | Higher spectral resolution           | Measurement shift |
| `C_low_res_noisy`      | Lower resolution with stronger noise | Measurement shift |
| `D_calibration_shift`  | Spectral calibration shift and warp  | Calibration shift |
| `E_window_shift`       | Different spectral window            | Window shift      |
| `F_nrb_family_shift`   | Different NRB shape family           | NRB shift         |
| `G_biochemical_source` | Lipid/protein-dominant chemistry     | Biochemical shift |
| `H_biochemical_target` | Nucleic/aromatic-dominant chemistry  | Biochemical shift |

For a more detailed explanation of each domain and suggested benchmark setups, see [`docs/domains.md`](docs/domains.md).

These domains are intended for cross-domain generalization experiments, for example:

* train on typical acquisition conditions and test on noisy spectra
* train on one biochemical composition and test on another
* evaluate whether retrieval methods are robust to NRB-family changes
* evaluate whether calibration shifts degrade Raman-retrieval quality

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rhouhou/CARSBench.git
cd CARSBench
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On some macOS/Linux systems, you may need to use `python3` instead of `python`.

Install the package in editable mode:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

For development plus analysis and plotting tools:

```bash
python -m pip install -e ".[dev,analysis]"
```

You can also install the local analysis requirements with:

```bash
python -m pip install -r requirements.txt
```

---

## Installation check

After installation, verify that CARSBench can be imported and that the domain registry is available:

```bash
python -c "import CARSBench as cb; print(cb.list_domains())"
```

Expected output should include the benchmark domains:

```text
['A_typical', 'B_high_res', 'C_low_res_noisy', 'D_calibration_shift', 'E_window_shift', 'F_nrb_family_shift', 'G_biochemical_source', 'H_biochemical_target']
```

You can also run the smoke test:

```bash
python scripts/00_smoke_test.py
```

The smoke test checks that dataset generation, sample writing, batch writing, and reading work correctly.

---

## Quickstart

Generate a small dataset from the reference domain:

```python
import CARSBench as cb

batch = cb.generate_dataset(
    num_samples=100,
    domain_name="A_typical",
    seed=42,
)

print(len(batch.samples))
print(cb.list_domains())
```

Access one simulated sample:

```python
sample = batch.samples[0]

axis = sample.axis
spectrum = sample.spectrum
target = sample.raman_target
```

Plot a simulated CARS/BCARS spectrum:

```python
import matplotlib.pyplot as plt

plt.plot(axis, spectrum)
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.title("Simulated CARS/BCARS spectrum")
plt.show()
```

---

## Command-line workflow

### 1. Run the smoke test

```bash
python scripts/00_smoke_test.py
```

This checks that the public API works, a small dataset can be generated, and sample/batch writing and reading work correctly.

### 2. Generate a small pilot dataset

```bash
python scripts/01_generate_full_dataset.py \
  --output-root data/carsbench_pilot \
  --samples-per-domain 500 \
  --chunk-size 250 \
  --seed 42 \
  --include-latents
```

### 3. Generate the full benchmark dataset

```bash
python scripts/01_generate_full_dataset.py \
  --output-root data/carsbench_v1/seed_42 \
  --samples-per-domain 5000 \
  --chunk-size 500 \
  --seed 42 \
  --include-latents
```

### 4. Generate multiple seeds

The benchmark design uses three seeds:

```text
42, 123, 777
```

You can either run `01_generate_full_dataset.py` manually for each seed or edit the `OUTPUT_ROOT` variable in:

```bash
scripts/01_generate_all_seeds.py
```

Then run:

```bash
python scripts/01_generate_all_seeds.py
```

### 5. Generate validation figures

```bash
python scripts/06_validate_spectra.py \
  --data-root data/carsbench_v1 \
  --output-dir figures/spectra_validation
```

### 6. Run general domain QC

```bash
python scripts/08_general_domain_qc.py \
  --data-root data/carsbench_v1 \
  --output-csv qc/general_qc_spectrum.csv \
  --value-key spectrum
```

You can also run QC on the clean intensity and Raman target:

```bash
python scripts/08_general_domain_qc.py \
  --data-root data/carsbench_v1 \
  --output-csv qc/general_qc_clean_intensity.csv \
  --value-key clean_intensity

python scripts/08_general_domain_qc.py \
  --data-root data/carsbench_v1 \
  --output-csv qc/general_qc_raman_target.csv \
  --value-key raman_target
```

### 7. Run domain-specific QC

```bash
python scripts/09_specific_domain_qc.py \
  --data-root data/carsbench_v1 \
  --output-csv qc/specific_domain_qc.csv
```

---

## Dataset format

Generated datasets are written in a chunked format.

For a more detailed explanation of the generated files and array keys, see [`docs/dataset_format.md`](docs/dataset_format.md).

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
        ...
    seed_123/
      ...
    seed_777/
      ...
```

Each `.npz` batch may contain arrays such as:

| Key               | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| `axis`            | Wavenumber axis                                             |
| `spectrum`        | Simulated measured CARS/BCARS spectrum                      |
| `raman_target`    | Raman-equivalent target signal                              |
| `clean_intensity` | Clean forward intensity before selected measurement effects |
| `envelope`        | Instrument/envelope contribution, when saved                |
| `baseline`        | Baseline contribution, when saved                           |
| `metadata_json`   | Per-sample simulation metadata                              |

When `--include-latents` is used, additional latent arrays may also be saved, such as resonant and non-resonant susceptibility components.

---

## Recommended dataset sizes

| Dataset type      | Samples per domain | Use case                     |
| ----------------- | -----------------: | ---------------------------- |
| Smoke test        |               3-10 | API and I/O check            |
| Pilot dataset     |                500 | Fast validation and plotting |
| Benchmark dataset |               5000 | Main cross-domain benchmark  |
| Large dataset     |             10000+ | Extended ML experiments      |

For most development work, start with the pilot dataset before generating the full benchmark.

---

## Quality-control outputs

The `qc/` folder stores CSV summaries from validation scripts.

Typical QC files include:

| File                             | Purpose                                              |
| -------------------------------- | ---------------------------------------------------- |
| `general_qc_spectrum.csv`        | Domain-level QC on measured spectra                  |
| `general_qc_clean_intensity.csv` | QC before selected detector/noise effects            |
| `general_qc_raman_target.csv`    | QC on Raman-equivalent target signals                |
| `specific_domain_qc.csv`         | Checks for expected domain-specific parameter shifts |

The QC workflow is intended to verify that each domain produces the expected type of variation before the dataset is used for ML benchmarking.

See [`qc/README.md`](qc/README.md) for more details.

---

## Python API

List available domains:

```python
import CARSBench as cb

domains = cb.list_domains()
print(domains)
```

Generate a single-domain dataset:

```python
batch = cb.generate_dataset(
    num_samples=100,
    domain_name="A_typical",
    seed=42,
)
```

Generate a multi-domain dataset:

```python
batch = cb.generate_multi_domain_dataset(
    domain_names=["A_typical", "C_low_res_noisy", "F_nrb_family_shift"],
    samples_per_domain=100,
    seed=42,
)
```

Use benchmark metrics:

```python
from CARSBench import rmse, mae, spectral_angle

error_rmse = rmse(prediction, target)
error_mae = mae(prediction, target)
angle = spectral_angle(prediction, target)
```

---

## Baseline benchmark example

CARSBench includes a lightweight baseline benchmark script:

```bash
python scripts/12_run_baseline_benchmark.py
```

This script evaluates simple non-learning baselines across all benchmark domains and saves the results to:

```text
results/benchmark/baseline_results.csv
```

The included baselines are intended as sanity checks, not as strong Raman-retrieval methods.

They help verify that:

* datasets can be generated across all domains
* Raman-equivalent targets are available
* benchmark metrics can be computed
* domain-level evaluation outputs can be saved and compared

Typical output metrics include:

| Metric         | Meaning                                                     |
| -------------- | ----------------------------------------------------------- |
| RMSE           | Root mean squared error between prediction and Raman target |
| MAE            | Mean absolute error between prediction and Raman target     |
| Spectral angle | Shape-based similarity between prediction and Raman target  |

The baseline benchmark provides a simple starting point for future comparisons with stronger retrieval methods, phase-retrieval pipelines, or machine-learning models.

See [`docs/baselines.md`](docs/baselines.md) for more details.

---

## Tests and CI

Run the test suite locally:

```bash
python -m pytest
```

Run formatting and linting checks:

```bash
python -m black --check src scripts tests
python -m ruff check src scripts tests
```

Apply formatting locally:

```bash
python -m black src scripts tests
python -m ruff check src scripts tests --fix
python -m black src scripts tests
```

GitHub Actions runs tests, formatting checks, linting checks, and the smoke test on each push and pull request.

---

## Repository structure

```text
CARSBench/
  docs/
    Documentation and project notes

  qc/
    Quality-control CSV outputs

  results/
    Lightweight benchmark summaries and result documentation

  scripts/
    00_smoke_test.py
    01_generate_all_seeds.py
    01_generate_full_dataset.py
    02_qa_simulation.py
    03_paper_figures.py
    04_boxplot.py
    05_categorical_bar.py
    06_validate_spectra.py
    07_validate_chemistry_GH.py
    08_general_domain_qc.py
    09_specific_domain_qc.py
    11_make_readme_figures.py
    12_run_baseline_benchmark.py

  src/CARSBench/
    benchmark/
      Metrics and baseline benchmark utilities

    configs/
      Default simulation configuration

    datasets/
      Sample schema, simulation, reading, writing, and batch generation

    domains/
      Domain registry and domain-specific parameter presets

    instrument/
      Instrument envelope, resolution, and measurement effects

    io/
      Input/output utilities

    physics/
      CARS/BCARS forward-model components

    spatial/
      Hyperspectral/spatial simulation utilities

    tasks/
      Benchmark task definitions

    utils/
      Utility functions

    viz/
      Plotting and visualization helpers

  tests/
    Unit and integration tests for API, domains, metrics, reproducibility, and I/O
```

---

## Reproducibility

CARSBench uses explicit random seeds for reproducible dataset generation.

The recommended benchmark seeds are:

```text
42, 123, 777
```

Each generated domain includes metadata and a manifest file so that simulation settings can be inspected after generation.

For detailed seed recommendations, generation commands, and reporting practices, see [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Documentation

Additional documentation is available in the [`docs/`](docs/) folder.

Recommended pages:

* [`docs/domains.md`](docs/domains.md)
* [`docs/dataset_format.md`](docs/dataset_format.md)
* [`docs/reproducibility.md`](docs/reproducibility.md)
* [`docs/baselines.md`](docs/baselines.md)

---

## Limitations

CARSBench is a simulation and benchmarking framework. It is intended for research, education, and portfolio demonstration.

Current limitations include:

* The simulator is not a substitute for experimental validation.
* The generated spectra are synthetic and depend on the assumptions in the simulation model.
* Full ML training pipelines are not yet included.
* Real-data validation is planned but not yet part of the core benchmark.
* Benchmark results should be interpreted as simulation-based evaluation, not experimental proof.

This project is **not intended for clinical diagnosis, medical decision-making, or deployment in real healthcare settings**.

---

## Roadmap

Planned improvements include:

* Expand test coverage for simulation physics, domain presets, and benchmark tasks
* Add stronger baseline benchmark methods
* Add simple ML baselines for Raman-retrieval evaluation
* Add calibration and error-analysis plots
* Add example cross-domain benchmark reports
* Add integration examples with `prCARS` and `CARSGuard`
* Add real-data comparison workflows
* Add optional experiment tracking with MLflow or Weights & Biases
* Add lightweight documentation pages for API usage

---

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for version history.

---

## Citation

If you use CARSBench in research, education, or benchmarking work, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

```bibtex
@misc{carsbench2026,
  title={CARSBench: A Simulation and Domain-Generalization Benchmark for BCARS/CARS Spectroscopy},
  author={Houhou, Rola},
  year={2026},
  note={Alpha research software},
  url={https://github.com/rhouhou/CARSBench}
}
```

---

## License

This project is licensed under the MIT License.
