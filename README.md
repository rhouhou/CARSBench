
# CARSBench

CARSBench is a simulation and benchmark framework for broadband  
Coherent Anti-Stokes Raman Scattering (BCARS) spectroscopy.

The framework enables **controlled domain-generalization experiments** for
machine learning models performing **Raman retrieval from CARS spectra**.

CARSBench provides:

- physically grounded BCARS forward simulation
- biochemical prototype-based Raman generation
- configurable instrument and acquisition domain shifts
- reproducible dataset generation
- benchmarking utilities for ML models

---

## Key idea

The simulation pipeline separates the generative process into four stages:

1. **Biochemical prototype library**

   Realistic correlated Raman peaks are generated from biochemical components
   such as lipid, protein, and nucleic acid prototypes.

2. **Clean Raman mixture generation**

   Random mixtures of prototypes create realistic sample-to-sample variability.

3. **CARS forward model**

   Raman susceptibility is combined with a non-resonant background:

4. **Measurement effects**

Domain-specific effects simulate realistic acquisition differences:

- spectral resolution
- detector noise
- baseline drift
- NRB variations
- spectral window shifts
- calibration errors

This design allows systematic study of **domain shift robustness**.

---

## Domain families

The benchmark currently includes the following domains.

| Domain | Description |
| ------ | ------------- |
| A_typical | typical BCARS acquisition |
| B_high_res | higher spectral resolution |
| C_low_res_noisy | lower resolution with higher noise |
| D_calibration_shift | spectral calibration shift |
| E_window_shift | different spectral window |
| F_nrb_family_shift | different NRB shape family |
| G_biochemical_source | lipid/protein dominant chemistry |
| H_biochemical_target | nucleic/aromatic dominant chemistry |

These domains create three types of shifts:

- measurement shift
- NRB shift
- biochemical composition shift

---

## Installation

Clone the repository.

```bash
git clone https://github.com/rhouhou/CARSBench.git
cd CARSBench
```

Install in editable mode.

```bash
pip install -e .
```

Optional development dependencies:

```bash
pip install -e ".[dev]"
```

---

## Quickstart

Generate a small dataset.

```python
import CARSBench as cb

batch = cb.generate_dataset(
    num_samples=100,
    domain_name="A_typical",
    seed=42,
)
print(len(batch.samples))
```

Access a sample:

```python
sample = batch.samples[0]

axis = sample.axis
spectrum = sample.spectrum
target = sample.raman_target
```

Plot a spectrum:

```python
import matplotlib.pyplot as plt

plt.plot(axis, spectrum)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Intensity")
plt.show()
```

---

## Project structure

```bash
CARSBench/
│
├── scripts/
│   ├── smoke_test.py
│   ├── generate_full_dataset.py
│   ├── qa_simulation.py
│   └── paper_figures.py
│
├── src/CARSBench/
│
├── tests/
│
└── data/ (ignored)
```

Library code lives inside src/.

Runnable workflows are in scripts/.

---

## Typical workflow

1. Run smoke test:

    ```bash
    python scripts/smoke_test.py
    ```

2. Generate a pilot dataset:

    ```bash
    python scripts/generate_full_dataset.py --samples-per-domain 500 --chunk-size 250
    ```

3. Run simulation QA:

    ```bash
    python scripts/qa_simulation.py
    ```

4. Generate full benchmark dataset:

    ```bash
    python scripts/generate_full_dataset.py --samples-per-domain 5000 --chunk-size 500
    ```

5. Generate figures:

    ```bash
    python scripts/paper_figures.py
    ```

---

## Dataset format

Datasets are written in chunked format:

```bash
data/
  carsbench_v1/
    A_typical/
      batches/
        batch_000.npz
        batch_001.npz
    B_high_res/
      ...
```

Chunked storage avoids large memory usage and allows interrupted runs to resume.

---

## Recommended dataset sizes

| dataset | samples per domain |
| ------ | ------------- |
| pilot | 500 |
| benchmark | 5000 |
| large | 10000 |

Start with the pilot dataset for validation.

---

## Citation

If you use CARSBench in your research, please cite:

```bash
@misc{carsbench2026,
  title={CARSBench: A Benchmark for Domain Generalization in Broadband CARS Spectroscopy},
  author={Rola Houhou},
  year={2026}
}
```

---

## License

MIT License

---
