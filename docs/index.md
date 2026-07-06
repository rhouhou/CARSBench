# CARSBench documentation

This folder contains supporting documentation for CARSBench.

CARSBench is an alpha-stage simulation and benchmarking framework for testing cross-domain generalization in CARS/BCARS Raman-retrieval models.

---

## Documentation pages

| Page                                       | Description                                                                                 |
| ------------------------------------------ | ------------------------------------------------------------------------------------------- |
| [`domains.md`](domains.md)                 | Explains the eight benchmark domains and their intended domain shifts                       |
| [`dataset_format.md`](dataset_format.md)   | Describes the generated dataset structure, `.npz` batch files, metadata, and manifest files |
| [`reproducibility.md`](reproducibility.md) | Explains recommended seeds, generation commands, and reproducibility practices              |

---

## Main concepts

CARSBench is built around four main ideas:

1. Generate synthetic CARS/BCARS spectra with controlled variation.
2. Define benchmark domains that represent realistic acquisition and biochemical shifts.
3. Save generated spectra in a reproducible chunked dataset format.
4. Use the generated datasets to evaluate Raman-retrieval and machine-learning models under cross-domain conditions.

---

## Recommended reading order

Start with:

1. [`domains.md`](domains.md)
2. [`dataset_format.md`](dataset_format.md)
3. [`reproducibility.md`](reproducibility.md)

This gives the most important background for understanding how CARSBench datasets are generated and how they should be used in benchmark experiments.

---

## Notes

The documentation is intentionally lightweight at this stage.

The goal is to make the repository easy to understand for:

* researchers
* reviewers
* collaborators
* recruiters
* hiring managers
* future users of the package

More detailed API documentation may be added later when the package interface becomes more stable.
