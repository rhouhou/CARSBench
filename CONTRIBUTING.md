# Contributing to CARSBench

Thank you for your interest in CARSBench.

CARSBench is an alpha-stage research and portfolio project for simulating CARS/BCARS spectra and evaluating cross-domain generalization in Raman-retrieval workflows.

Contributions, suggestions, bug reports, and documentation improvements are welcome.

---

## Development setup

Clone the repository:

```bash
git clone https://github.com/rhouhou/CARSBench.git
cd CARSBench
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package with development dependencies:

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ".[dev,analysis]"
```

---

## Running tests

Run the test suite with:

```bash
python3 -m pytest
```

The tests check the public API, available benchmark domains, and reproducibility behavior.

---

## Formatting and linting

CARSBench uses Black for formatting and Ruff for linting.

Format the code with:

```bash
python3 -m black src scripts tests
```

Run Ruff with:

```bash
python3 -m ruff check src scripts tests
```

To automatically fix safe Ruff issues:

```bash
python3 -m ruff check src scripts tests --fix
```

Before committing, it is recommended to run:

```bash
python3 -m black src scripts tests
python3 -m ruff check src scripts tests
python3 -m pytest
```

---

## Generated data and large files

Please do not commit large generated datasets.

Generated outputs such as full `.npz` datasets, large arrays, temporary experiment outputs, and local data folders should stay outside Git.

Recommended to commit:

* source code
* tests
* documentation
* small QC summary files
* small benchmark summary CSV files
* small example figures used in documentation

Recommended not to commit:

* full generated datasets
* large `.npz` files
* large `.npy` files
* model checkpoints
* temporary experiment folders
* local environment files

---

## Documentation updates

Documentation lives mainly in:

```text
README.md
docs/
qc/README.md
results/README.md
```

When adding a new feature, consider updating the relevant documentation page.

Useful documentation pages include:

* `docs/domains.md`
* `docs/dataset_format.md`
* `docs/reproducibility.md`
* `docs/baselines.md`

---

## Pull request checklist

Before opening a pull request, please check:

* the code runs locally
* tests pass
* Black formatting passes
* Ruff linting passes
* documentation is updated if needed
* generated large files are not included
* new behavior is covered by a test when possible

Recommended local check:

```bash
python3 -m black src scripts tests
python3 -m ruff check src scripts tests
python3 -m pytest
```

---

## Project scope

CARSBench focuses on:

* CARS/BCARS spectral simulation
* controlled benchmark domain shifts
* Raman-equivalent target generation
* reproducible synthetic dataset generation
* quality-control workflows
* baseline benchmark evaluation

Features outside this scope may be better suited for companion projects such as `prCARS` or `CARSGuard`.

---

## Safety and limitations

CARSBench is intended for research, education, and portfolio demonstration.

It is not intended for clinical diagnosis, medical decision-making, or deployment in real healthcare settings.
