# CARSBench

**CARSBench** is a physics-informed simulation and benchmarking framework for  
**Coherent Anti-Stokes Raman Scattering (CARS/BCARS)**.

It enables realistic modeling of:
- complex resonant and nonresonant susceptibilities (χR, χNRB)
- instrument variability across CARS/BCARS systems
- noise, calibration shifts, and spectral distortions
- cross-domain generalization for AI-based spectral reconstruction

---

## 🎯 Motivation

CARS and BCARS measurements are inherently:
- indirect (nonlinear optical response)
- affected by nonresonant background (NRB)
- sensitive to instrument-specific variations

Most existing approaches:
- rely heavily on simulated data
- do not account for variability across instruments
- fail to generalize beyond training conditions

**CARSBench addresses this gap** by introducing:
- variability-aware simulation
- domain-driven evaluation
- standardized benchmarking for generalization

---

## 🔬 What CARSBench Provides

### 1. Physics-Informed Simulation
- Complex susceptibility modeling:
  - χ_total = χ_R + χ_NRB
- BCARS intensity generation:
  - I(ω) = |χ_total(ω)|²
- Instrument effects:
  - spectral resolution (PSF)
  - envelope modulation
  - calibration shifts

---

### 2. Variability Modeling
Simulation spans realistic domains:
- spectral range and resolution
- noise regimes (Poisson + read noise)
- NRB magnitude and phase variability
- instrument-induced distortions

---

### 3. Domain-Based Benchmarking
Built-in evaluation using **Leave-One-Domain-Out (LODO)**:

Train on multiple domains → test on unseen domain

Domains include:
- Typical
- Noisy
- High-resolution
- Calibration-shifted

---

### 4. Flexible Data Generation
Supports:
- single spectra
- batched datasets
- hyperspectral images (H × W × N)

---

## 📦 Installation

```bash
pip install -e .