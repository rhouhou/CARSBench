# CARSBench v1 Benchmark Specification

**Version:** 1.0.0  
**Status:** Frozen  
**Primary machine-readable specification:** `specs/carsbench_v1.json`

## 1. Purpose

CARSBench is a controlled benchmark for evaluating the reliability and generalization of methods that recover Raman-like spectra from coherent anti-Stokes Raman scattering (CARS/BCARS) measurements.

The benchmark is designed around explicit distribution shifts rather than a single synthetic train/test split.

The central question is:

> How reliably do CARS-to-Raman retrieval methods generalize across acquisition physics, spectral coverage, biochemical diversity, and combinations of these shifts?

CARSBench v1 therefore separates different sources of out-of-distribution variation into individually controlled domains and compound stress conditions.

The benchmark does not claim that its procedural distributions reproduce the complete population distribution of experimental biological CARS/BCARS measurements. Its purpose is controlled and reproducible evaluation.

---

# 2. Canonical spectral representation

All CARSBench v1 spectra are represented on a common physical Raman-shift grid:

- minimum Raman shift: **400 cm⁻¹**
- maximum Raman shift: **3200 cm⁻¹**
- number of points: **1024**

The physical axis is fixed within all benchmark domains.

The approximate grid spacing is:

\[
\Delta\nu \approx
\frac{3200-400}{1024-1}
\approx 2.74\ \mathrm{cm^{-1}}.
\]

This fixed representation separates physical spectral generation from observation-window effects.

---

# 3. Observation-window representation

Spectral-coverage variation does not change the underlying physical Raman axis.

Instead, CARSBench uses:

\[
\boxed{
\text{canonical spectral grid}
+
\text{binary observation mask}
}
\]

The measured CARS/BCARS input is set to zero outside the observed region, while the full Raman target remains defined on the complete canonical grid.

The available observation windows are:

| Window | Range |
|---|---:|
| Full | 400–3200 cm⁻¹ |
| Fingerprint | 400–1800 cm⁻¹ |
| CH | 2700–3150 cm⁻¹ |
| Partial fingerprint | 800–1800 cm⁻¹ |
| Partial CH | 2800–3050 cm⁻¹ |

For the spectral-coverage OOD domain, these four restricted windows are sampled with probabilities:

| Window | Probability |
|---|---:|
| Fingerprint | 0.35 |
| CH | 0.35 |
| Partial fingerprint | 0.15 |
| Partial CH | 0.15 |

These probabilities are benchmark design parameters.

---

# 4. Resonant and non-resonant susceptibility

The CARS response is generated from resonant and non-resonant susceptibility components.

The intrinsic non-resonant susceptibility in CARSBench v1 is modeled as a real, frequency-independent reference term:

\[
\chi_{NR}(\nu)=1.
\]

Its normalized amplitude is therefore:

\[
|\chi_{NR}|=1.
\]

Frequency-dependent excitation and instrument effects are modeled separately through the effective system-response envelope.

This separation is intentional:

- intrinsic non-resonant susceptibility controls the coherent reference contribution;
- the system-response model controls broad spectral modulation;
- the R/NRB ratio controls global resonant strength.

---

# 5. Global resonant-to-NRB strength

CARSBench explicitly controls global resonant strength using:

\[
R=
\frac{
\max_\nu |\chi_R(\nu)|
}{
\operatorname{median}_\nu |\chi_{NR}(\nu)|
}.
\]

For every domain, the requested target ratio is sampled from a truncated log-normal distribution:

\[
\log R
\sim
\mathcal N(
\mu=-1.6917887382,
\sigma=0.6886309739
)
\]

subject to:

\[
0.02 \le R \le 2.5.
\]

The parameters were obtained from a stability audit of the original A-domain reference simulator using 25,000 generated samples.

The pooled reference statistics before explicit normalization were approximately:

| Statistic | R/NRB |
|---|---:|
| p01 | 0.0393 |
| p05 | 0.0607 |
| Median | 0.1828 |
| Mean | 0.2342 |
| p95 | 0.5785 |
| p99 | 0.9383 |
| Observed maximum | 2.385 |

The resulting fitted distribution is used as a **reference-matched benchmark control distribution**.

It must not be interpreted as a universal experimentally measured biological CARS R/NRB distribution.

After resonant susceptibility generation, CARSBench scales \(\chi_R\) so that the realized R/NRB ratio matches the requested target ratio.

The same global R/NRB distribution is used by all v1 domains. This prevents acquisition or biochemical domains from being unintentionally distinguished by overall resonant strength.

---

# 6. Raman lineshape convention

Individual resonant peaks use a complex Lorentzian susceptibility:

\[
\chi_j(\nu)
=
\frac{A_j}
{(\nu_j-\nu)-i\gamma_j}.
\]

Here:

- \(\nu_j\) is the center position;
- \(\gamma_j\) is the Lorentzian half-width at half maximum;
- therefore

\[
\mathrm{FWHM}=2\gamma_j.
\]

The biochemical prototype API defines peak amplitudes as approximate isolated peak heights in:

\[
\operatorname{Im}\chi_R.
\]

For a desired peak height \(h_j\), the Lorentzian numerator is therefore constructed as:

\[
A_j=h_j\gamma_j.
\]

This prevents linewidth changes from unintentionally changing the interpretation of the prototype amplitude parameter.

---

# 7. Biochemical prototype model

CARSBench v1 contains four biochemical prototype classes:

- lipid
- protein
- nucleic acid
- glycogen

Each prototype contains correlated Raman features with stochastic variation in:

- peak center;
- Lorentzian width;
- relative peak amplitude;
- optional peak presence.

Default stochastic peak variation is:

\[
\sigma_{\mathrm{center}}=2\ \mathrm{cm^{-1}}
\]

with selected exceptions such as the protein amide-I region.

Width variation uses a log-normal perturbation with:

\[
\sigma_{\log,\mathrm{width}}=0.15.
\]

Relative peak-height variation uses:

\[
\sigma_{\log,\mathrm{amplitude}}=0.25.
\]

The base peak positions are chosen from commonly reported biological Raman assignments.

However, the exact relative peak amplitudes, peak-presence probabilities, mixture concentration parameters, and procedural jitter distributions are benchmark design choices.

They are **not** claimed to be measured population frequencies.

---

# 8. Biochemical mixture model

Biochemical spectra are constructed from mixtures of prototype spectra:

\[
\chi_R(\nu)
=
\sum_j
w_j
\chi_j(\nu).
\]

Mixture weights are sampled from a symmetric Dirichlet distribution.

For the controlled chemistry benchmark:

\[
\alpha=3.0.
\]

Each selected biochemical component must satisfy:

\[
w_j\ge0.10.
\]

The weights satisfy:

\[
\sum_j w_j=1.
\]

No additional random prototype-level concentration scaling is applied.

No separate random global resonant scale is applied after biochemical mixture construction.

This gives each random variable one explicit role:

- peak variation controls within-component spectral shape;
- mixture weights control relative biochemical composition;
- target R/NRB controls global resonant strength.

---

# 9. Reference acquisition model

The reference A domain uses the following acquisition configuration.

## 9.1 Spectral resolution

Effective Gaussian instrument PSF:

\[
\mathrm{FWHM}\sim U(10,13)\ \mathrm{cm^{-1}}.
\]

This is an effective line-spread model rather than a claim about one universal BCARS spectrometer.

## 9.2 Effective system response

Reference envelope family:

`hybrid`

Reference ripple amplitude:

\[
0.
\]

## 9.3 Photon statistics

Detected photoelectron scale:

\[
N_e
\sim
\mathrm{LogUniform}(700,3000)
\]

photoelectrons per simulator intensity unit.

Photon counts are sampled using Poisson statistics.

## 9.4 Detector read noise

Gaussian read noise:

\[
\sigma_{\mathrm{read}}=0.7
\]

electrons.

## 9.5 Calibration

Reference calibration:

\[
\Delta\nu=0
\]

and

\[
\mathrm{warp}=0.
\]

## 9.6 Optical background

Optical background is disabled in the reference configuration:

- family: `none`
- fraction: `0`

## 9.7 Electronic baseline

The electronic baseline uses a polynomial/smooth correlated model.

Polynomial-scale parameter:

\[
U(0.008,0.015)
\]

Correlated component:

\[
U(0.002,0.005)
\]

with knot counts selected from:

\[
\{8,10,12\}.
\]

---

# 10. Individual benchmark domains

## A — `A_typical`

Reference domain.

No OOD shift is applied.

This is the source/reference domain for the acquisition-physics benchmark tracks.

---

## B — `B_high_res`

Tests higher spectral resolution.

Only the instrument PSF changes:

\[
\mathrm{FWHM}\sim U(7,9)\ \mathrm{cm^{-1}}.
\]

All other reference acquisition variables are inherited.

---

## C — `C_low_res`

Tests degraded spectral resolution.

Only the PSF changes:

\[
\mathrm{FWHM}\sim U(16,22)\ \mathrm{cm^{-1}}.
\]

---

## D — `D_high_noise`

Tests reduced photon-count conditions.

Photoelectron scale:

\[
N_e
\sim
\mathrm{LogUniform}(150,700).
\]

Read noise remains identical to the reference domain.

The intended change is therefore predominantly photon-statistics/SNR related.

---

## E — `E_calibration_shift`

Tests spectral-axis calibration mismatch.

Global shift:

\[
\Delta\nu
\sim
U(-3,3)\ \mathrm{cm^{-1}}.
\]

Smooth warp:

\[
W
\sim
U(-4,4)\ \mathrm{cm^{-1}}.
\]

Calibration error is applied to the measured spectrum while the Raman ground truth remains defined on the nominal canonical axis.

---

## F — `F_window_shift`

Tests spectral-coverage generalization.

The physical spectrum is always generated on:

\[
400-3200\ \mathrm{cm^{-1}}
\]

with 1024 points.

Only the observation mask changes.

The measured CARS/BCARS input is zero outside the sampled observation window.

The full Raman target is retained.

Thus F represents an information-coverage shift rather than a change to the underlying physical spectrum.

---

## G — `G_system_response_shift`

Tests effective excitation/instrument spectral-response generalization.

The reference `hybrid` response is replaced by:

`poly`

The intrinsic non-resonant susceptibility remains unchanged.

This prevents system-response variation from being conflated with intrinsic NRB variation.

---

# 11. Chemistry OOD domains

## H — `H_biochemical_source`

H is the chemistry source distribution.

Allowed components:

- lipid
- protein
- nucleic acid

Glycogen is completely absent.

Each spectrum contains:

\[
2\le N_{\mathrm{components}}\le3.
\]

Mixture concentration:

\[
\alpha=3.
\]

Every selected component satisfies:

\[
w_j\ge0.10.
\]

Minor random background peaks are disabled.

---

## I — `I_biochemical_target`

I is the held-out-component chemistry target.

Allowed components:

- lipid
- protein
- nucleic acid
- glycogen

Glycogen is required in every sample.

Each spectrum therefore contains unseen glycogen chemistry embedded within one or two familiar biochemical constituents.

Again:

\[
2\le N_{\mathrm{components}}\le3
\]

and

\[
w_j\ge0.10.
\]

This design intentionally creates a moderate chemistry OOD shift rather than replacing the complete biochemical distribution.

Quantitative validation showed substantial but incomplete separation between H and I spectral shapes.

Example spectral-angle statistics from the frozen chemistry audit were approximately:

| Comparison | Median angle |
|---|---:|
| H vs H | 0.602 rad |
| I vs I | 0.786 rad |
| H vs I | 0.883 rad |

The cross-domain distribution substantially overlaps the within-domain distributions, preventing chemistry membership from becoming a trivial near-orthogonal separation.

---

# 12. Compound OOD domains

## J — `J_compound_acquisition`

J combines four already validated acquisition shifts:

1. low spectral resolution;
2. high photon noise;
3. calibration mismatch;
4. altered system response.

Specifically:

\[
\mathrm{PSF\ FWHM}\sim U(16,22)
\]

\[
N_e\sim\mathrm{LogUniform}(150,700)
\]

\[
\Delta\nu\sim U(-3,3)
\]

\[
W\sim U(-4,4)
\]

and:

`envelope_family = poly`.

No spectral-coverage shift is included.

No chemistry override is included.

Therefore:

\[
\boxed{
A\rightarrow J
}
\]

is the compound acquisition-physics benchmark.

Paired validation confirmed that A and J can have identical latent Raman truth while their measured CARS/BCARS inputs differ because of acquisition physics.

---

## K — `K_joint_ood`

K combines the complete J acquisition shift with the held-out glycogen chemistry used in I.

Therefore K changes both:

- acquisition physics;
- biochemical distribution.

It does **not** include the F spectral-coverage shift.

The primary joint benchmark is:

\[
\boxed{
H\rightarrow K
}
\]

This tests simultaneous generalization to unfamiliar acquisition conditions and held-out biochemical structure.

---

# 13. Benchmark tracks

The frozen CARSBench v1 tracks are:

| Track | Source | Target |
|---|---|---|
| Higher resolution | A | B |
| Lower resolution | A | C |
| Noise | A | D |
| Calibration | A | E |
| Spectral coverage | A | F |
| System response | A | G |
| Chemistry | H | I |
| Compound acquisition | A | J |
| Joint OOD | H | K |

The individual domains are intended to diagnose failure mechanisms.

The compound domains evaluate interaction between shifts.

---

# 14. Why spectral coverage is excluded from compound J/K

F is deliberately not included in J or K.

Resolution, noise, calibration, and system-response changes modify the quality or transformation of available measurements.

F is qualitatively different because it removes parts of the measured spectral information.

Including F automatically in J/K would make it difficult to determine whether degradation arose from:

- acquisition-physics generalization;
- chemistry generalization;
- or missing information.

Spectral coverage is therefore retained as an independent benchmark axis in v1.

---

# 15. Deterministic generation

CARSBench uses deterministic per-sample seed derivation.

For master seed \(S\), domain \(D\), sample index \(i\), and random stream \(r\), a child seed is derived using BLAKE2b from:

\[
S|D|i|r.
\]

The 8-byte digest is converted to an unsigned little-endian integer and reduced modulo:

\[
2^{32}.
\]

Separate streams are used for:

- configuration sampling;
- simulation realization.

This design guarantees that a sample does not depend on:

- generation chunk size;
- the order in which domains are generated;
- whether a domain is generated alone or with other domains.

Exact saved-array and metadata equality are tested across chunk-size and domain-order changes.

---

# 16. Output structure

Each sample contains at minimum:

- canonical Raman-shift axis;
- measured CARS/BCARS input spectrum;
- Raman target;
- observation mask;
- metadata describing the realized simulation.

Additional arrays such as:

- clean intensity;
- envelope;
- baseline;
- resonant susceptibility;
- non-resonant susceptibility;

may be retained depending on generation settings.

For F, masked input coordinates are exactly zero while the Raman target remains defined on the full grid.

---

# 17. Interpretation of synthetic parameters

CARSBench distinguishes between three categories of parameters.

## 17.1 Physics-defined quantities

Examples include:

- complex Lorentzian susceptibility;
- HWHM/FWHM convention;
- Poisson photon statistics;
- explicit resonant-to-NRB scaling;
- calibration resampling;
- Gaussian PSF convolution.

## 17.2 Literature-informed benchmark ranges

Examples include:

- biological Raman peak locations;
- approximate biological linewidth ranges;
- reference and shifted spectral-resolution regimes;
- experimentally plausible photon-count regimes;
- calibration perturbation scales.

## 17.3 Procedural benchmark parameters

Examples include:

- exact relative prototype peak amplitudes;
- peak-presence probabilities;
- Dirichlet concentration;
- observation-window probabilities;
- some envelope-shape parameterizations;
- the fitted reference-matched R/NRB distribution.

Procedural parameters are chosen to produce controlled diversity and reproducible OOD conditions.

They must not be presented as universal experimental population distributions.

---

# 18. Known modeling limitations

CARSBench v1 is intentionally a reduced simulation model.

Important limitations include:

1. The effective spectral envelope is applied as a reduced system-response model rather than explicitly simulating full pump, Stokes, and probe electric fields.

2. Intrinsic \(\chi_{NR}\) is represented as a real frequency-independent constant.

3. Biochemical spectra are generated from procedural prototype mixtures rather than directly sampled from measured Raman datasets.

4. Peak-presence probabilities and relative amplitudes are not experimentally estimated population statistics.

5. The current benchmark does not model all experimental artifacts that may occur in a real BCARS instrument.

6. Cross-simulator and experimental-transfer evaluation are separate benchmark stages and are not replaced by the internal synthetic-domain tests.

These limitations are explicit design boundaries rather than hidden assumptions.

---

# 19. Validation performed before v1 freeze

Before freezing the v1 specification, the following checks were performed:

- prototype-generation bug audit;
- biochemical peak-count audit;
- calibration-direction audit;
- intrinsic NRB audit;
- resonant-to-NRB scaling audit;
- optical/electronic background separation;
- PSF boundary-condition tests;
- per-sample stochastic domain resolution;
- configuration/simulation seed separation;
- reference R/NRB stability over 25,000 spectra;
- bounded R/NRB control;
- chemistry spectral-angle audit;
- chemistry minimum-weight audit;
- F canonical-axis and observation-mask disk-level smoke test;
- exact chunk-size generation invariance;
- exact domain-order generation invariance;
- compound-domain configuration tests;
- paired A→J acquisition-only validation;
- H→K joint-shift validation;
- machine-readable v1 specification consistency tests.

The unit-test suite is required to remain green against the frozen specification.

---

# 20. Benchmark scope beyond v1 synthetic domains

The frozen synthetic-domain specification is the controlled core of CARSBench.

Planned subsequent benchmark layers include:

## Real Raman diversity

Measured Raman spectra can replace or complement procedural biochemical prototypes after the synthetic v1 benchmark is frozen.

This provides a stronger chemistry-diversity test without altering the controlled acquisition domains.

## Cross-simulator evaluation

Models trained using the CARSBench simulator should be tested against independently generated synthetic CARS/BCARS data.

This evaluates simulator-specific overfitting.

## Experimental transfer

Where appropriate experimental datasets are available, CARSBench-trained retrieval models should be evaluated without silently tuning the benchmark simulator to the target measurements.

This tests synthetic-to-experimental transfer.

These stages are complementary to, rather than replacements for, the controlled A–K domain structure.

---

# 21. Versioning rule

The file:

`specs/carsbench_v1.json`

is the authoritative machine-readable definition of CARSBench v1.

Changes that alter benchmark semantics, including:

- domain parameter ranges;
- chemistry composition;
- R/NRB distribution;
- observation-window definitions;
- compound-domain composition;
- canonical spectral axis;

must not silently modify v1.

Such changes require a new specification version.

Implementation refactoring that leaves the frozen benchmark semantics unchanged is permitted, provided the specification consistency and reproducibility tests continue to pass.