# CARSBench domain definitions

CARSBench is designed around controlled benchmark domains for testing cross-domain generalization in CARS/BCARS Raman-retrieval models.

Each domain represents a different type of shift that may occur in spectroscopy workflows, such as changes in acquisition quality, spectral calibration, non-resonant background, or biochemical composition.

## Domain overview

| Domain | Name                   | Main shift type            | Purpose                                                      |
| ------ | ---------------------- | -------------------------- | ------------------------------------------------------------ |
| A      | `A_typical`            | Reference domain           | Typical BCARS/CARS acquisition conditions                    |
| B      | `B_high_res`           | Spectral-resolution shift  | Tests performance on sharper spectral features               |
| C      | `C_low_res_noisy`      | Resolution and noise shift | Tests robustness to degraded spectra                         |
| D      | `D_calibration_shift`  | Calibration shift          | Tests robustness to spectral-axis shift and warp             |
| E      | `E_window_shift`       | Spectral-window shift      | Tests robustness to changed or partial spectral coverage     |
| F      | `F_nrb_family_shift`   | NRB-family shift           | Tests robustness to different non-resonant-background shapes |
| G      | `G_biochemical_source` | Biochemical source domain  | Represents one biochemical composition regime                |
| H      | `H_biochemical_target` | Biochemical target domain  | Represents a shifted biochemical composition regime          |

---

## A_typical

`A_typical` is the reference domain.

It represents typical simulated BCARS/CARS acquisition conditions and is used as the baseline domain for comparison with all other domains.

Expected behavior:

* moderate spectral resolution
* typical noise level
* typical non-resonant background
* typical spectral window
* typical biochemical mixture variability

Typical use:

* baseline training domain
* reference for QC comparisons
* sanity check for simulation behavior

---

## B_high_res

`B_high_res` represents a higher-resolution acquisition setting.

Compared with `A_typical`, spectra in this domain are expected to have sharper spectral features.

Expected behavior:

* narrower instrument point-spread function
* sharper peaks
* potentially easier visual peak separation
* possible mismatch when models are trained only on lower-resolution spectra

Typical use:

* test whether retrieval models generalize to sharper features
* evaluate resolution sensitivity
* compare peak-shape robustness

---

## C_low_res_noisy

`C_low_res_noisy` represents a degraded acquisition setting.

Compared with `A_typical`, spectra in this domain are expected to have lower spectral resolution and stronger noise.

Expected behavior:

* broader spectral features
* higher noise level
* reduced peak visibility
* more difficult Raman-target recovery

Typical use:

* stress-test retrieval models
* evaluate robustness to noisy measurement conditions
* compare denoising or regularized retrieval methods

---

## D_calibration_shift

`D_calibration_shift` represents spectral calibration errors.

This domain introduces shifts or warping in the wavenumber axis.

Expected behavior:

* spectral-axis offset
* possible nonlinear axis distortion
* peaks may appear shifted relative to the reference domain
* models trained on perfectly aligned data may degrade

Typical use:

* evaluate calibration robustness
* test preprocessing and alignment methods
* study sensitivity to spectral-axis errors

---

## E_window_shift

`E_window_shift` represents a change in spectral-window coverage.

This domain tests whether models remain robust when spectra are measured over a different wavenumber range.

Expected behavior:

* shifted or changed spectral window
* possible missing bands compared with the reference domain
* altered available biochemical information

Typical use:

* evaluate robustness to different acquisition windows
* test models under partial spectral coverage
* study transfer between different instrument settings

---

## F_nrb_family_shift

`F_nrb_family_shift` represents a non-resonant-background shift.

The non-resonant background is an important part of CARS/BCARS spectra. This domain changes the NRB shape family to test whether Raman-retrieval models overfit to a specific background style.

Expected behavior:

* different background shape
* different smooth intensity trend
* altered interference between resonant and non-resonant components
* possible degradation in retrieval methods that assume one NRB family

Typical use:

* test NRB robustness
* compare background-correction methods
* evaluate phase-retrieval stability under background mismatch

---

## G_biochemical_source

`G_biochemical_source` represents a source biochemical composition domain.

This domain is designed to emphasize one biochemical mixture regime, such as lipid/protein-dominant patterns.

Expected behavior:

* biochemical peak distributions characteristic of the source regime
* different relative peak intensities compared with target chemistry
* useful for source-domain training

Typical use:

* source domain for biochemical transfer experiments
* train models on one composition regime
* evaluate whether models learn general spectral structure or memorize source patterns

---

## H_biochemical_target

`H_biochemical_target` represents a target biochemical composition domain.

This domain is designed to differ from `G_biochemical_source`, for example by emphasizing nucleic-acid or aromatic patterns.

Expected behavior:

* shifted biochemical composition compared with `G_biochemical_source`
* different relative peak intensities
* different dominant spectral bands
* possible generalization gap for models trained only on source chemistry

Typical use:

* target domain for biochemical transfer experiments
* evaluate out-of-domain retrieval performance
* test domain-generalization methods

---

## Example benchmark setups

### Measurement robustness

Train on:

```text
A_typical
```

Test on:

```text
B_high_res
C_low_res_noisy
D_calibration_shift
E_window_shift
F_nrb_family_shift
```

Purpose:

Evaluate whether a Raman-retrieval model is robust to acquisition and measurement shifts.

---

### Biochemical generalization

Train on:

```text
G_biochemical_source
```

Test on:

```text
H_biochemical_target
```

Purpose:

Evaluate whether a model trained on one biochemical composition regime can generalize to another.

---

### Mixed-source training

Train on:

```text
A_typical
B_high_res
C_low_res_noisy
G_biochemical_source
```

Test on:

```text
D_calibration_shift
E_window_shift
F_nrb_family_shift
H_biochemical_target
```

Purpose:

Evaluate cross-domain generalization when the model sees multiple but not all domain types during training.

---

## Notes

The domain definitions are intended to create controlled simulation shifts, not to perfectly reproduce every experimental CARS/BCARS acquisition scenario.

The purpose of these domains is to make generalization failures measurable, interpretable, and reproducible.
