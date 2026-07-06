# Changelog

All notable changes to CARSBench will be documented in this file.

This project follows a simple versioned changelog format.

---

## [0.1.0] - 2026-07-06

### Added

- Initial alpha version of CARSBench
- Frequency-domain CARS/BCARS simulation workflow
- Eight benchmark domain presets:
  - `A_typical`
  - `B_high_res`
  - `C_low_res_noisy`
  - `D_calibration_shift`
  - `E_window_shift`
  - `F_nrb_family_shift`
  - `G_biochemical_source`
  - `H_biochemical_target`
- Raman-equivalent target generation
- Chunked dataset writing with `.npz` batches
- Per-sample metadata export
- Dataset manifest files
- Multi-seed generation workflow
- QC scripts for simulated spectra and domain-specific checks
- Basic visualization scripts
- README documentation for project purpose, domains, usage, and limitations
- Documentation pages for:
  - domain definitions
  - dataset format
  - reproducibility
- Basic API and reproducibility tests
- GitHub Actions CI for tests, formatting, and linting

### Notes

- Generated datasets are not included in the repository.
- This is an alpha-stage research and portfolio project.
- The project is not intended for clinical diagnosis or deployment in real healthcare settings.