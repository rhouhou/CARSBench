from __future__ import annotations

from .base import DomainConfig
from .registry import DomainRegistry


def _base_domain_tags(*extra: str) -> tuple[str, ...]:
    """
    Common tags used by all CARSBench benchmark domains.
    """
    return (
        "benchmark",
        "bcars",
        *extra,
    )


# ============================================================
# A: Reference domain
# ============================================================


def make_domain_a_typical() -> DomainConfig:
    """
    Reference acquisition domain.

    This domain does not introduce any additional shift relative to
    BASE_DEFAULTS. It defines the source/reference distribution used
    for the controlled acquisition-shift experiments.
    """
    return DomainConfig(
        name="A_typical",
        description=(
            "Reference BCARS acquisition domain using the baseline "
            "CARSBench acquisition and chemistry distributions."
        ),
        tags=_base_domain_tags(
            "reference",
            "typical",
        ),
        overrides={},
    )


# ============================================================
# B: Higher spectral resolution
# ============================================================


def make_domain_b_high_res() -> DomainConfig:
    """
    Higher-resolution acquisition domain.

    Only the spectrometer PSF differs from the reference domain.
    Noise, calibration, NRB distribution, chemistry, and other
    acquisition parameters remain inherited from BASE_DEFAULTS.
    """
    return DomainConfig(
        name="B_high_res",
        description=(
            "Higher spectral resolution under otherwise reference "
            "BCARS acquisition conditions."
        ),
        tags=_base_domain_tags(
            "resolution_shift",
            "high_res",
        ),
        overrides={
            "instrument": {
                "psf_fwhm": {
                    "dist": "uniform",
                    "low": 7.0,
                    "high": 9.0,
                },
            },
        },
    )


# ============================================================
# C: Lower spectral resolution
# ============================================================


def make_domain_c_low_res() -> DomainConfig:
    """
    Lower-resolution acquisition domain.

    Only the spectrometer PSF differs from the reference domain.
    """
    return DomainConfig(
        name="C_low_res",
        description=(
            "Lower spectral resolution under otherwise reference "
            "BCARS acquisition conditions."
        ),
        tags=_base_domain_tags(
            "resolution_shift",
            "low_res",
        ),
        overrides={
            "instrument": {
                "psf_fwhm": {
                    "dist": "uniform",
                    "low": 16.0,
                    "high": 22.0,
                },
            },
        },
    )


# ============================================================
# D: Noise shift
# ============================================================


def make_domain_d_high_noise() -> DomainConfig:
    """
    Reduced signal-quality domain.

    Resolution, calibration, NRB family, chemistry, and spectral
    coverage remain inherited from the reference domain.

    Only the stochastic detector/noise characteristics are changed.
    """
    return DomainConfig(
        name="D_high_noise",
        description=(
            "Higher-noise BCARS acquisition under otherwise "
            "reference acquisition conditions."
        ),
        tags=_base_domain_tags(
            "noise_shift",
            "high_noise",
        ),
        overrides={
            "noise": {
                "shot_scale": {
                    "dist": "log_uniform",
                    "low": 1e3,
                    "high": 5e4,
                },
                "read_sigma": {
                    "dist": "uniform",
                    "low": 3.0,
                    "high": 10.0,
                },
                "spike_prob": {
                    "dist": "uniform",
                    "low": 5e-4,
                    "high": 2e-3,
                },
            },
        },
    )


# ============================================================
# E: Calibration shift
# ============================================================


def make_domain_e_calibration_shift() -> DomainConfig:
    """
    Spectral-calibration mismatch domain.

    The underlying molecular spectrum and acquisition physics are
    generated on the nominal physical axis. The calibration mapping
    then introduces offset and smooth axis warping.
    """
    return DomainConfig(
        name="E_calibration_shift",
        description=(
            "Spectral calibration mismatch produced by wavenumber "
            "offset and smooth axis warping under otherwise reference "
            "acquisition conditions."
        ),
        tags=_base_domain_tags(
            "calibration_shift",
        ),
        overrides={
            "calibration": {
                "shift_cm1": {
                    "dist": "uniform",
                    "low": -10.0,
                    "high": 10.0,
                },
                "warp_cm1": {
                    "dist": "uniform",
                    "low": -15.0,
                    "high": 15.0,
                },
            },
        },
    )


# ============================================================
# F: Spectral-window shift
# ============================================================


def make_domain_f_window_shift() -> DomainConfig:
    """
    Spectral-coverage shift.

    This domain deliberately changes the measured Raman-shift window.

    NOTE
    ----
    The numerical window definitions and final ML representation
    should not be considered frozen yet. In the final benchmark,
    this domain will likely require a canonical spectral grid and
    explicit missing-region mask or another coordinate-aware
    representation.
    """
    return DomainConfig(
        name="F_window_shift",
        description=(
            "Spectral-coverage shift including fingerprint-only, "
            "CH-only, and partial spectral windows."
        ),
        tags=_base_domain_tags(
            "window_shift",
            "experimental",
        ),
        overrides={
            "axis": {
                "window_mode": {
                    "dist": "categorical",
                    "values": [
                        "fingerprint",
                        "ch",
                        "partial_fingerprint",
                        "partial_ch",
                    ],
                    "p": [
                        0.35,
                        0.35,
                        0.15,
                        0.15,
                    ],
                },
                "nu_min": {
                    "dist": "uniform",
                    "low": 500.0,
                    "high": 1200.0,
                },
                "nu_max": {
                    "dist": "uniform",
                    "low": 1800.0,
                    "high": 3100.0,
                },
                "num_points": 1024,
            },
        },
    )


# ============================================================
# G: NRB-family shift
# ============================================================


def make_domain_g_nrb_family_shift() -> DomainConfig:
    """
    Non-resonant-background shape shift.

    The reference distribution contains flat/poly NRB families.
    This target domain uses an unseen exponential-tilt family.

    NRB strength is intentionally inherited from BASE_DEFAULTS so
    that the shift isolates NRB shape rather than simultaneously
    changing shape and magnitude.
    """
    return DomainConfig(
        name="G_nrb_family_shift",
        description=(
            "Unseen non-resonant-background functional family under "
            "otherwise reference BCARS acquisition conditions."
        ),
        tags=_base_domain_tags(
            "nrb_shift",
            "nrb_family_shift",
        ),
        overrides={
            "nrb": {
                "family": "exp_tilt",
            },
        },
    )


# ============================================================
# H: Biochemical source domain
# ============================================================


def make_domain_h_biochemical_source() -> DomainConfig:
    """
    Source distribution for the chemistry-generalization task.

    The source chemistry is dominated by lipid/protein prototype
    mixtures while acquisition conditions remain identical to the
    reference domain.
    """
    return DomainConfig(
        name="H_biochemical_source",
        description=(
            "Chemistry-shift source domain dominated by lipid and "
            "protein prototype mixtures under reference BCARS "
            "acquisition conditions."
        ),
        tags=_base_domain_tags(
            "chemistry_shift",
            "source",
        ),
        overrides={
            "resonant": {
                "mode": "component",
                "max_components": 3,
                "allowed_components": [
                    "lipid",
                    "protein",
                ],
                "minor_background_max_peaks": 0,
                "component_weight_concentration": 3.0,
            },
        },
    )


# ============================================================
# I: Biochemical target domain
# ============================================================


def make_domain_i_biochemical_target() -> DomainConfig:
    """
    Target distribution for the chemistry-generalization task.

    The target chemistry is dominated by nucleic-acid/aromatic
    prototype mixtures while acquisition conditions remain identical
    to the chemistry-source domain.
    """
    return DomainConfig(
        name="I_biochemical_target",
        description=(
            "Chemistry-shift target domain enriched in nucleic-acid "
            "and aromatic prototype mixtures under reference BCARS "
            "acquisition conditions."
        ),
        tags=_base_domain_tags(
            "chemistry_shift",
            "target",
        ),
        overrides={
            "resonant": {
                "mode": "component",
                "max_components": 3,
                "allowed_components": [
                    "nucleic_acid",
                    "aromatic",
                ],
                "minor_background_max_peaks": 0,
                "component_weight_concentration": 3.0,
            },
        },
    )


# ============================================================
# Registry
# ============================================================


def build_default_registry() -> DomainRegistry:
    """
    Construct the default CARSBench domain registry.
    """
    registry = DomainRegistry()

    registry.register(
        make_domain_a_typical()
    )

    registry.register(
        make_domain_b_high_res()
    )

    registry.register(
        make_domain_c_low_res()
    )

    registry.register(
        make_domain_d_high_noise()
    )

    registry.register(
        make_domain_e_calibration_shift()
    )

    registry.register(
        make_domain_f_window_shift()
    )

    registry.register(
        make_domain_g_nrb_family_shift()
    )

    registry.register(
        make_domain_h_biochemical_source()
    )

    registry.register(
        make_domain_i_biochemical_target()
    )

    return registry