from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .base import DomainConfig
from .registry import DomainRegistry


# ============================================================
# Reusable frozen shift definitions
# ============================================================


LOW_RES_SHIFT = {
    "instrument": {
        "psf_fwhm": {
            "dist": "uniform",
            "low": 16.0,
            "high": 22.0,
        },
    },
}


HIGH_NOISE_SHIFT = {
    "noise": {
        "photoelectrons_per_intensity_unit": {
            "dist": "log_uniform",
            "low": 150.0,
            "high": 700.0,
        },
    },
}


CALIBRATION_SHIFT = {
    "calibration": {
        "shift_cm1": {
            "dist": "uniform",
            "low": -3.0,
            "high": 3.0,
        },
        "warp_cm1": {
            "dist": "uniform",
            "low": -4.0,
            "high": 4.0,
        },
    },
}


WINDOW_SHIFT = {
    "observation": {
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
    },
}


SYSTEM_RESPONSE_SHIFT = {
    "instrument": {
        "envelope_family": "poly",
    },
}


CHEMISTRY_SOURCE = {
    "resonant": {
        "mode": "component",
        "min_components": 2,
        "max_components": 3,
        "allowed_components": [
            "lipid",
            "protein",
            "nucleic_acid",
        ],
        "required_components": [],
        "minor_background_max_peaks": 0,
        "component_weight_concentration": 3.0,
        "min_component_weight": 0.10,
    },
}


CHEMISTRY_TARGET = {
    "resonant": {
        "mode": "component",
        "min_components": 2,
        "max_components": 3,
        "allowed_components": [
            "lipid",
            "protein",
            "nucleic_acid",
            "glycogen",
        ],
        "required_components": [
            "glycogen",
        ],
        "minor_background_max_peaks": 0,
        "component_weight_concentration": 3.0,
        "min_component_weight": 0.10,
    },
}


def _merge_overrides(
    *parts: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge frozen domain-shift definitions.

    Later parts overwrite earlier values only when the same leaf
    parameter is explicitly specified.
    """
    result: dict[str, Any] = {}

    for part in parts:
        for key, value in part.items():
            if (
                key in result
                and isinstance(
                    result[key],
                    dict,
                )
                and isinstance(
                    value,
                    Mapping,
                )
            ):
                result[key] = _merge_overrides(
                    result[key],
                    value,
                )
            else:
                result[key] = deepcopy(
                    value
                )

    return result


def _base_domain_tags(
    *extra: str,
) -> tuple[str, ...]:
    return (
        "benchmark",
        "bcars",
        *extra,
    )


# ============================================================
# A: reference
# ============================================================


def make_domain_a_typical() -> DomainConfig:
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
# B: high resolution
# ============================================================


def make_domain_b_high_res() -> DomainConfig:
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
# C: low resolution
# ============================================================


def make_domain_c_low_res() -> DomainConfig:
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
        overrides=_merge_overrides(
            LOW_RES_SHIFT
        ),
    )


# ============================================================
# D: high noise
# ============================================================


def make_domain_d_high_noise() -> DomainConfig:
    return DomainConfig(
        name="D_high_noise",
        description=(
            "Higher relative shot-noise domain produced by reduced "
            "detected photon counts under otherwise reference BCARS "
            "acquisition conditions."
        ),
        tags=_base_domain_tags(
            "noise_shift",
            "high_noise",
            "low_signal",
        ),
        overrides=_merge_overrides(
            HIGH_NOISE_SHIFT
        ),
    )


# ============================================================
# E: calibration
# ============================================================


def make_domain_e_calibration_shift() -> DomainConfig:
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
        overrides=_merge_overrides(
            CALIBRATION_SHIFT
        ),
    )


# ============================================================
# F: spectral coverage
# ============================================================


def make_domain_f_window_shift() -> DomainConfig:
    return DomainConfig(
        name="F_window_shift",
        description=(
            "Spectral-coverage OOD domain using fingerprint-only, "
            "CH-only, and partial observation windows on the common "
            "canonical Raman-shift grid."
        ),
        tags=_base_domain_tags(
            "window_shift",
            "coverage_shift",
            "observation_shift",
        ),
        overrides=_merge_overrides(
            WINDOW_SHIFT
        ),
    )


# ============================================================
# G: system response
# ============================================================


def make_domain_g_system_response_shift() -> DomainConfig:
    return DomainConfig(
        name="G_system_response_shift",
        description=(
            "OOD effective excitation/instrument spectral-response "
            "profile under otherwise reference BCARS conditions."
        ),
        tags=_base_domain_tags(
            "system_response_shift",
            "profile_shift",
        ),
        overrides=_merge_overrides(
            SYSTEM_RESPONSE_SHIFT
        ),
    )


# ============================================================
# H: chemistry source
# ============================================================


def make_domain_h_biochemical_source() -> DomainConfig:
    return DomainConfig(
        name="H_biochemical_source",
        description=(
            "Chemistry source domain containing mixed lipid, "
            "protein, and nucleic-acid Raman components."
        ),
        tags=_base_domain_tags(
            "chemistry_shift",
            "source",
            "known_chemistry",
        ),
        overrides=_merge_overrides(
            CHEMISTRY_SOURCE
        ),
    )


# ============================================================
# I: chemistry target
# ============================================================


def make_domain_i_biochemical_target() -> DomainConfig:
    return DomainConfig(
        name="I_biochemical_target",
        description=(
            "Chemistry OOD target containing held-out glycogen "
            "embedded within otherwise familiar lipid, protein, "
            "and/or nucleic-acid mixtures."
        ),
        tags=_base_domain_tags(
            "chemistry_shift",
            "target",
            "held_out_component",
        ),
        overrides=_merge_overrides(
            CHEMISTRY_TARGET
        ),
    )


# ============================================================
# J: compound acquisition OOD
# ============================================================


def make_domain_j_compound_acquisition() -> DomainConfig:
    """
    Compound acquisition shift.

    J combines four already validated individual acquisition shifts:

        low spectral resolution
        high photon noise
        calibration mismatch
        altered system response

    Spectral coverage remains complete so this condition tests
    degraded/shifted acquisition physics without removing spectral
    information.
    """
    return DomainConfig(
        name="J_compound_acquisition",
        description=(
            "Compound acquisition OOD domain combining lower spectral "
            "resolution, reduced photon counts, spectral-calibration "
            "mismatch, and altered effective system response."
        ),
        tags=_base_domain_tags(
            "compound_shift",
            "acquisition_shift",
            "physics_ood",
        ),
        overrides=_merge_overrides(
            LOW_RES_SHIFT,
            HIGH_NOISE_SHIFT,
            CALIBRATION_SHIFT,
            SYSTEM_RESPONSE_SHIFT,
        ),
    )


# ============================================================
# K: joint acquisition + chemistry OOD
# ============================================================


def make_domain_k_joint_ood() -> DomainConfig:
    """
    Joint acquisition-and-chemistry OOD target.

    K combines the complete J acquisition shift with the held-out
    glycogen chemistry used by I.

    This supports the joint benchmark:

        H_biochemical_source -> K_joint_ood

    where both acquisition physics and chemistry change.
    """
    return DomainConfig(
        name="K_joint_ood",
        description=(
            "Joint OOD target combining compound acquisition shifts "
            "with held-out glycogen chemistry embedded in familiar "
            "biochemical mixtures."
        ),
        tags=_base_domain_tags(
            "compound_shift",
            "joint_ood",
            "physics_ood",
            "chemistry_shift",
            "held_out_component",
        ),
        overrides=_merge_overrides(
            LOW_RES_SHIFT,
            HIGH_NOISE_SHIFT,
            CALIBRATION_SHIFT,
            SYSTEM_RESPONSE_SHIFT,
            CHEMISTRY_TARGET,
        ),
    )


# ============================================================
# Registry
# ============================================================


def build_default_registry() -> DomainRegistry:
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
        make_domain_g_system_response_shift()
    )

    registry.register(
        make_domain_h_biochemical_source()
    )

    registry.register(
        make_domain_i_biochemical_target()
    )

    registry.register(
        make_domain_j_compound_acquisition()
    )

    registry.register(
        make_domain_k_joint_ood()
    )

    return registry