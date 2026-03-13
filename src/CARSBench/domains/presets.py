from __future__ import annotations

from .base import DomainConfig
from .registry import DomainRegistry


def _base_domain_tags(*extra: str) -> tuple[str, ...]:
    return ("benchmark", "bcars", *extra)


def make_domain_a_typical() -> DomainConfig:
    return DomainConfig(
        name="A_typical",
        description="Typical bCARS acquisition with moderate resolution, moderate NRB, and moderate noise.",
        tags=_base_domain_tags("typical"),
        overrides={
            "axis": {
                "window_mode": {"dist": "choice", "values": ["full", "wide"]},
                "nu_min": {"dist": "uniform", "low": 350.0, "high": 600.0},
                "nu_max": {"dist": "uniform", "low": 3000.0, "high": 3300.0},
                "num_points": {"dist": "choice", "values": [1024, 2048]},
            },
            "instrument": {
                "psf_fwhm": {"dist": "uniform", "low": 8.0, "high": 14.0},
                "envelope_family": {"dist": "choice", "values": ["hybrid", "poly", "gaussian"]},
            },
            "noise": {
                "shot_scale": {"dist": "log_uniform", "low": 1e4, "high": 3e5},
                "read_sigma": {"dist": "uniform", "low": 0.5, "high": 5.0},
                "spike_prob": {"dist": "uniform", "low": 0.0, "high": 5e-4},
            },
            "calibration": {
                "shift_cm1": {"dist": "uniform", "low": -5.0, "high": 5.0},
                "warp_cm1": {"dist": "uniform", "low": -10.0, "high": 10.0},
            },
            "nrb": {
                "family": {"dist": "choice", "values": ["flat", "poly", "exp_tilt"]},
                "alpha": {"dist": "log_uniform", "low": 0.8, "high": 1.5},
                "phase_model": {"dist": "choice", "values": ["linear", "quadratic"]},
                "phase_total_change": {
                    "dist": "uniform",
                    "low": -1.0,
                    "high": 1.0,
                },
            },
        },
    )


def make_domain_b_high_res() -> DomainConfig:
    return DomainConfig(
        name="B_high_res",
        description="Careful acquisition with higher spectral resolution, stronger counts, and lower artifact rate.",
        tags=_base_domain_tags("high_res"),
        overrides={
            "instrument": {
                "psf_fwhm": {"dist": "uniform", "low": 5.0, "high": 10.0},
            },
            "noise": {
                "shot_scale": {"dist": "log_uniform", "low": 5e4, "high": 1e6},
                "read_sigma": {"dist": "uniform", "low": 0.25, "high": 2.0},
                "spike_prob": {"dist": "uniform", "low": 0.0, "high": 1e-4},
            },
        },
    )


def make_domain_c_low_res_noisy() -> DomainConfig:
    return DomainConfig(
        name="C_low_res_noisy",
        description="Lower resolution, lower counts, and stronger detector noise/artifacts.",
        tags=_base_domain_tags("low_res", "noisy"),
        overrides={
            "instrument": {
                "psf_fwhm": {"dist": "uniform", "low": 14.0, "high": 25.0},
            },
            "noise": {
                "shot_scale": {"dist": "log_uniform", "low": 1e3, "high": 5e4},
                "read_sigma": {"dist": "uniform", "low": 3.0, "high": 10.0},
                "spike_prob": {"dist": "uniform", "low": 5e-4, "high": 2e-3},
            },
            "baseline": {
                "family": "poly+ripple",
                "poly_std": {"dist": "uniform", "low": 0.008, "high": 0.02},
                "correlated_std": {"dist": "uniform", "low": 0.002, "high": 0.008},
                "correlated_knots": {"dist": "choice", "values": [8, 10, 12]},
            },
        },
    )


def make_domain_d_calibration_shift() -> DomainConfig:
    return DomainConfig(
        name="D_calibration_shift",
        description="Domain with stronger axis offset and wavenumber warping to simulate calibration mismatch.",
        tags=_base_domain_tags("calibration_shift"),
        overrides={
            "calibration": {
                "shift_cm1": {"dist": "uniform", "low": -10.0, "high": 10.0},
                "warp_cm1": {"dist": "uniform", "low": -15.0, "high": 15.0},
            },
        },
    )


def make_domain_e_window_shift() -> DomainConfig:
    return DomainConfig(
        name="E_window_shift",
        description="Different spectral windows and lower point counts, including fingerprint-only and CH-only settings.",
        tags=_base_domain_tags("window_shift"),
        overrides={
            "axis": {
                "window_mode": {
                    "dist": "categorical",
                    "values": ["fingerprint", "ch", "partial_fingerprint", "partial_ch"],
                    "p": [0.35, 0.35, 0.15, 0.15],
                },
                "num_points": {"dist": "choice", "values": [512, 1024, 2048]},
            },
        },
    )


def make_domain_f_nrb_family_shift() -> DomainConfig:
    return DomainConfig(
        name="F_nrb_family_shift",
        description="NRB-family shift with stronger exponential tilt and larger phase variation.",
        tags=_base_domain_tags("nrb_shift"),
        overrides={
            "nrb": {
                "family": {"dist": "choice", "values": ["exp_tilt", "poly"]},
                "alpha": {"dist": "log_uniform", "low": 1.0, "high": 5.0},
                "phase_model": {"dist": "choice", "values": ["linear", "quadratic"]},
                "phase_total_change": {
                    "dist": "uniform",
                    "low": -3.14159265359,
                    "high": 3.14159265359,
                },
            },
        },
    )

def make_domain_g_biochemical_source() -> DomainConfig:
    return DomainConfig(
        name="G_biochemical_source",
        description=(
            "Chemistry-shift source domain dominated by lipid/protein prototype mixtures "
            "under otherwise typical BCARS acquisition conditions."
        ),
        tags=_base_domain_tags("chemistry_shift", "source"),
        overrides={
            "resonant": {
                "mode": "component",
                "max_components": 3,
                "allowed_components": ["lipid", "protein"],
            },
            "nrb": {
                "family": "poly",
                "alpha": {"dist": "log_uniform", "low": 0.8, "high": 1.5},
                "phase_model": {"dist": "choice", "values": ["linear", "quadratic"]},
                "phase_total_change": {"dist": "uniform", "low": -1.0, "high": 1.0},
            },
            "instrument": {
                "psf_fwhm": {"dist": "uniform", "low": 8.0, "high": 14.0},
                "envelope_family": {"dist": "choice", "values": ["hybrid", "poly", "gaussian"]},
            },
            "noise": {
                "shot_scale": {"dist": "log_uniform", "low": 1e4, "high": 3e5},
                "read_sigma": {"dist": "uniform", "low": 0.5, "high": 5.0},
                "spike_prob": {"dist": "uniform", "low": 0.0, "high": 5e-4},
            },
            "calibration": {
                "shift_cm1": {"dist": "uniform", "low": -5.0, "high": 5.0},
                "warp_cm1": {"dist": "uniform", "low": -10.0, "high": 10.0},
            },
        },
    )


def make_domain_h_biochemical_target() -> DomainConfig:
    return DomainConfig(
        name="H_biochemical_target",
        description=(
            "Chemistry-shift target domain enriched in nucleic-acid/aromatic prototype mixtures "
            "under otherwise typical BCARS acquisition conditions."
        ),
        tags=_base_domain_tags("chemistry_shift", "target"),
        overrides={
            "resonant": {
                "mode": "component",
                "max_components": 3,
                "allowed_components": ["nucleic_acid", "aromatic"],
            },
            "nrb": {
                "family": "poly",
                "alpha": {"dist": "log_uniform", "low": 0.8, "high": 1.5},
                "phase_model": {"dist": "choice", "values": ["linear", "quadratic"]},
                "phase_total_change": {"dist": "uniform", "low": -1.0, "high": 1.0},
            },
            "instrument": {
                "psf_fwhm": {"dist": "uniform", "low": 8.0, "high": 14.0},
                "envelope_family": {"dist": "choice", "values": ["hybrid", "poly", "gaussian"]},
            },
            "noise": {
                "shot_scale": {"dist": "log_uniform", "low": 1e4, "high": 3e5},
                "read_sigma": {"dist": "uniform", "low": 0.5, "high": 5.0},
                "spike_prob": {"dist": "uniform", "low": 0.0, "high": 5e-4},
            },
            "calibration": {
                "shift_cm1": {"dist": "uniform", "low": -5.0, "high": 5.0},
                "warp_cm1": {"dist": "uniform", "low": -10.0, "high": 10.0},
            },
        },
    )

def build_default_registry() -> DomainRegistry:
    registry = DomainRegistry()

    registry.register(make_domain_a_typical())
    registry.register(make_domain_b_high_res())
    registry.register(make_domain_c_low_res_noisy())
    registry.register(make_domain_d_calibration_shift())
    registry.register(make_domain_e_window_shift())
    registry.register(make_domain_f_nrb_family_shift())
    registry.register(make_domain_g_biochemical_source())
    registry.register(make_domain_h_biochemical_target())

    return registry