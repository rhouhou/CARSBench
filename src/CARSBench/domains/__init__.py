from .base import DomainConfig, DomainSpec, ParameterBundle
from .presets import (
    build_default_registry,
    make_domain_a_typical,
    make_domain_b_high_res,
    make_domain_c_low_res_noisy,
    make_domain_d_calibration_shift,
    make_domain_e_window_shift,
    make_domain_f_nrb_family_shift,
)
from .registry import DomainRegistry
from .samplers import DomainSampler, merge_nested_dicts

__all__ = [
    "ParameterBundle",
    "DomainConfig",
    "DomainSpec",
    "DomainRegistry",
    "DomainSampler",
    "merge_nested_dicts",
    "build_default_registry",
    "make_domain_a_typical",
    "make_domain_b_high_res",
    "make_domain_c_low_res_noisy",
    "make_domain_d_calibration_shift",
    "make_domain_e_window_shift",
    "make_domain_f_nrb_family_shift",
]