from .base import DomainConfig, DomainSpec

from .presets import (
    build_default_registry,
    make_domain_a_typical,
    make_domain_b_high_res,
    make_domain_c_low_res,
    make_domain_d_high_noise,
    make_domain_e_calibration_shift,
    make_domain_f_window_shift,
    make_domain_g_nrb_family_shift,
    make_domain_h_biochemical_source,
    make_domain_i_biochemical_target,
)

from .registry import DomainRegistry

from .samplers import (
    DomainSampler,
    merge_nested_dicts,
)