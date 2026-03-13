from __future__ import annotations
from copy import deepcopy

BASE_DEFAULTS = {

    "axis": {
        "window_mode": "full",
        "nu_min": 400.0,
        "nu_max": 3200.0,
        "num_points": 1024,
    },

    "resonant": {
        "mode": "component",
        "max_components": 3,
        "allowed_components": None,
    },

    "nrb": {
        "family": "poly",
        "alpha": 1.0,
        "phase_model": "linear",
        "phase_total_change": 0.0,
        "phase_offset": 0.0,
    },

    "instrument": {
        "psf_fwhm": 10.0,
        "envelope_family": "hybrid",
    },

    "baseline": {
        "family": "poly",
        "poly_std": 0.01,
        "correlated_std": 0.003,
        "correlated_knots": 10,
    },

    "noise": {
        "shot_scale": 1e5,
        "read_sigma": 2.0,
        "spike_prob": 0.0,
        "spike_min": None,
        "spike_max": None,
    },

    "detector": {
        "min_value": 0.0,
        "max_value": None,
        "bit_depth": None,
        "full_scale": None,
    },

    "calibration": {
        "shift_cm1": 0.0,
        "warp_cm1": 0.0,
    },
}


def get_base_defaults() -> dict:
    """
    Return a deep copy of default configuration.
    """
    return deepcopy(BASE_DEFAULTS)