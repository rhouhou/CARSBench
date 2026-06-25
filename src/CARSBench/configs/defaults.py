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
        "family": {"dist": "choice", "values": ["flat", "poly"]},
        "alpha": {"dist": "log_uniform", "low": 0.8, "high": 1.5},
        "phase_model": {"dist": "choice", "values": ["linear", "quadratic"]},
        "phase_total_change": {"dist": "uniform", "low": -1.0, "high": 1.0},
        "phase_offset": 0.0,
    },

    "instrument": {
        "psf_fwhm": {"dist": "uniform", "low": 8.0, "high": 14.0},
        "envelope_family": {"dist": "choice", "values": ["hybrid", "poly", "gaussian"]},
    },

    "baseline": {
        "family": "poly",
        "poly_std": {"dist": "uniform", "low": 0.008, "high": 0.015},
        "correlated_std": {"dist": "uniform", "low": 0.002, "high": 0.005},
        "correlated_knots": {"dist": "choice", "values": [8, 10, 12]},
    },

    "noise": {
        "shot_scale": {"dist": "log_uniform", "low": 1e4, "high": 3e5},
        "read_sigma": {"dist": "uniform", "low": 0.5, "high": 5.0},
        "spike_prob": {"dist": "uniform", "low": 0.0, "high": 5e-4},
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
        "shift_cm1": {"dist": "uniform", "low": -5.0, "high": 5.0},
        "warp_cm1": {"dist": "uniform", "low": -10.0, "high": 10.0},
    },
}


def get_base_defaults() -> dict:
    """
    Return a deep copy of default configuration.
    """
    return deepcopy(BASE_DEFAULTS)