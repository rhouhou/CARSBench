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

        # No explicit ratio constraint by default.
        # We still record the realized ratio for every sample.
        "target_rnr_ratio": None,
    },

    "nrb": {
        "family": {
            "dist": "choice",
            "values": [
                "flat",
                "poly",
            ],
        },

        "alpha": {
            "dist": "log_uniform",
            "low": 0.8,
            "high": 1.5,
        },
    },

    "instrument": {
        "psf_fwhm": {
            "dist": "uniform",
            "low": 10.0,
            "high": 13.0,
        },

        "envelope_family": {
            "dist": "choice",
            "values": [
                "hybrid",
                "poly",
                "gaussian",
            ],
        },
    },

    # Optical background is kept disabled in the
    # reference configuration until justified experimentally.
    "optical_background": {
        "family": "none",
        "fraction": 0.0,
    },

    # This now represents electronic baseline / drift.
    "baseline": {
        "family": "poly",

        "poly_std": {
            "dist": "uniform",
            "low": 0.008,
            "high": 0.015,
        },

        "correlated_std": {
            "dist": "uniform",
            "low": 0.002,
            "high": 0.005,
        },

        "correlated_knots": {
            "dist": "choice",
            "values": [
                8,
                10,
                12,
            ],
        },
    },

    "noise": {
        "shot_scale": {
            "dist": "log_uniform",
            "low": 1e4,
            "high": 3e5,
        },

        "read_sigma": {
            "dist": "uniform",
            "low": 0.5,
            "high": 5.0,
        },

        "spike_prob": {
            "dist": "uniform",
            "low": 0.0,
            "high": 5e-4,
        },

        "spike_min": None,
        "spike_max": None,
    },

    "detector": {
        "min_value": 0.0,
        "max_value": None,
        "bit_depth": None,
        "full_scale": None,
    },

    # Reference domain is correctly calibrated.
    "calibration": {
        "shift_cm1": 0.0,
        "warp_cm1": 0.0,
    },
}


def get_base_defaults() -> dict:
    """
    Return a deep copy of the reference configuration.
    """
    return deepcopy(BASE_DEFAULTS)