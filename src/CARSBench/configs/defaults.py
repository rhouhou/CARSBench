from __future__ import annotations

from copy import deepcopy


# ============================================================
# Reference resonant-to-NRB strength distribution
# ============================================================
#
# Derived from the pooled A_typical stability audit:
#
#   5 seeds
#   5000 samples per seed
#   25,000 samples total
#
# Empirical pooled reference distribution:
#
#   min     = 0.01713
#   p01     = 0.03933
#   p05     = 0.06067
#   median  = 0.18281
#   mean    = 0.23419
#   p95     = 0.57849
#   p99     = 0.93828
#   max     = 2.38537
#
# Log-space fit:
#
#   mu      = -1.6917887382307994
#   sigma   =  0.6886309738595188
#
# CARSBench v1 uses this fitted distribution with conservative
# numerical guardrails.
#
# The guardrails are benchmark controls, not claimed physical
# limits of chi_R / chi_NR in experimental CARS or BCARS.
#
REFERENCE_RNR_LOG_MEAN = -1.6917887382307994
REFERENCE_RNR_LOG_SIGMA = 0.6886309738595188

REFERENCE_RNR_MIN = 0.02
REFERENCE_RNR_MAX = 2.5


BASE_DEFAULTS = {
    # ========================================================
    # Physical spectral axis
    # ========================================================
    "axis": {
        "window_mode": "full",
        "nu_min": 400.0,
        "nu_max": 3200.0,
        "num_points": 1024,
    },

    # ========================================================
    # Observation window
    # ========================================================
    #
    # Reference acquisition observes the complete physical axis.
    # Domain F overrides this section without changing the latent
    # physical spectrum.
    #
    "observation": {
        "window_mode": "full",
    },

    # ========================================================
    # Resonant susceptibility
    # ========================================================
    "resonant": {
        "mode": "component",
        "max_components": 3,
        "allowed_components": None,

        # Global resonant-to-NRB strength is sampled independently
        # from spectral chemistry.
        #
        # This keeps strength matched across acquisition and
        # biochemical domains.
        "target_rnr_ratio": {
            "dist": "truncated_lognormal",
            "mean": REFERENCE_RNR_LOG_MEAN,
            "sigma": REFERENCE_RNR_LOG_SIGMA,
            "low": REFERENCE_RNR_MIN,
            "high": REFERENCE_RNR_MAX,
        },
    },

    # ========================================================
    # Intrinsic non-resonant susceptibility
    # ========================================================
    "nrb": {
        # Dimensionless simulator normalization.
        #
        # Relative resonant/non-resonant strength is controlled
        # explicitly through target_rnr_ratio.
        "amplitude": 1.0,
    },

    # ========================================================
    # Instrument / effective spectral response
    # ========================================================
    "instrument": {
        "psf_fwhm": {
            "dist": "uniform",
            "low": 10.0,
            "high": 13.0,
        },

        "envelope_family": "hybrid",

        # Disabled in the reference configuration until an
        # experimentally justified ripple distribution is available.
        "ripple_amplitude": 0.0,
    },

    # ========================================================
    # Optical background
    # ========================================================
    "optical_background": {
        "family": "none",
        "fraction": 0.0,
    },

    # ========================================================
    # Electronic baseline / detector drift
    # ========================================================
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

    # ========================================================
    # Photon / read noise
    # ========================================================
    "noise": {
        "photoelectrons_per_intensity_unit": {
            "dist": "log_uniform",
            "low": 700.0,
            "high": 3000.0,
        },

        "read_noise_electrons": 0.7,
    },

    # ========================================================
    # Detector
    # ========================================================
    "detector": {
        "min_value": 0.0,
        "max_value": None,
        "bit_depth": None,
        "full_scale": None,
    },

    # ========================================================
    # Calibration
    # ========================================================
    "calibration": {
        "shift_cm1": 0.0,
        "warp_cm1": 0.0,
    },
}


def get_base_defaults() -> dict:
    """
    Return a deep copy of the reference CARSBench configuration.
    """
    return deepcopy(
        BASE_DEFAULTS
    )