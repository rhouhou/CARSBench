from CARSBench.configs.defaults import get_base_defaults
from CARSBench.domains import build_default_registry


def test_resolution_domains_are_non_overlapping():
    defaults = get_base_defaults()
    registry = build_default_registry()

    reference = defaults[
        "instrument"
    ]["psf_fwhm"]

    high_res = registry.get(
        "B_high_res"
    ).overrides[
        "instrument"
    ]["psf_fwhm"]

    low_res = registry.get(
        "C_low_res"
    ).overrides[
        "instrument"
    ]["psf_fwhm"]

    assert (
        high_res["high"]
        < reference["low"]
    )

    assert (
        reference["high"]
        < low_res["low"]
    )


def test_reference_spectral_axis():
    defaults = get_base_defaults()

    axis = defaults["axis"]

    assert axis["window_mode"] == "full"
    assert axis["nu_min"] == 400.0
    assert axis["nu_max"] == 3200.0
    assert axis["num_points"] == 1024

def test_high_noise_domain_uses_lower_photon_counts():
    defaults = get_base_defaults()
    registry = build_default_registry()

    reference_noise = defaults["noise"]

    high_noise = registry.get(
        "D_high_noise"
    ).overrides["noise"]

    reference_scale = reference_noise[
        "photoelectrons_per_intensity_unit"
    ]

    high_noise_scale = high_noise[
        "photoelectrons_per_intensity_unit"
    ]

    assert (
        high_noise_scale["high"]
        <= reference_scale["low"]
    )


def test_high_noise_domain_inherits_reference_read_noise():
    defaults = get_base_defaults()
    registry = build_default_registry()

    reference_read_noise = defaults[
        "noise"
    ]["read_noise_electrons"]

    high_noise_overrides = registry.get(
        "D_high_noise"
    ).overrides["noise"]

    assert "read_noise_electrons" not in high_noise_overrides

    assert reference_read_noise == 0.7

def test_reference_has_no_calibration_error():
    defaults = get_base_defaults()

    calibration = defaults[
        "calibration"
    ]

    assert calibration["shift_cm1"] == 0.0
    assert calibration["warp_cm1"] == 0.0


def test_calibration_shift_domain_changes_only_calibration():
    registry = build_default_registry()

    calibration = registry.get(
        "E_calibration_shift"
    ).overrides["calibration"]

    assert calibration[
        "shift_cm1"
    ]["low"] == -3.0

    assert calibration[
        "shift_cm1"
    ]["high"] == 3.0

    assert calibration[
        "warp_cm1"
    ]["low"] == -4.0

    assert calibration[
        "warp_cm1"
    ]["high"] == 4.0

def test_system_response_shift_does_not_modify_nrb():
    registry = build_default_registry()

    domain = registry.get(
        "G_system_response_shift"
    )

    assert "nrb" not in domain.overrides

    assert "instrument" in domain.overrides

    assert (
        domain.overrides[
            "instrument"
        ]["envelope_family"]
        == "poly"
    )