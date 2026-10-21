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