import CARSBench as cb


def test_expected_domains_are_available():
    domains = set(cb.list_domains())

    expected_domains = {
        "A_typical",
        "B_high_res",
        "C_low_res",
        "D_high_noise",
        "E_calibration_shift",
        "F_window_shift",
        "G_system_response_shift",
        "H_biochemical_source",
        "I_biochemical_target",
    }

    assert expected_domains.issubset(domains)
