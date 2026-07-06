import CARSBench as cb


def test_expected_domains_are_available():
    domains = set(cb.list_domains())

    expected_domains = {
        "A_typical",
        "B_high_res",
        "C_low_res_noisy",
        "D_calibration_shift",
        "E_window_shift",
        "F_nrb_family_shift",
        "G_biochemical_source",
        "H_biochemical_target",
    }

    assert expected_domains.issubset(domains)
