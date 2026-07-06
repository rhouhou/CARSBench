import CARSBench as cb


def test_all_domains_generate_samples():
    domains = [
        "A_typical",
        "B_high_res",
        "C_low_res_noisy",
        "D_calibration_shift",
        "E_window_shift",
        "F_nrb_family_shift",
        "G_biochemical_source",
        "H_biochemical_target",
    ]

    for domain_name in domains:
        batch = cb.generate_dataset(
            num_samples=2,
            domain_name=domain_name,
            seed=42,
        )

        assert len(batch.samples) == 2

        for sample in batch.samples:
            assert sample.domain_name == domain_name
            assert sample.axis is not None
            assert sample.spectrum is not None
            assert sample.raman_target is not None

            assert sample.axis.shape == sample.spectrum.shape
            assert sample.raman_target.shape == sample.spectrum.shape
