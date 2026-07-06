import CARSBench as cb


def test_generate_dataset_basic():
    batch = cb.generate_dataset(
        num_samples=5,
        domain_name="A_typical",
        seed=42,
    )

    assert len(batch.samples) == 5

    sample = batch.samples[0]

    assert sample.axis is not None
    assert sample.spectrum is not None
    assert sample.raman_target is not None

    assert sample.axis.shape == sample.spectrum.shape
    assert sample.raman_target.shape == sample.spectrum.shape