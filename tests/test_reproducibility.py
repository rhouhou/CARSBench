import numpy as np

import CARSBench as cb


def test_same_seed_is_reproducible():
    batch_1 = cb.generate_dataset(
        num_samples=3,
        domain_name="A_typical",
        seed=42,
    )

    batch_2 = cb.generate_dataset(
        num_samples=3,
        domain_name="A_typical",
        seed=42,
    )

    for sample_1, sample_2 in zip(batch_1.samples, batch_2.samples):
        np.testing.assert_allclose(sample_1.spectrum, sample_2.spectrum)
        np.testing.assert_allclose(sample_1.raman_target, sample_2.raman_target)