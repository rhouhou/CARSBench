import numpy as np

from CARSBench.api import generate_dataset
from CARSBench.datasets.reader import DatasetReader
from CARSBench.datasets.writer import DatasetWriter


def test_sample_and_batch_npz_roundtrip(tmp_path):
    batch = generate_dataset(
        num_samples=3,
        domain_name="A_typical",
        seed=42,
    )

    writer = DatasetWriter(tmp_path)
    reader = DatasetReader(tmp_path)

    sample = batch.samples[0]

    sample_path = writer.write_sample_npz(
        sample,
        relative_dir="samples",
    )

    loaded_sample = reader.read_sample_npz(sample_path)

    assert loaded_sample.sample_id == sample.sample_id
    assert loaded_sample.domain_name == sample.domain_name
    assert loaded_sample.axis.shape == sample.axis.shape
    assert loaded_sample.spectrum.shape == sample.spectrum.shape
    assert loaded_sample.raman_target.shape == sample.raman_target.shape

    np.testing.assert_allclose(loaded_sample.axis, sample.axis)
    np.testing.assert_allclose(loaded_sample.spectrum, sample.spectrum)
    np.testing.assert_allclose(loaded_sample.raman_target, sample.raman_target)

    writer.write_batch_npz(
        batch,
        filename="batch_test.npz",
        relative_dir="batches",
    )

    loaded_batch = reader.read_batch_npz(
        tmp_path / "batches" / "batch_test.npz",
    )

    assert len(loaded_batch.samples) == len(batch.samples)

    for original_sample, loaded_batch_sample in zip(
        batch.samples,
        loaded_batch.samples,
    ):
        assert loaded_batch_sample.domain_name == original_sample.domain_name
        np.testing.assert_allclose(
            loaded_batch_sample.spectrum,
            original_sample.spectrum,
        )
        np.testing.assert_allclose(
            loaded_batch_sample.raman_target,
            original_sample.raman_target,
        )
