from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np

from CARSBench.api import generate_dataset, list_domains
from CARSBench.datasets.reader import DatasetReader
from CARSBench.datasets.writer import DatasetWriter


TMP_DIR = Path("tmp_smoke_test")


def main() -> None:
    print("=== Smoke test ===")
    print("Domains:", list_domains())

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    batch = generate_dataset(
        num_samples=3,
        domain_name="A_typical",
        seed=42,
    )

    assert len(batch.samples) == 3

    sample = batch.samples[0]
    n = len(sample.axis)

    assert sample.spectrum.shape == (n,)
    assert sample.raman_target.shape == (n,)
    assert np.all(np.isfinite(sample.spectrum))
    assert np.all(np.isfinite(sample.raman_target))

    writer = DatasetWriter(TMP_DIR)
    reader = DatasetReader(TMP_DIR)

    sample_path = writer.write_sample_npz(sample, relative_dir="samples")
    loaded = reader.read_sample_npz(sample_path)

    assert loaded.sample_id == sample.sample_id
    assert loaded.domain_name == sample.domain_name
    assert loaded.spectrum.shape == sample.spectrum.shape

    writer.write_batch_npz(batch, filename="batch_test.npz", relative_dir="batches")
    batch_loaded = reader.read_batch_npz(TMP_DIR / "batches" / "batch_test.npz")
    assert len(batch_loaded.samples) == len(batch.samples)

    print("Smoke test passed.")


if __name__ == "__main__":
    main()