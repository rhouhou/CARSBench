from __future__ import annotations

from pathlib import Path

from CARSBench.datasets.reader import DatasetReader
from CARSBench.datasets.schema import SampleBatch, SpectrumSample


def read_sample(
    root: str | Path,
    path: str | Path,
) -> SpectrumSample:
    """
    Convenience wrapper for reading one sample NPZ.
    """
    return DatasetReader(root).read_sample_npz(path)


def read_batch(
    root: str | Path,
    path: str | Path,
) -> SampleBatch:
    """
    Convenience wrapper for reading one stacked batch NPZ.
    """
    return DatasetReader(root).read_batch_npz(path)


def read_samples_dir(
    root: str | Path,
    relative_dir: str | Path = "samples",
    limit: int | None = None,
) -> SampleBatch:
    """
    Convenience wrapper for reading many single-sample NPZ files.
    """
    return DatasetReader(root).read_samples_npz(relative_dir=relative_dir, limit=limit)


def read_metadata(
    root: str | Path,
    path: str | Path,
):
    """
    Convenience wrapper for reading metadata JSONL.
    """
    return DatasetReader(root).read_metadata_jsonl(path)