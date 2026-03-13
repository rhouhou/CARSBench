from __future__ import annotations

from pathlib import Path
from typing import Sequence

from CARSBench.datasets.schema import SampleBatch, SpectrumSample
from CARSBench.datasets.writer import DatasetWriter


def write_sample(
    root: str | Path,
    sample: SpectrumSample,
    relative_dir: str | Path = "samples",
    compress: bool = True,
):
    """
    Convenience wrapper for writing one sample NPZ.
    """
    return DatasetWriter(root).write_sample_npz(
        sample=sample,
        relative_dir=relative_dir,
        compress=compress,
    )


def write_samples(
    root: str | Path,
    samples: Sequence[SpectrumSample],
    relative_dir: str | Path = "samples",
    compress: bool = True,
):
    """
    Convenience wrapper for writing many sample NPZ files.
    """
    return DatasetWriter(root).write_samples_npz(
        samples=samples,
        relative_dir=relative_dir,
        compress=compress,
    )


def write_batch(
    root: str | Path,
    batch: SampleBatch,
    filename: str = "dataset_batch.npz",
    relative_dir: str | Path = "batches",
    compress: bool = True,
):
    """
    Convenience wrapper for writing one stacked batch NPZ.
    """
    return DatasetWriter(root).write_batch_npz(
        batch=batch,
        filename=filename,
        relative_dir=relative_dir,
        compress=compress,
    )


def write_dataset_bundle(
    root: str | Path,
    samples: Sequence[SpectrumSample],
    dataset_name: str,
    compress: bool = True,
    manifest_extra: dict | None = None,
):
    """
    Convenience wrapper for writing a full dataset bundle.
    """
    return DatasetWriter(root).write_dataset_bundle(
        samples=samples,
        dataset_name=dataset_name,
        compress=compress,
        manifest_extra=manifest_extra,
    )