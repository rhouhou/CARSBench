from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .schema import SampleBatch, SpectrumSample


class DatasetWriter:
    """
    Writer utilities for CARSBench datasets.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _ensure_dir(self, relative_dir: str | Path) -> Path:
        out_dir = self.root / relative_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def write_sample_npz(
        self,
        sample: SpectrumSample,
        relative_dir: str | Path = "samples",
        compress: bool = True,
    ) -> Path:
        out_dir = self._ensure_dir(relative_dir)
        path = out_dir / f"{sample.sample_id}.npz"

        arrays = sample.to_numpy_dict()
        arrays["metadata_json"] = np.array(
            json.dumps(sample.metadata.to_dict(), ensure_ascii=False),
            dtype=object,
        )

        if compress:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)

        return path

    def write_samples_npz(
        self,
        samples: Iterable[SpectrumSample],
        relative_dir: str | Path = "samples",
        compress: bool = True,
    ) -> list[Path]:
        paths: list[Path] = []

        for sample in samples:
            paths.append(
                self.write_sample_npz(
                    sample=sample,
                    relative_dir=relative_dir,
                    compress=compress,
                )
            )

        return paths

    def write_batch_npz(
        self,
        batch: SampleBatch,
        filename: str = "dataset_batch.npz",
        relative_dir: str | Path = "batches",
        compress: bool = True,
    ) -> Path:
        out_dir = self._ensure_dir(relative_dir)
        path = out_dir / filename

        stacked = batch.stack()

        stacked["metadata_json"] = np.array(
            [json.dumps(sample.metadata.to_dict(), ensure_ascii=False) for sample in batch.samples],
            dtype=object,
        )

        if compress:
            np.savez_compressed(path, **stacked)
        else:
            np.savez(path, **stacked)

        return path

    def write_metadata_jsonl(
        self,
        samples: Sequence[SpectrumSample],
        filename: str = "metadata.jsonl",
        relative_dir: str | Path = "metadata",
    ) -> Path:
        out_dir = self._ensure_dir(relative_dir)
        path = out_dir / filename

        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.metadata.to_dict(), ensure_ascii=False) + "\n")

        return path

    def write_manifest(
        self,
        samples: Sequence[SpectrumSample],
        filename: str = "manifest.json",
        relative_dir: str | Path = ".",
        extra: dict | None = None,
    ) -> Path:
        out_dir = self._ensure_dir(relative_dir)
        path = out_dir / filename

        manifest = {
            "num_samples": len(samples),
            "domains": sorted(set(s.domain_name for s in samples)),
            "generators": sorted(set(s.metadata.generator for s in samples)),
            "num_points": sorted(set(s.num_points for s in samples)),
            "sample_ids": [s.sample_id for s in samples],
        }

        if extra is not None:
            manifest["extra"] = extra

        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return path

    def write_dataset_bundle(
        self,
        samples: Sequence[SpectrumSample],
        dataset_name: str,
        write_individual_npz: bool = True,
        write_batch_npz: bool = True,
        write_metadata: bool = True,
        write_manifest: bool = True,
        compress: bool = True,
        manifest_extra: dict | None = None,
    ) -> dict[str, Path | list[Path]]:
        dataset_root = self.root / dataset_name
        dataset_root.mkdir(parents=True, exist_ok=True)

        nested_writer = DatasetWriter(dataset_root)
        outputs: dict[str, Path | list[Path]] = {}

        if write_individual_npz:
            outputs["sample_npz"] = nested_writer.write_samples_npz(
                samples=samples,
                relative_dir="samples",
                compress=compress,
            )

        if write_batch_npz:
            outputs["batch_npz"] = nested_writer.write_batch_npz(
                batch=SampleBatch(list(samples)),
                filename=f"{Path(dataset_name).name}.npz",
                relative_dir="batches",
                compress=compress,
            )

        if write_metadata:
            outputs["metadata_jsonl"] = nested_writer.write_metadata_jsonl(
                samples=samples,
                filename="metadata.jsonl",
                relative_dir="metadata",
            )

        if write_manifest:
            outputs["manifest"] = nested_writer.write_manifest(
                samples=samples,
                filename="manifest.json",
                relative_dir=".",
                extra=manifest_extra,
            )

        return outputs