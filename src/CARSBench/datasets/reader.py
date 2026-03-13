from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .schema import SampleBatch, SampleMetadata, SpectrumSample


class DatasetReader:
    """
    Reader utilities for CARSBench datasets.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _resolve(self, path: str | Path) -> Path:
        """
        Resolve paths robustly.

        Rules:
        - absolute path -> use directly
        - path already starting with root -> use directly
        - otherwise treat as relative to root
        """
        path = Path(path)

        if path.is_absolute():
            return path

        try:
            if str(path).startswith(str(self.root)):
                return path
        except Exception:
            pass

        return self.root / path

    @staticmethod
    def _parse_metadata_item(item) -> dict:
        if isinstance(item, np.ndarray) and item.shape == ():
            item = item.item()

        if isinstance(item, bytes):
            item = item.decode("utf-8")

        if isinstance(item, str):
            return json.loads(item)

        raise TypeError(f"Unsupported metadata type: {type(item)!r}")

    def read_sample_npz(self, path: str | Path) -> SpectrumSample:
        path = self._resolve(path)

        with np.load(path, allow_pickle=True) as data:
            if "metadata_json" in data:
                metadata = SampleMetadata(**self._parse_metadata_item(data["metadata_json"]))
            else:
                metadata = SampleMetadata(
                    sample_id=path.stem,
                    domain_name="unknown",
                )

            return SpectrumSample(
                axis=data["axis"],
                spectrum=data["spectrum"],
                raman_target=data["raman_target"],
                clean_intensity=data["clean_intensity"] if "clean_intensity" in data else None,
                chi_r_real=data["chi_r_real"] if "chi_r_real" in data else None,
                chi_r_imag=data["chi_r_imag"] if "chi_r_imag" in data else None,
                chi_nr_real=data["chi_nr_real"] if "chi_nr_real" in data else None,
                chi_nr_imag=data["chi_nr_imag"] if "chi_nr_imag" in data else None,
                envelope=data["envelope"] if "envelope" in data else None,
                baseline=data["baseline"] if "baseline" in data else None,
                metadata=metadata,
            )

    def read_samples_npz(
        self,
        relative_dir: str | Path = "samples",
        limit: Optional[int] = None,
    ) -> SampleBatch:
        directory = self._resolve(relative_dir)
        files = sorted(directory.glob("*.npz"))

        if limit is not None:
            files = files[:limit]

        return SampleBatch([self.read_sample_npz(f) for f in files])

    def read_batch_npz(self, path: str | Path) -> SampleBatch:
        path = self._resolve(path)

        with np.load(path, allow_pickle=True) as data:
            axis = data["axis"]
            spectrum = data["spectrum"]
            raman_target = data["raman_target"]

            if "metadata_json" in data:
                metadata_list = [self._parse_metadata_item(x) for x in data["metadata_json"]]
            else:
                metadata_list = [
                    {"sample_id": f"sample_{i:06d}", "domain_name": "unknown"}
                    for i in range(len(spectrum))
                ]

            optional_keys = [
                "clean_intensity",
                "chi_r_real",
                "chi_r_imag",
                "chi_nr_real",
                "chi_nr_imag",
                "envelope",
                "baseline",
            ]

            samples: list[SpectrumSample] = []

            for i in range(len(spectrum)):
                kwargs = {
                    "axis": axis[i],
                    "spectrum": spectrum[i],
                    "raman_target": raman_target[i],
                    "metadata": SampleMetadata(**metadata_list[i]),
                }

                for key in optional_keys:
                    if key in data:
                        kwargs[key] = data[key][i]

                samples.append(SpectrumSample(**kwargs))

            return SampleBatch(samples)

    def read_metadata_jsonl(self, path: str | Path) -> list[dict]:
        path = self._resolve(path)

        records: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def filter_by_domain(
        self,
        batch: SampleBatch,
        domain_names: Sequence[str],
    ) -> SampleBatch:
        domain_set = set(domain_names)
        return SampleBatch([s for s in batch if s.domain_name in domain_set])

    def filter_by_generator(
        self,
        batch: SampleBatch,
        generator_names: Sequence[str],
    ) -> SampleBatch:
        generator_set = set(generator_names)
        return SampleBatch([s for s in batch if s.metadata.generator in generator_set])

    def filter_by_num_points(
        self,
        batch: SampleBatch,
        num_points: int,
    ) -> SampleBatch:
        return SampleBatch([s for s in batch if s.num_points == num_points])