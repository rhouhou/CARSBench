from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Optional

import numpy as np


def _to_serializable(value: Any) -> Any:
    """
    Convert common Python / NumPy objects into JSON-safe structures.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()

    if is_dataclass(value):
        return _to_serializable(asdict(value))

    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_to_serializable(v) for v in value]

    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]

    return value


@dataclass
class SampleMetadata:
    """
    Metadata associated with one simulated sample.
    """

    sample_id: str
    domain_name: str
    seed: Optional[int] = None
    window_mode: Optional[str] = None
    generator: str = "frequency"
    parameters: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(asdict(self))


@dataclass
class SpectrumSample:
    """
    Canonical 1D CARSBench sample.
    """

    axis: np.ndarray
    spectrum: np.ndarray
    raman_target: np.ndarray

    clean_intensity: Optional[np.ndarray] = None

    chi_r_real: Optional[np.ndarray] = None
    chi_r_imag: Optional[np.ndarray] = None

    chi_nr_real: Optional[np.ndarray] = None
    chi_nr_imag: Optional[np.ndarray] = None

    envelope: Optional[np.ndarray] = None
    baseline: Optional[np.ndarray] = None

    metadata: SampleMetadata = field(
        default_factory=lambda: SampleMetadata(
            sample_id="unknown",
            domain_name="unknown",
        )
    )

    def __post_init__(self) -> None:
        self.axis = np.asarray(self.axis, dtype=np.float64)
        self.spectrum = np.asarray(self.spectrum, dtype=np.float64)
        self.raman_target = np.asarray(self.raman_target, dtype=np.float64)

        if self.clean_intensity is not None:
            self.clean_intensity = np.asarray(self.clean_intensity, dtype=np.float64)

        if self.chi_r_real is not None:
            self.chi_r_real = np.asarray(self.chi_r_real, dtype=np.float64)

        if self.chi_r_imag is not None:
            self.chi_r_imag = np.asarray(self.chi_r_imag, dtype=np.float64)

        if self.chi_nr_real is not None:
            self.chi_nr_real = np.asarray(self.chi_nr_real, dtype=np.float64)

        if self.chi_nr_imag is not None:
            self.chi_nr_imag = np.asarray(self.chi_nr_imag, dtype=np.float64)

        if self.envelope is not None:
            self.envelope = np.asarray(self.envelope, dtype=np.float64)

        if self.baseline is not None:
            self.baseline = np.asarray(self.baseline, dtype=np.float64)

        n = len(self.axis)
        self._validate_length("spectrum", self.spectrum, n)
        self._validate_length("raman_target", self.raman_target, n)
        self._validate_length("clean_intensity", self.clean_intensity, n)
        self._validate_length("chi_r_real", self.chi_r_real, n)
        self._validate_length("chi_r_imag", self.chi_r_imag, n)
        self._validate_length("chi_nr_real", self.chi_nr_real, n)
        self._validate_length("chi_nr_imag", self.chi_nr_imag, n)
        self._validate_length("envelope", self.envelope, n)
        self._validate_length("baseline", self.baseline, n)

    @staticmethod
    def _validate_length(name: str, arr: Optional[np.ndarray], n: int) -> None:
        if arr is None:
            return
        if len(arr) != n:
            raise ValueError(
                f"{name} has length {len(arr)}, expected {n} to match axis."
            )

    @property
    def num_points(self) -> int:
        return int(len(self.axis))

    @property
    def domain_name(self) -> str:
        return self.metadata.domain_name

    @property
    def sample_id(self) -> str:
        return self.metadata.sample_id

    def to_dict(self, include_arrays: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "metadata": self.metadata.to_dict(),
            "num_points": self.num_points,
        }

        if include_arrays:
            out.update(
                {
                    "axis": _to_serializable(self.axis),
                    "spectrum": _to_serializable(self.spectrum),
                    "raman_target": _to_serializable(self.raman_target),
                    "clean_intensity": _to_serializable(self.clean_intensity),
                    "chi_r_real": _to_serializable(self.chi_r_real),
                    "chi_r_imag": _to_serializable(self.chi_r_imag),
                    "chi_nr_real": _to_serializable(self.chi_nr_real),
                    "chi_nr_imag": _to_serializable(self.chi_nr_imag),
                    "envelope": _to_serializable(self.envelope),
                    "baseline": _to_serializable(self.baseline),
                }
            )

        return out

    def to_numpy_dict(self) -> Dict[str, Any]:
        def _f32(x):
            return np.asarray(x, dtype=np.float32)
        
        out: Dict[str, Any] = {
            "axis": _f32(self.axis),
            "spectrum": _f32(self.spectrum),
            "raman_target": _f32(self.raman_target),
        }

        optional_arrays = {
            "clean_intensity": self.clean_intensity,
            "chi_r_real": self.chi_r_real,
            "chi_r_imag": self.chi_r_imag,
            "chi_nr_real": self.chi_nr_real,
            "chi_nr_imag": self.chi_nr_imag,
            "envelope": self.envelope,
            "baseline": self.baseline,
        }

        for key, value in optional_arrays.items():
            if value is not None:
                out[key] = _f32(value)

        return out


@dataclass
class SampleBatch:
    """
    Container for a list of SpectrumSample objects.
    """

    samples: list[SpectrumSample]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def append(self, sample: SpectrumSample) -> None:
        self.samples.append(sample)

    def domain_names(self) -> list[str]:
        return [s.domain_name for s in self.samples]

    def sample_ids(self) -> list[str]:
        return [s.sample_id for s in self.samples]

    def to_metadata_table(self) -> list[dict]:
        return [s.to_dict(include_arrays=False) for s in self.samples]

    def stack(self) -> Dict[str, np.ndarray]:
        if len(self.samples) == 0:
            raise ValueError("Cannot stack an empty SampleBatch.")

        out = {
            "axis": np.stack([s.axis for s in self.samples], axis=0),
            "spectrum": np.stack([s.spectrum for s in self.samples], axis=0),
            "raman_target": np.stack([s.raman_target for s in self.samples], axis=0),
        }

        optional_keys = [
            "clean_intensity",
            "chi_r_real",
            "chi_r_imag",
            "chi_nr_real",
            "chi_nr_imag",
            "envelope",
            "baseline",
        ]

        for key in optional_keys:
            values = [getattr(s, key) for s in self.samples]
            if all(v is not None for v in values):
                out[key] = np.stack(values, axis=0)

        return out