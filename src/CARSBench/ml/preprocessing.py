from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import ChunkedNPZDataset


@dataclass(frozen=True)
class NormalizationStats:
    """
    Scalar source-training normalization statistics.

    Spectrum statistics are computed only over observed input points.

    Target statistics are computed over every point of every Raman
    target in the selected source-training subset.
    """

    spectrum_mean: float
    spectrum_std: float

    target_mean: float
    target_std: float

    num_samples: int
    num_observed_spectrum_values: int
    num_target_values: int

    def __post_init__(
        self,
    ) -> None:
        if self.num_samples <= 0:
            raise ValueError(
                "num_samples must be positive."
            )

        if (
            self.num_observed_spectrum_values
            <= 0
        ):
            raise ValueError(
                "num_observed_spectrum_values "
                "must be positive."
            )

        if self.num_target_values <= 0:
            raise ValueError(
                "num_target_values must be positive."
            )

        if not np.isfinite(
            self.spectrum_mean
        ):
            raise ValueError(
                "spectrum_mean must be finite."
            )

        if not np.isfinite(
            self.target_mean
        ):
            raise ValueError(
                "target_mean must be finite."
            )

        if (
            not np.isfinite(
                self.spectrum_std
            )
            or self.spectrum_std <= 0.0
        ):
            raise ValueError(
                "spectrum_std must be finite "
                "and positive."
            )

        if (
            not np.isfinite(
                self.target_std
            )
            or self.target_std <= 0.0
        ):
            raise ValueError(
                "target_std must be finite "
                "and positive."
            )

    def to_dict(
        self,
    ) -> dict:
        return {
            "spectrum_mean": float(
                self.spectrum_mean
            ),

            "spectrum_std": float(
                self.spectrum_std
            ),

            "target_mean": float(
                self.target_mean
            ),

            "target_std": float(
                self.target_std
            ),

            "num_samples": int(
                self.num_samples
            ),

            "num_observed_spectrum_values": int(
                self.num_observed_spectrum_values
            ),

            "num_target_values": int(
                self.num_target_values
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ) -> "NormalizationStats":
        return cls(
            spectrum_mean=float(
                data[
                    "spectrum_mean"
                ]
            ),

            spectrum_std=float(
                data[
                    "spectrum_std"
                ]
            ),

            target_mean=float(
                data[
                    "target_mean"
                ]
            ),

            target_std=float(
                data[
                    "target_std"
                ]
            ),

            num_samples=int(
                data[
                    "num_samples"
                ]
            ),

            num_observed_spectrum_values=int(
                data[
                    "num_observed_spectrum_values"
                ]
            ),

            num_target_values=int(
                data[
                    "num_target_values"
                ]
            ),
        )


def _mean_and_std_from_sums(
    *,
    value_sum: float,
    value_squared_sum: float,
    count: int,
    name: str,
) -> tuple[float, float]:
    if count <= 0:
        raise ValueError(
            f"No values available for {name}."
        )

    mean = (
        value_sum
        / count
    )

    variance = (
        value_squared_sum
        / count
        - mean * mean
    )

    # Small negative values are possible from floating-point
    # cancellation.
    variance = max(
        variance,
        0.0,
    )

    std = float(
        np.sqrt(
            variance
        )
    )

    if (
        not np.isfinite(
            std
        )
        or std <= 0.0
    ):
        raise ValueError(
            f"{name} standard deviation "
            f"is invalid: {std}"
        )

    return (
        float(mean),
        std,
    )


def fit_normalization(
    dataset: ChunkedNPZDataset,
    *,
    num_samples: int | None = None,
) -> NormalizationStats:
    """
    Fit normalization statistics using only a source training set.

    Parameters
    ----------
    dataset:
        Source-domain training dataset.

    num_samples:
        Number of prefix samples used to fit the statistics.

        This supports the nested CARSBench scale study:

            5k ⊂ 10k ⊂ 25k ⊂ 50k ⊂ 100k

        If None, the complete dataset is used.

    Notes
    -----
    Spectrum statistics:
        Only coordinates where observation_mask == 1 are included.

    Target statistics:
        Every Raman-target coordinate is included.

    Float64 accumulation is used even though stored spectra are
    float32.
    """
    if num_samples is None:
        num_samples = len(
            dataset
        )

    num_samples = int(
        num_samples
    )

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be positive."
        )

    if num_samples > len(
        dataset
    ):
        raise ValueError(
            "num_samples exceeds dataset size."
        )

    spectrum_sum = 0.0
    spectrum_squared_sum = 0.0
    spectrum_count = 0

    target_sum = 0.0
    target_squared_sum = 0.0
    target_count = 0

    for index in range(
        num_samples
    ):
        item = dataset[
            index
        ]

        spectrum = np.asarray(
            item[
                "spectrum"
            ],
            dtype=np.float64,
        )

        mask = np.asarray(
            item[
                "observation_mask"
            ],
            dtype=bool,
        )

        target = np.asarray(
            item[
                "target"
            ],
            dtype=np.float64,
        )

        observed = spectrum[
            mask
        ]

        spectrum_sum += float(
            np.sum(
                observed,
                dtype=np.float64,
            )
        )

        spectrum_squared_sum += float(
            np.sum(
                observed * observed,
                dtype=np.float64,
            )
        )

        spectrum_count += int(
            observed.size
        )

        target_sum += float(
            np.sum(
                target,
                dtype=np.float64,
            )
        )

        target_squared_sum += float(
            np.sum(
                target * target,
                dtype=np.float64,
            )
        )

        target_count += int(
            target.size
        )

    spectrum_mean, spectrum_std = (
        _mean_and_std_from_sums(
            value_sum=spectrum_sum,
            value_squared_sum=(
                spectrum_squared_sum
            ),
            count=spectrum_count,
            name="spectrum",
        )
    )

    target_mean, target_std = (
        _mean_and_std_from_sums(
            value_sum=target_sum,
            value_squared_sum=(
                target_squared_sum
            ),
            count=target_count,
            name="target",
        )
    )

    return NormalizationStats(
        spectrum_mean=(
            spectrum_mean
        ),

        spectrum_std=(
            spectrum_std
        ),

        target_mean=(
            target_mean
        ),

        target_std=(
            target_std
        ),

        num_samples=(
            num_samples
        ),

        num_observed_spectrum_values=(
            spectrum_count
        ),

        num_target_values=(
            target_count
        ),
    )


def normalize_spectrum(
    spectrum: np.ndarray,
    observation_mask: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    """
    Normalize a measured spectrum with source-training statistics.

    Missing coordinates remain exactly zero.
    """
    spectrum = np.asarray(
        spectrum,
        dtype=np.float32,
    )

    mask = np.asarray(
        observation_mask,
        dtype=np.float32,
    )

    if spectrum.shape != mask.shape:
        raise ValueError(
            "spectrum and observation_mask "
            "must have identical shapes."
        )

    normalized = (
        (
            spectrum
            - stats.spectrum_mean
        )
        / stats.spectrum_std
    ).astype(
        np.float32,
        copy=False,
    )

    # Missingness must not become an artificial standardized
    # intensity.
    normalized = np.where(
        mask > 0.5,
        normalized,
        0.0,
    ).astype(
        np.float32,
        copy=False,
    )

    return normalized


def normalize_target(
    target: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    target = np.asarray(
        target,
        dtype=np.float32,
    )

    return (
        (
            target
            - stats.target_mean
        )
        / stats.target_std
    ).astype(
        np.float32,
        copy=False,
    )


def denormalize_target(
    normalized_target: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    normalized_target = np.asarray(
        normalized_target,
        dtype=np.float32,
    )

    return (
        normalized_target
        * stats.target_std
        + stats.target_mean
    ).astype(
        np.float32,
        copy=False,
    )


def prepare_model_item(
    item: dict,
    stats: NormalizationStats,
) -> dict:
    """
    Convert one raw ChunkedNPZDataset item into the representation
    used for baseline training.
    """
    spectrum = normalize_spectrum(
        item[
            "spectrum"
        ],
        item[
            "observation_mask"
        ],
        stats,
    )

    mask = np.asarray(
        item[
            "observation_mask"
        ],
        dtype=np.float32,
    )

    target = normalize_target(
        item[
            "target"
        ],
        stats,
    )

    model_input = np.stack(
        [
            spectrum,
            mask,
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    return {
        "input": model_input,
        "target": target,

        "raw_target": np.asarray(
            item[
                "target"
            ],
            dtype=np.float32,
        ),

        "observation_mask": mask,

        "axis": np.asarray(
            item[
                "axis"
            ],
            dtype=np.float32,
        ),

        "dataset_index": item[
            "dataset_index"
        ],

        "sample_index": item[
            "sample_index"
        ],

        "domain_name": item[
            "domain_name"
        ],
    }