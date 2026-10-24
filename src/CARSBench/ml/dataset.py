from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np


class ChunkedNPZDataset:
    """
    Disk-backed reader for one generated CARSBench domain/split.

    Expected directory structure
    ----------------------------
    domain_root/
        manifest.json
        batches/
            batch_000.npz
            batch_001.npz
            ...

    Only one NPZ batch is cached in memory at a time.

    Returned item
    -------------
    {
        "input": float32 array, shape (2, L),
        "spectrum": float32 array, shape (L,),
        "observation_mask": float32 array, shape (L,),
        "target": float32 array, shape (1, L),
        "axis": float32 array, shape (L,),
        "dataset_index": int,
        "sample_index": int,
        "domain_name": str,
    }

    Model input channels
    --------------------
    channel 0 = measured CARS/BCARS spectrum
    channel 1 = binary observation mask
    """

    REQUIRED_KEYS = {
        "axis",
        "spectrum",
        "raman_target",
        "observation_mask",
    }

    def __init__(
        self,
        domain_root: str | Path,
    ) -> None:
        self.domain_root = Path(
            domain_root
        )

        if not self.domain_root.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: "
                f"{self.domain_root}"
            )

        self.manifest_path = (
            self.domain_root
            / "manifest.json"
        )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Missing manifest: "
                f"{self.manifest_path}"
            )

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            self.manifest = json.load(
                handle
            )

        self.domain_name = str(
            self.manifest[
                "domain_name"
            ]
        )

        self.num_samples = int(
            self.manifest[
                "num_samples"
            ]
        )

        self.start_index = int(
            self.manifest.get(
                "start_index",
                0,
            )
        )

        self.chunk_size = int(
            self.manifest[
                "chunk_size"
            ]
        )

        self.n_chunks = int(
            self.manifest[
                "n_chunks"
            ]
        )

        if self.num_samples <= 0:
            raise ValueError(
                "Manifest num_samples must be positive."
            )

        if self.chunk_size <= 0:
            raise ValueError(
                "Manifest chunk_size must be positive."
            )

        if self.n_chunks <= 0:
            raise ValueError(
                "Manifest n_chunks must be positive."
            )

        self.batch_dir = (
            self.domain_root
            / "batches"
        )

        if not self.batch_dir.exists():
            raise FileNotFoundError(
                f"Missing batch directory: "
                f"{self.batch_dir}"
            )

        self.batch_files = sorted(
            self.batch_dir.glob(
                "batch_*.npz"
            )
        )

        if len(
            self.batch_files
        ) != self.n_chunks:
            raise ValueError(
                "Number of batch files does not match manifest: "
                f"found {len(self.batch_files)}, "
                f"expected {self.n_chunks}."
            )

        expected_chunks = (
            self.num_samples
            + self.chunk_size
            - 1
        ) // self.chunk_size

        if expected_chunks != self.n_chunks:
            raise ValueError(
                "Manifest n_chunks is inconsistent with "
                "num_samples and chunk_size."
            )

        # -----------------------------------------------------
        # Validate the stored schema using the first chunk.
        # -----------------------------------------------------

        with np.load(
            self.batch_files[0],
            allow_pickle=False,
        ) as batch:
            keys = set(
                batch.files
            )

            missing = (
                self.REQUIRED_KEYS
                - keys
            )

            if missing:
                raise ValueError(
                    "Batch is missing required arrays: "
                    f"{sorted(missing)}"
                )

            first_spectrum = (
                batch[
                    "spectrum"
                ]
            )

            if first_spectrum.ndim != 2:
                raise ValueError(
                    "Expected spectrum array with shape "
                    "(N, L)."
                )

            self.spectral_length = int(
                first_spectrum.shape[
                    1
                ]
            )

        # -----------------------------------------------------
        # One-chunk cache.
        #
        # Each PyTorch worker will receive its own Dataset
        # instance/process state and therefore its own cache.
        # -----------------------------------------------------

        self._cached_chunk_index: (
            int | None
        ) = None

        self._cached_arrays: (
            dict[str, np.ndarray]
            | None
        ) = None

    def __len__(
        self,
    ) -> int:
        return self.num_samples

    def _validate_dataset_index(
        self,
        index: int,
    ) -> int:
        if index < 0:
            index += self.num_samples

        if (
            index < 0
            or index >= self.num_samples
        ):
            raise IndexError(
                f"Dataset index out of range: "
                f"{index}"
            )

        return index

    def chunk_index_for(
        self,
        index: int,
    ) -> int:
        """
        Return the NPZ chunk containing one dataset index.
        """
        index = (
            self._validate_dataset_index(
                index
            )
        )

        return (
            index
            // self.chunk_size
        )

    def local_index_for(
        self,
        index: int,
    ) -> int:
        """
        Return the row inside its NPZ chunk.
        """
        index = (
            self._validate_dataset_index(
                index
            )
        )

        return (
            index
            % self.chunk_size
        )

    def chunk_bounds(
        self,
        chunk_index: int,
    ) -> tuple[int, int]:
        """
        Dataset-index interval [start, end) for one chunk.
        """
        if (
            chunk_index < 0
            or chunk_index
            >= self.n_chunks
        ):
            raise IndexError(
                f"Chunk index out of range: "
                f"{chunk_index}"
            )

        start = (
            chunk_index
            * self.chunk_size
        )

        end = min(
            start
            + self.chunk_size,
            self.num_samples,
        )

        return start, end

    def _load_chunk(
        self,
        chunk_index: int,
    ) -> dict[str, np.ndarray]:
        if (
            self._cached_chunk_index
            == chunk_index
            and self._cached_arrays
            is not None
        ):
            return self._cached_arrays

        path = self.batch_files[
            chunk_index
        ]

        with np.load(
            path,
            allow_pickle=False,
        ) as batch:
            arrays = {
                "axis": np.asarray(
                    batch[
                        "axis"
                    ],
                    dtype=np.float32,
                ),

                "spectrum": np.asarray(
                    batch[
                        "spectrum"
                    ],
                    dtype=np.float32,
                ),

                "raman_target": (
                    np.asarray(
                        batch[
                            "raman_target"
                        ],
                        dtype=np.float32,
                    )
                ),

                "observation_mask": (
                    np.asarray(
                        batch[
                            "observation_mask"
                        ],
                        dtype=np.float32,
                    )
                ),
            }

        n_rows = int(
            arrays[
                "spectrum"
            ].shape[
                0
            ]
        )

        expected_start, expected_end = (
            self.chunk_bounds(
                chunk_index
            )
        )

        expected_rows = (
            expected_end
            - expected_start
        )

        if n_rows != expected_rows:
            raise ValueError(
                f"Batch {path.name} contains "
                f"{n_rows} rows, expected "
                f"{expected_rows}."
            )

        for key in [
            "axis",
            "spectrum",
            "raman_target",
            "observation_mask",
        ]:
            arr = arrays[
                key
            ]

            if arr.shape != (
                n_rows,
                self.spectral_length,
            ):
                raise ValueError(
                    f"{path.name}:{key} has "
                    f"shape {arr.shape}; expected "
                    f"({n_rows}, "
                    f"{self.spectral_length})."
                )

            if not np.all(
                np.isfinite(
                    arr
                )
            ):
                raise ValueError(
                    f"{path.name}:{key} contains "
                    "non-finite values."
                )

        mask = arrays[
            "observation_mask"
        ]

        if not np.all(
            np.logical_or(
                mask == 0.0,
                mask == 1.0,
            )
        ):
            raise ValueError(
                f"{path.name}: observation mask "
                "must be binary."
            )

        self._cached_chunk_index = (
            chunk_index
        )

        self._cached_arrays = arrays

        return arrays

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        index = (
            self._validate_dataset_index(
                index
            )
        )

        chunk_index = (
            self.chunk_index_for(
                index
            )
        )

        local_index = (
            self.local_index_for(
                index
            )
        )

        arrays = (
            self._load_chunk(
                chunk_index
            )
        )

        spectrum = (
            arrays[
                "spectrum"
            ][
                local_index
            ].copy()
        )

        mask = (
            arrays[
                "observation_mask"
            ][
                local_index
            ].copy()
        )

        target = (
            arrays[
                "raman_target"
            ][
                local_index
            ].copy()
        )

        axis = (
            arrays[
                "axis"
            ][
                local_index
            ].copy()
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

        target = target[
            np.newaxis,
            :
        ].astype(
            np.float32,
            copy=False,
        )

        return {
            "input": model_input,

            "spectrum": spectrum,

            "observation_mask": mask,

            "target": target,

            "axis": axis,

            "dataset_index": index,

            "sample_index": (
                self.start_index
                + index
            ),

            "domain_name": (
                self.domain_name
            ),
        }


def iter_chunk_shuffled_indices(
    dataset: ChunkedNPZDataset,
    *,
    seed: int,
    shuffle_chunks: bool = True,
    shuffle_within_chunks: bool = True,
) -> Iterator[int]:
    """
    Yield every dataset index exactly once while preserving
    chunk-local disk access.

    Why this exists
    ---------------
    Completely random sample-level shuffling causes frequent switching
    between compressed NPZ files.

    Instead:

    1. shuffle chunk order;
    2. optionally shuffle samples inside each chunk;
    3. process one chunk before moving to another.

    This retains stochastic training order while making NPZ access
    substantially more efficient.
    """
    rng = np.random.default_rng(
        seed
    )

    chunk_indices = np.arange(
        dataset.n_chunks,
        dtype=np.int64,
    )

    if shuffle_chunks:
        rng.shuffle(
            chunk_indices
        )

    for chunk_index_value in (
        chunk_indices
    ):
        chunk_index = int(
            chunk_index_value
        )

        start, end = (
            dataset.chunk_bounds(
                chunk_index
            )
        )

        indices = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        if shuffle_within_chunks:
            rng.shuffle(
                indices
            )

        for index in indices:
            yield int(
                index
            )