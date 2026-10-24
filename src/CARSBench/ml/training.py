from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import nn

from .dataset import (
    ChunkedNPZDataset,
    iter_chunk_shuffled_indices,
)
from .preprocessing import (
    NormalizationStats,
    denormalize_target,
    prepare_model_item,
)


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    batch_size: int = 64

    maximum_epochs: int = 100

    gradient_clip_norm: float = 1.0

    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1.0e-5

    def __post_init__(
        self,
    ) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.weight_decay < 0.0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.maximum_epochs <= 0:
            raise ValueError(
                "maximum_epochs must be positive."
            )

        if self.gradient_clip_norm <= 0.0:
            raise ValueError(
                "gradient_clip_norm must be positive."
            )

        if self.early_stopping_patience <= 0:
            raise ValueError(
                "early_stopping_patience must be positive."
            )

        if self.early_stopping_min_delta < 0.0:
            raise ValueError(
                "early_stopping_min_delta cannot be negative."
            )


@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_mse: float
    epochs_completed: int
    stopped_early: bool
    history: list[dict[str, float]]

    def to_dict(
        self,
    ) -> dict:
        return {
            "best_epoch": int(
                self.best_epoch
            ),

            "best_validation_mse": float(
                self.best_validation_mse
            ),

            "epochs_completed": int(
                self.epochs_completed
            ),

            "stopped_early": bool(
                self.stopped_early
            ),

            "history": self.history,
        }


def select_device(
    preference: str = "auto",
) -> torch.device:
    """
    Device selection policy.

    auto:
        CUDA -> MPS -> CPU
    """
    preference = (
        preference
        .strip()
        .lower()
    )

    valid = {
        "auto",
        "cuda",
        "mps",
        "cpu",
    }

    if preference not in valid:
        raise ValueError(
            f"Unknown device preference "
            f"{preference!r}. "
            f"Expected one of {sorted(valid)}."
        )

    if preference == "cpu":
        return torch.device(
            "cpu"
        )

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device(
            "cuda"
        )

    if preference == "mps":
        if not (
            hasattr(
                torch.backends,
                "mps",
            )
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "MPS was requested but is not available."
            )

        return torch.device(
            "mps"
        )

    # auto
    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


def seed_everything(
    seed: int,
) -> None:
    """
    Seed Python, NumPy and PyTorch RNGs.

    This controls initialization and data-order randomness within
    one software/hardware environment.

    Exact bitwise equality is not promised across different device
    backends or PyTorch versions.
    """
    seed = int(
        seed
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def _prefix_indices(
    indices: Iterable[int],
    *,
    num_samples: int,
) -> Iterator[int]:
    """
    Restrict an index stream to dataset indices [0, num_samples).
    """
    for index in indices:
        if index < num_samples:
            yield index


def _make_training_indices(
    dataset: ChunkedNPZDataset,
    *,
    num_samples: int,
    epoch_seed: int,
) -> Iterator[int]:
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

    all_indices = (
        iter_chunk_shuffled_indices(
            dataset,
            seed=epoch_seed,
            shuffle_chunks=True,
            shuffle_within_chunks=True,
        )
    )

    return _prefix_indices(
        all_indices,
        num_samples=num_samples,
    )


def _make_sequential_indices(
    dataset,
    *,
    num_samples: int | None = None,
) -> range:
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

    return range(
        num_samples
    )


def iter_tensor_batches(
    dataset,
    stats: NormalizationStats,
    *,
    indices: Iterable[int],
    batch_size: int,
    device: torch.device,
) -> Iterator[
    tuple[
        torch.Tensor,
        torch.Tensor,
        list[dict],
    ]
]:
    """
    Convert dataset items to normalized PyTorch batches.

    Yields
    ------
    inputs:
        (B, 2, L)

    targets:
        normalized Raman targets, shape (B, 1, L)

    prepared_items:
        retained for evaluation metadata/raw targets.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    batch_items: list[dict] = []

    for index in indices:
        raw_item = dataset[
            int(
                index
            )
        ]

        prepared = prepare_model_item(
            raw_item,
            stats,
        )

        batch_items.append(
            prepared
        )

        if len(
            batch_items
        ) == batch_size:
            inputs = np.stack(
                [
                    item[
                        "input"
                    ]
                    for item
                    in batch_items
                ],
                axis=0,
            )

            targets = np.stack(
                [
                    item[
                        "target"
                    ]
                    for item
                    in batch_items
                ],
                axis=0,
            )

            yield (
                torch.from_numpy(
                    inputs
                ).to(
                    device=device,
                    dtype=torch.float32,
                ),

                torch.from_numpy(
                    targets
                ).to(
                    device=device,
                    dtype=torch.float32,
                ),

                batch_items,
            )

            batch_items = []

    if batch_items:
        inputs = np.stack(
            [
                item[
                    "input"
                ]
                for item
                in batch_items
            ],
            axis=0,
        )

        targets = np.stack(
            [
                item[
                    "target"
                ]
                for item
                in batch_items
            ],
            axis=0,
        )

        yield (
            torch.from_numpy(
                inputs
            ).to(
                device=device,
                dtype=torch.float32,
            ),

            torch.from_numpy(
                targets
            ).to(
                device=device,
                dtype=torch.float32,
            ),

            batch_items,
        )


def train_one_epoch(
    model: nn.Module,
    dataset,
    stats: NormalizationStats,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    batch_size: int,
    num_samples: int,
    epoch_seed: int,
    gradient_clip_norm: float = 1.0,
) -> float:
    """
    Train for one complete pass through a source-training prefix.

    Returns mean normalized MSE over all target elements.
    """
    model.train()

    indices = _make_training_indices(
        dataset,
        num_samples=num_samples,
        epoch_seed=epoch_seed,
    )

    squared_error_sum = 0.0
    element_count = 0

    loss_function = nn.MSELoss(
        reduction="mean"
    )

    for (
        inputs,
        targets,
        _,
    ) in iter_tensor_batches(
        dataset,
        stats,
        indices=indices,
        batch_size=batch_size,
        device=device,
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            inputs
        )

        if (
            predictions.shape
            != targets.shape
        ):
            raise ValueError(
                "Model prediction and target "
                "shapes differ: "
                f"{tuple(predictions.shape)} vs "
                f"{tuple(targets.shape)}."
            )

        loss = loss_function(
            predictions,
            targets,
        )

        if not torch.isfinite(
            loss
        ):
            raise FloatingPointError(
                "Non-finite training loss."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=(
                gradient_clip_norm
            ),
        )

        optimizer.step()

        with torch.no_grad():
            difference = (
                predictions
                - targets
            )

            squared_error_sum += float(
                torch.sum(
                    difference
                    * difference
                ).item()
            )

            element_count += int(
                difference.numel()
            )

    if element_count == 0:
        raise RuntimeError(
            "Training epoch processed zero elements."
        )

    return (
        squared_error_sum
        / element_count
    )


def evaluate_normalized_mse(
    model: nn.Module,
    dataset,
    stats: NormalizationStats,
    *,
    device: torch.device,
    batch_size: int,
    num_samples: int | None = None,
) -> float:
    """
    Source-validation metric used for early stopping.

    No OOD dataset should be passed here for model selection.
    """
    model.eval()

    indices = _make_sequential_indices(
        dataset,
        num_samples=num_samples,
    )

    squared_error_sum = 0.0
    element_count = 0

    with torch.inference_mode():
        for (
            inputs,
            targets,
            _,
        ) in iter_tensor_batches(
            dataset,
            stats,
            indices=indices,
            batch_size=batch_size,
            device=device,
        ):
            predictions = model(
                inputs
            )

            difference = (
                predictions
                - targets
            )

            squared_error_sum += float(
                torch.sum(
                    difference
                    * difference
                ).item()
            )

            element_count += int(
                difference.numel()
            )

    if element_count == 0:
        raise RuntimeError(
            "Validation processed zero elements."
        )

    return (
        squared_error_sum
        / element_count
    )


class EarlyStopping:
    """
    Source-validation early stopping.

    The best model state is kept in CPU memory and optionally written
    to disk.
    """

    def __init__(
        self,
        *,
        patience: int,
        min_delta: float,
        checkpoint_path: (
            str
            | Path
            | None
        ) = None,
    ) -> None:
        if patience <= 0:
            raise ValueError(
                "patience must be positive."
            )

        if min_delta < 0.0:
            raise ValueError(
                "min_delta cannot be negative."
            )

        self.patience = int(
            patience
        )

        self.min_delta = float(
            min_delta
        )

        self.checkpoint_path = (
            Path(
                checkpoint_path
            )
            if checkpoint_path
            is not None
            else None
        )

        self.best_value = float(
            "inf"
        )

        self.best_epoch = -1

        self.bad_epochs = 0

        self.best_state: (
            dict
            | None
        ) = None

    def update(
        self,
        value: float,
        *,
        epoch: int,
        model: nn.Module,
    ) -> bool:
        """
        Update early-stopping state.

        Returns True when training should stop.
        """
        value = float(
            value
        )

        if not np.isfinite(
            value
        ):
            raise FloatingPointError(
                "Validation metric is non-finite."
            )

        improved = (
            value
            < (
                self.best_value
                - self.min_delta
            )
        )

        if improved:
            self.best_value = value

            self.best_epoch = int(
                epoch
            )

            self.bad_epochs = 0

            self.best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            if (
                self.checkpoint_path
                is not None
            ):
                self.checkpoint_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                torch.save(
                    {
                        "epoch": (
                            self.best_epoch
                        ),

                        "validation_mse": (
                            self.best_value
                        ),

                        "model_state_dict": (
                            self.best_state
                        ),
                    },
                    self.checkpoint_path,
                )

            return False

        self.bad_epochs += 1

        return (
            self.bad_epochs
            >= self.patience
        )

    def restore(
        self,
        model: nn.Module,
    ) -> None:
        if self.best_state is None:
            raise RuntimeError(
                "No best model state is available."
            )

        model.load_state_dict(
            self.best_state
        )


def train_with_early_stopping(
    model: nn.Module,
    train_dataset,
    validation_dataset,
    stats: NormalizationStats,
    *,
    model_seed: int,
    num_train_samples: int,
    config: TrainingConfig,
    device: torch.device,
    checkpoint_path: (
        str
        | Path
        | None
    ) = None,
) -> TrainingResult:
    """
    Train the baseline using only source train + source validation.

    OOD datasets are intentionally absent from this API.
    """
    if num_train_samples <= 0:
        raise ValueError(
            "num_train_samples must be positive."
        )

    if num_train_samples > len(
        train_dataset
    ):
        raise ValueError(
            "num_train_samples exceeds training dataset."
        )

    model.to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    stopper = EarlyStopping(
        patience=(
            config
            .early_stopping_patience
        ),

        min_delta=(
            config
            .early_stopping_min_delta
        ),

        checkpoint_path=(
            checkpoint_path
        ),
    )

    history: list[
        dict[str, float]
    ] = []

    stopped_early = False

    for epoch in range(
        config.maximum_epochs
    ):
        epoch_seed = (
            int(
                model_seed
            )
            + epoch
        )

        train_mse = train_one_epoch(
            model,
            train_dataset,
            stats,
            optimizer,

            device=device,

            batch_size=(
                config.batch_size
            ),

            num_samples=(
                num_train_samples
            ),

            epoch_seed=(
                epoch_seed
            ),

            gradient_clip_norm=(
                config
                .gradient_clip_norm
            ),
        )

        validation_mse = (
            evaluate_normalized_mse(
                model,
                validation_dataset,
                stats,

                device=device,

                batch_size=(
                    config.batch_size
                ),
            )
        )

        history.append(
            {
                "epoch": float(
                    epoch
                ),

                "train_normalized_mse": float(
                    train_mse
                ),

                "validation_normalized_mse": float(
                    validation_mse
                ),
            }
        )

        should_stop = (
            stopper.update(
                validation_mse,
                epoch=epoch,
                model=model,
            )
        )

        if should_stop:
            stopped_early = True
            break

    stopper.restore(
        model
    )

    return TrainingResult(
        best_epoch=(
            stopper.best_epoch
        ),

        best_validation_mse=(
            stopper.best_value
        ),

        epochs_completed=len(
            history
        ),

        stopped_early=(
            stopped_early
        ),

        history=history,
    )


def _spectral_angle_rows(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Spectral angle in radians for corresponding rows.

    prediction and target:
        shape (N, L)
    """
    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    dot = np.sum(
        prediction
        * target,
        axis=1,
    )

    prediction_norm = np.linalg.norm(
        prediction,
        axis=1,
    )

    target_norm = np.linalg.norm(
        target,
        axis=1,
    )

    denominator = (
        prediction_norm
        * target_norm
    )

    result = np.full(
        prediction.shape[
            0
        ],
        np.nan,
        dtype=np.float64,
    )

    valid = (
        denominator
        > np.finfo(
            np.float64
        ).eps
    )

    cosine = np.zeros_like(
        dot
    )

    cosine[
        valid
    ] = (
        dot[
            valid
        ]
        / denominator[
            valid
        ]
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    result[
        valid
    ] = np.arccos(
        cosine[
            valid
        ]
    )

    return result


def _row_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    if (
        prediction.shape
        != target.shape
    ):
        raise ValueError(
            "Prediction and target shapes differ."
        )

    difference = (
        prediction
        - target
    )

    rmse_values = np.sqrt(
        np.mean(
            difference
            * difference,
            axis=1,
        )
    )

    mae_values = np.mean(
        np.abs(
            difference
        ),
        axis=1,
    )

    angle_values = (
        _spectral_angle_rows(
            prediction,
            target,
        )
    )

    return {
        "rmse": rmse_values,
        "mae": mae_values,
        "spectral_angle": angle_values,
    }


def _summarize_metric(
    values: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = values[
        np.isfinite(
            values
        )
    ]

    if finite.size == 0:
        return {
            "mean": float(
                "nan"
            ),
            "median": float(
                "nan"
            ),
            "std": float(
                "nan"
            ),
            "p05": float(
                "nan"
            ),
            "p95": float(
                "nan"
            ),
        }

    return {
        "mean": float(
            np.mean(
                finite
            )
        ),

        "median": float(
            np.median(
                finite
            )
        ),

        "std": float(
            np.std(
                finite
            )
        ),

        "p05": float(
            np.percentile(
                finite,
                5.0,
            )
        ),

        "p95": float(
            np.percentile(
                finite,
                95.0,
            )
        ),
    }


def _masked_metrics_one_sample(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if not np.any(
        mask
    ):
        return {
            "rmse": float(
                "nan"
            ),
            "mae": float(
                "nan"
            ),
            "spectral_angle": float(
                "nan"
            ),
        }

    prediction_region = (
        np.asarray(
            prediction,
            dtype=np.float64,
        )[
            mask
        ]
    )

    target_region = (
        np.asarray(
            target,
            dtype=np.float64,
        )[
            mask
        ]
    )

    difference = (
        prediction_region
        - target_region
    )

    rmse_value = float(
        np.sqrt(
            np.mean(
                difference
                * difference
            )
        )
    )

    mae_value = float(
        np.mean(
            np.abs(
                difference
            )
        )
    )

    denominator = (
        np.linalg.norm(
            prediction_region
        )
        * np.linalg.norm(
            target_region
        )
    )

    if (
        denominator
        <= np.finfo(
            np.float64
        ).eps
    ):
        angle = float(
            "nan"
        )

    else:
        cosine = float(
            np.dot(
                prediction_region,
                target_region,
            )
            / denominator
        )

        cosine = float(
            np.clip(
                cosine,
                -1.0,
                1.0,
            )
        )

        angle = float(
            np.arccos(
                cosine
            )
        )

    return {
        "rmse": rmse_value,
        "mae": mae_value,
        "spectral_angle": angle,
    }


def evaluate_model(
    model: nn.Module,
    dataset,
    stats: NormalizationStats,
    *,
    device: torch.device,
    batch_size: int,
    num_samples: int | None = None,
) -> dict:
    """
    Evaluate in original Raman-target units.

    Report:
    - full-spectrum RMSE
    - full-spectrum MAE
    - full-spectrum spectral angle

    If missing spectral coverage is present, also report metrics on
    observed and unobserved regions separately.
    """
    model.eval()

    indices = _make_sequential_indices(
        dataset,
        num_samples=num_samples,
    )

    full_metric_values = {
        "rmse": [],
        "mae": [],
        "spectral_angle": [],
    }

    observed_metric_values = {
        "rmse": [],
        "mae": [],
        "spectral_angle": [],
    }

    unobserved_metric_values = {
        "rmse": [],
        "mae": [],
        "spectral_angle": [],
    }

    saw_missing_values = False

    processed_samples = 0

    with torch.inference_mode():
        for (
            inputs,
            _,
            prepared_items,
        ) in iter_tensor_batches(
            dataset,
            stats,
            indices=indices,
            batch_size=batch_size,
            device=device,
        ):
            normalized_predictions = (
                model(
                    inputs
                )
                .detach()
                .cpu()
                .numpy()
            )

            predictions = (
                denormalize_target(
                    normalized_predictions,
                    stats,
                )
            )

            targets = np.stack(
                [
                    item[
                        "raw_target"
                    ]
                    for item
                    in prepared_items
                ],
                axis=0,
            ).astype(
                np.float32,
                copy=False,
            )

            # Shape:
            # (B, 1, L) -> (B, L)
            predictions_2d = (
                predictions[
                    :,
                    0,
                    :
                ]
            )

            targets_2d = (
                targets[
                    :,
                    0,
                    :
                ]
            )

            metrics = _row_metrics(
                predictions_2d,
                targets_2d,
            )

            for metric_name in (
                full_metric_values
            ):
                full_metric_values[
                    metric_name
                ].extend(
                    metrics[
                        metric_name
                    ].tolist()
                )

            for local_index, item in enumerate(
                prepared_items
            ):
                mask = np.asarray(
                    item[
                        "observation_mask"
                    ],
                    dtype=bool,
                )

                observed = (
                    _masked_metrics_one_sample(
                        predictions_2d[
                            local_index
                        ],

                        targets_2d[
                            local_index
                        ],

                        mask,
                    )
                )

                for metric_name in (
                    observed_metric_values
                ):
                    observed_metric_values[
                        metric_name
                    ].append(
                        observed[
                            metric_name
                        ]
                    )

                missing_mask = (
                    ~mask
                )

                if np.any(
                    missing_mask
                ):
                    saw_missing_values = True

                    unobserved = (
                        _masked_metrics_one_sample(
                            predictions_2d[
                                local_index
                            ],

                            targets_2d[
                                local_index
                            ],

                            missing_mask,
                        )
                    )

                    for metric_name in (
                        unobserved_metric_values
                    ):
                        unobserved_metric_values[
                            metric_name
                        ].append(
                            unobserved[
                                metric_name
                            ]
                        )

                else:
                    for metric_name in (
                        unobserved_metric_values
                    ):
                        unobserved_metric_values[
                            metric_name
                        ].append(
                            float(
                                "nan"
                            )
                        )

            processed_samples += len(
                prepared_items
            )

    result = {
        "domain_name": getattr(
            dataset,
            "domain_name",
            "unknown",
        ),

        "num_samples": int(
            processed_samples
        ),

        "metrics": {
            name: _summarize_metric(
                np.asarray(
                    values
                )
            )
            for name, values
            in full_metric_values.items()
        },
    }

    if saw_missing_values:
        result[
            "spectral_regions"
        ] = {
            "observed": {
                name: _summarize_metric(
                    np.asarray(
                        values
                    )
                )
                for name, values
                in observed_metric_values.items()
            },

            "unobserved": {
                name: _summarize_metric(
                    np.asarray(
                        values
                    )
                )
                for name, values
                in unobserved_metric_values.items()
            },
        }

    return result


def save_training_artifacts(
    output_dir: str | Path,
    *,
    model: nn.Module,
    normalization_stats: (
        NormalizationStats
    ),
    training_config: TrainingConfig,
    training_result: TrainingResult,
    model_seed: int,
) -> None:
    """
    Persist everything needed to reproduce one trained baseline.
    """
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        output_dir
        / "model.pt",
    )

    with (
        output_dir
        / "normalization.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            normalization_stats.to_dict(),
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    metadata = {
        "model_seed": int(
            model_seed
        ),

        "training_config": (
            asdict(
                training_config
            )
        ),

        "training_result": (
            training_result.to_dict()
        ),
    }

    with (
        output_dir
        / "training.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )