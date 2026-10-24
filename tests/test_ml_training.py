from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch"
)

from torch import nn

from CARSBench.ml import (
    EarlyStopping,
    NormalizationStats,
    TrainingConfig,
    evaluate_model,
    evaluate_normalized_mse,
    seed_everything,
    select_device,
    train_one_epoch,
)


class ToyDataset:
    """
    Small in-memory dataset implementing the interface needed by the
    CARSBench trainer.

    target == spectrum, so an identity model has perfect metrics.
    """

    def __init__(
        self,
        *,
        num_samples: int = 12,
        spectral_length: int = 32,
        chunk_size: int = 4,
    ) -> None:
        rng = np.random.default_rng(
            42
        )

        self.spectra = rng.normal(
            size=(
                num_samples,
                spectral_length,
            )
        ).astype(
            np.float32
        )

        self.masks = np.ones(
            (
                num_samples,
                spectral_length,
            ),
            dtype=np.float32,
        )

        self.num_samples = (
            num_samples
        )

        self.chunk_size = (
            chunk_size
        )

        self.n_chunks = (
            num_samples
            + chunk_size
            - 1
        ) // chunk_size

        self.domain_name = (
            "toy"
        )

    def __len__(
        self,
    ) -> int:
        return (
            self.num_samples
        )

    def chunk_bounds(
        self,
        chunk_index: int,
    ) -> tuple[int, int]:
        start = (
            chunk_index
            * self.chunk_size
        )

        end = min(
            start
            + self.chunk_size,
            self.num_samples,
        )

        return (
            start,
            end,
        )

    def chunk_index_for(
        self,
        index: int,
    ) -> int:
        return (
            index
            // self.chunk_size
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        spectrum = (
            self.spectra[
                index
            ].copy()
        )

        mask = (
            self.masks[
                index
            ].copy()
        )

        target = spectrum[
            np.newaxis,
            :
        ].copy()

        axis = np.arange(
            spectrum.size,
            dtype=np.float32,
        )

        return {
            "input": np.stack(
                [
                    spectrum,
                    mask,
                ],
                axis=0,
            ),

            "spectrum": spectrum,

            "observation_mask": mask,

            "target": target,

            "axis": axis,

            "dataset_index": (
                index
            ),

            "sample_index": (
                index
            ),

            "domain_name": (
                self.domain_name
            ),
        }


class ToyMaskedDataset(
    ToyDataset
):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            num_samples=8,
            spectral_length=32,
            chunk_size=4,
        )

        self.masks[
            :,
            16:
        ] = 0.0

        self.spectra[
            :,
            16:
        ] = 0.0


class TinyCNN(
    nn.Module
):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.layer = nn.Conv1d(
            2,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        x,
    ):
        return self.layer(
            x
        )


class IdentitySpectrumModel(
    nn.Module
):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        # Dummy parameter allows standard model/device behavior.
        self.scale = nn.Parameter(
            torch.tensor(
                1.0
            )
        )

    def forward(
        self,
        x,
    ):
        return (
            x[
                :,
                0:1,
                :
            ]
            * self.scale
        )


def unit_stats(
    num_samples: int = 12,
) -> NormalizationStats:
    return NormalizationStats(
        spectrum_mean=0.0,
        spectrum_std=1.0,

        target_mean=0.0,
        target_std=1.0,

        num_samples=num_samples,

        num_observed_spectrum_values=(
            num_samples
            * 32
        ),

        num_target_values=(
            num_samples
            * 32
        ),
    )


def test_cpu_device_selection():
    device = select_device(
        "cpu"
    )

    assert (
        device.type
        == "cpu"
    )


def test_auto_device_is_supported():
    device = select_device(
        "auto"
    )

    assert (
        device.type
        in {
            "cpu",
            "cuda",
            "mps",
        }
    )


def test_seed_everything_reproduces_torch_values():
    seed_everything(
        101
    )

    first = torch.randn(
        8
    )

    seed_everything(
        101
    )

    second = torch.randn(
        8
    )

    assert torch.equal(
        first,
        second,
    )


def test_training_config_validation():
    with pytest.raises(
        ValueError
    ):
        TrainingConfig(
            batch_size=0
        )


def test_one_training_epoch_is_finite():
    dataset = ToyDataset()

    stats = unit_stats()

    seed_everything(
        101
    )

    model = TinyCNN()

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=1.0e-3,
        )
    )

    loss = train_one_epoch(
        model,
        dataset,
        stats,
        optimizer,

        device=torch.device(
            "cpu"
        ),

        batch_size=4,

        num_samples=len(
            dataset
        ),

        epoch_seed=101,

        gradient_clip_norm=1.0,
    )

    assert np.isfinite(
        loss
    )

    assert (
        loss
        >= 0.0
    )


def test_training_epoch_changes_parameters():
    dataset = ToyDataset()

    stats = unit_stats()

    seed_everything(
        101
    )

    model = TinyCNN()

    before = [
        parameter.detach()
        .clone()
        for parameter
        in model.parameters()
    ]

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=1.0e-2,
        )
    )

    train_one_epoch(
        model,
        dataset,
        stats,
        optimizer,

        device=torch.device(
            "cpu"
        ),

        batch_size=4,

        num_samples=len(
            dataset
        ),

        epoch_seed=101,
    )

    after = list(
        model.parameters()
    )

    assert any(
        not torch.equal(
            first,
            second,
        )
        for first, second
        in zip(
            before,
            after,
            strict=True,
        )
    )


def test_identity_model_has_zero_validation_mse():
    dataset = ToyDataset()

    stats = unit_stats()

    model = (
        IdentitySpectrumModel()
    )

    mse = evaluate_normalized_mse(
        model,
        dataset,
        stats,

        device=torch.device(
            "cpu"
        ),

        batch_size=5,
    )

    assert (
        mse
        == pytest.approx(
            0.0,
            abs=1.0e-12,
        )
    )


def test_identity_model_has_zero_denormalized_metrics():
    dataset = ToyDataset()

    stats = unit_stats()

    model = (
        IdentitySpectrumModel()
    )

    result = evaluate_model(
        model,
        dataset,
        stats,

        device=torch.device(
            "cpu"
        ),

        batch_size=5,
    )

    assert (
        result[
            "num_samples"
        ]
        == len(
            dataset
        )
    )

    assert (
        result[
            "metrics"
        ][
            "rmse"
        ][
            "mean"
        ]
        == pytest.approx(
            0.0,
            abs=1.0e-12,
        )
    )

    assert (
        result[
            "metrics"
        ][
            "mae"
        ][
            "mean"
        ]
        == pytest.approx(
            0.0,
            abs=1.0e-12,
        )
    )

    assert (
        result[
            "metrics"
        ][
            "spectral_angle"
        ][
            "mean"
        ]
        == pytest.approx(
            0.0,
            abs=1.0e-7,
        )
    )


def test_masked_dataset_reports_region_metrics():
    dataset = (
        ToyMaskedDataset()
    )

    stats = unit_stats(
        num_samples=8
    )

    model = (
        IdentitySpectrumModel()
    )

    result = evaluate_model(
        model,
        dataset,
        stats,

        device=torch.device(
            "cpu"
        ),

        batch_size=4,
    )

    assert (
        "spectral_regions"
        in result
    )

    assert (
        "observed"
        in result[
            "spectral_regions"
        ]
    )

    assert (
        "unobserved"
        in result[
            "spectral_regions"
        ]
    )


def test_early_stopping_tracks_best_model(
    tmp_path: Path,
):
    model = nn.Linear(
        1,
        1,
        bias=False,
    )

    checkpoint = (
        tmp_path
        / "best.pt"
    )

    stopping = EarlyStopping(
        patience=2,
        min_delta=0.01,
        checkpoint_path=checkpoint,
    )

    with torch.no_grad():
        model.weight.fill_(
            1.0
        )

    should_stop = stopping.update(
        1.0,
        epoch=0,
        model=model,
    )

    assert (
        should_stop
        is False
    )

    with torch.no_grad():
        model.weight.fill_(
            2.0
        )

    should_stop = stopping.update(
        0.8,
        epoch=1,
        model=model,
    )

    assert (
        should_stop
        is False
    )

    assert (
        stopping.best_epoch
        == 1
    )

    assert checkpoint.exists()

    # Two non-improving epochs.
    assert (
        stopping.update(
            0.805,
            epoch=2,
            model=model,
        )
        is False
    )

    assert (
        stopping.update(
            0.81,
            epoch=3,
            model=model,
        )
        is True
    )

    # Destroy current model weights.
    with torch.no_grad():
        model.weight.fill_(
            99.0
        )

    stopping.restore(
        model
    )

    assert (
        model.weight.item()
        == pytest.approx(
            2.0
        )
    )