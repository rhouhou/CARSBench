from __future__ import annotations

import pytest

torch = pytest.importorskip(
    "torch"
)

from CARSBench.ml import (
    CARSBaselineCNN,
    ResidualBlock1D,
    count_trainable_parameters,
)


def test_residual_block_preserves_shape():
    block = ResidualBlock1D(
        channels=32,
        kernel_size=7,
    )

    x = torch.randn(
        4,
        32,
        1024,
    )

    y = block(
        x
    )

    assert (
        y.shape
        == x.shape
    )


def test_baseline_output_shape():
    model = CARSBaselineCNN()

    x = torch.randn(
        8,
        2,
        1024,
    )

    y = model(
        x
    )

    assert (
        y.shape
        == (
            8,
            1,
            1024,
        )
    )


def test_baseline_handles_single_sample():
    model = CARSBaselineCNN()

    x = torch.randn(
        1,
        2,
        1024,
    )

    y = model(
        x
    )

    assert (
        y.shape
        == (
            1,
            1,
            1024,
        )
    )


def test_baseline_preserves_arbitrary_spectral_length():
    model = CARSBaselineCNN()

    x = torch.randn(
        3,
        2,
        257,
    )

    y = model(
        x
    )

    assert (
        y.shape
        == (
            3,
            1,
            257,
        )
    )


def test_model_output_is_finite():
    model = CARSBaselineCNN()

    x = torch.randn(
        4,
        2,
        1024,
    )

    with torch.no_grad():
        y = model(
            x
        )

    assert torch.all(
        torch.isfinite(
            y
        )
    )


def test_model_has_reasonable_parameter_count():
    model = CARSBaselineCNN(
        hidden_channels=64,
        num_blocks=6,
    )

    n_parameters = (
        count_trainable_parameters(
            model
        )
    )

    assert (
        n_parameters
        > 10000
    )

    assert (
        n_parameters
        < 5_000_000
    )


def test_same_initialization_seed_is_reproducible():
    torch.manual_seed(
        101
    )

    model_a = (
        CARSBaselineCNN()
    )

    torch.manual_seed(
        101
    )

    model_b = (
        CARSBaselineCNN()
    )

    params_a = list(
        model_a.parameters()
    )

    params_b = list(
        model_b.parameters()
    )

    assert (
        len(params_a)
        == len(params_b)
    )

    for first, second in zip(
        params_a,
        params_b,
        strict=True,
    ):
        assert torch.equal(
            first,
            second,
        )


def test_different_initialization_seeds_change_weights():
    torch.manual_seed(
        101
    )

    model_a = (
        CARSBaselineCNN()
    )

    torch.manual_seed(
        202
    )

    model_b = (
        CARSBaselineCNN()
    )

    differences = [
        not torch.equal(
            first,
            second,
        )
        for first, second in zip(
            model_a.parameters(),
            model_b.parameters(),
            strict=True,
        )
    ]

    assert any(
        differences
    )


def test_invalid_number_of_input_channels_is_rejected():
    model = CARSBaselineCNN()

    x = torch.randn(
        4,
        1,
        1024,
    )

    with pytest.raises(
        ValueError
    ):
        model(
            x
        )


def test_non_3d_input_is_rejected():
    model = CARSBaselineCNN()

    x = torch.randn(
        2,
        1024,
    )

    with pytest.raises(
        ValueError
    ):
        model(
            x
        )


def test_even_kernel_size_is_rejected():
    with pytest.raises(
        ValueError
    ):
        CARSBaselineCNN(
            kernel_size=6
        )