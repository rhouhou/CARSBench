from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    """
    Simple residual block preserving spectral length.
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels must be positive."
            )

        if kernel_size <= 0:
            raise ValueError(
                "kernel_size must be positive."
            )

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd so spectral "
                "length is preserved."
            )

        padding = (
            kernel_size // 2
        )

        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
            ),

            nn.GELU(),

            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )

        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.activation(
            x
            + self.block(
                x
            )
        )


class CARSBaselineCNN(nn.Module):
    """
    Simple 1-D residual CNN baseline for CARS-to-Raman retrieval.

    Input
    -----
    Tensor of shape:

        (batch, 2, spectral_length)

    Channels:

        0: measured CARS/BCARS spectrum
        1: observation mask

    Output
    ------
    Tensor of shape:

        (batch, 1, spectral_length)

    representing predicted Im(chi_R).
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be positive."
            )

        if hidden_channels <= 0:
            raise ValueError(
                "hidden_channels must be positive."
            )

        if num_blocks <= 0:
            raise ValueError(
                "num_blocks must be positive."
            )

        if kernel_size <= 0:
            raise ValueError(
                "kernel_size must be positive."
            )

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd."
            )

        padding = (
            kernel_size // 2
        )

        self.input_projection = nn.Sequential(
            nn.Conv1d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GELU(),
        )

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock1D(
                    hidden_channels,
                    kernel_size=kernel_size,
                )
                for _ in range(
                    num_blocks
                )
            ]
        )

        self.output_projection = nn.Conv1d(
            hidden_channels,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch, channels, spectral_length), "
                f"received {tuple(x.shape)}."
            )

        if (
            x.shape[1]
            != 2
        ):
            raise ValueError(
                "CARSBaselineCNN expects exactly "
                "2 input channels: spectrum and mask."
            )

        features = (
            self.input_projection(
                x
            )
        )

        features = (
            self.residual_blocks(
                features
            )
        )

        prediction = (
            self.output_projection(
                features
            )
        )

        return prediction


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Number of parameters updated during training.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )