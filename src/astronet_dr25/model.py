"""AstroNet local/global 1-D CNN, matching the Google configuration."""

from __future__ import annotations

import torch
from torch import nn


GLOBAL_LENGTH = 2001
LOCAL_LENGTH = 201
OFFICIAL_PARAMETER_COUNT = 9_940_193


def _conv_column(
    blocks: int,
    pool_size: int,
    initial_filters: int = 16,
    kernel_size: int = 5,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 1
    for block in range(blocks):
        out_channels = initial_filters * 2**block
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        # TensorFlow Keras MaxPool1D defaults to padding="valid".
        layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=2))
    return nn.Sequential(*layers)


class AstroNet(nn.Module):
    """Official 2001-point global + 201-point local AstroNet architecture.

    The network returns logits.  Use BCEWithLogitsLoss during training and
    torch.sigmoid(logits) for probabilities.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.global_column = _conv_column(blocks=5, pool_size=5)
        self.local_column = _conv_column(blocks=2, pool_size=7)

        # Official output sizes: global 59*256; local 46*32; total 16576.
        self.flattened_features = 16_576
        dense: list[nn.Module] = []
        in_features = self.flattened_features
        for _ in range(4):
            dense.append(nn.Linear(in_features, 512))
            dense.append(nn.ReLU())
            if dropout > 0:
                dense.append(nn.Dropout(dropout))
            in_features = 512
        dense.append(nn.Linear(512, 1))
        self.classifier = nn.Sequential(*dense)

    def forward(self, local_view: torch.Tensor, global_view: torch.Tensor) -> torch.Tensor:
        if local_view.ndim == 2:
            local_view = local_view.unsqueeze(1)
        if global_view.ndim == 2:
            global_view = global_view.unsqueeze(1)
        if local_view.shape[-1] != LOCAL_LENGTH:
            raise ValueError(f"Expected local length {LOCAL_LENGTH}, got {local_view.shape}")
        if global_view.shape[-1] != GLOBAL_LENGTH:
            raise ValueError(f"Expected global length {GLOBAL_LENGTH}, got {global_view.shape}")

        local_hidden = self.local_column(local_view).flatten(start_dim=1)
        global_hidden = self.global_column(global_view).flatten(start_dim=1)
        hidden = torch.cat([global_hidden, local_hidden], dim=1)
        if hidden.shape[1] != self.flattened_features:
            raise RuntimeError(
                f"Flattened feature mismatch: {hidden.shape[1]} != {self.flattened_features}"
            )
        return self.classifier(hidden).squeeze(1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
