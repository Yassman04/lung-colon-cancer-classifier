"""The CNN architecture.

Three conv blocks pick up tissue patterns at increasing scale (texture -> cell clusters ->
tissue structure), then a small two-layer head turns those features into 5 class scores.
"""

import torch
import torch.nn as nn

from . import CLASS_NAMES


def conv_block(in_ch: int, out_ch: int, spatial_dropout: float) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    ]
    if spatial_dropout > 0:
        layers.append(nn.Dropout2d(spatial_dropout))
    return nn.Sequential(*layers)


class CancerCNN(nn.Module):
    """Conv(3->16) -> Conv(16->32) -> Conv(32->64) -> FC(128) -> FC(5).

    Dropout is off by default: in testing it caused underfitting (val accuracy fell from
    95.9% to ~91.5%), because this small network needs all of its capacity. It can be
    switched back on with `dropout=` to reproduce that experiment.
    """

    def __init__(self, num_classes: int = len(CLASS_NAMES), image_size: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        spatial = dropout / 2 if dropout else 0.0
        self.features = nn.Sequential(
            conv_block(3, 16, spatial),
            conv_block(16, 32, spatial),
            conv_block(32, 64, spatial),
        )
        feature_size = 64 * (image_size // 8) ** 2  # three 2x2 pools: size / 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))  # raw logits; CrossEntropyLoss applies softmax


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
