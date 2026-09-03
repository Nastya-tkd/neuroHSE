"""
Simple 3D CNN for binary voxel classification (concordant vs discordant)
from a structural-MRI patch. Deliberately small, per the supervisor's
instruction to start simple and only go deeper (e.g. a residual/attention
net) if accuracy on the simple model stays near chance.
"""

import torch
import torch.nn as nn


class SimplePatchCNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=8):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=3, padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(c, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(c * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 2, c * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(c * 2, 1),
        )

    def forward(self, x):
        """x: (N, 1, p, p, p) -> logits (N,)"""
        x = self.features(x)
        return self.classifier(x).squeeze(-1)


class DeeperPatchCNN(nn.Module):
    """Fallback architecture (residual blocks) for step 3 of the plan, if
    SimplePatchCNN stays near chance accuracy."""

    class ResBlock(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv1 = nn.Conv3d(c, c, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(c)
            self.conv2 = nn.Conv3d(c, c, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm3d(c)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + x)

    def __init__(self, in_channels=1, base_channels=16, n_blocks=3):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=3, padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[self.ResBlock(c) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(c, c), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(c, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(-1)
