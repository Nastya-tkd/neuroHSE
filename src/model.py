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


class AttentionPatchCNN(nn.Module):
    """
    Conv encoder + transformer self-attention over the patch's spatial
    tokens, then a classification head - the sensible adaptation of a
    "U-Net with transformer" (like MS-DSA-NET, the FCD-segmentation model
    from the original article) to a per-patch binary label instead of dense
    segmentation. A real U-Net decoder reconstructs a full-resolution output
    map, which nothing here needs (we want one label per patch, not a
    voxel-wise map) - the decoder half would just be extra unused compute
    and parameters, so it's dropped and only the encoder + attention +
    pooling stays.

    Downsamples the p x p x p patch by 4x (stride-2 convs), flattens the
    remaining spatial grid into tokens, and lets a small transformer encoder
    mix information across the whole patch (global receptive field) before
    pooling to a single vector.
    """

    def __init__(self, in_channels=1, base_channels=16, n_heads=4, n_layers=2, patch_size=9):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=3, padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
            nn.Conv3d(c, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
            nn.Conv3d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(c * 2),
            nn.ReLU(inplace=True),
        )
        token_dim = c * 2
        grid = -(-patch_size // 4)  # ceil(patch_size / 4), spatial size after two stride-2 convs
        n_tokens = grid ** 3
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, token_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=n_heads, dim_feedforward=token_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(token_dim, 1)
        )

    def forward(self, x):
        """x: (N, 1, p, p, p) -> logits (N,)"""
        feats = self.stem(x)                          # (N, token_dim, g, g, g)
        n, c = feats.shape[0], feats.shape[1]
        tokens = feats.flatten(2).transpose(1, 2)      # (N, n_tokens, token_dim)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)                    # (N, token_dim)
        return self.classifier(pooled).squeeze(-1)
