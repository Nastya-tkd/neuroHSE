"""
Simple 3D CNN for binary voxel classification (concordant vs discordant)
from a structural-MRI patch. Deliberately small, per the supervisor's
instruction to start simple and only go deeper (e.g. a residual/attention
net) if accuracy on the simple model stays near chance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PatchUNet(nn.Module):
    """
    A real U-Net: convolutional encoder-decoder with skip connections,
    trained to output a dense per-voxel map over the whole patch, not just
    a single pooled vector like the other architectures here.

    This is a genuine architectural difference from SimplePatchCNN/
    DeeperPatchCNN/AttentionPatchCNN, all of which collapse the patch to a
    feature vector before ever making a spatial prediction. A U-Net instead
    predicts densely and lets the decoder's skip connections recombine
    fine (early-layer) and coarse (bottleneck) spatial detail at every
    output location - the thing U-Nets are for.

    The task here is one label per patch (the label belongs to the voxel
    at the patch's center), not a segmentation map, so there's no dense
    ground truth to supervise most of the output with. Resolved by reading
    off the decoder's prediction AT THE CENTER VOXEL as the classification
    logit - training and evaluation only ever look at that one location,
    so this is still a real classification model, just one that reaches
    its answer through a dense, skip-connected reconstruction instead of
    global pooling. Uses F.interpolate (not transposed-conv striding) for
    upsampling so odd patch sizes (9, 15, ...) work without shape mismatches
    against the encoder skip features.
    """

    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        c = base_channels

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=3, padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
                nn.Conv3d(cout, cout, kernel_size=3, padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, c)
        self.enc2 = conv_block(c, c * 2)
        self.bottleneck = conv_block(c * 2, c * 4)

        self.dec2 = conv_block(c * 4 + c * 2, c * 2)
        self.dec1 = conv_block(c * 2 + c, c)

        self.pool = nn.MaxPool3d(2, ceil_mode=True)
        self.out_conv = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x):
        """x: (N, 1, p, p, p) -> logits (N,), read from the output map's center voxel."""
        e1 = self.enc1(x)                              # (N, c,   p,  p,  p)
        e2 = self.enc2(self.pool(e1))                   # (N, 2c, p/2,p/2,p/2)
        b = self.bottleneck(self.pool(e2))              # (N, 4c, p/4,p/4,p/4)

        up2 = F.interpolate(b, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))     # (N, 2c, p/2,p/2,p/2)

        up1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))     # (N, c, p, p, p)

        out_map = self.out_conv(d1)                     # (N, 1, p, p, p)
        p = out_map.shape[-1]
        center = p // 2
        return out_map[:, 0, center, center, center]    # (N,)


class PatchBOLDNet(nn.Module):
    """
    Experiment 2: structural patch + per-voxel BOLD time series, two
    branches concatenated before the classifier head.

    Structural branch: same small conv trunk as SimplePatchCNN.
    BOLD branch: 1D conv stack over the time axis (a time series is a
    different kind of signal than a 3D patch - local temporal patterns,
    not spatial neighborhoods - so a 1D conv over time, not another 3D
    conv, is the appropriate match), then global average pooled.
    """

    def __init__(self, patch_channels=1, patch_base=8, bold_base=16, bold_len=400):
        super().__init__()
        c = patch_base
        self.patch_branch = nn.Sequential(
            nn.Conv3d(patch_channels, c, kernel_size=3, padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(c, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(c * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        patch_feat_dim = c * 2

        bc = bold_base
        self.bold_branch = nn.Sequential(
            nn.Conv1d(1, bc, kernel_size=7, padding=3),
            nn.BatchNorm1d(bc),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(bc, bc * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(bc * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        bold_feat_dim = bc * 2

        combined_dim = patch_feat_dim + bold_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(combined_dim, 1),
        )

    def forward(self, patch, bold_vec):
        """patch: (N, 1, p, p, p). bold_vec: (N, T). -> logits (N,)"""
        patch_feat = self.patch_branch(patch)
        bold_feat = self.bold_branch(bold_vec.unsqueeze(1))
        combined = torch.cat([patch_feat, bold_feat], dim=1)
        return self.classifier(combined).squeeze(-1)


class PatchBOLDConditionNet(nn.Module):
    """
    Same idea as PatchBOLDNet, but for a compact per-condition BOLD feature
    vector (src/bold_features.py:compute_condition_features - percent
    signal change per task condition, a handful of numbers) instead of the
    full raw time series. A small MLP is the appropriate match for a short,
    already-summarized feature vector - a 1D conv (built for finding
    patterns *along* a sequence) has nothing to do here.
    """

    def __init__(self, patch_channels=1, patch_base=8, n_bold_features=3, bold_hidden=16):
        super().__init__()
        c = patch_base
        self.patch_branch = nn.Sequential(
            nn.Conv3d(patch_channels, c, kernel_size=3, padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(c, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(c * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        patch_feat_dim = c * 2

        self.bold_branch = nn.Sequential(
            nn.Linear(n_bold_features, bold_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(bold_hidden, bold_hidden),
            nn.ReLU(inplace=True),
        )

        combined_dim = patch_feat_dim + bold_hidden
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(combined_dim, 1),
        )

    def forward(self, patch, bold_feat):
        """patch: (N, 1, p, p, p). bold_feat: (N, n_bold_features). -> logits (N,)"""
        patch_feat = self.patch_branch(patch)
        bold_feat = self.bold_branch(bold_feat)
        combined = torch.cat([patch_feat, bold_feat], dim=1)
        return self.classifier(combined).squeeze(-1)
