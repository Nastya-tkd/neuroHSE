"""
3D ResNet trunk matching Tencent/MedicalNet's architecture (Chen et al.,
"Med3D: Transfer Learning for 3D Medical Image Analysis"), reproduced here
(trunk only - conv1/bn1/relu/maxpool/layer1-4, no conv_seg segmentation
head, since the released checkpoint doesn't include one) so the pretrained
resnet_50_23dataset.pth checkpoint's state_dict loads with an exact key
match. Dilated (not strided) layer3/layer4 by design - MedicalNet keeps
spatial resolution higher than a standard ImageNet ResNet for segmentation,
so a P-voxel input patch only downsamples ~8x (conv1 stride2 x maxpool
stride2 x layer2 stride2) rather than the usual 32x.

conv1 takes 1 input channel already - single-channel T1 patches need no
adaptation, unlike a typical 3-channel ImageNet backbone.
"""

import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, dilation=dilation,
                      stride=stride, padding=dilation, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                                dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class MedicalNetResNetTrunk(nn.Module):
    """Bottleneck-ResNet50 trunk, output: (N, 2048, d, h, w) feature map."""

    def __init__(self, layers=(3, 4, 6, 3)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(512, layers[3], stride=1, dilation=4)

    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def load_pretrained_trunk(checkpoint_path):
    import torch
    model = MedicalNetResNetTrunk(layers=(3, 4, 6, 3))
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt["state_dict"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_backbone_features(backbone, patches, batch_size=32):
    """patches: (N, 1, p, p, p) float32 numpy array (already intensity-
    normalized, same as any other patch input in this repo). Runs the
    frozen trunk + global-average-pool in eval/no_grad mode (never trained,
    never sees gradients - only src.model.PretrainedFeatureHead trains) and
    returns (N, 2048) float32 numpy features."""
    import torch
    import numpy as np

    backbone.eval()
    pool = torch.nn.AdaptiveAvgPool3d(1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            xb = torch.from_numpy(patches[i:i + batch_size])
            out = backbone(xb)
            out = pool(out).flatten(1)
            feats.append(out.numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)
