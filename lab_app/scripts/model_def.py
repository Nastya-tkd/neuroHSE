"""A compact 2D U-Net sized for CPU training and inference on a small
preclinical MRI dataset. Not the paper's nnU-Net/EfficientNet-B0 baselines
(those assume GPU + a much larger training budget) -- this is a lab-ready
baseline tuned to run on an ordinary Windows laptop with no GPU.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """4-level U-Net, ~1.9M parameters. base=24 by default."""

    def __init__(self, in_ch=1, num_classes=4, base=24):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16

        self.enc1 = conv_block(in_ch, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.enc4 = conv_block(c3, c4)
        self.bottleneck = conv_block(c4, c5)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec4 = conv_block(c5, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = conv_block(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = conv_block(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = conv_block(c2, c1)

        self.head = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = SmallUNet()
    x = torch.zeros(1, 1, 192, 192)
    y = m(x)
    print("output shape:", y.shape, "params:", count_params(m))
