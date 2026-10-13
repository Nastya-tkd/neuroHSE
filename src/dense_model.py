"""
Более прямой (но всё ещё облегчённый) перенос MS-DSA-NET
(link.springer.com/article/10.1186/s13244-024-01803-8, модель сегментации
очагов ФКД) на нашу задачу: в отличие от src/model.py::MultiScaleAttentionPatchCNN
(которая по-прежнему классифицирует один патч 9^3), здесь сеть - настоящий
encoder-decoder, выдающий ПЛОТНУЮ карту concordant/discordant на весь объём
мозга за один проход, как и в оригинальной статье.

Чем это сознательно отличается от оригинала и почему:
  - Без MONAI, без MNI152, без FSL. Оригинал регистрирует всех пациентов в
    общий шаблон MNI152 через внешний бинарник FSL - у нас для этой задачи
    такая регистрация не нужна: каждый пациент обучается/оценивается в
    собственном пространстве (T2-пространство, общее для всех пациентов
    только по форме массива после паддинга, см. scripts/run_dense_segmentation.py),
    поскольку плотная метка (в отличие от их общей lesion-вероятностной
    карты) у каждого пациента своя и не требует общего анатомического
    пространства для обучения.
  - Один трансформер-мост на боттлнеке, а не четыре параллельных пути на
    разных масштабах, как в оригинале. На CPU self-attention по всем
    пространственным токенам целого объёма мозга (даже после троекратного
    пулинга) заметно дороже, чем на патче 9^3 - четыре моста на разных
    масштабах (как в оригинале или как в облегчённой версии для патчей,
    MultiScaleAttentionPatchCNN) здесь экономически не оправданы без GPU;
    если один мост на боттлнеке покажет хоть какой-то сигнал, можно
    добавить более мелкие масштабы отдельным экспериментом.
  - in_channels=1 (T1), out_channels=1 (один логит на воксель: >0 =
    concordant, <=0 = discordant) - у нас нет отдельного фона/очага, как у
    них (lesion/no-lesion), наша метка определена на каждом размеченном
    вокселе мозга, поэтому бинарный логит на воксель достаточен (как и в
    остальных частях проекта, где воксельная метка - {0,1}, см.
    src/train.py:select_labeled_coords).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, kernel_size=3, padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DenseConcordanceNet(nn.Module):
    """
    Encoder-decoder (3 уровня пулинга + боттлнек) со skip-соединениями, как
    в src/model.py::PatchUNet, но с self-attention трансформером на
    боттлнеке (аналог моста MS-DSA-NET) перед декодером - даёт каждому
    выходному вокселю доступ к глобальному контексту всей ВИДИМОЙ стороны
    объёма (видимость контролируется обнулением входа, см.
    src/patches.py::side_mask_volume и scripts/run_dense_segmentation.py),
    а не только к локальной окрестности, как в патчевых моделях.

    MaxPool3d(..., ceil_mode=True) и F.interpolate для upsampling - та же
    причина, что и в PatchUNet: форма входа (112, 112, 35 после паддинга)
    не делится на 8 без остатка, целочисленное округление в большую
    сторону и явный target size при upsampling избегают несовпадения форм
    со skip-признаками энкодера.
    """

    def __init__(self, img_size, in_channels=1, out_channels=1, base_channels=8, n_heads=4, n_layers=1):
        """img_size: (D, H, W) формы ВХОДА - нужна заранее (а не выводится
        лениво при первом forward), чтобы позиционное вложение боттлнека
        было обычным nn.Parameter, зарегистрированным до создания
        оптимизатора (лениво созданный параметр молча не попал бы в
        model.parameters(), собранные раньше первого forward)."""
        super().__init__()
        c = base_channels
        self.enc1 = _ConvBlock(in_channels, c)
        self.enc2 = _ConvBlock(c, c * 2)
        self.enc3 = _ConvBlock(c * 2, c * 4)
        self.bottleneck = _ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool3d(2, ceil_mode=True)

        self.token_dim = c * 8
        bottleneck_grid = list(img_size)
        for _ in range(3):  # три пулинга MaxPool3d(2, ceil_mode=True) -> ceil(d/2) на каждом уровне
            bottleneck_grid = [-(-d // 2) for d in bottleneck_grid]
        n_tokens = bottleneck_grid[0] * bottleneck_grid[1] * bottleneck_grid[2]
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, self.token_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim, nhead=n_heads, dim_feedforward=self.token_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dec3 = _ConvBlock(c * 8 + c * 4, c * 4)
        self.dec2 = _ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = _ConvBlock(c * 2 + c, c)
        self.out_conv = nn.Conv3d(c, out_channels, kernel_size=1)

    def _bottleneck_transformer(self, b):
        n, c, d, h, w = b.shape
        tokens = b.flatten(2).transpose(1, 2)  # (N, n_tokens, c)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        return tokens.transpose(1, 2).reshape(n, c, d, h, w)

    def forward(self, x):
        """x: (N, 1, D, H, W) -> логиты (N, 1, D, H, W), та же форма, что вход."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        b = self._bottleneck_transformer(b)

        up3 = F.interpolate(b, size=e3.shape[2:], mode="trilinear", align_corners=False)
        d3 = self.dec3(torch.cat([up3, e3], dim=1))

        up2 = F.interpolate(d3, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))

        up1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        return self.out_conv(d1)
