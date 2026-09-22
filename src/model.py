"""
Простая 3D CNN для бинарной классификации вокселей (concordant vs discordant)
по патчу структурной МРТ. Намеренно небольшая — по указанию руководителя
начать с простой модели и переходить к более глубокой (например,
residual/attention-сети) только если точность простой модели остаётся на
уровне случайного угадывания.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePatchCNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=8, n_classes=1):
        super().__init__()
        c = base_channels
        self.n_classes = n_classes
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
            nn.Linear(c * 2, n_classes),
        )

    def forward(self, x):
        """x: (N, 1, p, p, p) -> бинарные логиты (N,), если n_classes=1,
        иначе многоклассовые логиты (N, n_classes) - используется для
        3-классовой постановки concordant/discordant/unreliable."""
        x = self.features(x)
        out = self.classifier(x)
        return out.squeeze(-1) if self.n_classes == 1 else out


class DeeperPatchCNN(nn.Module):
    """Запасная архитектура (residual-блоки) для шага 3 плана, на случай если
    точность SimplePatchCNN остаётся на уровне случайного угадывания."""

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
    Сверточный энкодер + self-attention трансформера над пространственными
    токенами патча, затем классификационная голова - разумная адаптация
    идеи "U-Net с трансформером" (как в MS-DSA-NET, модели сегментации FCD
    из исходной статьи) к задаче бинарной метки на весь патч вместо плотной
    сегментации. Настоящий декодер U-Net восстанавливает выходную карту
    полного разрешения, что здесь не нужно (нужна одна метка на патч, а не
    повоксельная карта) - декодерная половина была бы лишними
    неиспользуемыми вычислениями и параметрами, поэтому она отброшена, и
    остаются только энкодер + attention + пулинг.

    Уменьшает разрешение патча p x p x p в 4 раза (свёртки со stride=2),
    превращает оставшуюся пространственную сетку в токены и позволяет
    небольшому трансформер-энкодеру смешивать информацию по всему патчу
    (глобальное рецептивное поле) перед пулингом в единый вектор.
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
        grid = -(-patch_size // 4)  # ceil(patch_size / 4), пространственный размер после двух сверток со stride=2
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
        """x: (N, 1, p, p, p) -> логиты (N,)"""
        feats = self.stem(x)                          # (N, token_dim, g, g, g)
        n, c = feats.shape[0], feats.shape[1]
        tokens = feats.flatten(2).transpose(1, 2)      # (N, n_tokens, token_dim)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)                    # (N, token_dim)
        return self.classifier(pooled).squeeze(-1)


class PatchUNet(nn.Module):
    """
    Настоящая U-Net: сверточный энкодер-декодер со skip-соединениями,
    обучаемый выдавать плотную повоксельную карту по всему патчу, а не
    просто один усреднённый вектор, как остальные архитектуры здесь.

    Это подлинное архитектурное отличие от SimplePatchCNN/DeeperPatchCNN/
    AttentionPatchCNN, которые все сворачивают патч в вектор признаков
    прежде, чем делать какое-либо пространственное предсказание. U-Net же
    предсказывает плотно и позволяет skip-соединениям декодера
    пересобирать тонкие (ранние слои) и грубые (bottleneck) пространственные
    детали в каждой выходной точке - именно для этого и нужны U-Net.

    Здесь задача - одна метка на патч (метка принадлежит вокселю в центре
    патча), а не карта сегментации, поэтому нет плотной разметки для
    обучения большей части выхода. Решается тем, что предсказание
    декодера В ЦЕНТРАЛЬНОМ ВОКСЕЛЕ читается как классификационный логит -
    обучение и оценка смотрят только на эту одну точку, так что это
    по-прежнему настоящая классификационная модель, просто получающая
    ответ через плотную реконструкцию со skip-соединениями, а не через
    глобальный пулинг. Для upsampling используется F.interpolate (а не
    транспонированная свёртка со stride), чтобы нечётные размеры патча
    (9, 15, ...) работали без несовпадения форм со skip-признаками энкодера.
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
        """x: (N, 1, p, p, p) -> логиты (N,), считанные из центрального вокселя выходной карты."""
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
    Эксперимент 2: структурный патч + BOLD-временной ряд по вокселю, две
    ветви объединяются перед классификационной головой.

    Структурная ветвь: тот же небольшой сверточный ствол, что и в
    SimplePatchCNN.
    BOLD-ветвь: стек 1D-сверток по оси времени (временной ряд - это иной
    тип сигнала, чем 3D-патч: локальные временные паттерны, а не
    пространственные окрестности, поэтому уместна именно 1D-свертка по
    времени, а не ещё одна 3D-свертка), затем глобальный average pooling.
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
        """patch: (N, 1, p, p, p). bold_vec: (N, T). -> логиты (N,)"""
        patch_feat = self.patch_branch(patch)
        bold_feat = self.bold_branch(bold_vec.unsqueeze(1))
        combined = torch.cat([patch_feat, bold_feat], dim=1)
        return self.classifier(combined).squeeze(-1)


class PatchBOLDConditionNet(nn.Module):
    """
    Та же идея, что и в PatchBOLDNet, но для компактного вектора
    BOLD-признаков по условию (src/bold_features.py:compute_condition_features
    - процентное изменение сигнала по каждому условию задачи, несколько
    чисел) вместо полного сырого временного ряда. Небольшой MLP - уместный
    выбор для короткого, уже агрегированного вектора признаков - 1D-свертке
    (созданной для поиска паттернов *вдоль* последовательности) здесь
    делать нечего.
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
        """patch: (N, 1, p, p, p). bold_feat: (N, n_bold_features). -> логиты (N,)"""
        patch_feat = self.patch_branch(patch)
        bold_feat = self.bold_branch(bold_feat)
        combined = torch.cat([patch_feat, bold_feat], dim=1)
        return self.classifier(combined).squeeze(-1)


class PretrainedFeatureHead(nn.Module):
    """Небольшой MLP-классификатор поверх заранее извлечённых признаков ствола
    MedicalNet ResNet50 (2048-мерных, после global-average-pooling) -
    обучается заново под нашу задачу, сам ствол остаётся замороженным
    (извлечение признаков - отдельный, разовый шаг, см.
    src/medicalnet_resnet.extract_backbone_features).
    Подключается в src/train.py:train_one_fold как любой другой
    model_factory: x имеет вид (N, 2048) вместо (N, 1, p, p, p), всё
    остальное общее."""

    def __init__(self, in_dim=2048, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RegionMLP(nn.Module):
    """Небольшой MLP над простым вектором признаков - используется, когда
    единицей классификации выступает целый анатомический регион (например,
    реальный parcel атласа Glasser, усреднённый по ROI), а не патч
    вокселей, так что пространственного патча для 3D CNN здесь в принципе
    нет. Тот же общий интерфейс (N, dim) -> (N,) логитов, что и у
    PretrainedFeatureHead, но названный отдельно, так как на входе здесь
    несколько сводных статистик по ROI, а не глубокие признаки backbone.
    n_classes=1 даёт бинарные логиты (как и везде в проекте); n_classes>1
    даёт многоклассовые логиты (используется для 3-классовой постановки
    concordant/discordant/unreliable)."""

    def __init__(self, in_dim=3, hidden=32, n_classes=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes),
        )
        self.n_classes = n_classes

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1) if self.n_classes == 1 else out
