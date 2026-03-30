"""
SE-ResNet-18: Squeeze-Excitation ResNet-18
──────────────────────────────────────────
Improvements over the baseline paper (plain ResNet-18):

  1. SE (Squeeze-Excitation) attention after every residual block
     → lets the network re-calibrate channel-wise feature responses
  2. Grayscale input adaptation via weight-averaged first conv layer
  3. Label-smoothed cross-entropy loss for better calibration
  4. Metadata MLP fusion hook (disabled for MedMNIST, ready for custom data)
  5. Dropout regularisation before final FC layers
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).
    Recalibrates channel-wise feature responses via global context.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SEResBlock(nn.Module):
    """Wraps a ResNet BasicBlock with a post-residual SE recalibration."""
    def __init__(self, block, channels: int):
        super().__init__()
        self.block = block
        self.se    = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.block(x))


class MetadataMLP(nn.Module):
    """Optional metadata encoder (age, gender → 32-d embedding)."""
    def __init__(self, in_dim: int = 2, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SEResNet18(nn.Module):
    """
    ResNet-18 adapted for 1-channel grayscale medical images.

    Architecture:
        Input (1×224×224)
        → Modified conv1 (1-channel, weight-averaged from ImageNet)
        → ResNet-18 stages with SE blocks (layer1–4)
        → AdaptiveAvgPool2d → (512-d)
        [optional] → concat MetadataMLP(meta) → (544-d)
        → Dropout → FC(128) → ReLU → Dropout → FC(n_classes)

    Attributes
    ----------
    gradcam_layer : the layer4 module, used as Grad-CAM target
    """

    def __init__(
        self,
        n_classes:    int   = 2,
        use_metadata: bool  = False,
        meta_dim:     int   = 2,
        dropout_p:    float = 0.3,
    ):
        super().__init__()
        self.use_metadata = use_metadata

        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Adapt first conv to accept 1-channel images.
        # Initialise by averaging the 3 RGB channels of the pretrained weights.
        old_conv = base.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool

        # Wrap each residual block in SE attention
        self.layer1 = self._add_se(base.layer1, 64)
        self.layer2 = self._add_se(base.layer2, 128)
        self.layer3 = self._add_se(base.layer3, 256)
        self.layer4 = self._add_se(base.layer4, 512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        feat_dim = 512
        if use_metadata:
            self.meta_mlp = MetadataMLP(meta_dim, 64, 32)
            feat_dim += 32

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p / 2),
            nn.Linear(128, n_classes),
        )

        # Exposed for Grad-CAM hooks
        self.gradcam_layer = self.layer4

    @staticmethod
    def _add_se(layer: nn.Sequential, channels: int) -> nn.Sequential:
        return nn.Sequential(*[SEResBlock(b, channels) for b in layer])

    def forward(
        self,
        x:    torch.Tensor,
        meta: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feat = self.avgpool(x).flatten(1)   # (B, 512)

        if self.use_metadata and meta is not None:
            feat = torch.cat([feat, self.meta_mlp(meta)], dim=1)

        return self.classifier(feat)        # (B, n_classes)


class LabelSmoothingCELoss(nn.Module):
    """
    Cross-entropy with label smoothing for better model calibration.
    Supports optional per-class weighting for imbalanced datasets.
    """
    def __init__(
        self,
        n_classes:  int,
        smoothing:  float = 0.1,
        weight:     torch.Tensor = None,
    ):
        super().__init__()
        self.smoothing   = smoothing
        self.n_classes   = n_classes
        self.weight      = weight
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = self.log_softmax(logits)

        # Build smooth target distribution
        smooth_val = self.smoothing / (self.n_classes - 1)
        soft = torch.full_like(log_probs, smooth_val)
        soft.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(soft * log_probs).sum(dim=1)

        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[targets]

        return loss.mean()


def build_model(
    n_classes:    int,
    device:       torch.device,
    use_metadata: bool = False,
) -> SEResNet18:
    model = SEResNet18(
        n_classes=n_classes,
        use_metadata=use_metadata,
        dropout_p=0.3,
    )
    return model.to(device)
