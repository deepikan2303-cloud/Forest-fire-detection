"""Model zoo: transfer-learning backbones from TorchVision plus a small CNN trained from scratch."""
import warnings

import torch.nn as nn
from torchvision import models

ARCHS = ["mobilenet_v3_small", "mobilenet_v3_large", "resnet18", "efficientnet_b0", "simple_cnn"]


class SimpleCNN(nn.Module):
    """Lightweight VGG-style baseline (~1.2M params) for training without pretrained weights."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def _load_tv(fn, weights, pretrained: bool):
    if not pretrained:
        return fn(weights=None)
    try:
        return fn(weights=weights)
    except Exception as exc:  # offline, proxy, etc.
        warnings.warn(f"Could not download pretrained weights ({exc}); using random init.")
        return fn(weights=None)


def build_model(arch: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    if arch == "simple_cnn":
        return SimpleCNN(num_classes)
    if arch == "mobilenet_v3_small":
        m = _load_tv(models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT, pretrained)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        m = _load_tv(models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT, pretrained)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "resnet18":
        m = _load_tv(models.resnet18, models.ResNet18_Weights.DEFAULT, pretrained)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = _load_tv(models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, pretrained)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {ARCHS}")
    return m


def head_module(model: nn.Module, arch: str) -> nn.Module:
    """The classification head (kept trainable while the backbone is frozen)."""
    return model.fc if arch == "resnet18" else model.classifier


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    head_ids = {id(p) for p in head_module(model, arch).parameters()}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


def count_params(model: nn.Module) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
