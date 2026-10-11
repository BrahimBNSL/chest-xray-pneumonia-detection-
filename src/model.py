"""Construction du modèle : backbone ResNet pré-entraîné et tête de classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from src.config import ModelConfig
from src.utils import get_logger

logger = get_logger(__name__)

HEAD_DROPOUT: float = 0.3

BACKBONES: dict[str, tuple] = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}


def build_model(cfg: ModelConfig, unfreeze_last_block: bool) -> nn.Module:
    """Construit le modèle : backbone pré-entraîné, tête de classification, gel sélectif."""
    constructor, weights = BACKBONES[cfg.backbone]
    model = constructor(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=HEAD_DROPOUT),
        nn.Linear(in_features, cfg.num_classes),
    )

    if cfg.freeze_backbone:
        freeze_backbone(model, unfreeze_last_block)

    trainable, total = count_parameters(model)
    logger.info(
        "Modèle %s : %s paramètres entraînables sur %s (dernier bloc dégelé : %s)",
        cfg.backbone, f"{trainable:,}", f"{total:,}", unfreeze_last_block,
    )
    return model


def freeze_backbone(model: nn.Module, unfreeze_last_block: bool) -> None:
    """Gèle le backbone, sauf le dernier bloc si demandé. La tête reste toujours entraînable."""
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = unfreeze_last_block and name.startswith("layer4.")


def set_train_mode(model: nn.Module) -> None:
    """Passe le modèle en mode entraînement, en gardant les couches BatchNorm gelées en mode évaluation."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            has_trainable = any(p.requires_grad for p in module.parameters(recurse=False))
            if not has_trainable:
                module.eval()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Retourne le nombre de paramètres entraînables et le nombre total de paramètres."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total