"""Ensemble de plusieurs entraînements du fine-tuning : moyenne des probabilités sur le test."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import OUTPUT_DIR, DataConfig, ModelConfig
from src.data import build_dataloaders
from src.features import evaluate
from src.model import build_model
from src.train import checkpoint_path, predict
from src.utils import get_device, get_logger, save_json

logger = get_logger(__name__)

SEEDS: tuple[int, ...] = (42, 1, 2)
ENSEMBLE_PATH = OUTPUT_DIR / "metrics" / "ensemble_resnet50.json"


def load_model(seed: int, device: torch.device) -> nn.Module:
    """Charge le meilleur modèle sauvegardé pour une graine donnée."""
    model = build_model(ModelConfig(backbone="resnet50", freeze_backbone=False), unfreeze_last_block=False)
    model.load_state_dict(torch.load(checkpoint_path(seed), map_location=device, weights_only=True))
    return model.to(device)


def run(seeds: tuple[int, ...] = SEEDS) -> dict:
    """Évalue chaque graine, puis la moyenne des probabilités, sur le même test."""
    device = get_device()
    bundle = build_dataloaders(DataConfig())
    criterion = nn.CrossEntropyLoss()

    par_graine: dict[str, dict] = {}
    probabilites: list[np.ndarray] = []
    labels: np.ndarray | None = None

    for seed in seeds:
        model = load_model(seed, device)
        _, probs, y = predict(model, bundle.test, criterion, device)
        if labels is None:
            labels = y
        elif not np.array_equal(labels, y):
            raise ValueError("Les étiquettes du test diffèrent entre les modèles")
        probabilites.append(probs)
        par_graine[f"graine_{seed}"] = evaluate(probs, y)
        logger.info(
            "Graine %d : balanced accuracy %.4f, ROC AUC %.4f",
            seed, par_graine[f"graine_{seed}"]["balanced_accuracy"], par_graine[f"graine_{seed}"]["roc_auc"],
        )

    ensemble = evaluate(np.mean(probabilites, axis=0), labels)
    scores = [m["balanced_accuracy"] for m in par_graine.values()]
    resume = {
        "graines": list(seeds),
        "balanced_accuracy_moyenne": float(np.mean(scores)),
        "balanced_accuracy_ecart_type": float(np.std(scores, ddof=1)),
        "ensemble": ensemble,
        "par_graine": par_graine,
    }
    save_json(resume, ENSEMBLE_PATH)
    logger.info(
        "Moyenne sur %d graines : balanced accuracy %.4f ± %.4f",
        len(seeds), resume["balanced_accuracy_moyenne"], resume["balanced_accuracy_ecart_type"],
    )
    logger.info(
        "Ensemble : balanced accuracy %.4f, ROC AUC %.4f, matrice %s",
        ensemble["balanced_accuracy"], ensemble["roc_auc"], ensemble["matrice_confusion"],
    )
    return resume


if __name__ == "__main__":
    run()