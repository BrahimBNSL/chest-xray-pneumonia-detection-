"""Fine-tuning complet de ResNet-50 : entraînement pour une graine donnée, arrêt anticipé et sauvegarde du meilleur modèle."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import OUTPUT_DIR, DataConfig, ModelConfig, TrainConfig
from src.data import build_dataloaders
from src.features import DEFAULT_THRESHOLD, POSITIVE_INDEX, evaluate
from src.model import build_model, set_train_mode
from src.utils import ensure_dir, get_device, get_logger, save_json, set_seed

logger = get_logger(__name__)

BACKBONE_LR_FACTOR: float = 0.1
HEAD_LR_FACTOR: float = 10.0
MAX_GRAD_NORM: float = 1.0
MIN_DELTA: float = 1e-4


def checkpoint_path(seed: int) -> Path:
    """Chemin du meilleur modèle pour une graine donnée."""
    return OUTPUT_DIR / "checkpoints" / f"resnet50_finetune_seed{seed}.pt"


def history_path(seed: int) -> Path:
    """Chemin de l'historique d'entraînement pour une graine donnée."""
    return OUTPUT_DIR / "metrics" / f"finetune_history_seed{seed}.json"


def results_path(seed: int) -> Path:
    """Chemin des résultats finaux pour une graine donnée."""
    return OUTPUT_DIR / "metrics" / f"finetune_resnet50_seed{seed}.json"


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> AdamW:
    """Crée AdamW avec deux groupes de taux : faible pour le backbone, plus élevé pour la tête."""
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("fc.")]
    return AdamW(
        [
            {"params": backbone_params, "lr": cfg.learning_rate * BACKBONE_LR_FACTOR},
            {"params": head_params, "lr": cfg.learning_rate * HEAD_LR_FACTOR},
        ],
        weight_decay=cfg.weight_decay,
    )


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    """Entraîne le modèle sur une epoch et retourne la perte moyenne."""
    set_train_mode(model)
    total_loss, total_count = 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(images), targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(targets)
        total_count += len(targets)
    return total_loss / total_count


@torch.no_grad()
def predict(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Calcule la perte, les probabilités de PNEUMONIA et les étiquettes, sans mise à jour des poids."""
    model.eval()
    total_loss, total_count = 0.0, 0
    probabilities, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        loss = criterion(logits.float(), targets)
        total_loss += loss.item() * len(targets)
        total_count += len(targets)
        probabilities.append(torch.softmax(logits.float(), dim=1)[:, POSITIVE_INDEX].cpu().numpy())
        labels.append(targets.cpu().numpy())
    return total_loss / total_count, np.concatenate(probabilities), np.concatenate(labels)


def run(seed: int = 42) -> dict:
    """Entraîne le fine-tuning pour une graine donnée, puis évalue le meilleur modèle sur le test.

    Le découpage des données reste fixé par DataConfig.seed. Seule la graine d'entraînement change.
    """
    data_cfg = DataConfig()
    model_cfg = ModelConfig(backbone="resnet50", freeze_backbone=False)
    train_cfg = TrainConfig()
    device = get_device()

    bundle = build_dataloaders(data_cfg)
    set_seed(seed)
    logger.info("Graine d'entraînement : %d", seed)

    model = build_model(model_cfg, unfreeze_last_block=False).to(device)
    criterion = nn.CrossEntropyLoss(weight=bundle.class_weights.to(device))
    optimizer = build_optimizer(model, train_cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    ensure_dir(checkpoint_path(seed).parent)
    history: list[dict] = []
    best_loss, best_epoch, epochs_without_gain = float("inf"), 0, 0

    for epoch in range(1, train_cfg.epochs + 1):
        started = time.perf_counter()
        train_loss = train_one_epoch(model, bundle.train, criterion, optimizer, scaler, device)
        val_loss, val_probs, val_labels = predict(model, bundle.val, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - started

        val_balanced = balanced_accuracy_score(val_labels, (val_probs >= DEFAULT_THRESHOLD).astype(int))
        history.append({
            "epoch": epoch,
            "perte_train": train_loss,
            "perte_validation": val_loss,
            "balanced_accuracy_validation": float(val_balanced),
            "duree_secondes": round(elapsed, 1),
        })
        logger.info(
            "Epoch %2d/%d : perte train %.4f, perte validation %.4f, balanced accuracy %.4f, durée %.0f s",
            epoch, train_cfg.epochs, train_loss, val_loss, val_balanced, elapsed,
        )

        if val_loss < best_loss - MIN_DELTA:
            best_loss, best_epoch, epochs_without_gain = val_loss, epoch, 0
            torch.save(model.state_dict(), checkpoint_path(seed))
            logger.info("Nouveau meilleur modèle sauvegardé (epoch %d)", epoch)
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= train_cfg.patience:
                logger.info("Arrêt anticipé : pas d'amélioration depuis %d epochs", train_cfg.patience)
                break

    save_json({"historique": history}, history_path(seed))

    model.load_state_dict(torch.load(checkpoint_path(seed), map_location=device, weights_only=True))
    _, test_probs, test_labels = predict(model, bundle.test, criterion, device)
    test_metrics = evaluate(test_probs, test_labels)

    results = {
        "modele": "ResNet-50 fine-tuning complet",
        "graine": seed,
        "meilleure_epoch": best_epoch,
        "meilleure_perte_validation": best_loss,
        "test": test_metrics,
    }
    save_json(results, results_path(seed))
    logger.info(
        "Test (graine %d, meilleure epoch %d, seuil 0,5) : balanced accuracy %.4f, ROC AUC %.4f, matrice %s",
        seed, best_epoch, test_metrics["balanced_accuracy"], test_metrics["roc_auc"],
        test_metrics["matrice_confusion"],
    )
    return results


def main() -> None:
    """Point d'entrée en ligne de commande : python -m src.train --seed 1."""
    parser = argparse.ArgumentParser(description="Fine-tuning ResNet-50 pour une graine donnée.")
    parser.add_argument("--seed", type=int, default=42, help="Graine d'entraînement")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()