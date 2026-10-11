"""Référence machine learning : régression logistique sur les caractéristiques d'un ResNet-50 gelé."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.config import CLASS_NAMES, OUTPUT_DIR, DataConfig, ModelConfig
from src.data import build_dataloaders, build_eval_transform
from src.model import build_model
from src.utils import ensure_dir, get_device, get_logger, save_json, set_seed

logger = get_logger(__name__)

C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
DEFAULT_THRESHOLD: float = 0.5
POSITIVE_INDEX: int = CLASS_NAMES.index("PNEUMONIA")
FEATURE_DIR: Path = OUTPUT_DIR / "features"
METRICS_PATH: Path = OUTPUT_DIR / "metrics" / "features_logreg.json"


def load_frozen_backbone(cfg: ModelConfig, device: torch.device) -> nn.Module:
    """Charge le backbone ResNet, entièrement gelé, sans sa tête de classification."""
    frozen_cfg = ModelConfig(backbone=cfg.backbone, num_classes=cfg.num_classes, freeze_backbone=True)
    model = build_model(frozen_cfg, unfreeze_last_block=False)
    model.fc = nn.Identity()
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_train_eval_loader(cfg: DataConfig, indices: list[int]) -> DataLoader:
    """Loader d'entraînement sans augmentation, pour extraire des caractéristiques fidèles aux images."""
    dataset = datasets.ImageFolder(
        cfg.data_dir / "train", transform=build_eval_transform(cfg.image_size)
    )
    return DataLoader(
        Subset(dataset, indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )


@torch.no_grad()
def extract_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les caractéristiques de chaque image et retourne le tableau et les étiquettes."""
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(images)
        features.append(output.float().cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def cached_features(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Charge les caractéristiques depuis le cache, ou les calcule puis les enregistre."""
    path = ensure_dir(cache_dir) / f"{name}.npz"
    if path.exists():
        data = np.load(path)
        logger.info("Caractéristiques chargées depuis le cache : %s", path.name)
        return data["x"], data["y"]

    x, y = extract_features(model, loader, device)
    np.savez_compressed(path, x=x, y=y)
    logger.info("Caractéristiques %s calculées : %s", name, x.shape)
    return x, y


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[object, float, float]:
    """Choisit le paramètre de régularisation C sur la validation, puis retourne le meilleur modèle."""
    best_model, best_c, best_score = None, None, -1.0
    for c in C_GRID:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=seed),
        )
        model.fit(x_train, y_train)
        score = balanced_accuracy_score(y_val, model.predict(x_val))
        logger.info("C = %g : balanced accuracy de validation %.4f", c, score)
        if score > best_score:
            best_model, best_c, best_score = model, c, score
    return best_model, best_c, best_score


def positive_probabilities(model: object, x: np.ndarray) -> np.ndarray:
    """Retourne la probabilité de la classe PNEUMONIA pour chaque image."""
    return model.predict_proba(x)[:, POSITIVE_INDEX]


def evaluate(probabilities: np.ndarray, y: np.ndarray) -> dict:
    """Calcule les métriques de classification au seuil par défaut de 0,5."""
    predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)
    report = classification_report(
        y, predictions, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    return {
        "seuil": DEFAULT_THRESHOLD,
        "accuracy": float((predictions == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "par_classe": {name: report[name] for name in CLASS_NAMES},
        "matrice_confusion": confusion_matrix(y, predictions).tolist(),
    }


def run() -> dict:
    """Exécute la référence de bout en bout : C choisi sur la validation, test évalué au seuil 0,5."""
    data_cfg = DataConfig()
    model_cfg = ModelConfig(backbone="resnet50", freeze_backbone=True)
    set_seed(data_cfg.seed)
    device = get_device()

    bundle = build_dataloaders(data_cfg)
    backbone = load_frozen_backbone(model_cfg, device)
    prefix = f"{model_cfg.backbone}_{data_cfg.image_size}"

    train_loader = build_train_eval_loader(data_cfg, bundle.train.dataset.indices)
    x_train, y_train = cached_features(f"{prefix}_train", backbone, train_loader, device, FEATURE_DIR)
    x_val, y_val = cached_features(f"{prefix}_val", backbone, bundle.val, device, FEATURE_DIR)
    x_test, y_test = cached_features(f"{prefix}_test", backbone, bundle.test, device, FEATURE_DIR)

    model, best_c, val_score = fit_logistic(x_train, y_train, x_val, y_val, data_cfg.seed)
    test_metrics = evaluate(positive_probabilities(model, x_test), y_test)

    results = {
        "modele": "ResNet-50 gelé + régression logistique",
        "C_retenu": best_c,
        "balanced_accuracy_validation": val_score,
        "test": test_metrics,
    }
    save_json(results, METRICS_PATH)
    logger.info(
        "Test (seuil 0,5) : balanced accuracy %.4f, ROC AUC %.4f, matrice %s",
        test_metrics["balanced_accuracy"], test_metrics["roc_auc"], test_metrics["matrice_confusion"],
    )
    return results


if __name__ == "__main__":
    run()