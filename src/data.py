"""Chargement des données, découpage par groupe et transformations des images."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, DataConfig
from src.utils import get_logger

logger = get_logger(__name__)

PNEUMONIA_PATTERN = re.compile(r"^(person\d+)_", re.IGNORECASE)
NORMAL_PATTERN = re.compile(r"^(?:NORMAL\d*-)?IM-(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class DataBundle:
    """Chargeurs de données et informations utiles à l'entraînement et à l'évaluation."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_labels: np.ndarray
    class_weights: torch.Tensor
    test_group_overlap: int


def build_train_transform(image_size: int) -> transforms.Compose:
    """Transformations d'entraînement : gris, redimensionnement, augmentation légère, normalisation."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(image_size: int) -> transforms.Compose:
    """Transformations de validation et de test : gris, redimensionnement, normalisation, sans augmentation."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image_folder(root: Path, transform: transforms.Compose) -> datasets.ImageFolder:
    """Charge un dossier au format ImageFolder et vérifie que les classes sont celles attendues."""
    if not root.exists():
        raise FileNotFoundError(f"Dossier introuvable : {root}")
    dataset = datasets.ImageFolder(root, transform=transform)
    if tuple(dataset.classes) != CLASS_NAMES:
        raise ValueError(
            f"Classes inattendues dans {root} : {dataset.classes}, attendues {CLASS_NAMES}"
        )
    return dataset


def extract_group_id(path: Path) -> str | None:
    """Extrait un identifiant patient approximatif à partir du nom de fichier.

    Les pneumonies sont nommées personXXX_..., donc le numéro identifie le patient.
    Les cas normaux sont nommés IM-XXXX-YYYY, avec un préfixe optionnel NORMAL2- : le numéro XXXX sert d'approximation du patient.
    Retourne None si le nom ne correspond à aucun des deux formats.
    """
    name = path.name
    match = PNEUMONIA_PATTERN.match(name)
    if match:
        return f"{path.parent.name}_{match.group(1).lower()}"
    match = NORMAL_PATTERN.match(name)
    if match:
        return f"{path.parent.name}_normal{int(match.group(1))}"
    return None


def build_groups(paths: list[Path]) -> tuple[np.ndarray, int]:
    """Attribue un identifiant de groupe à chaque image, et compte les noms non reconnus.

    Une image au nom non reconnu forme son propre groupe, ce qui évite de la regrouper à tort.
    """
    groups: list[str] = []
    unmatched = 0
    for path in paths:
        group_id = extract_group_id(path)
        if group_id is None:
            group_id = f"{path.parent.name}_{path.stem}"
            unmatched += 1
        groups.append(group_id)
    return np.asarray(groups), unmatched


def check_single_label_per_group(labels: np.ndarray, groups: np.ndarray) -> None:
    """Vérifie qu'un même groupe ne contient jamais deux classes différentes."""
    frame = pd.DataFrame({"label": labels, "group": groups})
    labels_per_group = frame.groupby("group")["label"].nunique()
    mixed = labels_per_group[labels_per_group > 1]
    if not mixed.empty:
        raise ValueError(f"Groupes contenant plusieurs classes : {mixed.index[:5].tolist()}")


def grouped_stratified_split(
    labels: np.ndarray, groups: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Découpe les indices par groupe, en conservant les proportions de classes.

    Tous les clichés d'un même groupe restent dans le même ensemble.
    Le nombre de plis est choisi pour que la validation représente environ val_fraction.
    """
    n_splits = max(2, round(1 / val_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    placeholder = np.zeros((len(labels), 1))
    train_idx, val_idx = next(splitter.split(placeholder, labels, groups))

    shared = np.intersect1d(groups[train_idx], groups[val_idx])
    if shared.size > 0:
        raise ValueError(f"Fuite détectée : {shared.size} groupes présents dans train et validation")
    return train_idx, val_idx


def count_group_overlap(groups_a: np.ndarray, groups_b: np.ndarray) -> int:
    """Compte les groupes présents dans les deux ensembles."""
    return len(set(groups_a.tolist()) & set(groups_b.tolist()))


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Calcule un poids inversement proportionnel à l'effectif de chaque classe."""
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"Une classe est absente de l'entraînement : {counts.tolist()}")
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(cfg: DataConfig) -> DataBundle:
    """Construit les chargeurs d'entraînement, de validation et de test."""
    train_root = cfg.data_dir / "train"
    test_root = cfg.data_dir / "test"

    train_aug = load_image_folder(train_root, build_train_transform(cfg.image_size))
    train_eval = load_image_folder(train_root, build_eval_transform(cfg.image_size))
    test_set = load_image_folder(test_root, build_eval_transform(cfg.image_size))

    labels = np.asarray(train_aug.targets)
    train_paths = [Path(path) for path, _ in train_aug.samples]
    groups, train_unmatched = build_groups(train_paths)
    check_single_label_per_group(labels, groups)

    train_idx, val_idx = grouped_stratified_split(labels, groups, cfg.val_fraction, cfg.seed)

    test_paths = [Path(path) for path, _ in test_set.samples]
    test_groups, test_unmatched = build_groups(test_paths)
    test_overlap = count_group_overlap(groups[train_idx], test_groups)

    train_set = Subset(train_aug, train_idx.tolist())
    val_set = Subset(train_eval, val_idx.tolist())

    generator = torch.Generator().manual_seed(cfg.seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    class_weights = compute_class_weights(train_labels, len(CLASS_NAMES))

    logger.info("Entraînement : %d images, %d groupes", len(train_idx), len(set(groups[train_idx])))
    logger.info("Validation   : %d images, %d groupes", len(val_idx), len(set(groups[val_idx])))
    logger.info("Test         : %d images, %d groupes", len(test_set), len(set(test_groups)))
    logger.info("Noms non reconnus : train %d, test %d", train_unmatched, test_unmatched)
    for name, n_train, n_val in zip(
        CLASS_NAMES,
        np.bincount(train_labels, minlength=len(CLASS_NAMES)),
        np.bincount(val_labels, minlength=len(CLASS_NAMES)),
    ):
        logger.info("Classe %-10s : train %d, validation %d", name, n_train, n_val)
    logger.info("Poids de classe : %s", [round(w, 3) for w in class_weights.tolist()])

    if test_overlap > 0:
        logger.warning(
            "%d identifiants approximatifs présents à la fois dans le train et dans le test : "
            "les scores de test peuvent être optimistes", test_overlap
        )

    return DataBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        train_labels=train_labels,
        class_weights=class_weights,
        test_group_overlap=test_overlap,
    )