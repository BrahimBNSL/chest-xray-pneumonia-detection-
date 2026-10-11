"""Configuration centralisée du projet.

Toutes les constantes et tous les hyperparamètres sont définis ici.
Les autres modules importent cette configuration au lieu de redéfinir des valeurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data" / "chest_xray"
OUTPUT_DIR: Path = ROOT_DIR / "outputs"

CLASS_NAMES: tuple[str, ...] = ("NORMAL", "PNEUMONIA")
POSITIVE_CLASS: str = "PNEUMONIA"

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataConfig:
    """Paramètres de chargement et de découpage des données."""

    data_dir: Path = DATA_DIR
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    val_fraction: float = 0.15
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction doit être dans ]0, 1[, reçu {self.val_fraction}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size doit être positif, reçu {self.batch_size}")
        if self.image_size <= 0:
            raise ValueError(f"image_size doit être positif, reçu {self.image_size}")


@dataclass(frozen=True)
class ModelConfig:
    """Paramètres de l'architecture."""

    backbone: str = "resnet50"
    num_classes: int = len(CLASS_NAMES)
    freeze_backbone: bool = True

    def __post_init__(self) -> None:
        if self.backbone not in {"resnet18", "resnet50"}:
            raise ValueError(f"backbone non supporté : {self.backbone}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes doit être au moins 2, reçu {self.num_classes}")


@dataclass(frozen=True)
class TrainConfig:
    """Paramètres d'entraînement du fine-tuning."""

    epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 4
    unfreeze_last_block: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs doit être positif, reçu {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate doit être positif, reçu {self.learning_rate}")
        if self.patience < 0:
            raise ValueError(f"patience doit être positive ou nulle, reçu {self.patience}")