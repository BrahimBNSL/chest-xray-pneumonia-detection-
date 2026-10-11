"""Moteur de prédiction : chargement du modèle et inférence sur une radiographie."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.config import CLASS_NAMES, OUTPUT_DIR, ModelConfig
from src.data import build_eval_transform
from src.features import DEFAULT_THRESHOLD, POSITIVE_INDEX
from src.model import build_model
from src.utils import get_device, get_logger

logger = get_logger(__name__)

DEFAULT_CHECKPOINT: Path = OUTPUT_DIR / "checkpoints" / "resnet50_finetune_seed42.pt"
MIN_SIDE_PIXELS: int = 128
DISPLAY_NAMES: dict[str, str] = {"NORMAL": "Normal", "PNEUMONIA": "Pneumonie"}


class InvalidImageError(ValueError):
    """Levée quand l'image fournie ne peut pas être analysée."""


@dataclass(frozen=True)
class Prediction:
    """Résultat d'une prédiction."""

    probabilite_pneumonie: float
    diagnostic: str
    confiance: float


class Predictor:
    """Charge le modèle une seule fois, puis prédit sur chaque image reçue."""

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT, image_size: int = 224) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint}")
        self.device = get_device()
        self.transform = build_eval_transform(image_size)
        self.model = build_model(
            ModelConfig(backbone="resnet50", freeze_backbone=False), unfreeze_last_block=False
        )
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()
        logger.info("Modèle chargé depuis %s (appareil : %s)", checkpoint.name, self.device)

    def _validate(self, image: Image.Image) -> Image.Image:
        """Vérifie que l'image est exploitable, sans la conserver."""
        if min(image.size) < MIN_SIDE_PIXELS:
            raise InvalidImageError(
                f"Image trop petite ({image.size[0]} × {image.size[1]} px) : "
                f"le minimum est {MIN_SIDE_PIXELS} px sur chaque côté."
            )
        return image.convert("L")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Prediction:
        """Retourne la probabilité de pneumonie, le diagnostic au seuil 0,5, et la confiance associée."""
        image = self._validate(image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probabilite = torch.softmax(self.model(tensor), dim=1)[0, POSITIVE_INDEX].item()

        if probabilite >= DEFAULT_THRESHOLD:
            diagnostic = DISPLAY_NAMES[CLASS_NAMES[POSITIVE_INDEX]]
            confiance = probabilite
        else:
            diagnostic = DISPLAY_NAMES[CLASS_NAMES[0]]
            confiance = 1 - probabilite

        return Prediction(
            probabilite_pneumonie=probabilite,
            diagnostic=diagnostic,
            confiance=confiance,
        )