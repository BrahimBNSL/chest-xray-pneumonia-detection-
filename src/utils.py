"""Outils transversaux : reproductibilité, choix du device, journalisation et sauvegarde."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def set_seed(seed: int) -> None:
    """Fixe les graines de tous les générateurs aléatoires utilisés dans le projet."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Retourne le GPU s'il est disponible, sinon le CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Crée un logger configuré une seule fois, sans doublons de handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def ensure_dir(path: Path) -> Path:
    """Crée le dossier s'il n'existe pas, puis le retourne."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_serializable(value: Any) -> Any:
    """Convertit les types NumPy et PyTorch en types Python natifs pour le JSON."""
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    """Sauvegarde un dictionnaire en JSON lisible, avec création du dossier parent."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    """Charge un fichier JSON et retourne un dictionnaire."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)