"""
Shared utilities: image loading, device handling, seeding.
No imports from other project modules to avoid circular imports.
"""
import os
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("t2i")


def get_device() -> str:
    """Return 'cpu' for deterministic, CPU-only execution."""
    return "cpu"


def set_seed(seed: int = 42) -> None:
    """Set all relevant seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image(image_path: str) -> Image.Image:
    """Load image from path; raise FileNotFoundError if missing."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(path).convert("RGB")
    return img


def ensure_data_dir() -> Path:
    """Ensure data/images exists; create sample.jpg if missing (minimal placeholder)."""
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data" / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample = data_dir / "sample.jpg"
    if not sample.exists():
        # Minimal 64x64 RGB image so pipeline can run
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:] = [200, 200, 200]
        Image.fromarray(arr).save(sample, quality=85)
        logger.info("Created placeholder data/images/sample.jpg")
    return data_dir
