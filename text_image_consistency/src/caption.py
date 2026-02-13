"""
BLIP image captioning. Singleton model loading.
Model: Salesforce/blip-image-captioning-base
"""
import logging
from typing import Optional

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from .utils import get_device, load_image

logger = logging.getLogger("t2i")

_model: Optional[BlipForConditionalGeneration] = None
_processor: Optional[BlipProcessor] = None


def _get_blip():
    global _model, _processor
    if _model is None:
        device = get_device()
        logger.info("Loading BLIP (Salesforce/blip-image-captioning-base)...")
        logger.info("Downloading model weights (~990MB) if not cached...")
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        _model.eval()
        logger.info("BLIP loaded successfully.")
    return _model, _processor


def generate_caption(image_path: str) -> str:
    """
    Generate dense image caption using BLIP.
    Returns a single string caption.
    """
    model, processor = _get_blip()
    device = get_device()
    image = load_image(image_path)
    # Unconditional caption (no text prompt)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50, num_beams=3)
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    return caption if caption else ""
