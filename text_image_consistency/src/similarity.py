"""
CLIP text-image similarity (normalized cosine).
Model: openai/clip-vit-base-patch32
"""
import logging
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .utils import get_device, load_image

logger = logging.getLogger("t2i")

_model: Optional[CLIPModel] = None
_processor: Optional[CLIPProcessor] = None


def _get_clip():
    global _model, _processor
    if _model is None:
        device = get_device()
        logger.info("Loading CLIP (openai/clip-vit-base-patch32)...")
        logger.info("Downloading model weights (~500MB) if not cached...")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        _model.eval()
        logger.info("CLIP loaded successfully.")
    return _model, _processor


def clip_similarity(image_path: str, text: str) -> float:
    """
    Compute CLIP similarity between text and image (normalized cosine).
    Returns value in [0, 1].
    """
    if not text or not text.strip():
        return 0.0
    model, processor = _get_clip()
    device = get_device()
    image = load_image(image_path)
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(
        device
    )
    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image: (1, 1) for one image, one text
        logits = outputs.logits_per_image
        # CLIP logits are cosine sim * temperature; normalize to [0,1] via sigmoid
        # Raw cosine in [-1,1] -> (cos + 1) / 2 for [0,1]
        probs = logits.softmax(dim=-1)
        score = probs[0, 0].item()
    # Ensure [0, 1]; softmax of single pair can be skewed, use cosine directly for [0,1]
    image_emb = outputs.image_embeds
    text_emb = outputs.text_embeds
    cos = torch.nn.functional.cosine_similarity(
        image_emb, text_emb, dim=-1
    ).item()
    # Map [-1, 1] -> [0, 1]
    return float((cos + 1.0) / 2.0)
