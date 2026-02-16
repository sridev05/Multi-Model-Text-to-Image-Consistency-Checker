"""
Grounding DINO: object-level grounding, entity presence verification.
Model: IDEA-Research/grounding-dino-tiny
"""
import logging
import os
import re
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .utils import get_device, load_image

os.environ["TRANSFORMERS_VERBOSITY"] = "info"
logger = logging.getLogger("t2i")

_model: Optional[AutoModelForZeroShotObjectDetection] = None
_processor: Optional[AutoProcessor] = None

GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def _get_grounding():
    global _model, _processor
    if _model is None:
        device = get_device()
        logger.info("Loading Grounding DINO (%s)...", GROUNDING_MODEL_ID)

        _processor = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_MODEL_ID
        ).to(device)
        _model.eval()

        logger.info("Grounding DINO loaded successfully.")
    return _model, _processor


def _extract_entities(text: str) -> List[str]:
    text = text.lower().strip().rstrip(".")
    if not text:
        return []

    parts = re.split(r"[,;]|\band\b|\bor\b|\.", text)
    entities = []

    for p in parts:
        p = p.strip()
        if 0 < len(p) <= 60:
            entities.append(p)

    # Deduplicate
    seen = set()
    unique = []
    for e in entities:
        e_norm = " ".join(e.split())
        if e_norm not in seen:
            seen.add(e_norm)
            unique.append(e_norm)

    if not unique:
        unique = [" ".join(text.split()[:10])]

    return unique[:10]


def grounding_score(image_path: str, text: str) -> float:
    """
    Returns score in [0,1]: fraction of entities detected in the image.
    """
    if not text.strip():
        return 0.0

    model, processor = _get_grounding()
    device = get_device()

    image: Image.Image = load_image(image_path)
    entities = _extract_entities(text)
    if not entities:
        return 0.0

    # Grounding DINO expects ONE string with period-separated phrases
    prompt = ". ".join(entities)
    if not prompt.endswith("."):
        prompt += "."

    inputs = processor(
        images=image,
        text=[prompt],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)

    # HF API: post_process_grounded_object_detection(outputs, input_ids, box_threshold, text_threshold, target_sizes)
    input_ids = inputs.get("input_ids", None)
    if input_ids is None:
        # Some versions expose it as an attribute on the BatchFeature
        input_ids = getattr(inputs, "input_ids", None)
    if input_ids is None:
        logger.warning("Grounding DINO: input_ids not found; returning 0.0 grounding score.")
        return 0.0

    try:
        # Try new API first (transformers >= 4.40)
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=input_ids,
            box_threshold=0.3,
            text_threshold=0.25,
            target_sizes=target_sizes,
        )
    except TypeError:
        # Fallback to API without threshold parameters
        results = processor.post_process_grounded_object_detection(
            outputs,
            target_sizes=target_sizes,
        )

    result = results[0]
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]

    if len(scores) == 0:
        return 0.0

    THRESHOLD = 0.3
    matched = 0

    for entity in entities:
        for label, score in zip(labels, scores):
            if score.item() >= THRESHOLD:
                # labels are text spans predicted by DINO
                if entity in label.lower() or label.lower() in entity:
                    matched += 1
                    break

    return matched / len(entities)
