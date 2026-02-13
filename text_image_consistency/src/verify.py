"""
Central orchestrator: run full pipeline and return evaluation dict.
"""
import logging
from typing import Any, Dict

from .caption import generate_caption
from .fusion import compute_fusion, verdict
from .grounding import grounding_score
from .ocr import ocr_score
from .similarity import clip_similarity
from .tifa import tifa_score
from .utils import set_seed

logger = logging.getLogger("t2i")


def evaluate(image_path: str, prompt: str) -> Dict[str, Any]:
    """
    Run full text-image consistency evaluation.
    Returns dict with: blip_caption, clip_score, grounding_score, tifa_score,
    ocr_score, forward_score, backward_score, final_score (0-100), verdict.
    """
    set_seed(42)
    # BLIP caption (used in forward as grounding signal and in backward)
    blip_caption = generate_caption(image_path)
    # Forward: text -> image
    clip_score = clip_similarity(image_path, prompt)
    grounding_score_val = grounding_score(image_path, prompt)
    tifa_score_val = tifa_score(image_path, prompt, blip_caption)
    ocr_score_val = ocr_score(image_path, prompt)
    # Backward: image -> text (BLIP caption) -> CLIP(image, caption)
    backward_score = clip_similarity(image_path, blip_caption)
    # Fusion
    fusion_out = compute_fusion(
        clip_score=clip_score,
        grounding_score=grounding_score_val,
        tifa_score=tifa_score_val,
        ocr_score=ocr_score_val,
        backward_score=backward_score,
    )
    final = fusion_out["final_score"]
    return {
        "blip_caption": blip_caption,
        "clip_score": round(clip_score, 4),
        "grounding_score": round(grounding_score_val, 4),
        "tifa_score": round(tifa_score_val, 4),
        "ocr_score": round(ocr_score_val, 4),
        "forward_score": fusion_out["forward_score"],
        "backward_score": fusion_out["backward_score"],
        "final_score": final,
        "verdict": verdict(final),
    }
