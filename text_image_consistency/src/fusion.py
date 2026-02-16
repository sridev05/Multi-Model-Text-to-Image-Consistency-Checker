"""
Bidirectional fusion: combine forward and backward scores;
produce final score in [0, 100] and verdict.
"""
import logging
from typing import Dict

logger = logging.getLogger("t2i")

# Weights for forward components (must sum to 1.0 for forward_score in [0,1])
W_CLIP = 0.30
W_GROUNDING = 0.25
W_TIFA = 0.25
W_OCR = 0.20

# Weights for bidirectional fusion
W_FORWARD = 0.6
W_BACKWARD = 0.4


def forward_score(
    clip_score: float,
    grounding_score: float,
    tifa_score: float,
    ocr_score: float,
) -> float:
    """Compute single forward score in [0, 1] from component scores."""
    return (
        W_CLIP * clip_score
        + W_GROUNDING * grounding_score
        + W_TIFA * tifa_score
        + W_OCR * ocr_score
    )


def bidirectional_fusion(forward: float, backward: float) -> float:
    """Fuse forward and backward scores; return value in [0, 1] (before scale to 100)."""
    return W_FORWARD * forward + W_BACKWARD * backward


def final_score_normalized(fused: float) -> float:
    """Normalize to [0, 100]."""
    return max(0.0, min(100.0, fused * 100.0))


def verdict(score_0_100: float) -> str:
    """Return MATCH | PARTIAL MATCH | MISMATCH from final score in [0, 100]."""
    if score_0_100 >= 70:
        return "MATCH"
    if score_0_100 >= 50:
        return "PARTIAL MATCH"
    return "MISMATCH"


def compute_fusion(
    clip_score: float,
    grounding_score: float,
    tifa_score: float,
    ocr_score: float,
    backward_score: float,
) -> Dict[str, float]:
    """
    Compute forward score, fused score, final 0-100 score.
    Returns dict with forward_score, backward_score, final_score (0-100).
    """
    fwd = forward_score(clip_score, grounding_score, tifa_score, ocr_score)
    fused = bidirectional_fusion(fwd, backward_score)
    final = final_score_normalized(fused)
    return {
        "forward_score": round(fwd, 4),
        "backward_score": round(backward_score, 4),
        "final_score": round(final, 2),
    }
