"""
PaddleOCR: extract visible text from image; compare with prompt claims.
"""
import logging
import re
from pathlib import Path
from typing import List

from .utils import load_image

logger = logging.getLogger("t2i")

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        logger.info("Loading PaddleOCR...")
        from paddleocr import PaddleOCR

        try:
            # Use default CPU configuration; newer versions parse common args internally.
            _ocr = PaddleOCR()
        except Exception as e:
            logger.warning("PaddleOCR unavailable (%s); OCR score will be neutral (1.0).", e)
            _ocr = None
    return _ocr


def _extract_text_from_image(image_path: str) -> str:
    """Run PaddleOCR and return concatenated recognized text (single string)."""
    ocr = _get_ocr()
    if ocr is None:
        return ""
    path = Path(image_path)
    if not path.exists():
        return ""
    # Newer PaddleOCR pipeline API does not accept `cls` kwarg; use default signature.
    try:
        result = ocr.ocr(str(path))
    except Exception as e:
        # Some Paddle/PaddleOCR builds on CPU can fail with low-level NotImplementedError.
        # In that case, we fall back to "no OCR text" so scoring stays neutral.
        logger.warning("PaddleOCR inference failed (%s); treating OCR text as empty.", e)
        return ""
    if not result or result[0] is None:
        return ""
    lines = []
    for line in result[0]:
        if line and len(line) >= 2:
            lines.append(line[1][0])
    return " ".join(lines).strip()


def _normalize(s: str) -> str:
    """Lowercase and collapse whitespace for comparison."""
    return re.sub(r"\s+", " ", s.lower().strip())


def ocr_score(image_path: str, text: str) -> float:
    """
    Extract visible text from image; compare with prompt.
    Returns score in [0, 1]: consistency between OCR text and prompt text claims.
    If prompt mentions no specific text, return 1.0 (no penalty).
    If prompt mentions text, score by overlap / absence of contradiction.
    """
    if not text or not text.strip():
        return 1.0
    ocr_text = _extract_text_from_image(image_path)
    prompt_norm = _normalize(text)
    ocr_norm = _normalize(ocr_text)
    # If no text expected in image (heuristic: no quoted strings, no "says", "reads", "written")
    # we score 1.0 if OCR is empty or small, else mild penalty for irrelevant text
    quoted = re.findall(r'"([^"]+)"', text) or re.findall(r"'([^']+)'", text)
    if not quoted and not re.search(r"\b(says|reads|written|text|sign|label)\b", prompt_norm):
        if not ocr_norm:
            return 1.0
        # Prompt doesn't explicitly require text; don't penalize
        return 1.0
    if not quoted:
        # Generic "text" mention; score by whether OCR exists
        return 1.0 if ocr_norm else 0.5
    # Check quoted strings (or expected text) against OCR
    hits = 0
    for q in quoted:
        if _normalize(q) in ocr_norm or q.lower() in ocr_norm:
            hits += 1
    return hits / len(quoted) if quoted else 1.0
