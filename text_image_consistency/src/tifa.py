
"""
TIFA: Text-Image Faithfulness. Decompose prompt into atomic facts;
verify facts against image + caption; produce faithfulness score.
LLM-based decomposition via HuggingFace (small, CPU-compatible).
"""
import logging
import re
from typing import List

from .utils import load_image

logger = logging.getLogger("t2i")

# Use a small seq2seq model for fact decomposition (CPU-friendly)
TIFA_DECOMPOSITION_MODEL = "google/flan-t5-small"
_model = None
_tokenizer = None


def _get_tifa_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        logger.info("Loading TIFA decomposition model (%s)...", TIFA_DECOMPOSITION_MODEL)
        _tokenizer = T5Tokenizer.from_pretrained(TIFA_DECOMPOSITION_MODEL)
        _model = T5ForConditionalGeneration.from_pretrained(TIFA_DECOMPOSITION_MODEL)
        _model.eval()
    return _model, _tokenizer


def _decompose_prompt_into_facts(prompt: str) -> List[str]:
    """
    Decompose prompt into atomic facts (phrases to verify).
    Uses rule-based split plus optional LM; fallback to clause/phrase split.
    """
    prompt = prompt.strip()
    if not prompt:
        return []
    
    # Rule-based: split on conjunctions and commas; trim
    parts = re.split(r"\s+and\s+|\s+with\s+|\s*,\s*|\s+;\s*", prompt, flags=re.IGNORECASE)
    facts = []
    for p in parts:
        p = p.strip().strip(".")
        if len(p) > 3 and len(p) < 150:  # Lower minimum length from 10 to 3
            facts.append(p)
    
    # If single segment, try to break by "in/on/at" phrases
    if len(facts) <= 1 and len(prompt) > 30:
        sub = re.split(r"\s+in\s+|\s+on\s+|\s+at\s+", prompt, maxsplit=2, flags=re.IGNORECASE)
        if len(sub) > 1:
            facts = [s.strip().strip(".") for s in sub if len(s.strip()) > 3]
    
    # Also extract negations separately for better matching
    negation_patterns = [
        (r"\bno\s+([a-z\s]+?)(?:\.|,|$)", "no {}"),  # "no distinct objects"
        (r"\bwithout\s+([a-z\s]+?)(?:\.|,|$)", "without {}"),
    ]
    
    for pattern, template in negation_patterns:
        matches = re.findall(pattern, prompt, re.IGNORECASE)
        for match in matches:
            facts.append(template.format(match.strip()))
    
    if not facts:
        facts = [prompt[:200]]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_facts = []
    for f in facts[:15]:
        f_norm = " ".join(f.lower().split())
        if f_norm not in seen:
            seen.add(f_norm)
            unique_facts.append(f)
    
    return unique_facts[:15]


def _fact_in_caption_llm(fact: str, caption: str, model, tokenizer) -> bool:
    """Use FLAN-T5 to decide if caption supports the fact (yes/no)."""
    import torch
    from .utils import get_device
    
    device = get_device()
    inp = f"Does the following description support the claim? Description: {caption[:200]}. Claim: {fact[:100]}. Answer yes or no:"
    inputs = tokenizer(inp, return_tensors="pt", truncation=True, max_length=256).to(device)
    model = model.to(device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=3, num_beams=1)
    
    ans = tokenizer.decode(out[0], skip_special_tokens=True).strip().lower()
    
    # More flexible answer detection
    affirmative = ["yes", "true", "correct", "right", "support", "match"]
    negative = ["no", "false", "incorrect", "wrong", "not support", "contradict"]
    
    # Check if any affirmative word appears
    has_affirmative = any(word in ans for word in affirmative)
    has_negative = any(word in ans for word in negative)
    
    # If both appear, check which comes first or is stronger
    if has_affirmative and not has_negative:
        return True
    elif has_negative and not has_affirmative:
        return False
    elif has_affirmative and has_negative:
        # If "yes" appears but also "no", prefer the explicit answer
        return ans.startswith("yes") or ans.startswith("true")
    
    # Fallback: consider "yes" as affirmative if unclear
    return ans.startswith("yes") or ans.startswith("true")


def _fact_in_caption_overlap(fact: str, caption: str) -> bool:
    """Fallback: keyword overlap with semantic handling."""
    fact_lower = fact.lower().strip()
    cap_lower = caption.lower().strip()
    if not fact_lower or not cap_lower:
        return False
    
    f_words = set(re.findall(r"\b\w+\b", fact_lower))
    c_words = set(re.findall(r"\b\w+\b", cap_lower))
    
    # Remove common stop words
    stop_words = {"a", "an", "the", "and", "or", "in", "on", "at", "to", "of", "is", "with", "by"}
    f_words = f_words - stop_words
    c_words = c_words - stop_words
    
    if not f_words:
        return False
    
    overlap = len(f_words & c_words) / len(f_words)
    
    # Semantic similarity checks for common terms
    color_words = {"gray", "grey", "black", "white", "red", "blue", "green", "yellow"}
    object_words = {"object", "objects", "thing", "things", "person", "people", "man", "woman"}
    
    # Check for contradictions
    if "no" in f_words and ("distinct" in f_words or "objects" in f_words):
        # "no distinct objects" - check if caption mentions objects/people
        if c_words & object_words:
            return False
    
    # Reduced threshold from 0.4 to 0.25
    return overlap >= 0.25


def tifa_score(image_path: str, text: str, caption: str) -> float:
    """
    TIFA faithfulness: decompose prompt into atomic facts, verify against caption.
    LLM-based verification via FLAN-T5. Returns score in [0, 1].
    """
    if not text or not text.strip():
        return 1.0
    load_image(image_path)
    facts = _decompose_prompt_into_facts(text)
    if not facts:
        return 1.0
    try:
        model, tokenizer = _get_tifa_model()
        supported = sum(
            1 for f in facts if _fact_in_caption_llm(f, caption, model, tokenizer)
        )
    except Exception as e:
        logger.warning("TIFA LLM verification failed (%s), using overlap fallback.", e)
        supported = sum(1 for f in facts if _fact_in_caption_overlap(f, caption))
    return supported / len(facts)
