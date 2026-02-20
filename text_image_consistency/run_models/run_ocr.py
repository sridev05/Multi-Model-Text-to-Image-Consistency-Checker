"""
Standalone PaddleOCR Runner
----------------------------------
Outputs:
- Extracted text
- Filtered prompt text
- OCR average confidence
- Final consistency score in [0, 1]
"""

import argparse
import os
import sys
import re
from difflib import SequenceMatcher
from paddleocr import PaddleOCR

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Hardcoded defaults
IMAGE_PATH = os.path.join(ROOT, "data", "images", "image.png")
DEFAULT_TEXT = "welcome word in white colour with voilet background"


# ----------------------------
# Utility Functions
# ----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def extract_text_words_from_prompt(prompt):
    """
    Remove descriptive words like colors/background
    Keep only possible visible text words
    """
    stop_words = {
        "in", "with", "on", "background", "colour", "color",
        "white", "black", "red", "blue", "green", "yellow",
        "violet", "voilet", "word", "text"
    }

    words = clean_text(prompt).split()
    filtered = [w for w in words if w not in stop_words]

    return " ".join(filtered)


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# ----------------------------
# OCR Processing
# ----------------------------

def run_ocr(image_path):
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

    result = ocr_engine.ocr(image_path, cls=True)

    if not result:
        return "", 0.0

    extracted_text = ""
    confidences = []

    for line in result:
        for word_info in line:
            text, conf = word_info[1]
            extracted_text += text + " "
            confidences.append(conf)

    extracted_text = clean_text(extracted_text)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return extracted_text.strip(), round(avg_conf, 4)


def compute_consistency(extracted_text, prompt_text):
    filtered_prompt = extract_text_words_from_prompt(prompt_text)

    if not extracted_text or not filtered_prompt:
        return 0.0, filtered_prompt

    score = similarity(extracted_text, filtered_prompt)
    return round(score, 4), filtered_prompt


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Standalone PaddleOCR Consistency Runner")
    parser.add_argument("--image", "-i", default=IMAGE_PATH, help="Image path")
    parser.add_argument("--text", "-t", default=DEFAULT_TEXT, help="Text prompt")
    args = parser.parse_args()

    print("=" * 60)
    print("PADDLE OCR CONSISTENCY CHECK")
    print("=" * 60)

    if not os.path.exists(args.image):
        print("ERROR: Image path does not exist.")
        print(args.image)
        return

    print(f"Image Path : {args.image}")
    print(f"Prompt     : {args.text}")
    print("-" * 60)

    extracted_text, avg_conf = run_ocr(args.image)

    print(f"Extracted Text       : {extracted_text}")
    print(f"OCR Avg Confidence   : {avg_conf}")

    score, filtered_prompt = compute_consistency(extracted_text, args.text)

    print(f"Filtered Prompt Text : {filtered_prompt}")
    print("-" * 60)
    print(f"FINAL OCR SCORE      : {score}   (range [0,1])")
    print("=" * 60)


if __name__ == "__main__":
    main()