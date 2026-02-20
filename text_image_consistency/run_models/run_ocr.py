"""
Standalone PaddleOCR runner.
Output: Consistency score in [0, 1] between extracted image text and prompt.
(If PaddleOCR fails on your system, score returns 1.0 as neutral.)
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.ocr import ocr_score
from src.utils import ensure_data_dir


# Hardcoded paths - change if needed
IMAGE_PATH = os.path.join(ROOT, "data", "images", "sample.jpg")
DEFAULT_TEXT = "welcome word in white colour with voilet background"


def main():
    ensure_data_dir()
    parser = argparse.ArgumentParser(description="Run PaddleOCR: text extraction + consistency")
    parser.add_argument("--image", "-i", default=IMAGE_PATH, help="Image path")
    parser.add_argument("--text", "-t", default=DEFAULT_TEXT, help="Text prompt")
    args = parser.parse_args()

    print("=" * 60)
    print("PaddleOCR")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Text:  {args.text}")
    print("-" * 60)

    score = ocr_score(args.image, args.text)
    print(f"OUTPUT: ocr_score = {score:.4f}  (range [0, 1])")
    print("=" * 60)


if __name__ == "__main__":
    main()
