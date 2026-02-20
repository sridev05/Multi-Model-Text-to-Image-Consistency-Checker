"""
Standalone TIFA runner.
Model: google/flan-t5-small (for fact verification)
Output: Faithfulness score in [0, 1] (fraction of prompt facts supported by caption).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.caption import generate_caption
from src.tifa import tifa_score
from src.utils import ensure_data_dir


# Hardcoded paths - change if needed
IMAGE_PATH = os.path.join(ROOT, "data", "images", "image.png")
DEFAULT_TEXT = "a lion face in black and white color"


def main():
    ensure_data_dir()
    parser = argparse.ArgumentParser(description="Run TIFA: text-image faithfulness")
    parser.add_argument("--image", "-i", default=IMAGE_PATH, help="Image path")
    parser.add_argument("--text", "-t", default=DEFAULT_TEXT, help="Text prompt")
    args = parser.parse_args()

    print("=" * 60)
    print("TIFA (google/flan-t5-small for fact verification)")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Text:  {args.text}")
    print("-" * 60)

    caption = generate_caption(args.image)
    print(f"BLIP caption (used by TIFA): \"{caption}\"")
    print("-" * 60)

    score = tifa_score(args.image, args.text, caption)
    print(f"OUTPUT: tifa_score = {score:.4f}  (range [0, 1])")
    print("=" * 60)


if __name__ == "__main__":
    main()
