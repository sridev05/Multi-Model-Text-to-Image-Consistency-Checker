"""
Standalone CLIP runner.
Model: openai/clip-vit-base-patch32
Output: Semantic similarity between text and image, in [0, 1].
"""
import argparse
import os
import sys

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.similarity import clip_similarity
from src.utils import ensure_data_dir


# Hardcoded paths - change these if needed
IMAGE_PATH = os.path.join(ROOT, "data", "images", "sample.jpg")
DEFAULT_TEXT = "a lion face in black and white color"


def main():
    ensure_data_dir()
    parser = argparse.ArgumentParser(description="Run CLIP: text-image similarity")
    parser.add_argument("--image", "-i", default=IMAGE_PATH, help="Image path")
    parser.add_argument("--text", "-t", default=DEFAULT_TEXT, help="Text prompt")
    args = parser.parse_args()

    print("=" * 60)
    print("CLIP (openai/clip-vit-base-patch32)")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Text:  {args.text}")
    print("-" * 60)

    score = clip_similarity(args.image, args.text)
    print(f"OUTPUT: clip_score = {score:.4f}  (range [0, 1])")
    print("=" * 60)


if __name__ == "__main__":
    main()
