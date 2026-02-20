"""
Standalone Grounding DINO runner.
Model: IDEA-Research/grounding-dino-tiny
Output: Entity presence score in [0, 1] (fraction of prompt entities detected).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.grounding import grounding_score
from src.utils import ensure_data_dir


# Hardcoded paths - change if needed
IMAGE_PATH = os.path.join(ROOT, "data", "images", "sample.jpg")
DEFAULT_TEXT = "a lion face in black and white color"


def main():
    ensure_data_dir()
    parser = argparse.ArgumentParser(description="Run Grounding DINO: entity detection")
    parser.add_argument("--image", "-i", default=IMAGE_PATH, help="Image path")
    parser.add_argument("--text", "-t", default=DEFAULT_TEXT, help="Text prompt")
    args = parser.parse_args()

    print("=" * 60)
    print("Grounding DINO (IDEA-Research/grounding-dino-tiny)")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Text:  {args.text}")
    print("-" * 60)

    score = grounding_score(args.image, args.text)
    print(f"OUTPUT: grounding_score = {score:.4f}  (range [0, 1])")
    print("=" * 60)


if __name__ == "__main__":
    main()
