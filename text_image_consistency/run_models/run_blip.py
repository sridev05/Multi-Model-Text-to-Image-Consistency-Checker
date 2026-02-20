"""
Standalone BLIP runner.
Model: Salesforce/blip-image-captioning-base
Output: Dense image caption (single string).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.caption import generate_caption
from src.utils import ensure_data_dir


# Hardcoded path - change if needed
IMAGE_PATH = os.path.join(ROOT, "data", "images", "sample.jpg")


def main():
    ensure_data_dir()
    parser = argparse.ArgumentParser(description="Run BLIP: image captioning")
    parser.add_argument("--image", "-i", default=IMAGE_PATH, help="Image path")
    args = parser.parse_args()

    print("=" * 60)
    print("BLIP (Salesforce/blip-image-captioning-base)")
    print("=" * 60)
    print(f"Image: {args.image}")
    print("-" * 60)

    caption = generate_caption(args.image)
    print(f"OUTPUT: blip_caption = \"{caption}\"")
    print("=" * 60)


if __name__ == "__main__":
    main()
