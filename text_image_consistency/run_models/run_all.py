"""
Run all five models sequentially and print outputs.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.caption import generate_caption
from src.grounding import grounding_score
from src.ocr import ocr_score
from src.similarity import clip_similarity
from src.tifa import tifa_score
from src.utils import ensure_data_dir


# Hardcoded paths - change if needed
IMAGE_PATH = os.path.join(ROOT, "data", "images", "sample.jpg")
DEFAULT_TEXT = "a lion face in black and white color"


def main():
    ensure_data_dir()
    parser = argparse.ArgumentParser(description="Run all models")
    parser.add_argument("--image", "-i", default=IMAGE_PATH, help="Image path")
    parser.add_argument("--text", "-t", default=DEFAULT_TEXT, help="Text prompt")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("RUNNING ALL MODELS")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Text:  {args.text}")
    print("=" * 60)

    # 1. BLIP
    print("\n[1/5] BLIP")
    caption = generate_caption(args.image)
    print(f"      blip_caption = \"{caption}\"")

    # 2. CLIP
    print("\n[2/5] CLIP")
    clip = clip_similarity(args.image, args.text)
    print(f"      clip_score = {clip:.4f}")

    # 3. Grounding DINO
    print("\n[3/5] Grounding DINO")
    grounding = grounding_score(args.image, args.text)
    print(f"      grounding_score = {grounding:.4f}")

    # 4. TIFA
    print("\n[4/5] TIFA")
    tifa = tifa_score(args.image, args.text, caption)
    print(f"      tifa_score = {tifa:.4f}")

    # 5. PaddleOCR
    print("\n[5/5] PaddleOCR")
    ocr = ocr_score(args.image, args.text)
    print(f"      ocr_score = {ocr:.4f}")

    print("\n" + "=" * 60)
    print("ALL OUTPUTS")
    print("=" * 60)
    print(f"  blip_caption:   \"{caption}\"")
    print(f"  clip_score:     {clip:.4f}")
    print(f"  grounding_score:{grounding:.4f}")
    print(f"  tifa_score:     {tifa:.4f}")
    print(f"  ocr_score:      {ocr:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
