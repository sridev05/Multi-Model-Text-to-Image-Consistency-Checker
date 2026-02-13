
"""
Entry point: hardcoded test case, formatted report.
"""
import os
import sys

# Ensure package root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ensure_data_dir
from src.verify import evaluate

def main() -> None:
    ensure_data_dir()
    image_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "images",
        "sample.jpg",
    )
    prompt = "A gray image with no distinct objects."
    result = evaluate(image_path, prompt)
    # Formatted report
    print("=" * 60)
    print("TEXT-IMAGE CONSISTENCY EVALUATION REPORT")
    print("=" * 60)
    print(f"Image:     {image_path}")
    print(f"Prompt:    {prompt}")
    print("-" * 60)
    print(f"BLIP caption:   {result['blip_caption']}")
    print(f"CLIP score:      {result['clip_score']:.4f}")
    print(f"Grounding score:{result['grounding_score']:.4f}")
    print(f"TIFA score:     {result['tifa_score']:.4f}")
    print(f"OCR score:      {result['ocr_score']:.4f}")
    print(f"Forward score:  {result['forward_score']:.4f}")
    print(f"Backward score: {result['backward_score']:.4f}")
    print("-" * 60)
    print(f"Final score:    {result['final_score']:.2f} / 100")
    print(f"Verdict:        {result['verdict']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
