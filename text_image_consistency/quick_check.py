"""
Quick check: Is Grounding DINO already downloaded?
"""
import sys
from pathlib import Path

model_id = "IDEA-Research/grounding-dino-tiny"
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

print("Checking cache for:", model_id)
print("Cache location:", cache_dir)

if cache_dir.exists():
    # Look for the model directory
    model_slug = model_id.replace("/", "--")
    matches = list(cache_dir.glob(f"models--{model_slug}*"))
    
    if matches:
        print(f"\nFound cache directory: {matches[0].name}")
        # Check for model files
        model_files = list(matches[0].rglob("*.safetensors")) + list(matches[0].rglob("*.bin"))
        if model_files:
            total_size = sum(f.stat().st_size for f in model_files if f.is_file())
            print(f"Found {len(model_files)} model files")
            print(f"Total size: {total_size / (1024**3):.2f} GB")
            if total_size > 1_000_000_000:  # > 1GB
                print("\n[INFO] Model appears fully downloaded!")
                print("Loading should be fast (no download needed).")
            else:
                print("\n[WARNING] Model files seem incomplete.")
                print("May need to download remaining files.")
        else:
            print("\n[INFO] Cache directory exists but no model files found.")
            print("Will download on first use.")
    else:
        print("\n[INFO] Model not in cache - will download (~1.4GB)")
else:
    print("\n[INFO] Cache directory doesn't exist yet.")
    print("Will create and download on first use.")

print("\n" + "="*60)
print("To test loading (this may take time):")
print("  python -c \"from transformers import AutoModelForZeroShotObjectDetection; AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny')\"")
print("="*60)
