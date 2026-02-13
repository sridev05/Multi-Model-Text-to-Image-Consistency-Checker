"""
Check if Grounding DINO model is already cached locally.
"""
import os
from pathlib import Path

def check_model_cache():
    """Check if model files exist in HuggingFace cache."""
    model_id = "IDEA-Research/grounding-dino-tiny"
    
    # Common cache locations
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface" / "transformers",
    ]
    
    print("=" * 60)
    print("CHECKING HUGGINGFACE CACHE")
    print("=" * 60)
    print(f"Model: {model_id}\n")
    
    
    found_files = []
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            print(f"Checking: {cache_dir}")
            # Look for model files
            patterns = [
                f"**/*{model_id.replace('/', '--')}*",
                f"**/*grounding*dino*tiny*",
                f"**/*grounding*dino*",
            ]
            for pattern in patterns:
                matches = list(cache_dir.glob(pattern))
                if matches:
                    found_files.extend(matches)
                    print(f"  Found {len(matches)} matches for pattern: {pattern}")
    
    if found_files:
        print(f"\n[OK] Model appears to be cached ({len(found_files)} files found)")
        print("  First few files:")
        for f in found_files[:5]:
            size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
            print(f"    {f.name} ({size_mb:.1f} MB)")
    else:
        print("\n[NOT FOUND] Model NOT found in cache - will download on first use")
    
    print("\n" + "=" * 60)
    print("TESTING DOWNLOAD WITH PROGRESS")
    print("=" * 60)
    
    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        import sys
        
        print("Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            model_id,
            force_download=False,  # Use cache if available
            resume_download=True,
        )
        print("[OK] Processor ready")
        
        print("\nDownloading model (this may take 5-10 minutes)...")
        print("Watch for progress bars below:\n")
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            force_download=False,
            resume_download=True,
        )
        print("\n[OK] Model downloaded/loaded successfully!")
        
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet: ping huggingface.co")
        print("2. Check disk space: df -h (Linux) or dir (Windows)")
        print("3. Try: pip install --upgrade transformers huggingface_hub")
        print("4. Set HF token if needed: export HF_TOKEN=your_token")

if __name__ == "__main__":
    check_model_cache()
