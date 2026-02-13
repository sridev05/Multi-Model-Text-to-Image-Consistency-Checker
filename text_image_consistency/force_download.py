"""
Force download Grounding DINO with explicit progress.
"""
import sys
import time
from pathlib import Path

print("=" * 60)
print("FORCING GROUNDING DINO DOWNLOAD")
print("=" * 60)
print("Model: IDEA-Research/grounding-dino-tiny")
print("Size: ~1.4 GB")
print("\nThis will:")
print("1. Clear incomplete cache (if any)")
print("2. Download model with progress bars")
print("3. Load model to verify")
print("\nPress Ctrl+C to cancel\n")

try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from huggingface_hub import snapshot_download, hf_hub_download
    
    model_id = "IDEA-Research/grounding-dino-tiny"
    
    # Check cache first
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_slug = model_id.replace("/", "--")
    cache_path = cache_dir / f"models--{model_slug}"
    
    if cache_path.exists():
        print(f"Found incomplete cache at: {cache_path}")
        print("Will download fresh copy...\n")
    
    print("[1/3] Downloading processor...")
    start = time.time()
    processor = AutoProcessor.from_pretrained(
        model_id,
        force_download=True,  # Force fresh download
    )
    print(f"      Done in {time.time() - start:.1f}s\n")
    
    print("[2/3] Downloading model weights...")
    print("      (Watch for progress bars - this may take 5-15 minutes)\n")
    start = time.time()
    
    # Use hf_hub_download for better progress visibility
    try:
        # Download main model file
        model_file = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
            force_download=True,
            resume_download=True,
        )
        print(f"      Downloaded: {Path(model_file).name}")
    except Exception as e:
        print(f"      Note: {e}")
        print("      Trying standard from_pretrained...")
    
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        force_download=True,
    )
    elapsed = time.time() - start
    print(f"\n      Done in {elapsed:.1f}s ({elapsed/60:.1f} minutes)\n")
    
    print("[3/3] Verifying model loads...")
    model.eval()
    print("      [OK] Model loaded and ready!\n")
    
    print("=" * 60)
    print("SUCCESS: Grounding DINO is ready to use!")
    print("=" * 60)
    print("\nYou can now run: python src/main.py")
    
except KeyboardInterrupt:
    print("\n\nCancelled by user.")
    print("Partial download may be cached - will resume on next run.")
    sys.exit(1)
except Exception as e:
    print(f"\n\nERROR: {type(e).__name__}: {e}")
    print("\nTroubleshooting:")
    print("1. Internet connection: ping huggingface.co")
    print("2. Disk space: Need ~5GB free")
    print("3. Firewall: May block HuggingFace downloads")
    print("4. Try: pip install --upgrade transformers huggingface_hub")
    print("5. Check HuggingFace status: https://status.huggingface.co/")
    import traceback
    traceback.print_exc()
    sys.exit(1)
