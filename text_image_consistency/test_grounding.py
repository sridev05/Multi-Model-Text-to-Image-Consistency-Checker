"""
Diagnostic script to test Grounding DINO loading separately.
Run this to check if Grounding DINO downloads/loads correctly.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("GROUNDING DINO LOAD TEST")
print("=" * 60)
print("This will download ~1.4GB model if not cached.")
print("Press Ctrl+C to cancel if stuck.\n")

try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    
    model_id = "IDEA-Research/grounding-dino-tiny"
    


    print(f"[1/2] Loading processor from {model_id}...")
    start = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"      ✓ Processor loaded in {time.time() - start:.1f}s\n")
    
    print(f"[2/2] Loading model from {model_id}...")
    print("      (This may take 2-5 minutes on first download)")
    start = time.time()
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    elapsed = time.time() - start
    print(f"      ✓ Model loaded in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Grounding DINO is working correctly!")
    print("=" * 60)
    
except KeyboardInterrupt:
    print("\n\nCancelled by user.")
    sys.exit(1)
except Exception as e:
    print(f"\n\nERROR: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Ensure ~5GB free disk space")
    print("3. Try: pip install --upgrade transformers")
    print("4. Check HuggingFace Hub status: https://status.huggingface.co/")
    sys.exit(1)
