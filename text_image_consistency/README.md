# Text–Image Consistency Evaluation System

Research-grade pipeline for evaluating alignment between a **text prompt** and a **generated image**. The system assumes the image is already generated (e.g. by Stable Diffusion, DALL·E, or Gemini) and scores text–image consistency via a forward (text→image) and backward (image→text→image) path.

## Architecture (matches specification)

```
INPUT
  Dataset source: COCO / Visual Genome
  Inputs: Text Prompt, Ground Truth Image (optional); required: Text Prompt + Generated Image

GENERATION STAGE (assumed, not implemented)
  Supported sources: Stable Diffusion, DALL·E, Gemini
  Output: Generated Image  ← system starts from here

CORE EVALUATION PIPELINE

  FORWARD (Text → Image)
    1. CLIP          : Semantic alignment Text ↔ Generated Image
    2. BLIP          : Dense image caption (semantic grounding signal)
    3. Grounding DINO : Object-level grounding; entity presence
    4. TIFA          : Decompose prompt into atomic facts; faithfulness vs image + caption
    5. PaddleOCR      : Extract visible text; compare with prompt
    → Forward Score ∈ [0, 1]

  BACKWARD (Image → Text → Image)
    1. BLIP caption as reverse prompt
    2. CLIP(Generated Image, BLIP Caption)
    → Backward Score ∈ [0, 1]

FINAL OUTPUT
  Bidirectional Score = w1·Forward + w2·Backward
  Final Score ∈ [0, 100]
  Verdict: 80–100 MATCH | 50–79 PARTIAL MATCH | 0–49 MISMATCH
```

## Model roles

| Component   | Model |
|------------|--------|
| CLIP       | `openai/clip-vit-base-patch32` (HuggingFace) |
| BLIP       | `Salesforce/blip-image-captioning-base` |
| Grounding  | `IDEA-Research/grounding-dino-tiny` (HuggingFace; spec references IDEACo/GroundingDINO) |
| OCR        | PaddleOCR |
| TIFA       | Decomposition: rule-based; verification: `google/flan-t5-small` (LLM-based faithfulness) |

All inference is **CPU-only** and **deterministic** (fixed seed).

## Installation

```bash
cd text_image_consistency
pip install -r requirements.txt
```

- **Python**: 3.10 or 3.11 recommended.
- **PaddlePaddle**: If `paddlepaddle` does not install a CPU build by default, install the CPU build for your OS from [PaddlePaddle](https://www.paddlepaddle.org.cn/).

Optional: download NLTK data once (for any NLP helpers):

```bash
python -c "import nltk; nltk.download('punkt', quiet=True)"
```

### First-run model downloads

On first run, the system downloads pre-trained models (~3GB total):
- **BLIP**: ~990MB
- **CLIP**: ~500MB  
- **Grounding DINO**: ~1.4GB (largest, may take 5-10 minutes on slow connections)
- **FLAN-T5** (TIFA): ~300MB
- **PaddleOCR**: ~100MB

**If Grounding DINO hangs during download:**
1. Check internet connection
2. Ensure sufficient disk space (~5GB free)
3. The download should show progress; if stuck >10 minutes, cancel (Ctrl+C) and retry
4. Models are cached after first download; subsequent runs are faster

## Run

```bash
python src/main.py
```

From the project root `text_image_consistency/`, so that `src` is the package. If you are inside `src/`:

```bash
cd text_image_consistency
python -m src.main
```

Or:

```bash
cd text_image_consistency
python src/main.py
```

The script uses a hardcoded test image (`data/images/sample.jpg`) and prompt. If `sample.jpg` is missing, a minimal placeholder is created so the pipeline runs.

## Example output

```
============================================================
TEXT-IMAGE CONSISTENCY EVALUATION REPORT
============================================================
Image:     .../data/images/sample.jpg
Prompt:    A gray image with no distinct objects.
------------------------------------------------------------
BLIP caption:   ...
CLIP score:      0.xxxx
Grounding score: 0.xxxx
TIFA score:     0.xxxx
OCR score:      0.xxxx
Forward score:  0.xxxx
Backward score: 0.xxxx
------------------------------------------------------------
Final score:    xx.xx / 100
Verdict:        MATCH | PARTIAL MATCH | MISMATCH
============================================================
```

## Programmatic use

```python
from src.verify import evaluate

result = evaluate("path/to/image.jpg", "A red cat on a sofa.")
print(result["final_score"], result["verdict"])
```

## Limitations and future work

- **Grounding DINO**: Spec lists IDEACo/GroundingDINO; this implementation uses HuggingFace `IDEA-Research/grounding-dino-tiny` for CPU compatibility and a single API.
- **TIFA**: Full TIFA uses VQA for fact verification; here facts are verified against the BLIP caption via a small LM (FLAN-T5) and a rule-based overlap fallback.
- **OCR**: Best when the prompt explicitly mentions text (e.g. quoted strings); otherwise the OCR score is neutral.
- **Performance**: All models run on CPU; first run downloads weights and can be slow.
- **Reproducibility**: Fixed seed and CPU-only give deterministic results across runs on the same machine.

Future work: GPU support, full TIFA/VQA pipeline, more prompt sources (COCO/Visual Genome loaders), and configurable weights.
