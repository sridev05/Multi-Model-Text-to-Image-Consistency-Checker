================================================================================
RUN MODELS - Standalone Execution
================================================================================
Image path and prompt are hardcoded (data/images/sample.jpg, "a lion face in
black and white color"). Run from project root: text_image_consistency/

Prerequisites:
  pip install -r requirements.txt

--------------------------------------------------------------------------------
RUN COMMANDS (simple - no args needed)
--------------------------------------------------------------------------------

  cd D:\projects\t2i\text_image_consistency

  python run_models/run_clip.py
  python run_models/run_blip.py
  python run_models/run_grounding.py
  python run_models/run_tifa.py
  python run_models/run_ocr.py
  python run_models/run_all.py

--------------------------------------------------------------------------------
What each outputs
--------------------------------------------------------------------------------
  run_clip.py      -> clip_score [0, 1]
  run_blip.py      -> blip_caption (string)
  run_grounding.py -> grounding_score [0, 1]
  run_tifa.py      -> tifa_score [0, 1]
  run_ocr.py       -> ocr_score [0, 1]
  run_all.py       -> all five outputs

--------------------------------------------------------------------------------
Optional: override image/prompt
--------------------------------------------------------------------------------
  python run_models/run_clip.py --image path/to/img.jpg --text "your prompt"
  python run_models/run_tifa.py --image path/to/img.jpg --text "your prompt"
  python run_models/run_ocr.py --image path/to/img.jpg --text "your prompt"

================================================================================
