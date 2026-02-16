"""
Debug TIFA score to see what's happening
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.tifa import _decompose_prompt_into_facts, _fact_in_caption_llm, _fact_in_caption_overlap, _get_tifa_model

prompt = "A gray image with no distinct objects."
caption = "a black and white photo of a man in a suit and tie"

print("=" * 60)
print("TIFA DEBUGGING")
print("=" * 60)
print(f"\nPrompt:  {prompt}")
print(f"Caption: {caption}")

# Test fact decomposition
facts = _decompose_prompt_into_facts(prompt)
print(f"\n[1] Decomposed Facts: {facts}")

# Test overlap method
print(f"\n[2] Testing Overlap Method:")
for fact in facts:
    overlap_result = _fact_in_caption_overlap(fact, caption)
    print(f"    - '{fact}' -> Overlap: {overlap_result}")

# Test LLM method
print(f"\n[3] Testing LLM Method:")
try:
    model, tokenizer = _get_tifa_model()
    print(f"    Model loaded: {model.__class__.__name__}")
    print(f"    Tokenizer: {tokenizer.__class__.__name__}")
    
    for fact in facts:
        print(f"\n    Testing fact: '{fact}'")
        
        # Run the LLM check
        inp = f"Does the following description support the claim? Description: {caption[:200]}. Claim: {fact[:100]}. Answer:"
        print(f"    Input: {inp[:100]}...")
        
        inputs = tokenizer(inp, return_tensors="pt", truncation=True, max_length=256)
        print(f"    Input tokens shape: {inputs['input_ids'].shape}")
        
        import torch
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4)
        
        ans = tokenizer.decode(out[0], skip_special_tokens=True).strip().lower()
        print(f"    Raw answer: '{ans}'")
        print(f"    Parsed (yes in first 10): {'yes' in ans[:10]}")
        
except Exception as e:
    print(f"    ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
