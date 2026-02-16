"""
Test TIFA with matching and non-matching prompts to verify it works correctly
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.tifa import _decompose_prompt_into_facts, _fact_in_caption_overlap, _get_tifa_model

print("=" * 70)
print("TIFA CORRECTNESS TEST")
print("=" * 70)

test_cases = [
    {
        "name": "MISMATCH - Contradictory prompt",
        "prompt": "A gray image with no distinct objects.",
        "caption": "a black and white photo of a man in a suit and tie",
        "expected_score": 0.0  # Should be 0 because prompt contradicts caption
    },
    {
        "name": "MATCH - Supporting prompt",
        "prompt": "A man in a suit and tie.",
        "caption": "a black and white photo of a man in a suit and tie",
        "expected_score": 1.0  # Should be 1.0 because facts match caption
    },
    {
        "name": "PARTIAL MATCH - Some supporting facts",
        "prompt": "A man wearing a suit with a tie and a hat.",
        "caption": "a black and white photo of a man in a suit and tie",
        "expected_score": 0.67  # 2 out of 3 facts match (hat is not mentioned)
    }
]

model, tokenizer = _get_tifa_model()

for test_case in test_cases:
    print(f"\nTest: {test_case['name']}")
    print(f"Prompt:  {test_case['prompt']}")
    print(f"Caption: {test_case['caption']}")
    
    facts = _decompose_prompt_into_facts(test_case['prompt'])
    print(f"Facts:   {facts}")
    
    results = []
    for fact in facts:
        supported = _fact_in_caption_overlap(fact, test_case['caption'])
        print(f"  - '{fact}' → {supported}")
        results.append(supported)
    
    score = sum(results) / len(results) if results else 0.0
    print(f"Score: {score:.4f} (expected ~{test_case['expected_score']:.4f})")
    
    if abs(score - test_case['expected_score']) <= 0.1:
        print("✓ PASS")
    else:
        print(f"⚠ Score differs from expectation")

print("\n" + "=" * 70)
print("IMPORTANT: TIFA Score of 0.0 is CORRECT when prompt contradicts caption!")
print("=" * 70)
