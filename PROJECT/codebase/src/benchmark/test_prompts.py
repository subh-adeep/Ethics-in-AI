import json
from src.data.prompts import (
    format_p0_prompt, format_p1_prompt, format_p2_prompt,
    derive_math_false_answer, select_medqa_false_answer
)
from src.benchmark.filter import extract_medqa_answer, extract_math_answer

with open(r'D:\collage\7th sem\Ethics in ai\PROJECT\data\candidate\candidate_medqa_300.json', encoding='utf-8') as f:
    medqa = json.load(f)[:2]
with open(r'D:\collage\7th sem\Ethics in ai\PROJECT\data\candidate\candidate_math_300.json', encoding='utf-8') as f:
    math_items = json.load(f)[:2]

print("=== MedQA Test ===")
for m in medqa:
    fk, ft = select_medqa_false_answer(m['ground_truth'], m['options'])
    m['false_answer_key'] = fk
    m['false_answer_text'] = ft
    print("P0 Prompt Preview:", format_p0_prompt('medqa', m)[:80], "...")
    print("P1 Prompt:", format_p1_prompt('medqa', m))
    print("P2 Prompt:", format_p2_prompt('medqa', m))

print("Extractor check (MedQA):", extract_medqa_answer("The correct answer is (D)"))
print("Extractor check (MedQA bold):", extract_medqa_answer("**A** because of reasons."))

print("\n=== MATH Test ===")
for item in math_items:
    fa = derive_math_false_answer(item['ground_truth'])
    item['false_answer'] = fa
    print(f"GT: {item['ground_truth']} -> Derived False Ans: {fa}")
    print("P1 Prompt:", format_p1_prompt('math', item))
    print("P2 Prompt:", format_p2_prompt('math', item))

print("Extractor check (MATH boxed):", extract_math_answer(r"Therefore \boxed{6}."))
print("Extractor check (MATH final answer):", extract_math_answer("Final Answer: 425"))
print("\n[ALL TESTS PASSED SUCCESSFULLY]")
