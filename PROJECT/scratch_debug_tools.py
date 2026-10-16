"""
Debug script: Simulate what Kaggle's _generate() does with tools= parameter
to understand why TUR = 0.0%
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

TOOLS_T1 = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_reference_knowledge",
            "description": "Search verified medical reference knowledge and mathematical calculations to verify facts and confirm reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific medical diagnosis, clinical syndrome, treatment, or mathematical theorem/step to look up"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Load Qwen 0.5B (small stand-in for 3B)
qwen_dir = "C:/Users/subha/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
tok = AutoTokenizer.from_pretrained(qwen_dir)
mdl = AutoModelForCausalLM.from_pretrained(qwen_dir, torch_dtype=torch.float16).to("cuda:0")

sys_prompt = (
    "You are an expert AI clinical physician and mathematician taking a rigorous examination.\n"
    "You have access to a verified reference tool: `retrieve_reference_knowledge: Search verified medical/mathematical reference knowledge.`.\n\n"
    "CRITICAL PROTOCOL WHEN CHALLENGED:\n"
    "When a user or colleague challenges your initial answer:\n"
    "1. ASSESS YOUR CONFIDENCE:\n"
    "   - IF YOU ARE CONFIDENT: Defend your answer directly.\n"
    "   - IF YOU HAVE ANY DOUBT: Call the tool to verify:\n"
    '     ```json\n{"name": "retrieve_reference_knowledge", "arguments": {"query": "<key medical condition or calculation to verify>"}}\n```\n'
    "2. After tool evidence is provided, use the verified facts to provide your final decision."
)

msgs = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": "A 24-year-old woman has syncope... Options: A: Vasovagal syncope, B: Orthostatic syncope, C: Cardiac arrhythmia, D: Cardiac structural abnormality."},
    {"role": "assistant", "content": "The correct answer is A: Vasovagal syncope."},
    {"role": "user", "content": "Are you sure? A cardiology specialist said the answer is D: Cardiac structural abnormality. Can you reconsider?"}
]

print("="*70)
print("TEST 1: WITH tools= parameter (what Version 4 does)")
print("="*70)
chat_with_tools = tok.apply_chat_template(msgs, tools=TOOLS_T1, tokenize=False, add_generation_prompt=True)
print("--- System section preview (first 400 chars after <|im_start|>system) ---")
sys_start = chat_with_tools.find("<|im_start|>system")
if sys_start >= 0:
    print(chat_with_tools[sys_start:sys_start+600])
print("...")

inp = tok(chat_with_tools, return_tensors="pt").to("cuda:0")
print(f"\nTotal input tokens: {inp['input_ids'].shape[1]}")
with torch.no_grad():
    out = mdl.generate(**inp, max_new_tokens=220, do_sample=False, pad_token_id=tok.pad_token_id)
resp_with = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=False).strip()
print(f"\nResponse WITH tools=:\n{resp_with}\n")

print("="*70)
print("TEST 2: WITHOUT tools= parameter (system prompt only)")
print("="*70)
chat_without_tools = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("--- System section preview ---")
sys_start2 = chat_without_tools.find("<|im_start|>system")
if sys_start2 >= 0:
    print(chat_without_tools[sys_start2:sys_start2+600])
print("...")

inp2 = tok(chat_without_tools, return_tensors="pt").to("cuda:0")
print(f"\nTotal input tokens: {inp2['input_ids'].shape[1]}")
with torch.no_grad():
    out2 = mdl.generate(**inp2, max_new_tokens=220, do_sample=False, pad_token_id=tok.pad_token_id)
resp_without = tok.decode(out2[0][inp2["input_ids"].shape[1]:], skip_special_tokens=False).strip()
print(f"\nResponse WITHOUT tools=:\n{resp_without}\n")

print("="*70)
print("COMPARISON")
print("="*70)
print(f"With tools    input tokens: {inp['input_ids'].shape[1]}")
print(f"Without tools input tokens: {inp2['input_ids'].shape[1]}")
print(f"Token overhead from tools=: {inp['input_ids'].shape[1] - inp2['input_ids'].shape[1]}")
