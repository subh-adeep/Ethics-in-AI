"""
Benchmarking Suite for GTX 1650 (4GB VRAM).
Measures memory usage (allocated & peak VRAM), inference latency (tokens/sec),
and validates tool calling capabilities using DuckDuckGo Web Search.
"""

import time
import json
import re
from typing import Dict, Any, Optional
import torch

from src.models.loader import (
    load_model_and_tokenizer,
    unload_model,
    get_gpu_memory_info,
    reset_peak_memory,
)
from src.tools.search import duckduckgo_search, format_search_results, SEARCH_TOOL_DEFINITION


def run_latency_benchmark(
    model,
    tokenizer,
    prompt: str = "Explain in two sentences what the principle of epistemic integrity means in artificial intelligence.",
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    """Measures generation throughput (tokens/sec) and peak memory."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    reset_peak_memory()
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    total_time = time.time() - start_time
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs.shape[1] - input_len
    tokens_per_sec = generated_tokens / total_time if total_time > 0 else 0.0
    
    mem = get_gpu_memory_info()
    generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    return {
        "generated_tokens": generated_tokens,
        "total_time_sec": round(total_time, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "peak_vram_mb": mem["max_allocated_mb"],
        "generated_sample": generated_text.strip(),
    }


def parse_tool_call(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parses tool call from generated text across common LLM output formats
    (Qwen/Hermes/JSON tool tags or markdown json blocks).
    """
    # 1. Look for <tool_call> ... </tool_call>
    tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL)
    if tool_call_match:
        try:
            return json.loads(tool_call_match.group(1).strip())
        except Exception:
            pass

    # 2. Look for ```json ... ``` containing 'name' or 'query'
    json_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    for block in json_blocks:
        try:
            data = json.loads(block)
            if "name" in data or "query" in data or "arguments" in data:
                return data
        except Exception:
            pass

    # 3. Direct JSON object
    direct_match = re.search(r"\{[\s\S]*?\"name\"[\s\S]*?\}", response_text)
    if direct_match:
        try:
            return json.loads(direct_match.group(0))
        except Exception:
            pass

    return None


def run_tool_calling_test(
    model,
    tokenizer,
    test_query: str = "Search the web for the latest space exploration news using duckduckgo_search.",
) -> Dict[str, Any]:
    """
    Tests whether the model emits a structured tool call, executes the tool,
    feeds back the search results, and completes the synthesis.
    """
    device = next(model.parameters()).device
    tool_prompt = (
        "You are an AI assistant equipped with external search capabilities.\n"
        "You have access to the following tool:\n"
        f"{json.dumps(SEARCH_TOOL_DEFINITION, indent=2)}\n\n"
        "To invoke this tool, output ONLY a JSON object formatted as:\n"
        '```json\n{"name": "duckduckgo_search", "arguments": {"query": "<your search query>"}}\n```\n'
        f"User: {test_query}\nAssistant:"
    )

    inputs = tokenizer(tool_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    first_turn_response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    tool_call = parse_tool_call(first_turn_response)

    test_result = {
        "tool_call_generated": tool_call is not None,
        "first_turn_raw": first_turn_response.strip(),
        "parsed_tool_call": tool_call,
        "tool_execution_success": False,
        "final_answer": "",
    }

    # If the model requested a tool call or mentioned query
    query_to_search = None
    if tool_call:
        if "arguments" in tool_call and isinstance(tool_call["arguments"], dict):
            query_to_search = tool_call["arguments"].get("query")
        elif "query" in tool_call:
            query_to_search = tool_call["query"]

    if not query_to_search:
        query_to_search = "latest space exploration news"

    # Execute the search tool
    search_results = duckduckgo_search(query_to_search, max_results=2)
    test_result["tool_execution_success"] = len(search_results) > 0 and search_results[0]["title"] != "Search Error"
    formatted_results = format_search_results(search_results)

    # Feed results back for final answer
    followup_prompt = (
        f"{tool_prompt}\n{first_turn_response}\n\n"
        f"Tool Result:\n{formatted_results}\n\n"
        "Assistant (summarizing retrieved information):"
    )
    followup_inputs = tokenizer(followup_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        followup_outputs = model.generate(
            **followup_inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    final_response = tokenizer.decode(followup_outputs[0][followup_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    test_result["final_answer"] = final_response.strip()
    return test_result


def benchmark_model(
    model_id: str,
    load_in_4bit: bool = True,
    vram_budget_mb: float = 4096.0,
) -> Dict[str, Any]:
    """Runs full benchmark pipeline for a single model."""
    print(f"\n==========================================")
    print(f"BENCHMARKING: {model_id}")
    print(f"==========================================")
    
    baseline_mem = get_gpu_memory_info()
    model, tokenizer, load_duration = load_model_and_tokenizer(model_id, load_in_4bit=load_in_4bit)
    loaded_mem = get_gpu_memory_info()

    print("\n--> Running generation latency benchmark...")
    latency_stats = run_latency_benchmark(model, tokenizer)
    print(f"    Speed: {latency_stats['tokens_per_sec']} tokens/s | Peak VRAM: {latency_stats['peak_vram_mb']} MB")

    print("\n--> Running DuckDuckGo tool calling test...")
    tool_stats = run_tool_calling_test(model, tokenizer)
    print(f"    Tool Call Emitted: {tool_stats['tool_call_generated']}")
    print(f"    Tool Executed: {tool_stats['tool_execution_success']}")

    peak_vram = max(latency_stats["peak_vram_mb"], loaded_mem["max_allocated_mb"])
    feasible = peak_vram <= (vram_budget_mb * 0.95) # 95% threshold of 4GB

    result = {
        "model_id": model_id,
        "load_duration_sec": round(load_duration, 2),
        "baseline_vram_mb": baseline_mem["allocated_mb"],
        "model_vram_mb": loaded_mem["allocated_mb"],
        "peak_vram_mb": round(peak_vram, 2),
        "vram_limit_mb": vram_budget_mb,
        "vram_headroom_mb": round(vram_budget_mb - peak_vram, 2),
        "is_feasible_gtx1650": feasible,
        "tokens_per_sec": latency_stats["tokens_per_sec"],
        "tool_call_supported": tool_stats["tool_call_generated"],
        "tool_stats": tool_stats,
        "latency_stats": latency_stats,
    }

    # Clean up model from memory
    print("\n--> Unloading model to free VRAM...")
    unload_model(model, tokenizer)
    after_mem = get_gpu_memory_info()
    print(f"    VRAM after cleanup: {after_mem['allocated_mb']} MB")
    
    return result
