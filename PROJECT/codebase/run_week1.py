"""
Week 1 Orchestrator: Local Inference, Benchmarking & Tool-Calling Validation.

Models evaluated on NVIDIA GeForce GTX 1650 (4GB VRAM):
1. Qwen 2.5 3B Instruct (4-bit)
2. Phi-4 Mini 3.8B Instruct (4-bit)
3. LLaMA 3.2 3B Instruct (4-bit)
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Setup paths and environment variables
ROOT_DIR = Path(__file__).resolve().parent
CODEBASE_DIR = ROOT_DIR
SRC_DIR = CODEBASE_DIR / "src"
RESULTS_DIR = CODEBASE_DIR / "results"
HF_CACHE_DIR = ROOT_DIR.parent / "model"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODEBASE_DIR))
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TORCH_HOME"] = str(ROOT_DIR.parent / "model" / "torch_cache")

import torch
import typing
import transformers.utils
if not hasattr(transformers.utils, "LossKwargs"):
    transformers.utils.LossKwargs = typing.TypedDict("LossKwargs", {})

from src.tools.search import duckduckgo_search, format_search_results
from src.benchmark.benchmark import benchmark_model


MODELS_TO_BENCHMARK = [
    {
        "name": "Qwen 2.5 3B Instruct",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "load_in_4bit": True,
        "alt_model_id": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
    },
    {
        "name": "Phi-4 Mini 3.8B Instruct",
        "model_id": "unsloth/Phi-4-mini-instruct-bnb-4bit",
        "load_in_4bit": True,
        "alt_model_id": "microsoft/Phi-4-mini-instruct",
    },
    {
        "name": "LLaMA 3.2 3B Instruct",
        "model_id": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "load_in_4bit": True,
        "alt_model_id": "unsloth/Llama-3.2-3B-Instruct",
    },
]


def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"   {title}")
    print("=" * 65)


def verify_search_tool():
    print_header("STEP 1: Standalone DuckDuckGo Search Verification")
    query = "latest space exploration news"
    print(f"Executing search for query: '{query}'...")
    results = duckduckgo_search(query, max_results=3)
    if not results or results[0]["title"] == "Search Error":
        print("[WARNING] DuckDuckGo search did not return results or encountered an error.")
        print(f"Details: {results}")
        return False
    
    print("[SUCCESS] DuckDuckGo returned results:")
    for i, res in enumerate(results, 1):
        print(f"  [{i}] {res['title']} -> {res['href'][:60]}...")
    return True


def run_week1_benchmarks(target_model_name: str = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print_header("ETHICS IN AI - WEEK 1 BENCHMARK & TOOL VALIDATION")
    print(f"Storage Directory (HDD): {HF_CACHE_DIR}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"Total VRAM: {total_vram:.1f} MiB")
    else:
        print("[WARNING] CUDA not detected! Models will attempt to run on CPU.")

    # 1. Verify search
    search_ok = verify_search_tool()

    # 2. Benchmark each model
    benchmark_results = []
    models_to_run = MODELS_TO_BENCHMARK
    if target_model_name:
        models_to_run = [m for m in MODELS_TO_BENCHMARK if target_model_name.lower() in m["name"].lower() or target_model_name.lower() in m["model_id"].lower()]

    for cfg in models_to_run:
        model_name = cfg["name"]
        model_id = cfg["model_id"]
        print_header(f"STEP 2: Evaluating {model_name}")
        try:
            res = benchmark_model(model_id=model_id, load_in_4bit=cfg.get("load_in_4bit", True))
            res["display_name"] = model_name
            benchmark_results.append(res)
        except Exception as e:
            print(f"[ERROR] Failed to benchmark {model_id}: {e}")
            # Try alternate model id if available (e.g. GPTQ)
            if "alt_model_id" in cfg:
                alt_id = cfg["alt_model_id"]
                print(f"[RETRY] Attempting alternative model: {alt_id}")
                try:
                    res = benchmark_model(model_id=alt_id, load_in_4bit=False)
                    res["display_name"] = model_name
                    benchmark_results.append(res)
                    continue
                except Exception as e2:
                    print(f"[ERROR] Alternative {alt_id} also failed: {e2}")

            benchmark_results.append({
                "model_id": model_id,
                "display_name": model_name,
                "error": str(e),
                "is_feasible_gtx1650": False,
            })

    # 3. Save report (merging with any prior benchmark results)
    report_file = RESULTS_DIR / "week1_benchmark_report.json"
    all_models_dict = {}
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                for m in existing_data.get("models", []):
                    all_models_dict[m.get("model_id")] = m
        except Exception:
            pass

    for m in benchmark_results:
        all_models_dict[m.get("model_id")] = m

    merged_models_list = list(all_models_dict.values())
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "search_verified": search_ok,
            "models": merged_models_list,
        }, f, indent=2)

    benchmark_results = merged_models_list

    print_header("BENCHMARK SUMMARY REPORT")
    print(f"{'Model':<25} | {'Load(s)':<8} | {'Peak VRAM':<10} | {'Tok/s':<8} | {'Tool Call':<10} | {'Feasible (4GB)':<14}")
    print("-" * 86)
    for b in benchmark_results:
        if "error" in b:
            print(f"{b['display_name']:<25} | {'ERROR':<8} | {'N/A':<10} | {'N/A':<8} | {'N/A':<10} | {'FAIL':<14}")
        else:
            load_s = f"{b['load_duration_sec']}s"
            vram = f"{b['peak_vram_mb']:.0f} MB"
            toks = f"{b['tokens_per_sec']:.1f}"
            tool = "YES" if b.get("tool_call_supported") else "NO"
            feasible = "PASS" if b.get("is_feasible_gtx1650") else "FAIL"
            print(f"{b['display_name']:<25} | {load_s:<8} | {vram:<10} | {toks:<8} | {tool:<10} | {feasible:<14}")
    
    print(f"\nReport written to: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 1 Benchmark and Tool Calling Suite")
    parser.add_argument("--model", type=str, default=None, help="Specific model to benchmark (qwen, phi, llama)")
    args = parser.parse_args()
    run_week1_benchmarks(args.model)
