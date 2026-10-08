"""
Model Loader for 4-bit Quantized Models on GTX 1650 (4GB VRAM).
Configures HF cache on D: drive and provides memory tracking and clean model unloading.
"""

import os
import gc
import time
from typing import Tuple, Optional
import torch

# Store all downloaded models strictly in the 'model' directory in the workspace root (on D: drive)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)

os.environ["HF_HOME"] = os.environ.get("HF_HOME", DEFAULT_MODEL_DIR)
os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", os.path.join(DEFAULT_MODEL_DIR, "torch_cache"))

import typing
import transformers.utils
if not hasattr(transformers.utils, "LossKwargs"):
    transformers.utils.LossKwargs = typing.TypedDict("LossKwargs", {})

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def get_gpu_memory_info() -> dict:
    """Returns allocated, reserved, and maximum VRAM in MiB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "max_allocated_mb": round(max_allocated, 2),
    }


def reset_peak_memory() -> None:
    """Resets the peak memory tracker in PyTorch."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def unload_model(model: Optional[PreTrainedModel] = None, tokenizer: Optional[PreTrainedTokenizer] = None) -> None:
    """Completely unloads the model from VRAM and forces garbage collection."""
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_model_and_tokenizer(
    model_id: str,
    device: str = "cuda",
    load_in_4bit: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, float]:
    """
    Loads a tokenizer and model in 4-bit precision with memory offload safety.

    Args:
        model_id: Hugging Face repository ID or local path.
        device: 'cuda' or 'cpu'.
        load_in_4bit: Whether to quantize weights using BitsAndBytes NF4.
        cache_dir: Directory to cache model files (defaults to D: drive hf_cache).

    Returns:
        Tuple of (model, tokenizer, load_duration_seconds)
    """
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", DEFAULT_MODEL_DIR)

    print(f"\n[Loader] Loading '{model_id}'...")
    print(f"[Loader] Storage location (HF_HOME): {cache_dir}")
    print(f"[Loader] Target hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    start_time = time.time()
    reset_peak_memory()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization config optimized for GTX 1650 (4GB)
    if load_in_4bit and device == "cuda" and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        # Cap GPU memory to 3.5GB to leave room for desktop/WDDM and prevent OOM
        max_memory = {0: "3500MiB", "cpu": "16GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            max_memory=max_memory,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

    load_duration = time.time() - start_time
    mem = get_gpu_memory_info()
    print(f"[Loader] Loaded '{model_id}' in {load_duration:.2f}s | VRAM Allocated: {mem['allocated_mb']} MB (Peak: {mem['max_allocated_mb']} MB)")

    return model, tokenizer, load_duration
