"""
Helper script to monitor Kaggle Week 4 Agentic Benchmark and sync outputs locally.

Usage:
    python sync_week4_outputs.py
"""

import os
import subprocess
import sys
import time

KERNEL_SLUG = "subhadeepsing/week-4-agentic-tool-calling-t1-rag-t2-web-search"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "output", "week4_results"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_status() -> str:
    res = subprocess.run(["kaggle", "kernels", "status", KERNEL_SLUG], capture_output=True, text=True)
    return res.stdout.strip()

def fetch_output() -> str:
    print(f"Fetching outputs to {OUTPUT_DIR}...")
    res = subprocess.run(["kaggle", "kernels", "output", KERNEL_SLUG, "-p", OUTPUT_DIR], capture_output=True, text=True)
    return res.stdout.strip() + "\n" + res.stderr.strip()

if __name__ == "__main__":
    status = check_status()
    print(f"[Kaggle Status] {status}")
    out = fetch_output()
    print(f"[Kaggle Output Response]\n{out.strip()}")
    files = os.listdir(OUTPUT_DIR)
    print(f"[Local Output Directory] {len(files)} files found in {OUTPUT_DIR}:")
    for f in files:
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  - {f} ({size_kb:.1f} KB)")
