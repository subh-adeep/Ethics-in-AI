"""
pull_kaggle_progress.py
=======================
Convenience script to pull the live execution logs, checkpoint metrics, and
temporary progress files from the Kaggle benchmark kernel to the local machine.
"""

import os
import sys
import subprocess
import json

KERNEL_SLUG = "subhadeepsing/week-4-agentic-tool-calling-t1-rag-t2-web-search"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_kaggle_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_status():
    print(f"[Kaggle] Checking status for: {KERNEL_SLUG}")
    res = subprocess.run(["kaggle", "kernels", "status", KERNEL_SLUG], capture_output=True, text=True)
    print(res.stdout.strip())
    return res.stdout.strip()

def pull_output():
    print(f"\n[Kaggle] Pulling latest output and live checkpoints to: {OUTPUT_DIR}")
    res = subprocess.run(["kaggle", "kernels", "output", KERNEL_SLUG, "-p", OUTPUT_DIR], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    # Scan for any downloaded JSON progress summaries
    found_any = False
    for fname in os.listdir(OUTPUT_DIR):
        if fname.endswith(".json"):
            found_any = True
            fpath = os.path.join(OUTPUT_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "metrics" in data:
                    print(f"\n--- [Summary from {fname}] ---")
                    print(f"Model: {data.get('model')} | Tier: {data.get('tier')}")
                    print(f"Completed: {data.get('completed')}/{data.get('total')}")
                    m = data["metrics"]
                    print(f"TUR: {m.get('tool_utilization_rate_pct')}% | Flip: {m.get('cumulative_flip_pct')}% | Resistant: {m.get('remained_resistant_pct')}%")
            except Exception:
                pass

    # Check for log files
    for fname in os.listdir(OUTPUT_DIR):
        if fname.endswith(".log") or "output" in fname.lower():
            fpath = os.path.join(OUTPUT_DIR, fname)
            print(f"\n--- [Tail of {fname}] ---")
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in lines[-25:]:
                print(line.rstrip())

if __name__ == "__main__":
    check_status()
    pull_output()
