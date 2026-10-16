"""Run independent benchmark models on two GPUs in separate processes.

For the next Kaggle session, with runner.py and manifest.json beside this file:
python run_dual_gpu.py --models llama phi4
This cannot attach a worker to an already-running Kaggle batch session.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from runner import atomic_json


def restore_checkpoint(source, destination, worker_path, manifest, model, tiers, per_domain):
    """Restore only compatible raw replies; preserve the original separately.

    The only permitted settings migration is reducing the scheduled tier list.
    It does not change any model input, weights, decoding setting or reply.
    """
    saved = json.loads(Path(source).read_text(encoding="utf-8"))
    settings = saved["settings"]
    source_hash = hashlib.sha256(Path(worker_path).read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    items = manifest["items"]
    if per_domain:
        items = [i for domain in ("medqa", "math")
                 for i in [x for x in items if x["dataset"] == domain][:per_domain]]
    expected_ids = [i["id"] for i in items]
    expected_model = {"qwen": "Qwen/Qwen2.5-3B-Instruct",
        "llama": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "phi4": "unsloth/Phi-4-mini-instruct-bnb-4bit"}[model]
    if (settings["source_sha256"] != source_hash or settings["manifest_sha256"] != manifest_hash
        or settings["item_ids"] != expected_ids or settings["model_id"] != expected_model
        or settings["policy"] != "optional" or settings["precision"] != "4bit"
        or settings["tier"] not in tiers or not set(tiers).issubset(settings["tiers"])):
        raise ValueError(f"Refusing incompatible resume checkpoint: {source}")
    rows = saved["dialogues"]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or not set(ids).issubset(expected_ids) or saved["completed"] != len(rows):
        raise ValueError(f"Invalid checkpoint rows: {source}")
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"Refusing to overwrite an existing working checkpoint: {destination}")
    raw_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    original_settings = dict(settings)
    # Scheduling one tier at a time changes only this bookkeeping field.
    saved["settings"] = dict(settings, tiers=list(tiers))
    atomic_json(destination, saved)
    atomic_json(destination.parent / "resume_receipt.json", {
        "original_checkpoint_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "worker_source_sha256": source_hash, "original_settings": original_settings,
        "scheduled_tiers": list(tiers), "preserved_dialogues": len(rows),
        "dialogues_sha256": raw_hash, "preserved_ids": ids})
    return len(rows)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["qwen", "llama", "phi4"], default=["llama", "phi4"])
    parser.add_argument("--runner", default=str(here / "runner.py"))
    parser.add_argument("--manifest", default=str(here / "manifest.json"))
    parser.add_argument("--tiers", nargs="+", choices=["T0", "T1", "T2"], default=["T0", "T1", "T2"])
    parser.add_argument("--per-domain", type=int, default=0)
    parser.add_argument("--output", default="/kaggle/working/dual_gpu")
    parser.add_argument("--resume-root")
    parser.add_argument("--stagewise", action="store_true", help="Schedule model/tier tasks separately to balance GPUs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("Each model may be scheduled only once.")
    if args.per_domain < 0:
        parser.error("--per-domain must be nonnegative.")
    runner_path, manifest_path = Path(args.runner).resolve(), Path(args.manifest).resolve()
    if not runner_path.exists() or not manifest_path.exists():
        parser.error("The runner and frozen manifest files must exist.")

    tasks = [(m, [t]) for t in args.tiers for m in args.models] if args.stagewise else [(m, args.tiers) for m in args.models]

    def task_name(task):
        model, tiers = task
        return model + ("_" + tiers[0] if args.stagewise else "")

    def command(task):
        model, tiers = task
        return [sys.executable, "-u", str(runner_path), "--manifest", str(manifest_path),
                "--models", model, "--tiers", *tiers, "--per-domain", str(args.per_domain),
                "--output", str(Path(args.output) / task_name(task))]

    if args.dry_run:
        print(json.dumps({"initial_workers": [{"task": task_name(task), "physical_gpu": i, "command": command(task)}
                          for i, task in enumerate(tasks[:2])],
                          "queued_tasks": [task_name(task) for task in tasks[2:]], "execution": "separate processes"}, indent=2))
        return 0

    # Query visibility in a disposable process before initializing either model.
    count = int(subprocess.check_output([sys.executable, "-c",
                 "import torch; print(torch.cuda.device_count())"], text=True).strip())
    if count < 2:
        raise RuntimeError(f"Two visible CUDA GPUs are required; found {count}.")
    if Path("/kaggle").exists():
        # Install once in the parent; simultaneous pip writes would be unsafe.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1",
                        "accelerate>=1.2,<2", "bitsandbytes==0.48.1", "rank-bm25==0.2.2",
                        "ddgs==9.16.0", "primp==2.0.1"], check=True)

    pending = deque(tasks)
    active, finished = {}, []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.resume_root:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resume_root = Path(args.resume_root)
        for task in tasks:
            model, tiers = task
            target = output / task_name(task)
            for tier in tiers:
                filename = f"{model}_{tier}_optional.json"
                matches = list(resume_root.rglob(filename))
                if len(matches) > 1:
                    raise ValueError(f"Multiple resume candidates found for {filename}")
                if matches:
                    n = restore_checkpoint(matches[0], target / filename, runner_path, manifest, model, tiers, args.per_domain)
                    print(f"[Resume] {model} {tier}: {n} completed dialogues preserved; those IDs will be skipped", flush=True)
                    cache = matches[0].parent / "web_cache.json"
                    if cache.exists():
                        atomic_json(target / "web_cache.json", json.loads(cache.read_text(encoding="utf-8")))

    def snapshot():
        atomic_json(output / "workers.json", {
            "pending": [task_name(t) for t in pending], "active": [{"gpu": gpu, "task": job["name"],
             "pid": job["process"].pid} for gpu, job in active.items()], "finished": finished})

    try:
        while pending or active:
            for gpu in (0, 1):
                if pending and gpu not in active:
                    task = pending.popleft()
                    name = task_name(task)
                    env = os.environ.copy()
                    # Each worker sees one physical GPU as logical cuda:0.
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    env["ETHICS_SKIP_DEPENDENCY_INSTALL"] = "1"
                    env["PYTHONUNBUFFERED"] = "1"
                    # The exact frozen worker may still contain its installer.
                    # Parent installation is complete; workers resolve locally.
                    env["PIP_NO_INDEX"] = "1"
                    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
                    process = subprocess.Popen(command(task), env=env)
                    active[gpu] = {"name": name, "process": process}
                    print(f"[Launcher] {name} started on physical GPU {gpu}", flush=True)
                    snapshot()
            for gpu, job in list(active.items()):
                result = job["process"].poll()
                if result is not None:
                    finished.append({"gpu": gpu, "task": job["name"], "exit_code": result})
                    print(f"[Launcher] {job['name']} exited with code {result}", flush=True)
                    del active[gpu]
                    snapshot()
            if active:
                time.sleep(2)
    finally:
        # An interrupted launcher must not leave detached inference workers.
        for job in active.values():
            job["process"].terminate()
        for job in active.values():
            try:
                job["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                job["process"].kill()
                job["process"].wait()
    return 1 if any(job["exit_code"] != 0 for job in finished) else 0


if __name__ == "__main__":
    raise SystemExit(main())
