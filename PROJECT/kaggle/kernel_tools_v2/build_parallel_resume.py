"""Package the exact worker and preserved checkpoints for a two-GPU restart."""
import base64
import hashlib
import json
from pathlib import Path
import zlib


def build():
    here = Path(__file__).resolve().parent
    source = (here / "resume_worker.py").read_bytes()
    worker_hash = hashlib.sha256(source).hexdigest()
    capture = here / "resume_capture" / "tools_v2"
    checkpoint = json.loads((capture / "qwen_T0_optional.json").read_text(encoding="utf-8"))
    if checkpoint["settings"]["source_sha256"] != worker_hash:
        raise ValueError("Frozen worker does not match the stopped experiment.")
    manifest = json.loads((here / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() != checkpoint["settings"]["manifest_sha256"]:
        raise ValueError("Manifest does not match the checkpoint.")
    assets = {"runner.py": base64.b64encode(source).decode(),
              "run_dual_gpu.py": base64.b64encode((here / "run_dual_gpu.py").read_text(encoding="utf-8").encode()).decode(),
              "manifest.json": base64.b64encode((here / "manifest.json").read_bytes()).decode()}
    backups = {p.name: base64.b64encode(p.read_bytes()).decode() for p in capture.glob("*.json")}
    bundle = base64.b64encode(zlib.compress(json.dumps({"assets": assets, "backups": backups}).encode(), 9)).decode()
    destination = here / "parallel_resume"
    destination.mkdir(exist_ok=True)
    bootstrap = '''import base64, json, zlib, subprocess, sys
from pathlib import Path
BUNDLE = "__BUNDLE__"
payload = json.loads(zlib.decompress(base64.b64decode(BUNDLE)))
assets = Path('/kaggle/working/parallel_assets')
backups = Path('/kaggle/working/preserved_checkpoint')
assets.mkdir(parents=True, exist_ok=True)
backups.mkdir(parents=True, exist_ok=True)
for name, encoded in payload['assets'].items():
    (assets / name).write_bytes(base64.b64decode(encoded))
for name, encoded in payload['backups'].items():
    (backups / name).write_bytes(base64.b64decode(encoded))
print('[Bootstrap] Restored the original checkpoint backup and byte-identical inference worker.', flush=True)
subprocess.run([sys.executable, '-u', str(assets / 'run_dual_gpu.py'),
    '--runner', str(assets / 'runner.py'), '--manifest', str(assets / 'manifest.json'),
    '--models', 'qwen', 'llama', 'phi4', '--tiers', 'T0', 'T1', 'T2',
    '--stagewise', '--per-domain', '0', '--resume-root', str(backups),
    '--output', '/kaggle/working/parallel_results'], check=True)
'''.replace("__BUNDLE__", bundle)
    (destination / "run_parallel_resume.py").write_bytes(bootstrap.encode())
    metadata = {"id": "subhadeepsing/ethics-tools-v2-parallel-resume",
        "title": "Ethics Tools V2 Parallel Resume", "code_file": "run_parallel_resume.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    receipt = {"preserved_qwen_T0": checkpoint["completed"], "worker_sha256": worker_hash,
               "manifest_sha256": checkpoint["settings"]["manifest_sha256"],
               "checkpoint_sha256": hashlib.sha256((capture / "qwen_T0_optional.json").read_bytes()).hexdigest(),
               "initial_tasks": ["GPU 0: qwen T0 resume", "GPU 1: llama T0"],
               "schedule": "Separate model/tier tasks on two independent GPU processes; Phi-4 is queued.",
               "evaluation": "Frozen scorer retained during inference; parser corrections are applied offline afterward."}
    (destination / "resume_package_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    build()
