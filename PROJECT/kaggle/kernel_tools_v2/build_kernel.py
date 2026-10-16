"""Freeze T0 cohort + prompts and check Week-2 all-correct provenance."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib
from collections import Counter


def build(full=False, models=None, output_directory=None, tiers=None):
    here = Path(__file__).resolve().parent
    project = here.parents[1]
    data = json.loads((project / "data/final/final_filtered_evaluation_set_500.json").read_text(encoding="utf-8"))
    logs = [json.loads((project / f"codebase/results/week3_logs/{m}_t0_results.json").read_text(encoding="utf-8"))
            for m in ("qwen", "llama", "phi4")]
    filters = [json.loads((project / f"codebase/results/week2_logs/{m}_eval.json").read_text(encoding="utf-8"))
               for m in ("qwen", "llama", "phi4")]
    mappings = [{d["id"]: d for d in log["dialogues"]} for log in logs]
    cohort = set(mappings[0])
    if any(set(m) != cohort for m in mappings):
        raise ValueError("T0 runs do not share the same cohort.")
    items = []
    historical_mismatches = []
    counts = Counter(i["id"] for i in data)
    excluded = sorted(i for i in cohort if counts[i] != 1)
    for original in data:
        if original["id"] not in cohort or original["id"] in excluded:
            continue
        item = dict(original)
        key = "medqa_eval" if item["dataset"] == "medqa" else "math_eval"
        if not all(f[key][item["id"]]["is_correct"] is True for f in filters):
            raise ValueError(f"Item was not originally correct in all models: {item['id']}")
        if not all(str(f[key][item["id"]]["ground_truth"]) == str(item["ground_truth"]) for f in filters):
            raise ValueError(f"Filter record label does not match the actual question: {item['id']}")
        false = item.get("false_answer_key") if item["dataset"] == "medqa" else item.get("false_answer")
        if str(item["ground_truth"]) == str(false):
            raise ValueError(f"Invalid labels: {item['id']}")
        for model, mapping in zip(("qwen", "llama", "phi4"), mappings):
            record = mapping[item["id"]]
            mismatch = [prompt for prompt, turn in (("p0_prompt", "turn1_p0"), ("p1_prompt", "turn2_p1"), ("p2_prompt", "turn3_p2"))
                        if record[turn]["prompt"] != item[prompt]]
            if mismatch:
                historical_mismatches.append({"id": item["id"], "model": model, "different_prompts": mismatch})
        items.append(item)
    if len(items) + len(excluded) != len(cohort):
        raise ValueError("Some T0 items are missing from the current dataset.")
    manifest = {"provenance": {"selection": "Week-2 all-three-models-correct intersection, restricted to shared completed T0 IDs",
        "original_rows_n": len(data), "original_unique_ids_n": len(cohort), "verified_cohort_n": len(items),
        "excluded_duplicate_ids": excluded,
        "domains": {d: sum(i["dataset"] == d for i in items) for d in ("medqa", "math")},
        "prompts": "Canonical current main-case dataset prompts, identical across new T0/T1/T2. Historical T0 is not a matched control; rerun T0.",
        "t1_condition": "privileged retrieval of ground-truth answer sheet, as proposed",
        "later_t0_p0_all_correct_n": len(set.intersection(*[{d['id'] for d in l['dialogues'] if d['turn1_p0']['is_correct']} for l in logs]))},
        "historical_prompt_audit": historical_mismatches, "items": items}
    destination = Path(output_directory).resolve() if output_directory else here
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    blob = base64.b64encode(zlib.compress(json.dumps(manifest, ensure_ascii=False).encode(), 9)).decode()
    source = (here / "runner.py").read_text(encoding="utf-8").replace("__MANIFEST__", blob)
    if full:
        source = source.replace('type=int, default=1, help=', 'type=int, default=0, help=')
    if models:
        source = source.replace('default=list(MODELS))', f'default={models!r})')
    if tiers:
        source = source.replace('default=["T0", "T1", "T2"])', f'default={tiers!r})')
    (destination / "run_kaggle_tools_v2.py").write_text(source, encoding="utf-8")
    suffix = ("full" if full else "pilot") + ("-" + "-".join(models) if models else "")
    metadata = {"id": "subhadeepsing/ethics-tools-v2-" + suffix,
        "title": "Ethics Tools V2 " + ("Full" if full else "Pilot") + (" " + " ".join(models) if models else ""), "code_file": "run_kaggle_tools_v2.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(data), "unique_ids": len(cohort), "excluded_ambiguous_ids": len(excluded),
                      "verified_questions": len(items), "domains": manifest["provenance"]["domains"]}))
    print("Built", metadata["id"], "code SHA256", hashlib.sha256(source.encode()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--models", nargs="+", choices=["qwen", "llama", "phi4"])
    parser.add_argument("--output-directory")
    parser.add_argument("--tiers", nargs="+", choices=["T0", "T1", "T2"])
    args = parser.parse_args()
    build(args.full, args.models, args.output_directory, args.tiers)
