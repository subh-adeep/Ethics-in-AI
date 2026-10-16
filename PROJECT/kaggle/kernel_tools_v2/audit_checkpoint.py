"""Inspect existing replies and parsing changes without running any model.

Example:
python audit_checkpoint.py qwen_T0_optional.json --manifest manifest.json --output audit.json
"""
import argparse
import json
from pathlib import Path
from runner import classify, EVALUATOR_VERSION


def audit(checkpoint_path, manifest_path, output_path):
    checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    items = {i["id"]: i for i in manifest["items"]}
    cases = []
    counts = {"responses": 0, "stored_unparsed": 0, "revised_unparsed": 0,
              "classification_changes": 0, "hit_token_limit": 0, "p0_not_correct": 0}
    for row in checkpoint["dialogues"]:
        item = items[row["id"]]
        counts["p0_not_correct"] += not row["p0_correct"]
        for pressure in ("p0", "p1", "p2"):
            turn = row[pressure]
            counts["responses"] += 1
            revised = classify(turn["text"], item)
            stored_unparsed = turn.get("extracted") is None
            changed = revised["extracted"] != turn.get("extracted") or revised["outcome"] != turn.get("outcome")
            counts["stored_unparsed"] += stored_unparsed
            counts["revised_unparsed"] += revised["outcome"] == "unparsed"
            counts["classification_changes"] += changed
            counts["hit_token_limit"] += bool(turn.get("hit_token_limit"))
            if stored_unparsed or changed or (pressure == "p0" and not row["p0_correct"]):
                cases.append({"id": row["id"], "pressure": pressure, "question": item["question"],
                    "options": item.get("options"), "ground_truth": item["ground_truth"],
                    "false_answer": item.get("false_answer_key") if item["dataset"] == "medqa" else item.get("false_answer"),
                    "response_raw": turn["text"], "generated_tokens": turn.get("generated_tokens"),
                    "hit_token_limit": turn.get("hit_token_limit"), "stored_extracted": turn.get("extracted"),
                    "stored_outcome": turn.get("outcome"), "revised_evaluation": revised,
                    "review_status": "manual_review_required"})
    report = {"checkpoint": str(Path(checkpoint_path).resolve()), "evaluator_revision": EVALUATOR_VERSION,
              "notice": "Offline audit only. Original checkpoint is unchanged; no model replies were generated.",
              "counts": counts, "cases": cases}
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print("Review cases:", str(Path(output_path).resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("manifest.json")))
    parser.add_argument("--output", default="checkpoint_audit.json")
    args = parser.parse_args()
    audit(args.checkpoint, args.manifest, args.output)
