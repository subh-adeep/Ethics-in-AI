"""Re-evaluate saved raw replies; never generate or fabricate model output."""
import argparse
import json
from pathlib import Path
from runner import classify, metrics, EVALUATOR_VERSION


def rejudge(source, manifest_path, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    items = {i["id"]: i for i in manifest["items"]}
    for path in source.glob("*_optional.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        run["evaluation_revision"] = EVALUATOR_VERSION
        run["original_metrics"] = run["metrics"]
        for row in run["dialogues"]:
            item = items[row["id"]]
            row["p0"].update(classify(row["p0"]["text"], item))
            row["p0_correct"] = row["p0"]["outcome"] == "resistant"
            for pressure in ("p1", "p2"):
                turn = row[pressure]
                turn.update(classify(turn["text"], item))
                evidence = turn["tool"].get("evidence")
                if evidence and evidence.get("supports_ground_truth") is True and turn["tool"]["tool_success"]:
                    turn["evidence_override"] = turn["outcome"] == "sycophantic"
        run["metrics"] = metrics(run["dialogues"])
        (destination / path.name).write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        print(path.name, "tool requests", run["metrics"]["tool_requested_n"],
              "unparsed", sum(run["metrics"][p + "_outcomes"]["unparsed"] for p in ("p1", "p2")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("manifest")
    parser.add_argument("destination")
    args = parser.parse_args()
    rejudge(args.source, args.manifest, args.destination)
