import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from run_dual_gpu import restore_checkpoint


class ResumeTests(unittest.TestCase):
    def fixture(self, folder):
        folder = Path(folder)
        worker = folder / "worker.py"
        worker.write_bytes(b"# frozen test worker\n")
        manifest = {"items": [{"id": "item1", "dataset": "medqa"}]}
        rows = [{"id": "item1", "p0": {"text": "Final Answer: A"}, "p1": {"text": "Final Answer: A"}}]
        settings = {"source_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            "item_ids": ["item1"], "model_id": "Qwen/Qwen2.5-3B-Instruct", "policy": "optional",
            "precision": "4bit", "tier": "T0", "tiers": ["T0", "T1", "T2"]}
        checkpoint = folder / "source.json"
        checkpoint.write_text(json.dumps({"settings": settings, "dialogues": rows, "completed": 1}), encoding="utf-8")
        return worker, manifest, checkpoint, rows

    def test_preserves_original_and_raw_dialogues(self):
        with tempfile.TemporaryDirectory() as folder:
            worker, manifest, checkpoint, rows = self.fixture(folder)
            original = checkpoint.read_bytes()
            target = Path(folder) / "output" / "checkpoint.json"
            self.assertEqual(restore_checkpoint(checkpoint, target, worker, manifest, "qwen", ["T0"], 0), 1)
            migrated = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(migrated["dialogues"], rows)
            self.assertEqual(migrated["settings"]["tiers"], ["T0"])
            self.assertEqual(checkpoint.read_bytes(), original)
            self.assertTrue((target.parent / "resume_receipt.json").exists())

    def test_rejects_changed_worker_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            worker, manifest, checkpoint, _ = self.fixture(folder)
            worker.write_bytes(b"# different worker\n")
            target = Path(folder) / "out.json"
            with self.assertRaises(ValueError):
                restore_checkpoint(checkpoint, target, worker, manifest, "qwen", ["T0"], 0)
            self.assertFalse(target.exists())

    def test_rejects_invalid_cohort_and_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            worker, manifest, checkpoint, _ = self.fixture(folder)
            target = Path(folder) / "out.json"
            with self.assertRaises(ValueError):
                restore_checkpoint(checkpoint, target, worker, {"items": []}, "qwen", ["T0"], 0)
            target.write_text("keep this", encoding="utf-8")
            with self.assertRaises(ValueError):
                restore_checkpoint(checkpoint, target, worker, manifest, "qwen", ["T0"], 0)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep this")


if __name__ == "__main__":
    unittest.main()
