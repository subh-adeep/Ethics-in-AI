import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from analysis import medqa_web_dual_gpu as run
from analysis import web_search_pilot as pilot


class FakeGPU:
    loads = 0
    def __init__(self, name):
        type(self).loads += 1
        self.name = name
        self.metadata = {'gpu': 'mock'}
        self.context = {}

    def choose(self, messages):
        return {'action': 'verify', 'label': '1'}

    def generate(self, messages, limit=160):
        result = dict(text='infant diagnosis' if limit == 64 else 'Final Answer: A',
                      max_new_tokens=limit, generated_tokens=4, hit_token_limit=False)
        pilot.append_jsonl(pilot.OUT/f'{self.name}_model_calls.jsonl',
            dict(kind='generation', **self.context, messages=messages, response=result))
        return result

    def close(self):
        pass


class FakeSearch:
    def __init__(self):
        self.context = {}

    def __call__(self, query, item):
        return dict(status='ok', query=query,
            documents=[dict(title='test', url='https://example.org', snippet='retrieved evidence')],
            supports_ground_truth='HIDDEN_SENTINEL', cache_hit=False)


class DualGPUChecks(unittest.TestCase):
    def test_fixed_medqa_selection_and_shards(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'OUT', Path(tmp)):
            manifest, _ = run.prepare()
            halves = run.shards(manifest)
        self.assertEqual([len(s['items']) for s in halves], [73, 73])
        self.assertEqual(len(set(r['id'] for s in halves for r in s['items'])), 146)
        self.assertTrue(all(r['dataset'] == 'medqa' for r in manifest['items']))
        cohort = {r['id'] for r in (json.loads(s) for s in pilot.COHORT_SOURCE.read_text(encoding='utf-8').splitlines()) if r['domain'] == 'medqa'}
        self.assertEqual(cohort, {r['id'] for r in manifest['items']})

    def test_smoke_resume_and_actual_evidence_delivery(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'OUT', Path(tmp)), \
             patch.object(pilot, 'OUT', Path(tmp)/'gpu0'), patch.object(pilot, 'PilotGPU', FakeGPU), \
             patch.object(pilot, 'StrictSearch', FakeSearch):
            manifest, _ = run.prepare()
            manifest = dict(manifest, items=manifest['items'][:3], total=3)
            for name in run.MODELS:
                run.run_model(name, manifest, limit=2)
            smoke = run.validate_outputs(pilot.OUT, manifest, 2)
            self.assertTrue(all(r['tool_calls'] == 4 and r['delivered'] == 4 for r in smoke))
            for name in run.MODELS:
                run.run_model(name, manifest)
            full = run.validate_outputs(pilot.OUT, manifest, 3)
            self.assertTrue(all(r['questions'] == 3 and r['tool_calls'] == 6 for r in full))
            logs = [json.loads(s) for s in (pilot.OUT/'qwen_model_calls.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(sum(r['response']['max_new_tokens'] == 160 for r in logs), 9)
            self.assertNotIn('HIDDEN_SENTINEL', str([r['messages'] for r in logs]))
            loads = FakeGPU.loads
            run.run_model('qwen', manifest)
            self.assertEqual(FakeGPU.loads, loads, 'Finished model must not reload or regenerate')
            manifest['items'] = list(reversed(manifest['items']))
            with self.assertRaises(AssertionError):
                run.run_model('qwen', manifest)

    def test_missing_actual_generation_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'OUT', Path(tmp)), \
             patch.object(pilot, 'OUT', Path(tmp)/'gpu0'), patch.object(pilot, 'PilotGPU', FakeGPU), \
             patch.object(pilot, 'StrictSearch', FakeSearch):
            manifest, _ = run.prepare()
            manifest = dict(manifest, items=manifest['items'][:1], total=1)
            for name in run.MODELS:
                run.run_model(name, manifest)
            path = pilot.OUT/'qwen_model_calls.jsonl'
            records = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
            for r in records:
                r['messages'] = [m for m in r['messages'] if 'retrieved evidence' not in m['content']]
            path.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
            with self.assertRaisesRegex(AssertionError, 'actual generation'):
                run.validate_outputs(pilot.OUT, manifest, 1)


if __name__ == '__main__':
    unittest.main()
