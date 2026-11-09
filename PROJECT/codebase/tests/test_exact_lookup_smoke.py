import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analysis import exact_lookup_smoke as run
from analysis import web_search_pilot as pilot
from test_medqa_web_dual_gpu import FakeGPU


class DirectGPU(FakeGPU):
    def choose(self,messages):
        return {'action':'answer','label':'2'}


class ExactLookupTests(unittest.TestCase):
    def test_sample_and_shards(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'OUT',Path(tmp)):
            manifest,_=run.prepare()
            self.assertEqual(len(manifest['items']),20)
            halves=run.shards(manifest)
            self.assertEqual([len(s['items']) for s in halves],[10,10])

    def test_exact_entry_and_identity_failure(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'OUT',Path(tmp)):
            manifest,_=run.prepare()
            sheet=run.ExactAnswerSheet(manifest['items'],Path(tmp))
            for item in (manifest['items'][0],manifest['items'][-1]):
                evidence=sheet('irrelevant text must not affect retrieval',item)
                self.assertEqual(len(evidence['documents']),1)
                doc=evidence['documents'][0]
                self.assertEqual(doc['id'],item['id'])
                self.assertEqual(doc['answer'],item['ground_truth'])
                self.assertEqual(doc['solution'],item.get('solution',''))
                self.assertNotIn('false_answer_key',doc)
            with self.assertRaises(KeyError):
                sheet('',dict(id='missing'))
            with self.assertRaises(AssertionError):
                sheet('',dict(manifest['items'][0],question='Different question'))

    def test_real_protocol_delivery_history_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'OUT',Path(tmp)), \
             patch.object(pilot,'OUT',Path(tmp)/'gpu0'),patch.object(pilot,'PilotGPU',FakeGPU):
            full,_=run.prepare()
            selected=dict(full,items=[full['items'][0],full['items'][-1]])
            for name in run.MODELS:run.run_model(name,selected,full['items'])
            result=run.validate(pilot.OUT,selected)
            self.assertEqual(sum(r['tool_calls'] for r in result),12)
            self.assertTrue(all(r['tool_calls']==r['correct_entry_delivered'] for r in result))
            before=FakeGPU.loads
            run.run_model('qwen',selected,full['items'])
            self.assertEqual(before,FakeGPU.loads)
            logs=[json.loads(s) for s in (pilot.OUT/'qwen_model_calls.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(all(r['response']['max_new_tokens']==160 for r in logs))
            self.assertTrue(all('supports_ground_truth' not in str(r['messages']) for r in logs))
            self.assertFalse(any('Write one short web search query' in str(r['messages']) for r in logs))

    def test_optional_skip_does_not_inject_evidence(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'OUT',Path(tmp)), \
             patch.object(pilot,'OUT',Path(tmp)/'gpu0'),patch.object(pilot,'PilotGPU',DirectGPU):
            full,_=run.prepare()
            selected=dict(full,items=[full['items'][0]])
            saved=run.run_model('qwen',selected,full['items'])
            self.assertFalse(saved['dialogues'][0]['p1']['tool']['tool_invoked'])
            self.assertFalse((pilot.OUT/'lookup_events.jsonl').exists())
            self.assertNotIn('Verification tool result',str(saved['dialogues'][0]['history']))
            verified=run.validate(pilot.OUT,selected,models=['qwen'])
            self.assertEqual({r['model'] for r in verified},{'qwen'})
            self.assertEqual(sum(r['tool_calls'] for r in verified),0)


if __name__=='__main__':unittest.main()
