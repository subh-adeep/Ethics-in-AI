from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analysis import exact_lookup_full as full
from analysis import exact_lookup_smoke as smoke
from analysis import web_search_pilot as pilot


class FullCohortTests(unittest.TestCase):
    def test_each_model_gets_the_full_fixed_cohort(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(full,'OUT',Path(tmp)):
            manifest,protocol=full.prepare()
            task=full.task_manifest(manifest)
            self.assertEqual(len(task['items']),361)
            self.assertEqual(dict(Counter(i['dataset'] for i in task['items'])),{'medqa':146,'math':215})
            self.assertEqual([i['dataset'] for i in task['items'][:2]],['medqa','math'])
            ids=[i['id'] for i in task['items']]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(set(ids),{r['id'] for r in manifest['items']})
            self.assertEqual(protocol['conversations'],1083)
            self.assertEqual(protocol['final_answer_turns'],3249)
            self.assertEqual(protocol['initial_gpu_models'],{'0':'qwen','1':'llama'})
            self.assertEqual(protocol['queued_model'],'phi4')
            self.assertEqual(full.prepare(),(manifest,protocol))

    def test_smoke_settings_and_canonical_prompts_preserved(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(full,'OUT',Path(tmp)):
            manifest,protocol=full.prepare()
        old=json.loads((smoke.OUT/'protocol.json').read_text(encoding='utf-8'))
        for field in ['setting','answer_max_new_tokens','original_prompts','original_system_prompt',
                      'original_single_token_action_selection','tool_name','lookup_method','entries_per_call',
                      'medqa_evidence','math_evidence','model_revisions']:
            self.assertEqual(protocol[field],old[field])
        source=json.loads(pilot.MANIFEST_SOURCE.read_text(encoding='utf-8'))
        lookup={r['id']:r for r in source['items']}
        self.assertTrue(all(r==lookup[r['id']] for r in manifest['items']))
        self.assertTrue(all(r['level'] in (1,2,'Level 1','Level 2') for r in manifest['items'] if r['dataset']=='math'))


if __name__=='__main__':unittest.main()
