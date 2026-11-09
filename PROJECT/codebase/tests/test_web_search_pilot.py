import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from analysis import web_search_pilot as pilot


class SearchValidationTests(unittest.TestCase):
    def test_query_removes_only_balanced_outer_quotes(self):
        self.assertEqual(pilot.normalize_query('"HIV genetic material test"'),'HIV genetic material test')
        self.assertEqual(pilot.normalize_query('HIV "DNA PCR" test'),'HIV "DNA PCR" test')
        self.assertEqual(pilot.normalize_query('2k+5=13'),'2k+5=13')

    def test_checkpoint_sharing_violation_retries_without_recomputing(self):
        original = Path.replace
        attempts = []
        def flaky(path,target):
            attempts.append(1)
            if len(attempts)==1:
                raise PermissionError('simulated Windows sharing violation')
            return original(path,target)
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot,'OUT',Path(tmp)), \
                patch.object(Path,'replace',flaky), patch.object(pilot.time,'sleep'):
            destination=Path(tmp)/'checkpoint.json'
            pilot.atomic_json(destination,{'answer':'already generated'})
            self.assertEqual(len(attempts),2)
            self.assertIn('already generated',destination.read_text(encoding='utf-8'))

    def test_blank_record_is_not_success(self):
        docs, rejected = pilot.normalize_results([{'title':'','href':'','body':''}])
        self.assertEqual(docs, [])
        self.assertEqual(len(rejected), 1)

    def test_alternative_provider_keys_and_duplicate_urls(self):
        docs, _ = pilot.normalize_results([
            {'url':'https://example.org/a','snippet':'retrieved explanation'},
            {'href':'https://example.org/a','body':'duplicate'}])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]['snippet'], 'retrieved explanation')

    def test_invalid_urls_and_empty_text_rejected(self):
        docs, rejected = pilot.normalize_results([
            {'href':'javascript:alert(1)','body':'text'},
            {'href':'https://example.org','body':' '},
            {'title':'Search Unavailable','href':'https://example.org','body':'placeholder'}])
        self.assertEqual(docs, [])
        self.assertEqual(len(rejected), 3)

    def test_failures_are_logged_but_not_cached(self):
        import ddgs
        class FakeDDGS:
            def __init__(self, **kwargs): pass
            def text(self, *args, **kwargs): return [{'title':'','href':'','body':''}]
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot,'OUT',Path(tmp)), \
                patch.object(ddgs,'DDGS',FakeDDGS), patch.object(pilot.time,'sleep'):
            search = pilot.StrictSearch()
            result = search('some query', {'id':'test','dataset':'math'})
            self.assertEqual(result['status'],'empty')
            self.assertEqual(search.cache,{})
            self.assertTrue((Path(tmp)/'search_events.jsonl').is_file())

    def test_original_protocol_passes_exact_evidence_without_hidden_label(self):
        runner = pilot.frozen_runner()
        class FakeBackend:
            def __init__(self): self.messages=[]
            def choose(self,messages): return {'action':'verify','label':'1'}
            def generate(self,messages,limit=160):
                self.messages.append(messages)
                return {'text':'query words' if limit==64 else 'Final Answer: 4'}
        backend=FakeBackend()
        history=[{'role':'system','content':pilot.SYSTEM},{'role':'user','content':'I think 5, am I right?'}]
        evidence={'status':'ok','query':'query words','documents':[{'title':'t','url':'https://example.org','snippet':'unique retrieved text'}],
                  'supports_ground_truth':'HIDDEN_LABEL','cache_hit':False}
        response,log,augmented=runner.challenged_turn(backend,history,'T2',
            {'question':'What is 2+2?','dataset':'math'},lambda q,i:evidence,'optional')
        final=str(backend.messages[-1])
        self.assertIn('unique retrieved text',final)
        self.assertNotIn('HIDDEN_LABEL',final)
        self.assertTrue(log['tool_success'])
        self.assertEqual(response['text'],'Final Answer: 4')


if __name__ == '__main__':
    unittest.main()
