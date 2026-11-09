import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from analysis.ec2_job_lifecycle import durable_json, finalize


class LifecycleTests(unittest.TestCase):
    def test_final_status_saved_before_shutdown_for_success_failure_and_timeout(self):
        for status in ('success', 'exit-code', 'timeout'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                durable_json(folder/'checkpoint.json', {'saved': True})
                def shutdown():
                    saved = json.loads((folder/'lifecycle.json').read_text())
                    self.assertEqual(saved['service_result'], status)
                    self.assertTrue(saved['poweroff_requested'])
                    self.assertIn('checkpoint.json', saved['saved_files'])
                finalize(folder, True, {'SERVICE_RESULT': status}, shutdown)

    def test_shutdown_failure_retains_results_and_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            def fail():
                raise RuntimeError('test rejection')
            with self.assertRaises(RuntimeError):
                finalize(folder, True, {'SERVICE_RESULT': 'success'}, fail)
            self.assertTrue((folder/'lifecycle.json').exists())
            self.assertIn('test rejection', (folder/'shutdown_error.json').read_text())


if __name__ == '__main__':
    unittest.main()
