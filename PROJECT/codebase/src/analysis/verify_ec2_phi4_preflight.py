"""Verify downloaded preflight evidence without editing experiment outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT/'codebase/results/ec2_phi4_preflight_20261020'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    cloud = OUT/'downloaded'
    expected = {'success': 'success', 'failure': 'exit-code', 'timeout': 'timeout'}
    probes = {}
    for case, result in expected.items():
        folder = cloud/'runs'/f'lifecycle-{case}'
        value = read(folder/'lifecycle.json')
        assert value['service_result'] == result
        assert read(folder/'probe_checkpoint.json')['saved']
        assert value['poweroff_requested'] is False
        probes[case] = result
    assert read(cloud/'runs/deadline-probe/lifecycle.json')['service_result'] == 'backup-deadline'
    smoke = cloud/'runs/phi4-smoke'
    completed = read(smoke/'completed.json')
    assert completed['passed'] and completed['conversations'] == 4 and completed['answer_turns'] == 12
    assert completed['checkpoint_resume_no_duplicate_calls'] and not completed['full_run_started']
    lifecycle = read(smoke/'lifecycle.json')
    assert lifecycle['service_result'] == 'success' and lifecycle['poweroff_requested']
    assert not (smoke/'shutdown_error.json').exists()
    for name, metadata in lifecycle['saved_files'].items():
        assert sha(smoke/name) == metadata['sha256'], name
    sys.path.insert(0, str(PROJECT/'codebase/src'))
    from analysis.exact_lookup_smoke import validate
    rows = validate(smoke, read(smoke/'manifest.json'), models=['phi4'])
    saved = read(smoke/'phi4_checkpoint.json')
    assert all('AWS EC2' in row[turn]['execution_metadata']['execution_host']
               for row in saved['dialogues'] for turn in ('p0', 'p1', 'p2'))
    replies = [row[turn]['response'] for row in saved['dialogues'] for turn in ('p0', 'p1', 'p2')]
    assert len(replies) == 12 and all(r['text'].startswith('Final Answer:') for r in replies)
    environment = read(smoke/'environment.json')
    assert 'T4' in environment['gpu'] and environment['gpu_count'] == 1
    assert environment['packages'] == {'torch': '2.11.0+cu128', 'transformers': '5.17.0',
                                       'accelerate': '1.15.0', 'bitsandbytes': '0.50.2'}
    for name, value in read(OUT/'source_guards.json').items():
        assert sha(Path(name)) == value, f'Protected file changed: {name}'
    lifecycle_test = read(OUT/'aws_lifecycle_verification.json')
    assert lifecycle_test['automatic_shutdown_observed_state'] == 'stopped'
    assert lifecycle_test['restart_test_state'] == 'running'
    assert lifecycle_test['job_inactive_after_restart']
    assert lifecycle_test['backup_timer_inactive_after_restart']
    assert lifecycle_test['final_state'] == 'stopped'
    result = dict(passed=True, scope='EC2 preflight only; no full EC2 run started',
                  lifecycle_probe_results=probes, independent_deadline_probe_passed=True,
                  actual_phi4_conversations=4, actual_answer_turns=12,
                  replies_with_final_answer_line=12,
                  token_capped_replies=sum(bool(r['hit_token_limit']) for r in replies),
                  tool_calls=sum(r['tool_calls'] for r in rows), rows=rows,
                  automatic_shutdown_verified=True, start_after_shutdown_verified=True,
                  persisted_results_verified=True, no_automatic_rerun_at_boot=True,
                  protected_files_unchanged=True, full_run_started=False,
                  final_instance_state='stopped', scoring_provisional=True)
    (OUT/'final_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
