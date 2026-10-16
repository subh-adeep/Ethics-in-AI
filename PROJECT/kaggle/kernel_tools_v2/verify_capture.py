"""Validate downloaded checkpoints without modifying or rescoring model replies."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from runner import MODELS, metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'parallel_assets/manifest.json').read_text(encoding='utf-8'))
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    source_hash = sha(root / 'parallel_assets/runner.py')
    ids = [item['id'] for item in manifest['items']]
    assert len(ids) == len(set(ids)) == 421, 'Invalid cohort'
    entries = []
    for model in MODELS:
        for tier in ('T0', 'T1', 'T2'):
            path = root / 'parallel_results' / f'{model}_{tier}' / f'{model}_{tier}_optional.json'
            checkpoint = json.loads(path.read_text(encoding='utf-8'))
            rows, settings = checkpoint['dialogues'], checkpoint['settings']
            done = [row['id'] for row in rows]
            assert len(done) == len(set(done)) == checkpoint['completed'], path
            assert done == ids[:len(done)], f'Unexpected question order: {path}'
            assert checkpoint['total'] == len(ids), path
            assert settings['source_sha256'] == source_hash, path
            assert settings['manifest_sha256'] == manifest_hash, path
            assert settings['item_ids'] == ids, path
            assert settings['tier'] == tier and settings['tiers'] == [tier], path
            assert settings['model_id'] == MODELS[model], path
            assert settings['policy'] == 'optional' and settings['precision'] == '4bit', path
            for row in rows:
                assert row['tier'] == tier, path
                for turn in ('p0', 'p1', 'p2'):
                    assert isinstance(row[turn]['text'], str), path
                    assert 'generated_tokens' in row[turn] and 'hit_token_limit' in row[turn], path
                for turn in ('p1', 'p2'):
                    assert 'tool' in row[turn], path
            assert metrics(rows) == checkpoint['metrics'], f'Metric totals mismatch: {path}'
            turns = [row[p] for row in rows for p in ('p1', 'p2')]
            called = [t for t in turns if t['tool']['tool_invoked']]
            successful = [t for t in called if t['tool']['tool_success']]
            failures = Counter(str(t['tool'].get('evidence', {}).get('error', 'unsuccessful/empty retrieval'))
                               for t in called if not t['tool']['tool_success'])
            entries.append({'model': model, 'tier': tier, 'completed': len(rows), 'total': len(ids),
                            'remaining': len(ids) - len(rows), 'path': str(path), 'sha256': sha(path),
                            'metrics': checkpoint['metrics'],
                            'after_successful_tool_outcomes': dict(Counter(t['outcome'] for t in successful)),
                            'tool_failure_messages': dict(failures),
                            'hit_token_limit': sum(bool(row[p]['hit_token_limit']) for row in rows for p in ('p0', 'p1', 'p2'))})
    files = [p for p in root.rglob('*') if p.is_file() and p.name not in
             ('verification_report.json', 'PROGRESS.md', 'backup.zip', 'backup.zip.sha256')]
    for path in files:
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'))
    completed = sum(e['completed'] for e in entries)
    report = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'verified',
              'notice': 'Original downloaded responses and stored scoring remain unchanged. Scoring is provisional.',
              'source_sha256': source_hash, 'manifest_sha256': manifest_hash,
              'completed_dialogues': completed, 'planned_dialogues': len(ids) * 9,
              'remaining_dialogues': len(ids) * 9 - completed,
              'entries': entries,
              'downloaded_files': {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files}}
    (root / 'verification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Preserved Kaggle experiment progress', '',
             'Remote run was manually stopped. All nine checkpoints parsed and passed cohort, source, settings, count and metric consistency checks.', '',
             f'Completed: **{completed}/{len(ids) * 9} dialogues**. Remaining: **{len(ids) * 9 - completed}**.', '',
             '| Model | T0 | T1 | T2 |', '|---|---:|---:|---:|']
    for model in MODELS:
        counts = [next(e['completed'] for e in entries if e['model'] == model and e['tier'] == tier) for tier in ('T0', 'T1', 'T2')]
        lines.append(f'| {model} | {counts[0]}/421 | {counts[1]}/421 | {counts[2]}/421 |')
    lines += ['', '## T1 outcomes after successful RAG calls', '',
              'Stored scorer labels; no offline rescoring has been applied. These are descriptive counts, not a causal comparison with T0.', '',
              '| Model | Calls | Resistant | Sycophantic | Other wrong | Unparsed |', '|---|---:|---:|---:|---:|---:|']
    for e in entries:
        if e['tier'] == 'T1':
            outcomes = e['after_successful_tool_outcomes']
            cells = [outcomes.get(k, 0) for k in ('resistant', 'sycophantic', 'unforced_error', 'unparsed')]
            lines.append(f"| {e['model']} | {e['metrics']['tool_success_n']} | " + ' | '.join(map(str, cells)) + ' |')
    lines += ['', 'The original raw responses, tool records, web caches, health checks, manifests and inference source are preserved in this directory.',
              'Do not rerun the old upload: a future resume must use these latest checkpoints.',
              'workers.json records launcher state before interruption; its active process IDs do not mean the cancelled cloud job is still running.', '']
    (root / 'PROGRESS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': 'verified', 'completed': completed, 'remaining': len(ids)*9-completed,
                      'counts': {e['model']+'_'+e['tier']: e['completed'] for e in entries}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('capture_directory')
    verify(parser.parse_args().capture_directory)
