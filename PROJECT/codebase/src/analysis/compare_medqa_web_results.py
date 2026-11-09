"""Read-only, consistently scored T0/T1/T2 comparison on the fixed 146 MedQA IDs."""
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.medqa_web_dual_gpu import OUT, MODELS
from analysis import web_search_pilot as pilot
from analysis.rescore_saved_replies import outcome
from benchmark.answer_parser import VERSION, parse_reply

NAMES = {'qwen': 'Qwen', 'llama': 'LLaMA', 'phi4': 'Phi-4'}
LABELS = ('resistant', 'sycophantic', 'unforced_error', 'unparsed')


def csv_file(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    folder = OUT/'comparison'
    folder.mkdir(exist_ok=True)
    cloud = OUT/'kaggle_download_final_v2/medqa_project/PROJECT/codebase/results/medqa_web_t2_20261019'
    manifest_path = OUT/'manifest.json'
    items = {i['id']: i for i in json.loads(manifest_path.read_text(encoding='utf-8'))['items']}
    assert len(items) == 146 and all(i['dataset'] == 'medqa' for i in items.values())
    baseline_path = pilot.COHORT_SOURCE
    baseline = [json.loads(s) for s in baseline_path.read_text(encoding='utf-8').splitlines()]
    records = []
    tool_rows = []
    sources = [manifest_path, baseline_path, Path(sys.modules['benchmark.answer_parser'].__file__)]
    for row in baseline:
        if row['domain'] != 'medqa':
            continue
        assert row['id'] in items and row['tier'] in ('T0', 'T1')
        item = items[row['id']]
        parsed = parse_reply(row['text'], 'medqa', options=item['options'], question=item['question'],
                             hit_token_limit=row['hit_token_limit'])
        assert parsed.answer == row['parsed_answer']
        label = outcome(parsed.answer, item)
        assert label == row['outcome']
        records.append(dict(model=row['model'], setting=row['tier'], id=row['id'], turn=row['turn'],
            parsed_answer=parsed.answer, outcome=label, conflict=parsed.conflict,
            hit_token_limit=row['hit_token_limit'], parser_version=VERSION))
    assert len(records) == 146*3*2*3
    for model in MODELS:
        source = cloud/f'{model}_checkpoint.json'
        sources.append(source)
        saved = json.loads(source.read_text(encoding='utf-8'))
        assert len(saved['dialogues']) == 146 and {r['id'] for r in saved['dialogues']} == set(items)
        for row in saved['dialogues']:
            item = items[row['id']]
            for turn in ('p0', 'p1', 'p2'):
                result = row[turn]
                response = result['response']
                parsed = parse_reply(response['text'], 'medqa', options=item['options'], question=item['question'],
                                     hit_token_limit=response['hit_token_limit'])
                assert asdict(parsed) == result['evaluation']['provisional_parser_v4']
                label = outcome(parsed.answer, item)
                records.append(dict(model=model, setting='T2', id=row['id'], turn=turn,
                    parsed_answer=parsed.answer, outcome=label, conflict=parsed.conflict,
                    hit_token_limit=response['hit_token_limit'], parser_version=VERSION))
                if turn != 'p0':
                    tool = result['tool']
                    called = bool(tool and tool['tool_invoked'])
                    tool_rows.append(dict(model=model, id=row['id'], turn=turn, invoked=called,
                        success=bool(called and tool['tool_success']),
                        status=tool['evidence']['status'] if called else 'not_called',
                        outcome=label, evidence_delivered=bool(called and result['delivery_check']['exact_observation_in_generation_input'])))
    lookup = {(r['model'], r['setting'], r['id'], r['turn']): r for r in records}
    assert len(lookup) == len(records) == 146*3*3*3
    summaries = []
    paired = []
    for model in MODELS:
        for setting in ('T0', 'T1', 'T2'):
            for turn in ('p0', 'p1', 'p2'):
                selected = [r for r in records if (r['model'], r['setting'], r['turn']) == (model, setting, turn)]
                assert len(selected) == 146 and {r['id'] for r in selected} == set(items)
                counts = Counter(r['outcome'] for r in selected)
                if turn == 'p0':
                    assert counts['resistant'] == 146, 'Fresh P0 comparison needs qualification'
                summary = dict(model=model, setting=setting, turn=turn, n=146)
                for label in LABELS:
                    summary[label+'_n'] = counts[label]
                    summary[label+'_pct'] = round(100*counts[label]/146, 4)
                summary['conflict_n'] = sum(r['conflict'] for r in selected)
                summaries.append(summary)
        for turn in ('p1', 'p2'):
            pairs = [(lookup[model, 'T0', ident, turn]['outcome'] == 'resistant',
                      lookup[model, 'T2', ident, turn]['outcome'] == 'resistant') for ident in items]
            paired.append(dict(model=model, turn=turn,
                correct_in_both=sum(a and b for a,b in pairs),
                correct_only_without_tool=sum(a and not b for a,b in pairs),
                correct_only_with_web=sum(not a and b for a,b in pairs),
                incorrect_or_unparsed_in_both=sum(not a and not b for a,b in pairs)))
    tools = []
    for model in MODELS:
        entry = dict(model=model)
        for turn in ('p1', 'p2'):
            selected = [r for r in tool_rows if r['model'] == model and r['turn'] == turn]
            assert len(selected) == 146
            called = sum(r['invoked'] for r in selected)
            entry.update({turn+'_calls': called, turn+'_call_rate_pct': round(100*called/146, 4),
                          turn+'_nonempty': sum(r['success'] for r in selected),
                          turn+'_failed_or_empty': sum(r['invoked'] and not r['success'] for r in selected)})
        entry['total_calls'] = entry['p1_calls']+entry['p2_calls']
        entry['nonempty'] = entry['p1_nonempty']+entry['p2_nonempty']
        entry['failed_or_empty'] = entry['total_calls']-entry['nonempty']
        tools.append(entry)
    def get(model, setting, turn):
        return next(r for r in summaries if (r['model'],r['setting'],r['turn']) == (model,setting,turn))
    lines = ['# MedQA: no tool, BM25 and web search', '',
        'All three settings use the same fixed 146 question IDs. Every model answered all 146 P0 questions correctly in the new web-search run as well as the saved no-tool run. The unchanged answer-parser-4.0 was applied consistently to every saved reply; no Gemma extraction or longer-budget rerun is substituted. Unparsed replies remain in the denominator.', '',
        'These updated no-tool numbers differ from the older PDF tables, which used the frozen original scorer. T0 means no tool; T1 means optional BM25 answer retrieval; T2 means optional web search. P1 is the ordinary-user wrong suggestion; P2 repeats it with claimed expertise.', '',
        '## Correct-answer rates', '',
        '| Model | Turn | No tool (%) | BM25 (%) | Web search (%) | Web minus no tool (percentage points) |',
        '|---|---|---:|---:|---:|---:|']
    for model in MODELS:
        for turn in ('p1', 'p2'):
            r0,r1,r2 = [get(model,s,turn)['resistant_pct'] for s in ('T0','T1','T2')]
            lines.append(f'| {NAMES[model]} | {turn.upper()} | {r0:.1f} | {r1:.1f} | {r2:.1f} | {r2-r0:+.1f} |')
    lines += ['', '## Web-search calls', '',
        '| Model | P1 calls / 146 | P2 calls / 146 | Total / 292 | Nonempty | Failed or empty |',
        '|---|---:|---:|---:|---:|---:|']
    for r in tools:
        lines.append(f"| {NAMES[r['model']]} | {r['p1_calls']} | {r['p2_calls']} | {r['total_calls']} | {r['nonempty']} | {r['failed_or_empty']} |")
    lines += ['', '## Interpretation', '',
        'Web search does not improve resistance overall in this run. Qwen and LLaMA have lower correct-answer rates at both turns. Phi-4 improves slightly at P1 but declines at P2. This is a comparison of the complete settings; it does not establish that retrieved text itself caused an error. Routing, query generation, added messages, retrieved snippets and runtime details also differ.', '',
        'Search returned nonempty text in 629 of 706 calls; 77 calls failed or were empty. Nonempty snippets have not yet been checked for relevance or answer support across the full run. The saved token-cap flags and unresolved/conflicting replies still need review before treating these as final scores.', '',
        'The report and original checkpoints were not changed. `outcome_summary.csv` contains all four outcome counts/rates, `tool_calls.csv` contains counts and per-turn rates, `paired_changes.csv` tracks matched gains/losses, and `scored_replies.csv` contains the consistently parsed answers for all settings.']
    (folder/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    csv_file(folder/'outcome_summary.csv', summaries)
    csv_file(folder/'tool_calls.csv', tools)
    csv_file(folder/'paired_changes.csv', paired)
    csv_file(folder/'scored_replies.csv', records)
    pilot.atomic_json(folder/'verification.json', dict(at=pilot.now(), parser=VERSION, n=146,
        evaluated_replies=len(records), every_setting_p0_correct=146,
        original_data_only=True, no_model_inference=True, outcomes_sum_to_146=True,
        sources={str(p):pilot.sha(p) for p in sources}, tool_call_summaries=tools))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
