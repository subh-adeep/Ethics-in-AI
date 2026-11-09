"""Portable private Kaggle controller. Payload is inserted by the local launcher."""
import base64
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import zlib

PAYLOAD_B64 = '__MEDQA_PAYLOAD__'


def main():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
        'transformers==5.17.0', 'accelerate==1.15.0', 'bitsandbytes==0.50.2', 'ddgs==9.16.0'], check=True)
    payload = json.loads(zlib.decompress(base64.b64decode(PAYLOAD_B64)))
    project = Path('/kaggle/working/medqa_project/PROJECT')
    for relative, text in payload['files'].items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    for package in ('analysis', 'benchmark'):
        (project/'codebase/src'/package/'__init__.py').write_text('', encoding='utf-8')
    out = project/'codebase/results/medqa_web_t2_20261019'
    out.mkdir(parents=True, exist_ok=True)
    for filename, value in [('manifest.json', payload['manifest']), ('protocol.json', payload['protocol']),
                            ('uploaded_source_hashes.json', payload['source_hashes'])]:
        (out/filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    import torch
    count = torch.cuda.device_count()
    assert count == 2, f'Requested two GPUs, received {count}; full run will not start'
    devices = [dict(index=i, name=torch.cuda.get_device_name(i),
                    memory_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(count)]
    environment = dict(python=sys.version, devices=devices, cuda=torch.version.cuda,
        packages={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'ddgs', 'primp']})
    (out/'cloud_environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    print('DUAL_GPU_CONFIRMED', json.dumps(devices), flush=True)
    from huggingface_hub import snapshot_download
    cache = Path('/tmp/ethics_medqa_t2_model_cache')
    for name, spec in payload['models'].items():
        print('DOWNLOAD_MODEL', name, spec['revision'], flush=True)
        snapshot_download(repo_id=spec['repo'], revision=spec['revision'], cache_dir=cache,
            allow_patterns=['*.json', '*.safetensors', '*.model', 'tokenizer*', '*.py', '*.txt'])
        refs = cache/('models--'+spec['repo'].replace('/', '--'))/'refs'
        refs.mkdir(parents=True, exist_ok=True)
        (refs/'main').write_text(spec['revision'], encoding='utf-8')
    sys.path.insert(0, str(project/'codebase/src'))
    from analysis import medqa_web_dual_gpu as run
    from analysis import web_search_pilot as pilot
    pilot.atomic_json(out/'controller_started.json', dict(at=pilot.now(), pid=os.getpid(), devices=devices))
    children = []
    readers = []

    def relay(process, gpu):
        with (out/f'gpu{gpu}_worker.log').open('a', encoding='utf-8') as stream:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(f'[GPU{gpu}] {line.rstrip()}', flush=True)

    try:
        for gpu in range(2):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                       TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            process = subprocess.Popen([sys.executable, '-m', 'analysis.medqa_web_dual_gpu', '--worker', str(gpu)],
                env=dict(env, PYTHONPATH=str(project/'codebase/src')), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
            children.append(process)
            thread = threading.Thread(target=relay, args=(process, gpu), daemon=True)
            thread.start()
            readers.append(thread)
        deadline = time.monotonic()+1800
        released = False
        while True:
            codes = [p.poll() for p in children]
            if any(c is not None and c != 0 for c in codes):
                raise RuntimeError(f'GPU worker failed: {codes}')
            if not released:
                ready_paths = [out/f'gpu{i}/smoke_ready.json' for i in range(2)]
                if all(p.exists() for p in ready_paths):
                    smoke = [json.loads(p.read_text(encoding='utf-8')) for p in ready_paths]
                    assert {s['physical_gpu'] for s in smoke} == {0, 1}
                    assert sum(r['nonempty'] for s in smoke for r in s['rows']) > 0, 'No model-selected nonempty search in smoke'
                    pilot.atomic_json(out/'smoke_passed.json', dict(at=pilot.now(), workers=smoke,
                        operational_only=True, relevance_and_accuracy_not_established=True))
                    released = True
                    print('SMOKE_PASSED_BOTH_GPUS FULL_MEDQA_RUN_RELEASED questions=146 models=3', flush=True)
                elif time.monotonic() > deadline:
                    raise TimeoutError('Dual GPU smoke did not finish within 30 minutes')
            if all(c == 0 for c in codes):
                assert released
                break
            time.sleep(3)
        combined = []
        for name in run.MODELS:
            rows = []
            for gpu in range(2):
                data = json.loads((out/f'gpu{gpu}/{name}_checkpoint.json').read_text(encoding='utf-8'))
                rows += data['dialogues']
            assert len(rows) == 146 and {r['id'] for r in rows} == {r['id'] for r in payload['manifest']['items']}
            pilot.atomic_json(out/f'{name}_checkpoint.json', dict(model=name, completed=146, total=146,
                dialogues=sorted(rows, key=lambda r:r['id']), shard_sources=['gpu0', 'gpu1'],
                protocol_sha256=pilot.sha(payload['protocol'])))
            combined.append(dict(model=name, questions=146, final_answer_turns=438))
        pilot.atomic_json(out/'cloud_completed.json', dict(at=pilot.now(), models=combined,
            conversations=438, final_answer_turns=1314, scoring_provisional=True))
        print('FULL_RUN_COMPLETE conversations=438 final_answer_turns=1314', flush=True)
    except Exception:
        pilot.atomic_json(out/'abort.json', dict(at=pilot.now(), error=traceback.format_exc()))
        print('CONTROLLER_ERROR', traceback.format_exc(), flush=True)
        raise
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
        for process in children:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for thread in readers:
            thread.join(timeout=5)


if __name__ == '__main__':
    main()
