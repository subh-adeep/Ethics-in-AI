"""Private dual-GPU smoke only; payload inserted before submission."""
import base64
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import zlib

PAYLOAD_B64='__EXACT_LOOKUP_PAYLOAD__'


def main():
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.17.0',
        'accelerate==1.15.0','bitsandbytes==0.50.2'],check=True)
    payload=json.loads(zlib.decompress(base64.b64decode(PAYLOAD_B64)))
    project=Path('/kaggle/working/exact_lookup_project/PROJECT')
    for relative,text in payload['files'].items():
        path=project/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text,encoding='utf-8')
    for package in ('analysis','benchmark'):
        (project/'codebase/src'/package/'__init__.py').write_text('',encoding='utf-8')
    out=project/'codebase/results/exact_lookup_smoke_20261019'
    out.mkdir(parents=True,exist_ok=True)
    for filename,value in [('manifest.json',payload['manifest']),('protocol.json',payload['protocol']),
                            ('source_hashes.json',payload['source_hashes'])]:
        (out/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    import torch
    assert torch.cuda.device_count()==2, 'Two GPUs required; refuse silent fallback'
    env=dict(python=sys.version,devices=[dict(index=i,name=torch.cuda.get_device_name(i)) for i in range(2)],
        packages={name:importlib.metadata.version(name) for name in ('torch','transformers','accelerate','bitsandbytes')})
    (out/'cloud_environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    print('DUAL_GPU_CONFIRMED',json.dumps(env['devices']),flush=True)
    from huggingface_hub import snapshot_download
    cache=Path('/tmp/ethics_exact_lookup_model_cache')
    for name,spec in payload['models'].items():
        snapshot_download(repo_id=spec['repo'],revision=spec['revision'],cache_dir=cache,
            allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','*.py','*.txt'])
        refs=cache/('models--'+spec['repo'].replace('/','--'))/'refs'
        refs.mkdir(parents=True,exist_ok=True)
        (refs/'main').write_text(spec['revision'],encoding='utf-8')
        print('MODEL_READY',name,spec['revision'],flush=True)
    sys.path.insert(0,str(project/'codebase/src'))
    from analysis import exact_lookup_smoke as run
    from analysis import web_search_pilot as pilot
    children=[]
    readers=[]
    def relay(process,gpu):
        with (out/f'gpu{gpu}_worker.log').open('a',encoding='utf-8') as f:
            for line in process.stdout:
                f.write(line);f.flush()
                print(f'[GPU{gpu}] {line.rstrip()}',flush=True)
    try:
        for gpu in range(2):
            process=subprocess.Popen([sys.executable,'-m','analysis.exact_lookup_smoke','--worker',str(gpu)],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',
                    TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(project/'codebase/src')),
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
            children.append(process)
            thread=threading.Thread(target=relay,args=(process,gpu),daemon=True)
            thread.start();readers.append(thread)
        import time
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):
                raise RuntimeError('A smoke worker failed')
            time.sleep(2)
        assert all(p.returncode==0 for p in children)
        summary=[]
        for gpu,shard in enumerate(run.shards(payload['manifest'])):
            summary+=run.validate(out/f'gpu{gpu}',shard)
        for name in run.MODELS:
            rows=[]
            for gpu in range(2):
                rows+=json.loads((out/f'gpu{gpu}/{name}_checkpoint.json').read_text(encoding='utf-8'))['dialogues']
            assert len(rows)==20 and {r['id'] for r in rows}=={i['id'] for i in payload['manifest']['items']}
            pilot.atomic_json(out/f'{name}_checkpoint.json',dict(model=name,completed=20,total=20,dialogues=rows))
        for domain in ('medqa','math'):
            assert sum(r['tool_calls'] for r in summary if r['dataset']==domain)>0, 'No real model-selected lookup for dataset'
        calls=sum(r['tool_calls'] for r in summary)
        record=dict(at=pilot.now(),passed=True,scope='Smoke only; full run not started',
            dataset_counts={'medqa':10,'math':10},models=3,conversations=60,answer_turns=180,
            tool_calls=calls,correct_matching_entries_delivered=calls,lookup_failures=0,rows=summary)
        pilot.atomic_json(out/'cloud_completed.json',record)
        print('SMOKE_COMPLETE',json.dumps(record),flush=True)
    except Exception:
        pilot.atomic_json(out/'abort.json',dict(at=pilot.now(),error=traceback.format_exc()))
        raise
    finally:
        for process in children:
            if process.poll() is None:process.terminate()
        for process in children:
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for thread in readers:thread.join(timeout=5)


if __name__=='__main__':
    main()
