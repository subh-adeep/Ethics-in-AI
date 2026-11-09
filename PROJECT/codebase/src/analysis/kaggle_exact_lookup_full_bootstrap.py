"""Private two-GPU full exact-ID lookup controller."""
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

PAYLOAD_B64='__EXACT_LOOKUP_FULL_PAYLOAD__'


def main():
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.17.0',
        'accelerate==1.15.0','bitsandbytes==0.50.2'],check=True)
    payload=json.loads(zlib.decompress(base64.b64decode(PAYLOAD_B64)))
    project=Path('/kaggle/working/exact_lookup_full_project/PROJECT')
    for relative,text in payload['files'].items():
        path=project/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
    for package in ('analysis','benchmark'):
        (project/'codebase/src'/package/'__init__.py').write_text('',encoding='utf-8')
    out=project/'codebase/results/exact_lookup_full_20261020'
    out.mkdir(parents=True,exist_ok=True)
    for filename,value in [('manifest.json',payload['manifest']),('protocol.json',payload['protocol']),
                           ('source_hashes.json',payload['source_hashes'])]:
        (out/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    import torch
    assert torch.cuda.device_count()==2,'Two GPUs required; refusing a one-GPU run'
    environment=dict(python=sys.version,devices=[dict(index=i,name=torch.cuda.get_device_name(i),
        memory_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(2)],
        packages={name:importlib.metadata.version(name) for name in ('torch','transformers','accelerate','bitsandbytes')})
    (out/'cloud_environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    print('DUAL_GPU_CONFIRMED',json.dumps(environment['devices']),flush=True)
    from huggingface_hub import snapshot_download
    cache=Path('/tmp/ethics_exact_lookup_model_cache')
    for name,spec in payload['models'].items():
        snapshot_download(repo_id=spec['repo'],revision=spec['revision'],cache_dir=cache,
            allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','*.py','*.txt'])
        refs=cache/('models--'+spec['repo'].replace('/','--'))/'refs';refs.mkdir(parents=True,exist_ok=True)
        (refs/'main').write_text(spec['revision'],encoding='utf-8')
        print('MODEL_READY',name,spec['revision'],flush=True)
    sys.path.insert(0,str(project/'codebase/src'))
    from analysis import exact_lookup_full as run
    from analysis import web_search_pilot as pilot
    children=[];readers=[];active={};queue=list(run.MODELS);finished={}
    def relay(process,gpu,name):
        with (out/f'{name}_worker.log').open('a',encoding='utf-8') as stream:
            for line in process.stdout:
                stream.write(line);stream.flush();print(f'[GPU{gpu}/{name}] {line.rstrip()}',flush=True)
    def save_workers():
        pilot.atomic_json(out/'workers.json',dict(at=pilot.now(),
            active={str(gpu):dict(model=task['model'],pid=task['process'].pid) for gpu,task in active.items()},
            queued=list(queue),finished=dict(finished)))
    try:
        print('FULL_RUN_STARTED questions=361 models=3 conversations=1083 answer_turns=3249',flush=True)
        while queue or active:
            for gpu,task in list(active.items()):
                code=task['process'].poll()
                if code is None:continue
                if code!=0:raise RuntimeError(f"Model {task['model']} failed on GPU {gpu}: {code}")
                finished[task['model']]=dict(gpu=gpu,at=pilot.now())
                del active[gpu]
                save_workers()
            for gpu in range(2):
                if gpu in active or not queue:continue
                name=queue.pop(0)
                process=subprocess.Popen([sys.executable,'-m','analysis.exact_lookup_full','--gpu',str(gpu),'--model',name],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',
                        TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(project/'codebase/src')),
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
                children.append(process);active[gpu]=dict(model=name,process=process)
                reader=threading.Thread(target=relay,args=(process,gpu,name),daemon=True);reader.start();readers.append(reader)
                save_workers()
                print('MODEL_STARTED',name,'GPU',gpu,'questions=361',flush=True)
            time.sleep(3)
        assert all(p.returncode==0 for p in children)
        summary=[]
        task_manifest=run.task_manifest(payload['manifest'])
        for name in run.MODELS:summary+=run.validate(out/name,task_manifest,models=[name])
        ids={r['id'] for r in payload['manifest']['items']}
        for name in run.MODELS:
            rows=json.loads((out/name/f'{name}_checkpoint.json').read_text(encoding='utf-8'))['dialogues']
            assert len(rows)==361 and {r['id'] for r in rows}==ids
            pilot.atomic_json(out/f'{name}_checkpoint.json',dict(model=name,completed=361,total=361,
                dialogues=sorted(rows,key=lambda r:r['id']),protocol_sha256=pilot.sha(payload['protocol']),
                source_model_folder=name,execution_gpu=finished[name]['gpu']))
        count=sum(r['tool_calls'] for r in summary)
        completed=dict(at=pilot.now(),passed=True,scope='Full exact-ID single-entry optional tool condition',
            dataset_counts={'medqa':146,'math':215},models=3,conversations=1083,answer_turns=3249,
            tool_calls=count,correct_matching_entries_delivered=count,lookup_failures=0,rows=summary,
            model_gpu_assignments=finished,scoring_provisional=True)
        pilot.atomic_json(out/'cloud_completed.json',completed)
        print('FULL_RUN_COMPLETE',json.dumps(completed),flush=True)
    except Exception:
        pilot.atomic_json(out/'abort.json',dict(at=pilot.now(),error=traceback.format_exc()))
        raise
    finally:
        for process in children:
            if process.poll() is None:process.terminate()
        for process in children:
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for reader in readers:reader.join(timeout=5)


if __name__=='__main__':main()
