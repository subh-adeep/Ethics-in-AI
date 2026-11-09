"""Launch the same fixed Phi-4 pilot privately on Kaggle; no main-run changes."""
import base64
import json
from pathlib import Path
import sys
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.web_search_pilot import OUT,PROJECT,CAPTURE,atomic_json,sha,now


def main():
    destination=OUT/'kaggle_phi4_bundle'
    destination.mkdir(exist_ok=True)
    paths=[
        PROJECT/'codebase/src/analysis/web_search_pilot.py',
        PROJECT/'codebase/src/analysis/local_gpu_followup.py',
        PROJECT/'codebase/src/analysis/audit_saved_reply_quality.py',
        PROJECT/'codebase/src/benchmark/answer_parser.py',
        CAPTURE/'parallel_assets/runner.py',
    ]
    payload={'files':{str(p.relative_to(PROJECT)).replace('\\','/'):p.read_text(encoding='utf-8') for p in paths},
             'manifest':json.loads((OUT/'manifest.json').read_text(encoding='utf-8')),
             'protocol':json.loads((OUT/'protocol.json').read_text(encoding='utf-8')),
             'source_hashes':{str(p.relative_to(PROJECT)):sha(p) for p in paths}}
    blob=base64.b64encode(zlib.compress(json.dumps(payload,ensure_ascii=False).encode(),9)).decode()
    bootstrap='''import base64, zlib, json, os, sys, subprocess, traceback
from pathlib import Path
from datetime import datetime, timezone
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.17.0','accelerate==1.15.0','bitsandbytes==0.50.2','ddgs==9.16.0'],check=True)
payload=json.loads(zlib.decompress(base64.b64decode(BLOB)))
project=Path('/kaggle/working/pilot_project/PROJECT')
for relative,text in payload['files'].items():
    path=project/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
for package in ['analysis','benchmark']:
    (project/'codebase/src'/package/'__init__.py').write_text('',encoding='utf-8')
model_cache=Path('/tmp/ethics_web_pilot_model_cache')
from huggingface_hub import snapshot_download
revision='cece1fd36f04ff79f55ec861f206ca4e16acea6e'
repo='unsloth/Phi-4-mini-instruct-bnb-4bit'
snapshot_download(repo_id=repo,revision=revision,cache_dir=model_cache,
                  allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','*.py','*.txt'])
refs=model_cache/'models--unsloth--Phi-4-mini-instruct-bnb-4bit/refs'
refs.mkdir(parents=True,exist_ok=True);(refs/'main').write_text(revision)
sys.path.insert(0,str(project/'codebase/src'))
from analysis import local_gpu_followup as local
local.MODEL_ROOT=model_cache
from analysis import web_search_pilot as pilot
pilot.OUT.mkdir(parents=True,exist_ok=True)
pilot.atomic_json(pilot.OUT/'manifest.json',payload['manifest'])
pilot.atomic_json(pilot.OUT/'protocol.json',payload['protocol'])
pilot.atomic_json(pilot.OUT/'uploaded_source_hashes.json',payload['source_hashes'])
import importlib.metadata,platform,torch
pilot.atomic_json(pilot.OUT/'cloud_environment.json',dict(python=sys.version,platform=platform.platform(),
    gpu=torch.cuda.get_device_name(0),model_revision=revision,
    packages={x:importlib.metadata.version(x) for x in ['torch','transformers','accelerate','bitsandbytes','ddgs','primp']}))
original_init=local.LocalGPU.__init__
def cloud_init(self,name):
    original_init(self,name)
    self.metadata['execution_host']='Kaggle private pilot'
    self.metadata['changed_hardware_from_original']='Pilot uses actual Kaggle GPU listed in gpu field; other pilot models use GTX 1650'
    self.metadata['model_weights_local_to_execution_host']=True
    self.metadata['model_downloaded_for_this_pilot']=True
local.LocalGPU.__init__=cloud_init
try:
    pilot.run_model('phi4',payload['manifest'])
    pilot.summary()
    pilot.atomic_json(pilot.OUT/'cloud_completed.json',dict(completed_at=pilot.now(),model='phi4',questions=20))
except Exception:
    pilot.append_jsonl(pilot.OUT/'runtime_errors.jsonl',dict(at=pilot.now(),error=traceback.format_exc()))
    raise
'''
    source=bootstrap.replace('BLOB',repr(blob))
    (destination/'run_pilot.py').write_text(source,encoding='utf-8')
    metadata=dict(id='subhadeepsing/ethics-web-pilot-phi4-20-questions-20261018',title='Ethics Web Pilot Phi4 20 Questions 20261018',
                  code_file='run_pilot.py',language='python',kernel_type='script',is_private=True,
                  enable_gpu=True,enable_tpu=False,enable_internet=True,dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[])
    atomic_json(destination/'kernel-metadata.json',metadata)
    # Explicit user authorization permits either Kaggle or local GPU for this pilot.
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    result=api.kernels_push(str(destination))
    record=dict(at=now(),kernel=metadata['id'],is_private=True,questions=20,
                reason='Run Phi-4 concurrently while laptop completes LLaMA',source_sha256=sha(destination/'run_pilot.py'),
                response=result.to_dict() if hasattr(result,'to_dict') else str(result))
    atomic_json(OUT/'kaggle_launch.json',record)
    print(json.dumps(record))


if __name__=='__main__':
    main()
