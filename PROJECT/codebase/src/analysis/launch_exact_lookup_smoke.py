"""Build and, when requested, execute the private exact-lookup smoke job."""
import argparse
import base64
import json
from pathlib import Path
import sys
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis import exact_lookup_smoke as run
from analysis import web_search_pilot as pilot
from analysis import local_gpu_followup as local
from analysis import medqa_web_dual_gpu as revisions


def bundle():
    manifest,protocol=run.prepare()
    files=[Path(run.__file__),Path(pilot.__file__),Path(local.__file__),Path(revisions.__file__),
        pilot.SRC/'analysis/audit_saved_reply_quality.py',pilot.SRC/'benchmark/answer_parser.py',pilot.RUNNER_SOURCE]
    payload=dict(files={p.relative_to(pilot.PROJECT).as_posix():p.read_text(encoding='utf-8') for p in files},
        manifest=manifest,protocol=protocol,source_hashes={p.relative_to(pilot.PROJECT).as_posix():pilot.sha(p) for p in files},
        models={name:dict(repo=local.MODEL_IDS[name],revision=run.REVISIONS[name]) for name in run.MODELS})
    blob=base64.b64encode(zlib.compress(json.dumps(payload,ensure_ascii=False).encode(),9)).decode()
    template=Path(__file__).with_name('kaggle_exact_lookup_bootstrap.py')
    source=template.read_text(encoding='utf-8').replace('__EXACT_LOOKUP_PAYLOAD__',blob)
    compile(source,'run_smoke.py','exec')
    folder=run.OUT/'kaggle_bundle';folder.mkdir(exist_ok=True)
    (folder/'run_smoke.py').write_text(source,encoding='utf-8')
    metadata=dict(id='subhadeepsing/exact-answer-lookup-smoke-20261019',title='Exact Answer Lookup Smoke 20261019',
        code_file='run_smoke.py',language='python',kernel_type='script',is_private=True,
        enable_gpu=True,enable_internet=True,machine_shape='NvidiaTeslaT4',
        dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[])
    pilot.atomic_json(folder/'kernel-metadata.json',metadata)
    guards=[pilot.PROJECT/'Midsem report/revision_from_rough_20261015/midsem_report.pdf',
            pilot.SRC/'benchmark/answer_parser.py',Path(pilot.__file__),revisions.OUT/'final_verification.json']
    pilot.atomic_json(run.OUT/'source_guards.json',dict(files={str(p):pilot.sha(p) for p in guards}))
    pilot.atomic_json(run.OUT/'bundle_verification.json',dict(source_sha256=pilot.sha(folder/'run_smoke.py'),
        manifest_sha256=pilot.sha(run.OUT/'manifest.json'),source_hashes=payload['source_hashes'],
        bootstrap_sha256=pilot.sha(template),shard_sizes=[len(s['items']) for s in run.shards(manifest)]))
    return folder,metadata


def launch():
    folder,metadata=bundle()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
    api=KaggleApi();api.authenticate()
    request=ApiSaveKernelRequest()
    request.slug=metadata['id'];request.new_title=metadata['title']
    request.text=(folder/'run_smoke.py').read_text(encoding='utf-8')
    request.language='python';request.kernel_type='script';request.is_private=True
    request.enable_gpu=True;request.enable_internet=True;request.machine_shape='NvidiaTeslaT4'
    request.kernel_execution_type=KernelExecutionType.SAVE_AND_RUN_ALL
    with api.build_kaggle_client() as client:result=client.kernels.kernels_api_client.save_kernel(request)
    if result.error:raise RuntimeError(str(result.error))
    record=dict(at=pilot.now(),kernel=metadata['id'],is_private=True,questions=20,models=3,
        scope='Smoke only',source_sha256=pilot.sha(folder/'run_smoke.py'),response=result.to_dict())
    pilot.atomic_json(run.OUT/'kaggle_launch.json',record)
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    if args.launch:launch()
    else:print(bundle()[1])
