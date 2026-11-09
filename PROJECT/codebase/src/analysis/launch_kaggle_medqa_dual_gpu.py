"""Package the isolated 146-question MedQA run and explicitly execute it privately."""
import argparse
import base64
import json
from pathlib import Path
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import medqa_web_dual_gpu as run
from analysis import web_search_pilot as pilot
from analysis import local_gpu_followup as local


def bundle():
    manifest, protocol = run.prepare()
    files = [Path(__file__), Path(run.__file__), Path(pilot.__file__), Path(local.__file__),
             pilot.SRC/'analysis/audit_saved_reply_quality.py',
             pilot.SRC/'benchmark/answer_parser.py', pilot.RUNNER_SOURCE]
    payload = dict(files={p.relative_to(pilot.PROJECT).as_posix(): p.read_text(encoding='utf-8') for p in files},
        manifest=manifest, protocol=protocol,
        models={name: dict(repo=local.MODEL_IDS[name], revision=run.REVISIONS[name]) for name in run.MODELS},
        source_hashes={p.relative_to(pilot.PROJECT).as_posix(): pilot.sha(p) for p in files})
    blob = base64.b64encode(zlib.compress(json.dumps(payload, ensure_ascii=False).encode(), 9)).decode()
    template_path = Path(__file__).with_name('kaggle_medqa_dual_gpu_bootstrap.py')
    template = template_path.read_text(encoding='utf-8')
    source = template.replace('__MEDQA_PAYLOAD__', blob)
    compile(source, 'run_medqa.py', 'exec')
    folder = run.OUT/'kaggle_bundle'
    folder.mkdir(exist_ok=True)
    (folder/'run_medqa.py').write_text(source, encoding='utf-8')
    metadata = dict(id='subhadeepsing/medqa-web-search-dual-gpu-20261019',
        title='MedQA Web Search Dual GPU 20261019', code_file='run_medqa.py',
        language='python', kernel_type='script', is_private=True,
        enable_gpu=True, enable_tpu=False, enable_internet=True,
        dataset_sources=[], competition_sources=[], kernel_sources=[], model_sources=[])
    pilot.atomic_json(folder/'kernel-metadata.json', metadata)
    report = pilot.PROJECT/'Midsem report/revision_from_rough_20261015/midsem_report.pdf'
    pilot.atomic_json(run.OUT/'source_guards.json', dict(at=pilot.now(),
        files={str(p): pilot.sha(p) for p in [report, pilot.SRC/'benchmark/answer_parser.py', Path(pilot.__file__)]}))
    pilot.atomic_json(run.OUT/'bundle_verification.json', dict(at=pilot.now(), questions=146,
        shard_sizes=[len(s['items']) for s in run.shards(manifest)],
        source_bytes=len(source.encode()), source_sha256=pilot.sha(folder/'run_medqa.py'),
        model_revisions=run.REVISIONS, source_hashes=payload['source_hashes'],
        bootstrap_template_sha256=pilot.sha(template_path)))
    return folder, metadata


def launch():
    folder, metadata = bundle()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
    api = KaggleApi()
    api.authenticate()
    request = ApiSaveKernelRequest()
    request.slug = metadata['id']
    request.new_title = metadata['title']
    request.text = (folder/'run_medqa.py').read_text(encoding='utf-8')
    request.language = 'python'
    request.kernel_type = 'script'
    request.is_private = True
    request.enable_internet = True
    request.enable_gpu = True
    request.machine_shape = 'NvidiaTeslaT4'
    request.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.save_kernel(request)
    if response.error:
        raise RuntimeError(str(response.error))
    record = dict(at=pilot.now(), requested_kernel=metadata['id'], is_private=True,
        questions=146, models=3, requested_gpus=2,
        source_sha256=pilot.sha(folder/'run_medqa.py'), response=response.to_dict())
    pilot.atomic_json(run.OUT/'kaggle_launch.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        launch()
    else:
        folder, metadata = bundle()
        print(json.dumps(dict(folder=str(folder), metadata=metadata)), flush=True)
