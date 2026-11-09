"""Bounded, read-only status/log snapshot for the private MedQA deployment."""
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.medqa_web_dual_gpu import OUT
from analysis.web_search_pilot import atomic_json, now


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest
    api = KaggleApi()
    api.authenticate()
    ref = 'subhadeepsing/medqa-web-search-dual-gpu-20261019'
    status = api.kernels_status(ref)
    record = dict(at=now(), kernel=ref, status=status.status.name, failure_message=status.failure_message)
    print('STATUS', json.dumps(record), flush=True)
    request = ApiGetKernelSessionLogsStreamRequest()
    request.user_name, request.kernel_slug = ref.split('/')
    request.wait_for_logs_url_seconds = 1
    text = ''
    try:
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
            kind = response.headers.get('content-type', '')
            record['log_content_type'] = kind
            if 'application/json' in kind:
                rows = response.json()
                text = '\n'.join(str(r.get('data', r)) for r in rows) if isinstance(rows, list) else json.dumps(rows)
            else:
                lines = []
                deadline = time.monotonic()+15
                timer = threading.Timer(20, response.close)
                timer.daemon = True
                timer.start()
                try:
                    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                        if line:
                            lines.append(line.decode('utf-8', 'replace') if isinstance(line, bytes) else line)
                        if time.monotonic() > deadline or len(lines) > 5000:
                            break
                except Exception as exc:
                    record['bounded_stream_note'] = type(exc).__name__
                finally:
                    timer.cancel()
                fragments = []
                for line in lines:
                    raw = line[5:].strip() if line.startswith('data:') else line
                    try:
                        value = json.loads(raw)
                        fragments.append(str(value.get('data', value)) if isinstance(value, dict) else str(value))
                    except (ValueError, TypeError):
                        fragments.append(raw)
                text = '\n'.join(fragments)
            response.close()
    except Exception as exc:
        record['log_error'] = f'{type(exc).__name__}: {exc}'
    stamp = record['at'].replace(':', '').replace('.', '_')
    folder = OUT/'monitoring'
    folder.mkdir(exist_ok=True)
    (folder/f'{stamp}.log').write_text(text, encoding='utf-8')
    indicators = ['DUAL_GPU_CONFIRMED', 'GPU_READY', 'SMOKE_READY', 'SMOKE_PASSED',
                  'FULL_RUN_STARTED', 'SHARD_COMPLETE', 'FULL_RUN_COMPLETE', 'CONTROLLER_ERROR',
                  'TURN ', 'Error:', 'Traceback', 'DOWNLOAD_MODEL']
    relevant = [s for s in text.splitlines() if any(k in s for k in indicators)]
    record['key_log_lines'] = relevant[-35:]
    record['log_characters'] = len(text)
    atomic_json(folder/f'{stamp}.json', record)
    atomic_json(OUT/'latest_status.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
