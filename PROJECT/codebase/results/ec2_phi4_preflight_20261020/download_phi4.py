"""Download only the experiment's pinned Phi-4 snapshot; no access keys."""
from pathlib import Path
from huggingface_hub import snapshot_download

repo = 'unsloth/Phi-4-mini-instruct-bnb-4bit'
revision = 'cece1fd36f04ff79f55ec861f206ca4e16acea6e'
cache = Path('/home/ubuntu/ethics-ec2/model-cache')
path = snapshot_download(repo_id=repo, revision=revision, cache_dir=cache,
                         allow_patterns=['*.json', '*.safetensors', '*.model', 'tokenizer*', '*.py', '*.txt'])
refs = cache/('models--'+repo.replace('/', '--'))/'refs'
refs.mkdir(parents=True, exist_ok=True)
(refs/'main').write_text(revision, encoding='utf-8')
print('PINNED_PHI4_READY', path, flush=True)
