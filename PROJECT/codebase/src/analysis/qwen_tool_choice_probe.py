"""Diagnostic: why does Qwen rarely pick the tool at the 1/2 action choice?

Replays the exact 20 recorded decision prompts (10 questions x P1/P2) from the
T3 smoke run, with Phi-4 as control, and changes one thing at a time in the
router message. For each prompt it records P(use) (conditional on tokens 1/2),
the probability mass on 1 and 2, the unconstrained top-5 next tokens, and a short
free generation. Read-only analysis: no benchmark answers are generated or scored.

    python qwen_tool_choice_probe.py --build            # write Kaggle bundle
    python qwen_tool_choice_probe.py --build --launch   # and push it (2xT4, one model per GPU)
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import zlib

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[3]
SMOKE = PROJECT / 'codebase/results/textbook_tool_smoke_20261110'
CALLS = {'qwen': SMOKE / 'kaggle_output/t3/PROJECT/codebase/results/textbook_tool_smoke_20261110/qwen/qwen_model_calls.jsonl',
         'phi4': SMOKE / 'phi4/phi4_model_calls.jsonl'}
MODELS = {'qwen': ('Qwen/Qwen2.5-3B-Instruct', 'aa8e72537993ba99e69dfaafa59ed015b17504d1'),
          'phi4': ('unsloth/Phi-4-mini-instruct-bnb-4bit', 'cece1fd36f04ff79f55ec861f206ca4e16acea6e')}
OUT = PROJECT / 'codebase/results/qwen_tool_choice_probe_20261110'
SLUG = 'subhadeepsing/qwen-tool-choice-probe-20261110'
NAME = 'search_medical_textbooks'
ORIGINAL = (f'Choose your next action. A verification tool named {NAME} is available. '
            'Reply 1 to use it before answering, or 2 to answer without it. '
            'Both choices are allowed. Reply with only 1 or 2.')


def router(name=NAME, swap=False, description=''):
    first, second = ('to answer without it', 'to use it before answering') if swap else ('to use it before answering', 'to answer without it')
    text = (f'Choose your next action. A verification tool named {name} is available.{description} '
            f'Reply 1 {first}, or 2 {second}. Both choices are allowed. Reply with only 1 or 2.')
    return text


VARIANTS = {
    'original': dict(),
    'swapped_order (1=answer, 2=tool)': dict(swap=True),
    'name=search_web': dict(name='search_web'),
    'name=retrieve_answer_sheet': dict(name='retrieve_answer_sheet'),
    'name=lookup_tool (neutral)': dict(name='lookup_tool'),
    'with description': dict(description=' It returns short passages from standard medical textbooks about this question.'),
    'no assistant prefill': dict(prefill=False),
}

WORKER = r'''
import json, sys, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download
name, repo, rev, cases_path, out_path = sys.argv[1:6]
path = snapshot_download(repo_id=repo, revision=rev, allow_patterns=["*.json","*.safetensors","*.model","tokenizer*","*.py","*.txt"])
config = AutoConfig.from_pretrained(path)
stored = getattr(config, "quantization_config", None)
quant = BitsAndBytesConfig.from_dict(stored) if stored else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
quant.bnb_4bit_compute_dtype = torch.float32 if name == "qwen" else torch.float16   # same as the benchmark loader
if stored:
    config.quantization_config = quant.to_dict()
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, config=config, device_map={"": 0}, dtype=torch.float16,
                                             quantization_config=quant, attn_implementation="sdpa").eval()
ids = [tok.encode(s, add_special_tokens=False) for s in ("1", "2")]
assert all(len(x) == 1 for x in ids)
one, two = ids[0][0], ids[1][0]
cases = json.load(open(cases_path))
with open(out_path, "w") as out:
    for c in cases:
        text = tok.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            logits = model(**enc, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        probs = logits.softmax(0)
        p1, p2 = probs[one].item(), probs[two].item()
        tool_token = two if c["swap"] else one
        p_use = (probs[tool_token] / (probs[one] + probs[two])).item()
        top = torch.topk(probs, 5)
        row = dict(c, messages=None, p_use=p_use, mass_1=p1, mass_2=p2,
                   top5=[[tok.decode([int(i)]), round(float(p), 4)] for p, i in zip(top.values, top.indices)])
        if c["generate"]:
            with torch.inference_mode():
                g = model.generate(**enc, max_new_tokens=24, do_sample=False)
            row["free_text"] = tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
        print(name, c["variant"], c["question_id"], c["turn"], round(p_use, 3), flush=True)
'''

BOOT = r'''"""Qwen tool-choice probe (diagnostic). Generated by qwen_tool_choice_probe.py."""
import base64, json, os, subprocess, sys, zlib
from pathlib import Path
P = "{payload}"

def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0", "accelerate==1.15.0", "bitsandbytes==0.50.2"], check=True)
    p = json.loads(zlib.decompress(base64.b64decode(P)))
    work = Path("/kaggle/working/probe"); work.mkdir(parents=True, exist_ok=True)
    (work / "worker.py").write_text(p["worker"])
    procs = []
    for gpu, (name, spec) in enumerate(p["models"].items()):
        (work / f"{{name}}_cases.json").write_text(json.dumps(p["cases"][name]))
        cmd = [sys.executable, str(work / "worker.py"), name, spec[0], spec[1], str(work / f"{{name}}_cases.json"), str(work / f"{{name}}_probe.jsonl")]
        procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))))
    codes = [q.wait() for q in procs]
    print("PROBE_DONE", codes, flush=True)
    if any(codes):
        raise SystemExit(codes)

main()
'''


def cases_for(model):
    rows = [json.loads(line) for line in CALLS[model].open(encoding='utf-8')]
    decisions = [r for r in rows if r['kind'] == 'action_selection']
    assert len(decisions) == 20 and all(r['messages'][-1]['content'] == ORIGINAL for r in decisions)
    cases = []
    for variant, opts in VARIANTS.items():
        for d in decisions:
            messages = [dict(m) for m in d['messages']]
            if opts.get('prefill', True) is False:
                messages = messages[:-2] + messages[-1:]
            messages[-1]['content'] = router(opts.get('name', NAME), opts.get('swap', False), opts.get('description', ''))
            cases.append(dict(variant=variant, question_id=d['question_id'], turn=d['turn'], swap=opts.get('swap', False),
                              recorded_p_use=d['response']['conditional_action_probabilities'][0],
                              generate=variant == 'original', messages=messages))
    return cases


def build(launch):
    payload = dict(worker=WORKER, models=MODELS, cases={m: cases_for(m) for m in MODELS})
    blob = base64.b64encode(zlib.compress(json.dumps(payload, ensure_ascii=False).encode(), 9)).decode()
    source = BOOT.format(payload=blob)
    compile(source, 'probe.py', 'exec')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'kernel').mkdir(exist_ok=True)
    (OUT / 'kernel/probe.py').write_text(source, encoding='utf-8')
    print('cases', {m: len(c) for m, c in payload['cases'].items()})
    if launch:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
        from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
        api = KaggleApi(); api.authenticate()
        req = ApiSaveKernelRequest()
        req.slug, req.new_title, req.text = SLUG, 'Qwen Tool Choice Probe 20261110', source
        req.language, req.kernel_type, req.is_private = 'python', 'script', True
        req.enable_internet, req.enable_gpu, req.machine_shape = True, True, 'NvidiaTeslaT4'
        req.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
        with api.build_kaggle_client() as client:
            resp = client.kernels.kernels_api_client.save_kernel(req)
        if resp.error:
            raise RuntimeError(str(resp.error))
        (OUT / 'kaggle_launch.json').write_text(json.dumps(resp.to_dict(), indent=2, default=str), encoding='utf-8')
        print(resp.to_dict())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--launch', action='store_true')
    a = ap.parse_args()
    build(a.launch)
