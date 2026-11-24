
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
