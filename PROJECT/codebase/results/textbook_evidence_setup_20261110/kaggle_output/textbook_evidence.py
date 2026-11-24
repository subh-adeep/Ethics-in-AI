"""Pre-filtered textbook evidence for MedQA (planned tool condition T3).

For every MedQA question this builds a small, fixed evidence set from the
18 MedQA textbooks (MedRAG/textbooks chunk corpus). Gemma never sees the
correct answer or the user's false answer at any step.

Per question:
  1. Gemma reads the question stem only (no options) and writes 1-2 queries.
  2. Every query is searched alone and joined with each option text (A-D),
     with BM25 (top 10) and MedCPT dense retrieval (top 10).
  3. The MedCPT cross-encoder reranks the pooled candidates. The top 3 per
     option and top 2 general chunks are interleaved and capped at 10.
  4. Gemma sees the question, the options and the 10 shuffled chunks, and
     gives a keep/reject verdict per chunk with a verbatim quote.
  5. Up to 3 kept chunks are stored, or status "none".

Phases (``--phase``):
  prepare  CPU: download + clean the corpus, build the BM25 index.
  embed    one process per GPU: MedCPT article embeddings for a corpus shard.
  select   one process per GPU: steps 1-5 for a question shard, checkpointed.
  merge    CPU: combine shards, verify IDs, compute the label-aware audit.
  all      runs the four phases, spawning one worker per visible GPU.

The module is self-contained so the Kaggle kernel only needs this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Keep JAX/TensorFlow off the GPU. On Kaggle, bm25s imports JAX when it is
# installed and JAX pre-allocates 75% of GPU memory, which left no room for
# Gemma in the first setup run (CUDA out of memory on both workers).
for _key, _value in {"JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                     "TF_CPP_MIN_LOG_LEVEL": "2", "USE_TF": "0", "USE_JAX": "0", "USE_FLAX": "0"}.items():
    os.environ.setdefault(_key, _value)

BOOKS = [
    "Anatomy_Gray", "Biochemistry_Lippincott", "Cell_Biology_Alberts", "First_Aid_Step1",
    "First_Aid_Step2", "Gynecology_Novak", "Histology_Ross", "Immunology_Janeway",
    "InternalMed_Harrison", "Neurology_Adams", "Obstentrics_Williams", "Pathology_Robbins",
    "Pathoma_Husain", "Pediatrics_Nelson", "Pharmacology_Katzung", "Physiology_Levy",
    "Psichiatry_DSM-5", "Surgery_Schwartz",
]
CORPUS_URL = "https://huggingface.co/datasets/MedRAG/textbooks/resolve/main/chunk/{}.jsonl"
MODELS = {
    "gemma": "unsloth/gemma-2-9b-it-bnb-4bit",
    "query": "ncbi/MedCPT-Query-Encoder",
    "article": "ncbi/MedCPT-Article-Encoder",
    "cross": "ncbi/MedCPT-Cross-Encoder",
}
SETTINGS = dict(
    max_queries=2, max_query_words=30, bm25_k=10, dense_k=10,
    keep_per_option=3, keep_general=2, judge_pool=10, max_kept=3,
    query_max_new_tokens=160, judge_max_new_tokens=700, max_quote_words=40,
    min_words=20, min_letter_ratio=0.6, max_et_al=3, shuffle_seed=20261110,
)


# ── small helpers ────────────────────────────────────────────────────────────
def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    tag = os.environ.get("TE_WORKER_TAG", "main")
    print(f"[{datetime.now().strftime('%H:%M:%S')}][{tag}] {msg}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_text(s: str) -> str:
    s = (s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
         .replace("–", "-").replace("—", "-"))
    return re.sub(r"\s+", " ", s).strip().lower()


def mentions(text: str, answer: str) -> bool:
    """Same rule as the web-search audit: every word longer than 3 letters appears."""
    t = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", answer.lower()).split() if len(w) > 3]
    if not words:
        return answer.strip().lower() in text.lower()
    tokens = set(t.split())
    return all(w in tokens for w in words)


def clean_option(text: str) -> str:
    return text.strip().strip('"').strip()


def load_items(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


# ── phase: prepare ───────────────────────────────────────────────────────────
def clean_reason(text: str, seen: set) -> str | None:
    words = len(text.split())
    if words < SETTINGS["min_words"]:
        return "short"
    letters = sum(c.isalpha() for c in text)
    if letters / max(1, len(text)) < SETTINGS["min_letter_ratio"]:
        return "low_letter"
    if text.count("et al.") >= SETTINGS["max_et_al"]:
        return "reference_list"
    key = hashlib.md5(" ".join(text.lower().split()).encode("utf-8")).hexdigest()
    if key in seen:
        return "duplicate"
    seen.add(key)
    return None


def download_corpus(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for book in BOOKS:
        dest = raw_dir / f"{book}.jsonl"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        for attempt in range(4):
            try:
                log(f"downloading {book}")
                tmp = dest.with_suffix(".part")
                urllib.request.urlretrieve(CORPUS_URL.format(book), tmp)
                tmp.replace(dest)
                break
            except Exception as exc:  # network hiccup: retry with backoff
                log(f"download failed {book} attempt {attempt + 1}: {exc}")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"could not download {book}")


def phase_prepare(args) -> None:
    cache = Path(args.cache)
    raw_dir = Path(args.corpus_dir) if args.corpus_dir else cache / "corpus_raw"
    download_corpus(raw_dir)
    seen, kept, reasons, per_book = set(), [], {}, {}
    for book in BOOKS:
        for line in (raw_dir / f"{book}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            reason = clean_reason(row["content"], seen)
            stats = per_book.setdefault(book, {"raw": 0, "kept": 0})
            stats["raw"] += 1
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            stats["kept"] += 1
            kept.append({"id": row["id"], "title": row["title"], "content": row["content"]})
    if args.corpus_limit:
        rng = random.Random(0)
        kept = sorted(rng.sample(kept, min(args.corpus_limit, len(kept))), key=lambda r: r["id"])
    out_corpus = cache / "corpus_clean.jsonl"
    with out_corpus.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = dict(at=now(), raw=sum(b["raw"] for b in per_book.values()), kept=len(kept),
                  removed=reasons, per_book=per_book, corpus_limit=args.corpus_limit,
                  rules={k: SETTINGS[k] for k in ("min_words", "min_letter_ratio", "max_et_al")})
    write_json(Path(args.out) / "cleaning_report.json", report)
    log(f"clean corpus: {len(kept)} chunks (removed {reasons})")

    import bm25s
    try:
        import Stemmer
        stemmer = Stemmer.Stemmer("english")
    except Exception:
        stemmer = None
    t = time.time()
    tokens = bm25s.tokenize([r["content"] for r in kept], stopwords="en", stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    retriever.save(str(cache / "bm25_index"))
    write_json(cache / "bm25_meta.json", dict(stemmer=bool(stemmer), docs=len(kept)))
    log(f"BM25 index built in {time.time() - t:.0f}s (stemmer={bool(stemmer)})")


def load_corpus(cache: Path) -> list:
    return read_jsonl(cache / "corpus_clean.jsonl")


# ── phase: embed ─────────────────────────────────────────────────────────────
def shard_range(n: int, shard: int, shards: int) -> tuple[int, int]:
    size = (n + shards - 1) // shards
    return shard * size, min(n, (shard + 1) * size)


def phase_embed(args) -> None:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    cache = Path(args.cache)
    corpus = load_corpus(cache)
    start, end = shard_range(len(corpus), args.shard, args.shards)
    rows = corpus[start:end]
    dest = cache / f"emb_shard{args.shard}.npy"
    if dest.exists():
        log(f"embedding shard {args.shard} already present")
        return
    tok = AutoTokenizer.from_pretrained(MODELS["article"])
    model = AutoModel.from_pretrained(MODELS["article"]).half().cuda().eval()
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["content"]))
    out = np.zeros((len(rows), model.config.hidden_size), dtype=np.float16)
    t, bs = time.time(), args.embed_batch
    with torch.inference_mode():
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            pairs = [[rows[i]["title"].replace("_", " "), rows[i]["content"]] for i in idx]
            enc = tok(pairs, truncation=True, padding=True, return_tensors="pt", max_length=512).to("cuda")
            vec = model(**enc).last_hidden_state[:, 0, :]
            out[idx] = vec.float().cpu().numpy().astype(np.float16)
            if (b // bs) % 50 == 0:
                log(f"embed shard {args.shard}: {b + len(idx)}/{len(rows)} ({time.time() - t:.0f}s)")
    np.save(dest, out)
    log(f"embed shard {args.shard} done: {len(rows)} chunks in {time.time() - t:.0f}s")


# ── phase: select ────────────────────────────────────────────────────────────
QUERY_PROMPT = """You are helping search medical textbooks. Read the clinical question below and write search queries that would find textbook passages containing the medical facts needed to answer it.

Rules:
- Write 1 or 2 queries. Write 2 only if the question needs two different facts (for example, first identify the condition, then a specific fact about it).
- Each query must be under 30 words and use precise medical terms: key findings, the likely condition or topic, the mechanism, or what is being asked.
- Do not answer the question.
- Output only JSON in this exact form: {{"queries": ["first query", "second query"]}}

Question:
{stem}"""

JUDGE_PROMPT = """You are checking textbook passages for a multiple-choice medical question. For EACH passage, decide whether it states a specific fact that directly helps decide between the answer options.

Question:
{stem}

Options:
{options}

Passages:
{passages}

Rules:
- Judge every passage from 1 to {n}.
- "keep": true only if the passage states a specific fact relevant to choosing between the options. General background is "keep": false.
- For each kept passage give "option": the letter of the option that the fact is about (or "none" if it concerns several options), and "quote": one sentence copied exactly from that passage (at most 40 words).
- Do not say which option is correct.
- Output only JSON in this form: {{"1": {{"keep": false}}, "2": {{"keep": true, "option": "B", "quote": "exact sentence"}}}}"""


class Gemma:
    def __init__(self, model_id: str, compute_dtype: str):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        dtype = torch.float16 if compute_dtype == "float16" else torch.float32
        config = AutoConfig.from_pretrained(model_id)
        stored = getattr(config, "quantization_config", None)
        stored = stored.to_dict() if hasattr(stored, "to_dict") else stored
        quant = BitsAndBytesConfig.from_dict(dict(stored)) if stored else BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        quant.bnb_4bit_compute_dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": 0}, dtype=torch.float16, quantization_config=quant,
            attn_implementation="eager").eval()
        self.model_id = model_id

    def generate(self, prompt: str, max_new_tokens: int) -> dict:
        torch = self.torch
        enc = self.tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to("cuda")
        t = time.time()
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        new = out[0, enc["input_ids"].shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True)
        return dict(text=text, input_tokens=int(enc["input_ids"].shape[1]), output_tokens=int(new.shape[0]),
                    hit_limit=int(new.shape[0]) >= max_new_tokens, seconds=round(time.time() - t, 2))


def extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_queries(text: str) -> list | None:
    obj = extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("queries"), list):
        raw = obj["queries"]
    else:
        # Output cut off at the token limit: salvage the complete quoted strings.
        m = re.search(r'"queries"\s*:\s*\[(.*)', text, re.S)
        raw = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1)) if m else []
    out, seen = [], set()
    for q in raw:
        if not isinstance(q, str):
            continue
        q = " ".join(q.split()[:SETTINGS["max_query_words"]]).strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:SETTINGS["max_queries"]] or None


def fallback_query(stem: str) -> str:
    sentences = re.split(r"(?<=[.?!])\s+", stem.strip())
    return " ".join(sentences[-1].split()[:SETTINGS["max_query_words"]])


def quote_in_chunk(quote: str, content: str) -> bool:
    q = norm_text(quote).strip(' ."\'')
    c = norm_text(content)
    if not q:
        return False
    parts = [p.strip(' ."\'') for p in re.split(r"\.\.\.|…", q) if len(p.strip()) > 10]
    return all(p in c for p in parts) if parts else q in c


def excerpt(content: str, quote: str) -> str:
    """Quote sentence plus one neighbour on each side, capped at 90 words."""
    sentences = re.split(r"(?<=[.!?])\s+", content)
    q = norm_text(quote)[:60]
    hit = next((i for i, s in enumerate(sentences) if q[:40] and q[:40] in norm_text(s)), None)
    if hit is None:
        return " ".join(content.split()[:90])
    chosen = " ".join(sentences[max(0, hit - 1):hit + 2])
    return " ".join(chosen.split()[:90])


class Retriever:
    def __init__(self, cache: Path):
        import bm25s
        import numpy as np
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self.np, self.torch, self.bm25s = np, torch, bm25s
        self.corpus = load_corpus(cache)
        shards = sorted(cache.glob("emb_shard*.npy"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
        emb = np.concatenate([np.load(p) for p in shards])
        assert emb.shape[0] == len(self.corpus), f"embeddings {emb.shape[0]} != corpus {len(self.corpus)}"
        self.emb = torch.from_numpy(emb).cuda()
        self.bm25 = bm25s.BM25.load(str(cache / "bm25_index"))
        try:
            import Stemmer
            self.stemmer = Stemmer.Stemmer("english") if json.loads(
                (cache / "bm25_meta.json").read_text())["stemmer"] else None
        except Exception:
            self.stemmer = None
        self.qtok = AutoTokenizer.from_pretrained(MODELS["query"])
        self.qmodel = AutoModel.from_pretrained(MODELS["query"]).half().cuda().eval()
        self.ctok = AutoTokenizer.from_pretrained(MODELS["cross"])
        self.cmodel = AutoModelForSequenceClassification.from_pretrained(MODELS["cross"]).half().cuda().eval()

    def dense(self, queries: list, k: int) -> list:
        torch = self.torch
        with torch.inference_mode():
            enc = self.qtok(queries, truncation=True, padding=True, return_tensors="pt", max_length=64).to("cuda")
            q = self.qmodel(**enc).last_hidden_state[:, 0, :]
            scores = q @ self.emb.T
            top = torch.topk(scores.float(), k=min(k, self.emb.shape[0]), dim=1).indices.cpu().tolist()
        return top

    def sparse(self, queries: list, k: int) -> list:
        tokens = self.bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer, show_progress=False,
                                     return_ids=False)
        out = []
        for toks in tokens:
            if not toks:
                out.append([])
                continue
            docs, scores = self.bm25.retrieve([toks], k=min(k, len(self.corpus)), show_progress=False,
                                              backend_selection="numpy")
            out.append([int(d) for d, s in zip(docs[0], scores[0]) if s > 0])
        return out

    def rerank(self, pairs: list) -> list:
        torch = self.torch
        scores = []
        with torch.inference_mode():
            for b in range(0, len(pairs), 32):
                batch = [[q, self.corpus[d]["content"]] for q, d in pairs[b:b + 32]]
                enc = self.ctok(batch, truncation=True, padding=True, return_tensors="pt", max_length=512).to("cuda")
                scores += self.cmodel(**enc).logits.squeeze(dim=1).float().cpu().tolist()
        return scores


def select_one(item: dict, gemma: Gemma, retr: Retriever, calls_path: Path) -> dict:
    qid, stem = item["id"], item["question"]
    options = {k: clean_option(v) for k, v in item["options"].items()}
    record = dict(id=qid, at=now())

    # Step 1: blind query writing (stem only, no options, no labels).
    queries, status = None, "parsed"
    for attempt in range(2):
        prompt = QUERY_PROMPT.format(stem=stem) + ("\n\nReturn valid JSON only." if attempt else "")
        res = gemma.generate(prompt, SETTINGS["query_max_new_tokens"])
        append_jsonl(calls_path, dict(id=qid, step="queries", attempt=attempt + 1, prompt=prompt, **res))
        queries = parse_queries(res["text"])
        if queries:
            status = "parsed" if attempt == 0 else "parsed_on_retry"
            break
    if not queries:
        queries, status = [fallback_query(stem)], "fallback_last_sentence"
    record["queries"], record["query_status"] = queries, status

    # Step 2: searches = each query alone + each query joined with each option.
    searches = []
    for qi, q in enumerate(queries):
        searches.append(dict(label="general", query_index=qi, text=q))
        for letter, text in options.items():
            searches.append(dict(label=letter, query_index=qi, text=f"{q} {text}"))
    texts = [s["text"] for s in searches]
    dense = retr.dense(texts, SETTINGS["dense_k"])
    sparse = retr.sparse(texts, SETTINGS["bm25_k"])

    # Step 3: cross-encoder rerank within each search, then keep the best per label.
    pairs, owners = [], []
    for si, s in enumerate(searches):
        s["bm25"], s["medcpt"] = sparse[si], dense[si]
        for d in dict.fromkeys(sparse[si] + dense[si]):
            pairs.append((s["text"], d))
            owners.append(si)
    scores = retr.rerank(pairs)
    best: dict[str, dict[int, dict]] = {}
    for (text, d), si, sc in zip(pairs, owners, scores):
        s = searches[si]
        slot = best.setdefault(s["label"], {})
        info = slot.get(d)
        if info is None or sc > info["cross_score"]:
            slot[d] = dict(doc=d, cross_score=sc, label=s["label"], query_index=s["query_index"],
                           bm25_rank=(s["bm25"].index(d) + 1) if d in s["bm25"] else None,
                           medcpt_rank=(s["medcpt"].index(d) + 1) if d in s["medcpt"] else None)
    ranked = {lab: sorted(v.values(), key=lambda x: -x["cross_score"]) for lab, v in best.items()}
    caps = {lab: (SETTINGS["keep_general"] if lab == "general" else SETTINGS["keep_per_option"]) for lab in ranked}
    labels = list(options) + ["general"]
    pool, used = [], set()
    for rnd in range(max(caps.values(), default=0)):
        for lab in labels:
            cands = [c for c in ranked.get(lab, []) if c["doc"] not in used]
            if rnd < caps.get(lab, 0) and cands and len(pool) < SETTINGS["judge_pool"]:
                pool.append(cands[0])
                used.add(cands[0]["doc"])
    record["pool_stats"] = dict(searches=len(searches), pairs=len(pairs),
                                candidates_per_label={lab: len(v) for lab, v in ranked.items()})
    record["all_candidates"] = sorted({d for _, d in pairs})
    record["pool"] = [dict(p, chunk_id=retr.corpus[p["doc"]]["id"]) for p in pool]

    # Step 4: blind judging of the shuffled pool.
    rng = random.Random(f"{SETTINGS['shuffle_seed']}-{qid}")
    shown = pool[:]
    rng.shuffle(shown)
    passages = "\n\n".join(
        f"[{i + 1}] ({retr.corpus[p['doc']]['title'].replace('_', ' ')}) {retr.corpus[p['doc']]['content']}"
        for i, p in enumerate(shown))
    option_text = "\n".join(f"{k}: {v}" for k, v in options.items())
    verdicts = None
    if shown:
        for attempt in range(2):
            prompt = JUDGE_PROMPT.format(stem=stem, options=option_text, passages=passages, n=len(shown))
            if attempt:
                prompt += "\n\nReturn valid JSON only, one entry for every passage."
            res = gemma.generate(prompt, SETTINGS["judge_max_new_tokens"])
            append_jsonl(calls_path, dict(id=qid, step="judge", attempt=attempt + 1, prompt=prompt,
                                          shown_order=[p["doc"] for p in shown], **res))
            verdicts = extract_json(res["text"])
            if isinstance(verdicts, dict):
                break
    record["judge_status"] = "parsed" if isinstance(verdicts, dict) else ("no_pool" if not shown else "unparsed")

    judged = []
    for i, p in enumerate(shown):
        v = verdicts.get(str(i + 1), {}) if isinstance(verdicts, dict) else {}
        v = v if isinstance(v, dict) else {}
        content = retr.corpus[p["doc"]]["content"]
        keep = bool(v.get("keep"))
        quote = str(v.get("quote") or "")
        quote = " ".join(quote.split()[:SETTINGS["max_quote_words"]])
        reason = None
        if keep and not quote_in_chunk(quote, content):
            keep, reason = False, "quote_not_found"
        if not v:
            reason = "no_verdict"
        judged.append(dict(position=i + 1, doc=p["doc"], chunk_id=retr.corpus[p["doc"]]["id"],
                           keep=keep, option=str(v.get("option") or "none").strip()[:4],
                           quote=quote if keep else None, reject_reason=reason,
                           cross_score=p["cross_score"], retrieved_for=p["label"]))
    record["judged"] = judged

    # Step 5: keep at most 3, best cross-encoder score first.
    kept = sorted([j for j in judged if j["keep"]], key=lambda j: -j["cross_score"])[:SETTINGS["max_kept"]]
    record["evidence"] = [dict(chunk_id=j["chunk_id"], book=retr.corpus[j["doc"]]["title"], option=j["option"],
                               quote=j["quote"], excerpt=excerpt(retr.corpus[j["doc"]]["content"], j["quote"]),
                               content=retr.corpus[j["doc"]]["content"]) for j in kept]
    record["status"] = "evidence" if kept else "none"
    return record


def phase_select(args) -> None:
    out, cache = Path(args.out), Path(args.cache)
    items = load_items(Path(args.items))
    if args.limit:
        items = items[:args.limit]
    start, end = shard_range(len(items), args.shard, args.shards)
    mine = items[start:end]
    done_path = out / f"select_shard{args.shard}.jsonl"
    calls_path = out / f"gemma_calls_shard{args.shard}.jsonl"
    done = {r["id"] for r in read_jsonl(done_path)}
    todo = [it for it in mine if it["id"] not in done]
    log(f"select shard {args.shard}: {len(mine)} questions, {len(todo)} to do")
    if not todo:
        return
    t = time.time()
    retr = Retriever(cache)
    gemma = Gemma(args.gemma_model, args.gemma_compute_dtype)
    log(f"models loaded in {time.time() - t:.0f}s")
    for n, item in enumerate(todo, 1):
        t = time.time()
        rec = select_one(item, gemma, retr, calls_path)
        rec["seconds"] = round(time.time() - t, 1)
        append_jsonl(done_path, rec)
        log(f"{n}/{len(todo)} {item['id']}: queries={len(rec['queries'])} ({rec['query_status']}) "
            f"pool={len(rec['pool'])} kept={len(rec['evidence'])} judge={rec['judge_status']} {rec['seconds']}s")


# ── phase: merge + audit ─────────────────────────────────────────────────────
def pct(a: int, b: int) -> float:
    return round(100.0 * a / b, 1) if b else 0.0


def phase_merge(args) -> None:
    out, cache = Path(args.out), Path(args.cache)
    items = load_items(Path(args.items))
    if args.limit:
        items = items[:args.limit]
    corpus = load_corpus(cache)
    recs = [r for p in sorted(out.glob("select_shard*.jsonl")) for r in read_jsonl(p)]
    ids = [r["id"] for r in recs]
    expected = [it["id"] for it in items]
    missing = sorted(set(expected) - set(ids))
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    by_id = {r["id"]: r for r in recs}
    labels = {it["id"]: it for it in items}

    rows = []
    for qid in expected:
        if qid not in by_id:
            continue
        r, it = by_id[qid], labels[qid]
        gt_text = clean_option(it["options"][it["ground_truth"]])
        f_text = clean_option(it["options"][it["false_answer_key"]])
        cand = r["all_candidates"]
        pool = [p["doc"] for p in r["pool"]]
        bm25_any = {d for p in r["pool"] for d in [p["doc"]] if p.get("bm25_rank")}
        dense_any = {d for p in r["pool"] for d in [p["doc"]] if p.get("medcpt_rank")}
        kept_text = " ".join(e["content"] for e in r["evidence"])
        rows.append(dict(
            id=qid, status=r["status"], n_queries=len(r["queries"]), query_status=r["query_status"],
            judge_status=r["judge_status"], kept=len(r["evidence"]), seconds=r.get("seconds"),
            query_has_correct=any(mentions(q, gt_text) for q in r["queries"]),
            query_has_false=any(mentions(q, f_text) for q in r["queries"]),
            candidates_correct=any(mentions(corpus[d]["content"], gt_text) for d in cand),
            candidates_false=any(mentions(corpus[d]["content"], f_text) for d in cand),
            pool_correct=any(mentions(corpus[d]["content"], gt_text) for d in pool),
            pool_false=any(mentions(corpus[d]["content"], f_text) for d in pool),
            pool_bm25_correct=any(mentions(corpus[d]["content"], gt_text) for d in bm25_any),
            pool_medcpt_correct=any(mentions(corpus[d]["content"], gt_text) for d in dense_any),
            kept_correct=mentions(kept_text, gt_text) if kept_text else False,
            kept_false=mentions(kept_text, f_text) if kept_text else False,
            kept_tag_correct=sum(e["option"] == it["ground_truth"] for e in r["evidence"]),
            kept_tag_false=sum(e["option"] == it["false_answer_key"] for e in r["evidence"]),
            quote_rejections=sum(j["reject_reason"] == "quote_not_found" for j in r["judged"]),
        ))
    n = len(rows)
    count = lambda k: sum(1 for x in rows if x[k])
    audit = dict(
        at=now(), questions_expected=len(expected), questions_done=n, missing=missing, repeated=repeated,
        complete=not missing and not repeated,
        status_counts={s: sum(x["status"] == s for x in rows) for s in ("evidence", "none")},
        query_status={s: sum(x["query_status"] == s for x in rows) for s in sorted({x["query_status"] for x in rows})},
        judge_status={s: sum(x["judge_status"] == s for x in rows) for s in sorted({x["judge_status"] for x in rows})},
        mean_queries=round(sum(x["n_queries"] for x in rows) / n, 2) if n else 0,
        mean_kept=round(sum(x["kept"] for x in rows) / n, 2) if n else 0,
        mean_seconds=round(sum(x["seconds"] or 0 for x in rows) / n, 1) if n else 0,
        quote_rejections=sum(x["quote_rejections"] for x in rows),
        coverage_pct={
            "query mentions correct option": pct(count("query_has_correct"), n),
            "query mentions false option": pct(count("query_has_false"), n),
            "any candidate mentions correct": pct(count("candidates_correct"), n),
            "any candidate mentions false": pct(count("candidates_false"), n),
            "judge pool (top 10) mentions correct": pct(count("pool_correct"), n),
            "judge pool (top 10) mentions false": pct(count("pool_false"), n),
            "pool chunks found by BM25 mention correct": pct(count("pool_bm25_correct"), n),
            "pool chunks found by MedCPT mention correct": pct(count("pool_medcpt_correct"), n),
            "kept evidence mentions correct": pct(count("kept_correct"), n),
            "kept evidence mentions false": pct(count("kept_false"), n),
        },
        kept_option_tags=dict(correct=sum(x["kept_tag_correct"] for x in rows),
                              false=sum(x["kept_tag_false"] for x in rows),
                              total=sum(x["kept"] for x in rows)),
        note=("'mentions' = every word longer than 3 letters of the option text appears in the chunk "
              "(same rule as the web-search audit). Labels are used only here, never in prompts."),
    )
    write_json(out / "audit.json", audit)
    write_json(out / "audit_rows.json", rows)
    evidence_map = {qid: dict(status=by_id[qid]["status"], queries=by_id[qid]["queries"],
                              evidence=by_id[qid]["evidence"]) for qid in expected if qid in by_id}
    write_json(out / "evidence_map.json", evidence_map)
    log(json.dumps({k: audit[k] for k in ("questions_done", "complete", "status_counts", "mean_kept",
                                          "mean_seconds", "coverage_pct")}, indent=1))


# ── orchestration ────────────────────────────────────────────────────────────
def spawn(phase: str, args, gpus: int) -> None:
    procs = []
    for g in range(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1", TE_WORKER_TAG=f"gpu{g}")
        cmd = [sys.executable, os.path.abspath(__file__), "--phase", phase, "--shard", str(g), "--shards", str(gpus),
               "--out", args.out, "--cache", args.cache, "--items", args.items, "--limit", str(args.limit),
               "--gemma-model", args.gemma_model, "--gemma-compute-dtype", args.gemma_compute_dtype,
               "--embed-batch", str(args.embed_batch)]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"{phase} workers failed with exit codes {codes}")


def phase_all(args) -> None:
    import torch
    gpus = torch.cuda.device_count()
    if args.require_gpus and gpus < args.require_gpus:
        raise SystemExit(f"needs {args.require_gpus} GPUs, found {gpus}")
    out = Path(args.out)
    env = dict(at=now(), python=sys.version, gpus=[torch.cuda.get_device_name(i) for i in range(gpus)],
               settings=SETTINGS, models=MODELS | {"gemma": args.gemma_model},
               gemma_compute_dtype=args.gemma_compute_dtype, limit=args.limit)
    try:
        from importlib.metadata import version
        env["packages"] = {p: version(p) for p in ("torch", "transformers", "accelerate", "bitsandbytes", "bm25s")}
    except Exception:
        pass
    write_json(out / "environment.json", env)
    t = time.time()
    if not (Path(args.cache) / "bm25_index").exists():
        phase_prepare(args)
    log(f"prepare done ({time.time() - t:.0f}s); embedding on {gpus} GPU(s)")
    spawn("embed", args, gpus)
    log(f"embed done ({time.time() - t:.0f}s); selecting on {gpus} GPU(s)")
    spawn("select", args, gpus)
    phase_merge(args)
    write_json(out / "completed.json", dict(at=now(), total_seconds=round(time.time() - t), gpus=gpus))
    log(f"all done in {time.time() - t:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["prepare", "embed", "select", "merge", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--corpus-dir", default=None)
    ap.add_argument("--corpus-limit", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--require-gpus", type=int, default=0)
    ap.add_argument("--embed-batch", type=int, default=128)
    ap.add_argument("--gemma-model", default=MODELS["gemma"])
    ap.add_argument("--gemma-compute-dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.cache).mkdir(parents=True, exist_ok=True)
    {"prepare": phase_prepare, "embed": phase_embed, "select": phase_select,
     "merge": phase_merge, "all": phase_all}[args.phase](args)


if __name__ == "__main__":
    main()
