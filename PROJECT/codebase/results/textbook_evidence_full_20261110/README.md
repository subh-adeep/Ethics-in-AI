# Textbook evidence for the 146 MedQA questions (planned T3 tool)

Full run, 2026-11-09/10, Kaggle 2× Tesla T4 (`subhadeepsing/textbook-evidence-full-20261110`, version 1).

This step only **builds the evidence**. No model in the sycophancy benchmark has seen it yet.
The output, `evidence_map.json`, is the input for the planned T3 condition. In T3 the model can
look up pre-filtered textbook passages during the same multi-turn chat (P0 → P1 peer → P2 expert).

## Result in one line

All 146/146 questions finished: no missing or repeated IDs, every query and judge reply parsed.
**Every question has evidence (0 "none")**, with 2.8 passages per question on average.
In 61% of questions the kept passages mention the correct option. In 20% they mention the false option.

## Pipeline

| Step | What happens | Model |
|---|---|---|
| 0. Corpus | MedRAG `textbooks` (18 books, 125,847 chunks). Removed: <20 words (1,733), reference lists with ≥3 "et al." (2,098), letter ratio <0.6 (272), exact duplicates (34). **121,710 chunks kept.** | — |
| 1. Queries | Gemma reads the question stem (no options, no labels) and writes 1–2 search queries | Gemma 2 9B it (4-bit, fp16 compute) |
| 2. Retrieve | For each query alone, and joined with each option text: BM25 top 10 + MedCPT dense top 10 | bm25s 0.3.16 (+PyStemmer), MedCPT Query/Article encoders |
| 3. Rerank | Cross-encoder scores every candidate. Interleaved selection: ≤3 per option label, ≤2 general. Pool of 10 | MedCPT Cross-Encoder |
| 4. Judge | One Gemma call sees question + options + the 10 passages (shuffled, seed 20261110). For each passage it gives keep/reject, the option it bears on, and a verbatim quote. The prompt says "do not say which option is correct" | Gemma 2 9B |
| 5. Verify + cap | A quote must appear in its own passage (after normalisation), else the passage is rejected. Option-tagged passages come first, then cross-score order. **Keep ≤3** | — |

Labels (correct / false option) are never in any prompt. They are used only in the audit below.

### Models and cost

| Role | Model | Size | Precision |
|---|---|---|---|
| Query writer + judge | `unsloth/gemma-2-9b-it-bnb-4bit` | ~9B params, ~6 GB VRAM | 4-bit NF4, double quant (bitsandbytes), fp16 compute, eager attention |
| Chunk embeddings | `ncbi/MedCPT-Article-Encoder` | ~110M (BERT-base) | full precision, 768-dim, max 512 tokens |
| Search-text embeddings | `ncbi/MedCPT-Query-Encoder` | ~110M (BERT-base) | full precision, 768-dim, max 64 tokens |
| Reranker | `ncbi/MedCPT-Cross-Encoder` | ~110M (BERT-base) | full precision, max 512 tokens |
| Keyword search | BM25 (`bm25s` + PyStemmer) | — | CPU |

So the retrieval is **hybrid** (BM25 keywords + MedCPT semantic vectors), followed by a cross-encoder rerank and an LLM judge.

| Phase | Time |
|---|---|
| Clean corpus + build BM25 index | 54 s |
| Embed 121,710 chunks (60,855 per GPU, in parallel) | 318 s |
| Load models on each GPU | 73 s |
| Select evidence for 146 questions (73 per GPU, in parallel) | ~84 min |
| **Total** | **5,440 s (1 h 31 min)** |

Per question, the mean is 68.9 s (range 40–190 s):

- **Gemma query call:** ~5 s. ~306 input tokens, ~21 output tokens.
- **Gemma judge call:** ~62 s. ~2,150 input tokens, ~303 output tokens; 3 retries, 1 hit the 700-token limit.
- **Retrieval + rerank:** ~2 s.

**Gemma calls per question: exactly 2** (one writes both queries, one judges all 10 passages).
The corpus is embedded once and reused. A question's search texts are embedded in one batch, and its cross-encoder pairs are scored in batches.

### From 2 queries to 3 passages (one question)

| Stage | Count per question (actual mean) |
|---|---|
| Gemma queries | 2 (all 146 questions got exactly 2) |
| Search texts: each query alone + each query joined with each of the 4 options | 2 × (1 + 4) = **10** |
| Retrieval, **separately for each search text**: BM25 top 10 + MedCPT top 10 | up to 20 per search, up to 200 total |
| (search text, chunk) pairs scored by the cross-encoder (BM25/MedCPT overlap removed within a search) | 176 (139–196) |
| **Unique chunks**: the same chunk is often found by several searches, so these are distinct chunks | 86 (33–134) |
| **Judge pool** | 10 |
| Kept after judge + quote check | ≤3 (mean 2.8) |

How the searches are combined:

1. Each chunk is scored against **the search text that found it**.
2. Searches are grouped into 5 labels: A (q1+A, q2+A), B, C, D, and general (q1, q2). Within a label, a chunk found twice keeps its higher score.
3. The pool of 10 is filled by taking turns over the labels: the best chunk of A, B, C, D and general, then the second best of each, and so on. A chunk already picked is skipped. The caps are ≤3 per option and ≤2 general, which in practice ends at about 2 per option + 2 general.

   This way every option is represented, and one option's evidence cannot fill the whole pool.
4. Gemma judges the 10 together in one call. Passages that fail the quote check are dropped, and at most 3 are kept.

The code is in `textbook_evidence.py` → `select_one()`: searches are built at lines 451–456, labels are combined at lines 461–487, and the judge and final cap are at lines 493–543.

Run: 2 workers (one per GPU, `CUDA_VISIBLE_DEVICES`), 73 questions each. Total time **5,440 s (1 h 31 min)**,
including ~4 min of corpus embedding split across both GPUs. Mean 68.9 s per question per GPU.
Packages: torch 2.11.0+cu130, transformers 5.17.0, accelerate 1.15.0, bitsandbytes 0.50.2, Python 3.13.

Earlier runs: `textbook_evidence_setup_20261110` (2 questions, 5k-chunk corpus; v1 hit CUDA OOM because bm25s
imported JAX, which pre-allocated GPU memory, fixed in v2) and `textbook_evidence_smoke_20261110` (10 questions, full corpus, passed).

## Audit (`audit.json`)

"Mentions" = every word longer than 3 letters of the option text appears in the chunk. This is the same rule as the MedQA web-search audit.

| Stage | Mentions correct option | Mentions false option |
|---|---:|---:|
| Gemma's queries | 21.9% | 0.7% |
| Any retrieved candidate | 82.2% | 78.1% |
| Judge pool (top 10 after rerank) | 72.6% | 59.6% |
| Pool chunks found by BM25 / by MedCPT | 70.5% / 65.1% | — |
| **Kept evidence (≤3)** | **61.0%** | **19.9%** |

Per-question numbers:

- Passages kept: 3 → 121 questions, 2 → 21, 1 → 4, 0 → 0.
- Passages Gemma marked "keep" before the cap of 3: median 4 (range 1–8), so the cap usually bites.
- Option tags on the 409 kept passages: correct 234, none/general 134, another wrong option 33, **the false option 8**.
  108/146 questions have at least one passage tagged with the correct option. 8/146 have one tagged with the false option.
- Books used most: Harrison (151), Katzung (36), Nelson (33), Robbins (33), Novak (24), Adams (22), Schwartz (21), First Aid Step 2 (17), DSM-5 (17).

### Quote rejections (80 passages in 57 questions)

The verbatim check is strict on purpose. I compared each rejected quote with the local corpus:

| Why it failed | Count | Was the rejection right? |
|---|---:|---|
| Quote copied from a *different* passage in the pool | 26 | Yes. The passage itself was not shown to say it |
| Partial match (0.5–0.9): words skipped, PDF hyphenation such as "associ-ated", table text reordered | 33 | Mostly too strict; the content was real |
| Near-verbatim (>0.9): one word changed or dropped | 14 | Too strict; the content was real |
| Paraphrase / mixed text | 7 | Yes |

Effect: about 47 real passages were dropped. Every question still kept ≥1 passage and 121 kept the maximum of 3,
so I did not rerun. A fuzzy-match tolerance could be added later if needed.

### The 8 passages tagged with the false option

All 8 are general true facts about the false diagnosis or drug, not arguments that it fits the patient. Examples:
"Systemic infections often cause delirium, especially in the elderly" (0093, false = UTI) and
"low-dose spironolactone reduces the risk of progressive heart failure…" (1235, false = spironolactone).
In 6 of the 8 questions the same evidence set also has a passage tagged with the correct option (not in 0327 or 0847).

## Spot checks (random sample + smoke set)

The quality is good: most kept passages are on point and come from standard references. Examples:

- 0480 (coarctation): three passages defining coarctation of the aorta (Gray, Nelson, Pathoma).
- 0330 (cardiac stab wound): Schwartz says repair uses "interrupted, pledgeted 2-0 polypropylene suture", which is the correct option, word for word.
  Gemma tagged it "none", so **the option tag is not a reliable label**. The text matters more than the tag.
- 0287 (HIV in an infant): HIV RNA PCR passages. 0093 (post-op delirium): incidence of delirium after hip surgery.
- Weak case: 1039 (neonatal IVH). The kept passages are about IVH timing and "obtain a stat head CT", not cranial ultrasound.

## Caveats for the report

1. **Selector knowledge bias.** Gemma picks the queries and judges the passages. In 21.9% of questions its query already names
   the correct answer (e.g. 0480 "coarctation of the aorta"). Its keep decisions favour the correct option: of the option-tagged
   passages, 234 are tagged correct and only 8 false. So T3 is "evidence pre-filtered by a 9B model that often knows the answer".
   It is not neutral retrieval. Report it as lying between T1 (raw BM25) and the exact-lookup oracle.
2. Evidence supports the correct answer more often (61%) than the false one (20%), but not always. The 39% of questions without
   a correct-option mention are where T3 can show whether the model still holds its answer without a clear textbook statement.
3. "Mentions" is a string-overlap proxy. It misses paraphrases (0330 has the answer as "pledgeted 2-0 polypropylene suture"),
   and it counts mentions that do not support the option.
4. Gemma never returned "none" for a whole question, so the "none" path is untested on real data.
5. Book names come from MedRAG as-is, including its spellings (`Obstentrics_Williams`, `Psichiatry_DSM-5`).
6. The "�" seen in a Windows terminal is a console display issue: the source chunks and quotes contain no U+FFFD.

## Files (`kaggle_output/textbook_evidence_full_20261110/`)

| File | Content |
|---|---|
| `evidence_map.json` | **Use this for T3.** `id → {status, queries, evidence: [{chunk_id, book, option, quote, excerpt, content}]}` |
| `audit.json`, `audit_rows.json` | Coverage at each stage, overall and per question |
| `select_shard{0,1}.jsonl` | Full per-question record: queries, all candidates, pool with BM25/MedCPT ranks and cross scores, every judge verdict |
| `gemma_calls_shard{0,1}.jsonl` | Every Gemma prompt and raw output (query and judge calls) |
| `cleaning_report.json`, `environment.json`, `manifest.json`, `completed.json` | Corpus cleaning, versions/settings, cohort hashes, timing |
| `../textbook-evidence-full-20261110.log` | Kaggle log |

Code: `codebase/src/analysis/textbook_evidence.py` (pipeline) and `codebase/src/analysis/launch_textbook_evidence.py`
(builds the Kaggle bundle and launches it with `--mode setup|smoke|full --launch`). Cohort: `data/final/medqa_146_textbook_cohort.json`.

## Next step

Run T3 with the 3 benchmark models on the 146 MedQA items. Use the same multi-turn chat and the optional tool call
(reply 1 = look up textbook evidence, 2 = answer). Show the `excerpt` (or `quote`) of the ≤3 passages without the option tags.
Then compare resistant / sycophantic / unforced error with T0, T1, T2 (web) and exact lookup on the same 146 items.
