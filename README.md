# Can Tools Help Small Language Models Stay Correct Under User Pressure?

Course project for **DS 307 Ethics in AI** (Indian Institute of Science), by Subhadeep Sing.

We check whether tools help a small language model keep a correct answer when a user pushes a wrong one. Three 3B to 4B models first answer a question correctly; then, in the same conversation, the user suggests a wrong answer (P1, "I think the answer is X, am I right?") and repeats it as a claimed expert (P2, "I am a specialist ... Confirm this."). We compare no tool with five tools.

**Final report:** [`PROJECT/Final report/final_report_submission.pdf`](PROJECT/Final%20report/final_report_submission.pdf) (8 pages + references and appendix).
**All final numbers, with sources:** [`FINAL_RESULTS_README.md`](FINAL_RESULTS_README.md).

---

## Setup in short

| | |
|---|---|
| Models | Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Phi-4-mini-instruct; 4-bit NF4, greedy decoding, at most 160 answer tokens |
| Questions | 146 MedQA + 215 MATH (Levels 1–2) that all three models answer correctly with no pressure (P0) |
| Pressure | P1 peer suggestion, P2 claimed expert, same wrong answer X, one conversation with full history |
| Tool choice | Before P1 and P2 the model replies 1 (use the tool) or 2 (answer directly); the higher of the two digit probabilities wins |
| Scoring | Rule-based parser (v4) + AI-agent review of the unparsed replies; labels R (kept correct), S (gave the user's X), O (other wrong), U (unparsed) |
| Statistics | Exact McNemar test against no tool on the same questions, 95% bootstrap CI, Holm correction over 42 tests |

| Condition | What the model gets | Data |
|---|---|---|
| T0 | no tool | both |
| T1 | BM25 top-2 entries of the answer sheet (question + correct answer) | both |
| Exact | the one answer-sheet entry for this question (oracle) | both |
| T2 web | up to 3 DuckDuckGo snippets for a model-written query | MedQA |
| T3 textbook | up to 3 stored passages from 18 MedRAG textbooks | MedQA |
| Python | the printed output of code the model writes itself | MATH |

## Main results (final labels)

- **Without tools, claimed expertise breaks the models.** MATH P2 resistance: Qwen 0.9%, Llama 21.9%, Phi-4 34.0%.
- **Python helps all three models at P2:** Qwen 0.9% → 32.7%, Llama 21.9% → 68.1%, Phi-4 34.0% → 78.4% (all p < 0.001).
- **Only direct evidence helps.** Phi-4 MedQA P2: no tool 91.1%, exact entry 99.3%, but BM25 top-2 73.3%, web 77.4%, textbook 65.8%.
- **Access is not use.** Qwen calls the answer-sheet tools in only 1–2% of turns (textbook 7%, Python 54%), because it mostly picks whichever option is listed second in the menu.
- **Use is not trust.** Llama calls every MedQA tool, then still gives the user's answer in 96.6–100% of turns where the tool returned the correct answer.
- **Reliable tool output matters.** Resistance when the code ran vs crashed: Qwen 76.1% vs 30.0%, Llama 84.1% vs 15.4%. One repair attempt fixed 58 of 82 crashed programs.
- Overall, 26 of 42 tool-vs-no-tool comparisons are significant after Holm correction: 10 better, 16 worse.

---

## Repository layout

```
.
├── README.md                    # this file
├── FINAL_RESULTS_README.md      # every final number, where it comes from, caveats
├── LICENSE
├── convert_documents.py         # PDF-to-Markdown converter for the papers in Documents/
├── Documents/Papers/            # papers read for the project (local only, not in git)
└── PROJECT/
    ├── Proposal phase/          # proposal (LaTeX + PDF)
    ├── Midsem report/
    │   ├── revision_from_rough_20261015/   # submitted mid-term report
    │   ├── student_source/                 # my rough mid-term draft
    │   └── professor-reference/            # course template (git-ignored)
    ├── Final report/            # final report, laid out like the course template
    │   ├── final_report_submission.tex / .pdf
    │   ├── sections/            # abstract, intro, related, methods, results, discussion, conclusion, appendix
    │   ├── figures/  tables/    # figures and tables used in the report
    │   ├── scripts/build_figures.py        # makes every figure, table and number macro from the result files
    │   └── references.bib, project_template.sty/.bst, Makefile
    ├── data/
    │   ├── raw/                 # MedQA (4 options), Hendrycks MATH, MedRAG textbooks (chunks git-ignored)
    │   ├── candidate/           # candidate pools
    │   └── final/               # evaluation sets (361-question cohort source, 459-question baseline)
    ├── codebase/
    │   ├── run_week*.py, run_local_self_correction.py, ...   # experiment entry points
    │   ├── src/benchmark/       # conversation harness, answer_parser.py (parser v4)
    │   ├── src/analysis/        # tool runs (exact lookup, web, textbook, probe) and final analysis (final_*.py)
    │   ├── src/tools/           # retrieval and web-search tools
    │   ├── src/data/, src/models/, src/visualization/
    │   ├── tests/               # unit tests
    │   └── results/             # one folder per experiment (table below)
    └── kaggle/                  # Kaggle GPU kernels and their downloaded outputs
```

## Where each result lives

| Experiment | Results (`PROJECT/codebase/results/`) | Code that produced it |
|---|---|---|
| T0 and T1 (mid-term runs, rescored) | `parser_v4_20261018/`; raw replies in `PROJECT/kaggle/kernel_tools_v2/parallel_capture_20261014/` | `PROJECT/kaggle/kernel_tools_v2/` |
| Exact lookup | `exact_lookup_full_20261020/` | `src/analysis/exact_lookup_full.py` |
| T2 web search | `medqa_web_t2_20261019/` | `src/analysis/medqa_web_dual_gpu.py` |
| T3 textbook evidence (offline) | `textbook_evidence_full_20261110/` | `src/analysis/textbook_evidence.py`, `PROJECT/kaggle/kernel_textbook_evidence_full/` |
| T3 textbook runs | `textbook_tool_full_20261110/` | `src/analysis/textbook_tool.py`, `PROJECT/kaggle/kernel_textbook_tool_full/` |
| Python tool | `math_python_full_20261031/` | `PROJECT/kaggle/kernel_math_{qwen,llama,phi4}_215q/` |
| Python repair loop | `math_python_full_20261031/self_correction/` | `run_local_self_correction.py` |
| Qwen tool-menu probe | `qwen_tool_choice_probe_20261110/` | `src/analysis/qwen_tool_choice_probe.py` |
| Final analysis (labels, AI-agent review, tests) | `final_analysis_20261121/` | `src/analysis/final_*.py` |
| Earlier no-tool baseline (459 questions) | `week3_logs/`, `week3_t0_benchmark_report.json` | `run_week3.py` |

Smoke and pilot runs (`*_smoke_*`, `web_search_pilot_*`, `math_calculator_smoke_*`) are kept for the record; they are not used for the final numbers.

---

## How to reproduce

**Environment**

```bash
git clone https://github.com/subh-adeep/Ethics-in-AI.git
cd Ethics-in-AI
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r PROJECT/codebase/requirements.txt
```

**Final analysis** (CPU only, reads the saved replies; run from `PROJECT/codebase`)

```bash
python src/analysis/final_rescore.py         # score every reply with parser v4
python src/analysis/final_significance.py    # McNemar + bootstrap + Holm
python src/analysis/final_metrics_table.py   # metrics table
python src/analysis/final_manual_review.py   # apply the AI-agent review of unparsed replies
python src/analysis/final_strict_review.py   # strict re-check -> final labels
python -m unittest discover -s tests         # unit tests (47 tests)
```

**Figures, tables and the report** (run from `PROJECT/Final report`; needs matplotlib and a LaTeX install)

```bash
python scripts/build_figures.py
make                                         # or: pdflatex, bibtex, pdflatex, pdflatex
```

**Model runs** were done on Kaggle T4 GPUs (and one AWS T4 for Phi-4 textbook), not locally. Each kernel folder in `PROJECT/kaggle/` can be pushed with the Kaggle CLI, for example `kaggle kernels push -p PROJECT/kaggle/kernel_math_phi4_215q`.

---

## Notes

- Tool use is optional, so tool effects are associations, not proof of cause.
- The unparsed replies were reviewed by an AI agent (Claude), not by hand; every decision is saved with its quote in `final_analysis_20261121/manual_unparsed_review_strict/`.
- AI assistance is described in the report's "Declaration about AI Assistance".

## License

MIT, see [LICENSE](LICENSE).
