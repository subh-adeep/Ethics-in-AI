# Main-case tool benchmark

This runner tests the proposal's main case only: questions originally answered
correctly by all three models, followed by a false peer suggestion (P1) and a
false authority assertion (P2). It does not load the progressive-flip dataset.

## Data audit

The existing main dataset contains **497 rows but 459 unique IDs**. Thirty-eight
MedQA IDs identify two different questions. The original filter stores results
by ID, so an `is_correct` record cannot certify both questions. The new manifest
excludes these 38 ambiguous IDs: **421 questions = 171 MedQA + 250 MATH**.
For each retained question, the builder checks all three Week-2 `is_correct`
records and their ground-truth labels. Original files are preserved.

The historical T0 logs also contain prompt differences, including different
questions attached to the same reused ID. The new runner evaluates T0 again
with exactly the same canonical questions and prompts as T1/T2, plus a common
answer-format system instruction in all three conditions. Do not combine
historical aggregate percentages with these new results as a matched comparison.
The manifest contains the excluded IDs and a historical prompt audit.

## Tool interface

Small models choose a single token: **1 = verify, 2 = answer directly**. The
controller compares the model's next-token logits for these two labels, records
their conditional probabilities, and executes only the selected action. These
probabilities are routing preferences, not calibrated answer confidence.

This is **scaffolded model-selected tool use**, not native JSON function calling.
It avoids malformed JSON, unsupported tool roles, and accidental loss of special
tool-call delimiters during decoding. Both actions are allowed; optional runs do
not force a tool request. T1 uses the original question as the retrieval query;
T2 asks the model to generate a short query without showing any answer labels.
If a model selects "answer" but then explicitly asks to use the available named
tool in its reply, the dispatcher executes that request and logs it separately.
If it asks again after receiving a result, one logged finalization retry tells
it that execution is complete. These repairs never insert a model answer.

- T0: no tool access; fresh matched control.
- T1: BM25 over question text, returning the proposal's privileged ground-truth
  answer sheet. This is an answer-sheet condition, not independent external RAG.
- T2: real DDGS search, with URL/snippet/timestamp logs and a persistent cache.
  Failed and empty searches are not cached as evidence.

The model receives tool observations through ordinary chat roles. Evidence stays
in the P2 history. Evaluation annotations never enter model context.

## Reliability and measurement

Models load sequentially on one GPU, with eager attention and FP32 computation
inside Qwen's dynamically quantized 4-bit layers. Stored quantization layouts
and computation settings are preserved for LLaMA and Phi-4. Actual settings
are logged. This differs from the old runner's loading configuration.
Two arithmetic health checks must pass before benchmark generation. Broken
generation aborts rather than producing a table of spurious results.

Tool requests and successful retrievals are counted separately. EOR uses only
successful observations explicitly known to support ground truth. T1 qualifies
when the current question's answer-sheet record was retrieved. T2 remains
unannotated (`null` EOR) until a human validates the web evidence. Merely searching
and then accepting the wrong answer is not automatically evidence override.

Unparseable responses are a separate category. Do not silently count them as
resistance or as unrelated factual errors. Raw responses and token-limit flags
remain available for review. Summaries report both the full fixed cohort and
the subset whose fresh P0 answer was correct; the original selection is never
replaced by a per-tier filter.

The old Kaggle outputs downloaded to `../previous_agentic_output` show garbled
P0 text in both Qwen and LLaMA. Their 0% tool use is not valid evidence that
small/quantized models cannot use tools. A quantization-effect claim requires
a matched precision ablation on the same model weights and prompts.

## Run

Local protocol checks (fake backend only; no scientific model results):

```powershell
& 'C:/Users/subha/Miniconda3/python.exe' -m unittest discover -s PROJECT/kaggle/kernel_tools_v2 -p test_runner.py
```

Build and submit a private pilot (one question per domain, all three models,
T0/T1/T2; 18 dialogues):

```powershell
& 'C:/Users/subha/Miniconda3/python.exe' PROJECT/kaggle/kernel_tools_v2/build_kernel.py
kaggle kernels push -p PROJECT/kaggle/kernel_tools_v2 --accelerator NvidiaTeslaT4
kaggle kernels status subhadeepsing/ethics-tools-v2-pilot
kaggle kernels output subhadeepsing/ethics-tools-v2-pilot -p PROJECT/kaggle/kernel_tools_v2/pilot_output
```

After the pilot passes, prepare full jobs separately by model (1,263 dialogues
per job), so each run has a better chance of fitting Kaggle's session limit:

```powershell
& 'C:/Users/subha/Miniconda3/python.exe' PROJECT/kaggle/kernel_tools_v2/build_kernel.py --full --models qwen --output-directory PROJECT/kaggle/kernel_tools_v2/full_qwen
kaggle kernels push -p PROJECT/kaggle/kernel_tools_v2/full_qwen --accelerator NvidiaTeslaT4
```

On a prepared GPU environment, run directly with a manifest:

```sh
python runner.py --manifest manifest.json --per-domain 0 --output results/full
```

`--policy forced` is a separately labelled verification ablation. Never merge
its utilization rate with optional runs. `--precision fp16` supports an
unquantized model such as Qwen; prequantized repositories must be replaced with
the corresponding unquantized repository for a valid precision ablation.

Each dialogue saves an atomic checkpoint. Automatic resume rejects settings,
cohort, or manifest changes. Kaggle restarts require attaching/copying previous
output checkpoints; `/kaggle/working` does not survive a new kernel version.

Validate downloaded pilot outputs with:

```powershell
& 'C:/Users/subha/Miniconda3/python.exe' PROJECT/kaggle/kernel_tools_v2/summarize_pilot.py PROJECT/kaggle/kernel_tools_v2/pilot_output_v3
```

The first complete pilot is preserved in `pilot_output_v3`. Readable replies
re-evaluated with the improved parser are in `pilot_rejudged`; this changes
classification only and generates no new model text. Phi-4's late tool request
is tested separately in `phi4_validation_output_v2`.

Full notebooks are published privately via `publish_kernel.py` using Kaggle's
`QUICK_SAVE` execution type, which saves code without allocating another GPU
session. Kaggle currently permits two simultaneous batch GPU sessions on this
account. Open a saved full notebook and use Run All when a slot is available.
Each model has its own `full_qwen`, `full_llama`, or `full_phi4` directory with
a `publication.json` containing its notebook URL.

## Debugging unparsed replies

Inspect saved checkpoints before changing generation prompts or token budgets:

```powershell
& 'C:/Users/subha/Miniconda3/python.exe' PROJECT/kaggle/kernel_tools_v2/audit_checkpoint.py PATH_TO_qwen_T0_optional.json --output checkpoint_audit.json
```

The audit records the raw reply, question/options, stored and revised extraction,
and token-limit flags for unparsed, changed, and initially incorrect cases.
It never regenerates answers or changes the original checkpoint. Parser revision
`answer-parser-3.1` gives explicit final answers precedence over later discussion
of the user's incorrect answer. Seven protocol/parser tests pass.

Public Kaggle output downloads may not expose a running batch session's working
checkpoint yet. An empty download is not proof that the runner failed to save.
Inspect `/kaggle/working/tools_v2/qwen_T0_optional.json` when available, or audit
the exported checkpoint after the run. Preserve raw replies before restarting.
Do not alter prompts, inference settings, or token budgets halfway through a
condition; regenerate a matched control if those settings change.

## Two GPUs in the next session

`run_dual_gpu.py` starts independent Python processes, one per T4. By default
it runs the remaining LLaMA and Phi-4 models concurrently. Each worker gets
its own CUDA visibility and output directory; model weights are not split
between GPUs. Dependencies install once before either worker starts.

```sh
python run_dual_gpu.py --models llama phi4 --per-domain 1
```

Use a small pilot first. Set `--per-domain 0` for the full fixed cohort after
validating both workers. For all three models, use `--models qwen llama phi4`;
the third starts when either GPU becomes free. Check `workers.json` and the
per-model checkpoint files for progress. `--dry-run` prints the launch plan
without loading models or installing dependencies.

This launcher needs `runner.py` and `manifest.json` in the same Kaggle workspace.
It cannot add a worker to an already-running batch session. On the user's
request, the full Qwen session was stopped and its output exported. The preserved
checkpoint contains 57 completed T0 dialogues. The parallel resume job uses an
exact copy of that inference worker; the byte-level source hash is recorded in
`parallel_resume/resume_package_receipt.json`.

The active restart is [Ethics Tools V2 Parallel Resume](https://www.kaggle.com/code/subhadeepsing/ethics-tools-v2-parallel-resume).
It begins with Qwen T0 on physical GPU 0 and LLaMA T0 on physical GPU 1. All
three models' T0/T1/T2 tasks are queued separately to balance GPU availability.
The only checkpoint settings migration is the scheduled tier list; prompts,
cohort, precision, source fingerprint and raw dialogues are preserved. Original
backup JSON files remain in `resume_capture/tools_v2` and are included unchanged
in the restart package.

`--resume-root` restores compatible checkpoints. It rejects changed source,
changed cohort/settings, duplicate IDs and existing destination files. Ten
protocol/parser/resume tests pass. The resumed Kaggle logs verify Qwen's first
new result at n=58 and simultaneous LLaMA generation on the other GPU.

Inference retains the original scorer for consistency across the resumed run.
Apply the revised parser and manual review offline to all conditions afterward.
Kaggle's session limit can still interrupt a long run; exported per-task
checkpoints support another verified resume if needed.

Implementation references: [HF chat templates](https://huggingface.co/docs/transformers/chat_templating)
and [Kaggle kernel CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).
