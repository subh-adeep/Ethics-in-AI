# Why Qwen rarely uses the tool — probe (10 November 2026)

## Question

In every optional-tool run with the 1/2 router, Qwen 2.5 3B almost never picks the tool. LLaMA and Phi-4 almost always do.
Is this a model limitation, a bug, or the prompt format?

## What was tested

The probe replays the exact 20 recorded decision prompts from the T3 smoke run (10 questions × P1/P2), on Kaggle T4,
with the same model revision and 4-bit loading. Phi-4 runs the same prompts as a control. One thing changes at a time in the router message.
Script: `codebase/src/analysis/qwen_tool_choice_probe.py`.
Raw rows: `kaggle_output/probe/{qwen,phi4}_probe.jsonl`. Each row has P(use), the probability on tokens 1 and 2, the top-5 next tokens, and a free generation.

## Result

| Router variant | Qwen mean P(use) | Qwen picks tool | Phi-4 mean P(use) | Phi-4 picks tool |
|---|---:|---:|---:|---:|
| Original (1 = use tool, 2 = answer) | 0.13 | 2/20 | 0.55 | 16/20 |
| **Order swapped (1 = answer, 2 = use tool)** | **1.00** | **20/20** | 0.81 | 20/20 |
| Tool name `search_web` | 0.36 | 6/20 | 0.61 | 16/20 |
| Tool name `retrieve_answer_sheet` | 0.05 | 1/20 | 0.64 | 16/20 |
| Tool name `lookup_tool` (neutral) | 0.06 | 1/20 | 0.69 | 19/20 |
| One-line description of what the tool returns | 0.37 | 7/20 | 0.73 | 17/20 |
| No "I will choose my next action." prefill | 0.05 | 1/20 | 0.73 | 20/20 |

Sanity checks:
- The original variant reproduces the recorded smoke-run probabilities exactly (max difference 0.000).
- Almost all probability is on the tokens 1 and 2: 1.00 for Qwen, 0.97–0.99 for Phi-4.
- Free generation gives the same digit, so this is what Qwen really answers, not a scoring artifact.

## Interpretation

1. **Qwen's choice is mostly a label/position bias toward "2".**
   With the original wording it says "2" (answer) 87% of the time. When "2" means "use the tool", it says "2" 100% of the time.
   So Qwen is not refusing the tool; it mostly picks whichever option is offered second.
2. **The wording changes it somewhat.** A web-search name or a one-line description raises P(use) to about 0.36,
   while "answer sheet" or a vague name lowers it. This matches the earlier runs:
   - T2 web: Qwen used the tool in 33% (P1) and 56% (P2) of turns;
   - exact lookup: 0–3%;
   - T3 textbook: 10%.
3. **Phi-4 is close to a coin flip but leans toward the tool.** Original 0.55, swapped 0.81. It also has a smaller "2" bias,
   but its choice depends more on meaning than Qwen's does.
4. **Qwen can use tools.** In the Python-tool run there was no 1/2 menu (the model writes a code block itself),
   and Qwen ran code in 110/215 (P1) and 124/215 (P2) turns.

**Conclusion:** the low tool use is mainly an artifact of the 1/2 routing format interacting with Qwen's strong preference for answering "2".
It is not a missing capability or a bug in the harness. In the optional-tool conditions (T1-optional, T2, exact, T3),
Qwen's results therefore mostly reflect *not getting* the evidence. They do not show how it reacts to the evidence.

## Implications for reporting and the full T3 run

- Report Qwen's tool-use rate next to its outcomes, and add a note about this bias.
- To measure the effect of the evidence itself, run a **forced-tool** condition: the evidence is always given, with no 1/2 choice.
  The frozen runner already supports this as `policy="forced"`.
  Keeping the optional condition unchanged keeps it comparable with T1/T2/exact.
- Another option is to randomise or counterbalance the 1/2 order per question. That would make T3 differ from the earlier conditions.
