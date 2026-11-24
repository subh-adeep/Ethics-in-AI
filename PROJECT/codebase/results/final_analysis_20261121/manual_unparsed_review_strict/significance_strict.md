# Significance tests — strict re-check of recovered answers

Same method as `significance/significance.md` (paired, exact McNemar, Holm over 42 tests, bootstrap 95% CI, 5,000 resamples, seed 20261014), on the strict labels. Verdicts from v4 and from the first manual review are shown.

## MedQA

| Model | Tool | Turn | n | T0 R % | Tool R % | Diff pp [95% CI] | Improved / worsened | p | Holm p | Verdict (strict) | Verdict (manual) | Verdict (v4) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen | T1 | P1 | 146 | 92.5 | 92.5 | +0.0 [+0.0, +0.0] | 0 / 0 | 1.000 | 1.000 | no clear change | no clear change | no clear change |
| Qwen | T1 | P2 | 146 | 24.7 | 24.7 | +0.0 [+0.0, +0.0] | 0 / 0 | 1.000 | 1.000 | no clear change | no clear change | no clear change |
| Qwen | T2 web | P1 | 146 | 92.5 | 72.6 | -19.9 [-27.4, -12.3] | 2 / 31 | <0.001 | <0.001 | worse | worse | worse |
| Qwen | T2 web | P2 | 146 | 24.7 | 13.0 | -11.6 [-18.5, -4.8] | 7 / 24 | 0.003 | 0.053 | worse (not after Holm) | worse | worse (not after Holm) |
| Qwen | Exact | P1 | 146 | 92.5 | 93.2 | +0.7 [-2.1, +4.1] | 3 / 2 | 1.000 | 1.000 | no clear change | no clear change | no clear change |
| Qwen | Exact | P2 | 146 | 24.7 | 26.0 | +1.4 [-4.8, +8.2] | 13 / 11 | 0.839 | 1.000 | no clear change | no clear change | no clear change |
| Qwen | T3 textbook | P1 | 146 | 92.5 | 88.4 | -4.1 [-8.9, +0.7] | 3 / 9 | 0.146 | 1.000 | no clear change | no clear change | no clear change |
| Qwen | T3 textbook | P2 | 146 | 24.7 | 24.7 | +0.0 [-6.8, +6.2] | 12 / 12 | 1.000 | 1.000 | no clear change | no clear change | no clear change |
| LLaMA | T1 | P1 | 146 | 50.0 | 0.7 | -49.3 [-57.5, -41.1] | 0 / 72 | <0.001 | <0.001 | worse | worse | worse |
| LLaMA | T1 | P2 | 146 | 49.3 | 1.4 | -47.9 [-56.2, -39.7] | 1 / 71 | <0.001 | <0.001 | worse | worse | worse |
| LLaMA | T2 web | P1 | 146 | 50.0 | 2.1 | -47.9 [-56.2, -39.7] | 0 / 70 | <0.001 | <0.001 | worse | worse | worse |
| LLaMA | T2 web | P2 | 146 | 49.3 | 0.0 | -49.3 [-57.5, -41.1] | 0 / 72 | <0.001 | <0.001 | worse | worse | worse |
| LLaMA | Exact | P1 | 146 | 50.0 | 0.0 | -50.0 [-58.2, -41.8] | 0 / 73 | <0.001 | <0.001 | worse | worse | worse |
| LLaMA | Exact | P2 | 146 | 49.3 | 0.7 | -48.6 [-56.8, -40.4] | 1 / 72 | <0.001 | <0.001 | worse | worse | worse |
| LLaMA | T3 textbook | P1 | 146 | 50.0 | 2.7 | -47.3 [-55.5, -39.0] | 0 / 69 | <0.001 | <0.001 | worse | worse | worse |
| LLaMA | T3 textbook | P2 | 146 | 49.3 | 1.4 | -47.9 [-56.2, -39.7] | 1 / 71 | <0.001 | <0.001 | worse | worse | worse |
| Phi-4 | T1 | P1 | 146 | 87.0 | 96.6 | +9.6 [+3.4, +15.8] | 18 / 4 | 0.004 | 0.059 | better (not after Holm) | better (not after Holm) | better (not after Holm) |
| Phi-4 | T1 | P2 | 146 | 91.1 | 73.3 | -17.8 [-26.0, -9.6] | 8 / 34 | <0.001 | 0.001 | worse | worse | worse |
| Phi-4 | T2 web | P1 | 146 | 87.0 | 90.4 | +3.4 [-2.1, +9.6] | 13 / 8 | 0.383 | 1.000 | no clear change | no clear change | no clear change |
| Phi-4 | T2 web | P2 | 146 | 91.1 | 77.4 | -13.7 [-20.5, -6.8] | 6 / 26 | <0.001 | 0.010 | worse | worse | worse |
| Phi-4 | Exact | P1 | 146 | 87.0 | 100.0 | +13.0 [+7.5, +18.5] | 19 / 0 | <0.001 | <0.001 | better | better | better |
| Phi-4 | Exact | P2 | 146 | 91.1 | 99.3 | +8.2 [+3.4, +13.0] | 13 / 1 | 0.002 | 0.031 | better | better | better |
| Phi-4 | T3 textbook | P1 | 146 | 87.0 | 74.7 | -12.3 [-19.9, -4.8] | 9 / 27 | 0.004 | 0.059 | worse (not after Holm) | worse (not after Holm) | worse (not after Holm) |
| Phi-4 | T3 textbook | P2 | 146 | 91.1 | 65.8 | -25.3 [-33.6, -17.1] | 4 / 41 | <0.001 | <0.001 | worse | worse | worse |

## MATH

| Model | Tool | Turn | n | T0 R % | Tool R % | Diff pp [95% CI] | Improved / worsened | p | Holm p | Verdict (strict) | Verdict (manual) | Verdict (v4) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen | T1 | P1 | 215 | 98.1 | 98.1 | +0.0 [+0.0, +0.0] | 0 / 0 | 1.000 | 1.000 | no clear change | no clear change | no clear change |
| Qwen | T1 | P2 | 215 | 0.9 | 1.4 | +0.5 [+0.0, +1.4] | 1 / 0 | 1.000 | 1.000 | no clear change | no clear change | no clear change |
| Qwen | Exact | P1 | 215 | 98.1 | 98.1 | +0.0 [+0.0, +0.0] | 0 / 0 | 1.000 | 1.000 | no clear change | no clear change | no clear change |
| Qwen | Exact | P2 | 215 | 0.9 | 1.4 | +0.5 [+0.0, +1.4] | 1 / 0 | 1.000 | 1.000 | no clear change | no clear change | no clear change |
| Qwen | Python | P1 | 214 | 98.1 | 92.5 | -5.6 [-9.3, -2.3] | 2 / 14 | 0.004 | 0.059 | worse (not after Holm) | worse | no clear change |
| Qwen | Python | P2 | 214 | 0.9 | 32.7 | +31.8 [+25.7, +38.3] | 68 / 0 | <0.001 | <0.001 | better | better | better |
| LLaMA | T1 | P1 | 213 | 89.7 | 33.3 | -56.3 [-62.9, -49.3] | 2 / 122 | <0.001 | <0.001 | worse | worse | worse |
| LLaMA | T1 | P2 | 213 | 22.1 | 1.4 | -20.7 [-26.3, -15.5] | 1 / 45 | <0.001 | <0.001 | worse | worse | worse |
| LLaMA | Exact | P1 | 215 | 89.8 | 60.5 | -29.3 [-36.7, -21.9] | 11 / 74 | <0.001 | <0.001 | worse | worse | worse |
| LLaMA | Exact | P2 | 215 | 21.9 | 24.7 | +2.8 [-5.1, +10.7] | 41 / 35 | 0.567 | 1.000 | no clear change | no clear change | no clear change |
| LLaMA | Python | P1 | 210 | 89.5 | 74.3 | -15.2 [-22.4, -7.6] | 16 / 48 | <0.001 | 0.001 | worse | worse | worse (not after Holm) |
| LLaMA | Python | P2 | 210 | 22.4 | 68.1 | +45.7 [+37.6, +53.3] | 107 / 11 | <0.001 | <0.001 | better | better | better |
| Phi-4 | T1 | P1 | 215 | 41.4 | 97.7 | +56.3 [+49.8, +62.8] | 123 / 2 | <0.001 | <0.001 | better | better | better |
| Phi-4 | T1 | P2 | 215 | 34.0 | 74.9 | +40.9 [+32.6, +48.8] | 105 / 17 | <0.001 | <0.001 | better | better | better |
| Phi-4 | Exact | P1 | 214 | 41.6 | 99.5 | +57.9 [+51.4, +64.5] | 124 / 0 | <0.001 | <0.001 | better | better | better |
| Phi-4 | Exact | P2 | 214 | 34.1 | 93.0 | +58.9 [+51.9, +65.9] | 129 / 3 | <0.001 | <0.001 | better | better | better |
| Phi-4 | Python | P1 | 208 | 40.9 | 81.7 | +40.9 [+32.2, +48.6] | 97 / 12 | <0.001 | <0.001 | better | better | better |
| Phi-4 | Python | P2 | 208 | 33.7 | 78.4 | +44.7 [+36.5, +52.4] | 102 / 9 | <0.001 | <0.001 | better | better | better |
