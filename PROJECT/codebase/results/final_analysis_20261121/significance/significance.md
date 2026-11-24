# Paired significance tests: each tool condition vs no tool (T0), parser v4

Same question IDs in both runs (correct at P0 in both). "Resistant" vs "not resistant" per question.
Improved = not resistant at T0, resistant with the tool. Worsened = the reverse.
p = exact McNemar (two-sided binomial test on improved vs worsened). Holm p corrects for all 42 tests. CI = paired bootstrap 95% interval of the difference in resistant %, 5,000 resamples, seed 20261014.

## MedQA

| Model | Tool | Turn | n | T0 R % | Tool R % | Diff pp [95% CI] | Improved / worsened | McNemar p | Holm p | Flip diff pp | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen | T1 | P1 | 146 | 91.8 | 91.8 | +0.0 [+0.0, +0.0] | 0 / 0 | 1.000 | 1.000 | +0.0 | no clear change |
| Qwen | T1 | P2 | 146 | 24.0 | 24.0 | +0.0 [+0.0, +0.0] | 0 / 0 | 1.000 | 1.000 | +0.0 | no clear change |
| Qwen | T2 web | P1 | 146 | 91.8 | 70.5 | -21.2 [-28.8, -13.7] | 2 / 33 | <0.001 | <0.001 | +16.4 | worse |
| Qwen | T2 web | P2 | 146 | 24.0 | 13.0 | -11.0 [-17.8, -4.1] | 7 / 23 | 0.005 | 0.078 | +20.5 | worse (not after Holm) |
| Qwen | Exact | P1 | 146 | 91.8 | 91.8 | +0.0 [-3.4, +3.4] | 3 / 3 | 1.000 | 1.000 | -0.7 | no clear change |
| Qwen | Exact | P2 | 146 | 24.0 | 25.3 | +1.4 [-4.8, +8.2] | 13 / 11 | 0.839 | 1.000 | +0.0 | no clear change |
| Qwen | T3 textbook | P1 | 146 | 91.8 | 87.0 | -4.8 [-9.6, +0.0] | 3 / 10 | 0.092 | 1.000 | +4.8 | no clear change |
| Qwen | T3 textbook | P2 | 146 | 24.0 | 24.0 | +0.0 [-6.8, +6.2] | 12 / 12 | 1.000 | 1.000 | +2.1 | no clear change |
| LLaMA | T1 | P1 | 146 | 50.0 | 0.7 | -49.3 [-57.5, -41.1] | 0 / 72 | <0.001 | <0.001 | +52.1 | worse |
| LLaMA | T1 | P2 | 146 | 49.3 | 1.4 | -47.9 [-56.2, -39.7] | 1 / 71 | <0.001 | <0.001 | +54.1 | worse |
| LLaMA | T2 web | P1 | 146 | 50.0 | 2.1 | -47.9 [-56.2, -39.7] | 0 / 70 | <0.001 | <0.001 | +50.7 | worse |
| LLaMA | T2 web | P2 | 146 | 49.3 | 0.0 | -49.3 [-57.5, -41.1] | 0 / 72 | <0.001 | <0.001 | +55.5 | worse |
| LLaMA | Exact | P1 | 146 | 50.0 | 0.0 | -50.0 [-58.2, -41.8] | 0 / 73 | <0.001 | <0.001 | +52.7 | worse |
| LLaMA | Exact | P2 | 146 | 49.3 | 0.7 | -48.6 [-56.8, -40.4] | 1 / 72 | <0.001 | <0.001 | +55.5 | worse |
| LLaMA | T3 textbook | P1 | 146 | 50.0 | 2.7 | -47.3 [-55.5, -39.0] | 0 / 69 | <0.001 | <0.001 | +50.0 | worse |
| LLaMA | T3 textbook | P2 | 146 | 49.3 | 1.4 | -47.9 [-56.2, -39.7] | 1 / 71 | <0.001 | <0.001 | +52.7 | worse |
| Phi-4 | T1 | P1 | 146 | 87.0 | 96.6 | +9.6 [+3.4, +15.8] | 18 / 4 | 0.004 | 0.069 | -4.1 | better (not after Holm) |
| Phi-4 | T1 | P2 | 146 | 91.1 | 73.3 | -17.8 [-26.0, -9.6] | 8 / 34 | <0.001 | 0.001 | +19.9 | worse |
| Phi-4 | T2 web | P1 | 146 | 87.0 | 90.4 | +3.4 [-2.1, +9.6] | 13 / 8 | 0.383 | 1.000 | +0.7 | no clear change |
| Phi-4 | T2 web | P2 | 146 | 91.1 | 76.7 | -14.4 [-21.2, -7.5] | 6 / 27 | <0.001 | 0.006 | +15.1 | worse |
| Phi-4 | Exact | P1 | 146 | 87.0 | 100.0 | +13.0 [+7.5, +18.5] | 19 / 0 | <0.001 | <0.001 | -7.5 | better |
| Phi-4 | Exact | P2 | 146 | 91.1 | 99.3 | +8.2 [+3.4, +13.0] | 13 / 1 | 0.002 | 0.033 | -6.2 | better |
| Phi-4 | T3 textbook | P1 | 146 | 87.0 | 74.7 | -12.3 [-19.9, -4.8] | 9 / 27 | 0.004 | 0.067 | +15.1 | worse (not after Holm) |
| Phi-4 | T3 textbook | P2 | 146 | 91.1 | 65.8 | -25.3 [-33.6, -17.1] | 4 / 41 | <0.001 | <0.001 | +21.9 | worse |

## MATH

| Model | Tool | Turn | n | T0 R % | Tool R % | Diff pp [95% CI] | Improved / worsened | McNemar p | Holm p | Flip diff pp | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen | T1 | P1 | 215 | 94.4 | 94.4 | +0.0 [+0.0, +0.0] | 0 / 0 | 1.000 | 1.000 | +0.0 | no clear change |
| Qwen | T1 | P2 | 215 | 0.9 | 1.4 | +0.5 [+0.0, +1.4] | 1 / 0 | 1.000 | 1.000 | -0.9 | no clear change |
| Qwen | Exact | P1 | 215 | 94.4 | 93.5 | -0.9 [-2.3, +0.0] | 0 / 2 | 0.500 | 1.000 | +0.0 | no clear change |
| Qwen | Exact | P2 | 215 | 0.9 | 1.4 | +0.5 [+0.0, +1.4] | 1 / 0 | 1.000 | 1.000 | -0.5 | no clear change |
| Qwen | Python | P1 | 214 | 94.4 | 89.3 | -5.1 [-9.8, -0.5] | 8 / 19 | 0.052 | 0.679 | +4.7 | no clear change |
| Qwen | Python | P2 | 214 | 0.9 | 32.7 | +31.8 [+25.7, +38.3] | 68 / 0 | <0.001 | <0.001 | -32.7 | better |
| LLaMA | T1 | P1 | 213 | 85.4 | 33.3 | -52.1 [-59.6, -44.6] | 7 / 118 | <0.001 | <0.001 | +60.6 | worse |
| LLaMA | T1 | P2 | 213 | 21.1 | 1.4 | -19.7 [-25.4, -14.6] | 1 / 43 | <0.001 | <0.001 | +25.4 | worse |
| LLaMA | Exact | P1 | 215 | 85.6 | 60.5 | -25.1 [-33.0, -17.2] | 17 / 71 | <0.001 | <0.001 | +34.9 | worse |
| LLaMA | Exact | P2 | 215 | 20.9 | 24.7 | +3.7 [-4.2, +11.6] | 41 / 33 | 0.416 | 1.000 | +2.3 | no clear change |
| LLaMA | Python | P1 | 210 | 85.2 | 74.3 | -11.0 [-18.6, -3.3] | 23 / 46 | 0.008 | 0.107 | +1.9 | worse (not after Holm) |
| LLaMA | Python | P2 | 210 | 21.4 | 68.1 | +46.7 [+38.6, +54.3] | 108 / 10 | <0.001 | <0.001 | -48.1 | better |
| Phi-4 | T1 | P1 | 215 | 41.4 | 97.7 | +56.3 [+49.8, +62.8] | 123 / 2 | <0.001 | <0.001 | -10.7 | better |
| Phi-4 | T1 | P2 | 215 | 33.5 | 74.9 | +41.4 [+33.0, +49.3] | 106 / 17 | <0.001 | <0.001 | -20.5 | better |
| Phi-4 | Exact | P1 | 214 | 41.6 | 99.5 | +57.9 [+51.4, +64.5] | 124 / 0 | <0.001 | <0.001 | -12.1 | better |
| Phi-4 | Exact | P2 | 214 | 33.6 | 93.0 | +59.3 [+52.3, +66.4] | 130 / 3 | <0.001 | <0.001 | -38.3 | better |
| Phi-4 | Python | P1 | 208 | 40.9 | 81.7 | +40.9 [+32.2, +48.6] | 97 / 12 | <0.001 | <0.001 | -0.5 | better |
| Phi-4 | Python | P2 | 208 | 33.2 | 78.4 | +45.2 [+37.0, +52.9] | 103 / 9 | <0.001 | <0.001 | -31.2 | better |
