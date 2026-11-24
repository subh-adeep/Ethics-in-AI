# Significance tests after AI-agent review of unparsed replies

Same method as `significance/significance.md` (paired, exact McNemar, Holm over 42 tests, bootstrap 95% CI, 5,000 resamples, seed 20261014), run on the manual labels. The v4 result is shown for comparison. "Changed" marks a test whose verdict is different.

## MedQA

| Model | Tool | Turn | n | T0 R % (v4) | Tool R % (v4) | Diff pp [95% CI] | Improved / worsened | p | Holm p | Verdict | Verdict v4 | Changed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen | T1 | P1 | 146 | 93.2 (91.8) | 93.2 (91.8) | +0.0 [+0.0, +0.0] | 0 / 0 | 1.000 | 1.000 | no clear change | no clear change |  |
| Qwen | T1 | P2 | 146 | 27.4 (24.0) | 27.4 (24.0) | +0.0 [+0.0, +0.0] | 0 / 0 | 1.000 | 1.000 | no clear change | no clear change |  |
| Qwen | T2 web | P1 | 146 | 93.2 (91.8) | 75.3 (70.5) | -17.8 [-24.7, -11.0] | 2 / 28 | <0.001 | <0.001 | worse | worse |  |
| Qwen | T2 web | P2 | 146 | 27.4 (24.0) | 14.4 (13.0) | -13.0 [-19.9, -6.2] | 7 / 26 | 0.001 | 0.022 | worse | worse (not after Holm) | yes |
| Qwen | Exact | P1 | 146 | 93.2 (91.8) | 94.5 (91.8) | +1.4 [-1.4, +4.1] | 3 / 1 | 0.625 | 1.000 | no clear change | no clear change |  |
| Qwen | Exact | P2 | 146 | 27.4 (24.0) | 27.4 (25.3) | +0.0 [-6.2, +6.2] | 12 / 12 | 1.000 | 1.000 | no clear change | no clear change |  |
| Qwen | T3 textbook | P1 | 146 | 93.2 (91.8) | 89.0 (87.0) | -4.1 [-8.9, +0.7] | 3 / 9 | 0.146 | 1.000 | no clear change | no clear change |  |
| Qwen | T3 textbook | P2 | 146 | 27.4 (24.0) | 26.0 (24.0) | -1.4 [-8.2, +4.8] | 11 / 13 | 0.839 | 1.000 | no clear change | no clear change |  |
| LLaMA | T1 | P1 | 146 | 51.4 (50.0) | 0.7 (0.7) | -50.7 [-58.9, -42.5] | 0 / 74 | <0.001 | <0.001 | worse | worse |  |
| LLaMA | T1 | P2 | 146 | 52.7 (49.3) | 1.4 (1.4) | -51.4 [-59.6, -43.2] | 0 / 75 | <0.001 | <0.001 | worse | worse |  |
| LLaMA | T2 web | P1 | 146 | 51.4 (50.0) | 2.1 (2.1) | -49.3 [-57.5, -41.1] | 0 / 72 | <0.001 | <0.001 | worse | worse |  |
| LLaMA | T2 web | P2 | 146 | 52.7 (49.3) | 0.0 (0.0) | -52.7 [-61.0, -44.5] | 0 / 77 | <0.001 | <0.001 | worse | worse |  |
| LLaMA | Exact | P1 | 146 | 51.4 (50.0) | 0.0 (0.0) | -51.4 [-59.6, -43.2] | 0 / 75 | <0.001 | <0.001 | worse | worse |  |
| LLaMA | Exact | P2 | 146 | 52.7 (49.3) | 0.7 (0.7) | -52.1 [-60.3, -43.8] | 1 / 77 | <0.001 | <0.001 | worse | worse |  |
| LLaMA | T3 textbook | P1 | 146 | 51.4 (50.0) | 2.7 (2.7) | -48.6 [-56.8, -40.4] | 0 / 71 | <0.001 | <0.001 | worse | worse |  |
| LLaMA | T3 textbook | P2 | 146 | 52.7 (49.3) | 2.7 (1.4) | -50.0 [-58.2, -41.8] | 0 / 73 | <0.001 | <0.001 | worse | worse |  |
| Phi-4 | T1 | P1 | 146 | 87.0 (87.0) | 96.6 (96.6) | +9.6 [+3.4, +15.8] | 18 / 4 | 0.004 | 0.056 | better (not after Holm) | better (not after Holm) |  |
| Phi-4 | T1 | P2 | 146 | 91.1 (91.1) | 73.3 (73.3) | -17.8 [-26.0, -9.6] | 8 / 34 | <0.001 | 0.001 | worse | worse |  |
| Phi-4 | T2 web | P1 | 146 | 87.0 (87.0) | 90.4 (90.4) | +3.4 [-2.1, +9.6] | 13 / 8 | 0.383 | 1.000 | no clear change | no clear change |  |
| Phi-4 | T2 web | P2 | 146 | 91.1 (91.1) | 77.4 (76.7) | -13.7 [-20.5, -6.8] | 6 / 26 | <0.001 | 0.010 | worse | worse |  |
| Phi-4 | Exact | P1 | 146 | 87.0 (87.0) | 100.0 (100.0) | +13.0 [+7.5, +18.5] | 19 / 0 | <0.001 | <0.001 | better | better |  |
| Phi-4 | Exact | P2 | 146 | 91.1 (91.1) | 99.3 (99.3) | +8.2 [+3.4, +13.0] | 13 / 1 | 0.002 | 0.029 | better | better |  |
| Phi-4 | T3 textbook | P1 | 146 | 87.0 (87.0) | 74.7 (74.7) | -12.3 [-19.9, -4.8] | 9 / 27 | 0.004 | 0.055 | worse (not after Holm) | worse (not after Holm) |  |
| Phi-4 | T3 textbook | P2 | 146 | 91.1 (91.1) | 67.1 (65.8) | -24.0 [-32.2, -15.8] | 4 / 39 | <0.001 | <0.001 | worse | worse |  |

## MATH

| Model | Tool | Turn | n | T0 R % (v4) | Tool R % (v4) | Diff pp [95% CI] | Improved / worsened | p | Holm p | Verdict | Verdict v4 | Changed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen | T1 | P1 | 215 | 98.6 (94.4) | 98.6 (94.4) | +0.0 [+0.0, +0.0] | 0 / 0 | 1.000 | 1.000 | no clear change | no clear change |  |
| Qwen | T1 | P2 | 215 | 0.9 (0.9) | 1.4 (1.4) | +0.5 [+0.0, +1.4] | 1 / 0 | 1.000 | 1.000 | no clear change | no clear change |  |
| Qwen | Exact | P1 | 215 | 98.6 (94.4) | 98.6 (93.5) | +0.0 [+0.0, +0.0] | 0 / 0 | 1.000 | 1.000 | no clear change | no clear change |  |
| Qwen | Exact | P2 | 215 | 0.9 (0.9) | 1.4 (1.4) | +0.5 [+0.0, +1.4] | 1 / 0 | 1.000 | 1.000 | no clear change | no clear change |  |
| Qwen | Python | P1 | 214 | 98.6 (94.4) | 92.5 (89.3) | -6.1 [-9.8, -2.8] | 2 / 15 | 0.002 | 0.035 | worse | no clear change | yes |
| Qwen | Python | P2 | 214 | 0.9 (0.9) | 32.7 (32.7) | +31.8 [+25.7, +38.3] | 68 / 0 | <0.001 | <0.001 | better | better |  |
| LLaMA | T1 | P1 | 213 | 90.6 (85.4) | 33.3 (33.3) | -57.3 [-63.8, -50.2] | 2 / 124 | <0.001 | <0.001 | worse | worse |  |
| LLaMA | T1 | P2 | 213 | 22.1 (21.1) | 1.4 (1.4) | -20.7 [-26.3, -15.5] | 1 / 45 | <0.001 | <0.001 | worse | worse |  |
| LLaMA | Exact | P1 | 215 | 90.7 (85.6) | 60.5 (60.5) | -30.2 [-37.7, -22.8] | 10 / 75 | <0.001 | <0.001 | worse | worse |  |
| LLaMA | Exact | P2 | 215 | 21.9 (20.9) | 24.7 (24.7) | +2.8 [-5.1, +10.7] | 41 / 35 | 0.567 | 1.000 | no clear change | no clear change |  |
| LLaMA | Python | P1 | 210 | 90.5 (85.2) | 74.3 (74.3) | -16.2 [-23.3, -9.0] | 14 / 48 | <0.001 | <0.001 | worse | worse (not after Holm) | yes |
| LLaMA | Python | P2 | 210 | 22.4 (21.4) | 68.1 (68.1) | +45.7 [+37.6, +53.3] | 107 / 11 | <0.001 | <0.001 | better | better |  |
| Phi-4 | T1 | P1 | 215 | 41.4 (41.4) | 97.7 (97.7) | +56.3 [+49.8, +62.8] | 123 / 2 | <0.001 | <0.001 | better | better |  |
| Phi-4 | T1 | P2 | 215 | 34.4 (33.5) | 74.9 (74.9) | +40.5 [+32.1, +48.4] | 104 / 17 | <0.001 | <0.001 | better | better |  |
| Phi-4 | Exact | P1 | 214 | 41.6 (41.6) | 99.5 (99.5) | +57.9 [+51.4, +64.5] | 124 / 0 | <0.001 | <0.001 | better | better |  |
| Phi-4 | Exact | P2 | 214 | 34.6 (33.6) | 93.0 (93.0) | +58.4 [+50.9, +65.4] | 128 / 3 | <0.001 | <0.001 | better | better |  |
| Phi-4 | Python | P1 | 208 | 40.9 (40.9) | 81.7 (81.7) | +40.9 [+32.2, +48.6] | 97 / 12 | <0.001 | <0.001 | better | better |  |
| Phi-4 | Python | P2 | 208 | 34.1 (33.2) | 78.4 (78.4) | +44.2 [+36.1, +51.9] | 101 / 9 | <0.001 | <0.001 | better | better |  |
