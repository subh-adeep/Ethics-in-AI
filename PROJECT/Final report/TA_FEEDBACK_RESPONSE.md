# Mid-term TA feedback (Arati Mohapatra, 21 Oct, grade B+) and where the final report addresses it

Each point is marked **[TA]** inside the outline boxes of the report, so you can see where to write it.

| # | TA comment (short) | Where in the final report | What to do when writing |
|---|---|---|---|
| 1 | Web-search milestone not met | Results §4.2 (web search), Table 2 row "Web search", Appendix F | Say it was completed on 19 Oct (MedQA, 146 Q × 3 models) and why MATH stayed a pilot |
| 2 | Some decisions not justified rigorously | Method §3.2 (4-bit, greedy, 160 tokens), §3.3 (top-2), Table 2 | Give one reason per setting; fill *[your reason]* for the 160-token cap and top-2 |
| 3 | Next steps should include ablations to explain behaviour | Results §4.2 list; Exact vs. T1, Qwen probe, crashed vs. clean code, repair loop | Present these as the analyses that explain the behaviour |
| 4 | Why MedQA and MATH is not motivated | Introduction (paragraph "Why MedQA and MATH"), Method §3.1 | Computation-checkable vs. knowledge claims; high stakes; one verifiable answer; contrast with SycEval's AMPS/MedQuad; expected differences |
| 5 | Introduction too short; motivation unclear | Introduction (4–5 paragraphs outlined) | Expand as outlined |
| 6 | Justify focusing on small models | Introduction ("Why small language models"), Related work (agentic sycophancy) | Local deployment, prior work tested large assistants, weaker tool-call formatting, your compute |
| 7 | Why only two pressure levels | Introduction ("Why two pressure levels"), Related work ("Pressure prompts in prior work") | P1 ≈ suggestion (Sharma et al.), P2 ≈ ethos rebuttal (SycEval); stronger levels add fake evidence that would compete with the tool |
| 8 | Related work should summarise each paper's findings | Related work (findings given per paper) | Check each finding in the paper itself before writing |
| 9 | Highlight the tool aspect in the main question sentence | Abstract, Introduction ("Why tools") | Start the question with external evidence / tools |
| 10 | Define the two (now six) conditions clearly; T0 used before defined | Abstract (no codes), Introduction ("What we did"), Table 2 | Define each condition in words before its code |
| 11 | "Changes little" unclear; give exact improvements | Abstract, Results | Always give points and both percentages |
| 12 | "Answer" too narrow; model should reject any false statement | Abstract, Introduction (problem paragraph) | Use "false claim" framing |
| 13 | Add a final section on implications (e.g. should developers foster more searching?) | New Discussion §5.1 "Implications for model developers" | Answer: more searching is not enough; checkable tools, interface audits, measure trust |
| 14 | Remove "rates should not be compared directly"; instead say why these benchmarks and whether results should match prior work | Introduction ("Why MedQA and MATH"); caveat moved to Limitations | — |
| 15 | Remove "supports keeping earlier messages" line (implicit) | Related work rewritten without it | — |
| 16 | Expand why the Li et al. line motivates your work | Related work ("Knowledge conflicts"): their findings + why your setting is new | — |
| 17 | Remove disclaimers from intro/related work | Moved to Limitations (first bullet) | — |
| 18 | Limited novelty is fine, but motivate small models | Introduction ("Why small language models") | — |
| 19 | Expand on what the datasets are; the proposal's query-structure section was useful | New Method §3.1 + Table 1 (one real query from each dataset) | Describe both datasets as outlined |
| 20 | Remove "selected analysis set" sentence; T1 P0 caveat before flip tables | Removed; one sentence in §3.5 on which questions enter each rate | — |
| 21 | "Numeric checks like what? Size bound?" | Method §3.1 (wrong answer X) | Must differ from the correct answer by > 10⁻⁴ and have absolute value ≤ 10⁸ |
| 22 | "A fixed fallback to what?" | Method §3.1 | Correct answer + 1, used for 1 of 215 questions |
| 23 | "What is 'it'? The code?" | Method §3.2 | "The controller code compares the probabilities …" |
| 24 | Cite how BM25 works | Method §3.3 + Table 2 (Robertson & Zaragoza 2009; bm25s) | — |
| 25 | Why two retrieved entries, not 5? | Method §3.3 | Give your reason; Exact vs. T1 now tests one vs. two |
| 26 | Explain prior work's pressure-prompt types and why your two | Related work ("Pressure prompts in prior work") | — |
| 27 | Surprising that the tool did more harm than good | Results §4.3 (Llama: −47 to −50 points with every MedQA tool) | — |
| 28 | Hypotheses for why Llama follows the user despite correct evidence | Results §4.3 (four hypotheses + testable next steps) | Present as hypotheses, not findings |
| 29 | "Looking forward to the results" (one vs. two entries; answer-only vs. solution) | Results §4.2 (one vs. two); Table 2 (answer-only replaced by Python, with reason) | — |
