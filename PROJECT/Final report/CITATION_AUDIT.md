# Citation audit for `references.bib`

Checked on 2026-11-21. The bibliography is for the final report and uses `project_template.bst` (natbib author-year).

## Summary

- **33 entries**: 8 carried over from the midsem bibliography and 25 new. All keys match the requested list and the `\citation` keys in `final_report.aux`.
- **32 entries fully verified** against an authoritative record that I opened. **1 entry is partly verified** (`holm1979`; see below). None were unverifiable.
- **Syntax check**: 33 entries, no duplicate keys, balanced braces, ASCII only (accents are written as LaTeX escapes), and every required field present. I also ran BibTeX 0.99e (MiKTeX) with `project_template.bst` on a scratch copy citing all keys: **0 warnings, 33 `\bibitem`s**.
- **How I checked**: I fetched the raw records myself instead of using memory. Sources were ACL Anthology `.bib` files and HTML pages, the arXiv API and abstract pages, the OpenReview API, NeurIPS proceedings `.bib` files and abstract pages, the PMLR page, Crossref, doi.org handle lookups, publisher pages, publisher OAI-PMH metadata endpoints, and zbMATH Open.
- **Two later requests (2026-11-23):** `jittham2026` was **not** added and is not cited (withdrawn by its author). `ranaldi2026` was added afterwards using the verified entry from the addendum (correct title "Learning Multilingual Agentic Policy to Control Sycophancy", authors Ranaldi and Giulia Pucci, EACL 2026). The bib now has 34 entries. Details are in the addendum at the end of this file.
- **Blocked sites**: JSTOR, DBLP, the OpenReview web/PDF pages, the AIES article page and the MDPI article page show bot challenges or return HTTP 403 to automated access. I did not try to get past them. Where that happened I used the publisher's official metadata interface (OAI-PMH or API) or Crossref, and I name the source below.

### Conventions

- `@inproceedings` for conference and Findings papers, with `booktitle` exactly as the proceedings name it.
- `@article` for journals and TMLR, and for arXiv-only papers (`journal = {arXiv preprint arXiv:XXXX.XXXXX}`). `project_template.bst` has no `eprint` field, so the `@misc`+eprint style would not render.
- `@book` for the bootstrap book.
- Each entry has a DOI if one exists, otherwise a URL. Never both.
- **arXiv-only papers use the year of arXiv version 1.** This is arXiv's `citation_date` and the usual convention. arXiv's own "Export BibTeX" uses the year of the *latest* version instead. That matters for `qwen2025`, flagged below.
- Very long author lists (more than 40 names) keep the first 8 and end with `and others`, or use the official team name where the key implies one (`qwen2025`, `gemma2024`). Lists of up to 22 names are given in full.
- No editors, months or conference locations, to keep the reference list compact. Capitals in titles are protected because the style sentence-cases titles.

### Corrections and flags (read these)

| Key | Issue | What the bib does |
|---|---|---|
| `robertson2009` | **Sources conflict.** The monograph's own masthead (publisher-typeset PDF) reads *Vol. 3, No. 4 (2009) 333–389*. The current publisher platform (Emerald, which now hosts FnTIR; nowpublishers.com redirects there) and Crossref list *Vol. 4, Issue 1–2, pp. 1–174*. | Uses the version of record: 3(4):333–389, plus the DOI. Pages 333–389 were missing from the listing and have been added. Anyone who follows the DOI will see the Emerald numbering. |
| `qwen2025` | Year does not match the key. arXiv v1 is 19 Dec 2024 and v2 is 3 Jan 2025. | `year = {2024}` (v1 convention). Change only the `year` field to 2025 if you want to cite v2. The key is unchanged as instructed. |
| `holm1979` | **Partly verified.** zbMATH confirms the title, sole author, journal, vol. 6, pp. 65–70 and 1979. The **issue number (2)** and the **JSTOR stable URL** could only be confirmed indirectly, because JSTOR blocks automated access. **No DOI exists.** OpenAlex claims `10.2307/4615733`, but that DOI is *not registered* (doi.org: handle not found; Crossref: 404). | Keeps `number = {2}` and `url = {https://www.jstor.org/stable/4615733}`, with no DOI. Delete those two fields if you want only directly verified fields. |
| `sharma2024` | The ICLR/OpenReview record lists **18 authors**. arXiv (v4) lists 19, adding Newton Cheng as 7th author. The ICLR PDF itself could not be opened (browser challenge). | Follows the venue record: all 18 authors. Booktitle changed to the official ICLR 2024 name and URL changed to OpenReview. If you prefer, truncate after Bowman with `and others`; the first six authors are the same in both records. |
| `fanous2025` | The published AIES version has **9 authors**: Sonnet Xu and Vasiliki Bikia are added, and "Ank A. Agarwal" becomes "Ank Agarwal". The midsem draft had the 7-author arXiv list. | Uses the published 9-author list. Entry type changed from `@article` to `@inproceedings`. Vol. 8 is kept. `number = {1}` is left out because the style cannot print volume and number together for `@inproceedings` (it would emit a warning). |
| `schick2023` | The NeurIPS version has **9 authors**: Eric Hambro is added as 6th author. arXiv (v1, the only version) has 8. | Uses the NeurIPS 9-author list. |
| `efron1993` | Publisher records differ in date and name. Crossref/Springer gives 1993 with the original print ISBN 0-412-04231-2, but its SpringerLink page now returns 404. The Taylor & Francis (CRC) e-book page gives 15 May 1994 and the imprint "Chapman and Hall/CRC". The Library of Congress record (via Open Library) gives Chapman & Hall, New York, 1993, and the series. | Chapman \& Hall, New York, 1993, volume 57 of *Monographs on Statistics and Applied Probability*. The DOI is the live T&F one, `10.1201/9780429246593`, for the same first edition. |
| `hendrycks2021`, `jin2021`, `li2026`, `kaur2024`, `sathyanathan2026` | The midsem *draft* bib cited the arXiv versions. | Now cite the published venue versions, matching the listing. |
| `hong2025` | DOI was missing. | `doi = {10.18653/v1/2025.findings-emnlp.121}` added. |

Minor, cosmetic only: the MedCPT journal title uses lower case where the arXiv title is in title case; the EMNLP title of `wolf2020` differs from its arXiv preprint; accents were restored from arXiv where the venue metadata had dropped them (Dessì, Rémi, Lukošiūtė). Details are in the per-entry notes.

The midsem draft's model-card `@misc` entries (`qwenmodel`, `llamamodel`, `phimodel`, `gemmamodel`) were not requested and are not included. The technical reports replace them.

---

## Per-entry record

### A. Entries carried over from the midsem bib

**`sharma2024`** — Towards Understanding Sycophancy in Language Models
- Source: OpenReview record `tvhaxkMKAn`, https://openreview.net/forum?id=tvhaxkMKAn, read through `https://api2.openreview.net/notes/search`. Also arXiv: https://arxiv.org/abs/2310.13548 (API and abstract page).
- Confirmed: ICLR 2024 (poster). Booktitle "The Twelfth International Conference on Learning Representations", 2024. No pages, no DOI.
- Notes: The full author list follows the venue record (18 names; arXiv has 19; see flags). Name forms come from OpenReview: "Esin DURMUS" is normalised to "Durmus", and "Scott R Johnston" and "Shauna M Kravec" get periods after the initials.

**`fanous2025`** — SycEval: Evaluating LLM Sycophancy
- Source: Crossref, https://api.crossref.org/works/10.1609/aies.v8i1.36598. AIES OJS OAI-PMH record, https://ojs.aaai.org/index.php/AIES/oai?verb=GetRecord&metadataPrefix=oai_dc&identifier=oai:ojs.aaai.org:article/36598. The article page https://ojs.aaai.org/index.php/AIES/article/view/36598 is behind a bot challenge. Also arXiv https://arxiv.org/abs/2502.08177 (comment "AIES 2025").
- Confirmed: *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society*, Vol. 8 No. 1 (2025), AIES-25 Main Track I. Pages 893–900, published 2025-10-15, publisher AAAI, DOI 10.1609/aies.v8i1.36598.
- Notes: 9 authors (see flags). Crossref registers the paper as a journal article in a serial (ISSN 3065-8365). So `@article` with volume 8, number 1 would also be valid; I used `@inproceedings` per your rule.

**`hong2025`** — Measuring Sycophancy of Language Models in Multi-turn Dialogues
- Source: https://aclanthology.org/2025.findings-emnlp.121/ and https://aclanthology.org/2025.findings-emnlp.121.bib
- Confirmed: Findings of the Association for Computational Linguistics: EMNLP 2025. Pages 2239–2259, November 2025, ACL, DOI 10.18653/v1/2025.findings-emnlp.121. Four authors, as listed.
- Notes: DOI added. The URL is dropped because the DOI is there. No errors.

**`li2026`** — How Large Language Models Balance Internal Knowledge with User and Document Assertions
- Source: https://aclanthology.org/2026.findings-acl.1267/ (and `.bib`). Also arXiv https://arxiv.org/abs/2604.22193 (comment "Findings of ACL 2026").
- Confirmed: Findings of the Association for Computational Linguistics: ACL 2026. Pages 25323–25346, July 2026, ACL, DOI 10.18653/v1/2026.findings-acl.1267.
- Notes: Matches the listing. No problems.

**`kaur2024`** — Evaluating Large Language Models for Health-related Queries with Presuppositions
- Source: https://aclanthology.org/2024.findings-acl.850/ (and `.bib`). Also arXiv https://arxiv.org/abs/2312.08800 (comment "Findings of ACL 2024").
- Confirmed: Findings of the Association for Computational Linguistics: ACL 2024. Pages 14308–14331, August 2024, ACL, DOI 10.18653/v1/2024.findings-acl.850.
- Notes: Matches the listing. No problems.

**`sathyanathan2026`** — Evaluating Reasoning Models for Queries with Presuppositions
- Source: https://aclanthology.org/2026.findings-acl.1201/ (and `.bib`). Also arXiv https://arxiv.org/abs/2605.03050 (comment "Accepted to ACL 2026 (Findings)").
- Confirmed: Findings of the Association for Computational Linguistics: ACL 2026. Pages 23991–24006, July 2026, ACL, DOI 10.18653/v1/2026.findings-acl.1201.
- Notes: Matches the listing. No problems.

**`hendrycks2021`** — Measuring Mathematical Problem Solving With the MATH Dataset
- Source: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html and its official BibTeX, https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/113-/bibtex. Also arXiv https://arxiv.org/abs/2103.03874.
- Confirmed: *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*, volume 1 (NeurIPS Datasets and Benchmarks 2021), 2021. The pages field is empty in the official record, and there is no DOI. Eight authors, as listed.
- Notes: The midsem draft cited arXiv. This entry now cites the proceedings version.

**`jin2021`** — What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams
- Source: Crossref, https://api.crossref.org/works/10.3390/app11146421. MDPI OAI-PMH record, https://oai.mdpi.com/oai/oai2.php?verb=GetRecord&metadataPrefix=oai_dc&identifier=oai:mdpi.com:/2076-3417/11/14/6421/. The article page https://www.mdpi.com/2076-3417/11/14/6421 returned HTTP 403. Also arXiv https://arxiv.org/abs/2009.13081.
- Confirmed: *Applied Sciences* 11(14):6421 (article number), published 2021-07-12, MDPI, DOI 10.3390/app11146421. Six authors, as listed.
- Notes: The midsem draft's URL pointed to the arXiv preprint, which has different title capitalisation and the comment "Submitted to AAAI 2021". It is replaced by the journal DOI. `{A}` is braced so the capital after "?" survives sentence-casing.

### B. New entries

**`qwen2025`** — Qwen2.5 Technical Report
- Source: https://arxiv.org/abs/2412.15115 (abstract page and API), and the v2 PDF title page, whose byline reads "Qwen Team".
- Confirmed: arXiv-only, arXiv:2412.15115. v1 is 19 Dec 2024 and v2 is 3 Jan 2025. arXiv lists the team label "Qwen" followed by 42 named authors.
- Notes: `author = {{Qwen Team}}` follows the byline. **The year is set to 2024 (v1)**, which does not match the key (see flags).

**`grattafiori2024`** — The Llama 3 Herd of Models
- Source: https://arxiv.org/abs/2407.21783 (abstract page and API).
- Confirmed: arXiv-only, arXiv:2407.21783. v1 is 31 Jul 2024 and v3 is 23 Nov 2024. arXiv lists 561 authors, starting with Aaron Grattafiori.
- Notes: Keeps the first 8 authors and ends with `and others`.

**`abouelenin2025`** — Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs
- Source: https://arxiv.org/abs/2503.01743 (abstract page and API).
- Confirmed: arXiv-only, arXiv:2503.01743. v1 is 3 Mar 2025 and v2 is 7 Mar 2025; the comment reads "39 pages".
- Notes: arXiv lists the team label "Microsoft" (plus a ":" separator) before 74 named authors. The bib starts with the first named author, Abouelenin, to match the key, keeps 8 names and ends with `and others`.

**`gemma2024`** — Gemma 2: Improving Open Language Models at a Practical Size
- Source: https://arxiv.org/abs/2408.00118 (abstract page and API), and the v3 PDF title page, whose byline reads "Gemma Team, Google DeepMind". The full list is in the paper's Contributions section.
- Confirmed: arXiv-only, arXiv:2408.00118. v1 is 31 Jul 2024 and v3 is 2 Oct 2024.
- Notes: `author = {{Gemma Team}}`.

**`dettmers2023`** — QLoRA: Efficient Finetuning of Quantized LLMs
- Source: https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html and the official `.bib`. Also arXiv https://arxiv.org/abs/2305.14314.
- Confirmed: *Advances in Neural Information Processing Systems* 36 (NeurIPS 2023), Main Conference Track. Pages 10088–10115, Curran Associates, DOI 10.52202/075280-0441 (doi.org handle resolves). Four authors.
- Notes: No problems.

**`robertson2009`** — The Probabilistic Relevance Framework: BM25 and Beyond
- Source:
  - The publisher-typeset now Publishers PDF, hosted on the first author's City, University of London page: https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf. Its masthead reads "Foundations and Trends in Information Retrieval Vol. 3, No. 4 (2009) 333–389, © 2009 S. Robertson and H. Zaragoza, DOI: 10.1561/1500000019".
  - Emerald: https://www.emerald.com/ftinr/article/4/1-2/1/1326508/The-Probabilistic-Relevance-Framework-BM25-and
  - Crossref: https://api.crossref.org/works/10.1561/1500000019
- Confirmed: *Foundations and Trends in Information Retrieval*, 2009, DOI 10.1561/1500000019. Authors Stephen Robertson and Hugo Zaragoza.
- Notes: **Volume, issue and pages conflict** (see flags). The bib uses 3(4):333–389 from the masthead.

**`lu2024bm25s`** — BM25S: Orders of magnitude faster lexical search via eager sparse scoring
- Source: https://arxiv.org/abs/2407.03618 (abstract page and API).
- Confirmed: arXiv-only, arXiv:2407.03618, submitted 4 Jul 2024, comment "Technical Report". Sole author Xing Han Lù, written `L{\`u}`. An OpenReview search found no venue version.
- Notes: No problems.

**`xiong2024`** — Benchmarking Retrieval-Augmented Generation for Medicine
- Source: https://aclanthology.org/2024.findings-acl.372/ (and `.bib`). Also arXiv https://arxiv.org/abs/2402.13178.
- Confirmed: Findings of the Association for Computational Linguistics: ACL 2024. Pages 6233–6251, August 2024, ACL, DOI 10.18653/v1/2024.findings-acl.372. Authors Xiong, Jin, Lu and Zhang.
- Notes: No problems.

**`jin2023medcpt`** — MedCPT: Contrastive Pre-trained Transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval
- Source: OUP article page, https://academic.oup.com/bioinformatics/article/39/11/btad651/7335842. Crossref, https://api.crossref.org/works/10.1093/bioinformatics/btad651. arXiv https://arxiv.org/abs/2307.00589, whose record links the DOI.
- Confirmed: *Bioinformatics* 39(11), article btad651, published 1 November 2023, OUP, DOI 10.1093/bioinformatics/btad651. Seven authors.
- Notes: `pages = {btad651}` is the article number; there is no page range. The title uses the journal's capitalisation (lower case after "Transformers with"). The style sentence-cases titles anyway, so this has no visible effect.

**`lewis2020`** — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Source: https://proceedings.neurips.cc/paper_files/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html and the official `.bib`. Also arXiv https://arxiv.org/abs/2005.11401 (comment "Accepted at NeurIPS 2020").
- Confirmed: *Advances in Neural Information Processing Systems* 33 (NeurIPS 2020). Pages 9459–9474, Curran Associates. The proceedings list no DOI. Twelve authors.
- Notes: No problems.

**`schick2023`** — Toolformer: Language Models Can Teach Themselves to Use Tools
- Source: https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html and the official `.bib`. Also arXiv https://arxiv.org/abs/2302.04761.
- Confirmed: *Advances in Neural Information Processing Systems* 36 (NeurIPS 2023), Main Conference Track. Pages 68539–68551, Curran Associates, DOI 10.52202/075280-2997 (handle resolves).
- Notes: **9 authors on NeurIPS versus 8 on arXiv** (Eric Hambro is added). The proceedings metadata spells "Dessi" and arXiv spells "Dessì"; the bib uses `Dess{\`\i}`.

**`yao2023`** — ReAct: Synergizing Reasoning and Acting in Language Models
- Source: OpenReview record `WE_vluYUL-X`, https://openreview.net/forum?id=WE_vluYUL-X, read through `https://api.openreview.net/notes/search`. Also arXiv https://arxiv.org/abs/2210.03629; the v3 comment says it is the ICLR camera-ready.
- Confirmed: ICLR 2023 (notable, top 5%). Booktitle "The Eleventh International Conference on Learning Representations", 2023. Seven authors.
- Notes: The name is given as "Karthik Narasimhan", as on arXiv and the camera-ready; OpenReview's profile form is "Karthik R Narasimhan".

**`gao2023pal`** — PAL: Program-aided Language Models
- Source: https://proceedings.mlr.press/v202/gao23f.html, including the BibTeX shown on that page.
- Confirmed: *Proceedings of the 40th International Conference on Machine Learning*, PMLR volume 202. Pages 10764–10799, 23–29 July 2023. Eight authors. PMLR issues no DOIs.
- Notes: No problems.

**`chen2023pot`** — Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks
- Source: OpenReview record `YfZ4ZPt8zd` ("Accepted by TMLR"), https://openreview.net/forum?id=YfZ4ZPt8zd. Also arXiv https://arxiv.org/abs/2211.12588 (comment "Published at TMLR 2023").
- Confirmed: *Transactions on Machine Learning Research*, 2023, ISSN 2835-8856. TMLR has no volume or pages. Four authors.
- Notes: The ISSN is left out, because the style would print it.

**`chen2024selfdebug`** — Teaching Large Language Models to Self-Debug
- Source: OpenReview record `KuPixIqPiq`, https://openreview.net/forum?id=KuPixIqPiq (API). Also arXiv https://arxiv.org/abs/2304.05128.
- Confirmed: ICLR 2024 (poster). Booktitle "The Twelfth International Conference on Learning Representations", 2024. Four authors (Schärli is written `Sch{\"a}rli`).
- Notes: No problems.

**`huang2024`** — Large Language Models Cannot Self-Correct Reasoning Yet
- Source: OpenReview record `IkmD3fKBPQ`, https://openreview.net/forum?id=IkmD3fKBPQ (API). Also arXiv https://arxiv.org/abs/2310.01798 (comment "ICLR 2024").
- Confirmed: ICLR 2024 (poster). Booktitle "The Twelfth International Conference on Learning Representations", 2024. Seven authors.
- Notes: No problems.

**`perez2023`** — Discovering Language Model Behaviors with Model-Written Evaluations
- Source: https://aclanthology.org/2023.findings-acl.847/ (and `.bib`). Also arXiv https://arxiv.org/abs/2212.09251.
- Confirmed: Findings of the Association for Computational Linguistics: ACL 2023. Pages 13387–13434, July 2023, ACL, DOI 10.18653/v1/2023.findings-acl.847. 63 authors.
- Notes: Keeps the first 8 authors and ends with `and others`. ACL Anthology spells the third author in ASCII ("Kamile Lukosiute"); the bib uses the accented arXiv spelling, `Luko{\v{s}}i{\=u}t{\.e}, Kamil{\.e}`.

**`wei2023`** — Simple synthetic data reduces sycophancy in large language models
- Source: https://arxiv.org/abs/2308.03958 (abstract page and API).
- Confirmed: arXiv-only, arXiv:2308.03958. v1 is 7 Aug 2023 and v2 is 15 Feb 2024. Five authors.
- Notes: OpenReview shows the paper was submitted to ICLR 2025 and not accepted, so it stays arXiv-only. Year 2023 (v1); arXiv's own BibTeX export says 2024.

**`laban2023`** — Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment
- Source: https://arxiv.org/abs/2311.08596 (abstract page and API).
- Confirmed: arXiv-only, arXiv:2311.08596. v1 is 14 Nov 2023 and v2 is 21 Feb 2024. Four authors. OpenReview and web searches found no venue version.
- Notes: `{C}` is braced so the capital after "?" survives sentence-casing.

**`xie2024`** — Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts
- Source: OpenReview record `auKAUJZMO6`, https://openreview.net/forum?id=auKAUJZMO6 (API). Also arXiv https://arxiv.org/abs/2305.13300 (comment "ICLR 2024 (Spotlight)").
- Confirmed: ICLR 2024 (spotlight). Booktitle "The Twelfth International Conference on Learning Representations", 2024. Five authors.
- Notes: The OpenReview title contains a stray double space, normalised here.

**`wu2024clasheval`** — ClashEval: Quantifying the tug-of-war between an LLM's internal prior and external evidence
- Source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/3aa291abc426d7a29fb08418c1244177-Abstract-Datasets_and_Benchmarks_Track.html and the official `.bib`. Also OpenReview `WGoCZl2itU` ("NeurIPS 2024 Track Datasets and Benchmarks Poster") and arXiv https://arxiv.org/abs/2404.10198.
- Confirmed: *Advances in Neural Information Processing Systems* 37 (NeurIPS 2024), Datasets and Benchmarks Track. Pages 33402–33422, Curran Associates, DOI 10.52202/079017-1053 (handle resolves). Three authors.
- Notes: The official booktitle does not name the track, so the track goes in `note`.

**`mcnemar1947`** — Note on the sampling error of the difference between correlated proportions or percentages
- Source: Springer page https://link.springer.com/article/10.1007/BF02295996 (citation metadata). Crossref, https://api.crossref.org/works/10.1007/BF02295996, which now lists Cambridge University Press as the current publisher. The Cambridge Core page also opened.
- Confirmed: *Psychometrika* 12(2):153–157, June 1947, DOI 10.1007/BF02295996. Sole author Quinn McNemar.
- Notes: Matches the listing. No problems.

**`holm1979`** — A Simple Sequentially Rejective Multiple Test Procedure
- Source: zbMATH Open record https://zbmath.org/3624650 (Zbl 0402.62058), read through `https://api.zbmath.org/v1/document/_search`. JSTOR, https://www.jstor.org/stable/4615733, returned a bot-challenge page, so I could not open it.
- Confirmed: *Scandinavian Journal of Statistics* 6:65–70 (1979). Sole author Sture Holm.
- Notes: **Partly verified.** zbMATH gives no issue number, and neither does OpenAlex. Issue 2 is corroborated only by a search-engine listing of JSTOR's issue page, "Vol. 6, No. 2, 1979" (https://www.jstor.org/stable/i412579), which lists the article. The JSTOR stable ID 4615733 is corroborated only by the OpenAlex record (which builds its identifier from that ID) and by secondary web pages. **There is no DOI.** OpenAlex's `10.2307/4615733` does not resolve (doi.org: handle not found; Crossref: 404), so it is left out.

**`efron1993`** — An Introduction to the Bootstrap
- Source:
  - Taylor & Francis: https://www.taylorfrancis.com/books/mono/10.1201/9780429246593/introduction-bootstrap-bradley-efron-tibshirani
  - Crossref (Springer record): https://api.crossref.org/works/10.1007/978-1-4899-4541-9
  - Open Library edition record derived from the Library of Congress catalogue: https://openlibrary.org/isbn/0412042312.json
- Confirmed: Bradley Efron and Robert J. Tibshirani. Chapman & Hall, New York, 1993, per the Library of Congress record (LCCN 93004489) and the Crossref/Springer record. Print ISBN 978-0-412-04231-7. *Monographs on Statistics and Applied Probability*, 57, per the Library of Congress record. DOI 10.1201/9780429246593 (T&F/CRC).
- Notes: The date and imprint differ between publisher records (see flags). The Springer DOI `10.1007/978-1-4899-4541-9` is registered, but its landing page returns 404, so the bib uses the live T&F DOI.

**`wolf2020`** — Transformers: State-of-the-Art Natural Language Processing
- Source: https://aclanthology.org/2020.emnlp-demos.6/ (and `.bib`). Also arXiv https://arxiv.org/abs/1910.03771.
- Confirmed: *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. Pages 38–45, October 2020, ACL, DOI 10.18653/v1/2020.emnlp-demos.6. 22 authors, all listed.
- Notes: The arXiv preprint has a different title ("HuggingFace's Transformers: ..."); the bib follows the EMNLP version. Names follow the Anthology page (Clément Delangue, Alexander M. Rush), except "Remi Louf", which is written R{\'e}mi as on arXiv.

---

## Addendum (2026-11-23): two further references, not added

Both were checked under the same rules. Neither was appended to `references.bib`, because the coordinator's instruction was not to add an entry when the paper cannot be found or its metadata differs. The bib still has 33 entries.

**`jittham2026`** — Agentic Scaffolding Amplifies Sycophantic Behavior in Large Language Models. **WITHDRAWN; do not cite.**
- Source: arXiv abstract page https://arxiv.org/abs/2608.21377 (v2) and https://arxiv.org/abs/2608.21377v1, plus the arXiv API. I also searched OpenReview and the web for a workshop version.
- Confirmed: The title is as listed, and the sole author is Thantham Jittham. v1 is dated 6 Jul 2026; its comment reads "13 pages, 4 figures. Accepted to the UAI 2026 Workshop on Safe AI". **v2 (1 Sep 2026) is a withdrawal.** The page says "This paper has been withdrawn by Thantham Jittham", and the comment says the reported results "do not correspond to the executed evaluation and are unsupported. The paper should not be cited". No workshop proceedings or accepted-papers page for the UAI 2026 Safe AI workshop could be found; the acceptance claim rests only on the v1 arXiv comment.
- Problem: The proposal's entry (`jittham2026agentic`, `journal={arXiv preprint / UAI Safe AI Workshop}`) and its related-work sentence ("demonstrated agentic sycophancy amplification under scaffolding") rely on results the author has retracted. Remove the citation, or at most mention that the paper was withdrawn.
- Main finding claimed in the (withdrawn) abstract: across 4,800 veracity judgments, agentic interaction scaffolding (feedback loops, reconsideration checkpoints, iterative refinement) "systematically amplifies sycophantic behavior". The abstract reports a mean accuracy drop of 6.3 points and larger effects in more capable models. The author now says these results are unsupported.

**`ranaldi2026`** — **metadata does not match the listing.** The paper at the listed URL is real and verified, but under a different title and second author.
- Source: https://aclanthology.org/2026.eacl-long.169/ (HTML page and `.bib`), and the PDF title page https://aclanthology.org/2026.eacl-long.169.pdf. I also searched the web and OpenReview for the listed title.
- Confirmed: **"Learning Multilingual Agentic Policy to Control Sycophancy"** by **Leonardo Ranaldi and Giulia Pucci**. *Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3664–3681, March 2026 (Rabat, Morocco), ACL, DOI 10.18653/v1/2026.eacl-long.169.
- Problem: The listed title, "Sycophancy as an Agentic Policy Problem", does not appear as the title of any paper I could find. It looks like a paraphrase of the paper's argument, which the proposal also uses ("formulated sycophancy as an agentic policy problem"). The listed second author, "André Pucci", is actually **Giulia Pucci**. The proposal's booktitle was also a shortened form, and it had no pages or DOI.
- Main finding stated in the abstract: the authors argue that sycophancy is "a policy-level failure" rather than a surface artefact. They train an agentic policy with an explicit action space (answer, counter misleading signals, ask for clarification) using a multi-objective reward. They report less sycophancy, better task performance, and generalisation across languages (three models, 11 languages per the introduction). It is a mitigation method, not an evaluation of agentic amplification, so check that the report's sentence about it matches this.
- If this is the intended paper, the following verified entry can be pasted into `references.bib`. It has been tested with `project_template.bst` (0 warnings).

```bibtex
@inproceedings{ranaldi2026,
  title     = {Learning Multilingual Agentic Policy to Control Sycophancy},
  author    = {Ranaldi, Leonardo and Pucci, Giulia},
  booktitle = {Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {3664--3681},
  publisher = {Association for Computational Linguistics},
  year      = {2026},
  doi       = {10.18653/v1/2026.eacl-long.169}
}
```

---

## Addendum (2026-11-24): claim check for `final_report_submission.tex`

The submission cites 33 papers (`ranaldi2026` is still in the bib but is no longer cited, so it is not printed). Every sentence that cites a paper was checked against the paper itself; the five less-known claims were checked against the source pages (arXiv / ACL Anthology).

Corrected in the report:
- `perez2023`: "Larger and RLHF-tuned models more often repeat a user's view" -> "Larger models more often repeat a user's view, and RLHF training does not remove this". The paper finds sycophancy grows with size; RLHF leaves it about the same (it does not increase it).
- `sharma2024`: "human preference data rewards agreement" -> "partly rewards agreement" (the paper says it "somewhat incentivizes" matching the user's beliefs).
- `li2026`: "models cannot reliably tell helpful from harmful" -> "most models cannot..." (the abstract says most models, 27 LLMs from 3 families).
- `kaur2024`: "a retrieval-augmented assistant was less affected" -> names the models: agreement rose for InstructGPT, ChatGPT and GPT-4, "but not Bing Copilot" (abstract: all models except Bing Copilot).
- `sathyanathan2026`: "reasoning models also fail to challenge many false presuppositions" -> "reasoning helps only a little: reasoning models still fail to challenge 26--42% of false presuppositions" (abstract).

Checked and correct as written: `fanous2025` (models, AMPS/MedQuad, 58.19%, 14.66%, 78.5%, ethos rebuttal), `hong2025`, `xie2024`, `wu2024clasheval`, `wei2023`, `laban2023`, `lewis2020`, `schick2023`, `yao2023`, `gao2023pal`, `chen2023pot`, `huang2024`, `chen2024selfdebug`, datasets, models, tools and statistics papers.

Minor note: `grattafiori2024` (The Llama 3 Herd of Models) describes Llama 3/3.1; Meta has no separate paper for Llama 3.2, so citing it for Llama-3.2-3B is the usual practice.

**2026-11-24, later:** at the author's request, `wei2023`, `laban2023`, `schick2023`, `yao2023` and `chen2023pot` were removed from the submission (their sentences were rewritten). They stay in `references.bib` but are no longer cited or printed. The submission now cites 28 papers.
