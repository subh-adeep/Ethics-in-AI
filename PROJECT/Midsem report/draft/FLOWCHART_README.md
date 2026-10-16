# Experiment flowchart and prompt section

The diagram is editable TikZ, with an independently compiled vector PDF. It shows the main analysis question selection and the two conversation branches. All six runs actually evaluated 421 questions; the 361-question intersection is selected for the main comparison after checking current T0 P0 correctness. It does not imply that only 361 questions were run or that all T1 P0 answers were correct.

## Insert the compiled figure

Use the report's existing graphicx package:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/experiment_flowchart_standalone.pdf}
\caption{Main comparison on 361 shared questions, with a real saved MATH example. All runs evaluated 421 questions. T1 offers optional BM25 answer retrieval. The displayed answers use stored extraction labels.}
\label{fig:experiment-flow}
\end{figure}
```

## Insert editable TikZ instead

Add to the preamble:

```latex
\usepackage{amsmath,graphicx,tikz}
\usetikzlibrary{arrows.meta,positioning,calc}
```

Then put this inside the figure environment, in place of includegraphics:

```latex
\resizebox{\linewidth}{!}{\input{figures/experiment_flowchart.tex}}
```

For the separate prompt subsection:

```latex
\input{experiment_prompts.tex}
```

For legend definitions, retrieval details and the saved-example caveat, use separate LaTeX text below the figure:

```latex
\input{flowchart_notes.tex}
```

The revised diagram reads horizontally and includes marks from the official Qwen, Meta LLaMA and Microsoft Hugging Face organisation profiles. Meta and Microsoft marks identify the model developers. Logo source URLs and hashes are recorded in figures/logos/sources.json. Copy the logos folder with the editable TikZ source. A vector search symbol marks retrieval. The legend is separate from the figure. Terminology is now BM25 answer retrieval; technically, retrieval followed by model generation is also a form of RAG, but the more specific term describes this implementation clearly.

These are ready-to-insert assets. The six-page main report has not been expanded automatically: inserting the full diagram and prompt section there would require rebalancing the page layout. Exact prompts can remain in the appendix.

Compile the standalone diagram from its figures directory:

```text
pdflatex -interaction=nonstopmode -halt-on-error experiment_flowchart_standalone.tex
```

## Horizontal layout

The source and PDF now read left to right. Use a landscape page for readability instead of shrinking all the labels to a narrow portrait column. In the preamble add `\usepackage{pdflscape}`. Then use:

```latex
\begin{landscape}
\begin{figure}[p]
\centering
\includegraphics[width=\linewidth]{figures/experiment_flowchart_standalone.pdf}
\caption{Shared-question selection and the no-tool/BM25 answer-retrieval comparison, illustrated with a real saved MATH question.}
\end{figure}
\end{landscape}
\input{flowchart_notes.tex}
```

Logo source profiles: https://huggingface.co/Qwen , https://huggingface.co/meta-llama , https://huggingface.co/microsoft .
