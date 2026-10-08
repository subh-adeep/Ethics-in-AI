# Ethics in AI - Research Papers Collection

This directory contains parsed Markdown versions of academic research papers on AI alignment, sycophancy, and evaluation, complete with extracted figures, diagrams, and formatted tables.

## Documents Index

| Document | Title | Pages | Extracted Images | Markdown File |
| :--- | :--- | :---: | :---: | :--- |
| [2310.13548v4.pdf](./2310.13548v4.pdf) | **TOWARDS UNDERSTANDING SYCOPHANCY IN LANGUAGE MODELS** | 35 | 20 ([folder](./2310.13548v4_images/)) | [2310.13548v4.md](./2310.13548v4.md) |
| [2502.08177v4.pdf](./2502.08177v4.pdf) | **SycEval: Evaluating LLM Sycophancy** | 8 | 6 ([folder](./2502.08177v4_images/)) | [2502.08177v4.md](./2502.08177v4.md) |
| [2604.22193v1 (1).pdf](./2604.22193v1 (1).pdf) | **How Large Language Models Balance Internal Knowledge with User and Document Assertions** | 24 | 20 ([folder](./2604.22193v1_-1-_images/)) | [2604.22193v1 (1).md](./2604.22193v1 (1).md) |
| [2604.22193v1.pdf](./2604.22193v1.pdf) | **How Large Language Models Balance Internal Knowledge with User and Document Assertions** | 24 | 20 ([folder](./2604.22193v1_images/)) | [2604.22193v1.md](./2604.22193v1.md) |
| [2606.08451v1.pdf](./2606.08451v1.pdf) | **Sycophancy as a Multilingual Alignment Failure: How Safety Degrades Across Languages, Topics, and Models** | 19 | 13 ([folder](./2606.08451v1_images/)) | [2606.08451v1.md](./2606.08451v1.md) |

## Directory Structure

```text
Documents/Papers/
├── 2310.13548v4.pdf
├── 2310.13548v4.md
├── 2310.13548v4_images/
├── 2502.08177v4.pdf
├── 2502.08177v4.md
├── 2502.08177v4_images/
├── 2604.22193v1 (1).pdf
├── 2604.22193v1 (1).md
├── 2604.22193v1_-1-_images/
├── 2604.22193v1.pdf
├── 2604.22193v1.md
├── 2604.22193v1_images/
├── 2606.08451v1.pdf
├── 2606.08451v1.md
├── 2606.08451v1_images/
└── README.md
```

## Parsing Details
- **Parser Engine**: PyMuPDF4LLM with PyMuPDF Layout (ONNX layout analyzer)
- **Image Resolution**: 150 DPI PNG
- **Image Linking**: Relative Markdown syntax `![...](<paper_stem>_images/...)` compatible with GitHub, Obsidian, and VS Code.
- **Encoding**: UTF-8
