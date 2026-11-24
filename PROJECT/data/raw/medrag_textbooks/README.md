---
task_categories:
- question-answering
language:
- en
tags:
- medical
- question answering
- large language model
- retrieval-augmented generation
size_categories:
- 100K<n<1M
---
# The Textbooks Corpus in MedRAG

This HF dataset contains the chunked snippets from the Textbooks corpus used in [MedRAG](https://arxiv.org/abs/2402.13178). It can be used for medical Retrieval-Augmented Generation (RAG).

## Dataset Details

### Dataset Descriptions

[Textbooks](https://github.com/jind11/MedQA) is a collection of 18 widely used medical textbooks, which are important references for students taking the United States Medical Licensing Examination (USLME).
In MedRAG, the textbooks are processed as chunks with no more than 1000 characters. 
We used the RecursiveCharacterTextSplitter from [LangChain](https://www.langchain.com/) to perform the chunking. 
This HF dataset contains our ready-to-use chunked snippets for the Textbooks corpus, including 125,847 snippets with an average of 182 tokens.

### Dataset Structure
Each row is a snippet of Textbooks, which includes the following features:

- id: a unique identifier of the snippet
- title: the title of the textbook from which the snippet is collected
- content: the content of the snippet
- contents: a concatenation of 'title' and 'content', which will be used by the [BM25](https://github.com/castorini/pyserini) retriever

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

```shell
git clone https://huggingface.co/datasets/MedRAG/textbooks
```

### Use in MedRAG

```python
>> from src.medrag import MedRAG

>> question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
>> options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
}

>> medrag = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="Textbooks")
>> answer, snippets, scores = medrag.answer(question=question, options=options, k=32) # scores are given by the retrieval system
```

## Citation
```shell
@article{xiong2024benchmarking,
    title={Benchmarking Retrieval-Augmented Generation for Medicine}, 
    author={Guangzhi Xiong and Qiao Jin and Zhiyong Lu and Aidong Zhang},
    journal={arXiv preprint arXiv:2402.13178},
    year={2024}
}
```