CODE LINIK : https://colab.research.google.com/drive/1fPAEiVf6ldzq3sneEI2Mcpm_Bz1bD-7U?usp=sharing

PDF LINK: https://arxiv.org/pdf/2109.00122

DATASET LINK: https://www.kaggle.com/datasets/visalakshiiyer/question-answering-financial-data/data


---

# 🧠 FinQA: Program Generation for Numerical Reasoning over Financial Text

FinQA (Financial Question Answering) is a system that performs numerical reasoning over financial documents such as earnings reports. It generates **programs** (step-by-step symbolic operations) to answer complex numerical questions that require arithmetic, logic, and contextual understanding.

This repository implements a complete **Retriever + Generator + Executor** pipeline inspired by the [FinQA paper](https://arxiv.org/abs/2110.10592), with enhancements for modularity and extensibility.

---

## 📌 Table of Contents

* [Project Overview](#project-overview)
* [Pipeline Architecture](#pipeline-architecture)
* [Components](#components)

  * [1. Retriever](#1-retriever)
  * [2. Generator](#2-generator)
  * [3. Executor](#3-executor)
* [Dataset](#dataset)
* [Training & Evaluation](#training--evaluation)
* [Results](#results)
* [Model Enhancements](#model-enhancements)
* [References](#references)

---

## 📄 Project Overview

The goal of FinQA is to answer financial questions that require reasoning over:

* Financial **tables** (structured data)
* **Pre-text** and **post-text** paragraphs (unstructured financial text)

Instead of directly predicting the answer, the model **generates a reasoning program**, executes it over the document, and derives the final result.

---

## 🏗️ Pipeline Architecture

## Machine Learning Lifecycle 
```
 ┌────────────┐     ┌───────────────┐     ┌────────────┐     ┌────────────┐
 │ Financial  │     │    Retriever  │     │  Generator │     │  Executor  │
 │ Document   │───▶ │ (BERT-based)  │───▶ │ (LSTM + BERT)│───▶│ Math Ops   │───▶Final Answer
 └────────────┘     └───────────────┘     └────────────┘     └────────────┘
                                                    
```

1. **Retriever**: Selects relevant sentences/tables from the document.
2. **Generator**: Generates a step-by-step reasoning program.
3. **Executor**: Executes the program over the table/text values to get the final answer.

---

## 🧩 Components

### 1. Retriever

* **Model**: BERT/RoBERTa-based encoder trained for sentence-level relevance classification.
* **Inputs**: Full document (pre-text, table, post-text) and question.
* **Outputs**: Binary labels for each sentence/table cell indicating its relevance.
* **Loss Function**: Binary Cross-Entropy

**Enhancements**:

* Option to use cosine similarity + neural classifier
* Handles table row/column vectorization

---

### 2. Generator

* **Encoder**: BERT-based contextual encoder

* **Decoder**: LSTM-based program generator

* **Vocabulary**:

  * Reserved operations: `add`, `subtract`, `multiply`, `divide`, `greater`, etc.
  * DSL tokens: intermediate variables `#0`, `#1`, ...
  * Table values and numeric constants from context

* **Step Memory**: Tracks previously generated operations (`#0`, `#1`) for multi-step reasoning

* **Training**:

  * Teacher forcing with program supervision
  * Cross-entropy loss over DSL tokens

* **Output**: List of DSL operations (program)

**Example**:

```python
Program: [select(table_value_1), select(table_value_2), add(#0, #1)]
```

---

### 3. Executor

* Symbolic interpreter for the DSL
* Executes operations like `add`, `subtract`, `select`, `greater`, etc.
* Supports:

  * Numeric precision
  * Intermediate variable storage
  * Type checking for safe math operations

---

## 📊 Dataset

* Based on the official [FinQA dataset](https://github.com/czyssrs/FinQA)

* JSON structure per sample:

  ```json
  {
    "pre_text": "...",
    "post_text": "...",
    "table": [["Header1", "Header2", ...], ["Row1", "Row2", ...], ...],
    "qa": "What is the revenue growth?",
    "program": ["select(...)", "divide(...)", "subtract(...)"],
    "gold_inds": [sentence indices used by retriever]
  }
  ```

* Custom preprocessing steps include:

  * Tokenization
  * Sentence splitting
  * Table value indexing
  * Operation mapping

---
---

## 📈 Results

| Component | Accuracy                       | Notes                     |
| --------- | ------------------------------ | ------------------------- |
| Retriever | \~84% (sentence-level F1)      | BERT-based                |
| Generator | \~68% (program match accuracy) | LSTM decoder              |
| Executor  | 100% (symbolic interpreter)    | Executes correct programs |

---

## 🔧 Model Enhancements

* ✅ Custom tokenizer and vocabulary expansion
* ✅ Integration of table schema as headers
* ✅ Attention over both table and text
* ✅ Robust step memory handling
* ✅ Support for program templates
* ✅ DSL support for numerical + logical ops

---

## 📚 References

* [FinQA: Financial Question Answering via Program Generation](https://arxiv.org/abs/2110.10592)
* [Official FinQA GitHub Repo](https://github.com/czyssrs/FinQA)

---
## MODEL
![Screenshot 2025-05-28 212430](https://github.com/user-attachments/assets/57894288-2de9-4aa3-b8ee-49bc25f1a5d5)
## infer
![Screenshot 2025-05-28 210849](https://github.com/user-attachments/assets/d592e4a5-c50d-478c-94c9-4610f93b0118)
