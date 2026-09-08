# Enterprise QA — Aurelia Bank Synthetic Research Corpus

> A full end-to-end **extractive Question Answering** research pipeline built on a synthetic banking corpus. Covers document ingestion, semantic retrieval, model fine-tuning, and rigorous evaluation — all locally, without any cloud dependencies.

---

## Table of Contents

- [Overview](#overview)
- [Project Highlights](#project-highlights)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Corpus & Dataset](#corpus--dataset)
- [Pipeline Phases](#pipeline-phases)
- [Results](#results)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Model Details](#model-details)
- [Evaluation Metrics](#evaluation-metrics)
- [File Reference](#file-reference)
- [Design Decisions](#design-decisions)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

**Enterprise QA** is a research-grade pipeline that demonstrates how to build a domain-adapted, extractive QA system entirely from scratch — from raw documents to a fine-tuned model with measurable quality improvements.

The corpus is the **Aurelia Bank Synthetic Research Corpus**: 10 synthetic internal banking policy documents (AsciiDoc format) covering retail banking, home and personal loans, corporate lending, KYC/AML compliance, credit risk, digital banking, and financial regulation. The fictional Aurelia Bank context ensures the data is safe to publish and experiment with without privacy concerns.

**Key capabilities demonstrated:**
- AsciiDoc ingestion and structured section extraction
- Dense semantic retrieval with `BAAI/bge-small-en-v1.5`
- Extractive QA with `deepset/tinyroberta-squad2` (base and fine-tuned)
- Character-level answer span validation and token-alignment
- Document-stratified train/validation/test splits
- Quantitative retrieval and QA evaluation with saved JSON reports

---

## Project Highlights

| Metric | Baseline (tinyRoBERTa-SQuAD2) | Fine-Tuned (Aurelia QA) | Improvement |
|---|---|---|---|
| **Exact Match** | 50.00% | **86.36%** | **+36.36 pp** |
| **Token F1** | 78.91% | **95.37%** | **+16.46 pp** |

| Retrieval Metric | Score |
|---|---|
| **Recall@1** | 63.64% (14/22) |
| **Recall@3** | 86.36% (19/22) |
| **Recall@5** | 90.91% (20/22) |

> All results measured on a **frozen test set of 22 examples** across 10 documents.

---

## Architecture

```
Raw AsciiDoc Documents (data/raw/)
         |
         v
 +----------------------+
 |  Document Processing |  data/process_documents.py
 |  - Parse AsciiDoc    |  -> sections.jsonl
 |  - Clean text        |  -> corpus.json
 |  - Assign record IDs |
 +----------------------+
         |
         v
 +----------------------+
 |  Semantic Embedding  |  retrieval/embed.py
 |  - BAAI/bge-small    |  -> embeddings.npy
 |  - 384-dim vectors   |  -> embedding_metadata.json
 |  - L2-normalized     |
 +----------------------+
         |
         v
 +----------------------+
 |  Semantic Search     |  retrieval/search.py
 |  - Cosine similarity |  (interactive REPL)
 |  - Top-K retrieval   |
 +----------------------+

 QA Dataset (data/qa_dataset/)
         |
         v
 +----------------------+
 |  Data Preparation    |  qa/prepare_qa_data.py
 |  - Validate spans    |  qa/tokenize_qa.py
 |  - Tokenize (512 tok)|  -> train / validation / test
 |  - Sliding windows   |
 +----------------------+
         |
         v
 +----------------------+
 |  Fine-Tuning         |  qa/train.py
 |  - tinyRoBERTa       |  -> models/tinyroberta-aurelia-qa/
 |  - 3 epochs, lr=2e-5 |
 +----------------------+
         |
         v
 +----------------------+
 |  Evaluation          |  qa/evaluate_qa.py
 |  - Baseline vs FT    |  -> evaluation/qa_evaluation.json
 |  - EM + Token F1     |
 +----------------------+
```

---

## Repository Structure

```
enterprise--qa/
|
+-- data/
|   +-- raw/                          # 10 AsciiDoc source documents
|   |   +-- 01_retail_banking_policy.adoc
|   |   +-- 02_home_loan_product_manual.adoc
|   |   +-- 03_personal_loan_product_manual.adoc
|   |   +-- 04_corporate_loan_policy.adoc
|   |   +-- 05_customer_onboarding_policy.adoc
|   |   +-- 06_kyc_aml_policy.adoc
|   |   +-- 07_interest_rate_fee_policy.adoc
|   |   +-- 08_credit_risk_eligibility_policy.adoc
|   |   +-- 09_financial_compliance_regulation.adoc
|   |   +-- 10_digital_banking_transaction_policy.adoc
|   |
|   +-- processed/                    # Generated - gitignored
|   |   +-- sections.jsonl            # 154 structured section records
|   |   +-- corpus.json               # Corpus-level metadata
|   |   +-- embeddings.npy            # 154 x 384 float32 matrix
|   |   +-- embedding_metadata.json   # Vector -> record mapping
|   |
|   +-- process_documents.py          # AsciiDoc parser + JSONL writer
|   +-- validate_processed_data.py    # Post-processing validation
|   |
|   +-- qa_dataset/
|       +-- split_qa_dataset.py       # Document-stratified splitter
|       +-- processed/                # Generated JSONL splits + tokenized datasets
|
+-- retrieval/
|   +-- embed.py                      # Generate + save BGE embeddings
|   +-- search.py                     # Interactive semantic search REPL
|   +-- evaluate_retrieval.py         # Recall@K evaluation
|   +-- index.py                      # (placeholder)
|
+-- qa/
|   +-- prepare_qa_data.py            # Validate QA spans
|   +-- tokenize_qa.py                # Token-align spans, sliding windows
|   +-- train.py                      # Hugging Face Trainer fine-tuning
|   +-- baseline.py                   # Baseline model evaluation
|   +-- evaluate_qa.py                # Base vs fine-tuned comparison
|   +-- apply_answer_corrections.py   # Span correction utilities
|   +-- check_token_boundaries.py     # Token boundary debugging
|   +-- inspect_tokenized_data.py     # Tokenized data inspector
|   +-- verify_exact_token_alignment.py
|   +-- verify_token_alignment.py
|
+-- evaluation/
|   +-- retrieval_evaluation.json     # Saved Recall@K results
|   +-- qa_evaluation.json            # Saved EM + F1 comparison
|   +-- baseline_predictions.json     # Per-example baseline predictions
|   +-- config.py                     # (reserved)
|
+-- models/
|   +-- tinyroberta-aurelia-qa/       # Fine-tuned model - gitignored
|       +-- model.safetensors         # ~311 MB
|       +-- config.json
|       +-- tokenizer.json
|       +-- train_results.json
|       +-- validation_results.json
|
+-- requirements.txt
+-- .gitignore
+-- README.md
```

---

## Corpus & Dataset

### Source Documents

The **Aurelia Bank Synthetic Research Corpus** consists of 10 structured AsciiDoc policy documents representing a realistic internal banking knowledge base. All names, figures, and regulatory references are synthetic and fictional.

| # | Document | Policy ID |
|---|---|---|
| 1 | Retail Banking Policy | AUR-RB-001 |
| 2 | Home Loan Product Manual | AUR-HL-002 |
| 3 | Personal Loan Product Manual | AUR-PL-003 |
| 4 | Corporate Loan Policy | AUR-CL-004 |
| 5 | Customer Onboarding Policy | AUR-OB-005 |
| 6 | KYC and AML Policy | AUR-KYC-006 |
| 7 | Interest Rate and Fee Policy | AUR-IRF-007 |
| 8 | Credit Risk and Eligibility Policy | AUR-CR-008 |
| 9 | Financial Compliance Regulation | AUR-FC-009 |
| 10 | Digital Banking and Transaction Policy | AUR-DB-010 |

### Processed Corpus Statistics

| Metric | Value |
|---|---|
| Source documents | 10 |
| Total section records | 154 |
| Embedding dimension | 384 |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Embeddings normalized | Yes (L2 unit-length) |

### QA Dataset Splits

Each document contributes 15 extractive QA examples. Splits are **document-stratified** (every document appears in every split).

| Split | Examples | Ratio |
|---|---|---|
| Train | ~105 | 70% |
| Validation | ~23 | 15% |
| **Test** | **22** | **15%** |

Each QA example contains:
- `id` — unique identifier (e.g. `01_retail_banking_policy__qa_002`)
- `document_id` — source document
- `context` — the passage containing the answer
- `question` — the question
- `answers.text[0]` — the exact answer string
- `answers.answer_start[0]` — character offset in context

---

## Pipeline Phases

### Phase 1 — Raw Document Corpus

The 10 AsciiDoc files reside in `data/raw/`. Each follows a consistent header structure:
- Document title (level-1 heading)
- Organization: `Aurelia Bank`
- `:doctype:` attribute
- `Policy ID:`, `Version:`, `Classification:` metadata fields
- Level-2+ sections containing policy content

### Phase 2 — Document Processing

**Script:** `data/process_documents.py`

Parses all `.adoc` files and produces a structured corpus:

1. **Metadata extraction** — Reads `Policy ID`, `Version`, `Classification`, organization, and `doctype` from each document header.
2. **Section parsing** — Walks headings (level 2-4), accumulates body text per section, and builds a hierarchical `title_path`.
3. **Text cleaning** — Strips trailing whitespace, collapses excessive blank lines, removes AsciiDoc continuation markers. Does **not** paraphrase, lemmatize, or otherwise change semantic content.
4. **Usability filtering** — Drops records with empty text, no alphanumeric content, or an alphanumeric character ratio below 0.20 (markup-only sections).
5. **Deterministic record IDs** — Assigned after filtering so IDs are stable across re-runs: `{document_id}__record_{N:03d}`.
6. **JSONL serialization** — All records written to `data/processed/sections.jsonl` (one JSON object per line).
7. **Corpus metadata** — Summary statistics written to `data/processed/corpus.json`.

**Output fields per record:**

```json
{
  "record_id": "01_retail_banking_policy__record_003",
  "document_id": "01_retail_banking_policy",
  "document_title": "Retail Banking Policy",
  "organization": "Aurelia Bank",
  "doctype": "article",
  "source_file": "01_retail_banking_policy.adoc",
  "metadata": {
    "policy_id": "AUR-RB-001",
    "version": "3.2",
    "classification": "Internal - Employee Use"
  },
  "heading_level": 2,
  "section": "Definitions",
  "subsection": null,
  "subsubsection": null,
  "title_path": ["Definitions"],
  "text": "Standard Retail Customer (SRC):: ..."
}
```

```bash
python data/process_documents.py
```

### Phase 3 — Semantic Retrieval

#### Phase 3B — Embedding Generation

**Script:** `retrieval/embed.py`

- Loads all 154 section records from `sections.jsonl`.
- Constructs embedding text: prepends the section hierarchy (`Section: X > Y\nContent: ...`) to provide contextual signal.
- Encodes using `BAAI/bge-small-en-v1.5` (384-dimensional, locally downloaded via `sentence-transformers`).
- L2-normalizes all vectors so cosine similarity reduces to a dot product.
- Saves the embedding matrix to `data/processed/embeddings.npy` and a vector-to-record mapping to `embedding_metadata.json`.

```bash
python retrieval/embed.py
```

#### Phase 3C — Interactive Semantic Search

**Script:** `retrieval/search.py`

Launches an interactive REPL for ad-hoc semantic queries against the corpus:

```
Query: What are the KYC requirements for non-resident customers?

SEMANTIC RETRIEVAL RESULTS
--------------------------
Rank       : 1
Similarity : 0.823456
Record ID  : 06_kyc_aml_policy__record_005
Document   : KYC and AML Policy
Section    : Enhanced Due Diligence
...
```

```bash
python retrieval/search.py
```

#### Phase 3D — Retrieval Evaluation

**Script:** `retrieval/evaluate_retrieval.py`

Evaluates Recall@K on the frozen test set by:
1. Mapping each test QA example to its **gold source record** (substring match within the same document, preferring the most specific/shortest record).
2. Embedding the test questions with BGE.
3. Retrieving the top-5 records per query.
4. Computing Recall@1, @3, and @5.

```bash
python retrieval/evaluate_retrieval.py
```

### Phase 4 — QA Dataset Construction

**Script:** `data/qa_dataset/split_qa_dataset.py`

- Reads `all_examples.jsonl` (the combined, manually validated QA dataset).
- Applies a **document-stratified split** (70 / 15 / 15) using the largest-remainder method to ensure each document is represented in every split.
- Writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and a `split_report.json`.
- Uses `SEED = 42` for full reproducibility.

### Phase 5 — Model Fine-Tuning

#### Phase 5B-1 — Data Validation

**Script:** `qa/prepare_qa_data.py`

Validates every QA record across all splits:
- Checks for required fields (`id`, `document_id`, `context`, `question`, `answers`).
- Verifies `answers.text` is exactly one element.
- Confirms `answer_start` is a non-negative integer.
- **Crucially**: confirms `context[answer_start : answer_start + len(answer_text)] == answer_text` — exact character-level span integrity.

```bash
python qa/prepare_qa_data.py
```

#### Phase 5B-2 — Tokenization

**Script:** `qa/tokenize_qa.py`

- Tokenizes all three splits using `deepset/tinyroberta-squad2`'s fast tokenizer (required for `offset_mapping`).
- `MAX_LENGTH = 512`, `DOC_STRIDE = 128` (overlapping sliding windows for long contexts).
- Maps character-level `answer_start`/`answer_end` to exact **token-level** `start_positions`/`end_positions` using token offset overlap.
- Windows that do not fully contain the answer are marked `has_answer=False` and receive `start_positions=0, end_positions=0`.
- Saves each split to `data/qa_dataset/processed/tokenized/{train,validation,test}` using Hugging Face `Dataset.save_to_disk()`.

```bash
python qa/tokenize_qa.py
```

#### Phase 5B-4 — Fine-Tuning

**Script:** `qa/train.py`

Fine-tunes `deepset/tinyroberta-squad2` on the Aurelia Bank train split:

| Hyperparameter | Value |
|---|---|
| Base model | `deepset/tinyroberta-squad2` |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Train batch size | 8 |
| Eval batch size | 8 |
| Weight decay | 0.01 |
| Seed | 42 |
| Eval strategy | per epoch |
| Save strategy | per epoch (keep best 2) |
| Report to | none (fully local) |

The best checkpoint (by validation loss) is loaded at the end. Model is saved to `models/tinyroberta-aurelia-qa/`.

```bash
python qa/train.py
```

**Training summary:**

| Metric | Value |
|---|---|
| Final train loss | 0.3159 |
| Final validation loss | 0.2483 |
| Runtime | ~762 seconds (~12.7 min) |
| Throughput | 0.414 samples/sec |

### Phase 6 — Evaluation

#### Phase 6A — Baseline Evaluation

**Script:** `qa/baseline.py`

Runs the unmodified `deepset/tinyroberta-squad2` on the frozen test set, saves per-example predictions to `evaluation/baseline_predictions.json`.

```bash
python qa/baseline.py
```

#### Phase 6B-6C — Model Comparison

**Script:** `qa/evaluate_qa.py`

Evaluates both the baseline and fine-tuned models side-by-side on the frozen test set. Reports:
- Exact Match (EM) — normalized string equality
- Token F1 — token-overlap F1 using normalized token counts

Writes the full comparison report to `evaluation/qa_evaluation.json`.

```bash
python qa/evaluate_qa.py
```

---

## Results

### QA Performance (22 test examples)

| Model | Exact Match | Token F1 |
|---|---|---|
| `deepset/tinyroberta-squad2` (baseline) | 50.00% | 78.91% |
| `tinyroberta-aurelia-qa` (fine-tuned) | **86.36%** | **95.37%** |
| **Improvement** | **+36.36 pp** | **+16.46 pp** |

The fine-tuned model achieves near-perfect scores on domain-specific questions (e.g., exact regulatory figures, defined terms, named persons) that the base model answers partially or incorrectly.

### Retrieval Performance (BGE-small, 22 test questions, 154 corpus records)

| Metric | Value |
|---|---|
| Recall@1 | 63.64% (14/22) |
| Recall@3 | 86.36% (19/22) |
| Recall@5 | 90.91% (20/22) |

The retrieval component successfully surfaces the correct source section within the top-5 results for over 90% of test questions — making it a viable first-stage retriever for a full RAG pipeline.

---

## Installation

### Prerequisites

- Python 3.10+
- `pip` (or `uv`/`conda`)
- A CUDA-capable GPU is **optional** but speeds up fine-tuning significantly. CPU training is supported.

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/enterprise--qa.git
cd enterprise--qa

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch
transformers
datasets
evaluate
accelerate
numpy
sentence-transformers
```

> **Note:** PyTorch is installed without a CUDA variant by default. For GPU support, install the appropriate CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Running the Pipeline

Run the scripts in the following order. Steps 1-3 generate the processed corpus and embeddings. Steps 4-7 prepare and fine-tune the QA model. Steps 8-11 evaluate everything.

```bash
# Step 1 — Process raw AsciiDoc documents
python data/process_documents.py

# Step 2 — Validate processed data
python data/validate_processed_data.py

# Step 3 — Generate sentence embeddings
python retrieval/embed.py

# Step 4 — (Optional) Interactive semantic search
python retrieval/search.py

# Step 5 — Evaluate retrieval quality
python retrieval/evaluate_retrieval.py

# Step 6 — Split QA dataset (if all_examples.jsonl exists)
python data/qa_dataset/split_qa_dataset.py

# Step 7 — Validate QA answer spans
python qa/prepare_qa_data.py

# Step 8 — Tokenize QA datasets
python qa/tokenize_qa.py

# Step 9 — Fine-tune the model
python qa/train.py

# Step 10 — Evaluate baseline
python qa/baseline.py

# Step 11 — Compare baseline vs fine-tuned
python qa/evaluate_qa.py
```

> **Important:** `data/processed/` and `models/` are gitignored (they contain large binary files). You must run the pipeline to regenerate them locally.

---

## Model Details

### Retrieval Model — `BAAI/bge-small-en-v1.5`

- Architecture: Encoder-only transformer
- Embedding dimension: 384
- Normalization: L2 unit-length (cosine similarity = dot product)
- Downloaded automatically by `sentence-transformers` on first run
- Embedding text format: `"Section: {hierarchy}\nContent: {text}"`

### QA Model — `deepset/tinyroberta-squad2` -> `tinyroberta-aurelia-qa`

- Architecture: `RobertaForQuestionAnswering` (6 layers, 768 hidden, 12 heads)
- Vocabulary size: 50,265
- Max position embeddings: 514
- Pre-trained on SQuAD 2.0 by deepset
- Fine-tuned for 3 epochs on Aurelia Bank domain data
- Task: **Extractive span prediction** (start + end token logits)
- Inference: `truncation="only_second"`, `max_length=512`

---

## Evaluation Metrics

### Exact Match (EM)

```
EM = 1  if  normalize(prediction) == normalize(reference)
EM = 0  otherwise
```

where `normalize` lowercases and collapses whitespace.

### Token F1

Token-overlap F1 between the normalized prediction and reference:

```
precision = common_tokens / len(prediction_tokens)
recall    = common_tokens / len(reference_tokens)
F1        = 2 * precision * recall / (precision + recall)
```

### Recall@K (Retrieval)

```
Recall@K = (# queries where gold record appears in top-K results) / (total queries)
```

The gold record for each QA example is the structured section record whose text contains the QA context (substring match), restricted to the same document, preferring the shortest (most specific) match.

---

## File Reference

| File | Phase | Purpose |
|---|---|---|
| `data/process_documents.py` | 2 | AsciiDoc parser, JSONL writer, corpus metadata |
| `data/validate_processed_data.py` | 2 | Post-processing integrity checks |
| `data/qa_dataset/split_qa_dataset.py` | 4 | Document-stratified train/val/test split |
| `retrieval/embed.py` | 3B | BGE embedding generation |
| `retrieval/search.py` | 3C | Interactive semantic search REPL |
| `retrieval/evaluate_retrieval.py` | 3D | Recall@K evaluation |
| `retrieval/index.py` | — | Placeholder |
| `qa/prepare_qa_data.py` | 5B-1 | Character-level span validation |
| `qa/tokenize_qa.py` | 5B-2 | Token alignment + sliding window tokenization |
| `qa/train.py` | 5B-4 | Hugging Face Trainer fine-tuning |
| `qa/baseline.py` | 6A | Baseline model inference + metrics |
| `qa/evaluate_qa.py` | 6B-6C | Baseline vs fine-tuned comparison |
| `qa/apply_answer_corrections.py` | — | Answer span correction utilities |
| `qa/check_token_boundaries.py` | — | Token boundary debugging |
| `qa/inspect_tokenized_data.py` | — | Tokenized dataset inspection |
| `qa/verify_exact_token_alignment.py` | — | Strict token alignment verification |
| `qa/verify_token_alignment.py` | — | Token alignment verification |
| `evaluation/retrieval_evaluation.json` | 3D | Saved retrieval results |
| `evaluation/qa_evaluation.json` | 6C | Saved QA comparison results |
| `evaluation/baseline_predictions.json` | 6A | Per-example baseline predictions |
| `models/tinyroberta-aurelia-qa/` | 5B-4 | Fine-tuned model weights + tokenizer |

---

## Design Decisions

**Why AsciiDoc?**
AsciiDoc supports rich structured markup (headings, definitions, attributes) without requiring a full XML parser. The heading-level hierarchy directly maps to the section/subsection/subsubsection fields used throughout the pipeline.

**Why conservative text cleaning?**
The pipeline deliberately avoids paraphrasing, lowercasing, stemming, or lemmatizing source text. Answer spans are character-level offsets into the original text — any normalization of the context would invalidate those offsets.

**Why document-stratified splits?**
With only 10 source documents, a random split risks putting entire documents in only one split. Stratification ensures every document is represented in train, validation, and test, making metrics more reliable and domain coverage consistent.

**Why tinyRoBERTa?**
`deepset/tinyroberta-squad2` is a 6-layer RoBERTa pre-trained on SQuAD 2.0 — a strong, fast baseline for extractive QA that runs on CPU in reasonable time. Its compact size (~82M parameters) makes local fine-tuning feasible without a GPU.

**Why BGE-small for retrieval?**
`BAAI/bge-small-en-v1.5` produces high-quality 384-dimensional embeddings with an excellent quality/size tradeoff. Because all vectors are L2-normalized at generation time, search reduces to a single matrix-vector dot product — no FAISS or ANN index required at this corpus size.

**Why no external experiment tracking?**
`report_to="none"` and `push_to_hub=False` keep the entire experiment local. This is intentional for a research project using a synthetic corpus — nothing is uploaded or logged externally.

---

## Limitations & Future Work

- **Corpus size:** 10 documents and ~150 QA examples is sufficient for demonstrating fine-tuning gains, but too small for production. The pipeline scales directly to larger corpora.
- **Generative QA:** The current pipeline is purely extractive. Adding a generative reader (e.g., a small LLM for answer synthesis) would enable abstractive answers.
- **Full RAG pipeline:** The retrieval and QA components are currently evaluated independently. Wiring them together end-to-end (retrieve -> read) is a natural next step.
- **ANN indexing:** At 154 records, brute-force cosine similarity is fast enough. For corpora with millions of records, `retrieval/index.py` should implement FAISS or similar ANN indexing.
- **Answer correction utilities:** Several scripts (`apply_answer_corrections.py`, `verify_*`) document the iterative process of fixing span alignment issues in the QA dataset. These would be removed or consolidated in a production codebase.
- **GPU training:** The fine-tuning run was measured at ~762 seconds on CPU (0.414 samples/sec). A modern GPU would reduce this to under 2 minutes.

---

## License

This project uses a **synthetic corpus** created solely for research purposes. The Aurelia Bank name, all policy content, figures, and regulatory references are entirely fictional. There are no real customer data, real banking policies, or real regulatory obligations represented.

---

*Built with PyTorch · Hugging Face Transformers · Sentence Transformers · NumPy*
