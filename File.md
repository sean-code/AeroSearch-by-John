# IR Aviation — Incident/Accident Narrative Retrieval

> Lightweight IR toolkit for aviation incident/accident narratives with Boolean/proximity search and BM25 ranking.

---

## 📁 Project Structure

ir_aviation/
├─ data/
│ └─ sample_docs.jsonl # Tiny narratives for smoke tests (JSON Lines)
├─ src/
│ ├─ init.py
│ ├─ preprocessing.py # Tokenization, stopword removal, sentence splitting
│ ├─ index.py # Inverted index with positional postings
│ ├─ boolean_query.py # Parser + Boolean + proximity (/n, /s, /p) operators
│ ├─ bm25.py # BM25 ranker over the same index
│ └─ eval.py # MAP, nDCG, Recall@k metrics
├─ notebooks/
│ └─ 00_quick_demo.ipynb # (Optional) exploratory demo
├─ run_demo.py # End-to-end quick run on sample docs
└─ README.md




### What goes where

| Path                          | Purpose |
|------------------------------|---------|
| `data/sample_docs.jsonl`     | Minimal dataset to validate the pipeline quickly. |
| `src/preprocessing.py`       | Text normalization utilities (tokenize, lower, strip, stopwords, sentence split). |
| `src/index.py`               | Builds/loads the inverted index with positions (docIDs → term → positions). |
| `src/boolean_query.py`       | Parses queries and evaluates AND/OR/NOT + proximity `/n`, `/s`, `/p`. |
| `src/bm25.py`                | BM25 scorer over the indexed corpus; returns ranked results. |
| `src/eval.py`                | Evaluation helpers (MAP, nDCG, Recall@k). |
| `notebooks/00_quick_demo.ipynb` | Interactive walkthrough (optional). |
| `run_demo.py`                | Small script that wires preprocessing → indexing → query → results. |

---

## 🚀 Quick Start

```bash
# 1) Create & activate a virtual env (
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run a smoke test on the tiny dataset
python run_demo.py --data data/sample_docs.jsonl --query "engine /3 failure AND (smoke OR odor)"
