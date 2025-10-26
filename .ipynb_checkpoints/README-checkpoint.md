# IR Aviation — Incident/Accident Narrative Retrieval

> Lightweight IR toolkit for aviation incident/accident narratives with Boolean/proximity search and BM25 ranking.

---



## 📁 Project Structure

```
ir_aviation/
  data/
    sample_docs.json            # (data for use for now) a few tiny narratives for smoke tests
  src/
    preprocessing.py             # tokenization, stopwords, sentence splits, normalization
    index.py                     # inverted index with positions
    boolean_query.py             # parser + Boolean/proximity (/n, /s, /p)
    bm25.py                      # BM25 ranker over the same index
    eval.py                      # MAP, nDCG, Recall@k
  main.py                    # quick end-to-end run on sample docs
  README.md
```




### What goes where

| Path                          | Purpose |
|------------------------------|---------|
| `data/sample_docs.json`     | Minimal dataset to validate the pipeline quickly. |
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
python main.py 

# Example After Load:                                                                              
[i] Loaded 41094 docs from data/sample_docs.json
[i] Built index over 41094 documents.

Type queries and press Enter.

- BM25 ranked search: just type your query text.
- Boolean/proximity search: prefix with  'bool: '  then use operators.
  Examples:
    bool: ("runway excursion" OR overrun) AND landing
    bool: (icing /5 pitot) AND (airspeed OR indications)
    bool: (approach AND mountainous) /s (cfit OR terrain)

Commands:
  :help             show this help
  :quit             exit
  :topk 20          set display cutoff



