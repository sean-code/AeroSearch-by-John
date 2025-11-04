# AeroSearch — Incident/Accident Narrative Retrieval

> Lightweight IR toolkit for aviation incident/accident narratives with Boolean/proximity search and BM25 ranking.
> Comprises Narratives Between 01/01/2019 and 11/01/2025 : as Referenced by NTSB

---



## 📁 Project Structure

```
ir_aviation/
  data/
    sample_docs.json            # tiny narratives for smoke tests (JSON array of objects)
  src/
    preprocessing.py            # tokenization, stopwords, sentence splits, normalization
    index.py                    # inverted index with positions (+ sentence bounds)
    boolean_query.py            # parser + Boolean/proximity (/n, /s, /p)
    bm25.py                     # BM25 ranker over the same index
    eval.py                     # MAP, nDCG, Recall@k
  main.py                       # end-to-end CLI with interactive search (REPL)
  app.py                       # UI Search platform
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
| `main.py`                | CLI loader (JSON/CSV), index builder, interactive search loop with pretty printing. |
| `app.py`                | User Interface by Streamlit- comprises index builder, interactive search and much more |

---

## 🚀 Quick Start

```bash
# (optional) create a venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Run a smoke test (auto-detects data/sample_docs.json if present)
python main.py


# Example After Load:                                                                              
``` 
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
```

Happy searching! ✈️

