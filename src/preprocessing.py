# src/preprocessing.py
from __future__ import annotations
import re
from typing import List, Tuple, Set

# Minimal stoplist (tweak as needed for your corpus)
STOPWORDS: Set[str] = {
    "the","a","an","and","or","but","of","to","in","for","on","at","by","with",
    "as","is","was","were","be","been","being","from","that","this","it","its",
    "into","over","after","before","than","then","so","such"
}

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_TOKEN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.IGNORECASE)

def sent_tokenize(text: str) -> List[str]:
    text = (text or "").strip()
    return [] if not text else _SENT_SPLIT.split(text)

def word_tokenize(text: str, lower: bool = True) -> List[str]:
    toks = _TOKEN.findall(text or "")
    return [t.lower() for t in toks] if lower else toks

def preprocess(text: str, stopwords: Set[str] = STOPWORDS) -> Tuple[List[str], List[List[str]]]:
    """
    Returns:
      flat_tokens: List[str]  (stopwords removed)
      sent_tokens: List[List[str]]  sentence-wise tokens (for /s)
    """
    sents = sent_tokenize(text)
    if not sents:
        toks = [t for t in word_tokenize(text) if t not in stopwords]
        return toks, [toks]
    flat, sent_tokens = [], []
    for s in sents:
        toks = [t for t in word_tokenize(s) if t not in stopwords]
        flat.extend(toks)
        sent_tokens.append(toks)
    return flat, sent_tokens

# Add to src/preprocessing.py
import html

def normalize_text(x: str) -> str:
    """
    Clean HTML entities and whitespace artifacts from NTSB text.
    - Unescape HTML
    - Replace carriage markers (&#x0D;) with spaces
    - Collapse multiple whitespace
    """
    if not x:
        return ""
    x = html.unescape(x)
    x = x.replace("\r", " ").replace("\n", " ").replace("\u00A0", " ").replace("&#x0D;", " ")
    x = " ".join(x.split())
    return x
