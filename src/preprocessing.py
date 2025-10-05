from typing import List, Tuple
import re

DEFAULT_STOPWORDS = {
    "the","a","an","and","or","but","of","to","in","for","on","at","by","with",
    "as","is","was","were","be","been","being","from","that","this","it","its",
}

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_TOKEN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.IGNORECASE)

def sent_tokenize(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return _SENT_SPLIT.split(text)

def word_tokenize(text: str, lower: bool=True) -> List[str]:
    toks = _TOKEN.findall(text)
    return [t.lower() for t in toks] if lower else toks

def preprocess(text: str, stopwords=DEFAULT_STOPWORDS) -> Tuple[List[str], List[List[str]]]:
    """
    Returns:
      - flat token list (stopwords removed)
      - sentence-wise token lists (for /s operator)
    """
    sents = sent_tokenize(text)
    sent_tokens = []
    flat = []
    for s in sents if sents else [text]:
        toks = word_tokenize(s)
        toks = [t for t in toks if t not in stopwords]
        sent_tokens.append(toks)
        flat.extend(toks)
    return flat, sent_tokens
