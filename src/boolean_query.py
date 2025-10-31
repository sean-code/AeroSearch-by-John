# src/boolean_query.py
from __future__ import annotations
import re
from typing import List, Set, Dict, Tuple
from .index import InvertedIndex

# Grammar tokens:
TOK = re.compile(r'"[^"]+"|\(|\)|/s|/p|/[0-9]+|AND|OR|NOT|[a-z0-9\-]+', re.IGNORECASE)

def tokenize_query(q: str) -> List[str]:
    return TOK.findall(q.lower())

def _docs(pl) -> Set[int]:
    return {d for d,_ in pl}

def _phrase_match(index: InvertedIndex, terms: List[str]) -> Set[int]:
    if not terms: return set()
    lists = [index.get_postings(t) for t in terms]
    if any(len(pl)==0 for pl in lists): return set()
    by_doc: Dict[int, List[List[int]]] = {}
    for pl in lists:
        for d, pos in pl:
            by_doc.setdefault(d, []).append(pos)
    out = set()
    for d, pos_lists in by_doc.items():
        if len(pos_lists) != len(terms): 
            continue
        first = pos_lists[0]
        s = set(first)
        for i, pl in enumerate(pos_lists[1:], start=1):
            s = {p for p in s if (p+i) in pl}
            if not s: break
        if s: out.add(d)
    return out

def _within_n(index: InvertedIndex, a: str, b: str, n: int) -> Set[int]:
    A = index.get_postings(a); B = index.get_postings(b)
    cand = _docs(A) & _docs(B)
    if not cand: return set()
    posA = {d: set(p) for d,p in A}
    posB = {d: set(p) for d,p in B}
    out = set()
    for d in cand:
        if any(abs(x-y) <= n for x in posA[d] for y in posB[d]):
            out.add(d)
    return out

def _same_sentence(index: InvertedIndex, a: str, b: str) -> Set[int]:
    A = index.get_postings(a); B = index.get_postings(b)
    cand = _docs(A) & _docs(B)
    if not cand: return set()
    posA = {d: set(p) for d,p in A}
    posB = {d: set(p) for d,p in B}
    out = set()
    for d in cand:
        for s,e in index.sent_bounds(d):
            if any(s<=x<=e for x in posA[d]) and any(s<=y<=e for y in posB[d]):
                out.add(d); break
    return out

def evaluate_query(index: InvertedIndex, q: str) -> List[int]:
    """
    Evaluate Boolean/proximity query with:
      AND, OR, NOT, "phrase", /n, /s, /p (paragraph≈/50), ()
    Returns sorted list of matching doc_ids.
    """
    tokens = tokenize_query(q)

    # Pack proximity triples: (left, /op, right) as atomic operands
    prox = []
    i = 0
    while i < len(tokens):
        if i+2 < len(tokens) and re.fullmatch(r'/[0-9]+|/s|/p', tokens[i+1]):
            prox.append((tokens[i], tokens[i+1], tokens[i+2]))
            i += 3
        else:
            prox.append(tokens[i]); i += 1

    def prec(t): return {'NOT':3, 'AND':2, 'OR':1}.get(t, 0)

    # Shunting yard → RPN
    output, stack = [], []
    for t in prox:
        if t == '(':
            stack.append(t)
        elif t == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if stack and stack[-1] == '(':
                stack.pop()
        elif isinstance(t, tuple):   # proximity atom
            output.append(t)
        elif str(t).upper() in ('AND','OR','NOT'):
            T = str(t).upper()
            while stack and stack[-1] != '(' and prec(stack[-1]) >= prec(T):
                output.append(stack.pop())
            stack.append(T)
        else:
            output.append(t)
    while stack: output.append(stack.pop())

    # Evaluate RPN
    def to_docs(op) -> Set[int]:
        if isinstance(op, tuple):
            a, o, b = op
            a, b = a.strip('"'), b.strip('"')
            if o == '/s': return _same_sentence(index, a, b)
            if o == '/p': return _within_n(index, a, b, 50)  # paragraph approx
            return _within_n(index, a, b, int(o[1:]))
        if isinstance(op, str) and op.startswith('"') and op.endswith('"'):
            terms = op.strip('"').split()
            return _phrase_match(index, terms)
        return _docs(index.get_postings(str(op)))

    st: List[Set[int]] = []
    for t in output:
        if t in ('AND','OR','NOT'):
            if t == 'NOT':
                b = st.pop()
                a = st.pop() if st else set()
                st.append(a - b)
            else:
                b, a = st.pop(), st.pop()
                st.append(a & b if t=='AND' else a | b)
        else:
            st.append(to_docs(t))
    return sorted(st[-1]) if st else []
