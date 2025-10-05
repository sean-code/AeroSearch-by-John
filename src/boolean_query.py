import re
from typing import List, Set, Tuple, Dict
from collections import defaultdict
from .index import InvertedIndex

# Very small parser for patterns like:
#   (icing /5 pitot) AND ("runway excursion" OR overrun) /s (landing OR rollout)
# Space alone is NOT an operator here—be explicit with AND/OR.
# You can adapt to your class’s exact grammar easily.

TOK = re.compile(r'"[^"]+"|\(|\)|/s|/p|/[0-9]+|AND|OR|NOT|[a-z0-9\-]+', re.IGNORECASE)

def tokenize_query(q: str) -> List[str]:
    return TOK.findall(q.lower())

def _postings_to_docs(postings) -> Set[int]:
    return {doc_id for doc_id, _ in postings}

def _merge_and(a: Set[int], b: Set[int]) -> Set[int]:
    return a & b

def _merge_or(a: Set[int], b: Set[int]) -> Set[int]:
    return a | b

def _merge_not(a: Set[int], b: Set[int]) -> Set[int]:
    return a - b

def _phrase_match(index: InvertedIndex, terms: List[str]) -> Set[int]:
    # consecutive positions (slop=0)
    if not terms:
        return set()
    postings_lists = [index.get_postings(t) for t in terms]
    if any(len(pl)==0 for pl in postings_lists):
        return set()
    # intersect by doc_id
    by_doc: Dict[int, List[List[int]]] = {}
    for pl in postings_lists:
        for doc_id, pos_list in pl:
            by_doc.setdefault(doc_id, []).append(pos_list)
    result = set()
    for doc_id, pos_lists in by_doc.items():
        # each term must have positions
        if len(pos_lists) != len(terms):
            continue
        # position merge: look for p, p+1, p+2...
        first_positions = pos_lists[0]
        others = pos_lists[1:]
        s = set(first_positions)
        for i, pl in enumerate(others, start=1):
            s = {p for p in s if (p+i) in pl}
            if not s: break
        if s:
            result.add(doc_id)
    return result

def _within_n(index: InvertedIndex, a_term: str, b_term: str, n: int) -> Set[int]:
    A = index.get_postings(a_term)
    B = index.get_postings(b_term)
    docsA = {d for d,_ in A}
    docsB = {d for d,_ in B}
    cand = docsA & docsB
    if not cand:
        return set()
    # map doc->positions
    posA = {d: set(p) for d,p in A}
    posB = {d: set(p) for d,p in B}
    out = set()
    for d in cand:
        pa, pb = posA[d], posB[d]
        # any |pa - pb| <= n
        close = any(abs(x - y) <= n for x in pa for y in pb)
        if close:
            out.add(d)
    return out

def _same_sentence(index: InvertedIndex, a_term: str, b_term: str) -> Set[int]:
    A = index.get_postings(a_term)
    B = index.get_postings(b_term)
    docs = {d for d,_ in A} & {d for d,_ in B}
    out = set()
    # dict of positions for quick lookup
    posA = {d: set(p) for d,p in A}
    posB = {d: set(p) for d,p in B}
    for d in docs:
        boundaries = index.sentence_boundaries(d)
        if not boundaries:
            continue
        a_pos, b_pos = posA[d], posB[d]
        for (s,e) in boundaries:
            if any(s <= x <= e for x in a_pos) and any(s <= y <= e for y in b_pos):
                out.add(d)
                break
    return out

def evaluate_query(index: InvertedIndex, q: str) -> List[int]:
    """
    Extremely simple evaluator:
    - Supports phrases "..."
    - Supports (term1 /n term2), /s, /p (as /50 approximation)
    - Combine with AND/OR/NOT and parentheses
    Returns a list of matching doc_ids (unordered set semantics).
    """
    # Convert to Reverse Polish (Shunting Yard) then evaluate
    tokens = tokenize_query(q)

    def precedence(tok):
        if tok in ('NOT',): return 3
        if tok in ('AND',): return 2
        if tok in ('OR',):  return 1
        return 0

    # preprocess proximity patterns like term /5 term into single pseudo-token
    i = 0
    prox_tokens = []
    while i < len(tokens):
        if i+2 < len(tokens) and re.fullmatch(r'/[0-9]+|/s|/p', tokens[i+1]):
            left, op, right = tokens[i], tokens[i+1], tokens[i+2]
            prox_tokens.append((left, op, right))   # pack a triple
            i += 3
        else:
            prox_tokens.append(tokens[i])
            i += 1

    # Shunting Yard
    output, stack = [], []
    for t in prox_tokens:
        if t == '(':
            stack.append(t)
        elif t == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if stack and stack[-1] == '(':
                stack.pop()
        elif isinstance(t, tuple):
            # proximity triple acts like an atomic operand → push to output
            output.append(t)
        elif t.upper() in ('AND','OR','NOT'):
            tU = t.upper()
            while stack and stack[-1] != '(' and precedence(stack[-1]) >= precedence(tU):
                output.append(stack.pop())
            stack.append(tU)
        else:
            # operand: term or "phrase"
            output.append(t)
    while stack:
        output.append(stack.pop())

    # Eval RPN
    def to_docs(operand) -> Set[int]:
        if isinstance(operand, tuple):  # proximity (left, op, right)
            a, op, b = operand
            if op.startswith('/'):
                if op == '/s':
                    return _same_sentence(index, a.strip('"'), b.strip('"'))
                elif op == '/p':
                    # approximate paragraph as within 50 tokens
                    return _within_n(index, a.strip('"'), b.strip('"'), 50)
                else:
                    n = int(op[1:])
                    return _within_n(index, a.strip('"'), b.strip('"'), n)
        if operand.startswith('"') and operand.endswith('"'):
            terms = operand.strip('"').split()
            return _phrase_match(index, terms)
        # simple term
        return {d for d,_ in index.get_postings(operand)}

    stack2: List[Set[int]] = []
    for t in output:
        if t in ('AND','OR','NOT'):
            if t == 'NOT':
                b = stack2.pop()
                a = stack2.pop() if stack2 else set()
                stack2.append(_merge_not(a,b))
            else:
                b, a = stack2.pop(), stack2.pop()
                stack2.append(_merge_and(a,b) if t=='AND' else _merge_or(a,b))
        else:
            stack2.append(to_docs(t))
    return sorted(stack2[-1]) if stack2 else []
