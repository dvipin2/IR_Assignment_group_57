"""
Core IR data structures and algorithms for the CISI assignment.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Dict, List, Optional, Set, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# NLTK bootstrap
# ---------------------------------------------------------------------------

def ensure_nltk():
    import ssl
    packages = [
        ("tokenizers/punkt_tab",              "punkt_tab"),
        ("tokenizers/punkt",                  "punkt"),
        ("corpora/stopwords",                 "stopwords"),
        ("corpora/wordnet",                   "wordnet"),
        ("corpora/omw-1.4",                   "omw-1.4"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    _orig = getattr(ssl, "_create_default_https_context", None)
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass
    for find_path, pkg_name in packages:
        try:
            nltk.data.find(find_path)
        except LookupError:
            try:
                nltk.download(pkg_name, quiet=True)
            except Exception:
                pass
    if _orig is not None:
        ssl._create_default_https_context = _orig

# download nltk
ensure_nltk()

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words("english"))


def tokenize(text: str) -> List[str]:
    return word_tokenize(text)


def lowercase(tokens: List[str]) -> List[str]:
    return [t.lower() for t in tokens]


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _stopwords]


def handle_hyphens(tokens: List[str]) -> List[str]:
    result = []
    for t in tokens:
        if "-" in t:
            parts = t.split("-")
            result.extend(parts)
            result.append(t.replace("-", ""))
        else:
            result.append(t)
    return result


def keep_alpha(tokens: List[str]) -> List[str]:
    return [t for t in tokens if re.match(r"^[a-zA-Z]+$", t)]


def stem_tokens(tokens: List[str]) -> List[str]:
    return [_stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    return [_lemmatizer.lemmatize(t) for t in tokens]


def full_preprocess(text: str, use_stemming: bool = True) -> List[str]:
    tokens = tokenize(text)
    tokens = lowercase(tokens)
    tokens = handle_hyphens(tokens)
    tokens = keep_alpha(tokens)
    tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = stem_tokens(tokens)
    else:
        tokens = lemmatize_tokens(tokens)
    return tokens


# ---------------------------------------------------------------------------
# Inverted Index
# ---------------------------------------------------------------------------

class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Set[int]] = defaultdict(set)

    def build(self, docs: dict, use_stemming: bool = True):
        self.index.clear()
        for doc_id, doc in docs.items():
            tokens = full_preprocess(doc.text, use_stemming=use_stemming)
            for token in tokens:
                self.index[token].add(doc_id)

    def search(self, query: str, use_stemming: bool = True) -> Set[int]:
        tokens = full_preprocess(query, use_stemming=use_stemming)
        if not tokens:
            return set()
        result = self.index.get(tokens[0], set()).copy()
        for t in tokens[1:]:
            result &= self.index.get(t, set())
        return result


# ---------------------------------------------------------------------------
# Biword Index
# ---------------------------------------------------------------------------

class BiwordIndex:
    def __init__(self):
        self.index: Dict[str, Set[int]] = defaultdict(set)

    def build(self, docs: dict):
        self.index.clear()
        for doc_id, doc in docs.items():
            tokens = full_preprocess(doc.text, use_stemming=False)
            for i in range(len(tokens) - 1):
                biword = f"{tokens[i]}_{tokens[i+1]}"
                self.index[biword].add(doc_id)

    def search(self, query: str) -> Tuple[Set[int], List[str]]:
        tokens = full_preprocess(query, use_stemming=False)
        if len(tokens) < 2:
            return set(), []
        biwords = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        result = self.index.get(biwords[0], set()).copy()
        for bw in biwords[1:]:
            result &= self.index.get(bw, set())
        return result, biwords


# ---------------------------------------------------------------------------
# Positional Index
# ---------------------------------------------------------------------------

class PositionalIndex:
    def __init__(self):
        # term -> {doc_id -> [positions]}
        self.index: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

    def build(self, docs: dict):
        self.index.clear()
        for doc_id, doc in docs.items():
            tokens = full_preprocess(doc.text, use_stemming=False)
            for pos, token in enumerate(tokens):
                self.index[token][doc_id].append(pos)

    def search_phrase(self, query: str) -> Set[int]:
        tokens = full_preprocess(query, use_stemming=False)
        if not tokens:
            return set()
        if len(tokens) == 1:
            return set(self.index.get(tokens[0], {}).keys())

        candidate_docs = set(self.index.get(tokens[0], {}).keys())
        for t in tokens[1:]:
            candidate_docs &= set(self.index.get(t, {}).keys())

        result = set()
        for doc_id in candidate_docs:
            positions = [self.index[t][doc_id] for t in tokens]
            # Check consecutive positions
            for start_pos in positions[0]:
                if all(
                    (start_pos + offset) in set(positions[offset])
                    for offset in range(1, len(tokens))
                ):
                    result.add(doc_id)
                    break
        return result


# ---------------------------------------------------------------------------
# BST Dictionary
# ---------------------------------------------------------------------------

@dataclass
class BSTNode:
    key: str
    postings: Set[int] = field(default_factory=set)
    left: Optional["BSTNode"] = None
    right: Optional["BSTNode"] = None


class BST:
    def __init__(self):
        self.root: Optional[BSTNode] = None
        self._comparisons = 0

    def insert(self, key: str, doc_id: int):
        self.root = self._insert(self.root, key, doc_id)

    def _insert(self, node: Optional[BSTNode], key: str, doc_id: int) -> BSTNode:
        if node is None:
            return BSTNode(key=key, postings={doc_id})
        if key < node.key:
            node.left = self._insert(node.left, key, doc_id)
        elif key > node.key:
            node.right = self._insert(node.right, key, doc_id)
        else:
            node.postings.add(doc_id)
        return node

    def search(self, key: str) -> Tuple[Optional[Set[int]], int]:
        self._comparisons = 0
        node = self._search(self.root, key)
        return (node.postings if node else None), self._comparisons

    def _search(self, node: Optional[BSTNode], key: str) -> Optional[BSTNode]:
        self._comparisons += 1
        if node is None or node.key == key:
            return node
        if key < node.key:
            return self._search(node.left, key)
        return self._search(node.right, key)

    def build(self, inverted_index: Dict[str, Set[int]]):
        items = sorted(inverted_index.items())
        def build_rec(pairs):
            if not pairs:
                return None
            mid = len(pairs) // 2
            term, postings = pairs[mid]
            node = BSTNode(key=term, postings=set(postings))
            node.left = build_rec(pairs[:mid])
            node.right = build_rec(pairs[mid+1:])
            return node
        self.root = build_rec(items)


# ---------------------------------------------------------------------------
# B-Tree (order 50)
# ---------------------------------------------------------------------------

ORDER = 50  # max children per node


@dataclass
class BTreeNode:
    keys: List[str] = field(default_factory=list)
    postings: List[Set[int]] = field(default_factory=list)
    children: List["BTreeNode"] = field(default_factory=list)
    is_leaf: bool = True


class BTree:
    def __init__(self, order: int = ORDER):
        self.t = max(2, order // 2)  # minimum degree
        self.root = BTreeNode(is_leaf=True)
        self._comparisons = 0

    def search(self, key: str) -> Tuple[Optional[Set[int]], int]:
        self._comparisons = 0
        result = self._search(self.root, key)
        return result, self._comparisons

    def _search(self, node: BTreeNode, key: str) -> Optional[Set[int]]:
        i = 0
        while i < len(node.keys):
            self._comparisons += 1
            if key == node.keys[i]:
                return node.postings[i]
            if key < node.keys[i]:
                break
            i += 1
        if node.is_leaf:
            return None
        return self._search(node.children[i], key)

    def insert(self, key: str, postings: Set[int]):
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(is_leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key, postings)

    def _insert_non_full(self, node: BTreeNode, key: str, postings: Set[int]):
        i = len(node.keys) - 1
        if node.is_leaf:
            node.keys.append("")
            node.postings.append(set())
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.postings[i + 1] = node.postings[i]
                i -= 1
            node.keys[i + 1] = key
            node.postings[i + 1] = postings
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == 2 * self.t - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key, postings)

    def _split_child(self, parent: BTreeNode, i: int):
        t = self.t
        child = parent.children[i]
        new_node = BTreeNode(is_leaf=child.is_leaf)

        parent.keys.insert(i, child.keys[t - 1])
        parent.postings.insert(i, child.postings[t - 1])
        parent.children.insert(i + 1, new_node)

        new_node.keys = child.keys[t:]
        new_node.postings = child.postings[t:]
        child.keys = child.keys[:t - 1]
        child.postings = child.postings[:t - 1]

        if not child.is_leaf:
            new_node.children = child.children[t:]
            child.children = child.children[:t]

    def build(self, inverted_index: Dict[str, Set[int]]):
        self.root = BTreeNode(is_leaf=True)
        for term in sorted(inverted_index.keys()):
            self.insert(term, inverted_index[term])


# ---------------------------------------------------------------------------
# K-gram Index
# ---------------------------------------------------------------------------

class KGramIndex:
    def __init__(self, k: int = 3):
        self.k = k
        self.index: Dict[str, Set[str]] = defaultdict(set)

    def build(self, vocabulary: Set[str]):
        self.index.clear()
        for term in vocabulary:
            padded = f"${term}$"
            for i in range(len(padded) - self.k + 1):
                self.index[padded[i:i + self.k]].add(term)

    def get_kgrams(self, term: str) -> List[str]:
        padded = f"${term}$"
        return [padded[i:i + self.k] for i in range(len(padded) - self.k + 1)]

    def wildcard_search(self, pattern: str) -> Set[str]:
        parts = pattern.split("*")
        if not parts or all(p == "" for p in parts):
            return set()

        candidate_sets = []
        n = len(parts)
        for i, part in enumerate(parts):
            if not part:
                continue
            # Only anchor start $ for first part, end $ for last part
            is_first = (i == 0)
            is_last  = (i == n - 1)
            padded = ("$" if is_first else "") + part + ("$" if is_last else "")
            kgrams = [padded[j:j + self.k] for j in range(len(padded) - self.k + 1)]
            matches = None
            for kg in kgrams:
                hits = self.index.get(kg, set())
                matches = hits if matches is None else matches & hits
            if matches is not None:
                candidate_sets.append(matches)

        if not candidate_sets:
            return set()

        result = candidate_sets[0]
        for s in candidate_sets[1:]:
            result = result & s

        # Post-filter: ensure pattern actually matches
        regex = re.compile("^" + re.escape(pattern).replace(r"\*", ".*") + "$")
        return {t for t in result if regex.match(t)}


# ---------------------------------------------------------------------------
# Edit Distance (Levenshtein)
# ---------------------------------------------------------------------------

def edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def spelling_correct(term: str, vocabulary: List[str], max_dist: int = 2) -> List[Tuple[str, int]]:
    candidates = [(w, edit_distance(term, w)) for w in vocabulary if abs(len(w) - len(term)) <= max_dist]
    candidates.sort(key=lambda x: x[1])
    return [(w, d) for w, d in candidates if d <= max_dist][:10]


# ---------------------------------------------------------------------------
# Soundex (Phonetic)
# ---------------------------------------------------------------------------

def soundex(name: str) -> str:
    if not name:
        return ""
    name = name.upper()
    code_map = {
        "BFPV": "1", "CGJKQSXYZ": "2", "DT": "3",
        "L": "4", "MN": "5", "R": "6"
    }

    def get_code(ch):
        for letters, code in code_map.items():
            if ch in letters:
                return code
        return "0"

    result = name[0]
    prev_code = get_code(name[0])
    for ch in name[1:]:
        code = get_code(ch)
        if code != "0" and code != prev_code:
            result += code
        prev_code = code
        if len(result) == 4:
            break
    return result.ljust(4, "0")


class PhoneticIndex:
    def __init__(self):
        self.index: Dict[str, Set[str]] = defaultdict(set)

    def build(self, vocabulary: Set[str]):
        self.index.clear()
        for term in vocabulary:
            code = soundex(term)
            self.index[code].add(term)

    def search(self, term: str) -> Set[str]:
        code = soundex(term.upper())
        return self.index.get(code, set())


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def precision_recall(retrieved: Set[int], relevant: Set[int]) -> Tuple[float, float]:
    if not retrieved:
        return 0.0, 0.0
    tp = len(retrieved & relevant)
    precision = tp / len(retrieved)
    recall = tp / len(relevant) if relevant else 0.0
    return round(precision, 4), round(recall, 4)


def f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return round(2 * p * r / (p + r), 4)
