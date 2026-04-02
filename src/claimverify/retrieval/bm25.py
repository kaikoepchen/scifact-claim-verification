"""BM25 sparse retrieval - adapted from Assignment 1."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import nltk

nltk.download("stopwords", quiet=True)


class BM25Retriever:
    """BM25 retriever with inverted index on a text corpus."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._stemmer = PorterStemmer()
        self._stopwords = set(stopwords.words("english"))

        # Index state
        self.N = 0
        self.avgdl = 0.0
        self.df: dict[str, int] = {}
        self.doc_len: dict[str, int] = {}
        self.doc_tf: dict[str, Counter] = {}
        self._built = False

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        tokens = [t for t in tokens if t not in self._stopwords]
        return [self._stemmer.stem(t) for t in tokens if t]

    def build_index(self, corpus: dict[str, str]) -> None:
        """Build inverted index from {doc_id: text} mapping."""
        total_len = 0
        self.N = 0
        self.df = {}
        self.doc_len = {}
        self.doc_tf = {}

        for doc_id, text in corpus.items():
            tokens = self.tokenize(text)
            tf = Counter(tokens)
            self.doc_tf[doc_id] = tf
            doc_length = len(tokens)
            self.doc_len[doc_id] = doc_length
            total_len += doc_length
            self.N += 1
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        self.avgdl = total_len / self.N if self.N > 0 else 0.0
        self._built = True

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _score_doc(self, query_tokens: list[str], doc_id: str) -> float:
        dl = self.doc_len.get(doc_id, 0)
        if dl == 0 or self.avgdl == 0:
            return 0.0
        tf = self.doc_tf.get(doc_id, Counter())
        score = 0.0
        for term in set(query_tokens):
            f = tf.get(term, 0)
            if f == 0:
                continue
            num = f * (self.k1 + 1)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self._idf(term) * (num / denom)
        return score

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve top-k documents for a query. Returns [(doc_id, score)]."""
        assert self._built, "Call build_index() first."
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        results = []
        for doc_id in self.doc_tf:
            s = self._score_doc(query_tokens, doc_id)
            if s > 0:
                results.append((doc_id, s))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def batch_retrieve(
        self, queries: dict[str, str], top_k: int = 10
    ) -> dict[str, list[tuple[str, float]]]:
        """Retrieve for multiple queries. Returns {query_id: [(doc_id, score)]}."""
        return {qid: self.retrieve(q, top_k) for qid, q in queries.items()}
