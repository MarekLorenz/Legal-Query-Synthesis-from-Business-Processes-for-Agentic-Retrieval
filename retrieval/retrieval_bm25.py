from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import pandas as pd
from rank_bm25 import BM25Okapi


@dataclass
class Document:
    doc_id: int
    text: str


@dataclass
class Query:
    text: str


@dataclass
class RankingResult:
    document: Document
    score: float
    rank: int


def tokenize(text: str) -> List[str]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    return [token for token in normalized.split() if token]


class BM25Index:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.tokenized_documents = [tokenize(doc.text) for doc in self.documents]
        self.index = BM25Okapi(self.tokenized_documents)

    def rank(self, query: Query, top_k: int = 10) -> List[RankingResult]:
        query_tokens = tokenize(query.text)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [RankingResult(document=self.documents[i], score=float(scores[i]), rank=rank + 1)
                for rank, i in enumerate(ranked_indices)]


def load_corpus(corpus_path: str, text_column: str = "requirement_text") -> List[Document]:
    df = pd.read_excel(corpus_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in corpus file: {corpus_path}")
    return [Document(doc_id=i, text=str(text)) for i, text in enumerate(df[text_column].astype(str).fillna(""))]


def build_bm25_index(corpus_path: str, text_column: str = "requirement_text") -> BM25Index:
    documents = load_corpus(corpus_path, text_column=text_column)
    return BM25Index(documents)


def bm25_top_k(query_text: str, corpus_path: str, top_k: int = 10, text_column: str = "requirement_text") -> List[RankingResult]:
    documents = load_corpus(corpus_path, text_column=text_column)
    index = BM25Index(documents)
    query = Query(text=query_text)
    return index.rank(query, top_k=top_k)
