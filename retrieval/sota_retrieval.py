from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words


@dataclass
class RetrievedPassage:
    query: str
    retrieval_query: str
    rel_text: str
    score: float
    method: str
    query_variant: str


def bm25_tokenizer(text: str) -> list[str]:
    tokens = []
    for token in str(text).lower().split():
        token = token.strip(string.punctuation)
        if token and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokens.append(token)
    return tokens


class SotaRetriever:
    """Notebook-compatible retrieval with BM25 candidates and cross-encoder reranking."""

    def __init__(
        self,
        corpus_texts: Iterable[str],
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SOTA retrieval. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.corpus_texts = [str(text) for text in corpus_texts]
        self.cross_encoder = CrossEncoder(cross_encoder_model)

        tokenized_corpus = [bm25_tokenizer(passage) for passage in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search_bm25_ce(
        self, n: int, query: str, query_variant: str, result_query: str | None = None
    ) -> list[RetrievedPassage]:
        output_query = result_query if result_query is not None else query
        n = min(n, len(self.corpus_texts))
        bm25_scores = self.bm25.get_scores(bm25_tokenizer(query))
        top_n = np.argpartition(bm25_scores, -n)[-n:]
        hits = [{"corpus_id": idx, "score": float(bm25_scores[idx])} for idx in top_n]
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)

        cross_inp = [[query, self.corpus_texts[hit["corpus_id"]]] for hit in hits]
        cross_scores = self.cross_encoder.predict(cross_inp)
        for idx in range(len(cross_scores)):
            hits[idx]["cross-score"] = float(cross_scores[idx])

        hits = sorted(hits, key=lambda x: x["cross-score"], reverse=True)
        return [
            RetrievedPassage(
                query=output_query,
                retrieval_query=query,
                rel_text=self.corpus_texts[hit["corpus_id"]].replace("\n", " "),
                score=hit["cross-score"],
                method="bm25_ce",
                query_variant=query_variant,
            )
            for hit in hits[:n]
        ]


def load_corpus_texts(corpus_path: str, text_column: str = "requirement_text") -> list[str]:
    df = pd.read_excel(corpus_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in corpus file: {corpus_path}")
    return df[text_column].astype(str).fillna("").tolist()
