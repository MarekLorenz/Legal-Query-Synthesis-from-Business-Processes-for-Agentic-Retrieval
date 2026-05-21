from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from retrieval.retrieval_bm25 import RankingResult


@dataclass
class RankedList:
    name: str
    results: List[RankingResult]


class ReciprocalRankFusion:
    def fuse(
        self,
        ranked_lists: List[RankedList],
        top_k: int = 10,
        k: int = 60,
        weights: Optional[List[float]] = None,
    ) -> List[RankingResult]:
        """
        RRF fusion: score(d) += w_i / (k + rank_i(d)). If weights is None, all w_i = 1.
        """
        if not ranked_lists:
            return []

        if weights is not None and len(weights) != len(ranked_lists):
            raise ValueError(
                f"weights length ({len(weights)}) must match ranked_lists ({len(ranked_lists)})"
            )

        fusion_scores = {}
        documents = {}

        for list_idx, ranked_list in enumerate(ranked_lists):
            w = 1.0 if weights is None else float(weights[list_idx])
            for position, result in enumerate(ranked_list.results, start=1):
                doc_id = result.document.doc_id
                fusion_scores[doc_id] = fusion_scores.get(doc_id, 0.0) + w * (1.0 / (k + position))
                documents[doc_id] = result.document

        merged = [
            RankingResult(document=documents[doc_id], score=score, rank=0)
            for doc_id, score in fusion_scores.items()
        ]
        merged.sort(key=lambda item: item.score, reverse=True)
        return [RankingResult(document=item.document, score=item.score, rank=index + 1)
                for index, item in enumerate(merged[:top_k])]
