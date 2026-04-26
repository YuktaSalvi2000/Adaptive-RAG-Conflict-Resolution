"""
reranker.py
===========
RankRAG-style pointwise LLM reranker.

Position in pipeline (matches diagram):
    stage2_retrieval  ─┐
                        ├─ merged chunks ──▶  reranker.py  ──▶  trust_scorer.py
    multi_hop         ─┘

Receives already-merged RetrievedChunk list from adaptive_retrieval.py,
scores each with llm_client.score_relevance(), returns sorted RankedChunk list.

Falls back to rrf_score ranking if LLM circuit breaker is open.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.config as config
import detect.llm_client as llm_client
from retrieval.stage2_retrieval import RetrievedChunk


# ─────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class RankedChunk:
    """
    RetrievedChunk + reranker_score.
    All fields from RetrievedChunk are preserved so trust_scorer.py
    can access retrieval_score, reranker_score, and label together.
    """
    chunk_id:       str
    doc_id:         str
    qid:            str
    source:         str
    label:          str | None      # 'gold' | 'misinfo' | 'noise' | None
    text:           str
    dense_score:    float
    bm25_score:     float
    rrf_score:      float
    reranker_score: float           # LLM pointwise [0,1]; falls back to rrf_score

    @classmethod
    def from_retrieved(cls, chunk: RetrievedChunk, reranker_score: float) -> "RankedChunk":
        return cls(
            chunk_id       = chunk.chunk_id,
            doc_id         = chunk.doc_id,
            qid            = chunk.qid,
            source         = chunk.source,
            label          = chunk.label,
            text           = chunk.text,
            dense_score    = chunk.dense_score,
            bm25_score     = chunk.bm25_score,
            rrf_score      = chunk.rrf_score,
            reranker_score = reranker_score,
        )


# ─────────────────────────────────────────────
# RERANKER
# ─────────────────────────────────────────────

class LLMReranker:
    """
    Pointwise LLM reranker.

    Sits after the stage2 / multi_hop merge in adaptive_retrieval.py.
    Scores every chunk in the pool regardless of which retrieval path
    produced it — so bridge-expanded chunks get scored on equal footing
    with normal hybrid chunks.
    """

    def rerank(
        self,
        query:  str,
        chunks: list[RetrievedChunk],
        top_k:  int = config.RERANKER_POOL,
    ) -> list[RankedChunk]:
        """
        Args:
            query  : the original question (not the augmented bridge query)
            chunks : merged pool from adaptive_retrieval (size ≤ TOP_K_RETRIEVAL)
            top_k  : how many RankedChunks to return

        Returns:
            List of RankedChunk sorted by reranker_score descending, len = top_k.
        """
        llm_up = not llm_client.is_disabled()
        ranked: list[RankedChunk] = []

        for chunk in chunks:
            if llm_up:
                score = llm_client.score_relevance(query, chunk.text)
            else:
                # Circuit breaker open → use normalised rrf_score as proxy
                score = chunk.rrf_score

            ranked.append(RankedChunk.from_retrieved(chunk, reranker_score=score))

        ranked.sort(key=lambda c: c.reranker_score, reverse=True)
        return ranked[:top_k]