"""
adaptive_retrieval.py  ★ NOVELTY
=================================
The central router. Reads the FailureProfile from failure_detector.py and
dispatches to the correct retrieval strategy, then runs reranker + trust scorer.

This is the "static pipeline → adaptive pipeline" transformation.

Pipeline position (matches diagram):
    failure_detector.py  ──▶  adaptive_retrieval.py ★
                                    /            \\
                         stage2_retrieval      multi_hop
                         (normal retrieval)  (bridge strategy)
                                    \\            /
                                      ↓ (merged)
                                   reranker.py
                                       ↓
                                 trust_scorer.py ★
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.config as config
from retrieval.stage2_retrieval import HybridRetriever, RetrievedChunk
from retrieval.multi_hop import MultiHopRetriever
from retrieval.reranker import LLMReranker, RankedChunk
from detect.failure_detector import FailureProfile, FailureType
from detect.trust_scorer import TrustScorer, AgentWeight


# ─────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class AdaptiveResult:
    """
    Everything downstream (madam_agent, aggregator, evaluator) needs
    from the retrieval stage, bundled in one object.
    """
    chunks:  list[RankedChunk]       # top-K after reranking  (size = TOP_K_FINAL)
    weights: list[AgentWeight]       # trust weight per chunk  (parallel list)
    profile: FailureProfile          # failure profile that drove routing
    strategy_used: str               # "multi-hop" or "hybrid"

# ─────────────────────────────────────────────
# ADAPTIVE RETRIEVER  ★
# ─────────────────────────────────────────────

class AdaptiveRetriever:
    """
    Routes each query to the right retrieval strategy based on its FailureProfile,
    then runs LLM reranking and trust scoring.

    Novel contribution:
      - BRIDGE  → MultiHopRetriever (two-pass retrieval)
      - default → HybridRetriever   (standard dense+BM25+RRF)
      - NOISE   → post-rerank filter: drop chunks below reranker threshold
      Both paths feed into the same reranker and trust scorer.
    """

    def __init__(
        self,
        base:     HybridRetriever,
        reranker: LLMReranker,
        multihop: MultiHopRetriever,
        trust:    TrustScorer,
    ):
        self.base     = base
        self.reranker = reranker
        self.multihop = multihop
        self.trust    = trust

    def retrieve(
        self,
        query:   str,
        profile: FailureProfile,
    ) -> AdaptiveResult:
        """
        Args:
            query   : the natural language question
            profile : FailureProfile from failure_detector.detect()

        Returns:
            AdaptiveResult with ranked chunks, trust weights, and the profile.
        """

        # ── Step 1: Route to retrieval strategy ─────────────
        # BRIDGE → two-pass multi-hop retrieval
        # everything else → standard hybrid retrieval
        if FailureType.BRIDGE in profile.failure_types:
            raw_chunks: list[RetrievedChunk] = self.multihop.retrieve(
            query,
            top_k=config.RERANKER_POOL
    )   
            strategy = "multi-hop"
        else:
            raw_chunks: list[RetrievedChunk] = self.base.query(
            query,
            top_k=config.RERANKER_POOL
    )
            strategy = "stage2_hybrid"

        if FailureType.NOISE in profile.failure_types:
            strategy += "+noise_filter"

        if FailureType.MISINFORMATION in profile.failure_types:
            strategy += "+trust_suppression"

        if FailureType.AMBIGUOUS in profile.failure_types:
            strategy += "+ambiguity_aware"

        # ── Step 2: Rerank (always — both paths merge here) ──
        # Pool size = RERANKER_POOL, output size = RERANKER_POOL
        # (TOP_K_FINAL slicing happens after noise filter below)
        ranked: list[RankedChunk] = self.reranker.rerank(
            query, raw_chunks, top_k=config.RERANKER_POOL
        )

        # ── Step 3: Noise filter ─────────────────────────────
        # If NOISE detected, drop chunks whose reranker score is too low
        # before computing trust weights — noisy chunks skew agreement scores
        if FailureType.NOISE in profile.failure_types:
            ranked = [
                c for c in ranked
                if c.reranker_score > config.NOISE_RERANKER_THRESHOLD
            ]
            # Guard: always keep at least 1 chunk so downstream doesn't crash
            if not ranked:
                ranked = self.reranker.rerank(query, raw_chunks, top_k=1)

        # ── Step 4: Trust scoring ★ ──────────────────────────
        weights: list[AgentWeight] = self.trust.score(ranked)

        # ── Step 5: Final top-K slice ────────────────────────
        final_chunks = ranked[:config.TOP_K_FINAL]
        final_weights = weights[:config.TOP_K_FINAL]

        return AdaptiveResult(
            chunks  = final_chunks,
            weights = final_weights,
            profile = profile,
            strategy_used=strategy,
        )