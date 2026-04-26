"""
trust_scorer.py  ★ NOVELTY
===========================
Computes a real-valued trust weight per chunk using three signals:

    weight = α·norm(retrieval) + β·norm(reranker) + γ·norm(agreement)

Signals:
    retrieval_score  — rrf_score from stage2_retrieval (hybrid BM25+dense)
    reranker_score   — LLM pointwise score from reranker.py
    agreement_score  — mean cosine similarity of this chunk's TF-IDF vector
                       to all other chunks in the pool
                       (low agreement → outlier → suspicious → lower weight)

This directly addresses Baseline Limitation: debate amplifies misinformation
because all agents are weighted equally. By down-weighting chunks that score
poorly on retrieval, reranking, AND cross-document agreement, misinformation
documents lose influence before they reach the aggregator.

Consumed by:
    adaptive_retrieval.py  — calls trust.score(ranked_chunks)
    aggregator.py          — uses AgentWeight.weight to vote-weight each agent
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.config as config
from retrieval.reranker import RankedChunk


# ─────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class AgentWeight:
    """
    Trust weight for one chunk/agent.
    Parallel to the RankedChunk list returned by adaptive_retrieval.
    """
    chunk_id:        str
    weight:          float    # final combined score in [0, 1]
    retrieval_score: float    # normalised rrf_score
    reranker_score:  float    # normalised LLM pointwise score
    agreement_score: float    # normalised cross-document cosine consensus


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _minmax_norm(values: np.ndarray) -> np.ndarray:
    """
    Min-max normalise to [0, 1].
    If all values are identical (range = 0), return uniform 1.0
    so every chunk gets equal weight on that signal rather than all-zero.
    """
    mn, mx = values.min(), values.max()
    if mx - mn < 1e-9:
        return np.ones_like(values)
    return (values - mn) / (mx - mn)


def _compute_agreement_scores(chunks: list[RankedChunk]) -> np.ndarray:
    """
    Compute cross-document agreement score for each chunk.

    Method:
        1. Embed all chunk texts with a lightweight TF-IDF vectorizer
           (fit on the pool itself — no external model needed).
        2. L2-normalise → cosine similarity = dot product.
        3. Agreement score for chunk i = mean cosine sim to all OTHER chunks.
           High agreement → chunk is consistent with the pool → trustworthy.
           Low agreement  → outlier (possibly misinformation or noise).

    Returns np.ndarray of shape (n_chunks,) with raw agreement scores.
    """
    texts = [c.text for c in chunks]

    if len(texts) == 1:
        # Single chunk — no peers to compare against; give full agreement
        return np.array([1.0])

    # Fit TF-IDF on the pool (bigrams capture more context than unigrams)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(texts)               # sparse (n, vocab)
    dense = normalize(tfidf, norm="l2").toarray()         # (n, vocab) float64

    # Cosine similarity matrix: sim[i, j] = dense[i] · dense[j]
    sim_matrix = dense @ dense.T                          # (n, n)

    # Mean similarity to all OTHER chunks (exclude self-similarity on diagonal)
    n = len(chunks)
    agreement = np.zeros(n)
    for i in range(n):
        others = [sim_matrix[i, j] for j in range(n) if j != i]
        agreement[i] = float(np.mean(others)) if others else 1.0

    return agreement


# ─────────────────────────────────────────────
# TRUST SCORER  ★
# ─────────────────────────────────────────────

class TrustScorer:
    """
    Assigns a trust weight to each chunk in the reranked pool.

    Args (from config):
        alpha : weight for retrieval score  (default 0.4)
        beta  : weight for reranker score   (default 0.4)
        gamma : weight for agreement score  (default 0.2)
        α + β + γ must equal 1.0
    """

    def __init__(
        self,
        alpha: float = config.TRUST_ALPHA,
        beta:  float = config.TRUST_BETA,
        gamma: float = config.TRUST_GAMMA,
    ):
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
            f"α+β+γ must equal 1.0, got {alpha+beta+gamma}"
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def score(self, chunks: list[RankedChunk]) -> list[AgentWeight]:
        """
        Compute trust weights for a list of RankedChunks.

        Args:
            chunks : output of LLMReranker.rerank() — ordered by reranker_score

        Returns:
            List of AgentWeight in the same order as chunks.
        """
        if not chunks:
            return []

        n = len(chunks)

        # ── Raw signal arrays ────────────────────────────────
        retrieval_raw = np.array([c.rrf_score      for c in chunks])
        reranker_raw  = np.array([c.reranker_score for c in chunks])
        agreement_raw = _compute_agreement_scores(chunks)

        # ── Normalise each signal to [0, 1] ──────────────────
        retrieval_norm = _minmax_norm(retrieval_raw)
        reranker_norm  = _minmax_norm(reranker_raw)
        agreement_norm = _minmax_norm(agreement_raw)

        # ── Combine: α·retrieval + β·reranker + γ·agreement ──
        combined = (
            self.alpha * retrieval_norm +
            self.beta  * reranker_norm  +
            self.gamma * agreement_norm
        )

        return [
            AgentWeight(
                chunk_id        = chunks[i].chunk_id,
                weight          = float(combined[i]),
                retrieval_score = float(retrieval_norm[i]),
                reranker_score  = float(reranker_norm[i]),
                agreement_score = float(agreement_norm[i]),
            )
            for i in range(n)
        ]


# ─────────────────────────────────────────────
# SELF-TEST  (python trust_scorer.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from retrieval.reranker import RankedChunk

    print("\n" + "═" * 60)
    print("  trust_scorer.py — self test")
    print("═" * 60)

    # Build synthetic chunks that simulate a misinformation scenario:
    #   chunk_0, chunk_1 : consistent gold docs (high agreement)
    #   chunk_2          : misinformation outlier (low agreement)
    fake_chunks = [
        RankedChunk(
            chunk_id="chunk_00000", doc_id="gold_A", qid="q1",
            source="ramdocs", label="gold",
            text="The Oberoi Group is a hotel company with its head office in Delhi. "
                 "Founded in 1934, it operates luxury hotels across six countries.",
            dense_score=0.82, bm25_score=14.3, rrf_score=0.031,
            reranker_score=0.9,
        ),
        RankedChunk(
            chunk_id="chunk_00001", doc_id="gold_B", qid="q1",
            source="ramdocs", label="gold",
            text="The Oberoi family is an Indian family famous for its involvement "
                 "in hotels through The Oberoi Group, headquartered in Delhi.",
            dense_score=0.79, bm25_score=12.1, rrf_score=0.028,
            reranker_score=0.85,
        ),
        RankedChunk(
            chunk_id="chunk_00002", doc_id="misinfo_X", qid="q1",
            source="ramdocs", label="misinfo",
            text="The Oberoi Group was founded by a British entrepreneur and its "
                 "headquarters are located in London, United Kingdom.",
            dense_score=0.41, bm25_score=5.2, rrf_score=0.011,
            reranker_score=0.3,
        ),
        RankedChunk(
            chunk_id="chunk_00003", doc_id="noise_Y", qid="q1",
            source="ramdocs", label="noise",
            text="Cricket is a bat-and-ball game played between two teams of eleven "
                 "players on a field with a 22-yard pitch.",
            dense_score=0.12, bm25_score=1.1, rrf_score=0.006,
            reranker_score=0.1,
        ),
    ]

    scorer = TrustScorer()
    weights = scorer.score(fake_chunks)

    print(f"\n  α={scorer.alpha}  β={scorer.beta}  γ={scorer.gamma}\n")
    print(f"  {'chunk_id':<15} {'label':<10} {'retrieval':>10} {'reranker':>10} "
          f"{'agreement':>10} {'WEIGHT':>10}")
    print("  " + "─" * 65)

    for chunk, w in zip(fake_chunks, weights):
        print(f"  {w.chunk_id:<15} {str(chunk.label):<10} "
              f"{w.retrieval_score:>10.3f} {w.reranker_score:>10.3f} "
              f"{w.agreement_score:>10.3f} {w.weight:>10.3f}")

    print()

    # Assertions — gold docs should outweigh misinfo/noise
    gold_weights    = [w.weight for w, c in zip(weights, fake_chunks) if c.label == "gold"]
    misinfo_weights = [w.weight for w, c in zip(weights, fake_chunks) if c.label == "misinfo"]
    noise_weights   = [w.weight for w, c in zip(weights, fake_chunks) if c.label == "noise"]

    assert all(g > m for g in gold_weights for m in misinfo_weights), \
        "FAIL: gold chunks should outweigh misinfo chunks"
    assert all(g > n for g in gold_weights for n in noise_weights), \
        "FAIL: gold chunks should outweigh noise chunks"
    assert all(0.0 <= w.weight <= 1.0 for w in weights), \
        "FAIL: weights must be in [0, 1]"

    print("  ✅  All assertions passed")
    print("      gold > misinfo > noise weight ordering confirmed")

    # Edge case: single chunk
    single = [fake_chunks[0]]
    single_weights = scorer.score(single)
    assert len(single_weights) == 1
    assert single_weights[0].weight == 1.0, "Single chunk should get weight 1.0"
    print("  ✅  Single-chunk edge case passed")

    # Edge case: empty list
    empty_weights = scorer.score([])
    assert empty_weights == []
    print("  ✅  Empty list edge case passed")

    print("\n" + "═" * 60 + "\n")