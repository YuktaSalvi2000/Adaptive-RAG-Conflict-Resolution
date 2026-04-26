"""
stage2_retrieval.py
===================
Hybrid Retrieval — Stage 2: Query-time retrieval.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

# allow running: python retrieval/stage2_retrieval.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.config as config
from index.stage1_indexing import BM25Index

@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    qid: str
    source: str
    label: str | None
    text: str
    dense_score: float
    bm25_score: float
    rrf_score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _rrf_score(
    dense_rank: int,
    bm25_rank: int,
    k: int = config.RRF_K,
    dense_weight: float = config.DENSE_WEIGHT,
    bm25_weight: float = config.BM25_WEIGHT,
) -> float:
    return (dense_weight / (k + dense_rank)) + (bm25_weight / (k + bm25_rank))


class HybridRetriever:
    def __init__(self):
        self._load_index()

    def _load_index(self):
        missing = [
            p for p in [
                config.CHUNKS_PATH,
                config.DENSE_MATRIX_PATH,
                config.DENSE_VOCAB_PATH,
                config.BM25_CORPUS_PATH,
            ]
            if not Path(p).exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing index artifacts:\n"
                + "\n".join(str(p) for p in missing)
                + "\nRun: python index/stage1_indexing.py"
            )

        with open(config.CHUNKS_PATH, encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)

        self.dense_matrix: np.ndarray = np.load(config.DENSE_MATRIX_PATH)

        with open(config.DENSE_VOCAB_PATH, "rb") as f:
            pipeline = pickle.load(f)

        self.vectorizer = pipeline["vectorizer"]
        self.svd = pipeline["svd"]

        with open(config.BM25_CORPUS_PATH, "rb") as f:
            bm25_data = pickle.load(f)

        self.bm25 = bm25_data["bm25"]

    def _embed_query(self, query: str) -> np.ndarray:
        tfidf_vec = self.vectorizer.transform([query])
        svd_vec = self.svd.transform(tfidf_vec)
        return normalize(svd_vec, norm="l2")[0].astype(np.float32)

    def _dense_query(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q_vec = self._embed_query(query)
        cosines = self.dense_matrix @ q_vec
        top_idx = np.argsort(cosines)[::-1][:top_k]
        return [(int(i), float(cosines[i])) for i in top_idx]

    def _bm25_query(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokens = _tokenize(query)
        return self.bm25.get_top_n(tokens, n=top_k)

    def _to_chunk(
        self,
        idx: int,
        dense_score: float = 0.0,
        bm25_score: float = 0.0,
        rrf_score: float = 0.0,
    ) -> RetrievedChunk:
        c = self.chunks[idx]
        return RetrievedChunk(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            qid=c["qid"],
            source=c["source"],
            label=c.get("label"),
            text=c["text"],
            dense_score=dense_score,
            bm25_score=bm25_score,
            rrf_score=rrf_score,
        )

    def _fuse_rrf(
        self,
        dense_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
        top_k: int,
    ) -> list[tuple[int, float, float, float]]:
        dense_rank = {idx: r + 1 for r, (idx, _) in enumerate(dense_results)}
        bm25_rank = {idx: r + 1 for r, (idx, _) in enumerate(bm25_results)}

        dense_score_map = {idx: s for idx, s in dense_results}
        bm25_score_map = {idx: s for idx, s in bm25_results}

        all_idx = set(dense_rank) | set(bm25_rank)

        worst_dense = len(dense_results) + 1
        worst_bm25 = len(bm25_results) + 1

        fused = []
        for idx in all_idx:
            dr = dense_rank.get(idx, worst_dense)
            br = bm25_rank.get(idx, worst_bm25)
            rrf = _rrf_score(dr, br)

            fused.append((
                idx,
                dense_score_map.get(idx, 0.0),
                bm25_score_map.get(idx, 0.0),
                rrf,
            ))

        fused.sort(key=lambda x: x[3], reverse=True)
        return fused[:top_k]

    def query(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVAL,
        mode: str = "hybrid",
    ) -> list[RetrievedChunk]:

        if mode == "dense":
            dense_results = self._dense_query(query, top_k)
            return [
                self._to_chunk(
                    idx,
                    dense_score=score,
                    bm25_score=0.0,
                    rrf_score=score,
                )
                for idx, score in dense_results
            ]

        if mode == "bm25":
            bm25_results = self._bm25_query(query, top_k)
            return [
                self._to_chunk(
                    idx,
                    dense_score=0.0,
                    bm25_score=score,
                    rrf_score=score,
                )
                for idx, score in bm25_results
            ]

        if mode != "hybrid":
            raise ValueError("mode must be one of: hybrid, dense, bm25")

        dense_results = self._dense_query(query, top_k)
        bm25_results = self._bm25_query(query, top_k)
        fused = self._fuse_rrf(dense_results, bm25_results, top_k)

        return [
            self._to_chunk(
                idx,
                dense_score=d_score,
                bm25_score=b_score,
                rrf_score=rrf,
            )
            for idx, d_score, b_score, rrf in fused
        ]


if __name__ == "__main__":
    retriever = HybridRetriever()
    query = "what nationality is the director of Spectre"
    results = retriever.query(query, top_k=config.TOP_K_FINAL, mode="hybrid")

    for i, r in enumerate(results, 1):
        print(f"\n#{i} rrf={r.rrf_score:.4f} dense={r.dense_score:.4f} bm25={r.bm25_score:.4f}")
        print(f"qid={r.qid} doc={r.doc_id} source={r.source} label={r.label}")
        print(r.text[:250])