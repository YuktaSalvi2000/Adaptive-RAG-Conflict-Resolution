"""
stage1_indexing.py
==================
Hybrid Retrieval System — Stage 1: Indexing Pipeline

Reads QueryExample objects from data_loader.py, chunks their documents,
builds dense (TF-IDF + SVD) and BM25 indexes, and saves 4 artefacts to index/.

Output layout:
  index/
    chunks.json       — list of chunk dicts with text + metadata
    dense_matrix.npy  — (n_chunks, DENSE_DIMS) float32 L2-normalised embeddings
    dense_vocab.pkl   — fitted TF-IDF + SVD pipeline (used at query time)
    bm25_corpus.pkl   — BM25Index object + tokenised corpus
"""

import json
import math
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.config as config
from data.data_loader import load_hotpotqa, load_ramdocs, QueryExample


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def log(msg: str, symbol: str = "▸"):
    print(f"  {symbol} {msg}", flush=True)


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping word chunks.
    chunk_size and overlap are word counts.
    """
    words = text.split()

    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    step = max(chunk_size - overlap, 1)

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


# ─────────────────────────────────────────────
# STEP 1 — LOAD QueryExamples
# ─────────────────────────────────────────────

def load_examples() -> list[QueryExample]:
    log(f"Loading HotpotQA from {config.HOTPOT_PATH}")
    hotpot = load_hotpotqa(config.HOTPOT_PATH, n=config.N_SAMPLES)
    log(f"Loaded {len(hotpot)} HotpotQA examples")

    log(f"Loading RAMDocs from {config.RAMDOCS_PATH}")
    ramdocs = load_ramdocs(config.RAMDOCS_PATH, n=config.N_SAMPLES)
    log(f"Loaded {len(ramdocs)} RAMDocs examples")

    return hotpot + ramdocs


# ─────────────────────────────────────────────
# STEP 2 — FLATTEN Documents → chunks
# ─────────────────────────────────────────────

def build_chunks(examples: list[QueryExample]) -> list[dict]:
    """
    For every Document inside every QueryExample, split text into
    overlapping chunks. Each chunk carries full provenance metadata
    so stage2_retrieval.py can trace back to the original document.

    chunk dict keys:
        chunk_id    — unique string e.g. "chunk_00042"
        doc_id      — from Document.doc_id
        qid         — parent QueryExample.qid (for evaluation tracing)
        source      — 'hotpotqa' | 'ramdocs'
        label       — 'gold' | 'misinfo' | 'noise' | None  (from Document.label)
        text        — the actual chunk text
        char_len    — character length of this chunk
    """
    log(f"Chunking documents (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    chunks = []
    chunk_counter = 0

    for example in examples:
        for doc in example.documents:
            text_chunks = chunk_text(doc.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            for sub_idx, chunk_text_val in enumerate(text_chunks):
                chunks.append({
                    "chunk_id":  f"chunk_{chunk_counter:05d}",
                    "doc_id":    doc.doc_id,
                    "qid":       example.qid,
                    "source":    example.source,
                    "label":     doc.label,          # gold | misinfo | noise | None
                    "sub_idx":   sub_idx,
                    "text":      chunk_text_val,
                    "char_len":  len(chunk_text_val),
                })
                chunk_counter += 1

    log(f"Created {len(chunks)} chunks from {sum(len(e.documents) for e in examples)} documents")
    return chunks


# ─────────────────────────────────────────────
# STEP 3 — DENSE EMBEDDINGS (TF-IDF + SVD)
# ─────────────────────────────────────────────

def build_dense_embeddings(chunks: list[dict]):
    """
    Fit TF-IDF + TruncatedSVD (LSA) on chunk corpus.

    Returns:
        embeddings    — np.ndarray (n_chunks, n_components), L2-normalised float32
        pipeline      — dict with fitted vectorizer + svd  (saved for query time)
    """
    log("Building TF-IDF matrix …")
    texts = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    log(f"TF-IDF: {tfidf_matrix.shape}, vocab={len(vectorizer.vocabulary_)}")

    n_components = min(
        config.DENSE_DIM,
        tfidf_matrix.shape[1] - 1,
        tfidf_matrix.shape[0] - 1,
    )
    log(f"Reducing to {n_components} dims with TruncatedSVD (LSA) …")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense = svd.fit_transform(tfidf_matrix)

    log("L2-normalising for cosine similarity …")
    embeddings = normalize(dense, norm="l2").astype(np.float32)
    log(f"Explained variance: {svd.explained_variance_ratio_.sum():.1%}")

    pipeline = {"vectorizer": vectorizer, "svd": svd, "n_components": n_components}
    return embeddings, pipeline


# ─────────────────────────────────────────────
# STEP 4 — BM25 INDEX
# ─────────────────────────────────────────────

class BM25Index:
    """
    Pure-Python BM25 (Robertson et al.).
    No external dependencies beyond numpy.
    """

    def __init__(self, tokenised_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus = tokenised_corpus
        self.n_docs = len(tokenised_corpus)

        self.dl    = np.array([len(doc) for doc in tokenised_corpus], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.n_docs else 1.0

        # Robertson IDF with smoothing
        df: dict[str, int] = Counter()
        for doc in tokenised_corpus:
            df.update(set(doc))
        self.idf: dict[str, float] = {
            term: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

        # Inverted index: term → [(doc_idx, term_freq), ...]
        self.inverted: dict[str, list[tuple[int, int]]] = {}
        for doc_idx, doc in enumerate(tokenised_corpus):
            for term, freq in Counter(doc).items():
                self.inverted.setdefault(term, []).append((doc_idx, freq))

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            k1, b, avgdl = self.k1, self.b, self.avgdl
            for doc_idx, tf in self.inverted.get(term, []):
                dl  = self.dl[doc_idx]
                num = tf * (k1 + 1)
                den = tf + k1 * (1 - b + b * dl / avgdl)
                scores[doc_idx] += idf * num / den
        return scores

    def get_top_n(self, query_tokens: list[str], n: int = 10) -> list[tuple[int, float]]:
        scores = self.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_idx]


def build_bm25_index(chunks: list[dict]) -> tuple["BM25Index", list[list[str]]]:
    log("Tokenising corpus for BM25 …")
    tokenised = [tokenize(c["text"]) for c in chunks]
    avg_toks  = sum(len(t) for t in tokenised) / max(len(tokenised), 1)
    log(f"Tokenised {len(tokenised)} chunks, avg {avg_toks:.1f} tokens/chunk")

    log("Building BM25 index …")
    bm25 = BM25Index(tokenised)
    log(f"BM25 ready — vocab={len(bm25.idf)}")
    return bm25, tokenised


# ─────────────────────────────────────────────
# STEP 5 — SAVE ARTEFACTS
# ─────────────────────────────────────────────

def save_artifacts(
    chunks: list[dict],
    embeddings: np.ndarray,
    dense_pipeline: dict,
    bm25: BM25Index,
    tokenised: list[list[str]],
):
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # chunks.json
    with open(config.CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, indent=2)
    log(f"Saved chunks        → {config.CHUNKS_PATH}  ({len(chunks)} chunks)")

    # dense_matrix.npy
    np.save(config.DENSE_MATRIX_PATH, embeddings)
    log(f"Saved dense matrix  → {config.DENSE_MATRIX_PATH}  {embeddings.shape}")

    # dense_vocab.pkl  (TF-IDF + SVD pipeline for query-time embedding)
    with open(config.DENSE_VOCAB_PATH, "wb") as f:
        pickle.dump(dense_pipeline, f)
    log(f"Saved dense vocab   → {config.DENSE_VOCAB_PATH}")

    # bm25_corpus.pkl
    with open(config.BM25_CORPUS_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenised": tokenised}, f)
    log(f"Saved BM25 corpus   → {config.BM25_CORPUS_PATH}")


# ─────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────

def verify_index():
    print("\n  ── Verification ──────────────────────────────")

    with open(config.CHUNKS_PATH) as f:
        chunks = json.load(f)
    assert len(chunks) > 0
    assert all(k in chunks[0] for k in ("chunk_id", "doc_id", "qid", "source", "label", "text"))
    log(f"chunks.json ✓  {len(chunks)} chunks")

    emb = np.load(config.DENSE_MATRIX_PATH)
    assert emb.shape[0] == len(chunks), "Row count mismatch"
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5), "Not L2-normalised"
    log(f"dense_matrix.npy ✓  shape={emb.shape}")

    with open(config.DENSE_VOCAB_PATH, "rb") as f:
        pipeline = pickle.load(f)
    assert "vectorizer" in pipeline and "svd" in pipeline
    log(f"dense_vocab.pkl ✓  vocab={len(pipeline['vectorizer'].vocabulary_)}")

    with open(config.BM25_CORPUS_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    assert bm25.n_docs == len(chunks)
    log(f"bm25_corpus.pkl ✓  n_docs={bm25.n_docs}, vocab={len(bm25.idf)}")

    # Smoke test — one BM25 query
    q = "what nationality is the director"
    results = bm25.get_top_n(tokenize(q), n=3)
    log(f"BM25 smoke test: '{q}'")
    for rank, (idx, score) in enumerate(results, 1):
        print(f"       #{rank}  score={score:.3f}  chunk={chunks[idx]['chunk_id']}  "
              f"label={chunks[idx]['label']}  text='{chunks[idx]['text'][:60]}...'")

    print("  ── Verification complete ✓ ───────────────────\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    t0 = time.time()
    print("\n" + "═" * 56)
    print("  Stage 1 — Indexing Pipeline")
    print("═" * 56)

    examples = load_examples()

    print()
    log("── Chunking ─────────────────────────────────────")
    chunks = build_chunks(examples)

    print()
    log("── Dense embeddings ─────────────────────────────")
    embeddings, dense_pipeline = build_dense_embeddings(chunks)

    print()
    log("── BM25 index ───────────────────────────────────")
    bm25, tokenised = build_bm25_index(chunks)

    print()
    log("── Saving artefacts ─────────────────────────────")
    save_artifacts(chunks, embeddings, dense_pipeline, bm25, tokenised)

    verify_index()

    elapsed = time.time() - t0
    print("═" * 56)
    print(f"  ✅  Index ready.")
    print(f"      {len(chunks)} chunks | {embeddings.shape[1]}-dim dense | BM25 vocab={len(bm25.idf)}")
    print(f"      Total time: {elapsed:.2f}s")
    print("═" * 56 + "\n")


if __name__ == "__main__":
    sys.exit(main())