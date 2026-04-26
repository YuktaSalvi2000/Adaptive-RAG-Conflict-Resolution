"""
config.py
---------
Single source of truth for every tunable value.
Everything imports from here; nothing hardcodes paths or params.
"""

from pathlib import Path

# Project folders
INDEX_DIR = Path("index")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

# Data paths
HOTPOT_PATH = DATA_DIR / "hotpot_dev.json"
RAMDOCS_PATH = DATA_DIR / "ramdocs_test.json"

# Dataset sampling
N_SAMPLES = 50

# Retrieval
TOP_K_RETRIEVAL = 10
TOP_K_FINAL = 3

MAX_BRIDGE_ENTITIES = 3
MULTIHOP_TOP_K = 10

DENSE_WEIGHT = 1.0
BM25_WEIGHT = 1.0
RRF_K = 60

# Reranker
RERANKER_POOL = 5

# Agents
MAX_AGENTS = 3
DEBATE_ROUNDS = 3

# Trust scorer
TRUST_ALPHA = 0.4
TRUST_BETA = 0.4
TRUST_GAMMA = 0.2

NOISE_RERANKER_THRESHOLD = 0.3
MIN_WEIGHT_THRESHOLD = 0.2

# LLM client
OPENAI_MODEL = "gpt-4o-mini"
LLM_MAX_TOKENS = 256

BACKOFF_BASE = 2.0
BACKOFF_MAX = 60.0
BACKOFF_RETRIES = 5

DEBATE_MAX_TOKENS = 16

# Failure detector
FAILURE_DETECTOR_CONFIDENCE = 0.9

BRIDGE_KEYWORDS = {
    "nationality", "born in", "directed by", "author of",
    "founded by", "located in", "owned by", "written by",
    "produced by", "member of", "part of", "head office",
    "capital of",
}

# Indexing
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_DIM = 384

# Output files
SUMMARY_METRICS_PATH = RESULTS_DIR / "summary_metrics.csv"
PER_QUERY_PATH = RESULTS_DIR / "per_query.csv"
TRACES_PATH = RESULTS_DIR / "traces.json"

CHUNKS_PATH = INDEX_DIR / "chunks.json"
DENSE_MATRIX_PATH = INDEX_DIR / "dense_matrix.npy"
DENSE_VOCAB_PATH = INDEX_DIR / "dense_vocab.pkl"
BM25_CORPUS_PATH = INDEX_DIR / "bm25_corpus.pkl"