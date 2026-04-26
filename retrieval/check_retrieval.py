import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# 🔑 IMPORTANT: load BM25Index into __main__
from index.stage1_indexing import BM25Index

from stage2_retrieval import HybridRetriever
from multi_hop import MultiHopRetriever

if __name__ == "__main__":
    base = HybridRetriever()
    mh = MultiHopRetriever(base)

    import json

    with open("data/hotpot_dev.json", encoding="utf-8") as f:
        data = json.load(f)

    query = data[0]["question"]
    print(query)

    print("\n=== Stage 2 normal retrieval ===")
    normal = base.query(query, top_k=5, mode="hybrid")
    for i, r in enumerate(normal, 1):
        print(f"\n#{i} rrf={r.rrf_score:.4f} doc={r.doc_id}")
        print(r.text[:180])

    print("\n=== Multi-hop retrieval ===")
    multihop = mh.retrieve(query, top_k=5)
    for i, r in enumerate(multihop, 1):
        print(f"\n#{i} rrf={r.rrf_score:.4f} doc={r.doc_id}")
        print(r.text[:180])