from __future__ import annotations

import re
import sys
from pathlib import Path

# allow running from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.config as config
from retrieval.stage2_retrieval import HybridRetriever, RetrievedChunk


_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")


def extract_bridge_entities(
    chunks: list[RetrievedChunk],
    max_entities: int = config.MAX_BRIDGE_ENTITIES,
) -> list[str]:
    candidate_text = " ".join(c.text for c in chunks[:3])

    proper_nouns = _PROPER_NOUN_RE.findall(candidate_text)
    years = _YEAR_RE.findall(candidate_text)

    seen = set()
    entities = []

    for e in proper_nouns + years:
        e = e.strip()
        if e and e.lower() not in seen:
            seen.add(e.lower())
            entities.append(e)

    return entities[:max_entities]


class MultiHopRetriever:
    """
    Adaptive bridge-question retriever.

    Used only when failure_detector marks the query as BRIDGE.
    Runs a second retrieval pass using extracted bridge entities.
    """

    def __init__(self, base_retriever: HybridRetriever):
        self.base = base_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = config.MULTIHOP_TOP_K,
    ) -> list[RetrievedChunk]:

        # Pass 1: normal hybrid retrieval
        pass1 = self.base.query(query, top_k=top_k, mode="hybrid")

        # Extract bridge entities from first-pass evidence
        entities = extract_bridge_entities(pass1)

        if not entities:
            return pass1

        # Pass 2: query expansion using bridge entities
        augmented_query = query + " " + " ".join(entities)
        pass2 = self.base.query(augmented_query, top_k=top_k, mode="hybrid")

        # Merge + deduplicate
        seen_ids = set()
        merged: list[RetrievedChunk] = []

        max_len = max(len(pass1), len(pass2))

        for i in range(max_len):
            for result_list in (pass1, pass2):
                if i < len(result_list):
                    chunk = result_list[i]
                    if chunk.chunk_id not in seen_ids:
                        seen_ids.add(chunk.chunk_id)
                        merged.append(chunk)

        # RetrievedChunk uses rrf_score, not hybrid_score
        merged.sort(key=lambda c: c.rrf_score, reverse=True)

        return merged[:top_k]