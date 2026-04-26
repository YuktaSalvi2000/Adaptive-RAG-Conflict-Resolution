"""
failure_detector.py  ★ NOVELTY
================================
Classifies each query into one or more failure types BEFORE retrieval happens.
This is what makes the pipeline adaptive rather than static.

Every prior system (MADAM-RAG, Self-RAG, RankRAG) applies one fixed strategy
to all queries. This module inspects the query + QueryExample metadata and
returns a FailureProfile that tells adaptive_retrieval.py which strategy to use.

Failure types (combinable via Flag):
    BRIDGE        → multi-hop question — route to multi_hop.py
    MISINFORMATION → conflicting evidence present — activate trust filtering
    NOISE         → irrelevant docs in pool — drop low-scoring chunks
    AMBIGUOUS     → multiple valid answers exist — surface all in aggregator

Feeds into:
    adaptive_retrieval.py  — routing decision (BRIDGE → multihop)
    aggregator.py          — resolution strategy per failure type
    evaluator.py           — conflict_type field in output
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Flag, auto
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.config as config
from data.data_loader import QueryExample


# ─────────────────────────────────────────────
# FAILURE TYPE FLAG
# ─────────────────────────────────────────────

class FailureType(Flag):
    """
    Combinable failure flags. A single query can have multiple types.
    Example: a bridge question in a RAMDocs example with wrong_answers
             gets BRIDGE | MISINFORMATION simultaneously.

    Usage in adaptive_retrieval.py:
        if FailureType.BRIDGE in profile.failure_types: ...
        if FailureType.MISINFORMATION in profile.failure_types: ...
    """
    NONE           = 0
    BRIDGE         = auto()   # multi-hop — needs multi_hop.py second pass
    MISINFORMATION = auto()   # conflicting evidence — needs trust filtering
    NOISE          = auto()   # irrelevant docs dominating — drop low-scorers
    AMBIGUOUS      = auto()   # multiple valid answers — present all


# ─────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class FailureProfile:
    """
    Result of FailureDetector.detect().
    Consumed by adaptive_retrieval.py and aggregator.py.
    """
    failure_types: FailureType
    confidence:    float        # 0–1, how certain the classification is
    rationale:     str          # human-readable explanation (for traces.json)


# ─────────────────────────────────────────────
# FAILURE DETECTOR  ★
# ─────────────────────────────────────────────

class FailureDetector:
    """
    Rule-based failure classifier. Runs BEFORE retrieval on every query.

    Rules are deterministic and dataset-agnostic — they work on both
    HotpotQA (uses .type field) and RAMDocs (uses .wrong_answers, .label).

    Rule-based is sufficient for the paper and swappable for an LLM
    classifier later by replacing the detect() method body.
    """

    # Keywords that signal a bridge (multi-hop) question even when
    # .type is not explicitly set (e.g. RAMDocs has no .type field)
    BRIDGE_KEYWORDS: frozenset[str] = frozenset(config.BRIDGE_KEYWORDS)

    def detect(self, query: str, example: QueryExample) -> FailureProfile:
        """
        Classify a query into one or more FailureTypes.

        Args:
            query   : the question string
            example : QueryExample from data_loader (carries metadata)

        Returns:
            FailureProfile with combined failure_types flag + rationale
        """
        types     = FailureType.NONE
        rationale = []
        q_lower   = query.lower()

        # ── Rule 1: BRIDGE ───────────────────────────────────
        # Signal A: HotpotQA explicitly labels bridge questions
        # Signal B: query contains bridge-indicator keywords
        is_explicit_bridge = (example.type == "bridge")
        has_bridge_keyword = any(kw in q_lower for kw in self.BRIDGE_KEYWORDS)

        if is_explicit_bridge or has_bridge_keyword:
            types |= FailureType.BRIDGE
            if is_explicit_bridge:
                rationale.append("explicit bridge type")
            if has_bridge_keyword:
                matched = [kw for kw in self.BRIDGE_KEYWORDS if kw in q_lower]
                rationale.append(f"bridge keywords: {matched}")

        # ── Rule 2: MISINFORMATION ───────────────────────────
        # RAMDocs provides wrong_answers when misinformation docs are present.
        # Any non-empty wrong_answers list means at least one doc is adversarial.
        if example.wrong_answers:
            types |= FailureType.MISINFORMATION
            rationale.append(
                f"wrong_answers present ({len(example.wrong_answers)} entries)"
            )

        # ── Rule 3: AMBIGUOUS ────────────────────────────────
        # Multiple gold answers = the question has more than one valid answer.
        # aggregator.py will surface all of them instead of picking one.
        if len(example.gold_answers) > 1:
            types |= FailureType.AMBIGUOUS
            rationale.append(
                f"multiple gold answers ({len(example.gold_answers)})"
            )

        # ── Rule 4: NOISE ────────────────────────────────────
        # RAMDocs explicitly labels documents as 'noise'.
        # HotpotQA context contains distractor paragraphs (no label),
        # so for HotpotQA we use a keyword heuristic: if the question is
        # hard-level, distractors are more likely to cause noise issues.
        has_noise_docs  = any(d.label == "noise" for d in example.documents)
        is_hard_hotpot  = (example.source == "hotpotqa" and
                           getattr(example, "level", None) == "hard")

        if has_noise_docs or is_hard_hotpot:
            types |= FailureType.NOISE
            if has_noise_docs:
                n_noise = sum(1 for d in example.documents if d.label == "noise")
                rationale.append(f"noise docs present ({n_noise})")
            if is_hard_hotpot:
                rationale.append("hard-level HotpotQA (distractor noise likely)")

        # ── Confidence ───────────────────────────────────────
        # Rule-based detector is deterministic, so confidence is fixed.
        # An LLM-based classifier would return a calibrated probability here.
        confidence = config.FAILURE_DETECTOR_CONFIDENCE if types != FailureType.NONE else 1.0

        return FailureProfile(
            failure_types=types,
            confidence=confidence,
            rationale="; ".join(rationale) if rationale else "no failure detected",
        )


# ─────────────────────────────────────────────
# SELF-TEST  (python failure_detector.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from data.data_loader import Document, QueryExample

    print("\n" + "═" * 60)
    print("  failure_detector.py — self test")
    print("═" * 60)

    detector = FailureDetector()

    def _run(label: str, example: QueryExample, expected_flags: list[FailureType]):
        profile = detector.detect(example.question, example)
        print(f"\n  [{label}]")
        print(f"    question : {example.question[:70]}")
        print(f"    types    : {profile.failure_types}")
        print(f"    conf     : {profile.confidence}")
        print(f"    rationale: {profile.rationale}")
        for flag in expected_flags:
            assert flag in profile.failure_types, \
                f"FAIL: expected {flag} in {profile.failure_types}"
        print(f"    ✅ flags confirmed: {expected_flags}")

    # ── Test 1: HotpotQA bridge question ────────────────────
    _run(
        label="HotpotQA bridge",
        example=QueryExample(
            qid="hp_001",
            question="The Oberoi family is part of a hotel company that has a head office in what city?",
            gold_answers=["Delhi"],
            documents=[
                Document(doc_id="Oberoi_family", text="The Oberoi family is an Indian family.", label=None),
                Document(doc_id="Oberoi_Group",  text="The Oberoi Group head office is in Delhi.", label=None),
            ],
            type="bridge",
            source="hotpotqa",
        ),
        expected_flags=[FailureType.BRIDGE],
    )

    # ── Test 2: RAMDocs misinformation + noise ───────────────
    _run(
        label="RAMDocs misinfo + noise",
        example=QueryExample(
            qid="ram_001",
            question="What is the population of Broken Bow?",
            gold_answers=["3,559 people"],
            wrong_answers=["10,000 people"],
            documents=[
                Document(doc_id="ram_001_doc_0", text="There were 3,559 people.", label="gold"),
                Document(doc_id="ram_001_doc_1", text="Population is 10,000.", label="misinfo"),
                Document(doc_id="ram_001_doc_2", text="Cricket is a bat-and-ball game.", label="noise"),
            ],
            source="ramdocs",
        ),
        expected_flags=[FailureType.MISINFORMATION, FailureType.NOISE],
    )

    # ── Test 3: Ambiguous (multiple gold answers) ────────────
    _run(
        label="Ambiguous multi-answer",
        example=QueryExample(
            qid="ram_002",
            question="Who directed the film?",
            gold_answers=["Director A", "Director B"],
            documents=[
                Document(doc_id="ram_002_doc_0", text="Directed by Director A.", label="gold"),
                Document(doc_id="ram_002_doc_1", text="Also credited to Director B.", label="gold"),
            ],
            source="ramdocs",
        ),
        expected_flags=[FailureType.AMBIGUOUS],
    )

    # ── Test 4: Keyword bridge (no explicit type) ────────────
    _run(
        label="Keyword bridge (no type field)",
        example=QueryExample(
            qid="ram_003",
            question="What is the nationality of the author of Dune?",
            gold_answers=["American"],
            documents=[
                Document(doc_id="ram_003_doc_0", text="Frank Herbert was an American author.", label="gold"),
            ],
            source="ramdocs",
        ),
        expected_flags=[FailureType.BRIDGE],
    )

    # ── Test 5: Clean question (no failures) ─────────────────
    example_clean = QueryExample(
        qid="hp_002",
        question="Which magazine was started first Arthur's Magazine or First for Women?",
        gold_answers=["Arthur's Magazine"],
        documents=[
            Document(doc_id="Arthurs", text="Arthur's Magazine (1844-1846).", label=None),
            Document(doc_id="FirstForWomen", text="First for Women started in 1989.", label=None),
        ],
        type="comparison",
        source="hotpotqa",
    )
    profile_clean = detector.detect(example_clean.question, example_clean)
    print(f"\n  [Clean comparison]")
    print(f"    question : {example_clean.question[:70]}")
    print(f"    types    : {profile_clean.failure_types}")
    print(f"    rationale: {profile_clean.rationale}")
    assert FailureType.BRIDGE not in profile_clean.failure_types, \
        "FAIL: comparison should not be flagged as BRIDGE"
    print(f"    ✅ BRIDGE correctly absent for comparison question")

    # ── Test 6: Combined flags (bridge + misinfo) ────────────
    _run(
        label="Combined BRIDGE + MISINFORMATION",
        example=QueryExample(
            qid="ram_004",
            question="What is the nationality of the director of Sinister?",
            gold_answers=["American"],
            wrong_answers=["British"],
            documents=[
                Document(doc_id="ram_004_doc_0", text="Scott Derrickson is an American director.", label="gold"),
                Document(doc_id="ram_004_doc_1", text="Scott Derrickson is British.", label="misinfo"),
            ],
            source="ramdocs",
        ),
        expected_flags=[FailureType.BRIDGE, FailureType.MISINFORMATION],
    )

    print("\n" + "═" * 60)
    print("  ✅  All tests passed")
    print("═" * 60 + "\n")