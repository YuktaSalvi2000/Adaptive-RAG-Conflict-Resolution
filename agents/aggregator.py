"""
aggregator.py  ★ NOVELTY 2 + 3
================================
Novelty 2 — Trust-weighted aggregation:
    Replaces baseline majority vote with weighted vote using AgentWeight
    from trust_scorer.py. Misinformation agents are down-weighted before
    their answers can influence the result.

Novelty 3 — Conflict-aware answer selection:
    Applies different resolution logic depending on FailureType:
        AMBIGUOUS      → _collect_all_valid : surface all distinct answers
        MISINFORMATION → _weighted_vote(penalise_low_trust=True)
        NOISE          → _weighted_vote(min_weight_threshold)
        default        → _weighted_vote standard

Input  : list[AgentTurn] + list[AgentWeight] + FailureType
Output : AggregatedAnswer

Consumed by: answer_generator.py, evaluator.py (conflict_type field)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.config as config
from agents.madam_agent import AgentTurn
from detect.trust_scorer import AgentWeight
from detect.failure_detector import FailureType


# ─────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class AggregatedAnswer:
    """
    Result of trust-weighted aggregation.
    Consumed by answer_generator.py and written to traces.json.
    """
    answer:        str              # winning answer string
    all_answers:   list[str]        # all distinct answers (for AMBIGUOUS)
    conflict_type: str              # 'none'|'ambiguous'|'misinformation'|'noise'|'mixed'
    winning_weight: float           # combined trust weight of the winning answer
    suppressed:    list[str]        # answers that were suppressed (low-trust)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase + strip for answer deduplication."""
    return text.lower().strip().rstrip(".")


def _conflict_label(failure_type: FailureType) -> str:
    """Map FailureType flags to a human-readable conflict_type string."""
    parts = []
    if FailureType.MISINFORMATION in failure_type:
        parts.append("misinformation")
    if FailureType.AMBIGUOUS in failure_type:
        parts.append("ambiguous")
    if FailureType.NOISE in failure_type:
        parts.append("noise")
    if not parts:
        return "none"
    return "|".join(parts)


def _build_weight_map(weights: list[AgentWeight]) -> dict[str, float]:
    """chunk_id → trust weight lookup."""
    return {w.chunk_id: w.weight for w in weights}


# ─────────────────────────────────────────────
# AGGREGATOR  ★
# ─────────────────────────────────────────────

class TrustWeightedAggregator:
    """
    Trust-weighted aggregator with conflict-aware resolution.

    The two novelties implemented here:
      2. Weighted vote  — each agent's vote is multiplied by its trust weight
                          instead of counting equally (baseline majority vote)
      3. Conflict-aware — the aggregation strategy branches on FailureType
                          so the resolution matches the source of the conflict
    """

    def aggregate(
        self,
        turns:        list[AgentTurn],
        weights:      list[AgentWeight],
        failure_type: FailureType,
    ) -> AggregatedAnswer:
        """
        Args:
            turns        : AgentTurn list from madam_agent.run_debate()
            weights      : AgentWeight list from trust_scorer.score() — parallel to turns
            failure_type : FailureType flags from failure_detector.detect()

        Returns:
            AggregatedAnswer with the winning answer + metadata
        """
        if not turns:
            return AggregatedAnswer(
                answer="I don't know",
                all_answers=[],
                conflict_type=_conflict_label(failure_type),
                winning_weight=0.0,
                suppressed=[],
            )

        weight_map = _build_weight_map(weights)

        # ── Novelty 3: branch on conflict type ───────────────
        if FailureType.AMBIGUOUS in failure_type:
            return self._collect_all_valid(turns, weight_map, failure_type)

        elif FailureType.MISINFORMATION in failure_type:
            return self._weighted_vote(
                turns, weight_map, failure_type,
                penalise_low_trust=True,
            )

        elif FailureType.NOISE in failure_type:
            return self._weighted_vote(
                turns, weight_map, failure_type,
                min_weight_threshold=config.MIN_WEIGHT_THRESHOLD,
            )

        else:
            return self._weighted_vote(turns, weight_map, failure_type)

    # ------------------------------------------------------------------
    # Strategy A — AMBIGUOUS: collect all distinct valid answers
    # ------------------------------------------------------------------

    def _collect_all_valid(
        self,
        turns:        list[AgentTurn],
        weight_map:   dict[str, float],
        failure_type: FailureType,
    ) -> AggregatedAnswer:
        """
        Novelty 3 — AMBIGUOUS resolution:
        Surface ALL distinct answers that come from agents above a minimum
        trust weight. Present them together rather than forcing one winner.
        """
        seen:     set[str]  = set()
        answers:  list[str] = []
        suppressed: list[str] = []

        # Sort by weight descending so highest-trust answers appear first
        ranked_turns = sorted(
            turns,
            key=lambda t: weight_map.get(t.chunk_id, 0.0),
            reverse=True,
        )

        for turn in ranked_turns:
            w     = weight_map.get(turn.chunk_id, 0.0)
            norm  = _normalise(turn.answer)

            if turn.answer.lower() in ("i don't know", "unknown", ""):
                continue

            if w < config.MIN_WEIGHT_THRESHOLD:
                suppressed.append(turn.answer)
                continue

            if norm not in seen:
                seen.add(norm)
                answers.append(turn.answer)

        if not answers:
            answers = ["I don't know"]

        # Primary answer = highest-trust answer; all_answers carries the full set
        primary        = answers[0]
        winning_weight = weight_map.get(ranked_turns[0].chunk_id, 0.0) if ranked_turns else 0.0

        return AggregatedAnswer(
            answer         = primary,
            all_answers    = answers,
            conflict_type  = _conflict_label(failure_type),
            winning_weight = winning_weight,
            suppressed     = suppressed,
        )

    # ------------------------------------------------------------------
    # Strategy B — Weighted vote (default + MISINFORMATION + NOISE)
    # ------------------------------------------------------------------

    def _weighted_vote(
        self,
        turns:                list[AgentTurn],
        weight_map:           dict[str, float],
        failure_type:         FailureType,
        penalise_low_trust:   bool  = False,    # MISINFORMATION mode
        min_weight_threshold: float = 0.0,      # NOISE mode
    ) -> AggregatedAnswer:
        """
        Novelty 2 — Trust-weighted majority vote:
        Each agent's vote for its answer is weighted by its trust score.
        Agents below the threshold are suppressed entirely.

        penalise_low_trust=True  (MISINFORMATION):
            Agents with weight < MIN_WEIGHT_THRESHOLD get vote weight = 0.
        min_weight_threshold > 0 (NOISE):
            Same mechanism but threshold comes from config.MIN_WEIGHT_THRESHOLD.
        """
        vote_totals: dict[str, float] = defaultdict(float)   # norm_answer → total weight
        answer_map:  dict[str, str]   = {}                    # norm → original casing
        suppressed:  list[str]        = []
        all_answers: list[str]        = []

        effective_threshold = (
            config.MIN_WEIGHT_THRESHOLD
            if (penalise_low_trust or min_weight_threshold > 0.0)
            else 0.0
        )

        for turn in turns:
            if turn.answer.lower() in ("i don't know", "unknown", ""):
                continue

            w    = weight_map.get(turn.chunk_id, 0.0)
            norm = _normalise(turn.answer)

            # Track all distinct answers seen
            if norm not in answer_map:
                answer_map[norm] = turn.answer
                all_answers.append(turn.answer)

            # Suppress agents below threshold (Novelty 2 — weighted not equal)
            if w < effective_threshold:
                suppressed.append(turn.answer)
                continue

            # Cast weighted vote (Novelty 2 — weight × confidence)
            vote_totals[norm] += w * turn.confidence

        if not vote_totals:
            # All agents suppressed or no valid answers — fall back to highest weight
            best = max(turns, key=lambda t: weight_map.get(t.chunk_id, 0.0))
            return AggregatedAnswer(
                answer         = best.answer or "I don't know",
                all_answers    = all_answers,
                conflict_type  = _conflict_label(failure_type),
                winning_weight = weight_map.get(best.chunk_id, 0.0),
                suppressed     = suppressed,
            )

        # Winning answer = highest cumulative weighted vote
        winning_norm   = max(vote_totals, key=lambda k: vote_totals[k])
        winning_answer = answer_map[winning_norm]
        winning_weight = vote_totals[winning_norm]

        return AggregatedAnswer(
            answer         = winning_answer,
            all_answers    = all_answers,
            conflict_type  = _conflict_label(failure_type),
            winning_weight = winning_weight,
            suppressed     = suppressed,
        )


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from agents.madam_agent import AgentTurn
    from detect.trust_scorer import AgentWeight
    from detect.failure_detector import FailureType

    print("\n" + "═" * 60)
    print("  aggregator.py — self test")
    print("═" * 60)

    agg = TrustWeightedAggregator()

    # Shared setup: 3 agents, 2 say Delhi (gold), 1 says London (misinfo)
    turns = [
        AgentTurn(chunk_id="c0", doc_id="gold_A", label="gold",
                  answer="Delhi", confidence=0.9,
                  retrieval_score=0.031, reranker_score=0.9),
        AgentTurn(chunk_id="c1", doc_id="gold_B", label="gold",
                  answer="Delhi", confidence=0.85,
                  retrieval_score=0.028, reranker_score=0.85),
        AgentTurn(chunk_id="c2", doc_id="misinfo", label="misinfo",
                  answer="London", confidence=0.8,
                  retrieval_score=0.011, reranker_score=0.3),
    ]
    weights = [
        AgentWeight(chunk_id="c0", weight=0.92, retrieval_score=1.0,
                    reranker_score=1.0, agreement_score=0.9),
        AgentWeight(chunk_id="c1", weight=0.88, retrieval_score=0.88,
                    reranker_score=0.92, agreement_score=0.87),
        AgentWeight(chunk_id="c2", weight=0.18, retrieval_score=0.2,
                    reranker_score=0.1, agreement_score=0.05),
    ]

    # ── Test 1: default weighted vote ───────────────────────
    result = agg.aggregate(turns, weights, FailureType.NONE)
    print(f"\n  [default]        answer='{result.answer}'  "
          f"conflict='{result.conflict_type}'  suppressed={result.suppressed}")
    assert result.answer == "Delhi", f"FAIL: expected Delhi, got {result.answer}"
    print("  ✅  Weighted vote correct (Delhi wins)")

    # ── Test 2: MISINFORMATION — London suppressed ───────────
    result = agg.aggregate(turns, weights, FailureType.MISINFORMATION)
    print(f"\n  [MISINFORMATION] answer='{result.answer}'  "
          f"suppressed={result.suppressed}")
    assert result.answer == "Delhi"
    assert "London" in result.suppressed, "FAIL: London should be suppressed"
    print("  ✅  Misinformation suppressed correctly")

    # ── Test 3: AMBIGUOUS — all answers surfaced ─────────────
    turns_amb = [
        AgentTurn(chunk_id="c0", doc_id="d0", label="gold",
                  answer="Mahesh Bhatt", confidence=0.9,
                  retrieval_score=0.03, reranker_score=0.9),
        AgentTurn(chunk_id="c1", doc_id="d1", label="gold",
                  answer="Raj Kapoor", confidence=0.8,
                  retrieval_score=0.025, reranker_score=0.8),
    ]
    weights_amb = [
        AgentWeight(chunk_id="c0", weight=0.9, retrieval_score=1.0,
                    reranker_score=1.0, agreement_score=0.5),
        AgentWeight(chunk_id="c1", weight=0.8, retrieval_score=0.8,
                    reranker_score=0.8, agreement_score=0.5),
    ]
    result = agg.aggregate(turns_amb, weights_amb, FailureType.AMBIGUOUS)
    print(f"\n  [AMBIGUOUS]      all_answers={result.all_answers}")
    assert len(result.all_answers) == 2, "FAIL: should surface both answers"
    print("  ✅  Both ambiguous answers surfaced")

    # ── Test 4: NOISE — low-weight agent ignored ─────────────
    result = agg.aggregate(turns, weights, FailureType.NOISE)
    print(f"\n  [NOISE]          answer='{result.answer}'  "
          f"suppressed={result.suppressed}")
    assert result.answer == "Delhi"
    print("  ✅  Noise agent suppressed, correct answer wins")

    # ── Test 5: empty turns ──────────────────────────────────
    result = agg.aggregate([], [], FailureType.NONE)
    assert result.answer == "I don't know"
    print("\n  ✅  Empty turns handled gracefully")

    print("\n" + "═" * 60)
    print("  ✅  All aggregator tests passed")
    print("═" * 60 + "\n")