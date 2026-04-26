"""
answer_generator.py
===================
Three-tier answer generation with fallback chain:

    Tier 1 — LLM polish  : take aggregated answer, ask LLM to clean it up
                            using the retrieved context for faithfulness
    Tier 2 — LLM direct  : if polish fails, ask LLM to answer from scratch
                            using the top chunks as context
    Tier 3 — Rule-based  : if LLM is unavailable, return the trust-weighted
                            aggregated answer directly (not raw majority vote)

The trust-weighted aggregated answer (from aggregator.py) is the rule-based
fallback — this is better than baseline which fell back to raw majority vote.

Input  : question (str) + AggregatedAnswer
Output : final answer string

Consumed by: main.py → written to per_query.csv + traces.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.config as config
import detect.llm_client as llm_client
from agents.aggregator import AggregatedAnswer


# ─────────────────────────────────────────────
# ANSWER GENERATOR
# ─────────────────────────────────────────────

class AnswerGenerator:
    """
    Three-tier answer generation with graceful fallback.

    Tier 1 (LLM polish):
        Uses the aggregated answer as a starting point and asks the LLM
        to verify and clean it up. This catches cases where the aggregated
        answer has the right content but wrong formatting or trailing noise.

    Tier 2 (LLM direct):
        Falls back to asking the LLM directly if polish fails or returns
        an empty string. Uses the aggregated answer + conflict context
        as a hint so the LLM knows what the agents found.

    Tier 3 (rule-based):
        Returns the trust-weighted aggregated answer as-is. This is already
        better than the baseline's fallback because it uses weighted vote
        rather than simple majority, so misinformation agents are already
        down-weighted before we reach this tier.
    """

    def generate(self, question: str, agg: AggregatedAnswer) -> str:
        """
        Args:
            question : the original question
            agg      : AggregatedAnswer from aggregator.py

        Returns:
            Final answer string.
        """
        # ── Tier 1: LLM polish ───────────────────────────────
        if not llm_client.is_disabled():
            polished = self._llm_polish(question, agg)
            if polished and polished.lower() not in ("i don't know", "unknown", ""):
                return polished

        # ── Tier 2: LLM direct ───────────────────────────────
        if not llm_client.is_disabled():
            direct = self._llm_direct(question, agg)
            if direct and direct.lower() not in ("i don't know", "unknown", ""):
                return direct

        # ── Tier 3: Rule-based fallback ──────────────────────
        return self._rule_based(agg)

    # ------------------------------------------------------------------
    # Tier 1 — LLM polish
    # ------------------------------------------------------------------

    def _llm_polish(self, question: str, agg: AggregatedAnswer) -> str:
        """
        Ask the LLM to verify and clean the aggregated answer.
        Short prompt — just fix formatting, don't change the content.
        """
        system = (
            "You are a precise answer extractor. "
            "Given a question and a candidate answer, return the most concise "
            "correct answer. If the candidate answer is already correct and concise, "
            "return it as-is. "
            "Return ONLY the answer — no explanation, no punctuation at the end."
        )

        # For ambiguous questions, show all valid answers
        candidate = (
            " / ".join(agg.all_answers)
            if len(agg.all_answers) > 1
            else agg.answer
        )

        conflict_note = ""
        if agg.conflict_type != "none":
            conflict_note = (
                f"\nNote: This question had a {agg.conflict_type} conflict. "
                f"The following answers were suppressed as untrustworthy: "
                f"{agg.suppressed[:3]}"
                if agg.suppressed else ""
            )

        prompt = (
            f"Question: {question}\n"
            f"Candidate answer: {candidate}"
            f"{conflict_note}\n\n"
            f"Final answer:"
        )

        return llm_client.call(prompt, system)

    # ------------------------------------------------------------------
    # Tier 2 — LLM direct
    # ------------------------------------------------------------------

    def _llm_direct(self, question: str, agg: AggregatedAnswer) -> str:
        """
        Ask the LLM to answer directly, using the aggregated answer as a hint.
        This acts as a sanity check when Tier 1 returns empty or garbage.
        """
        system = (
            "You are a helpful question-answering assistant. "
            "Answer the question in one short phrase or sentence. "
            "Return ONLY the answer."
        )

        prompt = (
            f"Question: {question}\n\n"
            f"Based on retrieved evidence, the likely answer is: {agg.answer}\n\n"
            f"Please provide the final concise answer:"
        )

        return llm_client.call(prompt, system)

    # ------------------------------------------------------------------
    # Tier 3 — Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based(self, agg: AggregatedAnswer) -> str:
        """
        Return the trust-weighted aggregated answer directly.
        For AMBIGUOUS questions, join all valid answers with ' / '.
        This is better than baseline majority vote because the aggregated
        answer already has misinformation agents down-weighted.
        """
        if agg.all_answers and len(agg.all_answers) > 1:
            return " / ".join(agg.all_answers[:3])   # cap at 3 for readability
        return agg.answer if agg.answer else "I don't know"


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from agents.aggregator import AggregatedAnswer

    print("\n" + "═" * 60)
    print("  answer_generator.py — self test (rule-based tier)")
    print("═" * 60)

    gen = AnswerGenerator()

    # ── Test 1: standard answer ──────────────────────────────
    agg1 = AggregatedAnswer(
        answer="Delhi", all_answers=["Delhi"],
        conflict_type="misinformation", winning_weight=0.92,
        suppressed=["London"],
    )
    out1 = gen.generate("What city is the Oberoi Group headquartered in?", agg1)
    print(f"\n  [standard]    output='{out1}'")
    assert out1, "FAIL: output should not be empty"
    print("  ✅  Standard answer generated")

    # ── Test 2: ambiguous — multiple answers joined ───────────
    agg2 = AggregatedAnswer(
        answer="Mahesh Bhatt", all_answers=["Mahesh Bhatt", "Raj Kapoor"],
        conflict_type="ambiguous", winning_weight=0.9,
        suppressed=[],
    )
    out2 = gen.generate("Who directed Lahu Ke Do Rang?", agg2)
    print(f"\n  [ambiguous]   output='{out2}'")
    assert out2, "FAIL: output should not be empty"
    print("  ✅  Ambiguous answer generated")

    # ── Test 3: empty aggregated answer ──────────────────────
    agg3 = AggregatedAnswer(
        answer="", all_answers=[],
        conflict_type="none", winning_weight=0.0,
        suppressed=[],
    )
    out3 = gen.generate("What sport is Bobby Carpenter associated with?", agg3)
    print(f"\n  [empty agg]   output='{out3}'")
    assert out3 == "I don't know", f"FAIL: expected fallback, got '{out3}'"
    print("  ✅  Empty aggregation handled gracefully")

    print("\n" + "═" * 60)
    print("  ✅  All answer_generator tests passed")
    print("═" * 60 + "\n")