"""
madam_agent.py
==============
Per-document agent logic (MADAM-RAG style).

Each chunk from AdaptiveResult gets its own agent. The agent reads only
its chunk + the question, produces an answer + confidence, then participates
in DEBATE_ROUNDS rounds where it can see other agents' answers.

Input  : AdaptiveResult.chunks (list[RankedChunk]) + weights (list[AgentWeight])
Output : list[AgentTurn]  — one per chunk, carries answer + confidence + scores

Consumed by: aggregator.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.config as config
import detect.llm_client as llm_client
from retrieval.reranker import RankedChunk
from detect.trust_scorer import AgentWeight


# ─────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class AgentTurn:
    """
    One agent's final answer after all debate rounds.
    Parallel to the chunks/weights lists from AdaptiveResult.
    """
    chunk_id:        str
    doc_id:          str
    label:           str | None     # gold | misinfo | noise | None
    answer:          str            # extracted answer string
    confidence:      float          # self-reported confidence in [0, 1]
    retrieval_score: float          # rrf_score from stage2 (passed through)
    reranker_score:  float          # LLM reranker score (passed through)
    debate_history:  list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────
# ANSWER EXTRACTION HELPERS
# ─────────────────────────────────────────────

def _extract_confidence(response: str) -> float:
    """
    Parse a confidence value from the LLM response.
    Looks for patterns like: "Confidence: 0.8", "confidence: 85%", "conf=0.7"
    Falls back to 0.5 if nothing parseable is found.
    """
    patterns = [
        r"confidence[:\s=]+([0-9]+\.?[0-9]*)\s*%",   # "confidence: 85%"
        r"confidence[:\s=]+([0-9]*\.?[0-9]+)",         # "confidence: 0.85"
        r"conf[:\s=]+([0-9]*\.?[0-9]+)",               # "conf=0.85"
    ]
    for pat in patterns:
        m = re.search(pat, response.lower())
        if m:
            val = float(m.group(1))
            return min(val / 100.0, 1.0) if val > 1.0 else val
    return 0.5


def _extract_answer(response: str) -> str:
    """
    Pull the answer out of the LLM response.
    Looks for "Answer: ..." prefix; falls back to first non-empty line.
    """
    for line in response.splitlines():
        line = line.strip()
        if line.lower().startswith("answer:"):
            return line[len("answer:"):].strip()
        if line.lower().startswith("my answer:"):
            return line[len("my answer:"):].strip()
    # Fallback: first non-empty line
    for line in response.splitlines():
        if line.strip():
            return line.strip()
    return response.strip()


def _rule_based_answer(question: str, chunk_text: str) -> tuple[str, float]:
    """
    Rule-based fallback when LLM is unavailable.
    Extracts the first sentence of the chunk as a proxy answer.
    Confidence is low (0.3) since this is a last resort.
    """
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text.strip())
    answer = sentences[0] if sentences else chunk_text[:100]
    return answer.strip(), 0.3


# ─────────────────────────────────────────────
# SINGLE AGENT
# ─────────────────────────────────────────────

def _run_agent(
    question:    str,
    chunk:       RankedChunk,
    history:     list[dict],
    round_num:   int,
) -> tuple[str, float]:
    """
    Run one agent for one debate round.

    Args:
        question  : the original question
        chunk     : the agent's document
        history   : other agents' answers from previous rounds
        round_num : current round (0-indexed)

    Returns:
        (answer, confidence)
    """
    if llm_client.is_disabled():
        return _rule_based_answer(question, chunk.text)

    # Build debate-aware prompt
    system = (
        "You are a fact-checking agent in a multi-agent debate. "
        "You have access to exactly one document passage. "
        "Extract the best answer to the question from your passage only. "
        "If your passage does not contain the answer, say 'I don't know'. "
        "Format your response as:\n"
        "Answer: <your answer>\n"
        "Confidence: <0.0 to 1.0>\n"
        "Be concise. One sentence maximum for the answer."
    )

    history_block = ""
    if history and round_num > 0:
        lines = [
            f"  Agent {h['agent_id']} said: '{h['answer']}' "
            f"(confidence {h['confidence']:.2f})"
            for h in history[-6:]   # cap history to last 6 to avoid token overflow
        ]
        history_block = "\n\nOther agents' answers so far:\n" + "\n".join(lines)
        history_block += (
            "\n\nConsider whether your document supports or contradicts these answers. "
            "You may revise your answer based on the debate."
        )

    prompt = (
        f"Question: {question}\n\n"
        f"Your document:\n{chunk.text[:800]}"
        f"{history_block}\n\n"
        f"Your response:"
    )

    raw = llm_client._call_api([
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ])

    if not raw:
        return _rule_based_answer(question, chunk.text)

    answer     = _extract_answer(raw)
    confidence = _extract_confidence(raw)
    return answer, confidence


# ─────────────────────────────────────────────
# MADAM AGENT  (multi-agent debate)
# ─────────────────────────────────────────────

class MadamAgent:
    """
    Runs multi-agent debate across all chunks.

    Round structure:
        Round 0: each agent answers independently (no history)
        Round 1+: each agent sees all previous answers and can revise

    The final AgentTurn for each chunk reflects its answer after all rounds.
    """

    def run_debate(
        self,
        question: str,
        chunks:   list[RankedChunk],
        weights:  list[AgentWeight],
        rounds:   int = config.DEBATE_ROUNDS,
    ) -> list[AgentTurn]:
        """
        Args:
            question : the original question
            chunks   : from AdaptiveResult.chunks  (one agent per chunk)
            weights  : from AdaptiveResult.weights (parallel list)
            rounds   : number of debate rounds (config.DEBATE_ROUNDS = 3)

        Returns:
            list[AgentTurn] — one per chunk, in same order as chunks/weights
        """
        if not chunks:
            return []

        # Cap agents to avoid excessive API calls
        n_agents = min(len(chunks), config.MAX_AGENTS)
        active_chunks  = chunks[:n_agents]
        active_weights = weights[:n_agents]

        # Track current answer + confidence per agent across rounds
        current_answers:     list[str]   = [""] * n_agents
        current_confidences: list[float] = [0.5] * n_agents
        all_histories:       list[list[dict]] = [[] for _ in range(n_agents)]

        for round_num in range(rounds):
            # Shared history = all agents' answers from the PREVIOUS round
            shared_history = [
                {
                    "agent_id":   i,
                    "answer":     current_answers[i],
                    "confidence": current_confidences[i],
                }
                for i in range(n_agents)
                if round_num > 0 and current_answers[i]   # skip empty first round
            ]

            for i, (chunk, weight) in enumerate(zip(active_chunks, active_weights)):
                answer, confidence = _run_agent(
                    question=question,
                    chunk=chunk,
                    history=shared_history,
                    round_num=round_num,
                )
                current_answers[i]     = answer
                current_confidences[i] = confidence
                all_histories[i].append({
                    "round":      round_num,
                    "answer":     answer,
                    "confidence": confidence,
                })

        # Build final AgentTurn objects
        turns = []
        for i, (chunk, weight) in enumerate(zip(active_chunks, active_weights)):
            turns.append(AgentTurn(
                chunk_id        = chunk.chunk_id,
                doc_id          = chunk.doc_id,
                label           = chunk.label,
                answer          = current_answers[i],
                confidence      = current_confidences[i],
                retrieval_score = chunk.rrf_score,
                reranker_score  = chunk.reranker_score,
                debate_history  = all_histories[i],
            ))

        return turns


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from retrieval.reranker import RankedChunk
    from detect.trust_scorer import AgentWeight, TrustScorer

    print("\n" + "═" * 60)
    print("  madam_agent.py — self test (LLM disabled fallback)")
    print("═" * 60)

    fake_chunks = [
        RankedChunk(
            chunk_id="chunk_00000", doc_id="gold_A", qid="q1",
            source="ramdocs", label="gold",
            text="The Oberoi Group is a hotel company with its head office in Delhi.",
            dense_score=0.82, bm25_score=14.3, rrf_score=0.031, reranker_score=0.9,
        ),
        RankedChunk(
            chunk_id="chunk_00001", doc_id="misinfo_X", qid="q1",
            source="ramdocs", label="misinfo",
            text="The Oberoi Group headquarters are located in London, United Kingdom.",
            dense_score=0.41, bm25_score=5.2, rrf_score=0.011, reranker_score=0.3,
        ),
    ]

    scorer  = TrustScorer()
    weights = scorer.score(fake_chunks)
    agent   = MadamAgent()

    turns = agent.run_debate(
        question="What city is the Oberoi Group headquartered in?",
        chunks=fake_chunks,
        weights=weights,
        rounds=1,
    )

    print(f"\n  Turns produced: {len(turns)}")
    for t in turns:
        print(f"\n  chunk_id   : {t.chunk_id}")
        print(f"  label      : {t.label}")
        print(f"  answer     : {t.answer}")
        print(f"  confidence : {t.confidence}")

    assert len(turns) == len(fake_chunks), "FAIL: one turn per chunk"
    assert all(t.answer for t in turns), "FAIL: all turns must have an answer"
    assert all(0.0 <= t.confidence <= 1.0 for t in turns), "FAIL: confidence in [0,1]"

    print("\n  ✅  All assertions passed")
    print("═" * 60 + "\n")