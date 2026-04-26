from __future__ import annotations

import os
import time
import random
import re
import sys
from pathlib import Path
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data.config as config

_client: OpenAI | None = None
_LLM_DISABLED = False
_disable_reason = ""


def _get_client() -> OpenAI | None:
    global _client, _LLM_DISABLED, _disable_reason

    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()

        if not api_key:
            _LLM_DISABLED = True
            _disable_reason = "Missing OPENAI_API_KEY"
            print(f"  [llm_client] ⚠  {_disable_reason}")
            return None

        _client = OpenAI(api_key=api_key)

    return _client


def is_disabled() -> bool:
    return _LLM_DISABLED


def disable_reason() -> str:
    return _disable_reason


def _rate_limit_pause() -> None:
    """
    Mandatory pause before every LLM call.
    Controlled by config.LLM_CALL_SLEEP (default 0.5s).
    At 0.5s → ~2 calls/sec → ~120/min, safely under OpenAI tier-1 limits.
    Increase to 1.0 if 429s persist.
    """
    pause = getattr(config, "LLM_CALL_SLEEP", 0.5)
    if pause > 0:
        time.sleep(pause)


def _call_api(messages: list[dict], max_tokens: int = config.LLM_MAX_TOKENS) -> str:
    global _LLM_DISABLED, _disable_reason

    if _LLM_DISABLED:
        return ""

    client = _get_client()
    if client is None:
        return ""

    base_delay = getattr(config, "BACKOFF_BASE",    2.0)
    max_delay  = getattr(config, "BACKOFF_MAX",     30.0)
    retries    = getattr(config, "BACKOFF_RETRIES", 5)
    model      = getattr(config, "LLM_MODEL",       "gpt-4o-mini")

    for attempt in range(1, retries + 1):
        try:
            _rate_limit_pause()

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            code = getattr(e, "status_code", None)

            # Permanent circuit breaker — bad key / no billing / forbidden
            if code in (401, 402, 403):
                _LLM_DISABLED = True
                _disable_reason = f"HTTP {code} — LLM permanently disabled"
                print(f"  [llm_client] ⚠  {_disable_reason}")
                return ""

            # Rate limit — exponential backoff with jitter
            if code == 429:
                wait = min(
                    base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 1.0),
                    max_delay,
                )
                print(f"  [llm_client] 429 rate-limit "
                      f"(attempt {attempt}/{retries}), waiting {wait:.1f}s")
                time.sleep(wait)
                continue

            # Any other error — log and retry
            print(f"  [llm_client] API error attempt {attempt}/{retries}: {e}")
            if attempt == retries:
                return ""
            wait = min(
                base_delay * (2 ** (attempt - 1)) + random.uniform(0.2, 1.0),
                max_delay,
            )
            time.sleep(wait)

    return ""


def call(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """General-purpose single-turn call."""
    return _call_api([
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ])


def score_relevance(question: str, doc_text: str) -> float:
    """
    Pointwise relevance scorer for reranker.py.
    Returns float in [0.0, 1.0]. Falls back to 0.0 if LLM unavailable.
    """
    if _LLM_DISABLED:
        return 0.0

    system = (
        "You are a relevance scoring assistant. "
        "Given a question and a passage, return only one integer from 0 to 10. "
        "No explanation. Just the number."
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Passage:\n{doc_text[:800]}\n\n"
        f"Relevance score (0-10):"
    )

    raw = _call_api([
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ], max_tokens=5)

    match = re.search(r"\b(10|[0-9])\b", raw.strip())
    if not match:
        return 0.0
    return max(0.0, min(float(match.group(1)) / 10.0, 1.0))


def debate_response(
    question: str,
    doc_text: str,
    history:  list[dict] | None = None,
) -> str:
    """
    Per-document agent response for madam_agent.py.
    Falls back to empty string if LLM unavailable.
    """
    if _LLM_DISABLED:
        return ""

    history_str = ""
    if history:
        history_str = "\n\nDebate so far:\n" + "\n".join(
            f"  Agent {h['agent']}: '{h['answer']}'  confidence={h['confidence']:.2f}"
            for h in history
        )

    system = (
        "You are a fact-checking agent in a multi-agent debate. "
        "Use only your document. "
        "If the answer is not in your document, say exactly: I don't know. "
        "Return only the shortest answer phrase — not a full sentence."
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Your document:\n{doc_text[:1000]}"
        f"{history_str}\n\n"
        f"Answer using only your document:"
    )

    return _call_api([
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ], max_tokens=getattr(config, "DEBATE_MAX_TOKENS", 32))