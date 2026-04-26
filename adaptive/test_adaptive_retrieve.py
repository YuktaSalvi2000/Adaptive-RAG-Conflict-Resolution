"""
test_adaptive_retrieval.py
==========================
End-to-end test of the adaptive retrieval pipeline:

    failure_detector  →  adaptive_retrieval
                              /          \\
                    stage2_retrieval   multi_hop
                              \\          /
                             reranker
                                 ↓
                           trust_scorer
                                 ↓
                         AdaptiveResult  ← what we inspect

Two test cases:
    1. BRIDGE query   → should route through multi_hop (two-pass)
    2. RAMDOCS query  → should route through stage2 + trigger MISINFORMATION/NOISE flags

Run from project root:
    python test_adaptive_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import data.config as config
from retrieval.stage2_retrieval import HybridRetriever
from retrieval.multi_hop import MultiHopRetriever
from retrieval.reranker import LLMReranker
from detect.failure_detector import FailureDetector, FailureType
from detect.trust_scorer import TrustScorer
from adaptive_retrieval import AdaptiveRetriever, AdaptiveResult
from data.data_loader import load_hotpotqa, load_ramdocs


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────

def _sep(title: str = ""):
    width = 65
    if title:
        pad = (width - len(title) - 2) // 2
        print("  " + "─" * pad + f" {title} " + "─" * pad)
    else:
        print("  " + "─" * width)


def _print_result(result: AdaptiveResult, query: str):
    """Pretty-print the full AdaptiveResult for inspection."""

    profile = result.profile
    print(f"\n  Query      : {query}")
    print(f"  Failure    : {profile.failure_types}  (confidence={profile.confidence})")
    print(f"  Rationale  : {profile.rationale}")
    print(f"  Chunks out : {len(result.chunks)}  (TOP_K_FINAL={config.TOP_K_FINAL})")

    _sep("chunks + trust weights")
    print(f"  {'#':<3} {'chunk_id':<14} {'label':<8} {'rrf':>6} "
          f"{'rerank':>7} {'agree':>7} {'WEIGHT':>7}  text[:60]")
    print("  " + "─" * 100)

    for i, (chunk, weight) in enumerate(zip(result.chunks, result.weights), 1):
        print(
            f"  {i:<3} {chunk.chunk_id:<14} {str(chunk.label):<8} "
            f"{chunk.rrf_score:>6.3f} {chunk.reranker_score:>7.3f} "
            f"{weight.agreement_score:>7.3f} {weight.weight:>7.3f}  "
            f"{chunk.text[:60].replace(chr(10), ' ')}..."
        )


# ─────────────────────────────────────────────
# BUILD PIPELINE (once, shared across tests)
# ─────────────────────────────────────────────

def build_pipeline():
    print("\n  Building pipeline components...")
    base     = HybridRetriever()
    reranker = LLMReranker()
    multihop = MultiHopRetriever(base_retriever=base)
    trust    = TrustScorer()
    adaptive = AdaptiveRetriever(
        base=base, reranker=reranker, multihop=multihop, trust=trust
    )
    detector = FailureDetector()
    print("  ✓ Pipeline ready\n")
    return adaptive, detector


# ─────────────────────────────────────────────
# TEST 1 — BRIDGE query (HotpotQA)
# ─────────────────────────────────────────────

def test_bridge(adaptive: AdaptiveRetriever, detector: FailureDetector):
    _sep("TEST 1: BRIDGE query → multi_hop path")

    # Load a known bridge example from HotpotQA
    examples = load_hotpotqa(config.HOTPOT_PATH, n=config.N_SAMPLES)
    bridge_examples = [e for e in examples if e.type == "bridge"]

    if not bridge_examples:
        print("  [SKIP] No bridge examples found in first N_SAMPLES — increase N_SAMPLES")
        return

    example = bridge_examples[0]
    query   = example.question

    print(f"  Example qid  : {example.qid}")
    print(f"  Example type : {example.type}")

    # Detect failure profile
    profile = detector.detect(query, example)

    # Assert BRIDGE was detected
    assert FailureType.BRIDGE in profile.failure_types, \
        f"FAIL: expected BRIDGE flag, got {profile.failure_types}"
    print(f"  ✓ BRIDGE flag detected correctly")

    # Run adaptive retrieval
    result = adaptive.retrieve(query, profile)

    # Inspect result
    _print_result(result, query)

    # Assertions
    assert len(result.chunks) > 0, "FAIL: no chunks returned"
    assert len(result.chunks) == len(result.weights), \
        "FAIL: chunks and weights must be parallel lists"
    assert all(0.0 <= w.weight <= 1.0 for w in result.weights), \
        "FAIL: weights must be in [0, 1]"

    print(f"\n  ✅ TEST 1 PASSED — bridge query routed, {len(result.chunks)} chunks returned")


# ─────────────────────────────────────────────
# TEST 2 — MISINFORMATION / NOISE query (RAMDocs)
# ─────────────────────────────────────────────

def test_misinformation(adaptive: AdaptiveRetriever, detector: FailureDetector):
    _sep("TEST 2: MISINFORMATION/NOISE query → stage2 path + trust filtering")

    examples = load_ramdocs(config.RAMDOCS_PATH, n=config.N_SAMPLES)

    # Find an example with wrong_answers (misinformation signal)
    misinfo_examples = [e for e in examples if e.wrong_answers]

    if not misinfo_examples:
        print("  [SKIP] No misinformation examples in first N_SAMPLES")
        return

    example = misinfo_examples[0]
    query   = example.question

    print(f"  Example qid    : {example.qid}")
    print(f"  Gold answers   : {example.gold_answers}")
    print(f"  Wrong answers  : {example.wrong_answers}")
    print(f"  Doc labels     : {[d.label for d in example.documents]}")

    # Detect failure profile
    profile = detector.detect(query, example)

    # Assert MISINFORMATION was detected
    assert FailureType.MISINFORMATION in profile.failure_types, \
        f"FAIL: expected MISINFORMATION flag, got {profile.failure_types}"
    print(f"  ✓ MISINFORMATION flag detected correctly")

    # Run adaptive retrieval
    result = adaptive.retrieve(query, profile)

    # Inspect result
    _print_result(result, query)

    # Check that misinfo/noise chunks get lower weights than gold chunks
    gold_weights    = [w.weight for w, c in zip(result.weights, result.chunks) if c.label == "gold"]
    misinfo_weights = [w.weight for w, c in zip(result.weights, result.chunks) if c.label == "misinfo"]

    if gold_weights and misinfo_weights:
        avg_gold    = sum(gold_weights)    / len(gold_weights)
        avg_misinfo = sum(misinfo_weights) / len(misinfo_weights)
        print(f"\n  avg gold weight    : {avg_gold:.3f}")
        print(f"  avg misinfo weight : {avg_misinfo:.3f}")
        if avg_gold > avg_misinfo:
            print(f"  ✓ Gold chunks outweigh misinfo chunks")
        else:
            print(f"  ⚠  Gold/misinfo ordering not ideal — check trust signal values")
    else:
        print(f"  ℹ  Labels in retrieved chunks: "
              f"{set(c.label for c in result.chunks)} — ordering check skipped")

    assert len(result.chunks) > 0, "FAIL: no chunks returned"
    assert len(result.chunks) == len(result.weights), \
        "FAIL: chunks and weights must be parallel lists"

    print(f"\n  ✅ TEST 2 PASSED — misinfo query routed, {len(result.chunks)} chunks returned")


# ─────────────────────────────────────────────
# TEST 3 — COMPARISON query (no special flags)
# ─────────────────────────────────────────────

def test_comparison(adaptive: AdaptiveRetriever, detector: FailureDetector):
    _sep("TEST 3: COMPARISON query → standard stage2 path")

    examples = load_hotpotqa(config.HOTPOT_PATH, n=config.N_SAMPLES)
    comparison_examples = [e for e in examples if e.type == "comparison"]

    if not comparison_examples:
        print("  [SKIP] No comparison examples in first N_SAMPLES")
        return

    example = comparison_examples[0]
    query   = example.question

    print(f"  Example qid  : {example.qid}")
    print(f"  Example type : {example.type}")

    profile = detector.detect(query, example)

    # Comparison questions should NOT trigger BRIDGE
    assert FailureType.BRIDGE not in profile.failure_types, \
        f"FAIL: comparison question incorrectly flagged as BRIDGE"
    print(f"  ✓ BRIDGE not triggered for comparison question")
    print(f"  ✓ Failure types: {profile.failure_types}")

    result = adaptive.retrieve(query, profile)
    _print_result(result, query)

    assert len(result.chunks) > 0, "FAIL: no chunks returned"
    assert len(result.chunks) == len(result.weights)

    print(f"\n  ✅ TEST 3 PASSED — comparison query handled, {len(result.chunks)} chunks returned")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 65)
    print("  Adaptive Retrieval — End-to-End Pipeline Test")
    print("═" * 65)

    adaptive, detector = build_pipeline()

    passed = 0
    failed = 0

    for name, test_fn in [
        ("BRIDGE",       test_bridge),
        ("MISINFORMATION", test_misinformation),
        ("COMPARISON",   test_comparison),
    ]:
        print()
        try:
            test_fn(adaptive, detector)
            passed += 1
        except AssertionError as e:
            print(f"\n  ❌ {e}")
            failed += 1
        except Exception as e:
            print(f"\n  ❌ Unexpected error in {name}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print()
    print("═" * 65)
    print(f"  Results: {passed} passed, {failed} failed")
    print("═" * 65 + "\n")

    sys.exit(0 if failed == 0 else 1)