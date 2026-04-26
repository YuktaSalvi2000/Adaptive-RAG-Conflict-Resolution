"""
main.py
=======
Orchestrator — runs the full adaptive RAG pipeline end to end.

Startup  : load data, build all components
Per query: detect → adaptive retrieve → debate → aggregate → generate → evaluate
Output   : results/summary_metrics.csv, results/per_query.csv, results/traces.json
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import data.config as config

# Needed for old BM25 pickle loading
from index.stage1_indexing import BM25Index

# Data
from data.data_loader import load_hotpotqa, load_ramdocs, QueryExample

# Retrieval
from retrieval.stage2_retrieval import HybridRetriever
from retrieval.multi_hop import MultiHopRetriever
from retrieval.reranker import LLMReranker

# Detection + trust
from detect.failure_detector import FailureDetector
from detect.trust_scorer import TrustScorer
from adaptive.adaptive_retrieval import AdaptiveRetriever

# Agents + resolution
from agents.madam_agent import MadamAgent
from agents.aggregator import TrustWeightedAggregator
from agents.answer_generator import AnswerGenerator

# Evaluation
from adaptive.metrics import (
    answer_em, answer_f1, supporting_fact_f1,
    joint_em, joint_f1, recall_at_k,
    multi_doc_hit, mrr, misinformation_suppressed,
    ambiguity_coverage, faithfulness_score,
)
from adaptive.evaluator import Evaluator


def log(msg: str, symbol: str = "▸"):
    print(f"  {symbol} {msg}", flush=True)


def _ensure_dirs():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _check_index():
    missing = [
        p for p in [
            config.CHUNKS_PATH,
            config.DENSE_MATRIX_PATH,
            config.DENSE_VOCAB_PATH,
            config.BM25_CORPUS_PATH,
        ]
        if not Path(p).exists()
    ]

    if missing:
        print("\n  ❌ Index artefacts missing:")
        for p in missing:
            print(f"     {p}")
        print("\n  Run first: python index/stage1_indexing.py\n")
        sys.exit(1)


def build_components():
    log("Building pipeline components...")

    base = HybridRetriever()
    reranker = LLMReranker()
    multihop = MultiHopRetriever(base_retriever=base)
    trust = TrustScorer()

    adaptive = AdaptiveRetriever(
        base=base,
        reranker=reranker,
        multihop=multihop,
        trust=trust,
    )

    detector = FailureDetector()
    agent = MadamAgent()
    aggregator = TrustWeightedAggregator()
    generator = AnswerGenerator()
    evaluator = Evaluator()

    log("All components ready")
    return adaptive, detector, agent, aggregator, generator, evaluator


def process_example(
    example: QueryExample,
    adaptive: AdaptiveRetriever,
    detector: FailureDetector,
    agent: MadamAgent,
    aggregator: TrustWeightedAggregator,
    generator: AnswerGenerator,
    evaluator: Evaluator,
) -> dict:

    query = example.question
    t0 = time.time()

    # Novelty 1: detect failure type before retrieval
    profile = detector.detect(query, example)

    # Novelty 1: route retrieval based on failure type
    result = adaptive.retrieve(query, profile)

    # MADAM-style per-document agents
    turns = agent.run_debate(
        question=query,
        chunks=result.chunks,
        weights=result.weights,
        rounds=config.DEBATE_ROUNDS,
    )

    # Novelty 3: conflict-aware aggregation using trust weights
    agg = aggregator.aggregate(
        turns=turns,
        weights=result.weights,
        failure_type=profile.failure_types,
    )

    # Final answer
    answer = generator.generate(query, agg)

    retrieved_doc_ids = [c.doc_id for c in result.chunks]

    metrics = {
        "answer_em": answer_em(answer, example.gold_answers),
        "answer_f1": answer_f1(answer, example.gold_answers),
        "supporting_fact_f1": supporting_fact_f1(answer, example.supporting_facts),
        "joint_em": joint_em(answer, example.gold_answers, example.supporting_facts),
        "joint_f1": joint_f1(answer, example.gold_answers, example.supporting_facts),
        "recall_at_k": recall_at_k(
            retrieved_doc_ids,
            example.gold_doc_ids,
            k=config.TOP_K_FINAL,
        ),
        "multi_doc_hit": multi_doc_hit(retrieved_doc_ids, example.gold_doc_ids),
        "mrr": mrr(retrieved_doc_ids, example.gold_doc_ids),
        "misinformation_suppressed": misinformation_suppressed(
            answer,
            example.wrong_answers,
        ),
        "ambiguity_coverage": ambiguity_coverage(answer, example.gold_answers),
        "faithfulness_score": faithfulness_score(
            answer,
            [c.text for c in result.chunks],
        ),
    }

    failure_mode, conflict_type = evaluator.classify(
        metrics=metrics,
        failure_type=profile.failure_types,
    )

    elapsed = time.time() - t0

    trace = {
        # Identity
        "qid": example.qid,
        "source": example.source,
        "question": query,
        "gold_answers": example.gold_answers,
        "wrong_answers": example.wrong_answers,
        "predicted_answer": answer,

        # Novelty 1: failure detection
        "failure_types": str(profile.failure_types),
        "detector_conf": getattr(profile, "confidence", None),
        "failure_rationale": getattr(profile, "rationale", None),

        # Novelty 1: adaptive retrieval routing
        "retrieval_strategy": getattr(result, "strategy_used", None),
        "n_chunks": len(result.chunks),
        "chunk_ids": [c.chunk_id for c in result.chunks],
        "doc_ids": [c.doc_id for c in result.chunks],
        "chunk_labels": [c.label for c in result.chunks],

        # Novelty 2: trust-weighted scoring evidence
        "trust_weights": [
            round(getattr(w, "weight", 0.0), 4) for w in result.weights
        ],
        "retrieval_scores": [
            round(getattr(w, "retrieval_score", 0.0), 4) for w in result.weights
        ],
        "reranker_scores": [
            round(getattr(w, "reranker_score", 0.0), 4) for w in result.weights
        ],
        "agreement_scores": [
            round(getattr(w, "agreement_score", 0.0), 4) for w in result.weights
        ],

        # Agent debate
        "n_agents": len(turns),
        "agent_answers": [getattr(t, "answer", "") for t in turns],
        "agent_confidences": [
            round(getattr(t, "confidence", 0.0), 4) for t in turns
        ],

        # Novelty 3: conflict-aware aggregation
        "agg_answer": getattr(agg, "answer", ""),
        "agg_conflict": getattr(agg, "conflict_type", None),
        "aggregation_mode": getattr(agg, "conflict_type", None),
        "suppressed_agents": getattr(agg, "suppressed_agents", []),
        "used_multiple_answers": len(example.gold_answers) > 1,

        # Evaluation classification
        "conflict_type": conflict_type,
        "failure_mode": failure_mode,

        # Metrics
        **metrics,

        # Timing
        "elapsed_s": round(elapsed, 3),
    }

    return trace


METRIC_KEYS = [
    "answer_em",
    "answer_f1",
    "supporting_fact_f1",
    "joint_em",
    "joint_f1",
    "recall_at_k",
    "multi_doc_hit",
    "mrr",
    "misinformation_suppressed",
    "ambiguity_coverage",
    "faithfulness_score",
]

PER_QUERY_FIELDS = [
    "qid",
    "source",
    "question",
    "gold_answers",
    "predicted_answer",

    # Novelty 1
    "failure_types",
    "detector_conf",
    "failure_rationale",
    "retrieval_strategy",

    # Novelty 2
    "trust_weights",
    "retrieval_scores",
    "reranker_scores",
    "agreement_scores",

    # Novelty 3
    "aggregation_mode",
    "suppressed_agents",
    "used_multiple_answers",

    # Other trace fields
    "conflict_type",
    "failure_mode",
    "n_chunks",
    "chunk_labels",
    "n_agents",
] + METRIC_KEYS + ["elapsed_s"]


def write_per_query(traces: list[dict]):
    with open(config.PER_QUERY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=PER_QUERY_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(traces)

    log(f"Saved per_query → {config.PER_QUERY_PATH}")


def write_traces(traces: list[dict]):
    with open(config.TRACES_PATH, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, default=str)

    log(f"Saved traces    → {config.TRACES_PATH}")


def write_summary(traces: list[dict]):
    def mean_metric(subset: list[dict], key: str) -> float:
        vals = [t.get(key) for t in subset if t.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    hotpot = [t for t in traces if t.get("source") == "hotpotqa"]
    ramdocs = [t for t in traces if t.get("source") == "ramdocs"]

    rows = []

    for metric in METRIC_KEYS:
        rows.append({
            "metric": metric,
            "overall": mean_metric(traces, metric),
            "hotpotqa": mean_metric(hotpot, metric),
            "ramdocs": mean_metric(ramdocs, metric),
        })

    # Extra novelty summary rows
    rows.extend([
        {
            "metric": "bridge_strategy_used",
            "overall": round(
                sum(1 for t in traces if str(t.get("failure_types", "")).find("BRIDGE") >= 0
                    and str(t.get("retrieval_strategy", "")).lower().find("multi") >= 0)
                / max(sum(1 for t in traces if "BRIDGE" in str(t.get("failure_types", ""))), 1),
                4,
            ),
            "hotpotqa": 0.0,
            "ramdocs": 0.0,
        },
        {
            "metric": "avg_trust_weight",
            "overall": _avg_list_field(traces, "trust_weights"),
            "hotpotqa": _avg_list_field(hotpot, "trust_weights"),
            "ramdocs": _avg_list_field(ramdocs, "trust_weights"),
        },
    ])

    with open(config.SUMMARY_METRICS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "overall", "hotpotqa", "ramdocs"],
        )
        writer.writeheader()
        writer.writerows(rows)

    log(f"Saved summary   → {config.SUMMARY_METRICS_PATH}")

    print(f"\n  {'Metric':<30} {'Overall':>8} {'HotpotQA':>10} {'RAMDocs':>9}")
    print("  " + "─" * 64)

    for row in rows:
        print(
            f"  {row['metric']:<30} "
            f"{row['overall']:>8.4f} "
            f"{row['hotpotqa']:>10.4f} "
            f"{row['ramdocs']:>9.4f}"
        )

    print()


def _avg_list_field(traces: list[dict], key: str) -> float:
    vals = []
    for t in traces:
        field = t.get(key, [])
        if isinstance(field, list):
            vals.extend([v for v in field if isinstance(v, (int, float))])
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def print_novelty_summary(traces: list[dict]):
    bridge = [t for t in traces if "BRIDGE" in str(t.get("failure_types", ""))]
    misinfo = [t for t in traces if "MISINFORMATION" in str(t.get("failure_types", ""))]
    noise = [t for t in traces if "NOISE" in str(t.get("failure_types", ""))]
    ambiguous = [t for t in traces if "AMBIGUOUS" in str(t.get("failure_types", ""))]

    print("\n" + "═" * 65)
    print("  Novelty Evidence Summary")
    print("═" * 65)

    print(f"  Failure-detected examples:")
    print(f"    BRIDGE         : {len(bridge)}")
    print(f"    MISINFORMATION : {len(misinfo)}")
    print(f"    NOISE          : {len(noise)}")
    print(f"    AMBIGUOUS      : {len(ambiguous)}")

    strategy_counts = {}
    for t in traces:
        strategy = str(t.get("retrieval_strategy", "unknown"))
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    print("\n  Retrieval strategies used:")
    for strategy, count in strategy_counts.items():
        print(f"    {strategy:<20}: {count}")

    print("\n  Aggregation modes used:")
    agg_counts = {}
    for t in traces:
        mode = str(t.get("aggregation_mode", "unknown"))
        agg_counts[mode] = agg_counts.get(mode, 0) + 1

    for mode, count in agg_counts.items():
        print(f"    {mode:<20}: {count}")

    print("\n  This proves:")
    print("    1. Failure detection was run before retrieval.")
    print("    2. Retrieval strategy was logged per query.")
    print("    3. Trust-score components were stored.")
    print("    4. Aggregation mode was stored per query.")
    print("═" * 65 + "\n")


def main():
    t_total = time.time()

    print("\n" + "═" * 65)
    print("  Adaptive RAG Pipeline — Main")
    print("═" * 65)

    _ensure_dirs()
    _check_index()

    log(f"Loading HotpotQA (n={config.N_SAMPLES}) from {config.HOTPOT_PATH}")
    hotpot_examples = load_hotpotqa(config.HOTPOT_PATH, n=config.N_SAMPLES)

    log(f"Loading RAMDocs  (n={config.N_SAMPLES}) from {config.RAMDOCS_PATH}")
    ramdocs_examples = load_ramdocs(config.RAMDOCS_PATH, n=config.N_SAMPLES)

    examples = hotpot_examples + ramdocs_examples

    log(
        f"Total examples   : {len(examples)} "
        f"({len(hotpot_examples)} hotpotqa + {len(ramdocs_examples)} ramdocs)"
    )

    print()
    adaptive, detector, agent, aggregator, generator, evaluator = build_components()

    print(f"\n{'═' * 65}")
    print(f"  Running pipeline over {len(examples)} examples...")
    print(f"{'═' * 65}\n")

    traces: list[dict] = []

    for idx, example in enumerate(examples, start=1):
        try:
            trace = process_example(
                example,
                adaptive,
                detector,
                agent,
                aggregator,
                generator,
                evaluator,
            )
            traces.append(trace)

        except Exception as e:
            traces.append({
                "qid": example.qid,
                "source": example.source,
                "question": example.question,
                "error": str(e),
                **{k: None for k in METRIC_KEYS},
            })
            print(f"  ⚠ [{example.qid}] error: {e}")

        if idx % 10 == 0 or idx == len(examples):
            last = traces[-1]
            em = last.get("answer_em", 0) or 0
            f1 = last.get("answer_f1", 0) or 0
            strategy = last.get("retrieval_strategy", "?")
            failure = last.get("failure_types", "?")

            print(
                f"  [{idx:>4}/{len(examples)}] "
                f"qid={example.qid[:20]:<20} "
                f"em={em:.2f} f1={f1:.2f} "
                f"failure={failure} "
                f"strategy={strategy}"
            )

    print(f"\n{'═' * 65}")
    print("  Writing results...")
    print(f"{'═' * 65}")

    write_per_query(traces)
    write_traces(traces)
    write_summary(traces)
    print_novelty_summary(traces)

    total_time = time.time() - t_total

    print(f"\n{'═' * 65}")
    print(
        f"  ✅ Done. {len(traces)} examples in {total_time:.1f}s "
        f"({total_time / max(len(traces), 1):.2f}s/query)"
    )
    print(f"{'═' * 65}\n")

    time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())