"""
data_loader.py
--------------
Loads HotpotQA and RAMDocs datasets into a unified schema.

Downstream consumers:
  - failure_detector.py  : uses .type, .wrong_answers, .gold_answers, doc.label
  - trust_scorer.py      : uses .gold_doc_ids
  - aggregator.py        : uses .wrong_answers, .gold_answers (ambiguity mode)
  - evaluator.py         : uses .gold_answers for metric computation
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """
    A single context document.

    HotpotQA : label is always None (no misinfo/noise concept in that dataset).
    RAMDocs  : label ∈ {'correct', 'misinfo', 'noise'}.
               'correct' maps to 'gold' internally so downstream code can
               use a single vocabulary: gold | misinfo | noise | None.
    """
    doc_id: str
    text: str
    label: Optional[str] = None          # 'gold' | 'misinfo' | 'noise' | None


@dataclass
class QueryExample:
    """
    Unified query container for both datasets.

    Fields used by the three novel components
    ------------------------------------------
    failure_detector:
        .type          — 'bridge' | 'comparison' | None
        .wrong_answers — non-empty list → MISINFORMATION flag
        .gold_answers  — len > 1       → AMBIGUOUS flag
        .documents     — any doc.label == 'noise' → NOISE flag

    trust_scorer:
        .gold_doc_ids  — used to compute cross-document agreement signal

    aggregator:
        .wrong_answers — answers to suppress in weighted vote
        .gold_answers  — all valid answers for ambiguity mode
    """
    qid: str
    question: str
    gold_answers: list[str]
    documents: list[Document]

    # HotpotQA-specific (None for RAMDocs)
    type: Optional[str] = None               # 'bridge' | 'comparison'
    supporting_facts: list[dict] = field(default_factory=list)

    # RAMDocs-specific (empty for HotpotQA)
    wrong_answers: list[str] = field(default_factory=list)
    gold_doc_ids: list[str] = field(default_factory=list)
    disambig_entities: list[str] = field(default_factory=list)

    # Source tag — lets pipeline components branch on dataset if needed
    source: str = "hotpotqa"                 # 'hotpotqa' | 'ramdocs'


# ---------------------------------------------------------------------------
# HotpotQA loader
# ---------------------------------------------------------------------------

def _hotpot_build_docs(context: dict) -> list[Document]:
    """
    HotpotQA context format:
        {
          "title":     ["Doc A", "Doc B", ...],
          "sentences": [["sent1", "sent2"], ["sent1", ...], ...]
        }
    We join sentences per document with a single space.
    """
    docs = []
    for title, sents in zip(context["title"], context["sentences"]):
        text = " ".join(s.strip() for s in sents)
        docs.append(Document(doc_id=title, text=text, label=None))
    return docs


def _hotpot_build_supporting(supporting_facts: dict) -> list[dict]:
    """
    HotpotQA supporting_facts format:
        {"title": [...], "sent_id": [...]}
    Convert to list of dicts for uniform access.
    """
    return [
        {"title": t, "sent_id": sid}
        for t, sid in zip(supporting_facts["title"], supporting_facts["sent_id"])
    ]


def load_hotpotqa(path: str | Path, n: int = 500) -> list[QueryExample]:
    """
    Load up to `n` examples from a HotpotQA JSON file.

    The file is a flat JSON array of question objects.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    examples = []
    for item in raw[:n]:
        docs = _hotpot_build_docs(item["context"])
        supporting = _hotpot_build_supporting(item["supporting_facts"])

        examples.append(QueryExample(
            qid=item.get("_id", item.get("id")),
            question=item["question"],
            gold_answers=[item["answer"]],       # HotpotQA has exactly one answer
            documents=docs,
            type=item.get("type"),               # 'bridge' | 'comparison'
            supporting_facts=supporting,
            wrong_answers=[],                    # not present in HotpotQA
            gold_doc_ids=[],                     # not applicable
            source="hotpotqa",
        ))

    return examples


# ---------------------------------------------------------------------------
# RAMDocs loader
# ---------------------------------------------------------------------------

# RAMDocs uses 'correct' for gold documents; normalise to 'gold' so downstream
# code uses a single vocabulary.
_RAMDOCS_LABEL_MAP = {
    "correct": "gold",
    "misinfo": "misinfo",
    "noise":   "noise",
}


def _ramdocs_build_docs(raw_docs: list[dict], qid: str) -> tuple[list[Document], list[str]]:
    """
    RAMDocs document format:
        {
          "text":   "...",
          "type":   "correct" | "misinfo" | "noise",
          "answer": "..."        # per-doc answer string (ignored at load time)
        }

    Returns (documents, gold_doc_ids).
    doc_id is synthesised as  <qid>_doc_<index>  because RAMDocs documents
    have no inherent identifier field.
    """
    docs = []
    gold_doc_ids = []

    for idx, d in enumerate(raw_docs):
        doc_id = f"{qid}_doc_{idx}"
        label = _RAMDOCS_LABEL_MAP.get(d.get("type", ""), None)
        docs.append(Document(doc_id=doc_id, text=d["text"], label=label))
        if label == "gold":
            gold_doc_ids.append(doc_id)

    return docs, gold_doc_ids


def load_ramdocs(path: str | Path, n: int = 500) -> list[QueryExample]:
    """
    Load up to `n` examples from a RAMDocs JSON file.

    The file is a flat JSON array of question objects.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    examples = []
    for item in raw[:n]:
        qid = item.get("id", f"ramdocs_{len(examples)}")
        docs, gold_doc_ids = _ramdocs_build_docs(item["documents"], qid)

        examples.append(QueryExample(
            qid=qid,
            question=item["question"],
            gold_answers=item.get("gold_answers", []),
            documents=docs,
            type=None,                           # RAMDocs has no bridge/comparison split
            supporting_facts=[],
            wrong_answers=item.get("wrong_answers", []),
            gold_doc_ids=gold_doc_ids,
            disambig_entities=item.get("disambig_entity", []),
            source="ramdocs",
        ))

    return examples


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

def load_all(
    hotpotqa_path: str | Path,
    ramdocs_path: str | Path,
    n_hotpot: int = 500,
    n_ramdocs: int = 500,
    shuffle: bool = False,
    seed: int = 42,
) -> list[QueryExample]:
    """
    Load both datasets and return a combined list.

    Args:
        hotpotqa_path : path to hotpot_dev.json (or hotpot_train.json)
        ramdocs_path  : path to ramdocs_test.json
        n_hotpot      : max HotpotQA samples
        n_ramdocs     : max RAMDocs samples
        shuffle       : if True, shuffle the combined list
        seed          : random seed for reproducibility

    Returns:
        Combined list of QueryExample objects.
        Each example's .source field indicates its origin dataset.
    """
    hotpot = load_hotpotqa(hotpotqa_path, n=n_hotpot)
    ramdocs = load_ramdocs(ramdocs_path, n=n_ramdocs)
    combined = hotpot + ramdocs

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(combined)

    return combined


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------

def _validate(examples: list[QueryExample], label: str) -> None:
    bridge = sum(1 for e in examples if e.type == "bridge")
    comparison = sum(1 for e in examples if e.type == "comparison")
    has_wrong = sum(1 for e in examples if e.wrong_answers)
    ambiguous = sum(1 for e in examples if len(e.gold_answers) > 1)
    has_noise = sum(1 for e in examples if any(d.label == "noise" for d in e.documents))
    has_misinfo = sum(1 for e in examples if any(d.label == "misinfo" for d in e.documents))
    has_gold_ids = sum(1 for e in examples if e.gold_doc_ids)

    print(f"\n{'='*55}")
    print(f"  {label}  (n={len(examples)})")
    print(f"{'='*55}")
    print(f"  type=bridge         : {bridge}")
    print(f"  type=comparison     : {comparison}")
    print(f"  wrong_answers set   : {has_wrong}  (→ MISINFORMATION flag)")
    print(f"  ambiguous answers   : {ambiguous}  (→ AMBIGUOUS flag)")
    print(f"  has noise docs      : {has_noise}  (→ NOISE flag)")
    print(f"  has misinfo docs    : {has_misinfo}")
    print(f"  gold_doc_ids set    : {has_gold_ids}  (→ trust_scorer agreement signal)")

    # Sample check
    ex = examples[0]
    print(f"\n  Sample qid          : {ex.qid}")
    print(f"  Sample question     : {ex.question[:70]}...")
    print(f"  Sample gold_answers : {ex.gold_answers}")
    print(f"  Sample doc count    : {len(ex.documents)}")
    if ex.documents:
        d = ex.documents[0]
        print(f"  Sample doc label    : {d.label}")
        print(f"  Sample doc text[:60]: {d.text[:60]}...")


if __name__ == "__main__":
    import sys

    # Adjust paths to match your local data/ folder layout
    hotpot_path = "data/hotpot_dev.json"
    ramdocs_path = "data/ramdocs_test.json"

    if not Path(hotpot_path).exists():
        print(f"[WARN] {hotpot_path} not found — skipping HotpotQA validation")
    else:
        hotpot_examples = load_hotpotqa(hotpot_path, n=500)
        _validate(hotpot_examples, "HotpotQA")

    if not Path(ramdocs_path).exists():
        print(f"[WARN] {ramdocs_path} not found — skipping RAMDocs validation")
    else:
        ramdocs_examples = load_ramdocs(ramdocs_path, n=500)
        _validate(ramdocs_examples, "RAMDocs")

    if Path(hotpot_path).exists() and Path(ramdocs_path).exists():
        combined = load_all(hotpot_path, ramdocs_path, shuffle=True)
        print(f"\n  Combined total: {len(combined)} examples")
        print(f"  HotpotQA : {sum(1 for e in combined if e.source == 'hotpotqa')}")
        print(f"  RAMDocs  : {sum(1 for e in combined if e.source == 'ramdocs')}")

    sys.exit(0)