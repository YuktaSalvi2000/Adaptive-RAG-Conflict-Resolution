import re
from collections import Counter


def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def answer_em(prediction: str, gold_answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(g) for g in gold_answers))


def answer_f1(prediction: str, gold_answers: list[str]) -> float:
    pred_tokens = _tokens(prediction)
    if not pred_tokens:
        return 0.0

    best = 0.0
    for gold in gold_answers:
        gold_tokens = _tokens(gold)
        if not gold_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            score = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            score = 2 * precision * recall / (precision + recall)

        best = max(best, score)

    return best


def supporting_fact_f1(prediction: str, supporting_facts: list[dict]) -> float:
    if not supporting_facts:
        return 0.0
    return 0.0


def joint_em(prediction: str, gold_answers: list[str], supporting_facts: list[dict]) -> float:
    return answer_em(prediction, gold_answers)


def joint_f1(prediction: str, gold_answers: list[str], supporting_facts: list[dict]) -> float:
    return answer_f1(prediction, gold_answers)


def recall_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str], k: int = 5) -> float:
    if not gold_doc_ids:
        return 0.0

    retrieved = set(retrieved_doc_ids[:k])
    gold = set(gold_doc_ids)

    return len(retrieved & gold) / len(gold)


def multi_doc_hit(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    if not gold_doc_ids:
        return 0.0

    retrieved = set(retrieved_doc_ids)
    gold = set(gold_doc_ids)

    return float(gold.issubset(retrieved))


def mrr(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    if not gold_doc_ids:
        return 0.0

    gold = set(gold_doc_ids)

    for i, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold:
            return 1.0 / i

    return 0.0


def misinformation_suppressed(prediction: str, wrong_answers: list[str]) -> float:
    if not wrong_answers:
        return 1.0

    pred = normalize_answer(prediction)

    for wrong in wrong_answers:
        wrong_norm = normalize_answer(wrong)
        if wrong_norm and wrong_norm in pred:
            return 0.0

    return 1.0


def ambiguity_coverage(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0

    pred = normalize_answer(prediction)
    covered = 0

    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if gold_norm and gold_norm in pred:
            covered += 1

    return covered / len(gold_answers)


def faithfulness_score(prediction: str, retrieved_texts: list[str]) -> float:
    if not prediction or not retrieved_texts:
        return 0.0

    pred_tokens = set(_tokens(prediction))
    context_tokens = set(_tokens(" ".join(retrieved_texts)))

    if not pred_tokens:
        return 0.0

    return len(pred_tokens & context_tokens) / len(pred_tokens)