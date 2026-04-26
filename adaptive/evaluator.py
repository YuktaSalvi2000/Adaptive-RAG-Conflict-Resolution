from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Evaluator:
    """
    Classifies outcome after metrics are computed.
    """

    def classify(self, metrics: dict, failure_type=None) -> tuple[str, str]:
        answer_em = metrics.get("answer_em", 0.0) or 0.0
        multi_hit = metrics.get("multi_doc_hit", 0.0) or 0.0
        misinfo_ok = metrics.get("misinformation_suppressed", 1.0)
        ambiguity = metrics.get("ambiguity_coverage", 0.0) or 0.0

        conflict_type = self._conflict_type(failure_type)

        if answer_em == 1.0:
            failure_mode = "correct"
        elif multi_hit == 1.0:
            failure_mode = "generation_failure"
        else:
            failure_mode = "retrieval_failure"

        if misinfo_ok == 0.0:
            failure_mode = "misinformation_failure"

        if conflict_type == "ambiguity" and ambiguity < 1.0:
            failure_mode = "ambiguity_undercoverage"

        return failure_mode, conflict_type

    def _conflict_type(self, failure_type) -> str:
        text = str(failure_type).upper()

        if "BRIDGE" in text:
            return "bridge"
        if "MISINFORMATION" in text:
            return "misinformation"
        if "NOISE" in text:
            return "noise"
        if "AMBIGUOUS" in text:
            return "ambiguity"

        return "none"