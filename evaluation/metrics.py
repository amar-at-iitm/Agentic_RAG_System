from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from rouge_score import rouge_scorer


@dataclass
class EvaluationResult:
    factual_consistency: float
    answer_faithfulness: float


def aggregate_factual_scores(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return float(np.mean(scores))


    return float(sum(verdicts) / len(verdicts))


def calculate_rouge(reference: str, candidate: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }
