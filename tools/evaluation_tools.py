from __future__ import annotations

from typing import List

import numpy as np


def factual_consistency(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return float(np.mean(scores))


def answer_faithfulness(verdicts: List[bool]) -> float:
    if not verdicts:
        return 0.0
    return float(sum(verdicts) / len(verdicts))
