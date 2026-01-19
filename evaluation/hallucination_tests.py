from __future__ import annotations

import json
from pathlib import Path

from pipelines import RAGPipeline

DATA_PATH = Path("evaluation/adversarial_queries.json")


def run_suite() -> None:
    pipeline = RAGPipeline()
    queries = json.loads(DATA_PATH.read_text())
    for case in queries:
        question = case["question"]
        print(f"Running: {case['id']}")
        result = pipeline.query(question)
        print(result["verification"])
        print("-" * 20)


if __name__ == "__main__":
    run_suite()
