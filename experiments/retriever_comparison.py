from __future__ import annotations

from pipelines import RAGPipeline


def compare(questions: list[str]) -> None:
    pipeline = RAGPipeline()
    for q in questions:
        result = pipeline.query(q)
        print(q)
        print(result["retrieved"])
        print()


if __name__ == "__main__":
    sample_questions = [
        "What is the methodology discussed in sample_report_1?",
        "List the datasets mentioned in sample_paper_1.",
    ]
    compare(sample_questions)
