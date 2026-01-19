from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List

from langchain_core.documents import Document

from agents import (
    AnswerAgent,
    PlannerAgent,
    ReasoningAgent,
    RetrieverAgent,
    VerifierAgent,
)


@dataclass
class OrchestratorResult:
    plan: List[Dict[str, Any]]
    retrieved: List[Document]
    draft: str
    verification: Dict[str, str]
    final_answer: str
    latency_seconds: float = 0.0
    token_usage: Dict[str, int] = None


class MultiAgentOrchestrator:
    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.reasoning = ReasoningAgent()
        self.verifier = VerifierAgent()
        self.answer = AnswerAgent()

    def run(self, question: str, top_k: int | None = None) -> OrchestratorResult:
        start_time = time.time()
        plan = self.planner.plan(question)
        retrieved = self.retriever.retrieve(question, plan, top_k=top_k)
        draft = self.reasoning.reason(question, retrieved)
        verification = self.verifier.verify(draft, retrieved)
        notes = verification.get("notes", "")
        final_answer = self.answer.finalize(draft=draft, notes=notes)
        end_time = time.time()
        
        # Placeholder for token usage - in real implementation this would come from agent callbacks
        token_usage = {
            "prompt_tokens": len(question.split()) + sum(len(d.page_content.split()) for d in retrieved),
            "completion_tokens": len(final_answer.split())
        }

        return OrchestratorResult(
            plan=plan,
            retrieved=retrieved,
            draft=draft,
            verification=verification,
            final_answer=final_answer,
            latency_seconds=end_time - start_time,
            token_usage=token_usage
        )
