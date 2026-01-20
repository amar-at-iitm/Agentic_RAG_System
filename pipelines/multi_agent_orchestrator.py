from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Set

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
    draft: str | None
    verification: Dict[str, str] | None
    final_answer: str
    latency_seconds: float = 0.0
    token_usage: Dict[str, int] | None = None


class MultiAgentOrchestrator:
    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.reasoning = ReasoningAgent()
        self.verifier = VerifierAgent()
        self.answer = AnswerAgent()

    @staticmethod
    def _agents_in_plan(plan: List[Dict[str, Any]]) -> Set[str]:
        return {step.get("agent") for step in plan if "agent" in step}

    def run(self, question: str, top_k: int | None = None) -> OrchestratorResult:
        start_time = time.time()

        # ---------- PLANNING ----------
        plan = self.planner.plan(question)
        agents = self._agents_in_plan(plan)

        retrieved: List[Document] = []
        draft: str | None = None
        verification: Dict[str, str] | None = None

        # ---------- RETRIEVAL ----------
        if "retriever" in agents:
            retrieved = self.retriever.retrieve(
                question,
                plan,
                top_k=top_k,
            )

        # ---------- REASONING ----------
        if "reasoning" in agents:
            draft = self.reasoning.reason(
                question,
                retrieved,
            )

        # ---------- VERIFICATION ----------
        if "verifier" in agents and draft is not None:
            verification = self.verifier.verify(
                draft,
                retrieved,
            )
        else:
            verification = {"verdict": "pass", "notes": ""}

        # ---------- ANSWER ----------
        if draft is None:
            # Direct answer (small talk or simple queries)
            final_answer = self.answer.finalize(
                draft=question,
                notes="",
            )
        else:
            final_answer = self.answer.finalize(
                draft=draft,
                notes=verification.get("notes", ""),
            )

        end_time = time.time()

        # ---------- TOKEN USAGE (approximate) ----------
        token_usage = {
            "prompt_tokens": len(question.split())
            + sum(len(d.page_content.split()) for d in retrieved),
            "completion_tokens": len(final_answer.split()),
        }

        return OrchestratorResult(
            plan=plan,
            retrieved=retrieved,
            draft=draft,
            verification=verification,
            final_answer=final_answer,
            latency_seconds=end_time - start_time,
            token_usage=token_usage,
        )
