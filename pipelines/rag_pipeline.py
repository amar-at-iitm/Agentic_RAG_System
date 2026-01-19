from __future__ import annotations

from typing import Any, Dict

import re
from pipelines.multi_agent_orchestrator import MultiAgentOrchestrator


class RAGPipeline:
    def __init__(self) -> None:
        self.orchestrator = MultiAgentOrchestrator()

    @staticmethod
    def scrub_pii(text: str) -> str:
        # Simple regex for email and phone numbers
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
        
        text = re.sub(email_pattern, "[EMAIL_REDACTED]", text)
        text = re.sub(phone_pattern, "[PHONE_REDACTED]", text)
        return text

    def query(self, question: str, top_k: int | None = None) -> Dict[str, Any]:
        clean_question = self.scrub_pii(question)
        result = self.orchestrator.run(clean_question, top_k=top_k)
        return {
            "plan": result.plan,
            "retrieved": [
                {
                    **doc.metadata,
                    "preview": doc.page_content[:400],
                }
                for doc in result.retrieved
            ],
            "draft": result.draft,
            "verification": result.verification,
            "answer": result.final_answer,
        }
