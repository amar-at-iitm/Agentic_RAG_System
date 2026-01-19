from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from agents.base_agent import BaseAgent

PROMPT_PATH = Path("config/agent_prompts.yaml")
MODEL_PATH = Path("config/model_config.yaml")


class ReasoningAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("reasoning", PROMPT_PATH, MODEL_PATH)

    def reason(self, question: str, docs: List[Document]) -> str:
        formatted_chunks = self._format_docs(docs)
        return self.invoke(question=question, chunks=formatted_chunks)

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        return "\n\n".join(
            f"{doc.metadata.get('chunk_id', 'chunk')}\n{doc.page_content}" for doc in docs
        )
