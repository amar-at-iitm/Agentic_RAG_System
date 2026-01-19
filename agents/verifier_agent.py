from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

from agents.base_agent import BaseAgent

PROMPT_PATH = Path("config/agent_prompts.yaml")
MODEL_PATH = Path("config/model_config.yaml")


class VerifierAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("verifier", PROMPT_PATH, MODEL_PATH)

    def verify(self, draft: str, docs: List[Document]) -> Dict[str, str]:
        formatted_chunks = self._format_docs(docs)
        result = self.invoke(draft=draft, chunks=formatted_chunks)
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                return {
                    "verdict": parsed.get("verdict", "fail"),
                    "notes": parsed.get("notes", "No verifier notes supplied."),
                }
        except json.JSONDecodeError:
            pass
        return {"verdict": "fail", "notes": result}

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        return "\n\n".join(
            f"{doc.metadata.get('chunk_id', 'chunk')}\n{doc.page_content}" for doc in docs
        )
