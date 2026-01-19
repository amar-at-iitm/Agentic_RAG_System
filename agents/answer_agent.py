from __future__ import annotations

from pathlib import Path

from agents.base_agent import BaseAgent

PROMPT_PATH = Path("config/agent_prompts.yaml")
MODEL_PATH = Path("config/model_config.yaml")


class AnswerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("answer", PROMPT_PATH, MODEL_PATH)

    def finalize(self, draft: str, notes: str) -> str:
        return self.invoke(draft=draft, notes=notes)
