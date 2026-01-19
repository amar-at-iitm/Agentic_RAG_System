from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agents.base_agent import BaseAgent

PROMPT_PATH = Path("config/agent_prompts.yaml")
MODEL_PATH = Path("config/model_config.yaml")


class PlannerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("planner", PROMPT_PATH, MODEL_PATH)

    def plan(self, question: str) -> List[Dict[str, Any]]:
        raw = self.invoke(question=question)
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return [{"step": 1, "action": raw}]
