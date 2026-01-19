from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools.config_loader import load_yaml_file
from tools.model_factory import build_llm


class BaseAgent:
    """Shared functionality for all agents."""

    def __init__(
        self,
        agent_name: str,
        prompt_config_path: Path,
        model_config_path: Path,
    ) -> None:
        prompts = load_yaml_file(prompt_config_path)
        if agent_name not in prompts:
            raise KeyError(f"Prompt missing for agent '{agent_name}'")
        agent_prompt = prompts[agent_name]
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", agent_prompt["system"]),
                ("human", agent_prompt["human"]),
            ]
        )
        self.agent_name = agent_name
        self.model_config_path = model_config_path
        self.llm: BaseLanguageModel = build_llm(agent_name, model_config_path)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, **kwargs: Any) -> str:
        result = self.chain.invoke(kwargs)
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            return json.dumps(result)
        return str(result)

    def with_llm_override(self, llm: Optional[BaseLanguageModel]) -> "BaseAgent":
        if llm is None:
            return self
        self.llm = llm
        self.chain = self.prompt | self.llm | StrOutputParser()
        return self

    @property
    def name(self) -> str:
        return self.agent_name
