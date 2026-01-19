from __future__ import annotations

import os
from typing import Any, Dict

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseLanguageModel

from tools.config_loader import load_yaml_file


def build_llm(agent_name: str, model_config_path: Path | str) -> BaseLanguageModel:
    config = load_yaml_file(model_config_path)
    global_settings: Dict[str, Any] = config.get("global", {})
    agent_overrides: Dict[str, Any] = config.get("agents", {}).get(agent_name, {})

    merged: Dict[str, Any] = {**global_settings, **agent_overrides}
    enforced = merged.get("enable_claude_sonnet_4_5", False) or (
        merged.get("client_policy", {}).get("enforce")
    )

    if enforced:
        merged["provider"] = "anthropic"
        merged["model"] = "claude-sonnet-4.5"

    provider = merged.get("provider", "anthropic")
    model = merged.get("model", "claude-sonnet-4.5")
    temperature = float(merged.get("temperature", 0.2))
    max_tokens = int(merged.get("max_output_tokens", 1024))

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY missing. Required to enable Claude Sonnet 4.5 for all clients."
            )
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=api_key,
        )

    if provider == "ollama":
        return ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            keep_alive="5m",
        )

    raise ValueError(f"Unsupported provider '{provider}' in model_config.yaml")
