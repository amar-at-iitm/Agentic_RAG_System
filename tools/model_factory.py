from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseLanguageModel

from tools.config_loader import load_yaml_file


def _build_ollama(cfg: Dict[str, Any]) -> BaseLanguageModel:
    return ChatOllama(
        model=cfg.get("model", "mistral"),
        temperature=float(cfg.get("temperature", 0.1)),
        num_predict=int(cfg.get("max_output_tokens", 512)),
        keep_alive="5m",
    )


def build_llm(agent_name: str, model_config_path: Path | str) -> BaseLanguageModel:
    config = load_yaml_file(model_config_path)

    global_cfg = config.get("global", {})
    agent_cfg = config.get("agents", {}).get(agent_name, {})
    merged: Dict[str, Any] = {**global_cfg, **agent_cfg}

    provider = merged.get("provider", "anthropic")
    model = merged.get("model", "claude-sonnet-4.5")
    temperature = float(merged.get("temperature", 0.2))
    max_tokens = int(merged.get("max_output_tokens", 1024))

    enforced = bool(
        merged.get("enable_claude_sonnet_4_5", False)
        or merged.get("client_policy", {}).get("enforce", False)
    )

    fallback_cfg: Dict[str, Any] = merged.get("fallback", {})

    # ---------- PRIMARY: ANTHROPIC ----------
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if api_key:
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                anthropic_api_key=api_key,
            )

        # Claude unavailable → fallback
        if fallback_cfg:
            return _build_ollama(fallback_cfg)

        # Enforced with no fallback → hard fail
        if enforced:
            raise EnvironmentError(
                "Claude Sonnet 4.5 is enforced but ANTHROPIC_API_KEY is missing "
                "and no fallback is configured."
            )

    # ---------- OLLAMA ----------
    if provider == "ollama":
        return _build_ollama(merged)

    raise ValueError(
        f"Unsupported provider '{provider}' and no valid fallback found."
    )
