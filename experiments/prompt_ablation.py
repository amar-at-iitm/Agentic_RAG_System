from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

PROMPT_PATH = Path("config/agent_prompts.yaml")


def tweak_prompt(agent: str, new_system_prompt: str) -> None:
    prompts: Dict[str, Dict[str, str]] = yaml.safe_load(PROMPT_PATH.read_text())
    prompts.setdefault(agent, {})["system"] = new_system_prompt
    PROMPT_PATH.write_text(yaml.safe_dump(prompts, sort_keys=False))


if __name__ == "__main__":
    tweak_prompt("answer", "You are a cheerful assistant.")
