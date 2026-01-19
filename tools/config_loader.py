from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_file(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data
