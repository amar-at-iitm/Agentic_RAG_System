from pathlib import Path

from tools.config_loader import load_yaml_file


def test_claude_sonnet_policy_enabled() -> None:
    config = load_yaml_file(Path("config/model_config.yaml"))
    assert config["global"].get("enable_claude_sonnet_4_5") is True
    assert config["global"].get("model") == "claude-sonnet-4.5"
