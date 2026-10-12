"""Configuration loader for GateID Vision."""
from pathlib import Path
import yaml

def load_config(config_path: str = "config.yaml") -> dict:
    """Load and validate configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
