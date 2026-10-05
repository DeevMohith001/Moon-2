"""
Configuration loader utility for YAML configuration files.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_config(name: str = "config") -> Dict[str, Any]:
    """
    Load a YAML configuration file from configs/ directory.
    Supports names with or without '.yaml' extension.
    """
    if name.endswith(".yaml") or name.endswith(".yml"):
        filename = name
    else:
        filename = f"{name}.yaml"

    path = CONFIGS_DIR / filename
    if not path.exists():
        # Fall back to checking root or prompts directory
        alt_path = PROJECT_ROOT / filename
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_prompt_config(prompt_name: str = "sentiment_prompt") -> Dict[str, Any]:
    """
    Load a prompt YAML configuration from configs/prompts/.
    """
    if prompt_name.endswith(".yaml") or prompt_name.endswith(".yml"):
        filename = prompt_name
    else:
        filename = f"{prompt_name}.yaml"

    path = CONFIGS_DIR / "prompts" / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
