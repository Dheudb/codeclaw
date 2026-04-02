"""
Persistent configuration for CodeClaw.

Stores model/provider settings in ~/.codeclaw/config.json so they survive
across terminal sessions. Values are loaded into os.environ on startup and
written back when /model changes them.
"""

import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".codeclaw"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Keys we persist — maps config key → environment variable name
_PERSISTED_KEYS = {
    "model_provider": "CODECLAW_MODEL_PROVIDER",
    "anthropic_model": "ANTHROPIC_MODEL",
    "openai_model": "OPENAI_MODEL",
    "anthropic_base_url": "ANTHROPIC_BASE_URL",
    "openai_base_url": "OPENAI_BASE_URL",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "max_output_tokens": "CODECLAW_MAX_OUTPUT_TOKENS",
}


def _ensure_dir():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Read config from disk. Returns empty dict on any failure."""
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_config(cfg: dict):
    """Write config dict to disk."""
    _ensure_dir()
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def load_config_into_env():
    """
    Called once at startup. For each persisted key, if the env var is NOT
    already set by the user's shell, inject the saved value.
    """
    cfg = load_config()
    for cfg_key, env_key in _PERSISTED_KEYS.items():
        val = cfg.get(cfg_key, "")
        if val and not os.environ.get(env_key):
            os.environ[env_key] = str(val)


def save_current_env_to_config():
    """
    Snapshot current env vars back into the config file.
    Only writes non-empty values.
    """
    cfg = load_config()
    for cfg_key, env_key in _PERSISTED_KEYS.items():
        val = os.environ.get(env_key, "")
        if val:
            cfg[cfg_key] = val
        else:
            cfg.pop(cfg_key, None)
    save_config(cfg)
