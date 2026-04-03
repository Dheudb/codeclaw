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
    "max_turns": "CODECLAW_MAX_TURNS",
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


# ── Model Output Token Limits ──────────────────────────────────────

# Slot-reservation cap: most requests use < 5k output tokens, so the
# main loop caps to DEFAULT_CAPPED to avoid wasting slot capacity.
# On overflow the loop escalates to ESCALATED.
DEFAULT_CAPPED_MAX_TOKENS = 8_000
ESCALATED_MAX_TOKENS = 64_000
COMPACT_MAX_OUTPUT_TOKENS = 20_000


def get_model_max_output_tokens(model: str) -> tuple:
    """Return (default, upper_limit) for the given model.

    Env override: set CODECLAW_MODEL_MAX_OUTPUT_TOKENS=<number> to force
    a specific limit (useful for local models with small output windows).

    Only genuinely old models with hard low limits need special-casing.
    Everything else gets a generous 64k default; most APIs silently cap
    if the model can't actually produce that many tokens.
    """
    env_val = os.environ.get("CODECLAW_MODEL_MAX_OUTPUT_TOKENS", "")
    if env_val:
        try:
            v = int(env_val)
            if v > 0:
                return v, v
        except ValueError:
            pass
    m = model.lower()
    if any(k in m for k in ("claude-3-opus", "claude-3-haiku", "gpt-3.5")):
        return 4_096, 4_096
    if any(k in m for k in ("claude-3-sonnet", "3-5-sonnet", "3-5-haiku")):
        return 8_192, 8_192
    return 64_000, 128_000


async def safe_llm_call(client, model: str, provider: str, messages: list,
                        max_tokens: int, _retried: bool = False):
    """LLM call with automatic max_tokens fallback.

    If the API rejects the request because max_tokens is too large
    (common with local models), retries once at half the value.
    """
    try:
        if provider == "openai":
            resp = await client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
            )
            return str(
                getattr(getattr(resp.choices[0], "message", None), "content", "") or ""
            )
        else:
            resp = await client.messages.create(
                model=model, messages=messages, max_tokens=max_tokens,
            )
            for block in resp.content:
                if getattr(block, "type", "") == "text":
                    return getattr(block, "text", "")
            return ""
    except Exception as e:
        err = str(e).lower()
        if not _retried and ("max_tokens" in err or "max_output" in err
                             or "maximum" in err or "too large" in err):
            fallback = max(1024, max_tokens // 2)
            return await safe_llm_call(
                client, model, provider, messages, fallback, _retried=True
            )
        raise
