"""Explicit model networking, shared by main agents, workers and compaction."""
import os

import httpx

from codeclaw.core.config import load_config


def model_network_mode():
    return os.environ.get("CODECLAW_NETWORK_MODE") or load_config().get("network_mode", "direct")


def model_http_client(client_type=None):
    # httpx also reads the Windows registry when trust_env=True. A system
    # proxy can silently break otherwise reachable local/custom model APIs.
    if client_type is not None:
        # SDK factories supply their own timeout/limits and the HTTP library
        # version they require (recent Anthropic SDKs use httpx2).
        return client_type(trust_env=model_network_mode() == "system", follow_redirects=True)
    return httpx.AsyncClient(
        trust_env=model_network_mode() == "system",
        timeout=httpx.Timeout(600, connect=10),
        follow_redirects=True,
    )
