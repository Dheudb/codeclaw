import os
import unittest
from unittest.mock import patch

from codeclaw.core.network import model_http_client


class ModelNetworkTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_factories_use_their_compatible_http_library(self):
        from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient as AnthropicHttp
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient as OpenAIHttp
        for sdk, factory in [(AsyncAnthropic, AnthropicHttp), (AsyncOpenAI, OpenAIHttp)]:
            async with sdk(api_key="fixture", http_client=model_http_client(factory)) as client:
                self.assertFalse(client._client.trust_env)

    async def test_direct_mode_ignores_implicit_windows_proxy(self):
        with patch.dict(os.environ, {"CODECLAW_NETWORK_MODE": ""}), patch("codeclaw.core.network.load_config", return_value={}), patch("httpx._utils.getproxies", return_value={"https": "http://127.0.0.1:1"}) as proxies:
            async with model_http_client() as client:
                self.assertFalse(client.trust_env)
                proxies.assert_not_called()

    async def test_system_mode_explicitly_enables_proxy_discovery(self):
        with patch.dict(os.environ, {"CODECLAW_NETWORK_MODE": ""}), patch("codeclaw.core.network.load_config", return_value={"network_mode": "system"}), patch("httpx._utils.getproxies", return_value={"https": "http://127.0.0.1:1"}) as proxies:
            async with model_http_client() as client:
                self.assertTrue(client.trust_env)
                proxies.assert_called_once()
