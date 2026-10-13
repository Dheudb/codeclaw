"""Shared project MCP settings with redacted environment/header snapshots."""
import copy
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


def validate_server(config):
    if not isinstance(config, dict):
        raise ValueError("MCP configuration must be an object")
    transport = config.get("type") or ("http" if config.get("url") else "stdio")
    if transport == "streamable-http":
        transport = "http"
    if transport not in {"stdio", "http"}:
        raise ValueError("Supported MCP transports: stdio, http (Streamable HTTP)")
    result = {"type": transport, "enabled": config.get("enabled", True) is not False}
    for key in ("label", "catalogId"):
        if config.get(key):
            result[key] = str(config[key])[:200]
    if transport == "stdio":
        command = config.get("command", "")
        args = config.get("args", [])
        if not isinstance(command, str) or not command.strip() or "\x00" in command:
            raise ValueError("Provide an executable in command")
        if not isinstance(args, list) or any(not isinstance(v, str) or "\x00" in v for v in args):
            raise ValueError("args must be a JSON array of strings")
        result.update(command=command.strip(), args=args)
        mapping = "env"
    else:
        url = config.get("url", "")
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError("Provide an HTTP(S) URL without embedded credentials or a fragment")
        result["url"] = url
        mapping = "headers"
    values = config.get(mapping, {})
    if not isinstance(values, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in values.items()):
        raise ValueError(mapping + " must be a JSON object of strings")
    for key, value in values.items():
        if mapping == "env" and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("Invalid environment variable name")
        if mapping == "headers" and (not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key) or "\r" in value or "\n" in value):
            raise ValueError("Invalid HTTP header")
        if "\x00" in value:
            raise ValueError("MCP values cannot contain NUL")
    result[mapping] = dict(values)
    return result


def validate_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,47}", name):
        raise ValueError("MCP name must start with a letter; use up to 48 letters, digits, - or _")


class MCPConfigStore:
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        if not self.path.exists():
            return {"mcpServers": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("mcpServers", {}), dict):
                raise ValueError()
            data.setdefault("mcpServers", {})
            return data
        except (ValueError, OSError):
            raise ValueError("mcp_servers.json is invalid; repair the file before saving")

    def write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, target = tempfile.mkstemp(prefix=".mcp-", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(target, self.path)
        finally:
            if os.path.exists(target):
                os.unlink(target)

    def snapshot(self):
        servers = []
        for name, raw in self.read()["mcpServers"].items():
            if not isinstance(raw, dict):
                servers.append({"name": name, "error": "Configuration must be an object", "enabled": False})
                continue
            config = copy.deepcopy(raw)
            error = ""
            try:
                config = validate_server(raw)
            except ValueError as exc:
                error = str(exc)
                config = {"enabled": raw.get("enabled", True) is not False, "type": str(raw.get("type", ""))}
            # Both env and headers may contain credentials. Show names, never values.
            for key in ("env", "headers"):
                values = config.get(key, {})
                config[key] = {k: "" for k in values} if isinstance(values, dict) else {}
            # Existing configurations may put a credential in URL query parameters.
            url = config.get("url", "")
            if isinstance(url, str) and "?" in url:
                config["url"] = url.split("?", 1)[0]
                config["savedUrlQuery"] = True
            servers.append({**config, "name": name, "error": error})
        return servers

    def save(self, name, config, *, existing=False):
        validate_name(name)
        data = self.read()
        previous = data["mcpServers"].get(name)
        if previous is not None and not existing:
            raise ValueError("A server with this name already exists")
        if existing and previous is None:
            raise ValueError("MCP server no longer exists")
        config = copy.deepcopy(config)
        if isinstance(previous, dict) and existing:
            for key in ("env", "headers"):
                for field, value in config.get(key, {}).items():
                    if value == "" and field in previous.get(key, {}):
                        config[key][field] = previous[key][field]
            if config.pop("savedUrlQuery", False) and config.get("url") == str(previous.get("url", "")).split("?", 1)[0]:
                config["url"] = previous["url"]
        data["mcpServers"][name] = validate_server(config)
        self.write(data)

    def toggle(self, name, enabled):
        data = self.read()
        if name not in data["mcpServers"]:
            raise ValueError("Unknown MCP server")
        data["mcpServers"][name]["enabled"] = bool(enabled)
        self.write(data)

    def remove(self, name):
        data = self.read()
        if name not in data["mcpServers"]:
            raise ValueError("Unknown MCP server")
        del data["mcpServers"][name]
        self.write(data)
