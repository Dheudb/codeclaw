"""Curated MCP catalog plus opt-in browsing of the official registry."""
import copy
import hashlib
import json
import re
from urllib.parse import urlsplit

import httpx

from xuloom.protocols.mcp_config import validate_server

REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0/servers"
BUILTIN = [
    {"id": "exa", "name": "Exa Search", "description": "网络搜索与网页正文读取。远程服务；可选 API Key。", "category": "web", "source": "精选", "homepage": "https://exa.ai/docs/get-started/exa-mcp", "config": {"type": "http", "url": "https://mcp.exa.ai/mcp", "headers": {"x-api-key": "${EXA_API_KEY}"}}, "inputs": [{"name": "EXA_API_KEY", "description": "Exa API Key（可留空，使用服务提供的免密额度）", "optional": True, "secret": True}], "requirements": "需要联网；服务额度和费用由 Exa 决定。"},
    {"id": "brave-search", "name": "Brave Search", "description": "网页、新闻及其他搜索工具。", "category": "web", "source": "精选", "homepage": "https://github.com/brave/brave-search-mcp-server", "config": {"type": "stdio", "command": "npx", "args": ["-y", "@brave/brave-search-mcp-server", "--transport", "stdio"], "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"}}, "inputs": [{"name": "BRAVE_API_KEY", "description": "Brave Search API Key", "secret": True}], "requirements": "需要 Node.js / npx 和 Brave Search API Key。"},
    {"id": "fetch", "name": "Fetch", "description": "读取网页并提取可供模型处理的内容。", "category": "web", "source": "精选", "homepage": "https://github.com/modelcontextprotocol/servers/tree/main/src/fetch", "config": {"type": "stdio", "command": "uvx", "args": ["mcp-server-fetch"]}, "inputs": [], "requirements": "需要安装 uv，确保 uvx 在 PATH 中。"},
    {"id": "filesystem", "name": "Filesystem", "description": "访问你指定目录内的文件。", "category": "devtools", "source": "精选", "homepage": "https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem", "config": {"type": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "${DIRECTORY}"]}, "inputs": [{"name": "DIRECTORY", "description": "允许访问的目录（绝对路径）"}], "requirements": "需要 Node.js / npx。仅填写你希望该服务访问的目录。"},
    {"id": "github", "name": "GitHub", "description": "仓库、Issue、Pull Request 等开发工具。", "category": "devtools", "source": "精选", "homepage": "https://github.com/github/github-mcp-server", "config": {"type": "http", "url": "https://api.githubcopilot.com/mcp/", "headers": {"Authorization": "Bearer ${GITHUB_PAT}"}}, "inputs": [{"name": "GITHUB_PAT", "description": "GitHub Personal Access Token", "secret": True}], "requirements": "需要 GitHub PAT；可用操作由 Token 权限决定。"},
]


def slug(value):
    text = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")[:36]
    return "mcp-" + text + "-" + hashlib.sha256(value.encode()).hexdigest()[:7]


def registry_entry(row):
    """Only offer registry records whose complete launch contract can be represented."""
    server = row.get("server", row)
    if not isinstance(server, dict) or not isinstance(server.get("name"), str):
        return None
    repository = server.get("repository")
    homepage = server.get("websiteUrl") or (repository.get("url", "") if isinstance(repository, dict) else "")
    title = server.get("title")
    base = {"id": slug(server["name"]), "name": title if isinstance(title, str) and title else server["name"], "description": str(server.get("description", ""))[:1500], "source": "官方注册表", "category": "other", "homepage": homepage if isinstance(homepage, str) else "", "inputs": []}
    for remote in server.get("remotes", []):
        if remote.get("type") != "streamable-http":
            continue
        url = remote.get("url", "")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or "{" in url:
            continue
        headers = {}
        inputs = {}
        supported = True
        for header in remote.get("headers", []):
            name = header.get("name")
            if not name:
                supported = False
                break
            value = header.get("value")
            variables = header.get("variables", {})
            if variables:
                if not isinstance(value, str):
                    supported = False
                    break
                for key, spec in variables.items():
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                        supported = False
                        break
                    value = value.replace("${" + key + "}", "{" + key + "}").replace("{" + key + "}", "${" + key + "}")
                    inputs[key] = {"name": key, "description": spec.get("description") or key, "secret": True, "optional": not spec.get("isRequired", True)}
            elif value is None:
                key = re.sub(r"[^A-Za-z0-9_]", "_", name).upper()
                value = "${" + key + "}"
                inputs[key] = {"name": key, "description": header.get("description") or name, "secret": True, "optional": not header.get("isRequired", True)}
            # Some registry records publish a token template without a variables map.
            if isinstance(value, str):
                for key in re.findall(r"\$?\{([A-Za-z_][A-Za-z0-9_]*)\}", value):
                    value = value.replace("${" + key + "}", "{" + key + "}").replace("{" + key + "}", "${" + key + "}")
                    inputs.setdefault(key, {"name": key, "description": header.get("description") or key, "secret": True, "optional": not header.get("isRequired", True)})
            headers[name] = value
        if supported:
            base.update(config={"type": "http", "url": url, "headers": headers}, inputs=list(inputs.values()), requirements="第三方远程服务；按服务要求配置凭据。")
            return checked_entry(base)
    for package in server.get("packages", []):
        kind = package.get("registryType")
        identifier = package.get("identifier", "")
        version = package.get("version", "")
        pattern = r"(?:@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*" if kind == "npm" else r"[A-Za-z0-9][A-Za-z0-9._-]*"
        if kind not in {"npm", "pypi"} or not re.fullmatch(pattern, identifier) or (version and not re.fullmatch(r"[A-Za-z0-9.+_-]+", version)):
            continue
        # Dynamic runtime/argument contracts need an explicit manual configuration.
        if package.get("runtimeArguments"):
            continue
        args = []
        supported = True
        for arg in package.get("packageArguments", []):
            if arg.get("variables") or arg.get("isRepeated") or "value" not in arg:
                supported = False
                break
            if arg.get("type") == "named":
                if not arg.get("name"):
                    supported = False
                    break
                args.append(str(arg["name"]))
            args.append(str(arg["value"]))
        if not supported:
            continue
        env, inputs = {}, []
        for spec in package.get("environmentVariables", []):
            name = spec.get("name", "")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or spec.get("variables"):
                supported = False
                break
            if "value" in spec:
                env[name] = str(spec["value"])
            else:
                env[name] = "${" + name + "}"
                inputs.append({"name": name, "description": spec.get("description") or name, "secret": spec.get("isSecret", True), "optional": not spec.get("isRequired", False), "defaultValue": str(spec.get("default", ""))})
        if supported:
            target = identifier + (("@" if kind == "npm" else "==") + version if version else "")
            base.update(config={"type": "stdio", "command": "npx" if kind == "npm" else "uvx", "args": (["-y"] if kind == "npm" else []) + [target] + args, "env": env}, inputs=inputs, requirements="需要 " + ("Node.js / npx" if kind == "npm" else "uv / uvx") + "。首次连接会启动第三方程序。")
            return checked_entry(base)
    return None


def checked_entry(entry):
    try:
        validate_server(entry["config"])
        for spec in entry["inputs"]:
            spec["description"] = str(spec.get("description", spec["name"]))[:1000]
        return entry
    except (ValueError, TypeError):
        return None


def materialize(entry, values):
    if not isinstance(values, dict):
        raise ValueError("MCP inputs must be an object")
    for spec in entry["inputs"]:
        if not spec.get("optional") and not str(values.get(spec["name"], "")).strip():
            raise ValueError("Missing value: " + spec["name"])
    def expand(value):
        if isinstance(value, str):
            return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: str(values.get(m[1], "")), value)
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        return value
    config = expand(copy.deepcopy(entry["config"]))
    for key in ("env", "headers"):
        config[key] = {k: v for k, v in config.get(key, {}).items() if v.strip()}
    config.update(label=entry["name"], catalogId=entry["id"], enabled=False)
    return validate_server(config)


class MCPMarket:
    def __init__(self):
        self.entries = {entry["id"]: entry for entry in BUILTIN}

    async def search(self, query="", source="builtin", cursor=""):
        if source == "builtin":
            query = query.strip().lower()
            return {"entries": [copy.deepcopy(e) for e in BUILTIN if query in (e["name"] + " " + e["description"]).lower()], "cursor": "", "skipped": 0}
        if source != "registry":
            raise ValueError("Unknown MCP catalog")
        params = {"limit": 30, "version": "latest"}
        if query:
            params["search"] = str(query)[:200]
        if cursor:
            params["cursor"] = str(cursor)[:2000]
        try:
            async with httpx.AsyncClient(timeout=12, follow_redirects=False) as client:
                async with client.stream("GET", REGISTRY_URL, params=params) as response:
                    response.raise_for_status()
                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        raw.extend(chunk)
                        if len(raw) > 4 * 1024 * 1024:
                            raise ValueError("Registry response is too large")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError()
            rows = data.get("servers", [])
            if not isinstance(rows, list):
                raise ValueError()
        except (httpx.HTTPError, ValueError):
            raise ValueError("官方 MCP 注册表暂时不可达或响应无效。可继续使用精选目录，稍后重试。")
        entries, skipped = [], 0
        for row in rows[:100]:
            try:
                entry = registry_entry(row)
            except (AttributeError, TypeError, ValueError):
                entry = None
            if entry:
                self.entries[entry["id"]] = entry
                if not any(e["id"] == entry["id"] for e in entries):
                    entries.append(entry)
            else:
                skipped += 1
        metadata = data.get("metadata")
        cursor = metadata.get("nextCursor", "") if isinstance(metadata, dict) else ""
        return {"entries": entries, "cursor": cursor[:2000] if isinstance(cursor, str) else "", "skipped": skipped}
