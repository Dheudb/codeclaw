# Xuloom

A local general-purpose Agent framework with **Desktop + CLI on one Python agent engine**.

[中文](README.md) · [Documentation](docs/README.md) · [Development](docs/development.md)

Xuloom supports independent conversations and projects, with model conversations, file and shell
tools, Skills, custom subagents, MCP connections, permissions and session history.
The desktop uses Electron and React. It is currently a preview.

## Desktop

[**Download Windows x64 portable**](https://github.com/Dheudb/xuloom/releases/tag/v0.1.0-beta.10) · [All releases](https://github.com/Dheudb/xuloom/releases)

Download `Xuloom-v0.1.0-beta.2-windows-x64.zip` from Assets, extract the entire
folder, and run `Xuloom.exe`. The automatic Source code downloads contain source,
not a runnable application. Keep the `resources` directory beside the executable.

The Windows portable package includes Python and the MCP dependencies. Extract the
entire package and run `Xuloom.exe`; users do not install Python or select an
interpreter. Open a project and configure the model endpoint in Settings.
Some third-party local MCP servers still require their own Node.js, uv or other runtime.

This repository contains source. Run `npm run pack` in `desktop/` to build a
Windows package. Public downloads are published in Releases; the desktop GitHub
Actions workflow currently runs only when manually dispatched.

## Source setup

Use Python 3.12+, Node.js 24+, Git and ripgrep for development. The base CLI supports
Python 3.9+; the MCP extra needs Python 3.10+.

```bash
git clone https://github.com/Dheudb/xuloom.git xuloom
cd xuloom
python -m venv .venv
# Activate .venv using your shell's command
python -m pip install -e ".[mcp]"
cd desktop
npm ci
npm run dev
```

Source builds discover the repository's `.venv`; developers can override it with
`XULOOM_PYTHON`. Portable packages always use their bundled runtime.
Run `xuloom` from the project you want to work on to use the CLI.

## Capabilities

- Project sessions and streaming conversations, shared with the CLI after saving.
- Skills discovery, import, validation, `$` selection and resource loading.
- Built-in and custom subagents with foreground/background tasks and cancellation.
- MCP curated catalog, official registry browsing, stdio/Streamable HTTP configuration and connection tests.
- Built-in DuckDuckGo search, static page extraction and external search MCP tools.
- File/shell tools, permission prompts, planning and Git changes.

Anthropic Messages and OpenAI-compatible Chat Completions are implemented; this is
not a guarantee that every model or gateway supports every agent feature.
The current development source adds session writer leases and checkpoint revision checks; this is not yet in the beta.2 download. Read-only viewing remains available, and a second writer must wait for release. See [development scope and migration](docs/general-framework.md).

MCP OAuth, legacy SSE, automatic updates,
signed installers and macOS/Linux desktop distributions are not implemented.
The Feishu integration has been removed.

## Repository

| Path | Purpose |
|---|---|
| `xuloom/` | Python engine, CLI, desktop bridge, Skills, subagents and protocols |
| `desktop/` | Electron, React, build scripts and UI tests |
| `tests/` | Python tests and offline MCP protocol fixtures |
| `examples/` | Importable Skills and agent definitions |
| `docs/` | Usage, configuration, architecture and development guides |

Credentials, sessions, virtual environments and generated packages are ignored by
Git. See [development](docs/development.md) for test and packaging commands.

[MIT License](LICENSE)
