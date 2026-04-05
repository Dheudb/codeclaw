<p align="center">
  <pre align="center">
     ▄██▄              ▄██▄
     ████  ▄████████▄  ████
     ████ █  ██  ██  █ ████
     ▀██ █ ▒▒ vv ▒▒ █ ██▀
     ▀██████████▀
     CodeClaw
  </pre>
</p>

<h1 align="center">CodeClaw</h1>

<p align="center">
  <b>Open-source agentic coding engine for the terminal.</b>
</p>

<p align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.9-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/badge/platform-Windows%20|%20macOS%20|%20Linux-lightgrey" />
</p>

---

<a id="english"></a>

## English

### What is CodeClaw?

CodeClaw is a **fully open-source, Python-native agentic coding CLI** — a ground-up rewrite inspired by Claude Code's architecture. It connects to any LLM provider (Anthropic, OpenAI, or local models) and gives you an AI coding agent right in your terminal.

Unlike hosted solutions, **you own the code, you choose the model, you control the data**.

### Features

| Category | Capabilities |
|---|---|
| **Agent Core** | Multi-turn agentic loop with tool use, budget-based context management, auto-continuation on token limits, session persistence & resume |
| **Tool System** | 20+ built-in tools (aligned to 100K char soft limits) — file read/write/edit, bash, ripgrep, glob, web fetch, REPL, notebook, browser, LSP |
| **Sub-agents** | Spawn child agents, fork mode (inherits parent context), built-in agent types (explore, plan, verify, general-purpose) |
| **Coordinator** | Multi-agent team collaboration — create teams, assign roles, inter-agent messaging |
| **Planning** | Plan mode with tool restrictions, Ultraplan deep-planning sub-agent |
| **Code Intelligence** | LSP integration for real-time diagnostics, auto-commit proposals with attribution, git-worktree sandboxing |
| **Context Management** | Prompt caching (static/dynamic split), FRC (Function Result Clearing), auto-compaction, attachment injection (file diffs, todos, skill recommendations) |
| **Verification** | Auto-spawns a verification sub-agent after 3+ file modifications |
| **Protocols** | MCP (Model Context Protocol) bridge, LSP client/manager |
| **Model Flexibility** | Runtime hot-swap via `/model` — switch provider, model, API URL, key without restarting. Config persists in `~/.codeclaw/config.json` |
| **Feishu Integration** | Built-in Feishu (Lark) WebSocket Gateway for remote team orchestration and interactive UI bridging |
| **Security** | Multi-layer permission system, risk classification, sandbox isolation |

### Quickstart

**Prerequisites:**
- Python ≥ 3.9
- [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) installed and on PATH

**Install:**

```bash
git clone https://github.com/Dheudb/codeclaw.git
cd codeclaw
pip install -e .
```

**Run:**

```bash
codeclaw
```

On first launch, use `/model` to configure your provider:

```
❯ /model

Select protocol:
  1. Anthropic protocol
  2. OpenAI protocol
  3. Local (OpenAI-compatible)

Model name: claude-sonnet-4-20250514
API Base URL: https://api.anthropic.com
API Key: sk-ant-...

✓ Configuration saved to ~/.codeclaw/config.json
```

**Environment variables** (alternative to `/model`):

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_MODEL="claude-sonnet-4-20250514"

# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o"

# OpenAI-compatible local server
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL="my-local-model"

# Provider selection
export CODECLAW_MODEL_PROVIDER="anthropic"  # or "openai"

# Optional
export CODECLAW_MAX_OUTPUT_TOKENS="32000"
```

### CLI Commands

| Command | Description |
|---|---|
| `/model` | Interactively configure provider, model, URL, API key |
| `/model status` | Show current model runtime |
| `/palette` | Open command palette |
| `/plan` | Enable plan mode (read-only tools only) |
| `/plan off` | Disable plan mode |
| `/coordinator` | Enable multi-agent coordinator mode |
| `/tools` | View tool activity |
| `/todos` | View structured todo list |
| `/status` | View runtime status |
| `/sessions` | List past sessions |
| `/resume` | Resume a previous session |
| `/quit` | Exit |

### Architecture

```
codeclaw/
├── main.py              # CLI entry point & TUI
├── setup.py             # Package configuration
├── core/
│   ├── engine.py        # Agent loop, LLM calls, streaming
│   ├── context.py       # System prompt construction
│   ├── config.py        # Persistent configuration
│   ├── permissions.py   # Permission & approval system
│   ├── messages.py      # UUID-tracked message system
│   ├── memory.py        # Context compaction
│   ├── frc.py           # Function Result Clearing
│   ├── verification.py  # Verification sub-agent
│   ├── attachments.py   # Auto-injection (diffs, todos, skills)
│   ├── session.py       # Session persistence
│   ├── hooks.py         # Lifecycle hooks
│   ├── team.py          # Multi-agent team coordination
│   ├── ultraplan.py     # Deep planning sub-agent
│   ├── sandbox.py       # Git-worktree isolation
│   └── ...
├── tools/
│   ├── bash_tool.py     # Shell command execution
│   ├── file_read_tool.py
│   ├── file_edit_tool.py
│   ├── file_write_tool.py
│   ├── grep_tool.py     # ripgrep-powered search
│   ├── glob_tool.py     # File pattern matching
│   ├── agent_tool.py    # Sub-agent spawning & fork
│   ├── web_tool.py      # Web search & fetch
│   ├── browser_tool.py  # Browser automation
│   ├── repl_tool.py     # Python REPL
│   └── ...
└── protocols/
    ├── mcp_bridge.py    # Model Context Protocol
    ├── lsp_client.py    # Language Server Protocol client
    └── lsp_manager.py   # LSP diagnostics manager
```

### License

MIT

---

<a id="中文"></a>

## 中文

### CodeClaw 是什么？

CodeClaw 是一个**完全开源的、纯 Python 实现的 AI 编程智能体 CLI 引擎**——参考 Claude Code 架构从零重写。它可以连接任何 LLM 供应商（Anthropic、OpenAI 或本地模型），在终端里为你提供一个 AI 编程代理。

与托管方案不同，**代码是你的，模型你选，数据你控**。

### 功能特性

| 分类 | 能力 |
|---|---|
| **智能体核心** | 多轮对话智能体循环、工具调用、基于预算的大上下文窗口管理、token 耗尽自动续写、会话持久化 |
| **工具系统** | 20+ 内置工具（全面对齐 100K 字符软限制）——文件读/写/编辑、Bash、ripgrep、Glob、Web、REPL、LSP |
| **子代理** | 生成子代理、Fork 模式（继承父上下文）、内置代理类型（explore、plan、verify、general-purpose） |
| **协调器** | 多代理团队协作——创建团队、分配角色、代理间消息传递 |
| **规划** | Plan 模式（限制为只读工具）、Ultraplan 深度规划子代理 |
| **代码智能** | LSP 实时诊断集成、自动提交建议与归因、Git Worktree 沙盒隔离 |
| **上下文管理** | Prompt 缓存（静态/动态分离）、FRC（函数结果清理）、自动压缩、附件自动注入（文件 diff、待办、技能推荐） |
| **验证** | 修改 3+ 个文件后自动生成验证子代理 |
| **协议** | MCP（模型上下文协议）桥接、LSP 客户端/管理器 |
| **模型灵活性** | 通过 `/model` 命令运行时热切换——无需重启即可更换供应商、模型、API 地址、密钥。配置持久化到 `~/.codeclaw/config.json` |
| **飞书网关** | 内置飞书 (Lark) WebSocket 网关，支持通过 IM 远程直接操控智能体团队与跨端交互式选择题渲染 |
| **安全** | 多层权限系统、风险分类、沙盒隔离 |

### 快速开始

**前置要求：**
- Python ≥ 3.9
- 安装 [ripgrep](https://github.com/BurntSushi/ripgrep)（`rg` 命令需在 PATH 中）

**安装：**

```bash
git clone https://github.com/Dheudb/codeclaw.git
cd codeclaw
pip install -e .
```

**启动：**

```bash
codeclaw
```

首次启动后，使用 `/model` 配置供应商：

```
❯ /model

选择协议：
  1. Anthropic 协议
  2. OpenAI 协议
  3. 本地模型（OpenAI 兼容）

Model name: claude-sonnet-4-20250514
API Base URL: https://api.anthropic.com
API Key: sk-ant-...

✓ 配置已保存到 ~/.codeclaw/config.json
```

**环境变量**（`/model` 的替代方案）：

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_MODEL="claude-sonnet-4-20250514"

# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o"

# OpenAI 兼容的本地服务器
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL="my-local-model"

# 供应商选择
export CODECLAW_MODEL_PROVIDER="anthropic"  # 或 "openai"

# 可选
export CODECLAW_MAX_OUTPUT_TOKENS="32000"
```

### CLI 命令

| 命令 | 说明 |
|---|---|
| `/model` | 交互式配置供应商、模型、URL、API 密钥 |
| `/model status` | 查看当前模型运行时状态 |
| `/palette` | 打开命令面板 |
| `/plan` | 启用 Plan 模式（仅限只读工具） |
| `/plan off` | 关闭 Plan 模式 |
| `/coordinator` | 启用多代理协调器模式 |
| `/tools` | 查看工具活动 |
| `/todos` | 查看结构化待办列表 |
| `/status` | 查看运行时状态 |
| `/sessions` | 列出历史会话 |
| `/resume` | 恢复历史会话 |
| `/quit` | 退出 |

### 架构

```
codeclaw/
├── main.py              # CLI 入口 & 终端 UI
├── setup.py             # 包配置
├── core/
│   ├── engine.py        # 智能体循环、LLM 调用、流式处理
│   ├── context.py       # 系统提示词构建
│   ├── config.py        # 持久化配置
│   ├── permissions.py   # 权限审批系统
│   ├── messages.py      # UUID 追踪消息系统
│   ├── memory.py        # 上下文压缩
│   ├── frc.py           # 函数结果清理
│   ├── verification.py  # 验证子代理
│   ├── attachments.py   # 自动注入（diff、待办、技能）
│   ├── session.py       # 会话持久化
│   ├── hooks.py         # 生命周期钩子
│   ├── team.py          # 多代理团队协调
│   ├── ultraplan.py     # 深度规划子代理
│   ├── sandbox.py       # Git Worktree 隔离
│   └── ...
├── tools/
│   ├── bash_tool.py     # Shell 命令执行
│   ├── file_read_tool.py
│   ├── file_edit_tool.py
│   ├── file_write_tool.py
│   ├── grep_tool.py     # 基于 ripgrep 的搜索
│   ├── glob_tool.py     # 文件模式匹配
│   ├── agent_tool.py    # 子代理生成 & Fork
│   ├── web_tool.py      # 网页搜索 & 抓取
│   ├── browser_tool.py  # 浏览器自动化
│   ├── repl_tool.py     # Python REPL
│   └── ...
└── protocols/
    ├── mcp_bridge.py    # 模型上下文协议
    ├── lsp_client.py    # 语言服务器协议客户端
    └── lsp_manager.py   # LSP 诊断管理器
```

### 许可证

MIT
