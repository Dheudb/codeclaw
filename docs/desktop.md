# CodeClaw Desktop Preview

CodeClaw Desktop adds an Electron + React workbench to the existing Python engine.
The CLI remains available. Each open desktop session owns a separate Python process
whose working directory is the selected project, avoiding process-global directory
changes across projects.

## Run from source

Prerequisites: Node.js 24+, Python 3.9+, Git, and ripgrep on PATH.

```powershell
# Repository root
python -m pip install -e .
cd desktop
npm ci
npm run dev
```

For a production frontend build, run `npm run build` followed by `npm start`.
`npm run pack` creates a portable Windows application under
`desktop/release/CodeClaw-win32-x64/CodeClaw.exe` (architecture follows the host).
The portable preview bundles Electron, the interface and Python source. It requires
an existing Python interpreter with the package dependencies; it does not bundle
Python. Choose that interpreter in Settings, or set `CODECLAW_PYTHON` before launch.

Open a project folder, configure a provider/model in Settings, and submit a task.
OpenAI-compatible endpoints use the OpenAI protocol. Model IDs are passed through
to the provider. The existing `~/.codeclaw/config.json` stores model configuration;
the desktop receives only a `hasKey` flag, not the saved API key. An optional
`CODECLAW_CONFIG_DIR` overrides the configuration directory for isolated testing.

## Skills

Discovery precedence is project `.codeclaw/skills`, `.agents/skills`,
`.claude/skills`, imported directories, `CODECLAW_SKILLS_PATHS`, and the same three
directories under the user's home. The first valid, enabled skill of a given name
wins. The UI shows invalid, disabled and shadowed entries explicitly.

Use the Skills panel to import a single skill directory or a collection. Imports
mount the existing directory, including its resources; they do not copy it.
Project-specific roots and disabled IDs persist in `.codeclaw/capabilities.json`.
Refresh after changing external skill files. This registry is also used by the CLI.

```text
my-skill/
  SKILL.md
  references/
  scripts/
  assets/
```

`SKILL.md` needs YAML frontmatter with `name` and `description`; the name must
match its directory and use lowercase ASCII letters, digits and single hyphens.
Metadata is included in the model's tool instructions. The model can load the full
skill with `skill_tool(action="read", skill_name="my-skill")`; users can also select
it in the desktop to include it explicitly in the next task. Referenced text is
available through `action="resource", resource_path="references/example.md"`.
Resource traversal outside the skill directory is rejected, including symlinks.

The loaded instructions identify the absolute skill directory and project directory
separately. References and scripts resolve from the skill directory; project work
resolves from the project. Script execution uses the existing tool permission gate.
The optional `compatibility` field is displayed as an environment requirement.
Platform-specific tools, connectors and runtimes must be installed separately;
importing a skill does not provide them. Experimental `allowed-tools` metadata does
not grant permissions or override the session's policy.

Try `examples/skills/project-review` using Import Skill Directory.

## Custom subagents

Agent definitions are Markdown files directly inside `.codeclaw/agents`,
`.agents/agents`, `.claude/agents`, an imported agent directory, or the equivalent
global directories. Project definitions precede imported and global definitions.
Custom definitions cannot override built-in agent names.

```markdown
---
name: reviewer
description: Review implementation and report concrete issues
model: inherit
tools: Read, Grep, Glob, Skill
skills: [project-review]
---
You are an independent reviewer. Inspect the task and report findings.
```

`tools` accepts a list or comma-separated names. Supported aliases are Read, Write,
Edit, Bash, Grep, Glob and Skill; exact CodeClaw tool names are also accepted.
An empty list grants no tools, while omitted tools use the allowed inherited set.
Per-command patterns such as `Bash(git:*)` are rejected explicitly. `model` is a
provider model ID or `inherit`. Models use the parent's provider and credentials;
per-agent provider switching is outside this preview. Required skills must be
available and enabled or the definition is shown as invalid.

The agent tool supports foreground tasks and up to four background workers:

```python
agent_tool(prompt="Inspect the implementation", subagent_type="reviewer",
           run_in_background=True, model="provider-model-id")
agent_tool(action="list")
agent_tool(action="result", agent_id="subagent-...")
agent_tool(action="cancel", agent_id="subagent-...")
```

Each worker has its own message context, tool set, permission memory and transcript.
Its working directory remains the parent project. Workers share selected runtime
services and the project filesystem; parallel writes are not isolated worktrees.
Background workers do not merge their plan/todo state over the parent's state.
The desktop receives status/result events, and the parent receives completed
background reports at the next model turn. Team members register when a team is
active and receive messages at model-turn boundaries.

Import `examples/agents` after importing the example skill to try `reviewer`.

## Desktop lifecycle and verification

- Chat uses structured engine events over stdio, not terminal-output scraping.
- Tool permission requests, user questions and plan approval wait for UI responses.
- Stop cancels the foreground turn, pending interactions and background subagents.
- Sessions persist in each project's `.codeclaw/sessions`; history can be reopened.
- Running workers from a prior process are marked interrupted, not silently replayed.
- Git Changes shows tracked diffs and lists untracked paths separately.
- Existing project `mcp_servers.json` is connected through the Python MCP bridge
  when the optional MCP package is installed. Each connection has its own owner
  task, including transport cleanup and failure isolation.
- The renderer has no Node integration and uses an allowlisted IPC interface.

```powershell
# Repository root
python -m unittest discover -s tests -v
cd desktop
npm test
npm run build
npm run test:e2e
```

The desktop E2E starts a local simulated OpenAI-compatible service and real Python
engines. It checks skill injection, a permission-gated file write, two background
subagents, project isolation and session restoration without spending model credits.
It does not measure the quality of any real model or validate every third-party
skill. Screenshots are generated in the ignored `desktop/test-results` directory.

Automatic updates, a plugin marketplace, bundled Python, signed installers and
arbitrary platform-plugin compatibility remain outside this first desktop preview.

## MCP 市场与联网搜索

打开项目后，点击侧边栏底部的 **MCP 市场**。

- **精选目录**：包含 Exa Search、Brave Search、Fetch、Filesystem 和 GitHub。
- **官方 MCP 注册表**：按名称或用途搜索、加载下一页。仅展示当前能完整表达
  连接方式及启动参数的记录；其余条目可以参考服务文档手动配置。网络失败不影响精选目录。
- **我的服务**：编辑配置、测试连接、启用、停用和移除。
- **自定义服务**：支持本地 stdio 程序及远程 Streamable HTTP MCP，包括手动配置请求头。
  OAuth 登录和旧版 SSE 连接暂未实现。

安装只保存配置，新服务默认停用。**测试连接**会实际启动本地程序或访问远程服务，
执行 MCP 握手并发现工具，不需要配置模型 API。测试本身不执行搜索工具。
启用后，下一次任务会加载服务；已有会话也可点击“刷新连接”。其他同项目会话会在下一次
任务前重新读取配置，正在执行的任务继续使用它启动时的连接。CLI 在启动时读取同一份配置。
工具执行仍走现有权限流程。服务本身的凭据、额度和费用由服务提供方决定。

MCP 需要 **Python 3.10+**，使用该解释器在仓库根目录运行：

```powershell
python -m pip install -e ".[mcp]"
```

在“设置与模型”选择安装了这些依赖的 Python 解释器，然后重新打开项目/会话。
Python 3.9 仍可使用原有 CLI、桌面和内置搜索，但不能运行当前官方 Python MCP SDK。
市场会显示当前解释器及缺失依赖，不会静默忽略已启用的服务。

配置写入项目根目录 `mcp_servers.json`，与 CLI 共用。它可能包含凭据，已加入忽略规则，
也不会自动附加到模型上下文。环境变量和 HTTP 请求头的已保存值不会回传界面，
编辑时留空保留原值，删除字段清除。凭据仍以明文存储在本地配置文件；这不是系统密钥库。
手动把凭据写入命令参数时，参数会显示在配置预览中，请优先使用环境变量或请求头。

内置 DuckDuckGo 搜索保持独立，不需要 MCP 或搜索 API Key；请求在后台线程运行，
避免阻塞流式对话与停止操作。它不提供服务可用性保证；搜索为空或失败会明确返回结果。
内置 WebFetch 提取静态 HTML 的纯文本，动态网页仍需浏览器工具。
配置搜索 MCP 后，模型可以选择它提供的搜索工具；当前没有自动切换搜索源的路由规则。

```powershell
# Run with the Python interpreter containing the MCP extra
python -m unittest discover -s tests -v
cd desktop
$env:CODECLAW_PYTHON = "C:\path\to\python.exe"
npm run test:mcp
```

MCP 验收使用隐藏 Electron 窗口、本机 MCP 协议测试服务和模拟模型 API，
覆盖市场安装、凭据保留/脱敏、stdio/HTTP 握手、权限审批后的工具调用以及停用重连。
不调用真实模型或付费搜索服务。第三方市场条目实际联网可用性需要使用自己的配置验证。
