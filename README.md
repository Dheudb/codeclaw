# CodeClaw

本地通用 Agent 框架：**桌面 App + CLI，共用 Python 智能体引擎**。

[English](README.en.md) · [使用文档](docs/README.md) · [开发说明](docs/development.md)

CodeClaw 提供普通对话、文件与命令工具、Skills、自定义子代理和 MCP 服务；领域工作流程通过扩展加载。
桌面端采用 Electron + React，CLI 可以单独使用。当前桌面版处于预览阶段。

## 从哪里开始

| 我想做什么 | 入口 |
|---|---|
| 直接使用桌面程序 | [桌面使用说明](docs/desktop.md) |
| 在终端使用 CodeClaw | [CLI 使用说明](docs/cli.md) |
| 接入模型或本地推理服务 | [模型配置与兼容范围](docs/models.md) |
| 添加 Skills 或子代理 | [Skills 与子代理](docs/capabilities.md) |
| 子代理续接、容量查询和命令输入输出 | [通用执行接口](docs/framework.md) |
| 通用化改造、会话占用与兼容迁移 | [通用框架进展与边界](docs/general-framework.md) |
| 安装搜索等外部工具 | [MCP 市场与搜索](docs/mcp.md) |
| 修改代码、运行测试或打包 | [开发说明](docs/development.md) · [项目结构](docs/architecture.md) |

## 桌面版

[**下载 Windows x64 便携版**](https://github.com/Dheudb/codeclaw/releases/tag/v0.1.0-beta.7) · [所有版本](https://github.com/Dheudb/codeclaw/releases)

在 Release 的 Assets 中下载 `CodeClaw-v0.1.0-beta.7-windows-x64.zip`，完整解压后运行 `CodeClaw.exe`。
请下载成品 ZIP；GitHub 自动提供的 Source code 是源码。程序依赖同目录的 `resources`，不能只复制 EXE。
Windows 便携包内置 Python 运行时和 MCP 依赖。
**使用便携包不需要安装 Python、Node.js，也不需要选择解释器。**
在“设置与模型”填写协议、API 地址、模型 ID 和 Key，点击“新建会话”即可使用默认工作区。
默认目录按当前系统和用户自动设置，可在设置中修改；处理已有代码时再选择“打开项目”。
仅打开界面、管理 Skills 或配置 MCP 不需要调用模型。

本仓库保存源码；Windows 包由 `npm run pack` 构建，输出到 `desktop/release/`。
公开下载包在 Releases 发布；GitHub Actions 目前仅手动触发。
第三方本地 MCP 服务可能需要它自己的 Node.js、uv 或其他运行环境，服务安装卡片会说明。

## 从源码启动

开发环境建议使用 Python 3.12+、Node.js 24+ 和 Git。CLI 基础功能兼容 Python 3.9+；
MCP 需要 Python 3.10+。`rg` 用于本地代码搜索，需要在 PATH 中。

```powershell
git clone https://github.com/Dheudb/codeclaw.git
cd codeclaw
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[mcp]"
cd desktop
npm ci
npm run dev
```

源码桌面端自动寻找仓库中的 `.venv`；开发者也可以用 `CODECLAW_PYTHON` 指定运行环境。
这只影响源码开发，正式便携包始终使用随应用附带的运行时。

CLI 可以直接使用同一虚拟环境：

```powershell
# 仓库根目录安装后，进入你要处理的项目目录
C:\path\to\codeclaw\.venv\Scripts\codeclaw.exe
```

## 当前能力与边界

| 能力 | 当前实现 |
|---|---|
| 桌面与 CLI | 共用引擎、模型配置和同一项目的会话格式 |
| Skills | 项目/用户/导入目录发现，校验、输入 `$` 选择和按需加载资源 |
| 子代理 | 内置/自定义角色、并行批次、容量查询、消息投递、按 ID 续接及取消 |
| MCP | 精选目录、官方注册表浏览、自定义 stdio / Streamable HTTP、连接测试和启停 |
| 联网 | 内置 DuckDuckGo 搜索、静态网页读取，以及外部搜索 MCP |
| 工具与审批 | 文件读写、命令执行、规划、权限卡片和 Git 变更查看 |

- 模型接入支持 Anthropic Messages 和 OpenAI 兼容 Chat Completions；不同模型的工具调用、推理和多模态能力需要单独验证。
- 当前源码为会话增加跨进程单写者锁与检查点版本校验：另一个入口可以查看历史，执行或修改历史前需要原写者释放。没有实时同步，也不能通过删除锁文件接管。此项属于尚未发布的通用化改造，不能据此认定 beta.2 下载包已具备。
- 默认提示词和内置子代理保持领域中立；编程流程通过可选的 `code-development` Skill 加载，普通文件修改不再自动触发编程检查。
- 工作区隔离使用 worktree 或目录副本，不限制进程、网络或宿主文件访问，不是操作系统安全沙箱。
- MCP OAuth 登录、旧版 SSE 连接、插件市场、自动更新和签名安装器尚未实现。
- Windows 便携包已经内置运行时；macOS/Linux 桌面分发尚未提供。
- 飞书集成已经移除，项目以本地桌面和 CLI 使用为主。

## 项目目录

```text
codeclaw/       Python 引擎、CLI、桌面桥接、Skills、代理和 MCP
desktop/        Electron 主进程、React 界面、构建脚本和界面测试
tests/          Python 测试及本地 MCP 协议测试服务
examples/       可导入的 Skills 和自定义代理示例
docs/           使用、配置、架构和开发文档
.github/        自动测试与 Windows 便携包构建
setup.py        Python 依赖与 CLI 入口的统一定义
```

会话、密钥、虚拟环境、依赖目录和构建产物不进入 Git。
模型配置在 `~/.codeclaw/config.json`，会话集中存储在当前用户数据目录的 `CodeClaw/history/`（Linux 为 `codeclaw/history/`），
MCP 配置在项目根目录 `mcp_servers.json`。

## 开发验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd desktop
npm test
npm run build
npm run test:e2e
npm run test:mcp
```

端到端测试使用隐藏窗口、本机协议测试服务和模拟模型 API，不需要真实模型 Key。
完整打包与便携性检查见[开发说明](docs/development.md)。

## 许可证

[MIT](LICENSE)
