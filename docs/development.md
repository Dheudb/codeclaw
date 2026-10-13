# 开发与打包

## 开发环境

建议 Python 3.12+、Node.js 24+、Git、ripgrep。使用仓库根目录 `.venv` 管理 Python 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[mcp]"
cd desktop
npm ci
npm run dev
```

源码桌面端自动寻找 `.venv`，兼容已有 `.venv-mcp`；开发时可用 `CODECLAW_PYTHON` 覆盖。
用户界面不提供解释器选择。便携包始终使用其内置运行时，不读取历史解释器设置或开发覆盖值。

## 依赖的唯一来源

Python 依赖和 CLI 入口在 `setup.py`。`requirements.txt` 只引用当前包，
避免维护两份不同的依赖列表。可选 extras：

| Extra | 用途 |
|---|---|
| `mcp` | MCP SDK，Python 3.10+ |
| `browser` | Python Playwright；还需执行 `python -m playwright install chromium` |
| `tokenizers` | 可选的 tiktoken / Transformers Token 计数 |

基础 CLI 支持 Python 3.9；完整桌面构建使用 Python 3.12+。
前端依赖在 `desktop/package.json` 与 `package-lock.json`，使用 `npm ci` 安装。

## 验证命令

```powershell
# 仓库根目录
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd desktop
npm test
npm run build
npm run test:e2e
npm run test:mcp
```

Electron 端到端验收默认隐藏窗口，使用本地模拟模型 API 和 MCP 协议测试服务，
不调用真实模型或付费搜索。测试覆盖 Skills、审批、后台代理、项目隔离、会话恢复、
MCP 安装与连接、凭据保留和停用重连。截图保存在 `desktop/test-results/`。
本地模型质量、真实搜索源可用性和第三方服务账户权限需要单独验证。

## Windows 便携包

```powershell
cd desktop
# 如构建环境没有仓库 .venv，可以指定构建用 Python 3.12+
$env:CODECLAW_BUILD_PYTHON = "C:\path\to\python.exe"
npm run pack
```

`prepare-runtime.py` 从构建机的标准 Windows CPython 中复制解释器和标准库，
通过正常 wheel 安装把 CodeClaw 与 MCP 依赖放进 `desktop/build/python/`。
它不复制虚拟环境的绝对路径，也不复制构建机全部 site-packages。
运行时清单记录 Python 版本、依赖版本和源码指纹；变化时会重新生成。

`pack.cjs` 组合 Electron、前端、内置 Python 和引擎，输出到
`desktop/release/CodeClaw-win32-x64/`。运行时使用 Python 隔离模式，
不依赖系统 Python、用户 site-packages 或 PYTHONPATH。
完整目录可以移动到其他位置；运行时和第三方许可信息随包保留。

成品验收：

```powershell
$env:CODECLAW_TEST_EXECUTABLE = (Resolve-Path "release/CodeClaw-win32-x64/CodeClaw.exe").Path
npm run test:e2e
npm run test:mcp
```

GitHub Actions 验证 Python 3.9/3.12、前端和 Windows 成品，并上传便携包与测试截图。
`desktop/build`、`desktop/release`、虚拟环境、缓存和本地配置不提交 Git。
