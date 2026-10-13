# CLI

## 安装与启动

基础 CLI 需要 Python 3.9+。建议使用 Python 3.12 的虚拟环境，便于同时使用 MCP。

```bash
python -m venv .venv
# 激活虚拟环境后
python -m pip install -e ".[mcp]"
codeclaw --help
```

进入要处理的项目目录后运行 `codeclaw`。首次使用 `/model` 配置模型。
从源码安装的 CLI 需要本机 Python；Windows 桌面便携包的用户不需要配置 Python。

## 常用入口

```text
codeclaw                         交互式会话
codeclaw -p "解释这个项目"         带初始任务启动
codeclaw --continue               恢复当前项目最近一次会话
codeclaw --resume SESSION_ID      恢复指定会话
```

`-p` 是交互式会话的初始消息，不代表执行完自动退出的批处理模式。

| 命令 | 用途 |
|---|---|
| `/model`、`/model status` | 配置、查看模型连接 |
| `/palette` | 命令面板 |
| `/plan`、`/plan off` | 切换规划模式 |
| `/coordinator` | 多代理协调模式 |
| `/tools`、`/todos`、`/status` | 查看工具、待办及运行状态 |
| `/sessions` | 列出当前项目会话 |
| `/resume <ID>` | 恢复会话 |
| `/quit` | 退出 |

Skills、子代理和 MCP 与桌面端共用引擎实现。CLI 在启动时读取项目 `mcp_servers.json`；
修改配置后重启 CLI 以加载。桌面可以在 MCP 市场中编辑这份配置。
