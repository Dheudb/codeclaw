# 通用执行接口

CLI 与桌面端使用同一套执行引擎。下列能力通过模型工具调用提供，不需要额外启动服务。

## 子代理

`agent_tool` 支持：

| action | 行为 |
| --- | --- |
| `spawn` | 新建独立代理；`count=1..10` 对同一任务批量并行；不同任务可同轮多次调用 |
| `capacity` | 返回实际并发上限、使用中与剩余容量 |
| `list` / `result` | 返回 JSON 状态；`executing` 表示当前是否真的有执行任务，不用历史状态推断存活 |
| `send` | 将 `prompt` 排入指定代理的邮箱，在下一次模型调用前读取；不会自动唤醒空闲代理 |
| `followup` | 按 `agent_id` 继续原代理，保留消息历史、角色指令和模型；运行中需先等待或使用 `send` |
| `wait` | 等待指定 `agent_ids` 中任意一个结束，最长 60 秒；超时保留正在执行的任务 |
| `cancel` | 取消任务及其命令会话，保留历史供后续继续 |
| `close` | 释放空闲代理及其命令会话，保留历史；关闭后不能继续该代理 |

`spawn` 默认保留文本回执；自动调度应指定 `response_format="json"`。
`wait` 的结构化模式使用 `response_format="json"`，默认等待 30 秒。
旧文本 `wait` 等待所选批次全部完成，同样有超时上限。
状态中的 `running_commands` 与 `pending_message_count` 单独计数，不等同于模型正在执行。

默认总容量 10，前台与后台共用容量。可在工作目录 `.codeclaw/capabilities.json` 中设置：

```json
{"agent_concurrency": 24}
```

范围为 1–80；单批仍最多 10 个。新上限在没有活跃代理时生效，容量不足时整批拒绝，不虚报启动。
较高上限不会突破供应商的限流，实际请求仍可能因接口容量失败。

历史位于用户级集中存储。重启后未完成任务显示 `interrupted`，不会自动重跑；可明确 `followup` 同一代理。
旧版代理缺少完整角色快照时会明确拒绝恢复，而非用默认角色替代。
更换供应商或 API 地址后，需要恢复原连接才能继续原代理；API Key 不写入角色快照。

## 自定义角色

继续使用 `.codeclaw/agents/*.md`（也支持 `.agents/agents`、`.claude/agents` 及导入目录）：

```markdown
---
name: evidence-reader
description: 查找资料并给出来源
model: inherit
tools: Read, Grep, Glob, Skill, web_search_tool, web_fetch_tool
web_search: enabled
mcp_servers: []
read_only: true
---
围绕分配的问题检索和阅读资料，区分证据与推断。
```

`web_search: disabled` 排除搜索、抓取和浏览器工具；`enabled` 只允许继承宿主已有能力，不会安装新工具。
省略 `mcp_servers` 继承宿主可用 MCP；`[]` 禁用 MCP；字符串列表只保留指定服务。
`read_only: true` 只保留声明为只读的工具，命令工具也会被排除。这是工具可用性约束，不是操作系统沙箱。
角色工具和潜在工具一起过滤，不能通过搜索工具重新激活被排除的工具。默认子代理不再嵌套委派。
不支持的 `sandbox_mode` 会报错，不会假装已有相应隔离。

子代理默认继承父任务的模型、供应商和思考设置。`reasoning_effort` 可显式设置 `low/medium/high/xhigh`，
目前通过 OpenAI 兼容接口传递，由供应商和具体模型决定是否支持；不兼容会报错，不自动忽略。
Anthropic 使用现有 thinking 配置继承，不接受此显式覆盖字段。

## 命令会话

新增 `command_tool`，保留已有 `bash_tool` 和后台任务接口。

- `start`：优先使用 `argv`（程序和参数数组），也可使用 `command` 执行 shell；默认工作目录为当前会话目录。
- `stdin`：直接传入 UTF-8 内容，不需要 shell 引号、临时文件或 Base64；默认发送后关闭输入。
- `keep_stdin_open=true`：保留管道输入，后续 `write` 发送文本，`eof=true` 关闭输入。
- `poll`：单次最长等待 10 秒，不因等待超时终止命令；执行总期限 `timeout_ms` 默认 120 秒、最大 600 秒。
- `read`：通过 `command_id`、`stream=stdout/stderr` 和 `offset` 分页，下一页使用返回的 `next_offset` 字节游标。
- `cancel`：停止该命令及其进程组/进程树；`list` 只展示创建代理拥有的命令。
- `capabilities`：查询实际能力；当前支持交互管道，不支持依赖真实 TTY 的终端程序。

标准输出和错误输出分开保存到集中存储中的会话工具资源目录。分页不会静默删除中间内容，
UTF-8 多字节字符跨页保留；只有读到 `eof=true` 且命令成功结束（`completed`、退出码 0），才能把整段控制 JSON 当作成功响应。
每个输出流上限 50 MiB，超限明确返回 `output_limit` 并终止进程。
命令 ID 限创建它的代理使用；同一代理继续任务时可继续访问。
正常完成子代理模型回合不会自动停止其独立命令，但取消/关闭代理或停止会话会清理它们。
重启后保留输出，旧进程显示 `interrupted`，不会按历史 PID 重新连接或终止进程。

启动、写输入和取消都遵循现有批准权限，Plan 阶段不能借此执行命令。

## MCP

现有 `mcp_servers.json` 每项服务可设置 `tool_timeout_sec`（1–600，默认 20 秒）。
界面编辑其他字段会保留已设置的超时。支持含连字符的合法环境变量名称，凭据仍按原规则隐藏。

## 验证

`tests/test_agent_lifecycle.py` 与 `tests/test_command_sessions.py` 覆盖生命周期、权限和命令输入输出。
`npm run test:framework` 通过隔离的模型接口驱动真实桌面引擎，验证跨回合/重启继续、消息投递、
模型与推理参数、并发查询、UTF-8 输入以及命令历史输出。
