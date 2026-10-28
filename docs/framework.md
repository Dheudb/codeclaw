# 通用执行接口

CLI 与桌面端使用同一套执行引擎。下列能力通过模型工具调用提供，不需要额外启动服务。

## 程序调用（当前开发源码）

`run(text, selected_skills=[...])` 返回 `RunResult`；`submit_message` 的最终事件包含同一回执。使用 `async with QueryEngine(...) as engine`，或在 `finally` 中 `await engine.aclose()`，释放连接、所属任务和会话写者锁。

`await engine.agents(action, **parameters)` 沿用 `agent_tool` 的动作与字段，返回 Python 字典。上层无需从自然语言提取 ID，且所有调用仍经过原权限入口和持久化执行日志。返回字段如下：

| 字段 | 语义 |
| --- | --- |
| `api_version` | 当前为 `1`；新增可选字段向后兼容，改变字段语义需升版本 |
| `session_id` | 框架真实会话 ID |
| `tool_use_id` | 本次授权与持久化执行记录的调用 ID |
| `ok` | 协调操作成功并已保存；不表示所有子任务成功或领域验收通过 |
| `data` | 原生工具生成的完整结构化结果，超过展示阈值也不会只返回截断预览 |
| `error` | 失败时的 `code` 和 `message`，此时 `data=null` |

```python
# request_id 应由上层在首次发送前持久保存，断线后复用。
launch = await engine.agents("spawn", prompt="列出给定材料中的证据缺口",
    count=2, run_in_background=True, request_id="saved-request-42")
if launch["ok"]:
    ids = [row["agent_id"] for row in launch["data"]["agents"]]
    while True:
        state = await engine.agents("wait", agent_ids=ids, timeout_ms=30000)
        if not state["ok"] or all(not row["executing"] for row in state["data"]["agents"]):
            break
```

调用方需要配置审批处理器或明确选择审批策略；此示例不自动授予权限。检查 worker 的 `status` 和 `receipt` 判断真实执行结果。

`spawn` 的可选 `request_id` 支持 1–128 个字母、数字、`_`、`-`。相同参数重复提交返回原记录与 `reused=true`，重启恢复后也不会再启动。相同 ID 修改参数会拒绝。注册中断只保存部分批次时明确报错，调用方应检查实际记录，不能盲目重新发起。省略 ID 的独立调用每次都是新任务。

`list/result/wait/capacity` 读取原生生命周期，`cancel` 停止所属工作，`followup` 明确继续同一 worker。重复读取结果不会重新执行。重启后的历史 `running` 记录显示为 `interrupted`；它不是当前存活证明，不会自动重跑。`wait` 等待任一目标结束即返回，要汇总全部结果需检查所有目标的 `executing`。

事件复用 `engine.tool_context["capability_event"]` 回调：持久化注册后通知启动/状态，保存结果与完成记录后通知完成。回调应快速返回且不抛异常。断线可能丢通知，上层应通过 `list/result` 恢复实际记录，不能仅因缺少通知重新启动任务。

`session_busy` 表示另一个写者占用，`session_conflict` 表示快照已过期；此时没有执行模型或工具。`tool_error` 表示参数、权限或工具操作失败；`persistence_error` 表示无法确认完整存储，可能已有副作用，需要读取实际记录后处理。框架没有为此增加另一套 Job 状态，也不承诺外部系统的 exactly-once 事务。

## 工具发现

目标模式、正式非交互 CLI 和定时消息使用同一引擎与会话，见 [目标与调度](goals-and-schedules.md)。

文件、命令、搜索、网页读取、Skill 和原生委派默认可用。浏览器、Notebook、REPL、LSP、工作区隔离、旧后台任务及团队工具通过 `tool_search_tool` 搜索并激活。激活只改变可见性，不改变权限、Plan 限制或中断恢复确认。

显式列出专用工具的自定义角色可直接加载相应工具，不要求额外允许发现工具。角色限制同时作用于默认和潜在工具，包括只读与 MCP 服务范围，不能通过发现重新引入被排除的能力。

命令工具的 `wait_ms` 单次等待范围为 0–120000 毫秒（最多 120 秒），程序完成时提前返回。它与控制进程总运行时长的 `timeout_ms` 分开设置，等待期间仍可停止任务。

## 工具调用恢复

DSML 中单独用 `arguments` 包裹完整 JSON 参数对象的兼容格式会直接解包；重复字段、混合包裹与普通参数、未知字段仍拒绝执行，解包后继续使用原工具的参数校验和权限检查。

DSML 调用格式错误会反馈具体工具和参数问题，请模型修正后继续；原生调用中的无效 JSON 不会被当成空参数执行。参数校验错误也在执行前反馈。连续失败最多修正两次，单次用户任务合计最多六次修正，之后明确中断并保留已完成的文件、调用结果和子代理记录。主代理与子代理使用同一恢复逻辑，不自动重放成功的操作。

桌面端折叠过程记录保留修正状态。被拒绝的 DSML 原始输出保存在集中历史的 `tool-results/<会话 ID>/protocol-errors/`，用于本地诊断。工具定义保留嵌套参数的完整 JSON Schema；Skill 指令和引用资源不会再被单条工具输出的 5000 字符预览替代，整体上下文预算仍生效。

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
`run_in_background` 允许主代理在子任务执行期间继续编排；最终回答会等待本会话运行中的子代理全部结束，并收集生成回答期间刚返回的结果再汇总。失败、取消或没有文本输出的任务不能算作成功；等待期间仍可停止本轮。
`wait` 的结构化模式使用 `response_format="json"`，默认等待 30 秒。
旧文本 `wait` 等待所选批次全部完成，同样有超时上限。
状态中的 `running_commands` 与 `pending_message_count` 单独计数，不等同于模型正在执行。

默认总容量 10，前台与后台共用容量。可在工作目录 `.xuloom/capabilities.json` 中设置：

```json
{"agent_concurrency": 24}
```

范围为 1–80；单批仍最多 10 个。新上限在没有活跃代理时生效，容量不足时整批拒绝，不虚报启动。
较高上限不会突破供应商的限流，实际请求仍可能因接口容量失败。

历史位于用户级集中存储。重启后未完成任务显示 `interrupted`，不会自动重跑；可明确 `followup` 同一代理。
旧版代理缺少完整角色快照时会明确拒绝恢复，而非用默认角色替代。
更换供应商或 API 地址后，需要恢复原连接才能继续原代理；API Key 不写入角色快照。

## 自定义角色

外部 Skill 可将自己的角色放在技能目录的 `agents/*.md`，并在该技能根目录提供 `mcp_servers.json`。
加载有效、未被同名技能覆盖的 Skill 时，会发现这些角色与 MCP 配置；它们保留在外部目录，不复制进应用或源码。
MCP 的 `command` / `args` 支持 `${python}`（当前 Xuloom 解释器）和 `${skill}`（此技能的绝对目录）。
工作区的 MCP 配置优先于技能默认配置；可在 MCP 管理中禁用技能提供的服务。
角色仍遵循下述工具与权限限制，不会因随 Skill 挂载而获得额外权限。

继续使用 `.xuloom/agents/*.md`（也支持 `.agents/agents`、`.claude/agents` 及导入目录）：

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
- `poll`：单次最长等待 120 秒，不因等待超时终止命令；执行总期限 `timeout_ms` 默认 120 秒、最大 600 秒。
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

技能可在 SKILL.md 的 YAML 中声明 `metadata.xuloom.completion-protocol: workflow-json-v1`。
只有本回合明确选择或通过 `skill_tool read` 加载的技能才启用完成检查。
其命令运行时可输出完整 JSON，以 `run_handle`（也可位于 `artifact_locator`）标识流程，
以布尔 `workflow_finished` 表示完成状态；带 `run_handle` 和 `next_command` 的初始化响应也表示未完成。
同一流程返回 `workflow_finished: true` 才清除未完成状态；明确受阻、失败或取消时可返回
`workflow_status: blocked/failed/cancelled`，并提供原因。检查只读取成功结束命令的完整 stdout，
不从普通文本、分页片段、非零退出码或其他代理输出推断完成状态，不自动执行返回的命令。
命令结果即使分页或进入溢出文件，检查仍使用最多 1 MB 的完整原始输出。
模型只描述下一步却结束时，最多连续提醒两次、每用户回合最多六次；仍未执行则显示中断，保留已有结果。
停止操作优先，新用户回合重新启用选定协议，不强行继续上一回合。普通聊天和未声明协议的技能不受影响。

`tests/test_agent_lifecycle.py` 与 `tests/test_command_sessions.py` 覆盖生命周期、权限和命令输入输出。
`npm run test:framework` 通过隔离的模型接口驱动真实桌面引擎，验证跨回合/重启继续、消息投递、
模型与推理参数、并发查询、UTF-8 输入以及命令历史输出。
