# Ultra 主动委派校准（2026-10-30）

## 发现

排查确认 Ultra 可正常保存并进入模型请求，`agent_tool` 也属于默认可见工具；不是简单的菜单未生效或默认容量不足。

实际策略只要求模型“检查”独立调查/验证机会，没有明确鼓励执行委派。同时工具描述将子任务限定为复杂多步工作，且工具说明错误地建议“最终回答需要结果时，使用 run_in_background=false”。实际上需要结果与提前阻塞是两回事；独立任务可以先后台启动，主代理继续推进，汇总前再等待。

这些指令会抬高委派门槛，但仅凭源码不能证明它们是模型不派发的全部原因。

## Codex 对照

本地固定提交 [`c9750d7`](https://github.com/openai/codex/tree/c9750d7f50052b29f53def02b5cf2160ae49a3d9) 的 [`multi_agent_mode_instructions.rs`](https://github.com/openai/codex/blob/c9750d7f50052b29f53def02b5cf2160ae49a3d9/codex-rs/core/src/context/multi_agent_mode_instructions.rs) 将明确请求模式和主动模式作为不同的 developer 指令片段。具体启用还经过 [`session/multi_agents.rs`](https://github.com/openai/codex/blob/c9750d7f50052b29f53def02b5cf2160ae49a3d9/codex-rs/core/src/session/multi_agents.rs) 的配置与模型消息解析，不能说只有一个固定提示词。

官方 [Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents) 区分本地 Codex 的显式/项目指令委派和 ChatGPT Work Ultra 的主动委派；[Responses Multi-agent](https://developers.openai.com/api/docs/guides/responses-multi-agent) 也说明可通过 developer 指令调整主动委派策略。不能把这些入口的默认行为混为“所有 Codex 都会自动拆任务”。

Xuloom 继续让协作档位独立于模型思考强度，不复制 Codex 中 Ultra 思考强度到协作模式的绑定。

## 修改

- Ultra：有收益的独立研究、不同方案、验证任务应主动尽早派发；主代理保留关键主线和最终综合。简单或严格串行任务直接处理。
- Auto 保留选择性委派，Solo 继续禁止委派。权限、思考强度、宿主容量、累计额度和递归规则不变。
- 工具描述改为可交给 worker 的有边界的独立任务，不再要求每个子任务本身都是复杂多步任务。
- 独立任务优先后台发起，只有确实无其他工作可推进时才前台等待；最终仍复用现有结果等待和完成保护。
- 没有增加隐藏规划调用、自动起代理、固定数量或“未派发就反复请求模型”的机制。原程序调用的 `run_in_background` 默认值仍是 false，未改变既有 API 的阻塞/返回语义。
- 修改现有策略和工具说明，没有增加全局提示段。Ultra 策略文本由 1127 减至 1079 字符；数字限额继续由原 Runtime 执行，可查询实际 capacity。

## 验证与边界

协作专项 14 项通过，包含新增的真实请求载荷检查：Ultra 主动策略和非阻塞工具指引确实进入模型请求；Auto 保持选择性策略；简单问题不强制派发；模型直接回答不会触发隐藏重试或额外代理。

- 子代理协作、生命周期、排队、故障回执、程序接口、最终等待等六组定向回归共 38 项通过；没有把它冒充整个 346 项套件的新全量验收。
- 源码桌面协作端到端通过：无模型保存、会话隔离、重启、忙碌保护、停止，以及真实发出的请求载荷中的主动策略和工具指引。
- 软件自带 Python 加载暂存资源、更新后的安装资源，各运行协作 14 项，均通过。
- 原 EXE 资源更新为 beta.20 / 构建 `20261030-153746`；更新后的原 EXE 协作端到端也通过。没有另建启动程序，没有强制结束用户应用。
- 116 个 Python 模块的语法与未定义名字检查通过，TypeScript/Vite 构建通过。

没有调用用户远端模型，不能据此声称子代理调用率已达到某个百分比；需要实际目标模型的相同任务对照才能衡量行为改善。此记录未包含用户聊天正文或私人 Skill。
