# Skills 与子代理

## 桌面配置入口

打开项目后，在右侧 Skills / 代理面板点击“怎么配置”，可以查看当前项目和全局的真实目录、
文件模板，并打开对应目录。可点击“添加示例到当前项目”，同时创建 `project-review` Skill 和
`reviewer` 子代理；存在同名文件时会拒绝覆盖。面板支持刷新、查看指令或代理定义。

在任务输入框输入 `$` 弹出 Skill 列表；继续输入名称或描述筛选，↑/↓ 选择，Enter/Tab 确认，Esc 关闭。
列表直接扫描真实技能目录，显示名称、描述和来源，只列出有效且未被覆盖的技能。
选中后以可移除标签挂载，可继续输入 `$` 选择多个技能；发送时读取对应的完整 `SKILL.md`。
也可使用右侧“选择使用”。子代理在角色卡片中点“分配任务”，补全任务后发送。
模型需支持工具调用，才能实际启动子代理；导入一个定义文件本身不会启动任务。

## Skills

发现顺序为项目 `.codeclaw/skills`、`.agents/skills`、`.claude/skills`、导入目录、
`CODECLAW_SKILLS_PATHS`、用户目录下对应的三个位置，最后是随应用分发的 `codeclaw/skills/builtin`。
同名的第一个有效的 Skill 生效，自定义技能可以覆盖内置技能。
所有格式正确的技能默认可用，无需启用开关；旧的停用配置不再影响可用性。界面显示无效和被覆盖的条目。

```text
my-skill/
  SKILL.md
  references/
  scripts/
  assets/
```

```markdown
---
name: my-skill
description: Describe when and how to use this skill.
---

Instructions go here.
```

名称应与目录名一致，使用小写 ASCII 字母、数字和单个连字符。导入会挂载已有目录，
不会复制文件；移动或删除原目录会影响使用。每次打开 `$` 列表都会重新扫描目录，也可在右侧点击刷新。
名称和描述进入模型上下文，完整指令与资源按需读取。选择“选择使用”会显式加载指令。
资源访问限制在 Skill 根目录；`allowed-tools` 等声明目前不构成权限授予。

可以导入 `examples/skills/project-review` 体验示例。

源码自带三个真实的技能目录：`code-review`（代码审查）、`debug-issue`（故障排查）、
`web-research`（联网研究与链接引用）。它们与用户技能共用发现和加载逻辑，同时适用于 CLI 和桌面应用。
安装包包含这些 `SKILL.md`，用户无需单独下载，也不会将内置技能复制到用户项目里。

## 自定义子代理

定义放在项目/用户的 `.codeclaw/agents`、`.agents/agents`、`.claude/agents`，
也可以在代理面板导入目录。每个代理使用一个 Markdown 文件：

```markdown
---
name: reviewer
description: Review code independently.
model: inherit
tools: Read, Grep, Glob, Skill
skills: [project-review]
---

Inspect the task and report concrete findings. Do not modify files.
```

工具别名支持 Read、Write、Edit、Bash、Grep、Glob、Skill，以及实际工具名。
省略 `tools` 会继承可用工具集合；空列表表示不提供工具。
引用的 Skills 必须有效。自定义定义不能覆盖内置角色。

子代理有独立消息上下文，使用共享项目文件系统，仍受工具权限控制。
子代理默认只接收分配的任务，角色指令和 Skill 放在系统提示中，不继承主会话的聊天请求。
需要参考主会话时可设置 `inherit_context=true`；历史仅供参考，当前任务仍由 `prompt` 指定。
同一批前台子代理调用可并发执行，后台任务最多同时运行十个；代理面板可以查看结果和取消后台任务。
给多个子代理分配同一个任务时，模型应一次调用 `agent_tool(prompt="你好", description="问候测试", count=10)`。
`count` 支持 1–10；一次创建全部独立任务并并发请求模型，结果按创建顺序返回，卡片显示编号和启动时间。
不同任务可在同一回复中提交多个调用。模型在多轮回复中逐个调用仍会逐个创建；供应商的并发限额也可能让已启动的请求排队。
需要立即比较或汇总多个回答时使用前台并发调用；后台任务可通过 `agent_tool action=wait` 等待一个或全部任务的实际结果。
子代理的计划和待办独立维护，不会覆盖主代理的计划和待办；文件修改仍共享项目，分配任务时应避免修改同一个文件。
`model` 可以是 `inherit` 或同一连接提供的模型 ID。

先导入示例 Skill，再导入 `examples/agents`，即可使用示例 reviewer。

## 移动会话到其他项目

将独立会话或项目会话拖到侧栏的目标项目，也可在会话菜单选择“移动到项目…”。
历史、过程记录、草稿、名称和置顶状态保留；后续文件操作、命令、Skills、子代理和 MCP 使用目标项目配置。
运行中的会话及子代理需要先停止。移动会话不会移动项目代码文件。
移动前会检查目标记录冲突，失败时保留原记录；移动后重启仍恢复到目标项目。
