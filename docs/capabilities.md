# Skills 与子代理

## 桌面配置入口

打开项目后，在右侧 Skills / 代理面板点击“怎么配置”，可以查看当前项目和全局的真实目录、
文件模板，并打开对应目录。可点击“添加示例到当前项目”，同时创建 `project-review` Skill 和
`reviewer` 子代理；存在同名文件时会拒绝覆盖。面板支持刷新、查看指令或代理定义。

Skill 用“用于下次任务”加载；子代理在角色卡片中点“分配任务”，补全任务后发送。
模型需支持工具调用，才能实际启动子代理；导入一个定义文件本身不会启动任务。

## Skills

发现顺序为项目 `.codeclaw/skills`、`.agents/skills`、`.claude/skills`、导入目录、
`CODECLAW_SKILLS_PATHS`，最后是用户目录下对应的三个位置。同名的第一个有效且启用的 Skill 生效。
界面显示无效、禁用和被覆盖的条目。

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
不会复制文件；移动或删除原目录会影响使用。修改外部文件后点击刷新。
名称和描述进入模型上下文，完整指令与资源按需读取。选择“用于下次任务”会显式加载指令。
资源访问限制在 Skill 根目录；`allowed-tools` 等声明目前不构成权限授予。

可以导入 `examples/skills/project-review` 体验示例。

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
引用的 Skills 必须有效且启用。自定义定义不能覆盖内置角色。

子代理有独立消息上下文，使用共享项目文件系统，仍受工具权限控制。
子代理默认只接收分配的任务，角色指令和 Skill 放在系统提示中，不继承主会话的聊天请求。
需要参考主会话时可设置 `inherit_context=true`；历史仅供参考，当前任务仍由 `prompt` 指定。
同一批前台子代理调用可并发执行，后台任务最多同时运行十个；代理面板可以查看结果和取消后台任务。
需要立即比较或汇总多个回答时使用前台并发调用；后台任务可通过 `agent_tool action=wait` 等待一个或全部任务的实际结果。
子代理的计划和待办独立维护，不会覆盖主代理的计划和待办；文件修改仍共享项目，分配任务时应避免修改同一个文件。
`model` 可以是 `inherit` 或同一连接提供的模型 ID。

先导入示例 Skill，再导入 `examples/agents`，即可使用示例 reviewer。
