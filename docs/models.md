# 模型配置与兼容范围

桌面在“设置与模型”配置，CLI 通过 `/model` 配置。两者共用
`~/.codeclaw/config.json`。`CODECLAW_CONFIG_DIR` 可以为开发或测试指定独立目录。
配置文件中的 Key 以本地明文保存；界面读取配置时只收到 `hasKey` 标记。
已保存的 Key 在输入框中显示遮罩和“已保存”提示，留空保存会保留原值。

## 支持的接口

| 接口 | 当前实现 |
|---|---|
| Anthropic Messages | 流式对话、工具调用及相关消息格式 |
| OpenAI 兼容 Chat Completions | 文本流、工具定义与工具调用结果 |
| 仅提供 Responses 的接口 | 尚未实现 |
| 其他厂商原生接口 | 需要适配，或使用其兼容端点 |

模型连接可以在打开项目前保存。选择 OpenAI 协议时，推荐填写 API 根地址，例如 `http://localhost:8000/v1`；
也接受完整的 `/chat/completions` 地址，保存时会自动转换为根地址。填写服务实际提供的模型 ID。
如果本地服务不校验 Key，可填写占位值；SDK 初始化仍可能要求非空 Key。

## 环境变量

```powershell
$env:CODECLAW_MODEL_PROVIDER = "openai"
$env:OPENAI_BASE_URL = "http://localhost:8000/v1"
$env:OPENAI_MODEL = "your-served-model-id"
$env:OPENAI_API_KEY = "local-placeholder"
```

Anthropic 对应 `ANTHROPIC_MODEL`、`ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`。
CLI 启动时，已设置的环境变量优先于从配置文件注入的值。桌面会话使用设置页保存的连接配置，
恢复历史时也使用当前连接配置，避免旧历史里的地址覆盖新设置。

## 直连与代理

模型请求默认直连，不会自动读取 Windows 系统代理。设置页可改为“系统代理 / 环境代理”，
此时会使用系统代理以及 `HTTP_PROXY`、`HTTPS_PROXY`、`NO_PROXY`。
CLI 可用 `CODECLAW_NETWORK_MODE=system` 启用代理，或 `direct` 直连。
主代理、子代理和上下文压缩共用此规则；证书校验始终开启。

## 兼容边界

能返回聊天文本，不等于能完整执行编程任务。需要分别验证文本流、工具调用、工具结果回传、
Skill 指令和子代理执行。上下文大小、输出上限、推理字段和图片能力目前没有完整的逐模型适配。
OpenAI 兼容路径目前主要处理文本和工具；不保证所有推理或多模态扩展。

自定义子代理可以选择模型 ID，连接协议和服务地址沿用父引擎，不能据此任意跨供应商切换。
本地模型的联网搜索由 CodeClaw 的独立工具或 MCP 完成，仍要求推理接口正确输出工具调用。
