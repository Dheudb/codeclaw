import { useState } from "react";
import { X, KeyRound, Check, MonitorCog } from "lucide-react";
import type { EngineEvent, ModelConfig } from "./types";
import Markdown from "./Markdown";

export function SettingsDialog({
  config,
  python,
  close,
  save,
  choosePython,
}: {
  config?: ModelConfig;
  python: string;
  close: () => void;
  save: (value: ModelConfig & { apiKey: string }) => Promise<void>;
  choosePython: () => Promise<void>;
}) {
  const [form, setForm] = useState({
    provider: config?.provider || "openai",
    model: config?.model || "",
    baseUrl: config?.baseUrl || "",
    hasKey: !!config?.hasKey,
    apiKey: "",
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  async function submit() {
    setSaving(true);
    setError("");
    try {
      await save(form);
      close();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }
  return (
    <div className="overlay" onClick={close}>
      <section
        role="dialog"
        aria-modal="true"
        aria-label="工作台设置"
        className="dialog"
        onClick={(e) => e.stopPropagation()}
      >
        <header>
          <div>
            <span className="eyebrow">WORKBENCH SETTINGS</span>
            <h2>配置你的工作台</h2>
          </div>
          <button className="icon-button" aria-label="关闭设置" onClick={close}>
            <X size={18} />
          </button>
        </header>
        <div className="form-section">
          <h3>
            <KeyRound size={16} /> 模型连接
          </h3>
          {!config && (
            <p className="notice">先打开项目，再为编程代理配置模型。</p>
          )}
          <label>
            API 协议
            <select
              value={form.provider}
              onChange={(e) =>
                setForm({
                  ...form,
                  provider: e.target.value,
                  model: "",
                  baseUrl: "",
                  hasKey: false,
                })
              }
            >
              <option value="openai">OpenAI / 兼容 API / 本地模型</option>
              <option value="anthropic">Anthropic</option>
            </select>
          </label>
          <label>
            模型 ID
            <input
              placeholder="输入供应商提供的完整模型 ID"
              value={form.model}
              onChange={(e) => setForm({ ...form, model: e.target.value })}
            />
          </label>
          <label>
            API 地址
            <input
              placeholder="留空使用协议默认地址"
              value={form.baseUrl}
              onChange={(e) => setForm({ ...form, baseUrl: e.target.value })}
            />
          </label>
          <label>
            API Key{" "}
            <span className="muted">
              {form.hasKey
                ? "已配置，留空保留"
                : "本地无鉴权服务可填 not-needed"}
            </span>
            <input
              type="password"
              autoComplete="off"
              value={form.apiKey}
              onChange={(e) => setForm({ ...form, apiKey: e.target.value })}
            />
          </label>
          <p className="fine-print">
            沿用 CodeClaw 本机配置；密钥不会返回聊天界面。
          </p>
        </div>
        <div className="form-section">
          <h3>
            <MonitorCog size={16} /> Python 运行环境
          </h3>
          <div className="python-row">
            <code>{python}</code>
            <button onClick={choosePython}>选择解释器</button>
          </div>
          <p className="fine-print">
            解释器需安装 CodeClaw 依赖，修改后对新打开的会话生效。MCP 服务需要
            Python 3.10+ 和 MCP 依赖。
          </p>
        </div>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        <footer>
          <button onClick={close}>取消</button>
          <button
            className="primary"
            disabled={!config || saving || !form.model.trim()}
            onClick={submit}
          >
            <Check size={15} />
            {saving ? "保存中…" : "保存配置"}
          </button>
        </footer>
      </section>
    </div>
  );
}

export function InteractionCard({
  interaction,
  respond,
}: {
  interaction: EngineEvent;
  respond: (id: string, answer: unknown) => Promise<void>;
}) {
  const [answers, setAnswers] = useState<Record<string, string | string[]>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function submit(answer: unknown) {
    setBusy(true);
    setError("");
    try {
      await respond(interaction.requestId, answer);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }
  return (
    <section className="interaction" aria-label="待处理请求">
      <span className="eyebrow">需要你的输入</span>
      {interaction.type === "permission" && (
        <>
          <h3>允许执行 {interaction.request.tool_name}？</h3>
          <pre>{interaction.request.summary}</pre>
          <div className="actions">
            <button disabled={busy} onClick={() => submit("deny")}>
              拒绝
            </button>
            <button disabled={busy} onClick={() => submit("always")}>
              本会话允许此工具
            </button>
            <button
              disabled={busy}
              className="primary"
              onClick={() => submit("allow")}
            >
              允许一次
            </button>
          </div>
        </>
      )}
      {interaction.type === "plan_approval" && (
        <>
          <h3>审阅执行计划</h3>
          <div className="plan-content">
            <Markdown text={interaction.content} />
          </div>
          <div className="actions">
            <button disabled={busy} onClick={() => submit("deny")}>
              继续修改
            </button>
            <button
              disabled={busy}
              className="primary"
              onClick={() => submit("allow")}
            >
              批准并执行
            </button>
          </div>
        </>
      )}
      {interaction.type === "questions" && (
        <>
          <h3>补充任务信息</h3>
          {interaction.questions.map((q: any) => (
            <div key={q.question} className="question">
              <p>{q.question}</p>
              <div className="choices">
                {q.options.map((option: any) => {
                  const selected = q.multiSelect
                    ? (answers[q.question] || []).includes(option.label)
                    : answers[q.question] === option.label;
                  return (
                    <button
                      key={option.label}
                      className={selected ? "chosen" : ""}
                      onClick={() =>
                        setAnswers((previous) => {
                          const current = previous[q.question];
                          return {
                            ...previous,
                            [q.question]: q.multiSelect
                              ? Array.isArray(current) &&
                                current.includes(option.label)
                                ? current.filter((v) => v !== option.label)
                                : [
                                    ...(Array.isArray(current) ? current : []),
                                    option.label,
                                  ]
                              : option.label,
                          };
                        })
                      }
                    >
                      {option.label}
                      <small>{option.description}</small>
                    </button>
                  );
                })}
              </div>
              <input
                placeholder="或者输入你的回答"
                onChange={(e) =>
                  setAnswers({ ...answers, [q.question]: e.target.value })
                }
              />
            </div>
          ))}
          <div className="actions">
            <button
              disabled={
                busy ||
                interaction.questions.some(
                  (q: any) => !answers[q.question]?.length,
                )
              }
              className="primary"
              onClick={() => submit(answers)}
            >
              提交回答
            </button>
          </div>
        </>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
