import { useEffect, useRef, useState } from "react";
import { X, KeyRound, Check, FolderOpen, Archive } from "lucide-react";
import type { EngineEvent, ModelConfig, SessionCatalog } from "./types";
import Markdown from "./Markdown";
import ArchivePanel from "./ArchivePanel";

export function SettingsDialog({
  config,
  close,
  save,
  defaultDirectory,
  chooseDirectory,
  saveDirectory,
  catalog,
  workspaceAction,
}: {
  config?: ModelConfig;
  close: () => void;
  save: (value: ModelConfig & { apiKey: string }) => Promise<void>;
  defaultDirectory: string;
  chooseDirectory: () => Promise<string | null>;
  saveDirectory: (value: string) => Promise<string>;
  catalog: SessionCatalog;
  workspaceAction: (method: string, params: Record<string, unknown>) => Promise<void>;
}) {
  const [tab, setTab] = useState("model");
  const dialogRef = useRef<HTMLElement>(null);
  const [directory, setDirectory] = useState(defaultDirectory);
  const [directoryBusy, setDirectoryBusy] = useState(false);
  const [directorySaved, setDirectorySaved] = useState(false);
  const [form, setForm] = useState({
    provider: config?.provider || "openai",
    model: config?.model || "",
    baseUrl: config?.baseUrl || "",
    hasKey: !!config?.hasKey,
    apiKey: "",
    networkMode: config?.networkMode || "direct",
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    dialogRef.current?.querySelector<HTMLButtonElement>("[role=tab]")?.focus();
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !event.isComposing) { event.preventDefault(); event.stopPropagation(); close(); }
    };
    window.addEventListener("keydown", escape, true);
    return () => { window.removeEventListener("keydown", escape, true); previous?.focus(); };
  }, []);
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
        className="dialog settings-dialog"
        ref={dialogRef}
        onKeyDown={(event) => {
          if (event.key !== "Tab") return;
          const buttons = Array.from(dialogRef.current?.querySelectorAll<HTMLElement>("button:not(:disabled), input:not(:disabled), select:not(:disabled)") || [])
            .filter((element) => element.tabIndex >= 0 && !!element.getClientRects().length);
          const first = buttons[0], last = buttons[buttons.length - 1];
          if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
          else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <header>
          <div>
            <span className="eyebrow">WORKBENCH SETTINGS</span>
            <h2>设置</h2>
          </div>
          <button className="icon-button" aria-label="关闭设置" onClick={close}>
            <X size={18} />
          </button>
        </header>
        <div className="settings-layout">
          <nav role="tablist" aria-label="设置分类" className="settings-nav" aria-orientation="vertical"
            onKeyDown={(event) => {
              if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
              event.preventDefault();
              const buttons = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>("[role=tab]"));
              const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
              buttons[event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : (index + (event.key === "ArrowUp" ? -1 : 1) + buttons.length) % buttons.length]?.click();
            }}>
            {[{ id: "model", label: "模型连接", icon: KeyRound }, { id: "workspace", label: "工作区", icon: FolderOpen }, { id: "archives", label: "已归档会话", icon: Archive }].map(({ id, label, icon: Icon }) =>
              <button key={id} role="tab" id={"settings-tab-" + id} aria-selected={tab === id} aria-controls={"settings-panel-" + id} tabIndex={tab === id ? 0 : -1}
                onClick={(event) => { setTab(id); setError(""); event.currentTarget.focus(); }}><Icon size={16} />{label}</button>)}
          </nav>
          <div className="settings-content">
        <div className="form-section" role="tabpanel" id="settings-panel-model" aria-labelledby="settings-tab-model" hidden={tab !== "model"}>
          <h3>
            <KeyRound size={16} /> 模型连接
          </h3>
          <p className="notice">保存模型连接后，新建会话即可使用。</p>
          <label>
            API 协议
            <select
              aria-label="API 协议"
              value={form.provider}
              onChange={(e) =>
                setForm({
                  ...form,
                  provider: e.target.value,
                  model: "",
                  baseUrl: "",
                  hasKey: false,
                  apiKey: "",
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
              aria-label="模型 ID"
              placeholder="输入供应商提供的完整模型 ID"
              value={form.model}
              onChange={(e) => setForm({ ...form, model: e.target.value })}
            />
          </label>
          <label>
            API 地址
            <input
              aria-label="API 地址"
              placeholder="API 根地址或完整的 /chat/completions 地址"
              value={form.baseUrl}
              onChange={(e) => setForm({ ...form, baseUrl: e.target.value })}
            />
          </label>
          <label>
            API Key{" "}
            <span className="muted">
              {form.hasKey
                ? "✓ 已保存；输入新 Key 可替换"
                : "本地无鉴权服务可填 not-needed"}
            </span>
            <input
              aria-label="API Key"
              type="password"
              autoComplete="off"
              placeholder={form.hasKey ? "••••••••••••••••（已保存）" : "输入 API Key"}
              value={form.apiKey}
              onChange={(e) => setForm({ ...form, apiKey: e.target.value })}
            />
          </label>
          <label>
            连接方式
            <select aria-label="连接方式" value={form.networkMode}
              onChange={(e) => setForm({ ...form, networkMode: e.target.value as "direct" | "system" })}>
              <option value="direct">直连（本地模型和兼容 API 默认）</option>
              <option value="system">系统代理 / 环境代理</option>
            </select>
          </label>
          <p className="fine-print">
            沿用 CodeClaw 本机配置；密钥不会返回聊天界面。
          </p>
        {error && tab === "model" && <p role="alert" className="error">{error}</p>}
        <footer>
          <button onClick={close}>取消</button>
          <button className="primary" disabled={saving || !form.model.trim()} onClick={submit}><Check size={15} />{saving ? "保存中…" : "保存配置"}</button>
        </footer>
        </div>
        <div className="form-section" role="tabpanel" id="settings-panel-workspace" aria-labelledby="settings-tab-workspace" hidden={tab !== "workspace"}>
          <h3><FolderOpen size={16} /> 默认工作目录</h3>
          <p className="fine-print">按当前系统和用户自动设置。未选择项目时，新会话使用此目录；已有项目和历史不会移动。</p>
          <label>目录路径
            <input aria-label="默认工作目录" value={directory} onChange={(e) => { setDirectory(e.target.value); setDirectorySaved(false); }} />
          </label>
          <div className="actions">
            <button disabled={directoryBusy} onClick={async () => {
              setDirectoryBusy(true); setError("");
              try { const value = await chooseDirectory(); if (value) { setDirectory(value); setDirectorySaved(false); } }
              catch (e) { setError(String(e)); } finally { setDirectoryBusy(false); }
            }}>选择文件夹</button>
            <button disabled={directoryBusy || !directory.trim()} onClick={async () => {
              setDirectoryBusy(true); setError(""); setDirectorySaved(false);
              try { setDirectory(await saveDirectory(directory)); setDirectorySaved(true); }
              catch (e) { setError(String(e)); } finally { setDirectoryBusy(false); }
            }}>{directoryBusy ? "处理中…" : "保存默认目录"}</button>
            {directorySaved && <span role="status">默认目录已保存</span>}
          </div>
          {error && tab === "workspace" && <p role="alert" className="error">{error}</p>}
        </div>
        {tab === "archives" && <div role="tabpanel" id="settings-panel-archives" aria-labelledby="settings-tab-archives"><ArchivePanel catalog={catalog} action={workspaceAction} /></div>}
          </div>
        </div>
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
