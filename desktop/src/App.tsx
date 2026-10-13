import { useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  ArrowUpRight,
  Bot,
  Check,
  ChevronRight,
  Code2,
  Folder,
  FolderOpen,
  GitBranch,
  LoaderCircle,
  MessageSquare,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  Puzzle,
  Settings2,
  ShieldCheck,
  Square,
  X,
  Store,
} from "lucide-react";
import type { InitialSession, SessionView, ModelConfig } from "./types";
import { applyEvent, beginTurn, createSession } from "./state";
import CapabilityPanel from "./CapabilityPanel";
import { InteractionCard, SettingsDialog } from "./Dialogs";
import Markdown from "./Markdown";
import McpDialog from "./McpDialog";

const basename = (value: string) =>
  value.split(/[\\/]/).filter(Boolean).pop() || value;

export default function App() {
  const [sessions, setSessions] = useState<SessionView[]>([]);
  const [activeId, setActiveId] = useState("");
  const [recent, setRecent] = useState<string[]>([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [globalModel, setGlobalModel] = useState<ModelConfig>();
  const [mcpOpen, setMcpOpen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(true);
  const [panelTab, setPanelTab] = useState("skills");
  const [opening, setOpening] = useState(false);
  const [notice, setNotice] = useState("");
  const bottom = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const active = sessions.find((item) => item.id === activeId);

  useEffect(() => {
    if (!window.codeclaw) {
      setNotice("请通过 CodeClaw 桌面程序打开工作台。");
      return;
    }
    window.codeclaw
      .invoke("app.initialize")
      .then((value) => {
        setRecent(value.projects || []);
        setGlobalModel(value.config);
      })
      .catch((error) => setNotice(String(error)));
    return window.codeclaw.onEvent((event) =>
      setSessions((previous) =>
        previous.map((session) => applyEvent(session, event)),
      ),
    );
  }, []);
  useEffect(() => {
    bottom.current?.scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
    });
  }, [active?.messages, active?.interactions]);
  useEffect(() => {
    input.current?.focus();
  }, [activeId]);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "n") {
        event.preventDefault();
        if (!opening) active ? openSession() : openProject();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  function update(patch: Partial<SessionView>, id = activeId) {
    setSessions((previous) =>
      previous.map((session) =>
        session.id === id ? { ...session, ...patch } : session,
      ),
    );
  }
  function onError(error: unknown) {
    setNotice(
      String(error).replace(
        /^Error: Error invoking remote method '[^']+': Error: /,
        "",
      ),
    );
  }
  async function invoke(method: string, params: Record<string, unknown> = {}) {
    return window.codeclaw.invoke(method, { session: activeId, ...params });
  }
  function accept(initial: InitialSession) {
    setSessions((previous) => [...previous, createSession(initial)]);
    setActiveId(initial.id);
    setRecent((previous) =>
      [
        initial.project,
        ...previous.filter(
          (item) => item.toLowerCase() !== initial.project.toLowerCase(),
        ),
      ].slice(0, 12),
    );
  }
  async function openProject(project?: string) {
    setOpening(true);
    setNotice("");
    try {
      const initial = await window.codeclaw.invoke("project.open", {
        path: project,
      });
      if (initial) accept(initial);
    } catch (error) {
      onError(error);
    } finally {
      setOpening(false);
    }
  }
  async function openSession(resumeId?: string) {
    if (!active) return;
    if (resumeId) {
      const existing = sessions.find(
        (item) =>
          item.project === active.project && item.persistedId === resumeId,
      );
      if (existing) {
        setActiveId(existing.id);
        return;
      }
    }
    setOpening(true);
    try {
      const initial = await invoke("session.open", { resumeId });
      accept(initial);
      if (resumeId) update({ persistedId: resumeId }, initial.id);
    } catch (error) {
      onError(error);
    } finally {
      setOpening(false);
    }
  }
  async function closeSession(id: string) {
    const session = sessions.find((item) => item.id === id);
    if (
      session?.busy ||
      session?.workers.some((worker) => worker.status === "running")
    ) {
      onError("先停止当前任务，再关闭会话。");
      return;
    }
    try {
      await window.codeclaw.invoke("session.close", { session: id });
      setSessions((previous) => previous.filter((item) => item.id !== id));
      if (activeId === id)
        setActiveId(sessions.find((item) => item.id !== id)?.id || "");
    } catch (error) {
      onError(error);
    }
  }
  async function refresh() {
    if (active) update({ capabilities: await invoke("capabilities.list") });
  }
  async function send() {
    if (!active || active.busy || !active.draft.trim()) return;
    const id = active.id,
      text = active.draft.trim(),
      selected = active.selected,
      mode = active.mode;
    setSessions((previous) =>
      previous.map((session) =>
        session.id === id ? beginTurn(session, text) : session,
      ),
    );
    try {
      await window.codeclaw.invoke("prompt", {
        session: id,
        text,
        skills: selected,
        mode,
      });
    } catch (error) {
      setSessions((previous) =>
        previous.map((session) =>
          applyEvent(session, {
            session: id,
            type: "turn_error",
            message: String(error),
          }),
        ),
      );
    }
  }
  async function saveModel(value: ModelConfig & { apiKey: string }) {
    const config = await invoke("config.save", value);
    setGlobalModel(config);
    if (active) update({ config });
  }
  async function openSettings() {
    try {
      setGlobalModel(await window.codeclaw.invoke("config.get"));
      setSettingsOpen(true);
    } catch (error) {
      onError(error);
    }
  }
  const title =
    active?.messages
      .find((message) => message.role === "user")
      ?.text.split("\n")[0] || "新会话";
  const enabledSkills =
    active?.capabilities.skills.filter(
      (skill) => skill.valid && skill.enabled && !skill.shadowed,
    ) || [];
  const runningWorkers =
    active?.workers.filter((worker) => worker.status === "running").length || 0;

  return (
    <div className={"workbench " + (!panelOpen ? "panel-hidden" : "")}>
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">
            <Code2 size={23} strokeWidth={2.4} />
          </span>
          <strong>
            CodeClaw<span>DESKTOP</span>
          </strong>
          <span className="preview-badge">预览</span>
        </div>
        <button
          className="new-task"
          disabled={opening}
          onClick={() => (active ? openSession() : openProject())}
        >
          <Plus size={17} />
          新建会话<span>Ctrl N</span>
        </button>
        <button
          className="sidebar-action"
          disabled={opening}
          onClick={() => openProject()}
        >
          <FolderOpen size={16} />
          打开项目{opening && <LoaderCircle size={14} className="spin" />}
        </button>
        <div className="sidebar-scroll">
          <div className="sidebar-section">
            <span>工作区</span>
            <span className="count">
              {new Set(sessions.map((session) => session.project)).size}
            </span>
          </div>
          {!sessions.length && (
            <p className="sidebar-hint">打开本地目录，开始你的第一个任务。</p>
          )}
          {sessions.map((session) => (
            <div
              className={
                "session-item " + (session.id === activeId ? "active" : "")
              }
              key={session.id}
            >
              <button
                className="session-select"
                onClick={() => setActiveId(session.id)}
              >
                <span className="session-symbol">
                  {session.busy ? (
                    <LoaderCircle size={15} className="spin" />
                  ) : (
                    <MessageSquare size={15} />
                  )}
                </span>
                <span>
                  <strong>
                    {session.messages
                      .find((message) => message.role === "user")
                      ?.text.slice(0, 25) || "新会话"}
                  </strong>
                  <small>{basename(session.project)}</small>
                </span>
              </button>
              <button
                className="close-session"
                aria-label="关闭会话"
                onClick={() => closeSession(session.id)}
              >
                <X size={12} />
              </button>
            </div>
          ))}
          {active && (
            <>
              <div className="sidebar-section history-label">
                <span>项目历史</span>
                <span className="count">{active.sessions.length}</span>
              </div>
              {!active.sessions.length && (
                <p className="sidebar-hint">
                  完成任务后，会话自动保存在项目中。
                </p>
              )}
              {active.sessions.slice(0, 12).map((session) => (
                <button
                  className="history-item"
                  key={session.id}
                  onClick={() => openSession(session.id)}
                >
                  <MessageSquare size={13} />
                  <span>{session.title}</span>
                </button>
              ))}
            </>
          )}
          {recent.length > 0 && (
            <>
              <div className="sidebar-section history-label">
                <span>最近项目</span>
              </div>
              {recent.map((project) => (
                <button
                  className="history-item project-item"
                  key={project}
                  title={project}
                  disabled={opening}
                  onClick={() => openProject(project)}
                >
                  <Folder size={14} />
                  <span>{basename(project)}</span>
                  <ArrowUpRight size={12} />
                </button>
              ))}
            </>
          )}
        </div>
        <div className="sidebar-bottom">
          <button
            className="sidebar-action"
            disabled={!active}
            title={
              !active ? "先打开项目，再管理 MCP 服务" : "为当前项目管理外部工具"
            }
            onClick={() => setMcpOpen(true)}
          >
            <Store size={16} />
            MCP 市场
          </button>
          <div className="local-status">
            <span />
            本地工作区<span className="version">v0.1.0</span>
          </div>
          <button className="sidebar-action" onClick={openSettings}>
            <Settings2 size={16} />
            设置与模型
          </button>
        </div>
      </aside>
      <main className="main-panel">
        <header className="topbar">
          <div className="breadcrumbs">
            <Folder size={15} />
            <button
              title={active?.project}
              onClick={() => active && invoke("project.reveal").catch(onError)}
            >
              {active ? basename(active.project) : "工作台"}
            </button>
            <ChevronRight size={13} />
            <span>{active ? title.slice(0, 35) : "开始"}</span>
          </div>
          <div className="topbar-actions">
            <span className="local-tag">
              <span />
              {active?.busy ? "执行中" : "本地"}
            </span>
            <button
              className="icon-button"
              aria-label={panelOpen ? "收起能力面板" : "展开能力面板"}
              onClick={() => setPanelOpen(!panelOpen)}
            >
              {panelOpen ? (
                <PanelRightClose size={18} />
              ) : (
                <PanelRightOpen size={18} />
              )}
            </button>
          </div>
        </header>
        {notice && (
          <div className="global-notice" role="alert">
            <span>{notice}</span>
            <button aria-label="关闭提示" onClick={() => setNotice("")}>
              <X size={15} />
            </button>
          </div>
        )}
        <div className="conversation">
          {!active || !active.messages.length ? (
            <div className="welcome">
              <div className="welcome-symbol">
                <Code2 size={32} />
              </div>
              <div className="eyebrow">YOUR LOCAL AGENT WORKSPACE</div>
              <h1>
                {active ? "今天想构建什么？" : "把想法变成可运行的代码。"}
              </h1>
              <p>
                {active
                  ? "描述你的任务，连接模型，让代理在这个项目里开始工作。"
                  : "连接你的模型、挂载本地项目，用 Skills 和子代理构建自己的工作流。"}
              </p>
              {!active ? (
                <button
                  className="primary open-project"
                  disabled={opening}
                  onClick={() => openProject()}
                >
                  <FolderOpen size={17} />
                  {opening ? "正在打开…" : "打开本地项目"}
                  <ArrowUpRight size={15} />
                </button>
              ) : (
                <>
                  <div className="project-context">
                    <GitBranch size={14} />
                    <span title={active.project}>{active.project}</span>
                  </div>
                  <div className="starter-grid">
                    {[
                      {
                        icon: Code2,
                        title: "理解这个项目",
                        text: "分析代码结构、技术栈和启动方式。",
                        prompt:
                          "请分析这个项目的代码结构、技术栈和启动方式，说明核心模块的职责。",
                      },
                      {
                        icon: Puzzle,
                        title: "使用你的 Skills",
                        text: "挂载技能，复用自己的工作流。",
                        action: () => {
                          setPanelOpen(true);
                          setPanelTab("skills");
                        },
                      },
                      {
                        icon: Bot,
                        title: "把任务交给子代理",
                        text: "独立探索、规划与验证。",
                        prompt:
                          "请使用 explore 子代理分析项目结构，并总结最值得关注的模块。",
                      },
                    ].map(({ icon: Icon, title, text, prompt, action }) => (
                      <button
                        key={title}
                        onClick={() =>
                          action ? action() : update({ draft: prompt })
                        }
                      >
                        <Icon size={19} />
                        <strong>{title}</strong>
                        <span>{text}</span>
                        <ArrowUpRight size={14} />
                      </button>
                    ))}
                  </div>
                </>
              )}
              {!active && (
                <div className="welcome-features">
                  <span>
                    <ShieldCheck size={14} />
                    项目保存在本地
                  </span>
                  <span>
                    <Puzzle size={14} />
                    按需挂载 Skills
                  </span>
                  <span>
                    <Bot size={14} />
                    独立子代理
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div className="transcript">
              {active.messages.map((message, index) => (
                <article key={index} className={"message " + message.role}>
                  <div className="message-avatar">
                    {message.role === "assistant" ? <Code2 size={16} /> : "你"}
                  </div>
                  <div className="message-body">
                    <div className="message-author">
                      {message.role === "assistant" ? "CodeClaw" : "你"}
                      {message.role === "assistant" &&
                        active.busy &&
                        index === active.streamIndex && (
                          <span className="working-label">
                            <span />
                            处理中
                          </span>
                        )}
                    </div>
                    <div className="markdown">
                      {message.text ? (
                        <Markdown text={message.text} />
                      ) : active.busy ? (
                        <div className="thinking">
                          <span />
                          <span />
                          <span />
                        </div>
                      ) : (
                        <span className="muted">此次任务未返回文本。</span>
                      )}
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
          {active?.interactions.map((interaction) => (
            <InteractionCard
              key={interaction.requestId}
              interaction={interaction}
              respond={(requestId, answer) =>
                invoke("respond", { requestId, answer })
              }
            />
          ))}
          <div ref={bottom} />
        </div>
        <div className="composer-area">
          {active?.error && (
            <div className="session-error" role="alert">
              {active.error}
              <button
                aria-label="清除会话提示"
                onClick={() => update({ error: "" })}
              >
                <X size={13} />
              </button>
            </div>
          )}
          {active && runningWorkers > 0 && (
            <button
              className="worker-strip"
              onClick={() => {
                setPanelOpen(true);
                setPanelTab("agents");
              }}
            >
              <LoaderCircle size={13} className="spin" />
              {runningWorkers} 个子代理正在执行任务
              <ChevronRight size={13} />
            </button>
          )}
          <div className={"composer " + (!active ? "disabled" : "")}>
            {active && active.selected.length > 0 && (
              <div className="selected-skills">
                {active.selected.map((name) => (
                  <button
                    key={name}
                    onClick={() =>
                      update({
                        selected: active.selected.filter(
                          (item) => item !== name,
                        ),
                      })
                    }
                  >
                    <Puzzle size={12} />
                    {name}
                    <X size={11} />
                  </button>
                ))}
              </div>
            )}
            <textarea
              ref={input}
              aria-label="任务输入"
              placeholder={
                active
                  ? "描述你的任务，或选择一个 Skill 开始…"
                  : "打开项目后开始任务…"
              }
              disabled={!active || active.busy}
              value={active?.draft || ""}
              onChange={(e) => update({ draft: e.target.value })}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            <div className="composer-toolbar">
              <div className="composer-options">
                <select
                  aria-label="执行模式"
                  disabled={!active || active.busy}
                  value={active?.mode || "normal"}
                  onChange={(e) => update({ mode: e.target.value })}
                >
                  <option value="normal">Agent</option>
                  <option value="plan">Plan</option>
                </select>
                <span className="toolbar-divider" />
                <button className="model-button" onClick={openSettings}>
                  <span
                    className={
                      active?.config.model ? "model-dot connected" : "model-dot"
                    }
                  />
                  <span>{active?.config.model || "连接模型"}</span>
                  <ChevronRight size={12} />
                </button>
              </div>
              <div className="send-options">
                {active?.loaded.length ? (
                  <span
                    className="loaded-indicator"
                    title={active.loaded.join(", ")}
                  >
                    <Check size={12} />
                    {active.loaded.length} Skills
                  </span>
                ) : null}
                {active?.busy ? (
                  <button
                    className="send-button stop"
                    aria-label="停止任务"
                    onClick={() => invoke("stop").catch(onError)}
                  >
                    <Square size={15} fill="currentColor" />
                  </button>
                ) : (
                  <button
                    className="send-button"
                    aria-label="发送任务"
                    disabled={!active?.draft.trim()}
                    onClick={send}
                  >
                    <ArrowUp size={19} />
                  </button>
                )}
              </div>
            </div>
          </div>
          <div className="composer-caption">
            <span>
              <ShieldCheck size={12} />
              工具执行遵循会话权限
            </span>
            <span>
              {enabledSkills.length} Skills 可用 ·{" "}
              {active?.capabilities.agents.filter(
                (agent) => agent.valid && !agent.shadowed,
              ).length || 0}{" "}
              代理角色
              <span className="shortcut-hint">
                Enter 发送 · Shift Enter 换行
              </span>
            </span>
          </div>
        </div>
      </main>
      {panelOpen && (
        <CapabilityPanel
          key={activeId || "empty"}
          session={active}
          tab={panelTab}
          setTab={setPanelTab}
          invoke={invoke}
          refresh={refresh}
          update={update}
          onError={onError}
        />
      )}
      {settingsOpen && (
        <SettingsDialog
          key={activeId}
          config={active?.config || globalModel}
          close={() => setSettingsOpen(false)}
          save={saveModel}
        />
      )}
      {mcpOpen && active && (
        <McpDialog
          key={active.id}
          sessionId={active.id}
          busy={
            active.busy ||
            active.workers.some((worker) => worker.status === "running")
          }
          invoke={invoke}
          close={() => setMcpOpen(false)}
        />
      )}
    </div>
  );
}
