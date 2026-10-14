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
import type { SessionView, ModelConfig, SessionCatalog, PermissionMode, FileChange } from "./types";
import { applyEvent, beginTurn, createSession } from "./state";
import CapabilityPanel from "./CapabilityPanel";
import { InteractionCard, SettingsDialog } from "./Dialogs";
import Markdown from "./Markdown";
import McpDialog from "./McpDialog";
import SessionSidebar from "./SessionSidebar";
import { CatalogPreferences } from "./catalog-preferences";
import PanelResizer from "./PanelResizer";
import TurnProcess from "./TurnProcess";
import SkillInput from "./SkillInput";
import FileChanges, { relativeFile } from "./FileChanges";
import FilePreview from "./FilePreview";
import { clampPanelWidth, DEFAULT_PANEL_WIDTH } from "./panel-width";

const basename = (value: string) =>
  value.split(/[\\/]/).filter(Boolean).pop() || value;

export default function App() {
  const [sessions, setSessions] = useState<SessionView[]>([]);
  const [activeId, setActiveId] = useState("");
  const [catalog, setCatalog] = useState<SessionCatalog>({ projects: [], view: { sort: "recent" } });
  const preferences = useRef(new CatalogPreferences(catalog));
  const [defaultDirectory, setDefaultDirectory] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [globalModel, setGlobalModel] = useState<ModelConfig>();
  const [modelChanging, setModelChanging] = useState(false);
  const [permissionChanging, setPermissionChanging] = useState(false);
  const [mcpOpen, setMcpOpen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(true);
  const [panelTab, setPanelTab] = useState("skills");
  const [panelWidth, setPanelWidth] = useState(DEFAULT_PANEL_WIDTH);
  const [viewport, setViewport] = useState(window.innerWidth);
  const [opening, setOpening] = useState(true);
  const [notice, setNotice] = useState("");
  const [buildId, setBuildId] = useState("");
  const [filePreview, setFilePreview] = useState<{ path: string; change?: FileChange; initialView: "diff" | "content" }>();
  const bottom = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const navigation = useRef(0);
  const active = sessions.find((item) => item.id === activeId);
  function receiveCatalog(value: SessionCatalog) {
    setCatalog(preferences.current.receive(value));
  }

  useEffect(() => {
    if (!window.codeclaw) {
      setNotice("请通过 CodeClaw 桌面程序打开工作台。");
      return;
    }
    window.codeclaw
      .invoke("app.initialize")
      .then((value) => {
        setGlobalModel(value.config);
        setBuildId(value.buildId || "");
        setDefaultDirectory(value.workspaceSettings.defaultDirectory);
        if (Number.isFinite(value.layout?.inspectorWidth)) setPanelWidth(value.layout.inspectorWidth);
        receiveCatalog(value.catalog);
        setSessions((value.restored || []).map(createSession));
        setActiveId(value.activeId || "");
        if (value.restoreErrors?.length) setNotice(value.restoreErrors.join("\n"));
      })
      .catch((error) => setNotice(String(error)))
      .finally(() => setOpening(false));
    return window.codeclaw.onEvent((event) => {
      if (event.type === "catalog_changed") {
        receiveCatalog(event.catalog);
        return;
      }
      if (event.type === "session_unloaded") {
        setSessions((previous) => previous.filter((session) => session.id !== event.session));
        return;
      }
      setSessions((previous) =>
        previous.map((session) => applyEvent(session, event)),
      );
    });
  }, []);
  useEffect(() => {
    const resize = () => setViewport(window.innerWidth);
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);
  useEffect(() => {
    if (!active || opening) return;
    const id = active.id, draft = active.draft;
    const timer = setTimeout(() => window.codeclaw.invoke("session.update", { id, patch: { draft } }).catch(onError), 500);
    return () => clearTimeout(timer);
  }, [activeId, active?.draft, opening]);
  useEffect(() => {
    if (activeId) window.codeclaw.invoke("session.activate", { session: activeId }).catch(onError);
  }, [activeId]);
  useEffect(() => {
    bottom.current?.scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
    });
  }, [active?.messages, active?.interactions]);
  useEffect(() => {
    input.current?.focus();
    setFilePreview(undefined);
  }, [activeId, active?.project]);
  function openFile(path: string, change?: FileChange) {
    if (!active) return;
    const initialView = change ? "diff" : "content";
    change ||= active.fileChanges?.find((file) => file.path === path || relativeFile(file.path, active.project) === path.replace(/^\.[\\/]/, ""));
    setFilePreview({ path, change, initialView });
    setPanelOpen(true);
    setPanelWidth((width) => Math.max(width, 700));
  }
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "n") {
        event.preventDefault();
        if (!opening && !settingsOpen && !mcpOpen) openSession();
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
  async function workspaceAction(method: string, params: Record<string, unknown>) {
    const preferenceWrite = method === "catalog.view" ? preferences.current.begin(params as Partial<SessionCatalog["view"]>) : undefined;
    if (preferenceWrite) setCatalog(preferenceWrite.catalog);
    const changesSelection = !method.startsWith("catalog.") && !method.startsWith("archive.") &&
      (method !== "session.update" || !!(params.patch as { archived?: boolean } | undefined)?.archived) &&
      (method !== "project.update" || !!(params.patch as { archived?: boolean } | undefined)?.archived);
    const intent = changesSelection ? ++navigation.current : navigation.current;
    if (changesSelection) setOpening(true);
    setNotice("");
    try {
      if (active?.draft && ["project.open", "session.new", "session.open", "session.select", "session.move"].includes(method))
        await window.codeclaw.invoke("session.update", { id: active.id, patch: { draft: active.draft } });
      const result = await window.codeclaw.invoke(method, params);
      if (!result) return;
      if (preferenceWrite) setCatalog(preferences.current.finish(preferenceWrite.id, result));
      else if (result.catalog) receiveCatalog(result.catalog);
      else if (method.startsWith("catalog.")) receiveCatalog(result);
      if (result.id) {
        setSessions((previous) => [...previous.filter((session) => session.id !== result.id && session.id !== result.movedFrom), createSession(result)]);
      }
      if (changesSelection && intent === navigation.current) {
        if ("activeId" in result) setActiveId(result.activeId);
        else if (result.existingId || result.id) setActiveId(result.existingId || result.id);
      }
    } catch (error) {
      if (preferenceWrite) {
        setCatalog(preferences.current.fail(preferenceWrite.id));
        window.codeclaw.invoke("catalog.refresh").then(receiveCatalog).catch(() => {});
      }
      throw error;
    } finally {
      if (changesSelection && intent === navigation.current) setOpening(false);
    }
  }
  async function openProject(project?: string) {
    try {
      await workspaceAction("project.open", { path: project });
    } catch (error) {
      onError(error);
    }
  }
  async function openSession(resumeId?: string) {
    try {
      if (resumeId && active) await workspaceAction("session.open", { session: activeId, resumeId });
      else await workspaceAction("session.new", {});
    } catch (error) {
      onError(error);
    }
  }
  async function refresh() {
    if (active) update({ capabilities: await invoke("capabilities.list") });
  }
  async function send() {
    if (opening || modelChanging || permissionChanging || !active || active.busy || !active.draft.trim()) return;
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
    setSessions((previous) => previous.map((session) => session.busy ? session : { ...session, config }));
  }
  async function selectModel(value: string) {
    if (modelChanging) return;
    setModelChanging(true);
    try {
      const [connectionId, model] = JSON.parse(value);
      const config = await invoke("config.select", { connectionId, model });
      setGlobalModel(config);
      setSessions((previous) => previous.map((session) => session.busy ? session : { ...session, config }));
    } catch (error) { onError(error); }
    finally { setModelChanging(false); }
  }
  async function setPermissionMode(mode: PermissionMode) {
    if (!active || permissionChanging) return;
    const id = active.id;
    setPermissionChanging(true);
    try {
      const result = await window.codeclaw.invoke("permissions.set", { session: id, mode });
      update({ permissionMode: result.permissionMode }, id);
    } catch (error) { onError(error); }
    finally { setPermissionChanging(false); }
  }
  async function openSettings() {
    try {
      const [config, workspace] = await Promise.all([window.codeclaw.invoke("config.get"), window.codeclaw.invoke("workspace.get")]);
      setGlobalModel(config);
      setDefaultDirectory(workspace.defaultDirectory);
      setSettingsOpen(true);
    } catch (error) {
      onError(error);
    }
  }
  const title = catalog.projects.flatMap((project) => project.sessions).find((row) => row.id === activeId)?.title ||
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
    <div className={"workbench " + (!panelOpen ? "panel-hidden" : "")} style={{ "--inspector-width": clampPanelWidth(panelWidth, viewport) + "px" } as React.CSSProperties}>
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
          onClick={() => openSession()}
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
        <SessionSidebar catalog={catalog} activeId={activeId} live={sessions} opening={opening}
          action={workspaceAction} openProject={openProject} onError={onError} />
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
            本地工作区<span className="version" title={buildId ? "构建 " + buildId : "开发版本"}>v0.1.0{buildId ? " · " + buildId.slice(-4) : ""}</span>
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
              {active?.scope === "standalone" ? "独立会话" : active ? catalog.projects.find((project) => project.path.toLowerCase() === active.project.toLowerCase())?.name || basename(active.project) : "工作台"}
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
                {active?.scope === "standalone" ? "开始一个新会话。" : active ? "今天想构建什么？" : "把想法变成可运行的代码。"}
              </h1>
              <p>
                {active?.scope === "standalone" ? "直接开始对话；文件和命令任务使用此会话的专属工作目录。" : active
                  ? "描述你的任务，连接模型，让代理在这个项目里开始工作。"
                  : "连接模型即可开始；会话自动使用默认工作目录，也可以打开自己的项目。"}
              </p>
              {!active ? (
                <button
                  className="primary open-project"
                  disabled={opening}
                  onClick={() => openSession()}
                >
                  <FolderOpen size={17} />
                  {opening ? "正在打开…" : "开始新会话"}
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
                        title: active.scope === "standalone" ? "梳理一个任务" : "理解这个项目",
                        text: active.scope === "standalone" ? "明确目标、拆解步骤，开始推进。" : "分析代码结构、技术栈和启动方式。",
                        prompt:
                          active.scope === "standalone" ? "请帮我梳理要完成的任务，先确认我的目标和需求。" : "请分析这个项目的代码结构、技术栈和启动方式，说明核心模块的职责。",
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
                    {message.trace && <TurnProcess trace={message.trace} />}
                    <div className="markdown">
                      {message.text ? (
                        <Markdown text={message.text} onOpenFile={openFile} />
                      ) : active.busy && index === active.streamIndex ? (
                        <div className="thinking">
                          <span />
                          <span />
                          <span />
                        </div>
                      ) : !message.trace ? (
                        <span className="muted">此次任务未返回文本。</span>
                      ) : null}
                    </div>
                    {!!message.trace?.files?.length && <FileChanges files={message.trace.files} project={active.project} onOpen={openFile} />}
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
            <SkillInput key={activeId} sessionId={activeId} draft={active?.draft || ""}
              selected={active?.selected || []} capabilities={active?.capabilities}
              disabled={opening || !active || active.busy} inputRef={input} update={update} send={send} />
            <div className="composer-toolbar">
              <div className="composer-options">
                <select
                  aria-label="执行模式"
                  disabled={opening || !active || active.busy}
                  value={active?.mode || "normal"}
                  onChange={(e) => update({ mode: e.target.value })}
                >
                  <option value="normal">Agent</option>
                  <option value="plan">Plan</option>
                </select>
                <select className={"permission-picker " + (active?.permissionMode === "full" ? "full-access" : "")} aria-label="访问权限"
                  title={active?.permissionMode === "full" ? "本会话的主代理和子代理执行工具时无需批准" : "文件修改、命令和需要审批的外部工具会请求批准"}
                  disabled={opening || !active || permissionChanging} value={active?.permissionMode || "ask"}
                  onChange={(event) => void setPermissionMode(event.target.value as PermissionMode)}>
                  <option value="ask">请求批准</option><option value="full">完全访问</option>
                </select>
                <span className="toolbar-divider" />
                {globalModel?.connections?.some((connection) => connection.models.length) ? <select
                  className="model-picker" aria-label="选择模型" title="切换 API 与模型"
                  disabled={opening || modelChanging || active?.busy || !!runningWorkers}
                  value={JSON.stringify([globalModel.connectionId, globalModel.model])}
                  onChange={(event) => void selectModel(event.target.value)}>
                  {globalModel.connections.filter((connection) => connection.models.length).map((connection) =>
                    <optgroup key={connection.connectionId} label={connection.connectionName}>
                      {connection.models.map((model) => <option key={model} value={JSON.stringify([connection.connectionId, model])}>{model}</option>)}
                    </optgroup>)}
                </select> : <button className="model-button" onClick={openSettings}>连接模型<ChevronRight size={12} /></button>}
                <button className="icon-button" aria-label="管理 API 与模型" title="管理 API 与模型" onClick={openSettings}><Settings2 size={13} /></button>
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
                    disabled={opening || modelChanging || permissionChanging || !active?.draft.trim()}
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
              {active?.permissionMode === "full" ? "完全访问：工具执行无需批准" : "请求批准：需要审批的工具会先确认"}
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
        <div className="inspector-shell">
        <PanelResizer width={clampPanelWidth(panelWidth, viewport)} viewport={viewport} change={setPanelWidth}
          save={(inspectorWidth) => { void window.codeclaw.invoke("layout.save", { inspectorWidth }).catch(onError); }} />
        {filePreview && active ? <FilePreview sessionId={activeId} project={active.project}
          version={active.fileChanges?.find((file) => file.path === (filePreview.change?.path || filePreview.path) || relativeFile(file.path, active.project) === filePreview.path)?.updatedAt}
          {...filePreview} changes={active.fileChanges || []} close={() => setFilePreview(undefined)} onOpen={openFile} /> : <CapabilityPanel
            onOpenFile={openFile}
          key={activeId || "empty"}
          session={active}
          tab={panelTab}
          setTab={setPanelTab}
          invoke={invoke}
          refresh={refresh}
          update={update}
          onError={onError}
        />}
        </div>
      )}
      {settingsOpen && (
        <SettingsDialog
          key={activeId}
          config={globalModel || active?.config}
          catalog={catalog}
          workspaceAction={workspaceAction}
          defaultDirectory={defaultDirectory}
          chooseDirectory={() => window.codeclaw.invoke("workspace.choose")}
          saveDirectory={async (directory) => {
            const value = await window.codeclaw.invoke("workspace.save", { defaultDirectory: directory });
            setDefaultDirectory(value.defaultDirectory);
            return value.defaultDirectory;
          }}
          close={() => setSettingsOpen(false)}
          save={saveModel}
          discoverModels={(value) => window.codeclaw.invoke("config.models", value)}
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
