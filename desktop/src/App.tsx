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
  Pencil,
  Play,
  Paperclip,
} from "lucide-react";
import type { SessionView, ModelConfig, SessionCatalog, PermissionMode, FileChange, Attachment } from "./types";
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
import { clampPanelWidth, clampSidebarWidth, defaultSidebarWidth, DEFAULT_PANEL_WIDTH } from "./panel-width";
import useConversationScroll from "./useConversationScroll";
import Attachments, { AttachmentPreview } from "./Attachments";

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
  const [sidebarWidth, setSidebarWidth] = useState<number>();
  const [viewport, setViewport] = useState(window.innerWidth);
  const [opening, setOpening] = useState(true);
  const [notice, setNotice] = useState("");
  const [buildId, setBuildId] = useState("");
  const [importing, setImporting] = useState(false);
  const [fileDrag, setFileDrag] = useState(false);
  const [attachmentPreview, setAttachmentPreview] = useState<{ name: string; source: string }>();
  const dragDepth = useRef(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const [filePreview, setFilePreview] = useState<{ path: string; change?: FileChange; initialView: "diff" | "content" }>();
  const input = useRef<HTMLTextAreaElement>(null);
  const navigation = useRef(0);
  const active = sessions.find((item) => item.id === activeId);
  const conversationScroll = useConversationScroll(activeId, active?.messages, active?.interactions);
  const effectiveSidebarWidth = clampSidebarWidth(sidebarWidth ?? defaultSidebarWidth(viewport), viewport, panelOpen);
  const draftSessions = useRef(sessions);
  draftSessions.current = sessions;
  const lastUser = active?.messages.slice().reverse().find((message) => message.role === "user");
  const paused = !active?.busy && ["cancelled", "interrupted"].includes(active?.messages.at(-1)?.trace?.status || "");
  function receiveCatalog(value: SessionCatalog) {
    setCatalog(preferences.current.receive(value));
  }

  useEffect(() => {
    const flush = () => window.codeclaw?.flushDrafts(draftSessions.current.map((session) => ({ id: session.id, draft: session.draft, draftEditId: session.editingId || "", draftAttachments: session.attachments.map((item) => item.id) })));
    window.addEventListener("beforeunload", flush);
    window.addEventListener("codeclaw-before-quit", flush);
    return () => { window.removeEventListener("beforeunload", flush); window.removeEventListener("codeclaw-before-quit", flush); };
  }, []);
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
        if (Number.isFinite(value.layout?.sidebarWidth)) setSidebarWidth(value.layout.sidebarWidth);
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
    if (!active || opening || importing) return;
    const id = active.id, draft = active.draft, draftEditId = active.editingId || "", draftAttachments = active.attachments.map((item) => item.id);
    const timer = setTimeout(() => window.codeclaw.invoke("session.update", { id, patch: { draft, draftEditId, draftAttachments } }).catch(onError), 500);
    return () => clearTimeout(timer);
  }, [activeId, active?.draft, active?.editingId, active?.attachments, opening, importing]);
  useEffect(() => {
    if (activeId) window.codeclaw.invoke("session.activate", { session: activeId }).catch(onError);
  }, [activeId]);
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
      if (active && ["project.open", "session.new", "session.open", "session.select", "session.move"].includes(method))
        await window.codeclaw.invoke("session.update", { id: active.id, patch: { draft: active.draft, draftEditId: active.editingId || "", draftAttachments: active.attachments.map((item) => item.id) } });
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
  function editLastPrompt() {
    if (!active || active.busy || !lastUser?.id) return;
    update({ editingId: lastUser.id, draftBeforeEdit: active.draft, draft: lastUser.text, attachmentsBeforeEdit: active.attachments, attachments: lastUser.attachments || [] });
    requestAnimationFrame(() => input.current?.focus());
  }
  async function send(continuation?: string) {
    if (opening || importing || modelChanging || permissionChanging || !active || active.busy || (!(continuation || active.draft).trim() && !active.attachments.length)) return;
    conversationScroll.followLatest();
    const id = active.id,
      text = (continuation || active.draft).trim() || "请查看这些附件。",
      selected = active.selected,
      mode = active.mode,
      attached = continuation ? [] : active.attachments,
      replaceMessageId = continuation ? undefined : active.editingId;
    setSessions((previous) =>
      previous.map((session) =>
        session.id === id ? replaceMessageId ? { ...session, busy: true, error: "" } : beginTurn(session, text) : session,
      ),
    );
    try {
      await window.codeclaw.invoke("prompt", {
        session: id,
        text,
        skills: selected,
        mode,
        replaceMessageId,
        attachments: attached.map((item) => item.id),
      });
    } catch (error) {
      setSessions((previous) =>
        previous.map((session) =>
          applyEvent(session.id === id ? { ...session, draft: text, editingId: replaceMessageId, attachments: attached } : session, {
            session: id,
            type: "turn_error",
            message: String(error),
          }),
        ),
      );
    }
  }
  async function importFiles(files: File[]) {
    if (!files.length) return;
    if (importing || opening) { onError(importing ? "正在添加附件，请稍后再添加" : "会话正在打开，请稍后再添加附件"); return; }
    if (active?.busy) { onError("请先停止当前任务，再添加附件"); return; }
    setImporting(true);
    try {
      let id = activeId;
      const existing = active?.attachments.map((item) => item.id) || [];
      if (!active) {
        const value = await window.codeclaw.invoke("session.new", { scope: "standalone" });
        if (value.catalog) receiveCatalog(value.catalog);
        setSessions((previous) => [...previous, createSession(value)]);
        id = value.id; setActiveId(id);
      }
      const items = await window.codeclaw.importFiles(id, files, existing);
      setSessions((previous) => previous.map((session) => session.id === id ? { ...session, attachments: items } : session));
    } catch (error) { onError(error); }
    finally { setImporting(false); }
  }
  async function openAttachment(item: Attachment) {
    try {
      if (item.kind === "image") setAttachmentPreview(await invoke("attachments.preview", { id: item.id }));
      else await invoke("attachments.open", { id: item.id });
    } catch (error) { onError(error); }
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
  const runningCommands = Object.values(active?.commands || {}).filter((command) => command.status === "running").length;

  return (
    <div className={"workbench " + (!panelOpen ? "panel-hidden" : "")} style={{ "--sidebar-width": effectiveSidebarWidth + "px", "--inspector-width": clampPanelWidth(panelWidth, viewport, effectiveSidebarWidth) + "px" } as React.CSSProperties}
      onDragEnter={(event) => { if (Array.from(event.dataTransfer.types).includes("Files")) { event.preventDefault(); dragDepth.current++; setFileDrag(true); } }}
      onDragLeave={() => { if (--dragDepth.current <= 0) { dragDepth.current = 0; setFileDrag(false); } }}
      onDragOver={(event) => { if (Array.from(event.dataTransfer.types).includes("Files")) { event.preventDefault(); event.dataTransfer.dropEffect = active?.busy || importing ? "none" : "copy"; } }}
      onDrop={(event) => { if (Array.from(event.dataTransfer.types).includes("Files")) { event.preventDefault(); event.stopPropagation(); dragDepth.current = 0; setFileDrag(false); void importFiles(Array.from(event.dataTransfer.files)); } }}>
      {fileDrag && <div className="file-drop-overlay"><Paperclip size={28} /><strong>{active?.busy ? "请先停止当前任务" : "松开以添加附件"}</strong><span>图片、PDF 和本地文件</span></div>}
      <aside className="sidebar">
        <PanelResizer side="left" width={effectiveSidebarWidth} viewport={viewport} panelOpen={panelOpen} change={setSidebarWidth}
          save={(sidebarWidth) => { void window.codeclaw.invoke("layout.save", { sidebarWidth }).catch(onError); }} />
        <div className="brand">
          <span className="brand-mark">
            <Code2 size={23} strokeWidth={2.4} />
          </span>
          <strong>
            CodeClaw
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
        <div className="conversation" ref={conversationScroll.ref} onScroll={conversationScroll.onScroll} onWheel={conversationScroll.onWheel}>
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
                        text: active.scope === "standalone" ? "明确目标，拆解步骤。" : "分析代码结构、技术栈和启动方式。",
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
                <article key={index} className={"message " + message.role} aria-label={message.role === "user" ? "你的消息" : "CodeClaw 回复"}>
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
                    {!!message.attachments?.length && <Attachments items={message.attachments} open={openAttachment} />}
                    {!!message.trace?.files?.length && <FileChanges files={message.trace.files} project={active.project} onOpen={openFile} />}
                    {message === lastUser && message.id && <button className="edit-prompt-button" aria-label="编辑上一条提示词" title="编辑上一条提示词（空输入框连按两次 Esc）"
                      disabled={opening || active.busy || !!runningWorkers || !!runningCommands} onClick={editLastPrompt}><Pencil size={13} /></button>}
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
        </div>
        <div className="composer-area">
          {active?.error && !paused && (
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
          {!!runningCommands && !active?.busy && <button className="worker-strip" onClick={() => invoke("stop").catch(onError)}><Square size={13} />{runningCommands} 个后台命令正在运行 · 停止所有后台任务</button>}
          {paused && !active?.editingId && <div className="paused-actions" role="status"><span>任务已停止，已收到的内容保留。</span>
            <button disabled={opening || importing || modelChanging || permissionChanging || !!active?.draft.trim() || !!active?.attachments.length} title={active?.draft.trim() || active?.attachments.length ? "请先发送或清空当前草稿" : "保留上下文，继续未完成的任务"} onClick={() => void send("请从刚才停止的位置继续完成上一项任务。先核对已完成的操作与当前文件状态，避免重复执行；按原任务要求完成并汇报结果。")}><Play size={13} />继续执行</button>
            {lastUser?.id && <button disabled={opening || !!runningWorkers || !!runningCommands} onClick={editLastPrompt}><Pencil size={13} />编辑提示词</button>}
          </div>}
          <div className={"composer " + (!active ? "disabled" : "")} onPaste={(event) => { if (event.clipboardData.files.length) { event.preventDefault(); void importFiles(Array.from(event.clipboardData.files)); } }}>
            {active?.editingId && <div className="editing-prompt"><span>编辑上一条提示词 · 发送后重新执行，已修改的文件保留</span>
              <button aria-label="取消编辑提示词" disabled={active.busy} onClick={() => update({ editingId: undefined, draft: active.draftBeforeEdit || "", draftBeforeEdit: undefined, attachments: active.attachmentsBeforeEdit || [], attachmentsBeforeEdit: undefined })}><X size={13} /></button></div>}
            {active && <Attachments items={active.attachments} open={openAttachment} disabled={importing || active.busy} remove={(id) => update({ attachments: active.attachments.filter((item) => item.id !== id) })} />}
            {importing && <div className="attachment-loading" role="status"><LoaderCircle size={13} className="spin" />正在添加附件…</div>}
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
              disabled={opening || !active || active.busy} inputRef={input} update={update} send={send} editPrevious={editLastPrompt} />
            <div className="composer-toolbar">
              <div className="composer-options">
                <input ref={fileInput} type="file" multiple hidden aria-label="选择附件文件" onChange={(event) => { const files = Array.from(event.target.files || []); event.target.value = ""; void importFiles(files); }} />
                <button className="attach-button" type="button" aria-label="添加附件" title="添加文件或图片，也可直接拖入或粘贴图片" disabled={opening || importing || active?.busy} onClick={() => fileInput.current?.click()}><Paperclip size={16} /></button>
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
                  title={active?.permissionMode === "full" ? "默认自动执行；显式权限规则、Plan 限制和中断恢复确认仍然生效" : "文件修改、命令和需要审批的外部工具会请求批准"}
                  disabled={opening || !active || permissionChanging} value={active?.permissionMode || "ask"}
                  onChange={(event) => void setPermissionMode(event.target.value as PermissionMode)}>
                  <option value="ask">请求批准</option><option value="full">完全访问</option>
                </select>
                <span className="toolbar-divider" />
                {globalModel?.connections?.some((connection) => connection.models.length) ? <select
                  className="model-picker" aria-label="选择模型" title="切换 API 与模型"
                  disabled={opening || modelChanging || active?.busy || !!runningWorkers || !!runningCommands}
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
                    disabled={opening || importing || modelChanging || permissionChanging || (!active?.draft.trim() && !active?.attachments.length)}
                    onClick={() => void send()}
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
              {active?.permissionMode === "full" ? "完全访问：默认自动执行，遵守显式权限规则" : "请求批准：需要审批的工具会先确认"}
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
        <PanelResizer width={clampPanelWidth(panelWidth, viewport, effectiveSidebarWidth)} viewport={viewport} sidebarWidth={effectiveSidebarWidth} change={setPanelWidth}
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
          deleteConnection={async (connectionId) => {
            const config = await invoke("config.delete", { connectionId, confirm: true });
            setGlobalModel(config);
            setSessions((previous) => previous.map((session) => ({ ...session, config })));
            return config;
          }}
          discoverModels={(value) => window.codeclaw.invoke("config.models", value)}
        />
      )}
      {attachmentPreview && <AttachmentPreview image={attachmentPreview} close={() => setAttachmentPreview(undefined)} />}
      {mcpOpen && active && (
        <McpDialog
          key={active.id}
          sessionId={active.id}
          busy={
            active.busy ||
            !!runningCommands ||
            active.workers.some((worker) => worker.status === "running")
          }
          invoke={invoke}
          close={() => setMcpOpen(false)}
        />
      )}
    </div>
  );
}
