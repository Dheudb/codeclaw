const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const fs = require("node:fs");
const path = require("node:path");
const { SessionCatalog, projectKey, sessionKey, atomicJSON } = require("./catalog.cjs");
const { EngineTransport } = require("./transport.cjs");
const { resolveEnginePython } = require("./runtime.cjs");
const { defaultWorkspace } = require("./workspace.cjs");
if (process.env.CODECLAW_USER_DATA)
  app.setPath("userData", process.env.CODECLAW_USER_DATA);
const sessions = new Map();
let window;
let quitting = false;
let configTransport;
let activeSessionId;
let restoration;
const catalog = new SessionCatalog(settings, saveSettings);
let workspaceQueue = Promise.resolve();
let catalogRevision = 0;
const engineMethods = new Set([
  "prompt",
  "respond",
  "stop",
  "changes",
  "capabilities.list",
  "capabilities.toggle",
  "skills.read",
  "agents.read",
  "agents.cancel",
  "config.save",
  "config.select",
  "mcp.list",
  "mcp.market",
  "mcp.install",
  "mcp.save",
  "mcp.toggle",
  "mcp.remove",
  "mcp.test",
  "mcp.reload",
  "capabilities.example",
]);
function settingsPath() {
  return path.join(app.getPath("userData"), "workbench.json");
}
function settings() {
  if (!fs.existsSync(settingsPath())) return { projects: [] };
  return JSON.parse(fs.readFileSync(settingsPath(), "utf8"));
}
function saveSettings(value) {
  atomicJSON(settingsPath(), value);
}
function workspaceSettings() {
  const saved = settings().defaultDirectory;
  return { defaultDirectory: typeof saved === "string" && path.isAbsolute(saved)
    ? saved : defaultWorkspace({ home: app.getPath("home") }) };
}
function ensureWorkspace(directory) {
  if (typeof directory !== "string" || !path.isAbsolute(directory.trim()))
    throw new Error("默认工作目录必须是完整的绝对路径");
  directory = directory.trim();
  fs.mkdirSync(directory, { recursive: true });
  if (!fs.statSync(directory).isDirectory()) throw new Error("请选择一个目录");
  return fs.realpathSync(directory);
}
function broadcastCatalog() {
  const snapshot = { ...catalog.snapshot(), revision: ++catalogRevision };
  if (window && !window.isDestroyed())
    window.webContents.send("workbench:event", { type: "catalog_changed", catalog: snapshot });
  return snapshot;
}
function send(id, payload) {
  const source = sessions.get(id);
  if (source && payload.type === "turn_finished") source.busy = false;
  if (source && payload.type === "subagent_status") source.workers.set(payload.agent.agent_id, payload.agent.status);
  if (payload.type === "session_started" && sessions.has(id)) {
    sessions.get(id).resumeId = payload.sessionId;
    saveWorkspace();
  }
  if (window && !window.isDestroyed())
    window.webContents.send("workbench:event", { session: id, ...payload });
  if (payload.type === "engine_exit") sessions.delete(id);
  if (["session_started", "sessions_changed"].includes(payload.type)) broadcastCatalog();
}
function saveWorkspace() {
  const active = sessions.get(activeSessionId);
  saveSettings({
    ...settings(),
    activeConversation: active ? { project: active.project, resumeId: active.resumeId, scope: active.scope } : null,
    workspace: [...sessions.entries()].map(([id, session]) => ({
      project: session.project,
      resumeId: session.resumeId,
      scope: session.scope,
      active: id === activeSessionId,
    })),
  });
}
function latestSavedSession(project, scope = "project") {
  try {
    return catalog.list(project).filter((row) => row.scope === scope && !row.archived && !row.error)
      .sort((a, b) => b.updated - a.updated)[0]?.sessionId;
  } catch {
    return undefined;
  }
}
async function restoreWorkspace() {
  const saved = settings();
  const legacyEntries = Array.isArray(saved.workspace) ? saved.workspace :
    (saved.projects || []).map((project) => ({ project, resumeId: latestSavedSession(project) }))
      .filter((entry) => entry.resumeId).slice(0, 1);
  const selected = Object.prototype.hasOwnProperty.call(saved, "activeConversation") ? saved.activeConversation :
    legacyEntries.find((entry) => entry.active) || legacyEntries[0];
  const entries = selected ? [selected] : [];
  const restored = [], errors = [];
  for (const entry of entries.slice(0, 8)) {
    try {
      if (entry.scope !== "standalone" && catalog.snapshot().projects.some((project) => projectKey(project.path) === projectKey(entry.project) && project.archived)) continue;
      let resumeId = entry.resumeId;
      if (resumeId && (!/^[a-zA-Z0-9_-]+$/.test(resumeId) ||
        !fs.existsSync(path.join(entry.project, ".codeclaw", "sessions", resumeId + ".json"))))
        resumeId = latestSavedSession(entry.project, entry.scope || "project");
      if (resumeId && catalog.list(entry.project).find((row) => row.sessionId === resumeId)?.archived)
        resumeId = latestSavedSession(entry.project, entry.scope || "project");
      if (!resumeId && entry.scope === "standalone") continue;
      const initial = await createSession(entry.project, resumeId);
      restored.push(initial);
      activeSessionId = initial.id;
    } catch {
      errors.push("无法恢复项目：" + entry.project);
    }
  }
  activeSessionId ||= restored[0]?.id;
  return { restored, activeId: activeSessionId, restoreErrors: errors };
}
function engineRoot() {
  return app.isPackaged
    ? path.join(process.resourcesPath, "engine")
    : path.resolve(__dirname, "../..");
}
function globalConfig() {
  if (!configTransport || configTransport.closed) {
    const directory = app.getPath("userData");
    fs.mkdirSync(directory, { recursive: true });
    configTransport = new EngineTransport({
      python: resolveEnginePython({
        packaged: app.isPackaged,
        resourcesPath: process.resourcesPath,
        projectRoot: path.resolve(__dirname, "../.."),
      }),
      isolated: app.isPackaged,
      engineRoot: engineRoot(),
      project: directory,
      onEvent: () => {},
    });
  }
  return configTransport;
}
async function createSession(project, resumeId) {
  project = fs.realpathSync(project);
  if (!fs.statSync(project).isDirectory()) throw new Error("项目目录不存在");
  const row = resumeId ? catalog.find(sessionKey(project, resumeId)) : catalog.create(project);
  resumeId = row.sessionId;
  const id = row.id;
  if (sessions.has(id)) return { existingId: id };
  if (sessions.size >= 8) {
    const idle = [...sessions.entries()].find(([key, source]) => key !== activeSessionId && !running(source));
    if (!idle) throw new Error("已有 8 个运行中的会话，请先停止一个任务");
    await disposeSession(idle[0]);
  }
  const transport = new EngineTransport({
    python: resolveEnginePython({
      packaged: app.isPackaged,
      resourcesPath: process.resourcesPath,
      projectRoot: path.resolve(__dirname, "../.."),
    }),
    isolated: app.isPackaged,
    engineRoot: engineRoot(),
    project,
    onEvent: (event) => send(id, event),
  });
  sessions.set(id, { transport, project, resumeId, scope: row.scope, busy: false, workers: new Map() });
  try {
    const initial = await transport.request("initialize", {
      sessionId: resumeId,
      permissionMode: row.permissionMode,
    });
    return { id, ...initial, scope: row.scope, title: row.title, savedDraft: row.draft || "" };
  } catch (error) {
    sessions.delete(id);
    await transport.close();
    throw error;
  }
}
function running(source) {
  return source.busy || [...source.workers.values()].includes("running");
}
async function disposeSession(id) {
  const source = sessions.get(id);
  if (!source) return;
  await source.transport.close();
  sessions.delete(id);
  if (window && !window.isDestroyed()) window.webContents.send("workbench:event", { type: "session_unloaded", session: id });
}
async function selectRow(row) {
  if (row.archived || row.scope !== "standalone" && catalog.snapshot().projects.some((project) => projectKey(project.path) === projectKey(row.project) && project.archived))
    throw new Error("请先在设置的已归档会话页面恢复此会话");
  if (row.scope !== "standalone") catalog.updateProject(row.project, { open: true });
  const initial = await createSession(row.project, row.sessionId);
  activeSessionId = row.id;
  saveWorkspace();
  return { ...initial, activeId: row.id, catalog: broadcastCatalog() };
}
async function fallback(project, excluded = [], scope = "project") {
  const snapshot = catalog.snapshot();
  const eligible = [...snapshot.standalone, ...snapshot.projects.filter((entry) => entry.open && !entry.archived)
    .flatMap((entry) => entry.sessions)].filter((row) => !row.archived && !row.error && !excluded.includes(row.id));
  const next = eligible.filter((row) => row.scope === scope && projectKey(row.project) === projectKey(project))
    .sort((a, b) => b.updated - a.updated)[0] || eligible.sort((a, b) => b.updated - a.updated)[0];
  if (next) return selectRow(next);
  const projectScope = snapshot.projects.find((entry) => projectKey(entry.path) === projectKey(project) && entry.open && !entry.archived);
  if (projectScope && scope !== "standalone") return selectRow(catalog.create(projectScope.path));
  activeSessionId = undefined;
  saveWorkspace();
  return { activeId: "", catalog: broadcastCatalog() };
}
async function invoke(method, params) {
  if (method === "external.open") {
    const url = new URL(params.url);
    if (!["https:", "http:"].includes(url.protocol) || url.username || url.password)
      throw new Error("只支持 HTTP/HTTPS 网页链接");
    await shell.openExternal(url.href);
    return true;
  }
  if (method === "app.initialize") {
    restoration ||= restoreWorkspace();
    return {
      ...settings(),
      ...await restoration,
      catalog: broadcastCatalog(),
      version: app.getVersion(),
      buildId: JSON.parse(fs.readFileSync(path.join(app.getAppPath(), "package.json"), "utf8")).buildId || "",
      config: await globalConfig().request("config.get"),
      workspaceSettings: workspaceSettings(),
    };
  }
  if (method === "workspace.get") return workspaceSettings();
  if (method === "layout.save") {
    if (!Number.isFinite(params.inspectorWidth) || params.inspectorWidth < 280 || params.inspectorWidth > 760)
      throw new Error("右侧面板宽度必须在 280 到 760 之间");
    const layout = { inspectorWidth: Math.round(params.inspectorWidth) };
    saveSettings({ ...settings(), layout });
    return layout;
  }
  if (method === "workspace.save") {
    const defaultDirectory = ensureWorkspace(params.defaultDirectory);
    saveSettings({ ...settings(), defaultDirectory });
    return { defaultDirectory };
  }
  if (method === "workspace.choose") {
    const result = await dialog.showOpenDialog(window, {
      title: "选择默认工作目录", properties: ["openDirectory", "createDirectory"],
      defaultPath: workspaceSettings().defaultDirectory,
    });
    return result.canceled ? null : result.filePaths[0];
  }
  if (method === "catalog.refresh") return broadcastCatalog();
  if (method === "archive.restore") {
    if ((typeof params.id === "string") === (typeof params.path === "string")) throw new Error("请选择要恢复的会话或项目");
    catalog.restoreArchive(params);
    return { catalog: broadcastCatalog() };
  }
  if (method === "catalog.view") {
    catalog.setView(params);
    return broadcastCatalog();
  }
  if (method === "session.activate") {
    if (sessions.has(params.session)) {
      activeSessionId = params.session;
      saveWorkspace();
    }
    return true;
  }
  if (method === "config.get") return globalConfig().request("config.get");
  if (method === "config.models") return globalConfig().request(method, params);
  if (["config.save", "config.select"].includes(method) && !params.session)
    return globalConfig().request(method, params);
  if (method === "project.open") {
    let project = params.path;
    if (!project) {
      const result = await dialog.showOpenDialog(window, {
        title: "打开项目目录",
        properties: ["openDirectory"],
      });
      if (result.canceled) return null;
      project = result.filePaths[0];
    }
    if (catalog.snapshot().projects.some((entry) => projectKey(entry.path) === projectKey(fs.realpathSync(project)) && entry.archived))
      throw new Error("项目已归档，请在设置的已归档会话页面恢复");
    project = catalog.addProject(project);
    const active = sessions.get(activeSessionId);
    const resumeId = active && active.scope !== "standalone" && projectKey(active.project) === projectKey(project) ? active.resumeId : latestSavedSession(project);
    return selectRow(resumeId ? catalog.find(sessionKey(project, resumeId)) : catalog.create(project));
  }
  if (method === "session.select") return selectRow(catalog.find(params.id));
  if (method === "session.open" || method === "session.new") {
    let project = params.path;
    const scope = method === "session.new" && !project ? "standalone" : method === "session.open" ? sessions.get(params.session)?.scope || "project" : "project";
    if (method === "session.new" && !project) {
      project = ensureWorkspace(workspaceSettings().defaultDirectory);
    }
    const source = method === "session.new" ? { project } : sessions.get(params.session);
    if (!source || scope !== "standalone" && !catalog.snapshot().projects.some((entry) => projectKey(entry.path) === projectKey(source.project))) throw new Error("项目不存在");
    if (scope !== "standalone" && catalog.snapshot().projects.some((entry) => projectKey(entry.path) === projectKey(source.project) && entry.archived))
      throw new Error("请先在设置的已归档会话页面恢复项目");
    if (params.resumeId) return selectRow(catalog.find(sessionKey(source.project, params.resumeId)));
    const empty = catalog.list(source.project).find((row) => row.scope === scope && !row.archived && !row.error && !row.draft &&
      row.title === "新会话" && !running(sessions.get(row.id) || { busy: false, workers: new Map() }) &&
      JSON.parse(fs.readFileSync(catalog.file(row.project, row.sessionId), "utf8")).messages?.length === 0);
    return selectRow(empty || (scope === "standalone" ? catalog.createStandalone(source.project) : catalog.create(source.project)));
  }
  if (method === "session.update") {
    const ids = [...new Set(params.ids || [params.id])];
    for (const id of ids) catalog.update(catalog.find(id), params.patch || {});
    if (params.patch?.archived && ids.includes(activeSessionId)) {
      const active = catalog.find(activeSessionId);
      return fallback(active.project, ids, active.scope);
    }
    return { catalog: broadcastCatalog() };
  }
  if (method === "session.fork") {
    const row = catalog.find(params.id);
    if (sessions.has(row.id) && running(sessions.get(row.id))) throw new Error("先停止当前任务，再复制会话");
    return selectRow(catalog.fork(row));
  }
  if (method === "session.delete") {
    if (params.confirm !== true) throw new Error("请确认删除会话");
    const rows = [...new Set(params.ids || [params.id])].map((id) => catalog.find(id));
    if (!params.stop && rows.some((row) => sessions.has(row.id) && running(sessions.get(row.id))))
      throw new Error("会话正在运行，请先停止任务或选择停止并删除");
    const wasActive = rows.find((row) => row.id === activeSessionId);
    for (const row of rows) await disposeSession(row.id);
    for (const row of rows) catalog.delete(row);
    if (wasActive) return fallback(wasActive.project, rows.map((row) => row.id), wasActive.scope);
    saveWorkspace();
    return { catalog: broadcastCatalog() };
  }
  if (["project.update", "project.close", "project.remove", "project.delete"].includes(method)) {
    const project = catalog.snapshot().projects.find((entry) => projectKey(entry.path) === projectKey(params.path));
    if (!project) throw new Error("项目不存在");
    if (method === "project.update") {
      catalog.updateProject(project.path, params.patch || {});
      if (params.patch?.archived && sessions.get(activeSessionId)?.scope !== "standalone" && sessions.get(activeSessionId)?.project === project.path) return fallback(project.path);
      return { catalog: broadcastCatalog() };
    }
    if (["project.delete", "project.remove"].includes(method) && params.confirm !== true) throw new Error("请确认移除项目");
    const live = [...sessions.entries()].filter(([, value]) => value.scope !== "standalone" && projectKey(value.project) === projectKey(project.path));
    if (!params.stop && live.some(([, source]) => running(source))) throw new Error("项目有运行中的任务，请先停止任务");
    const wasActive = live.some(([id]) => id === activeSessionId);
    for (const [id] of live) await disposeSession(id);
    if (method === "project.delete") for (const row of project.sessions) catalog.delete(row);
    if (method === "project.close") catalog.updateProject(project.path, { open: false });
    else catalog.removeProject(project.path);
    if (wasActive) return fallback(project.path);
    saveWorkspace();
    return { catalog: broadcastCatalog() };
  }
  if (method === "session.close") {
    const source = sessions.get(params.session);
    if (source) {
      await disposeSession(params.session);
      if (activeSessionId === params.session) activeSessionId = sessions.keys().next().value;
      saveWorkspace();
    }
    return true;
  }
  const source = sessions.get(params.session);
  if (!source) throw new Error("请先打开项目");
  if (method === "permissions.set") {
    if (!["ask", "full"].includes(params.mode)) throw new Error("请选择请求批准或完全访问");
    const row = catalog.find(params.session), previous = row.permissionMode;
    catalog.update(row, { permissionMode: params.mode });
    try {
      const result = await source.transport.request(method, { mode: params.mode });
      broadcastCatalog();
      return result;
    } catch {
      // A lost reply must never leave a hidden full-access engine running.
      await disposeSession(params.session);
      catalog.update(row, { permissionMode: previous });
      broadcastCatalog();
      throw new Error("切换权限失败，当前会话引擎已停止，请重新打开会话");
    }
  }
  if (method === "project.reveal") {
    await shell.openPath(source.project);
    return true;
  }
  if (method === "capabilities.directory") {
    const directory = await source.transport.request(method, params);
    await shell.openPath(directory);
    return true;
  }
  if (method === "capabilities.import") {
    if (!["skills", "agents"].includes(params.kind))
      throw new Error("能力类型无效");
    const result = await dialog.showOpenDialog(window, {
      title:
        params.kind === "skills"
          ? "导入 Skill 或技能集合目录"
          : "导入子代理定义目录",
      properties: ["openDirectory"],
    });
    if (result.canceled) return null;
    return source.transport.request(method, {
      kind: params.kind,
      path: result.filePaths[0],
    });
  }
  if (!engineMethods.has(method)) throw new Error("不支持的操作");
  const { session: _session, ...payload } = params;
  if (method === "prompt") {
    if (source.busy) throw new Error("此会话已有运行中的任务");
    source.busy = true;
    const row = catalog.find(params.session);
    catalog.update(row, { draft: "", ...(row.title === "新会话" ? { title: String(params.text || "").split("\n")[0].slice(0, 80) || "新会话" } : {}) });
    broadcastCatalog();
    try { return await source.transport.request(method, payload); }
    catch (error) { source.busy = false; throw error; }
  }
  return source.transport.request(method, payload);
}
app.whenReady().then(() => {
  window = new BrowserWindow({
    show: process.env.CODECLAW_TEST_HEADLESS !== "1",
    width: 1440,
    height: 940,
    minWidth: 1050,
    minHeight: 700,
    backgroundColor: "#fafaf9",
    title: "CodeClaw",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-navigate", (event) => event.preventDefault());
  window.webContents.session.setPermissionRequestHandler(
    (_webContents, _permission, callback) => callback(false),
  );
  ipcMain.handle("workbench:invoke", async (event, method, params) => {
    if (
      event.sender !== window.webContents ||
      event.senderFrame !== window.webContents.mainFrame
    )
      throw new Error("Invalid IPC sender");
    if (typeof method !== "string" || !params || typeof params !== "object")
      throw new Error("Invalid request");
    if (/^(app\.initialize|layout\.save|permissions\.set|config\.(save|select)|workspace\.save|archive\.restore|project\.(open|update|close|remove|delete)|session\.(open|new|select|update|fork|delete|close|activate)|catalog\.)/.test(method)) {
      const operation = workspaceQueue.then(() => invoke(method, params));
      workspaceQueue = operation.catch(() => {});
      return operation;
    }
    return invoke(method, params);
  });
  window.on("focus", () => { if (restoration) broadcastCatalog(); });
  const url = !app.isPackaged && process.env.CODECLAW_DEV_URL;
  if (url && url !== "http://127.0.0.1:5178")
    throw new Error("Invalid development server URL");
  if (url) window.loadURL(url);
  else window.loadFile(path.join(__dirname, "../dist/index.html"));
});
app.on("window-all-closed", () => app.quit());
app.on("before-quit", (event) => {
  if (quitting || (!sessions.size && !configTransport)) return;
  event.preventDefault();
  quitting = true;
  Promise.allSettled(
    [...sessions.values()]
      .map(({ transport }) => transport.close())
      .concat(configTransport ? [configTransport.close()] : []),
  ).finally(() => {
    sessions.clear();
    app.quit();
  });
});
