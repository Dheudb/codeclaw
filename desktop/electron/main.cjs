const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { EngineTransport } = require("./transport.cjs");
if (process.env.CODECLAW_USER_DATA)
  app.setPath("userData", process.env.CODECLAW_USER_DATA);
const sessions = new Map();
let window;
let quitting = false;
const engineMethods = new Set([
  "prompt",
  "respond",
  "stop",
  "changes",
  "capabilities.list",
  "capabilities.toggle",
  "skills.read",
  "agents.cancel",
  "config.save",
  "mcp.list",
  "mcp.market",
  "mcp.install",
  "mcp.save",
  "mcp.toggle",
  "mcp.remove",
  "mcp.test",
  "mcp.reload",
]);
function settingsPath() {
  return path.join(app.getPath("userData"), "workbench.json");
}
function settings() {
  try {
    return JSON.parse(fs.readFileSync(settingsPath(), "utf8"));
  } catch {
    return { projects: [], python: process.env.CODECLAW_PYTHON || "python" };
  }
}
function saveSettings(value) {
  fs.mkdirSync(app.getPath("userData"), { recursive: true });
  const target = settingsPath();
  fs.writeFileSync(target + ".tmp", JSON.stringify(value, null, 2));
  fs.renameSync(target + ".tmp", target);
}
function send(id, payload) {
  if (window && !window.isDestroyed())
    window.webContents.send("workbench:event", { session: id, ...payload });
}
function engineRoot() {
  return app.isPackaged
    ? path.join(process.resourcesPath, "engine")
    : path.resolve(__dirname, "../..");
}
function projectKey(value) {
  return process.platform === "win32"
    ? path.resolve(value).toLowerCase()
    : path.resolve(value);
}
async function createSession(project, resumeId) {
  project = fs.realpathSync(project);
  if (!fs.statSync(project).isDirectory()) throw new Error("项目目录不存在");
  if (sessions.size >= 8) throw new Error("已打开 8 个会话，请先关闭一个会话");
  const id = crypto.randomUUID();
  const transport = new EngineTransport({
    python: settings().python || "python",
    engineRoot: engineRoot(),
    project,
    onEvent: (event) => send(id, event),
  });
  sessions.set(id, { transport, project });
  try {
    const initial = await transport.request("initialize", {
      sessionId: resumeId,
    });
    return { id, ...initial };
  } catch (error) {
    sessions.delete(id);
    await transport.close();
    throw error;
  }
}
async function invoke(method, params) {
  if (method === "external.open") {
    const url = new URL(params.url);
    if (url.protocol !== "https:" || url.username || url.password)
      throw new Error("只支持 HTTPS 文档链接");
    await shell.openExternal(url.href);
    return true;
  }
  if (method === "app.initialize")
    return { ...settings(), version: app.getVersion() };
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
    const result = await createSession(project);
    const config = settings();
    config.projects = [
      result.project,
      ...(config.projects || []).filter(
        (p) => projectKey(p) !== projectKey(result.project),
      ),
    ].slice(0, 12);
    saveSettings(config);
    return result;
  }
  if (method === "session.open") {
    const source = sessions.get(params.session);
    if (!source) throw new Error("会话不存在");
    return createSession(source.project, params.resumeId);
  }
  if (method === "session.close") {
    const source = sessions.get(params.session);
    if (source) {
      sessions.delete(params.session);
      await source.transport.close();
    }
    return true;
  }
  if (method === "python.choose") {
    const result = await dialog.showOpenDialog(window, {
      title: "选择 Python 解释器",
      properties: ["openFile"],
      filters:
        process.platform === "win32"
          ? [{ name: "Python", extensions: ["exe"] }]
          : [],
    });
    if (result.canceled) return null;
    const config = settings();
    config.python = result.filePaths[0];
    saveSettings(config);
    return config.python;
  }
  const source = sessions.get(params.session);
  if (!source) throw new Error("请先打开项目");
  if (method === "project.reveal") {
    await shell.openPath(source.project);
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
    return invoke(method, params);
  });
  const url = !app.isPackaged && process.env.CODECLAW_DEV_URL;
  if (url && url !== "http://127.0.0.1:5178")
    throw new Error("Invalid development server URL");
  if (url) window.loadURL(url);
  else window.loadFile(path.join(__dirname, "../dist/index.html"));
});
app.on("window-all-closed", () => app.quit());
app.on("before-quit", (event) => {
  if (quitting || !sessions.size) return;
  event.preventDefault();
  quitting = true;
  Promise.allSettled(
    [...sessions.values()].map(({ transport }) => transport.close()),
  ).finally(() => {
    sessions.clear();
    app.quit();
  });
});
