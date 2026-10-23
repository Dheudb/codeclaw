const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { historyRoot, migrateProject, rebase } = require("./history-storage.cjs");
const { dataDirectory } = require("./workspace.cjs");

const validId = (id) => typeof id === "string" && /^[a-zA-Z0-9_-]+$/.test(id);
const projectKey = (project) => process.platform === "win32"
  ? path.resolve(project).toLowerCase() : path.resolve(project);
const sessionKey = (project, id) => crypto.createHash("sha256")
  .update(projectKey(project)).digest("hex").slice(0, 16) + ":" + id;
const basename = (project) => path.basename(project);

function atomicJSON(target, value) {
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const temporary = target + "." + crypto.randomUUID() + ".tmp";
  try {
    fs.writeFileSync(temporary, JSON.stringify(value, null, 2), { encoding: "utf8", flag: "wx" });
    fs.renameSync(temporary, target);
  } finally {
    if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
  }
}

/** Durable sidebar data. Runtime processes are a cache, not the session list. */
class SessionCatalog {
  constructor(readSettings, writeSettings, dataRoot = dataDirectory({ home: require("node:os").homedir() })) {
    this.readSettings = readSettings;
    this.writeSettings = writeSettings;
    this.dataRoot = dataRoot;
  }
  directory(project) {
    const base = path.resolve(project);
    const root = path.join(migrateProject(base, this.dataRoot, atomicJSON), "sessions");
    if (fs.existsSync(root) && !projectKey(fs.realpathSync(root)).startsWith(projectKey(path.resolve(this.dataRoot)) + path.sep))
      throw new Error("会话目录必须位于集中存储中");
    return root;
  }
  file(project, id) {
    if (!validId(id)) throw new Error("会话 ID 无效");
    const root = this.directory(project);
    const target = path.join(root, id + ".json");
    if (fs.existsSync(target) && (fs.lstatSync(target).isSymbolicLink() ||
      path.dirname(fs.realpathSync(target)) !== fs.realpathSync(root)))
      throw new Error("会话文件必须位于项目的历史目录中");
    return target;
  }
  indexPath(project) { return path.join(this.directory(project), "..", "index.json"); }
  artifacts(project) { return path.join(historyRoot(project, this.dataRoot), "tool-results"); }
  index(project) {
    const target = this.indexPath(project);
    if (!fs.existsSync(target)) return Object.create(null);
    const data = JSON.parse(fs.readFileSync(target, "utf8"));
    if (!data || data.version !== 1 || typeof data.sessions !== "object" || !data.sessions)
      throw new Error("会话索引损坏，请先修复 session-index.json");
    return Object.assign(Object.create(null), data.sessions);
  }
  saveIndex(project, sessions) { atomicJSON(this.indexPath(project), { version: 1, sessions }); }
  list(project) {
    const root = this.directory(project);
    if (!fs.existsSync(root)) return [];
    const meta = this.index(project);
    return fs.readdirSync(root).filter((name) => name.endsWith(".json") && validId(name.slice(0, -5)))
      .flatMap((name) => {
        const sessionId = name.slice(0, -5);
        try {
          const target = this.file(project, sessionId);
          const stat = fs.statSync(target);
          const raw = JSON.parse(fs.readFileSync(target, "utf8"));
          const messages = Array.isArray(raw) ? raw : raw.messages || [];
          const first = messages.find((message) => message.role === "user" && !message._is_meta &&
            !["team", "subagent"].includes(message._origin));
          const text = typeof first?.content === "string" ? first.content :
            (first?.content || []).filter((block) => block.type === "text").map((block) => block.text).join("\n");
          const item = meta[sessionId] || {};
          return [{ id: sessionKey(project, sessionId), sessionId, project,
            scope: item.scope === "standalone" ? "standalone" : "project",
            title: item.title || text.split("\n")[0].slice(0, 80) || "新会话",
            updated: stat.mtimeMs, created: item.created || stat.birthtimeMs || stat.mtimeMs,
            pinned: !!item.pinned, archived: !!item.archived, draft: item.draft || "", draftEditId: item.draftEditId || "", draftAttachments: item.draftAttachments || [], permissionMode: item.permissionMode === "full" ? "full" : "ask" }];
        } catch (error) {
          // Keep an unreadable record visible and deletable instead of hiding it.
          return [{ id: sessionKey(project, sessionId), sessionId, project,
            scope: meta[sessionId]?.scope === "standalone" ? "standalone" : "project",
            title: meta[sessionId]?.title || "无法读取的会话", updated: 0, created: 0,
            pinned: !!meta[sessionId]?.pinned, archived: !!meta[sessionId]?.archived, permissionMode: meta[sessionId]?.permissionMode === "full" ? "full" : "ask",
            error: error.message }];
        }
      });
  }
  snapshot() {
    const settings = this.readSettings();
    const projects = [...new Set(settings.projects || [])].map((project) => {
      const meta = settings.projectMeta?.[projectKey(project)] || {};
      try {
        return { path: project, name: meta.name || basename(project), open: meta.open !== false,
          pinned: !!meta.pinned, archived: !!meta.archived, collapsed: !!meta.collapsed,
          sessions: this.list(project).filter((row) => row.scope !== "standalone") };
      } catch (error) {
        return { path: project, name: meta.name || basename(project), open: false,
          pinned: !!meta.pinned, archived: !!meta.archived, sessions: [], error: error.message };
      }
    });
    const roots = [...new Map([...(settings.standaloneDirectories || []), ...(settings.projects || [])]
      .map((directory) => [projectKey(directory), directory])).values()];
    const standalone = roots.flatMap((directory) => {
      try { return this.list(directory).filter((row) => row.scope === "standalone"); }
      catch { return []; }
    });
    const sort = settings.sidebar?.sort;
    return { projects, standalone, view: { sort: ["recent", "oldest", "name", "created"].includes(sort) ? sort : "recent" } };
  }
  find(id) {
    const snapshot = this.snapshot();
    const row = [...snapshot.standalone, ...snapshot.projects.flatMap((project) => project.sessions)].find((row) => row.id === id);
    if (!row) throw new Error("会话不存在，可能已被删除");
    return row;
  }
  addProject(project) {
    project = fs.realpathSync(project);
    if (!fs.statSync(project).isDirectory()) throw new Error("项目目录不存在");
    const settings = this.readSettings();
    settings.projects = [project, ...(settings.projects || []).filter((p) => projectKey(p) !== projectKey(project))];
    settings.projectMeta ||= {};
    settings.projectMeta[projectKey(project)] = { ...settings.projectMeta[projectKey(project)], open: true, archived: false };
    this.writeSettings(settings);
    return project;
  }
  updateProject(project, patch) {
    const settings = this.readSettings();
    if (!(settings.projects || []).some((p) => projectKey(p) === projectKey(project))) throw new Error("项目不存在");
    settings.projectMeta ||= {};
    const next = { ...settings.projectMeta[projectKey(project)] };
    for (const field of ["pinned", "archived", "collapsed", "open"]) {
      if (field in patch) next[field] = !!patch[field];
    }
    if ("name" in patch) {
      const name = String(patch.name).trim();
      if (!name || name.length > 100) throw new Error("项目名称需为 1–100 个字符");
      next.name = name;
    }
    settings.projectMeta[projectKey(project)] = next;
    this.writeSettings(settings);
  }
  removeProject(project) {
    const settings = this.readSettings();
    settings.projects = (settings.projects || []).filter((p) => projectKey(p) !== projectKey(project));
    if (settings.projectMeta) delete settings.projectMeta[projectKey(project)];
    this.writeSettings(settings);
  }
  createStandalone(directory) {
    const base = fs.realpathSync(directory);
    directory = path.join(base, "chat-" + crypto.randomUUID());
    fs.mkdirSync(directory, { recursive: false });
    this.seedWorkspaceConfiguration(base, directory);
    const settings = this.readSettings();
    settings.standaloneDirectories = [directory, ...(settings.standaloneDirectories || []).filter((p) => projectKey(p) !== projectKey(directory))];
    this.writeSettings(settings);
    const row = this.create(directory, [], undefined, "standalone");
    const index = this.index(directory); index[row.sessionId].workspaceIsolated = true; this.saveIndex(directory, index);
    return row;
  }
  seedWorkspaceConfiguration(base, target) {
    const settingsFile = path.join(base, ".codeclaw/capabilities.json");
    const inherited = fs.existsSync(settingsFile) ? JSON.parse(fs.readFileSync(settingsFile, "utf8")) : {};
    const roots = (kind) => [".codeclaw", ".agents", ".claude"].map((name) => path.join(base, name, kind)).filter((root) => fs.existsSync(root));
    const skills = [...new Set([...(inherited.skill_roots || []), ...roots("skills")])], agents = [...new Set([...(inherited.agent_roots || []), ...roots("agents")])];
    if (fs.existsSync(settingsFile) || skills.length || agents.length) atomicJSON(path.join(target, ".codeclaw/capabilities.json"), { ...inherited, skill_roots: skills, agent_roots: agents });
    const mcp = path.join(base, "mcp_servers.json");
    if (fs.existsSync(mcp) && !fs.lstatSync(mcp).isSymbolicLink()) fs.copyFileSync(mcp, path.join(target, "mcp_servers.json"), fs.constants.COPYFILE_EXCL);
  }
  upgradeStandaloneWorkspaces(base) {
    const roots = [...new Set([...(this.readSettings().standaloneDirectories || []), ...(this.readSettings().projects || [])])];
    const rows = roots.flatMap((directory) => {
      try { return this.list(directory).filter((row) => row.scope === "standalone" && !row.error && !this.index(directory)[row.sessionId]?.workspaceIsolated); }
      catch { return []; } // snapshot() keeps damaged records and migration conflicts visible.
    });
    if (!rows.length) return;
    fs.mkdirSync(base, { recursive: true });
    const { moveSession } = require("./session-move.cjs");
    for (const row of rows) {
      const target = path.join(base, "chat-" + crypto.randomUUID()); fs.mkdirSync(target);
      this.seedWorkspaceConfiguration(row.project, target);
      const raw = JSON.parse(fs.readFileSync(this.file(row.project, row.sessionId), "utf8")), owned = new Set();
      function collect(value) {
        if (!value || typeof value !== "object") return;
        if (value.type === "tool_use" && ["file_write_tool", "file_edit_tool", "notebook_edit_tool"].includes(value.name)) {
          const filename = value.input?.file_path || value.input?.notebook_path; if (filename) owned.add(filename);
        }
        if (value.path && value.operation && "added" in value) owned.add(value.path);
        for (const child of Object.values(value)) collect(child);
      }
      collect(raw);
      for (const filename of owned) {
        const source = path.resolve(row.project, filename), relative = path.relative(row.project, source);
        if (!relative || relative === ".." || relative.startsWith(".." + path.sep) || path.isAbsolute(relative) || relative.split(path.sep).some((name) => name.startsWith("."))) continue;
        if (!fs.existsSync(source) || !fs.statSync(source).isFile() || fs.lstatSync(source).isSymbolicLink()) continue;
        const canonical = fs.realpathSync(source), check = path.relative(row.project, canonical);
        if (check === ".." || check.startsWith(".." + path.sep) || path.isAbsolute(check)) continue;
        const destination = path.join(target, relative); fs.mkdirSync(path.dirname(destination), { recursive: true }); fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
      }
      const pendingSettings = this.readSettings();
      pendingSettings.standaloneDirectories = [...new Set([...(pendingSettings.standaloneDirectories || []), target])];
      this.writeSettings(pendingSettings);
      const moved = moveSession(this, row, target, "standalone"), settings = this.readSettings();
      settings.standaloneDirectories = [...new Set([...(settings.standaloneDirectories || []), moved.project])];
      const matches = (entry) => entry && projectKey(entry.project) === projectKey(row.project) && entry.resumeId === row.sessionId;
      if (matches(settings.activeConversation)) settings.activeConversation = { ...settings.activeConversation, project: moved.project, scope: "standalone" };
      if (settings.workspace) settings.workspace = settings.workspace.map((entry) => matches(entry) ? { ...entry, project: moved.project, scope: "standalone" } : entry);
      this.writeSettings(settings);
    }
  }
  create(project, messages = [], title, scope = "project") {
    const id = crypto.randomUUID();
    atomicJSON(this.file(project, id), { messages, metadata: {} });
    const index = this.index(project);
    index[id] = { created: Date.now(), scope, ...(title ? { title } : {}) };
    this.saveIndex(project, index);
    return this.list(project).find((row) => row.sessionId === id);
  }
  update(row, patch) {
    const index = this.index(row.project);
    const next = { ...index[row.sessionId] };
    if ("title" in patch) {
      const title = String(patch.title).trim();
      if (!title || title.length > 200) throw new Error("会话名称需为 1–200 个字符");
      next.title = title;
    }
    for (const field of ["pinned", "archived"]) if (field in patch) next[field] = !!patch[field];
    if ("draft" in patch) next.draft = String(patch.draft || "");
    if ("draftAttachments" in patch) {
      if (!Array.isArray(patch.draftAttachments) || patch.draftAttachments.length > 10 || !patch.draftAttachments.every(validId)) throw new Error("附件草稿无效");
      next.draftAttachments = [...new Set(patch.draftAttachments)];
    }
    if ("draftEditId" in patch) {
      if (patch.draftEditId && !validId(patch.draftEditId)) throw new Error("编辑目标无效");
      next.draftEditId = patch.draftEditId || "";
    }
    if ("permissionMode" in patch) {
      if (!["ask", "full"].includes(patch.permissionMode)) throw new Error("请选择请求批准或完全访问");
      next.permissionMode = patch.permissionMode;
    }
    index[row.sessionId] = next;
    this.saveIndex(row.project, index);
  }
  fork(row) {
    const raw = JSON.parse(fs.readFileSync(this.file(row.project, row.sessionId), "utf8"));
    const copy = row.scope === "standalone" ? this.createStandalone(path.dirname(row.project)) : this.create(row.project, [], row.title + "（副本）", row.scope);
    try {
      const source = path.join(this.artifacts(row.project), row.sessionId), destination = path.join(this.artifacts(copy.project), copy.sessionId);
      const attachmentDirectory = path.join(source, "attachments");
      if (fs.existsSync(attachmentDirectory)) fs.cpSync(attachmentDirectory, path.join(destination, "attachments"), {
        recursive: true, errorOnExist: true, force: false,
        filter: (file) => { if (fs.lstatSync(file).isSymbolicLink()) throw new Error("无法复制链接形式的附件"); return true; },
      });
      atomicJSON(this.file(copy.project, copy.sessionId), { messages: rebase(Array.isArray(raw) ? raw : raw.messages || [], source, destination), metadata: {} });
      this.update(copy, { title: row.title + "（副本）" });
      return this.find(copy.id);
    } catch (error) { this.delete(copy); throw error; }
  }
  delete(row) {
    // Runtime disposal must complete before this is called, otherwise closing
    // Python can persist the conversation again and resurrect the deleted row.
    const target = this.file(row.project, row.sessionId);
    const index = this.index(row.project);
    const artifacts = path.resolve(this.artifacts(row.project), row.sessionId);
    // Clean owned outputs before removing the durable conversation/index. If
    // Windows holds an attachment open, the conversation stays visible/retryable.
    if (fs.existsSync(artifacts)) {
      const dataRoot = projectKey(fs.realpathSync(this.dataRoot)) + path.sep;
      if (fs.lstatSync(artifacts).isSymbolicLink() || !projectKey(fs.realpathSync(artifacts)).startsWith(dataRoot))
        throw new Error("会话附件目录必须位于集中存储中，记录已保留");
      try { fs.rmSync(artifacts, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 }); }
      catch (error) {
        if (!["EPERM", "EACCES", "EBUSY", "ENOTEMPTY"].includes(error.code)) throw error;
        const failure = new Error("会话附件仍被占用或无法访问。会话记录已保留，请关闭相关文件或预览后重试删除。");
        failure.cause = error;
        throw failure;
      }
    }
    const subagents = path.join(this.directory(row.project), "subagents");
    if (fs.existsSync(subagents) && !fs.lstatSync(subagents).isSymbolicLink()) {
      for (const name of fs.readdirSync(subagents).filter((name) => name.endsWith(".json"))) {
        const child = path.join(subagents, name);
        if (fs.lstatSync(child).isSymbolicLink()) continue;
        let raw;
        try { raw = JSON.parse(fs.readFileSync(child, "utf8")); } catch { continue; }
        if (raw.session_id === row.sessionId) fs.unlinkSync(child);
      }
    }
    if (fs.existsSync(target)) fs.unlinkSync(target);
    delete index[row.sessionId];
    this.saveIndex(row.project, index);
  }
  setView(patch) {
    const settings = this.readSettings();
    const view = { sort: this.snapshot().view.sort };
    if (patch.sort && ["recent", "oldest", "name", "created"].includes(patch.sort)) view.sort = patch.sort;
    this.writeSettings({ ...settings, sidebar: view });
  }
  restoreArchive({ id, path: project }) {
    const row = id ? this.find(id) : null;
    if (row?.scope === "standalone") {
      if (!fs.statSync(row.project).isDirectory()) throw new Error("默认工作目录不存在或无法访问");
      this.update(row, { archived: false });
      return;
    }
    const scope = this.snapshot().projects.find((entry) => projectKey(entry.path) === projectKey(row ? row.project : project));
    if (!scope) throw new Error("归档项目不存在");
    try {
      if (!fs.statSync(scope.path).isDirectory()) throw new Error("not a directory");
    } catch { throw new Error("项目目录不存在或无法访问，请恢复目录后再试"); }
    if (row) this.update(row, { archived: false });
    this.updateProject(scope.path, { archived: false, open: true, collapsed: false });
  }
}
module.exports = { SessionCatalog, projectKey, sessionKey, atomicJSON };
