const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

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
  constructor(readSettings, writeSettings) {
    this.readSettings = readSettings;
    this.writeSettings = writeSettings;
  }
  directory(project) {
    const base = fs.realpathSync(project);
    const root = path.join(base, ".codeclaw", "sessions");
    if (fs.existsSync(root) && !projectKey(fs.realpathSync(root)).startsWith(projectKey(base) + path.sep))
      throw new Error("会话目录必须位于当前项目中");
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
  indexPath(project) { return path.join(this.directory(project), "..", "session-index.json"); }
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
            title: item.title || text.split("\n")[0].slice(0, 80) || "新会话",
            updated: stat.mtimeMs, created: item.created || stat.birthtimeMs || stat.mtimeMs,
            pinned: !!item.pinned, archived: !!item.archived, draft: item.draft || "", permissionMode: item.permissionMode === "full" ? "full" : "ask" }];
        } catch (error) {
          // Keep an unreadable record visible and deletable instead of hiding it.
          return [{ id: sessionKey(project, sessionId), sessionId, project,
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
          sessions: this.list(project) };
      } catch (error) {
        return { path: project, name: meta.name || basename(project), open: false,
          pinned: !!meta.pinned, archived: !!meta.archived, sessions: [], error: error.message };
      }
    });
    const sort = settings.sidebar?.sort;
    return { projects, view: { sort: ["recent", "oldest", "name", "created"].includes(sort) ? sort : "recent" } };
  }
  find(id) {
    const row = this.snapshot().projects.flatMap((project) => project.sessions).find((row) => row.id === id);
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
  create(project, messages = [], title) {
    const id = crypto.randomUUID();
    atomicJSON(this.file(project, id), { messages, metadata: {} });
    const index = this.index(project);
    index[id] = { created: Date.now(), ...(title ? { title } : {}) };
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
    if ("permissionMode" in patch) {
      if (!["ask", "full"].includes(patch.permissionMode)) throw new Error("请选择请求批准或完全访问");
      next.permissionMode = patch.permissionMode;
    }
    index[row.sessionId] = next;
    this.saveIndex(row.project, index);
  }
  fork(row) {
    const raw = JSON.parse(fs.readFileSync(this.file(row.project, row.sessionId), "utf8"));
    return this.create(row.project, Array.isArray(raw) ? raw : raw.messages || [], row.title + "（副本）");
  }
  delete(row) {
    // Runtime disposal must complete before this is called, otherwise closing
    // Python can persist the conversation again and resurrect the deleted row.
    const target = this.file(row.project, row.sessionId);
    if (fs.existsSync(target)) fs.unlinkSync(target);
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
    const index = this.index(row.project);
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
