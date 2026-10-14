const fs = require("node:fs");
const path = require("node:path");
const { projectKey } = require("./catalog.cjs");
const { rebase } = require("./history-storage.cjs");

function targetProject(catalog, directory) {
  if (typeof directory !== "string" || !path.isAbsolute(directory)) throw new Error("请选择目标项目");
  const canonical = fs.realpathSync(directory);
  const project = catalog.snapshot().projects.find((entry) => projectKey(entry.path) === projectKey(canonical));
  if (!project || project.archived || !fs.statSync(canonical).isDirectory()) throw new Error("目标项目不存在或已归档");
  return canonical;
}

// Only session-owned records move. No project source, Skills or MCP configuration is copied.
function prepareMove(catalog, row, directory, scope = "project") {
  const target = scope === "standalone" ? fs.realpathSync(directory) : targetProject(catalog, directory);
  if (row.error || row.archived && scope !== "standalone") throw new Error("请先修复或恢复此会话，再移动");
  const sameDirectory = projectKey(row.project) === projectKey(target);
  const sourceIndex = catalog.index(row.project), targetIndex = sameDirectory ? sourceIndex : catalog.index(target);
  if (!sameDirectory && (fs.existsSync(catalog.file(target, row.sessionId)) || Object.hasOwn(targetIndex, row.sessionId)))
    throw new Error("目标项目中已有相同 ID 的会话，移动已取消，原记录保留");
  const sourceFile = catalog.file(row.project, row.sessionId);
  const original = fs.readFileSync(sourceFile);
  const raw = JSON.parse(original.toString("utf8"));
  if (!(Array.isArray(raw) || Array.isArray(raw.messages))) throw new Error("会话记录格式不正确");
  let payload = rebase(Array.isArray(raw) ? { messages: raw, metadata: {} } : structuredClone(raw), catalog.artifacts(row.project), catalog.artifacts(target));
  if (scope === "standalone") payload = rebase(payload, row.project, target);
  const files = [];
  function assertOwned(project, candidate) {
    const base = fs.realpathSync(catalog.dataRoot), relative = path.relative(base, candidate);
    if (relative.startsWith(".." + path.sep) || relative === ".." || path.isAbsolute(relative)) throw new Error("会话资源必须位于项目目录中");
    let probe = base;
    for (const name of relative.split(path.sep)) {
      probe = path.join(probe, name);
      if (fs.existsSync(probe) && fs.lstatSync(probe).isSymbolicLink()) throw new Error("无法移动链接形式的会话资源");
    }
  }
  function add(source, destination, content) {
    assertOwned(row.project, source); assertOwned(target, destination);
    if (fs.lstatSync(source).isSymbolicLink()) throw new Error("无法移动链接形式的会话资源");
    if (fs.existsSync(destination)) throw new Error("目标项目存在同名会话资源，原记录保留");
    const original = fs.readFileSync(source);
    files.push({ source, destination, content: content || original, original, stat: fs.statSync(source) });
  }
  if (!sameDirectory) {
    const metadata = payload.metadata ||= {};
    // Restoring these would reuse the old cwd, shell processes, pending writes or approvals.
    for (const field of ["sandbox", "browser", "shell_tasks", "permissions", "memory_files", "file_state_cache", "artifacts",
      "incremental_write_queue", "auto_commit", "content_replacements", "security_classifier", "vcr", "team", "ultraplan", "memory_extractor", "hooks"])
      delete metadata[field];
    metadata.workspace_moves = [...(Array.isArray(metadata.workspace_moves) ? metadata.workspace_moves : []),
      { from: row.project, to: target, at: Date.now() }];
    add(sourceFile, catalog.file(target, row.sessionId), Buffer.from(JSON.stringify(payload, null, 2)));
    const children = path.join(catalog.directory(row.project), "subagents");
    const owned = new Set((metadata.subagents || []).map((record) => record.agent_id));
    if (fs.existsSync(children)) {
      if (fs.lstatSync(children).isSymbolicLink()) throw new Error("无法移动链接形式的子代理记录");
      for (const name of fs.readdirSync(children).filter((name) => /^[a-zA-Z0-9_-]+\.json$/.test(name))) {
        const source = path.join(children, name);
        if (fs.lstatSync(source).isSymbolicLink()) continue;
        let child; try { child = JSON.parse(fs.readFileSync(source, "utf8")); } catch { continue; }
        if (child.session_id === row.sessionId || !child.session_id && owned.has(name.slice(0, -5)))
          add(source, path.join(catalog.directory(target), "subagents", name));
      }
    }
    function collect(source, destination) {
      assertOwned(row.project, source); assertOwned(target, destination);
      if (!fs.existsSync(source)) return;
      const stat = fs.lstatSync(source);
      if (stat.isSymbolicLink()) throw new Error("无法移动链接形式的会话资源");
      if (stat.isDirectory()) for (const name of fs.readdirSync(source)) collect(path.join(source, name), path.join(destination, name));
      else add(source, destination);
    }
    collect(path.join(catalog.artifacts(row.project), row.sessionId), path.join(catalog.artifacts(target), row.sessionId));
  }
  return { target, sameDirectory, sourceIndex, targetIndex, files };
}

function moveSession(catalog, row, directory, scope = "project") {
  const plan = prepareMove(catalog, row, directory, scope);
  const { target, sameDirectory, sourceIndex, targetIndex, files } = plan;
  const metadata = { ...sourceIndex[row.sessionId], title: row.title, created: row.created, scope, ...(scope === "standalone" ? { workspaceIsolated: true } : {}) };
  if (sameDirectory) {
    catalog.saveIndex(target, { ...sourceIndex, [row.sessionId]: metadata });
    return catalog.list(target).find((entry) => entry.sessionId === row.sessionId);
  }
  const written = [], removed = [];
  const nextSource = { ...sourceIndex }; delete nextSource[row.sessionId];
  try {
    for (const file of files) {
      fs.mkdirSync(path.dirname(file.destination), { recursive: true });
      fs.writeFileSync(file.destination, file.content, { flag: "wx" }); written.push(file);
      fs.utimesSync(file.destination, file.stat.atime, file.stat.mtime);
    }
    catalog.saveIndex(target, { ...targetIndex, [row.sessionId]: metadata });
    catalog.saveIndex(row.project, nextSource);
    for (const file of files) {
      if (!fs.readFileSync(file.source).equals(file.original)) throw new Error("会话记录已被其他程序修改，请刷新后重试");
      fs.unlinkSync(file.source); removed.push(file);
    }
  } catch (error) {
    // Retain all original bytes and metadata if any copy/index/delete operation fails.
    for (const file of removed) fs.writeFileSync(file.source, file.original, { flag: "wx" });
    for (const file of written) if (fs.existsSync(file.destination)) fs.unlinkSync(file.destination);
    catalog.saveIndex(row.project, sourceIndex); catalog.saveIndex(target, targetIndex);
    throw error;
  }
  return catalog.list(target).find((entry) => entry.sessionId === row.sessionId);
}
module.exports = { prepareMove, moveSession };
