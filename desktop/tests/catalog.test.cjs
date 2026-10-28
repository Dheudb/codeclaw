const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { SessionCatalog, sessionKey, projectKey, atomicJSON } = require("../electron/catalog.cjs");
const { moveSession } = require("../electron/session-move.cjs");
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "xuloom-catalog-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const project = path.join(root, "项目 with spaces"); fs.mkdirSync(project);
  let settings = { projects: [] };
  const catalog = new SessionCatalog(() => settings, (value) => { settings = value; }, path.join(root, "data"));
  catalog.addProject(project);
  return { root, project, catalog, settings: () => settings };
}

test("goal titles use the visible objective rather than internal continuation instructions", (t) => {
  const {catalog,project}=fixture(t);
  const row=catalog.create(project);
  atomicJSON(catalog.file(project,row.sessionId),{messages:[{role:"user",content:"Internal goal continuation instructions",_display_text:"用户原始目标"}],metadata:{}});
  assert.equal(catalog.list(project)[0].title,"用户原始目标");
});

test("moving an independent conversation changes its directory and preserves history, traces, draft and owned child results", (t) => {
  const { catalog, root, project: base } = fixture(t);
  const target = path.join(root, "target"); fs.mkdirSync(target); catalog.addProject(target);
  const row = catalog.createStandalone(base), project = row.project;
  catalog.update(row, { title: "待迁移", draft: "unfinished", pinned: true, permissionMode: "full" });
  const payload = { messages: [{ role: "user", content: "old history" }, { role: "assistant", content: "answer" }], metadata: {
    turn_traces: [{ id: "turn", elapsedMs: 1234 }], subagents: [{ agent_id: "owned", status: "completed", result: "child answer" }],
    memory_files: { cwd: project }, incremental_write_queue: { pending: ["old-write"] }, permissions: { always_allow_tools: ["bash_tool"] },
  } };
  atomicJSON(catalog.file(project, row.sessionId), payload);
  const child = path.join(catalog.directory(project), "subagents/owned.json"), unrelated = path.join(catalog.directory(project), "subagents/other.json");
  atomicJSON(child, { session_id: row.sessionId, messages: [{ role: "assistant", content: "child answer" }] });
  atomicJSON(unrelated, { session_id: "another", messages: [] });
  const proof = path.join(catalog.artifacts(project), row.sessionId, "result.txt");
  fs.mkdirSync(path.dirname(proof), { recursive: true }); fs.writeFileSync(proof, "large tool output");
  fs.writeFileSync(path.join(project, "source.py"), "keep source code");
  const moved = moveSession(catalog, catalog.find(row.id), target);
  assert.equal(moved.id, sessionKey(target, row.sessionId)); assert.equal(moved.scope, "project");
  assert.equal(moved.draft, "unfinished"); assert.equal(moved.title, "待迁移"); assert(moved.pinned); assert.equal(moved.permissionMode, "full");
  const saved = JSON.parse(fs.readFileSync(catalog.file(target, row.sessionId)));
  assert.deepEqual(saved.messages, payload.messages); assert.deepEqual(saved.metadata.turn_traces, payload.metadata.turn_traces);
  assert.deepEqual(saved.metadata.subagents, payload.metadata.subagents);
  assert.equal(saved.metadata.memory_files, undefined); assert.equal(saved.metadata.incremental_write_queue, undefined); assert.equal(saved.metadata.permissions, undefined);
  assert.equal(projectKey(saved.metadata.workspace_moves[0].to), projectKey(fs.realpathSync(target)));
  assert(!fs.existsSync(catalog.file(project, row.sessionId))); assert(!fs.existsSync(child)); assert(fs.existsSync(unrelated));
  assert(fs.existsSync(path.join(catalog.directory(target), "subagents/owned.json")));
  assert.equal(fs.readFileSync(path.join(catalog.artifacts(target), row.sessionId, "result.txt"), "utf8"), "large tool output");
  assert.equal(fs.readFileSync(path.join(project, "source.py"), "utf8"), "keep source code");
  assert.equal(catalog.snapshot().standalone.length, 0); assert.throws(() => catalog.find(row.id));
});

test("same-directory independent-to-project conversion preserves transcript bytes", (t) => {
  const { catalog, project: base } = fixture(t), row = catalog.createStandalone(base);
  const project = row.project; catalog.addProject(project);
  const file = catalog.file(project, row.sessionId), bytes = fs.readFileSync(file);
  const moved = moveSession(catalog, row, project);
  assert.equal(moved.id, row.id); assert.equal(moved.scope, "project");
  assert.deepEqual(fs.readFileSync(file), bytes); assert.equal(catalog.snapshot().standalone.length, 0);
});

test("move collisions preserve both records and retry succeeds after resolving the collision", (t) => {
  const { catalog, root, project } = fixture(t);
  const target = path.join(root, "target"); fs.mkdirSync(target); catalog.addProject(target);
  const row = catalog.create(project); catalog.update(row, { title: "keep me", draft: "keep draft" });
  const original = fs.readFileSync(catalog.file(project, row.sessionId));
  const collision = catalog.file(target, row.sessionId); atomicJSON(collision, { messages: [{ role: "user", content: "unrelated target" }] });
  const otherBytes = fs.readFileSync(collision);
  assert.throws(() => moveSession(catalog, row, target), /相同 ID/);
  assert.deepEqual(fs.readFileSync(collision), otherBytes); fs.unlinkSync(collision);
  assert.deepEqual(fs.readFileSync(catalog.file(project, row.sessionId)), original);
  assert.equal(catalog.find(row.id).draft, "keep draft");
  const moved = moveSession(catalog, row, target);
  assert.equal(moved.draft, "keep draft");
  assert(!fs.existsSync(catalog.file(project, row.sessionId))); assert(fs.existsSync(collision));
});
test("one durable row per conversation; metadata survives CLI history writes", (t) => {
  const { project, catalog } = fixture(t);
  const row = catalog.create(project);
  catalog.update(row, { title: "自定义名称", pinned: true, archived: true, draft: "未发送内容" });
  atomicJSON(catalog.file(project, row.sessionId), { messages: [{ role: "user", content: "CLI 对话" }], metadata: { model_runtime: {} } });
  const records = catalog.list(project);
  assert.equal(records.length, 1);
  assert.equal(records[0].title, "自定义名称");
  assert.equal(records[0].draft, "未发送内容");
  assert(records[0].archived && records[0].pinned);
  assert.equal(records[0].id, row.id);
  catalog.update(row, { archived: false });
  assert.equal(catalog.list(project)[0].archived, false);
});

test("standalone sessions keep their scope even when their working directory is also a project", (t) => {
  const { catalog, project } = fixture(t);
  const projectRow = catalog.create(project);
  const standalone = catalog.createStandalone(project);
  assert.deepEqual(catalog.snapshot().projects[0].sessions.map((row) => row.id), [projectRow.id]);
  assert.deepEqual(catalog.snapshot().standalone.map((row) => row.id), [standalone.id]);
  catalog.update(standalone, { title: "独立任务", archived: true, draft: "keep draft" });
  catalog.updateProject(project, { archived: true });
  catalog.restoreArchive({ id: standalone.id });
  assert(catalog.snapshot().projects[0].archived, "Restoring an independent conversation must not restore a project");
  assert.equal(catalog.find(standalone.id).draft, "keep draft");
  assert.equal(catalog.fork(catalog.find(standalone.id)).scope, "standalone");
  catalog.removeProject(project);
  assert.equal(catalog.snapshot().projects.length, 0);
  assert.equal(catalog.snapshot().standalone.length, 2);
  assert.equal(catalog.find(standalone.id).scope, "standalone");
  assert(fs.existsSync(catalog.file(project, projectRow.sessionId)));
});

test("independent history spans default-directory changes without registering projects", (t) => {
  const { catalog, root, project } = fixture(t);
  catalog.removeProject(project);
  const other = path.join(root, "other-default"); fs.mkdirSync(other);
  const first = catalog.createStandalone(project), second = catalog.createStandalone(other);
  const snapshot = catalog.snapshot();
  assert.equal(snapshot.projects.length, 0);
  assert.deepEqual(new Set(snapshot.standalone.map((row) => row.id)), new Set([first.id, second.id]));
  catalog.delete(first);
  assert.equal(catalog.find(second.id).scope, "standalone");
});
test("archive restoration reopens the project but preserves other archives and transcript bytes", (t) => {
  const { project, catalog } = fixture(t);
  const first = catalog.create(project), other = catalog.create(project);
  catalog.update(first, { archived: true, draft: "保留草稿", pinned: true });
  catalog.update(other, { archived: true });
  catalog.updateProject(project, { archived: true, collapsed: true, open: false });
  const original = fs.readFileSync(catalog.file(project, first.sessionId));
  catalog.restoreArchive({ id: first.id });
  assert.deepEqual(fs.readFileSync(catalog.file(project, first.sessionId)), original);
  const scope = catalog.snapshot().projects[0];
  assert(scope.open && !scope.archived && !scope.collapsed);
  const restored = catalog.find(first.id);
  assert(!restored.archived && restored.pinned);
  assert.equal(restored.draft, "保留草稿");
  assert(catalog.find(other.id).archived);
  catalog.restoreArchive({ path: project });
  assert(catalog.find(other.id).archived);
});
test("legacy archive visibility is discarded and cannot be enabled by preferences", (t) => {
  const { catalog } = fixture(t);
  catalog.writeSettings({ ...catalog.readSettings(), sidebar: { showArchived: true, sort: "name" } });
  assert.deepEqual(catalog.snapshot().view, { sort: "name" });
  catalog.setView({ showArchived: true, sort: "oldest" });
  assert.deepEqual(catalog.readSettings().sidebar, { sort: "oldest" });
});
test("permission mode is durable per conversation and forks start with approval", (t) => {
  const { catalog, project } = fixture(t);
  const first = catalog.create(project), second = catalog.create(project);
  assert.equal(first.permissionMode, "ask");
  catalog.update(first, { permissionMode: "full" });
  assert.equal(catalog.find(first.id).permissionMode, "full");
  assert.equal(catalog.find(second.id).permissionMode, "ask");
  assert.equal(catalog.fork(catalog.find(first.id)).permissionMode, "ask");
  assert.throws(() => catalog.update(first, { permissionMode: "auto" }), /请求批准或完全访问/);
  assert.equal(catalog.find(first.id).permissionMode, "full");
});
test("delete removes parent and owned child transcripts, preserves other histories and project files", (t) => {
  const { project, catalog } = fixture(t);
  const a = catalog.create(project), b = catalog.create(project);
  const children = path.join(catalog.directory(project), "subagents"); fs.mkdirSync(children);
  atomicJSON(path.join(children, "owned.json"), { session_id: a.sessionId });
  atomicJSON(path.join(children, "other.json"), { session_id: b.sessionId });
  fs.writeFileSync(path.join(project, "code.py"), "print('keep')");
  catalog.delete(a);
  assert(!fs.existsSync(catalog.file(project, a.sessionId)));
  assert(!fs.existsSync(path.join(children, "owned.json")));
  assert(fs.existsSync(path.join(children, "other.json")));
  assert(fs.existsSync(path.join(project, "code.py")));
  assert.deepEqual(catalog.list(project).map((row) => row.id), [b.id]);
});

test("readonly owned attachment copies can be deleted", (t) => {
  const { project, catalog } = fixture(t);
  const row = catalog.create(project);
  catalog.update(row, { title: "retain on failure", draft: "keep draft" });
  const target = catalog.file(project, row.sessionId), before = fs.readFileSync(target);
  const artifacts = path.join(catalog.artifacts(project), row.sessionId);
  fs.mkdirSync(artifacts, { recursive: true });
  const attachment = path.join(artifacts, "readonly.html"); fs.writeFileSync(attachment, "owned copy"); fs.chmodSync(attachment, 0o444);
  catalog.delete(row);
  assert(!fs.existsSync(target)); assert(!fs.existsSync(artifacts));
});

test("a real Windows attachment lock leaves the conversation retryable until released", { skip: process.platform !== "win32", timeout: 15000 }, async (t) => {
  const { spawn } = require("node:child_process");
  const { project, catalog } = fixture(t);
  const row = catalog.create(project), target = catalog.file(project, row.sessionId);
  catalog.update(row, { title: "Windows locked file" });
  const artifacts = path.join(catalog.artifacts(project), row.sessionId);
  fs.mkdirSync(artifacts, { recursive: true });
  const attachment = path.join(artifacts, "locked.html"); fs.writeFileSync(attachment, "owned attachment");
  const script = '$locked=[IO.File]::Open($env:XULOOM_LOCK_FIXTURE,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read); [Console]::Out.WriteLine("LOCKED"); [Console]::Out.Flush(); [Console]::In.ReadLine() | Out-Null; $locked.Dispose()';
  const child = spawn("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script], {
    windowsHide: true, stdio: ["pipe", "pipe", "pipe"], env: { ...process.env, XULOOM_LOCK_FIXTURE: attachment },
  });
  const exited = new Promise((resolve) => child.once("exit", resolve));
  t.after(async () => { child.stdin.end(); await exited; });
  await new Promise((resolve, reject) => {
    let output = "";
    child.stdout.on("data", (chunk) => { output += chunk; if (output.includes("LOCKED")) resolve(); });
    child.once("error", reject);
    child.once("exit", (code) => { if (!output.includes("LOCKED")) reject(new Error("Lock helper failed: " + code)); });
  });
  assert.throws(() => catalog.delete(row), /会话记录已保留/);
  assert(fs.existsSync(target)); assert.equal(catalog.find(row.id).title, "Windows locked file");
  child.stdin.end("\n"); await exited;
  catalog.delete(row);
  assert(!fs.existsSync(target)); assert(!fs.existsSync(artifacts));
});
test("removing a project keeps histories and reimport finds them; closing is persisted separately", (t) => {
  const { project, catalog } = fixture(t);
  const row = catalog.create(project);
  catalog.updateProject(project, { name: "显示名称", open: false, pinned: true });
  assert.equal(catalog.snapshot().projects[0].open, false);
  assert.equal(catalog.snapshot().projects[0].name, "显示名称");
  catalog.removeProject(project);
  assert.equal(catalog.snapshot().projects.length, 0);
  assert(fs.existsSync(catalog.file(project, row.sessionId)));
  catalog.addProject(project);
  assert.equal(catalog.snapshot().projects[0].sessions.length, 1);
});
test("copied session IDs in different projects have distinct UI identities", (t) => {
  const { root, project, catalog } = fixture(t);
  const second = path.join(root, "another"); fs.mkdirSync(second); catalog.addProject(second);
  const row = catalog.create(project);
  atomicJSON(catalog.file(second, row.sessionId), { messages: [] });
  const other = catalog.list(second)[0];
  assert.notEqual(row.id, other.id);
  assert.equal(catalog.find(row.id).project, project);
});
test("fork copies conversation only and keeps a new identity without running state", (t) => {
  const { project, catalog } = fixture(t);
  const source = catalog.create(project, [{ role: "user", content: "hello" }], "原会话");
  const copy = catalog.fork(source);
  assert.notEqual(copy.sessionId, source.sessionId);
  const raw = JSON.parse(fs.readFileSync(catalog.file(project, copy.sessionId), "utf8"));
  assert.deepEqual(raw.metadata, {});
  assert.equal(raw.messages[0].content, "hello");
  assert.equal(copy.title, "原会话（副本）");
});
test("history catalog has no arbitrary 50-record cutoff and malformed history stays visible", (t) => {
  const { project, catalog } = fixture(t);
  const root = catalog.directory(project); fs.mkdirSync(root, { recursive: true });
  for (let i = 0; i < 65; i++) atomicJSON(catalog.file(project, "session-" + i), { messages: [] });
  fs.writeFileSync(catalog.file(project, "broken"), "{");
  assert.equal(catalog.list(project).length, 66);
  const broken = catalog.find(sessionKey(project, "broken"));
  assert(broken.error);
  catalog.delete(broken);
  assert.equal(catalog.list(project).length, 65);
});
test("invalid IDs and corrupt indexes are refused without overwriting user data", (t) => {
  const { project, catalog } = fixture(t);
  assert.throws(() => catalog.file(project, "../../config"));
  const row = catalog.create(project);
  const index = catalog.indexPath(project); fs.writeFileSync(index, "broken-index");
  assert.throws(() => catalog.update(row, { title: "new" }));
  assert.equal(fs.readFileSync(index, "utf8"), "broken-index");
});
