const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { SessionCatalog, sessionKey, atomicJSON } = require("../electron/catalog.cjs");
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "codeclaw-catalog-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const project = path.join(root, "项目 with spaces"); fs.mkdirSync(project);
  let settings = { projects: [] };
  const catalog = new SessionCatalog(() => settings, (value) => { settings = value; });
  catalog.addProject(project);
  return { root, project, catalog, settings: () => settings };
}
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
