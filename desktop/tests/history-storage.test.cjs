const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const { execFileSync } = require("node:child_process");
const { SessionCatalog, atomicJSON } = require("../electron/catalog.cjs");
const { historyRoot } = require("../electron/history-storage.cjs");
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "xuloom-central-test-")); t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const project = path.join(root, "项目"), data = path.join(root, "data"); fs.mkdirSync(project);
  let settings = { projects: [project] };
  const catalog = new SessionCatalog(() => settings, (value) => { settings = value; }, data);
  return { root, project, data, catalog };
}
test("every standalone creation owns a distinct working directory and all records stay outside workspaces", (t) => {
  const { project, data, catalog } = fixture(t), first = catalog.createStandalone(project), second = catalog.createStandalone(project);
  assert.notEqual(first.project, second.project); assert.equal(path.dirname(first.project), project);
  fs.writeFileSync(path.join(first.project, "same.html"), "first"); fs.writeFileSync(path.join(second.project, "same.html"), "second");
  assert.equal(fs.readFileSync(path.join(first.project, "same.html"), "utf8"), "first");
  for (const row of [first, second]) {
    assert(fs.existsSync(catalog.file(row.project, row.sessionId))); assert(catalog.file(row.project, row.sessionId).startsWith(data));
    assert(!fs.existsSync(path.join(row.project, ".xuloom")));
  }
  const fork = catalog.fork(first); assert.notEqual(fork.project, first.project);
  catalog.delete(first); assert(fs.existsSync(path.join(first.project, "same.html"))); assert.equal(catalog.snapshot().standalone.length, 2);
});
test("legacy transcripts, drafts, archives, child records and spilled outputs migrate once without changing project configuration", (t) => {
  const { project, data, catalog } = fixture(t);
  const legacy = path.join(project, ".xuloom"), proof = path.join(legacy, "tool-results/old/result.txt");
  atomicJSON(path.join(legacy, "sessions/old.json"), { messages: [{ role: "assistant", content: "Read " + proof }], metadata: { turn_traces: [{ id: "turn" }] } });
  atomicJSON(path.join(legacy, "sessions/subagents/child.json"), { session_id: "old", messages: [] });
  atomicJSON(path.join(legacy, "session-index.json"), { version: 1, sessions: { old: { title: "旧会话", archived: true, draft: "草稿", pinned: true } } });
  fs.mkdirSync(path.dirname(proof), { recursive: true }); fs.writeFileSync(proof, "large result");
  atomicJSON(path.join(legacy, "capabilities.json"), { imported_roots: ["keep"] });
  const row = catalog.list(project)[0]; assert(row.archived && row.pinned); assert.equal(row.draft, "草稿");
  const stored = JSON.parse(fs.readFileSync(catalog.file(project, "old")));
  assert(stored.messages[0].content.includes(path.join(historyRoot(project, data), "tool-results/old/result.txt")));
  assert(fs.existsSync(path.join(catalog.directory(project), "subagents/child.json")));
  assert.equal(fs.readFileSync(path.join(catalog.artifacts(project), "old/result.txt"), "utf8"), "large result");
  for (const name of ["sessions", "session-index.json", "tool-results"]) assert(!fs.existsSync(path.join(legacy, name)));
  assert(fs.existsSync(path.join(legacy, "capabilities.json"))); assert.equal(catalog.list(project).length, 1);
  catalog.delete(row); assert.equal(catalog.list(project).length, 0);
  assert(!fs.existsSync(path.join(catalog.artifacts(project), "old")));
});
test("CLI and desktop use identical storage and CLI saves preserve desktop metadata", (t) => {
  const { root, project, data, catalog } = fixture(t), row = catalog.create(project);
  catalog.update(row, { draft: "keep", archived: true, title: "desktop name" });
  const repo = path.resolve(__dirname, "../..");
  const python = process.env.XULOOM_TEST_PYTHON || [".venv", ".venv-mcp"].map((name) => path.join(repo, name, process.platform === "win32" ? "Scripts/python.exe" : "bin/python")).find((filename) => fs.existsSync(filename)) || (process.platform === "win32" ? "python" : "python3");
  const output = execFileSync(python, ["-c", "import sys,json;from xuloom.core.session import SessionManager;m=SessionManager(cwd=sys.argv[1]);assert m.load_session_state(sys.argv[2])['_valid'];assert m.save_session_state(sys.argv[2],[{'role':'user','content':'CLI shared'}]);print(json.dumps({'directory':m.session_dir}))", project, row.sessionId], { cwd: repo, windowsHide: true, encoding: "utf8", env: { ...process.env, XULOOM_DATA_DIR: data, PYTHONPATH: repo, USERPROFILE: path.join(root, "home"), HOME: path.join(root, "home") } });
  assert.equal(JSON.parse(output).directory.toLowerCase(), catalog.directory(project).toLowerCase());
  assert.equal(catalog.find(row.id).draft, "keep"); assert.equal(catalog.find(row.id).title, "desktop name");
  assert(catalog.find(row.id).archived); assert.equal(JSON.parse(fs.readFileSync(catalog.file(project, row.sessionId))).messages[0].content, "CLI shared");
  assert(!fs.existsSync(path.join(project, ".xuloom/sessions")));
});
test("legacy independent conversations are isolated on upgrade and owned outputs are retained", async (t) => {
  const { project, root, catalog } = fixture(t);
  const record = { messages: [{ role: "assistant", content: [{ type: "tool_use", id: "write", name: "file_write_tool", input: { file_path: "pelican.html", content: "original" } }] }], metadata: {} };
  atomicJSON(path.join(project, ".xuloom/sessions/one.json"), record);
  atomicJSON(path.join(project, ".xuloom/sessions/two.json"), record);
  atomicJSON(path.join(project, ".xuloom/session-index.json"), { version: 1, sessions: { one: { scope: "standalone", draft: "keep", pinned: true }, two: { scope: "standalone", archived: true } } });
  fs.writeFileSync(path.join(project, "pelican.html"), "latest existing file");
  catalog.writeSettings({ ...catalog.readSettings(), standaloneDirectories: [project], activeConversation: { project, resumeId: "one", scope: "standalone" } });
  await catalog.upgradeStandaloneWorkspaces(path.join(root, "workspaces"));
  const rows = catalog.snapshot().standalone;
  assert.equal(rows.length, 2); assert.notEqual(rows[0].project, rows[1].project);
  for (const row of rows) assert.equal(fs.readFileSync(path.join(row.project, "pelican.html"), "utf8"), "latest existing file");
  assert.equal(fs.readFileSync(path.join(project, "pelican.html"), "utf8"), "latest existing file");
  const selected = rows.find((row) => row.sessionId === "one"); assert.equal(selected.draft, "keep"); assert(selected.pinned);
  assert.equal(catalog.readSettings().activeConversation.project, selected.project);
  assert(rows.find((row) => row.sessionId === "two").archived);
  await catalog.upgradeStandaloneWorkspaces(path.join(root, "workspaces")); assert.equal(catalog.snapshot().standalone.length, 2);
});

test("a lost legacy-upgrade reply restores active and open conversation associations on restart", async (t) => {
  const { project, root, catalog, data } = fixture(t);
  const { mutateSync } = require("../electron/storage-helper.cjs");
  const row = catalog.create(project, [{ role: "user", content: "legacy" }], "legacy", "standalone");
  const index = catalog.indexPath(project), metadata = JSON.parse(fs.readFileSync(index));
  metadata.sessions[row.sessionId].workspaceIsolated = false; atomicJSON(index, metadata);
  const entry = { project, resumeId: row.sessionId, scope: "standalone", id: "keep-ui-identity" };
  catalog.writeSettings({ ...catalog.readSettings(), standaloneDirectories: [project], activeConversation: entry, workspace: [entry] });
  const base = path.join(root, "workspaces");
  await assert.rejects(catalog.upgradeStandaloneWorkspaces(base, (spec) => {
    const result = mutateSync(catalog, spec);
    if (spec.action === "move") throw new Error("lost reply");
    return result;
  }), /lost reply/);
  assert.equal(catalog.readSettings().activeConversation.project, project);
  assert(!fs.existsSync(catalog.file(project, row.sessionId)));
  const restarted = new SessionCatalog(catalog.readSettings, catalog.writeSettings, data);
  await restarted.upgradeStandaloneWorkspaces(base);
  const settings = restarted.readSettings(), moved = restarted.snapshot().standalone[0];
  assert.equal(settings.activeConversation.project, moved.project);
  assert.equal(settings.workspace[0].project, moved.project); assert.equal(settings.workspace[0].id, entry.id);
  assert.equal(moved.sessionId, row.sessionId); assert.equal(fs.readdirSync(base).length, 1);
  await restarted.upgradeStandaloneWorkspaces(base); assert.equal(fs.readdirSync(base).length, 1);
});
