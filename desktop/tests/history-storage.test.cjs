const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const { execFileSync } = require("node:child_process");
const { SessionCatalog, atomicJSON } = require("../electron/catalog.cjs");
const { historyRoot } = require("../electron/history-storage.cjs");
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "codeclaw-central-test-")); t.after(() => fs.rmSync(root, { recursive: true, force: true }));
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
    assert(!fs.existsSync(path.join(row.project, ".codeclaw")));
  }
  const fork = catalog.fork(first); assert.notEqual(fork.project, first.project);
  catalog.delete(first); assert(fs.existsSync(path.join(first.project, "same.html"))); assert.equal(catalog.snapshot().standalone.length, 2);
});
test("legacy transcripts, drafts, archives, child records and spilled outputs migrate once without changing project configuration", (t) => {
  const { project, data, catalog } = fixture(t);
  const legacy = path.join(project, ".codeclaw"), proof = path.join(legacy, "tool-results/old/result.txt");
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
  const python = process.env.CODECLAW_TEST_PYTHON || [".venv", ".venv-mcp"].map((name) => path.join(repo, name, process.platform === "win32" ? "Scripts/python.exe" : "bin/python")).find((filename) => fs.existsSync(filename)) || (process.platform === "win32" ? "python" : "python3");
  const output = execFileSync(python, ["-c", "import sys,json;from codeclaw.core.session import SessionManager;m=SessionManager(cwd=sys.argv[1]);assert m.load_session_state(sys.argv[2])['_valid'];assert m.save_session_state(sys.argv[2],[{'role':'user','content':'CLI shared'}]);print(json.dumps({'directory':m.session_dir}))", project, row.sessionId], { cwd: repo, windowsHide: true, encoding: "utf8", env: { ...process.env, CODECLAW_DATA_DIR: data, PYTHONPATH: repo, USERPROFILE: path.join(root, "home"), HOME: path.join(root, "home") } });
  assert.equal(JSON.parse(output).directory.toLowerCase(), catalog.directory(project).toLowerCase());
  assert.equal(catalog.find(row.id).draft, "keep"); assert.equal(catalog.find(row.id).title, "desktop name");
  assert(catalog.find(row.id).archived); assert.equal(JSON.parse(fs.readFileSync(catalog.file(project, row.sessionId))).messages[0].content, "CLI shared");
  assert(!fs.existsSync(path.join(project, ".codeclaw/sessions")));
});
test("legacy independent conversations are isolated on upgrade and owned outputs are retained", async (t) => {
  const { project, root, catalog } = fixture(t);
  const record = { messages: [{ role: "assistant", content: [{ type: "tool_use", id: "write", name: "file_write_tool", input: { file_path: "pelican.html", content: "original" } }] }], metadata: {} };
  atomicJSON(path.join(project, ".codeclaw/sessions/one.json"), record);
  atomicJSON(path.join(project, ".codeclaw/sessions/two.json"), record);
  atomicJSON(path.join(project, ".codeclaw/session-index.json"), { version: 1, sessions: { one: { scope: "standalone", draft: "keep", pinned: true }, two: { scope: "standalone", archived: true } } });
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
