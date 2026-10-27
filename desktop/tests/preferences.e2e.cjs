const { _electron: electron } = require("playwright");
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), http = require("node:http"), assert = require("node:assert/strict");
const { spawn } = require("node:child_process"), { once } = require("node:events");
const { resolveEnginePython } = require("../electron/runtime.cjs");
const { atomicJSON, projectKey } = require("../electron/catalog.cjs");
const { historyRoot } = require("../electron/history-storage.cjs");
const root = path.resolve(__dirname, ".."), fixture = fs.mkdtempSync(path.join(os.tmpdir(), "codeclaw-preferences-e2e-"));
const data = path.join(fixture, "data"), config = path.join(fixture, "config"), home = path.join(fixture, "home");
for (const directory of [data, config, home]) fs.mkdirSync(directory);
atomicJSON(path.join(data, "workbench.json"), { projects: [], activeConversation: null, layout: { sidebarWidth: 280, inspectorWidth: 360 } });
atomicJSON(path.join(config, "config.json"), { model_provider: "openai", openai_model: "", openai_api_key: "", openai_base_url: "http://127.0.0.1:9/v1" });
const apps = [], errors = [];
let holder, requests = 0;
const server = http.createServer((req, res) => {
  let raw = ""; req.on("data", (chunk) => raw += chunk);
  req.on("end", () => {
    if (!JSON.parse(raw).stream) return res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({ choices: [{ message: { role: "assistant", content: "fixture memory" } }] }));
    requests++;
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    res.end('data: {"id":"fixture","choices":[{"index":0,"delta":{"content":"PREFERENCES_TASK_COMPLETED"},"finish_reason":null}]}\n\ndata: {"id":"fixture","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n');
  });
});
async function launch(label) {
  const env = { ...process.env, USERPROFILE: home, HOME: home, CODECLAW_DATA_DIR: data,
    CODECLAW_USER_DATA: path.join(fixture, "browser-" + label), CODECLAW_CONFIG_DIR: config, CODECLAW_TEST_HEADLESS: "1", CODECLAW_SKILLS_PATHS: "" };
  for (const key of ["ELECTRON_RUN_AS_NODE", "CODECLAW_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "CODECLAW_MODEL_PROVIDER"]) delete env[key];
  const app = await electron.launch({ executablePath: process.env.CODECLAW_TEST_EXECUTABLE, args: process.env.CODECLAW_TEST_EXECUTABLE ? [] : [root], env, timeout: 30000 });
  apps.push(app);
  const page = await app.firstWindow(); page.setDefaultTimeout(45000); page.on("pageerror", (error) => errors.push(error.message));
  await page.waitForFunction(() => document.querySelector(".new-task")?.disabled === false);
  return { app, page };
}
const invoke = (page, method, params = {}) => page.evaluate(({ method, params }) => window.codeclaw.invoke(method, params), { method, params });
async function main() {
  try {
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    atomicJSON(path.join(config, "config.json"), { model_provider: "openai", openai_model: "fixture", openai_api_key: "fixture", openai_base_url: `http://127.0.0.1:${server.address().port}/v1` });
    const [one, two] = await Promise.all([launch("one"), launch("two")]);
    await Promise.all([invoke(one.page, "layout.save", { sidebarWidth: 324 }), invoke(two.page, "layout.save", { inspectorWidth: 516 })]);
    const projects = Array.from({ length: 6 }, (_, index) => path.join(fixture, "project-" + index));
    for (const project of projects) fs.mkdirSync(project);
    for (let index = 0; index < projects.length; index += 2)
      await Promise.all([invoke(one.page, "project.open", { path: projects[index] }), invoke(two.page, "project.open", { path: projects[index + 1] })]);
    await Promise.all([invoke(one.page, "project.update", { path: projects[0], patch: { name: "renamed", pinned: true } }),
      invoke(two.page, "project.update", { path: projects[1], patch: { collapsed: true } })]);
    const [createdOne] = await Promise.all([invoke(one.page, "session.new"), invoke(two.page, "session.new")]);
    let saved = JSON.parse(fs.readFileSync(path.join(data, "workbench.json")));
    assert.equal(saved.projects.length, 6); assert.deepEqual(saved.layout, { sidebarWidth: 324, inspectorWidth: 516 });
    assert.equal(saved.projectMeta[projectKey(projects[0])].name, "renamed"); assert(saved.projectMeta[projectKey(projects[0])].pinned);
    assert(saved.projectMeta[projectKey(projects[1])].collapsed);
    assert.equal(saved.standaloneDirectories.length, 2); assert.notEqual(...saved.standaloneDirectories);
    for (const directory of saved.standaloneDirectories) assert.equal(fs.readdirSync(path.join(historyRoot(directory, data), "sessions")).filter((name) => name.endsWith(".json")).length, 1);
    await Promise.all([invoke(one.page, "project.remove", { path: projects[0], confirm: true }), invoke(two.page, "layout.save", { inspectorWidth: 540 })]);
    saved = JSON.parse(fs.readFileSync(path.join(data, "workbench.json")));
    assert(!saved.projects.includes(projects[0])); assert(!saved.projectMeta[projectKey(projects[0])]);
    assert.equal(saved.projects.length, 5); assert.equal(saved.layout.inspectorWidth, 540); assert.equal(saved.layout.sidebarWidth, 324);
    assert(saved.workspace.some((entry) => entry.project === saved.activeConversation.project && entry.resumeId === saved.activeConversation.resumeId));
    const packaged = !!process.env.CODECLAW_TEST_EXECUTABLE;
    const python = resolveEnginePython({ packaged, resourcesPath: packaged ? path.join(path.dirname(process.env.CODECLAW_TEST_EXECUTABLE), "resources") : undefined, projectRoot: path.dirname(root) });
    holder = spawn(python, [...(packaged ? ["-I"] : []), path.join(root, "../tests/fixtures/settings_writer.py"), "unused", "hold"], {
      windowsHide: true, stdio: ["pipe", "pipe", "pipe"], cwd: path.dirname(root), env: { ...process.env, CODECLAW_DATA_DIR: data, PYTHONPATH: path.dirname(root) },
    });
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("Preference holder readiness timeout")), 10000);
      let output = "";
      holder.stdout.on("data", (chunk) => { output += chunk; if (output.includes("OWNED")) { clearTimeout(timer); resolve(); } });
      holder.once("exit", (code) => { clearTimeout(timer); if (!output.includes("OWNED")) reject(new Error("Preference holder exited: " + code)); });
    });
    await one.page.evaluate(() => { window.preferenceTestEvents = []; window.codeclaw.onEvent((event) => window.preferenceTestEvents.push(event)); });
    await invoke(one.page, "prompt", { session: createdOne.id, text: "Say hello", skills: [], mode: "normal" });
    await one.page.getByRole("alert").filter({ hasText: "工作台设置更新失败" }).waitFor();
    await one.page.waitForFunction(() => window.preferenceTestEvents.some((event) => event.type === "turn_finished"));
    assert.equal(requests, 1);
    assert.equal(await one.page.evaluate(() => window.preferenceTestEvents.filter((event) => ["turn_error", "engine_exit"].includes(event.type)).length), 0);
    const released = once(holder, "exit"); holder.stdin.end("release\n"); await released;
    await invoke(one.page, "session.activate", { session: createdOne.id });
    saved = JSON.parse(fs.readFileSync(path.join(data, "workbench.json")));
    const savedBytes = fs.readFileSync(path.join(data, "workbench.json"));
    fs.writeFileSync(path.join(data, "workbench.json"), "{broken");
    await one.app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].emit("focus"));
    await assert.rejects(invoke(one.page, "catalog.refresh"));
    assert.equal(fs.readFileSync(path.join(data, "workbench.json"), "utf8"), "{broken");
    fs.writeFileSync(path.join(data, "workbench.json"), savedBytes);
    await invoke(one.page, "session.activate", { session: createdOne.id });
    const snapshot = saved.activeConversation;
    await Promise.all([one.app.close(), two.app.close()]);
    const restarted = await launch("restart");
    const result = await invoke(restarted.page, "catalog.refresh");
    assert.equal(result.projects.length, 5); assert.equal(result.standalone.length, 2);
    assert.equal((await invoke(restarted.page, "app.initialize")).activeConversation.resumeId, snapshot.resumeId);
    assert.deepEqual(errors, []);
    console.log("Preferences E2E passed: two actual Electron instances preserve project additions, metadata, distinct workspaces, widths and removal; real preference-lock denial reports a notice while a local protocol-model turn completes; corrupt preferences stay intact without crashing on focus, release/retry and restart succeed; no remote model");
  } finally {
    if (holder && holder.exitCode === null) { const released = once(holder, "exit"); holder.stdin.end("release\n"); await released; }
    await Promise.allSettled(apps.map((app) => app.close()));
    server.closeAllConnections(); server.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
