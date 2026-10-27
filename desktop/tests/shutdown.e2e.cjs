const { _electron: electron } = require("playwright");
const { spawn } = require("node:child_process"), { once } = require("node:events");
const readline = require("node:readline");
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), http = require("node:http"), assert = require("node:assert/strict");
const { atomicJSON } = require("../electron/catalog.cjs"), { historyRoot } = require("../electron/history-storage.cjs");
const { resolveEnginePython } = require("../electron/runtime.cjs");
const root = path.resolve(__dirname, ".."), repository = path.dirname(root), fixture = fs.mkdtempSync(path.join(os.tmpdir(), "codeclaw-shutdown-e2e-"));
const data = path.join(fixture, "data"), project = path.join(fixture, "project"), config = path.join(fixture, "config"), home = path.join(fixture, "home");
for (const directory of [data, project, config, home]) fs.mkdirSync(directory);
const env = { ...process.env, CODECLAW_DATA_DIR: data, CODECLAW_USER_DATA: data, CODECLAW_CONFIG_DIR: config,
  USERPROFILE: home, HOME: home, CODECLAW_TEST_HEADLESS: "1", CODECLAW_SKILLS_PATHS: "", PYTHONPATH: repository };
for (const name of ["ELECTRON_RUN_AS_NODE", "CODECLAW_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "CODECLAW_MODEL_PROVIDER"]) delete env[name];
const checkpoint = path.join(historyRoot(project, data), "sessions/shutdown.json");
atomicJSON(checkpoint, { messages: [], metadata: {} });
atomicJSON(path.join(data, "workbench.json"), { projects: [project], activeConversation: { project, resumeId: "shutdown", scope: "project" } });
let app, page, holder, lines, requests = 0; const errors = [];
const server = http.createServer((req, res) => {
  let raw = ""; req.on("data", (chunk) => raw += chunk);
  req.on("end", () => {
    if (!JSON.parse(raw).stream) return res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({ choices: [{ message: { role: "assistant", content: "fixture memory" } }] }));
    requests++;
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    res.end('data: {"id":"fixture","choices":[{"index":0,"delta":{"content":"SHUTDOWN_RESULT_SAVED"},"finish_reason":null}]}\n\ndata: {"id":"fixture","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n');
  });
});
async function bounded(promise, label) {
  let timer;
  try { return await Promise.race([promise, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(label + " timed out")), 20000); })]); }
  finally { clearTimeout(timer); }
}
async function launch() {
  app = await electron.launch({ executablePath: process.env.CODECLAW_TEST_EXECUTABLE, args: process.env.CODECLAW_TEST_EXECUTABLE ? [] : [root], env, timeout: 30000 });
  page = await app.firstWindow(); page.setDefaultTimeout(45000); page.on("pageerror", (error) => errors.push(error.message));
  await page.locator(".session-item.active").waitFor(); await page.waitForFunction(() => !document.querySelector(".new-task")?.disabled);
}
async function release() {
  if (holder && holder.exitCode === null) { const closed = once(holder, "exit"); holder.stdin.end("release\n"); await bounded(closed, "checkpoint holder exit"); }
  lines?.close(); holder = undefined;
}
async function main() {
  if (process.platform !== "win32") { console.log("SKIP: checkpoint delete-sharing denial requires Windows; common shutdown contracts run in Python on all platforms"); return; }
  try {
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    atomicJSON(path.join(config, "config.json"), { model_provider: "openai", openai_model: "fixture", openai_api_key: "fixture", openai_base_url: `http://127.0.0.1:${server.address().port}/v1` });
    await launch();
    const input = page.getByRole("textbox", { name: "任务输入" }); await input.fill("Say hello"); await input.press("Enter");
    await page.getByText("SHUTDOWN_RESULT_SAVED", { exact: true }).waitFor(); await page.waitForFunction(() => !document.querySelector(".send-button.stop"));
    const id = await page.locator(".session-item.active").getAttribute("data-session-id");
    await page.evaluate((id) => window.codeclaw.invoke("agents.list", { session: id }), id);
    const original = fs.readFileSync(checkpoint);
    const python = resolveEnginePython({ packaged: false, projectRoot: repository, env });
    holder = spawn(python, [path.join(repository, "tests/fixtures/checkpoint_holder.py"), checkpoint], { cwd: repository, env, windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
    lines = readline.createInterface({ input: holder.stdout }); let stderr = ""; holder.stderr.on("data", (chunk) => stderr += chunk);
    const ready = once(lines, "line"), earlyExit = new Promise((_, reject) => holder.once("exit", (code) => reject(new Error("Holder exited " + code + ": " + stderr))));
    assert.equal((await bounded(Promise.race([ready, earlyExit]), "checkpoint holder readiness"))[0], "OWNED");
    const draft = "退出前保留的草稿 α 😀"; await input.fill(draft);
    await app.evaluate(({ app }) => app.quit());
    await page.getByRole("alert").filter({ hasText: "关闭引擎失败，窗口和草稿已保留" }).waitFor();
    assert.equal(app.process().exitCode, null); assert.equal(await input.inputValue(), draft); assert.deepEqual(fs.readFileSync(checkpoint), original);
    assert.equal(JSON.parse(fs.readFileSync(path.join(historyRoot(project, data), "index.json"))).sessions.shutdown.draft, draft);
    await release();
    const exited = once(app.process(), "exit"); await app.evaluate(({ app }) => app.quit()); await bounded(exited, "shutdown retry");
    assert(JSON.parse(fs.readFileSync(checkpoint)).revision > JSON.parse(original).revision);
    await launch(); assert.equal(await page.getByRole("textbox", { name: "任务输入" }).inputValue(), draft);
    assert.equal(requests, 1); assert.deepEqual(errors, []);
    console.log("Shutdown E2E passed: actual Windows checkpoint sharing denial blocks engine shutdown and app quit, original checkpoint/window/draft survive, release/retry saves final checkpoint, restart restores draft without repeating the model request");
  } finally {
    await release();
    if (app) {
      try { await bounded(app.close(), "isolated app cleanup"); }
      catch (error) { await app.evaluate(({ app }) => app.exit(1)).catch(() => {}); throw error; }
    }
    server.closeAllConnections(); server.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
