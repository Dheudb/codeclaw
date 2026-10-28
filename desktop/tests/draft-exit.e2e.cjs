// Real storage denial must preserve the renderer and its unsent text on close/reload/quit.
const { _electron: electron } = require("playwright");
const { spawn } = require("node:child_process");
const { once } = require("node:events");
const readline = require("node:readline");
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), assert = require("node:assert/strict");
const { atomicJSON } = require("../electron/catalog.cjs");
const { historyRoot } = require("../electron/history-storage.cjs");
const { resolveEnginePython } = require("../electron/runtime.cjs");
const root = path.resolve(__dirname, ".."), repository = path.dirname(root);
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "xuloom-draft-exit-"));
const project = path.join(fixture, "project"), data = path.join(fixture, "data"), config = path.join(fixture, "config"), home = path.join(fixture, "home");
for (const directory of [project, data, config, home]) fs.mkdirSync(directory);
const env = { ...process.env, XULOOM_USER_DATA: data, XULOOM_DATA_DIR: data, XULOOM_CONFIG_DIR: config,
  USERPROFILE: home, HOME: home, XULOOM_TEST_HEADLESS: "1", PYTHONPATH: repository, XULOOM_SKILLS_PATHS: "" };
for (const name of ["ELECTRON_RUN_AS_NODE", "XULOOM_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "XULOOM_MODEL_PROVIDER"]) delete env[name];
atomicJSON(path.join(config, "config.json"), { model_provider: "openai", openai_model: "", openai_api_key: "", openai_base_url: "http://127.0.0.1:9/v1" });
atomicJSON(path.join(historyRoot(project, data), "sessions/draft.json"), { messages: [], metadata: {} });
atomicJSON(path.join(data, "workbench.json"), { projects: [project], workspace: [{ project, resumeId: "draft", active: true }] });
const options = { executablePath: process.env.XULOOM_TEST_EXECUTABLE, args: process.env.XULOOM_TEST_EXECUTABLE ? [] : [root], env, timeout: 30000 };
let app, page, holder, lines;
const failures = [];
async function bounded(promise, label) {
  let timer;
  try { return await Promise.race([promise, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("Timed out: " + label)), 20000); })]); }
  finally { clearTimeout(timer); }
}
async function launch() {
  app = await electron.launch(options); page = await app.firstWindow(); page.setDefaultTimeout(45000);
  page.on("pageerror", (error) => failures.push(error.message));
  await page.locator(".session-item.active").waitFor();
  await page.waitForFunction(() => !document.querySelector(".new-task")?.disabled);
}
async function release() {
  if (holder && holder.exitCode === null) {
    const exited = once(holder, "exit"); holder.stdin.end("release\n"); await exited;
  }
  lines?.close(); holder = undefined;
}
async function main() {
  try {
    await launch();
    const python = resolveEnginePython({ packaged: false, projectRoot: repository, env });
    holder = spawn(python, [path.join(repository, "tests/fixtures/index_holder.py"), project], { cwd: repository, env,
      windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
    lines = readline.createInterface({ input: holder.stdout });
    let stderr = ""; holder.stderr.on("data", (chunk) => stderr += chunk);
    const ready = once(lines, "line");
    const ended = new Promise((_, reject) => holder.once("exit", (code) => reject(new Error("Index holder exited: " + code + " " + stderr))));
    assert.equal((await bounded(Promise.race([ready, ended]), "index holder readiness"))[0], "OWNED");
    const input = page.getByRole("textbox", { name: "任务输入" });
    const original = "未保存草稿 α 😀";
    await input.click();
    await input.fill(original);
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].close());
    await page.getByText(/草稿保存失败，窗口已保留/).waitFor();
    assert.equal(app.process().exitCode, null); assert.equal(await input.inputValue(), original);
    console.log("Close denied with the original draft intact");
    await page.evaluate(() => { window.__draftExitNonce = "same-document"; });
    await bounded(app.evaluate(async ({ BrowserWindow }) => {
      const web = BrowserWindow.getAllWindows()[0].webContents;
      let timer;
      try {
        const blocked = new Promise((resolve, reject) => { web.once("will-prevent-unload", resolve); timer = setTimeout(() => reject(new Error("Reload was not blocked")), 15000); });
        web.reload(); await blocked;
      } finally { clearTimeout(timer); }
    }), "reload protection");
    assert.equal(await page.evaluate(() => window.__draftExitNonce), "same-document");
    assert.equal(await input.inputValue(), original);
    await release();
    const expected = "恢复存储后立即退出的最新草稿 😀";
    await input.fill(expected);
    const closed = once(app.process(), "exit");
    await app.evaluate(({ app }) => { app.quit(); });
    await bounded(closed, "normal quit after storage recovered");
    const index = JSON.parse(fs.readFileSync(path.join(historyRoot(project, data), "index.json"), "utf8"));
    assert.equal(index.sessions.draft.draft, expected);
    await launch();
    assert.equal(await page.getByRole("textbox", { name: "任务输入" }).inputValue(), expected);
    assert.deepEqual(failures, []);
    console.log("Draft exit E2E passed: real index denial blocks close/reload, original text remains, immediate retry saves latest draft and restart restores it; no model connection configured");
  } catch (error) {
    console.error("Fixture:", fixture);
    console.error(error);
    if (page && !page.isClosed()) console.error(await page.locator("body").innerText().catch(() => "unavailable"));
    throw error;
  } finally {
    await release();
    if (app) {
      try { await bounded(app.close(), "isolated test app cleanup"); }
      catch (error) {
        await app.evaluate(({ app }) => { app.exit(1); }).catch(() => {});
        throw error;
      }
    }
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
