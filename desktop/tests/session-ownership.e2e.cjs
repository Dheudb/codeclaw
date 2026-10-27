// A real Python writer competes with the actual hidden Electron application.
const { _electron: electron } = require("playwright");
const { spawn } = require("node:child_process");
const { once } = require("node:events");
const readline = require("node:readline");
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), http = require("node:http"), assert = require("node:assert/strict");
const { atomicJSON, sessionKey } = require("../electron/catalog.cjs");
const { historyRoot } = require("../electron/history-storage.cjs");
const root = path.resolve(__dirname, "..");
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "codeclaw-ownership-e2e-"));
const project = path.join(fixture, "workspace"), other = path.join(fixture, "target"), data = path.join(fixture, "data"), config = path.join(fixture, "config"), home = path.join(fixture, "home");
for (const directory of [project, other, data, config, home]) fs.mkdirSync(directory);
const env = { ...process.env, USERPROFILE: home, HOME: home, CODECLAW_DATA_DIR: data, CODECLAW_USER_DATA: data, CODECLAW_CONFIG_DIR: config, CODECLAW_TEST_HEADLESS: "1", CODECLAW_SKILLS_PATHS: "", PYTHONPATH: path.dirname(root) };
for (const key of ["ELECTRON_RUN_AS_NODE", "CODECLAW_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "CODECLAW_MODEL_PROVIDER"]) delete env[key];
let app, page, owner, lines, requests = 0;
const server = http.createServer((req, res) => {
  let raw = ""; req.on("data", (chunk) => raw += chunk);
  req.on("end", () => {
    if (!JSON.parse(raw).stream) {
      res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({ choices: [{ message: { role: "assistant", content: "fixture memory" } }] }));
      return;
    }
    requests++;
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    res.end('data: {"id":"fixture","choices":[{"index":0,"delta":{"role":"assistant","content":"RECOVERED_WRITER"},"finish_reason":null}]}\n\ndata: {"id":"fixture","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n');
  });
});
async function main() {
  try {
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    atomicJSON(path.join(config, "config.json"), { model_provider: "openai", openai_model: "fixture", openai_api_key: "fixture", openai_base_url: `http://127.0.0.1:${server.address().port}/v1` });
    const python = process.env.CODECLAW_PYTHON || path.join(path.dirname(root), ".venv-mcp/Scripts/python.exe");
    owner = spawn(python, [path.join(root, "../tests/fixtures/session_writer.py"), project, "hold", "shared"], { env, cwd: path.dirname(root), windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
    let stderr = ""; owner.stderr.on("data", (chunk) => stderr += chunk);
    lines = readline.createInterface({ input: owner.stdout });
    assert.equal((await once(lines, "line"))[0], "OWNED", stderr);
    atomicJSON(path.join(data, "workbench.json"), { projects: [project, other], workspace: [{ project, resumeId: "shared", active: true }] });
    const history = path.join(historyRoot(project, data), "sessions/shared.json"), before = fs.readFileSync(history);
    app = await electron.launch({ executablePath: process.env.CODECLAW_TEST_EXECUTABLE, args: process.env.CODECLAW_TEST_EXECUTABLE ? [] : [root], env, timeout: 30000 });
    page = await app.firstWindow(); page.setDefaultTimeout(45000);
    const errors = []; page.on("pageerror", (error) => errors.push(error.message));
    await page.locator(".session-item.active").waitFor();
    const id = sessionKey(project, "shared");
    assert.equal(await page.locator(".user-message").count() > 0 || (await page.locator("body").innerText()).includes("first"), true, "Owned history remains readable");
    await page.evaluate(({ id }) => window.codeclaw.invoke("prompt", { session: id, text: "Must not request the model", skills: [], mode: "normal" }), { id });
    await page.getByText(/Session is in use by another writer/).waitFor();
    assert.equal(requests, 0);
    for (const [method, params] of [["session.fork", { id }], ["session.move", { id, path: other }], ["session.delete", { id, confirm: true }]]) {
      const error = await page.evaluate(async ({ method, params }) => { try { await window.codeclaw.invoke(method, params); return ""; } catch (error) { return String(error); } }, { method, params });
      assert.match(error, /Session is in use by another writer/, method);
      assert.deepEqual(fs.readFileSync(history), before, method + " must preserve the owner's checkpoint");
    }
    const exited = once(owner, "exit"); owner.stdin.end("release\n"); await exited;
    // Reopen after failed deletion disposed the read-only engine transport.
    await page.locator('.session-item[data-session-id="' + id + '"] .session-select').click();
    const input = page.getByRole("textbox", { name: "任务输入" });
    await input.fill("Continue after the writer released"); await input.press("Enter");
    await page.getByText("RECOVERED_WRITER", { exact: true }).waitFor();
    await page.waitForFunction(() => !document.querySelector(".send-button.stop"));
    assert.equal(requests, 1);
    assert.equal(errors.length, 0, errors.join("\n"));
    assert(JSON.parse(fs.readFileSync(history)).revision > 1);
    console.log("Session ownership E2E passed: read-only history, no model request before ownership, blocked fork/move/delete, unchanged checkpoint, writer release and successful continuation");
  } catch (error) {
    console.error("Requests:", requests);
    if (page) console.error("Fixture UI:", await page.locator("body").innerText().catch(() => "unavailable"));
    throw error;
  } finally {
    if (app) await app.close().catch(() => {});
    if (owner && owner.exitCode === null) { const exited = once(owner, "exit"); owner.stdin.end("release\n"); await exited; }
    if (lines) lines.close();
    server.closeAllConnections(); server.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
