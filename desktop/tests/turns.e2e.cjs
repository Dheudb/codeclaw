const { _electron: electron } = require("playwright");
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), http = require("node:http"), assert = require("node:assert/strict");
const { atomicJSON } = require("../electron/catalog.cjs");
const root = path.resolve(__dirname, "..");
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "codeclaw-turns-e2e-"));
const project = path.join(fixture, "project"), data = path.join(fixture, "profile"), config = path.join(fixture, "config");
for (const directory of [project, data, config]) fs.mkdirSync(directory);
fs.writeFileSync(path.join(project, "proof.txt"), "TURN_OUTPUT_PROOF");
atomicJSON(path.join(data, "workbench.json"), { projects: [], activeConversation: null, defaultDirectory: project });
let app, page;
const errors = [];
const server = http.createServer((req, res) => {
  let raw = ""; req.on("data", (chunk) => raw += chunk);
  req.on("end", async () => {
    const body = JSON.parse(raw);
    if (!body.stream) { res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({ choices: [{ message: { role: "assistant", content: "memory" } }] })); return; }
    const task = [...body.messages].reverse().find((m) => m.role === "user" && typeof m.content === "string" && m.content.startsWith("TRACE_"));
    const after = task ? body.messages.slice(body.messages.indexOf(task) + 1) : [];
    const calls = after.filter((m) => m.role === "tool").length;
    if (task?.content === "TRACE_ERROR" && calls) { res.writeHead(401, { "Content-Type": "application/json" }).end(JSON.stringify({ error: { message: "fixture error", type: "authentication_error" } })); return; }
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    const chunk = (delta, finish_reason = null) => res.write("data: " + JSON.stringify({ id: "fixture", object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason }] }) + "\n\n");
    const multi = task?.content === "TRACE_MULTI" && calls < 2;
    const cancel = task?.content === "TRACE_CANCEL" && !calls;
    const error = task?.content === "TRACE_ERROR" && !calls;
    if (multi || cancel || error) {
      chunk({ role: "assistant", content: cancel ? "先请求批准，再写文件。" : calls ? "再检查一次，确认结果。" : "我先检查文件内容。" });
      await new Promise((resolve) => setTimeout(resolve, multi && !calls ? 1400 : 30));
      chunk({ tool_calls: [{ index: 0, id: `call_${calls}`, type: "function", function: { name: cancel ? "file_write_tool" : "file_read_tool", arguments: JSON.stringify({ file_path: path.join(project, cancel ? "cancelled.txt" : "proof.txt"), ...(cancel ? { content: "never write" } : {}) }) } }] });
      chunk({}, "tool_calls");
    } else {
      chunk({ role: "assistant", content: task?.content === "TRACE_SECOND" ? "第二轮的独立回答。" : "文件检查完成，这是最终答案。" });
      chunk({}, "stop");
    }
    res.end("data: [DONE]\n\n");
  });
});
async function launch() {
  const env = { ...process.env, CODECLAW_USER_DATA: data, CODECLAW_CONFIG_DIR: config, CODECLAW_TEST_HEADLESS: "1" };
  for (const key of ["ELECTRON_RUN_AS_NODE", "CODECLAW_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "CODECLAW_MODEL_PROVIDER"]) delete env[key];
  app = await electron.launch({ executablePath: process.env.CODECLAW_TEST_EXECUTABLE, args: process.env.CODECLAW_TEST_EXECUTABLE ? [] : [root], env, timeout: 30000 });
  page = await app.firstWindow(); page.setDefaultTimeout(60000); page.on("pageerror", (error) => errors.push(error.message));
  await page.waitForFunction(() => document.querySelector(".new-task")?.disabled === false);
}
async function send(text) { await page.getByRole("textbox", { name: "任务输入" }).fill(text); await page.getByRole("button", { name: "发送任务", exact: true }).click(); }
async function finished() { await page.waitForFunction(() => !document.querySelector(".send-button.stop")); }
async function main() {
  try {
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    atomicJSON(path.join(config, "config.json"), { model_provider: "openai", openai_model: "turn-fixture", openai_api_key: "fixture-key", openai_base_url: `http://127.0.0.1:${server.address().port}/v1` });
    await launch(); await page.locator(".new-task").click(); await page.locator(".session-item.active").waitFor();
    await send("TRACE_MULTI");
    await page.locator('.turn-process[data-turn-status="running"][open]').waitFor();
    await finished();
    const first = page.locator(".turn-process").first();
    assert.equal(await first.getAttribute("open"), null);
    assert.match(await first.locator(":scope > summary").innerText(), /用时 [1-9]\d*秒/);
    await first.locator(":scope > summary").click();
    assert.equal(await first.locator(".turn-commentary").count(), 2);
    assert((await first.innerText()).includes("我先检查文件内容。"));
    assert((await first.innerText()).includes("再检查一次，确认结果。"));
    assert.equal(await first.locator(".turn-tool").count(), 2);
    await first.locator(".turn-tool > summary").first().click();
    assert((await first.innerText()).includes("TURN_OUTPUT_PROOF"));
    assert(!(await first.innerText()).includes("这是最终答案"));
    await send("TRACE_SECOND"); await finished();
    assert.equal(await page.locator(".turn-process").count(), 2);
    assert.equal(await page.locator(".turn-process").nth(1).locator(".turn-tool").count(), 0);
    await send("TRACE_CANCEL"); await page.getByRole("button", { name: "允许一次", exact: true }).waitFor();
    await page.getByRole("button", { name: "停止任务", exact: true }).click(); await finished();
    assert.equal(await page.locator(".turn-process").nth(2).getAttribute("data-turn-status"), "cancelled");
    assert.equal(await page.locator(".turn-process").nth(2).getAttribute("open"), null);
    assert(!fs.existsSync(path.join(project, "cancelled.txt")));
    await send("TRACE_ERROR"); await finished();
    assert.equal(await page.locator(".turn-process").nth(3).getAttribute("data-turn-status"), "error");
    const summary = await first.locator(":scope > summary").innerText();
    await app.close(); await launch();
    assert.equal(await page.locator(".turn-process").count(), 4);
    assert.equal(await page.locator(".turn-process[open]").count(), 0);
    assert.equal(await page.locator(".turn-process").first().locator(":scope > summary").innerText(), summary);
    await page.locator(".turn-process").first().locator(":scope > summary").click();
    assert.equal(await page.locator(".turn-process").first().locator(".turn-commentary").count(), 2);
    assert.equal(await page.locator(".message.assistant").count(), 4);
    await page.locator(".message.assistant").first().scrollIntoViewIfNeeded();
    fs.mkdirSync(path.join(root, "test-results"), { recursive: true });
    await page.screenshot({ path: path.join(root, "test-results/turn-process.png") });
    assert.deepEqual(errors, []);
    console.log("Turn process E2E passed: live expanded timer, completed collapse, commentary/tool results retained, per-turn isolation, cancellation/error and restart persistence");
  } finally { if (app) await app.close().catch(() => {}); server.closeAllConnections(); server.close(); }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
