const { _electron: electron } = require("playwright");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), assert = require("node:assert/strict"), http = require("node:http");
const { SessionCatalog, atomicJSON } = require("../electron/catalog.cjs");
const root = path.resolve(__dirname, "..");
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "xuloom-collaboration-"));
const data = path.join(fixture, "data"), config = path.join(fixture, "config");
for (const dir of [data, config]) fs.mkdirSync(dir, { recursive: true });
let settings = {};
const catalog = new SessionCatalog(() => settings, (value) => { settings = value; }, data);
const first = catalog.createStandalone(fixture);
catalog.update(first, { title: "Collaboration A" });
settings.activeConversation = { project: first.project, resumeId: first.sessionId, scope: "standalone" };
atomicJSON(path.join(data, "workbench.json"), settings);
const requests = [], errors = [];
const requestWaiters = new Set();
function nextRequest() {
  return new Promise((resolve, reject) => {
    const receive = () => { clearTimeout(timer); requestWaiters.delete(receive); resolve(); };
    const timer = setTimeout(() => { requestWaiters.delete(receive); reject(new Error("Local request did not arrive")); }, 30000);
    timer.unref(); requestWaiters.add(receive);
  });
}
let app, page, pending;
const server = http.createServer((req, res) => {
  let raw = ""; req.on("data", (part) => { raw += part; }); req.on("end", () => {
    requests.push(JSON.parse(raw)); pending = res;
    for (const receive of [...requestWaiters]) receive();
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    res.write('data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n');
  });
});
async function launch() {
  const env = { ...process.env, XULOOM_DATA_DIR: data, XULOOM_USER_DATA: data, XULOOM_CONFIG_DIR: config,
    HOME: path.join(fixture, "home"), USERPROFILE: path.join(fixture, "home"), XULOOM_TEST_HEADLESS: "1" };
  for (const key of ["ELECTRON_RUN_AS_NODE", "XULOOM_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "XULOOM_MODEL_PROVIDER", "XULOOM_COLLABORATION_MODE"]) delete env[key];
  app = await electron.launch({ executablePath: process.env.XULOOM_TEST_EXECUTABLE, args: process.env.XULOOM_TEST_EXECUTABLE ? [] : [root], env });
  page = await app.firstWindow(); page.setDefaultTimeout(30000); page.on("pageerror", (error) => errors.push(error.message));
  await page.waitForFunction(() => document.querySelector(".new-task")?.disabled === false);
}
const picker = () => page.getByLabel("协作档位", { exact: true });
async function choose(mode) {
  await picker().selectOption(mode);
  await page.waitForFunction((mode) => { const el = document.querySelector(".collaboration-picker"); return el.value === mode && !el.disabled; }, mode);
}
(async () => {
  try {
    await launch();
    assert.equal(await picker().inputValue(), "auto");
    // Local preference does not depend on an API being configured.
    await choose("solo");
    assert.equal(await page.getByLabel("访问权限", { exact: true }).inputValue(), "ask");
    assert.equal(JSON.parse(fs.readFileSync(catalog.file(first.project, first.sessionId))).metadata.collaboration_mode, "solo");
    await app.close(); app = null;
    await launch(); assert.equal(await picker().inputValue(), "solo");
    await choose("ultra");
    const originalId = await page.locator(".session-item.active").getAttribute("data-session-id");
    await page.locator(".new-task").click();
    await page.waitForFunction((id) => document.querySelector(".session-item.active")?.getAttribute("data-session-id") !== id && document.querySelector(".new-task")?.disabled === false, originalId);
    assert.equal(await picker().inputValue(), "auto", "A new conversation has independent default policy");
    await page.locator(".session-item").filter({ hasText: "Collaboration A" }).click();
    await page.waitForFunction(() => document.querySelector(".collaboration-picker")?.value === "ultra");
    await page.getByRole("button", { name: "展开能力面板", exact: true }).click();
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1100, 850));
    assert.equal(await page.locator(".composer-toolbar").evaluate((el) => el.scrollWidth > el.clientWidth), false, "Options and stop/send stay accessible in a narrow conversation");
    await app.close(); app = null;
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    const base = `http://127.0.0.1:${server.address().port}/v1`;
    atomicJSON(path.join(config, "config.json"), { model_provider: "openai", active_model_connection: "fixture", model_connections: [{
      id: "fixture", name: "Fixture API", provider: "openai", model: "collaboration-fixture", models: ["collaboration-fixture", "unknown-fixture"],
      api_key: "fixture", base_url: base, model_reasoning: { "collaboration-fixture": { adapter: "deepseek", efforts: ["low", "high", "max"],
        canDisable: true, thinking: "default", effort: "high", budgetTokens: 2048 } },
    }] });
    await launch();
    await page.getByLabel("Thinking 模式", { exact: true }).selectOption("disabled");
    await page.waitForFunction(() => document.querySelector(".thinking-picker")?.value === "disabled" && !document.querySelector(".thinking-picker").disabled);
    assert(await page.getByLabel("推理强度", { exact: true }).isDisabled());
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1100, 850));
    assert.equal(await page.locator(".composer-toolbar").evaluate(el => el.scrollWidth > el.clientWidth), false, "Thinking controls must fit a narrow composer");
    const firstRequest = nextRequest();
    await page.getByRole("textbox", { name: "任务输入" }).fill("Hold for stop test");
    await page.getByRole("button", { name: "发送任务", exact: true }).click();
    await page.getByRole("button", { name: "停止任务", exact: true }).waitFor();
    await firstRequest;
    assert.equal(await picker().isDisabled(), true);
    const rejected = await page.evaluate(async (id) => { try {
      await window.xuloom.invoke("collaboration.set", { session: id, mode: "solo" }); return "unexpected success";
    } catch (error) { return String(error); } }, originalId);
    assert(rejected.includes("当前任务"), rejected);
    await page.getByRole("button", { name: "停止任务", exact: true }).click();
    await picker().waitFor(); await page.waitForFunction(() => !document.querySelector(".collaboration-picker").disabled);
    assert(requests.length > 0 && JSON.stringify(requests[0]).includes("# Collaboration: ultra"));
    assert(JSON.stringify(requests[0].messages).includes("Proactive delegation is enabled"));
    const system = requests[0].messages.filter((message) => message.role === "system").map((message) => message.content).join("\n");
    assert.equal(system.split("# Collaboration: ultra").length - 1, 1);
    assert(system.indexOf("# Collaboration: ultra") > system.indexOf("# Tool-Specific Rules"));
    assert(system.includes("instead of doing every branch yourself"));
    assert(JSON.stringify(requests[0].messages).includes("run_in_background=true"));
    assert(!JSON.stringify(requests[0].messages).includes("run_in_background=false when the current answer requires"));
    assert(requests[0].tools.some((tool) => tool.function.name === "agent_tool" && tool.function.description.includes("bounded independent assignment")));
    assert(requests[0].tools.some((tool) => tool.function.name === "spawn_agent" && !tool.function.parameters.properties.action));
    assert.deepEqual(requests[0].thinking, { type: "disabled" });
    assert.equal(requests[0].reasoning_effort, undefined, "Disabled thinking must omit conflicting effort");
    await page.getByLabel("Thinking 模式", { exact: true }).selectOption("enabled");
    await page.getByLabel("推理强度", { exact: true }).selectOption("max");
    await page.waitForFunction(() => document.querySelector(".reasoning-picker")?.value === "max" && !document.querySelector(".reasoning-picker").disabled);
    const enabledRequest = nextRequest();
    await page.getByRole("textbox", { name: "任务输入" }).fill("Enabled thinking wire check");
    await page.getByRole("button", { name: "发送任务", exact: true }).click();
    await page.waitForFunction(() => document.querySelector('[aria-label="停止任务"]'));
    await enabledRequest;
    await page.getByRole("button", { name: "停止任务", exact: true }).click();
    await page.waitForFunction(() => !document.querySelector(".collaboration-picker").disabled);
    assert.deepEqual(requests.at(-1).thinking, { type: "enabled" });
    assert.equal(requests.at(-1).reasoning_effort, "max");
    await page.getByLabel("选择模型", { exact: true }).selectOption(JSON.stringify(["fixture", "unknown-fixture"]));
    await page.getByLabel("Thinking 模式", { exact: true }).waitFor({ state: "detached" });
    assert.equal(await page.getByLabel("推理强度", { exact: true }).count(), 0, "Unknown models must not inherit another model's controls");
    await page.getByRole("button", { name: "设置与模型", exact: true }).click();
    await page.getByLabel("推理参数格式", { exact: true }).selectOption("qwen-local");
    await page.getByRole("button", { name: "保存配置", exact: true }).click();
    await page.getByRole("dialog", { name: "工作台设置", exact: true }).waitFor({ state: "hidden" });
    await page.getByLabel("Thinking 模式", { exact: true }).waitFor();
    assert.equal(await page.getByLabel("推理强度", { exact: true }).count(), 0);
    const saved = JSON.parse(fs.readFileSync(path.join(config, "config.json"), "utf8"));
    assert.equal(saved.model_connections[0].model_reasoning["unknown-fixture"].adapter, "qwen-local");
    assert.equal(saved.model_connections[0].model_reasoning["collaboration-fixture"].effort, "max");
    await choose("solo");
    assert.deepEqual(errors, []);
    console.log("Collaboration/reasoning E2E passed: mode persistence, native spawn schema, model-specific settings, SDK wire parameters, unknown-model reset, narrow layout and stop; hidden local fixture only");
  } finally {
    pending?.end(); if (app) await app.close().catch(() => {});
    server.closeAllConnections(); server.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
