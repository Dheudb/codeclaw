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
let app, page, pending;
const server = http.createServer((req, res) => {
  let raw = ""; req.on("data", (part) => { raw += part; }); req.on("end", () => {
    requests.push(JSON.parse(raw)); pending = res;
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
    atomicJSON(path.join(config, "config.json"), { model_provider: "openai", openai_model: "collaboration-fixture", openai_api_key: "fixture", openai_base_url: `http://127.0.0.1:${server.address().port}/v1` });
    await launch();
    await page.getByRole("textbox", { name: "任务输入" }).fill("Hold for stop test");
    await page.getByRole("button", { name: "发送任务", exact: true }).click();
    await page.getByRole("button", { name: "停止任务", exact: true }).waitFor();
    assert.equal(await picker().isDisabled(), true);
    const rejected = await page.evaluate(async (id) => { try {
      await window.xuloom.invoke("collaboration.set", { session: id, mode: "solo" }); return "unexpected success";
    } catch (error) { return String(error); } }, originalId);
    assert(rejected.includes("当前任务"), rejected);
    await page.getByRole("button", { name: "停止任务", exact: true }).click();
    await picker().waitFor(); await page.waitForFunction(() => !document.querySelector(".collaboration-picker").disabled);
    assert(requests.length > 0 && JSON.stringify(requests[0]).includes("# Collaboration: ultra"));
    await choose("solo");
    assert.deepEqual(errors, []);
    console.log("Collaboration E2E passed: model-free save, session isolation, restart restore, narrow layout, busy rejection and stop; hidden local fixture only");
  } finally {
    pending?.end(); if (app) await app.close().catch(() => {});
    server.closeAllConnections(); server.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
