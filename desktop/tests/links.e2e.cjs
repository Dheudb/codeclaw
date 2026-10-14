const { historyRoot } = require("../electron/history-storage.cjs");
const { _electron: electron } = require("playwright");
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), assert = require("node:assert/strict");
const { atomicJSON } = require("../electron/catalog.cjs");
const root = path.resolve(__dirname, "..");
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "codeclaw-links-e2e-"));
const project = path.join(fixture, "project"), data = path.join(fixture, "profile"), config = path.join(fixture, "config");
for (const directory of [project, data, config]) fs.mkdirSync(directory);
const history = path.join(project, ".codeclaw/sessions/links.json");
atomicJSON(history, { messages: [{ role: "user", content: "查看来源链接" }, { role: "assistant", content: "Sources:\n- [Chemistry paper](https://arxiv.org/abs/2506.17238)\n- [HTTP source](http://example.org/paper?from=search#abstract)\n- https://example.org/bare\n- [Unsafe](javascript:alert(1))\n- ![Image](https://example.org/image.png)" }], metadata: {} });
const original = fs.readFileSync(history);
atomicJSON(path.join(data, "workbench.json"), { projects: [project], workspace: [{ project, resumeId: "links", active: true }] });
let app, page;
async function opened(count) {
  for (let i = 0; i < 50; i++) {
    const urls = await app.evaluate(() => globalThis.linkOpens);
    if (urls.length === count) return urls;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  assert.fail("External link did not reach the browser opener");
}
async function main() {
  try {
    const env = { ...process.env, USERPROFILE: path.join(fixture, "home"), HOME: path.join(fixture, "home"), CODECLAW_USER_DATA: data, CODECLAW_CONFIG_DIR: config, CODECLAW_TEST_HEADLESS: "1" };
    for (const key of ["ELECTRON_RUN_AS_NODE", "CODECLAW_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "CODECLAW_MODEL_PROVIDER"]) delete env[key];
    app = await electron.launch({ executablePath: process.env.CODECLAW_TEST_EXECUTABLE, args: process.env.CODECLAW_TEST_EXECUTABLE ? [] : [root], env, timeout: 30000 });
    page = await app.firstWindow(); page.setDefaultTimeout(30000);
    const errors = []; page.on("pageerror", (error) => errors.push(error.message));
    await page.locator(".session-item.active").waitFor();
    await app.evaluate(({ shell }) => {
      globalThis.linkOpens = [];
      shell.openExternal = async (url) => {
        if (globalThis.failLinkOpen) throw new Error("fixture browser unavailable");
        globalThis.linkOpens.push(url);
      };
    });
    const initialURL = page.url();
    await page.getByRole("link", { name: "Chemistry paper", exact: true }).click();
    assert.deepEqual(await opened(1), ["https://arxiv.org/abs/2506.17238"]);
    await page.getByRole("link", { name: "HTTP source", exact: true }).focus();
    await page.keyboard.press("Enter");
    assert.equal((await opened(2))[1], "http://example.org/paper?from=search#abstract");
    await page.getByRole("link", { name: "https://example.org/bare", exact: true }).click({ button: "middle" });
    assert.equal((await opened(3))[2], "https://example.org/bare");
    assert.equal(await page.getByRole("link", { name: "Unsafe", exact: true }).count(), 0);
    assert.equal(await page.locator(".markdown img").count(), 0);
    for (const url of ["javascript:alert(1)", "file:///C:/Windows/system.ini", "data:text/html,test", "https://user:password@example.org/"]) {
      assert.equal(await page.evaluate(async (url) => {
        try { await window.codeclaw.invoke("external.open", { url }); return true; } catch { return false; }
      }, url), false);
    }
    await app.evaluate(() => { globalThis.failLinkOpen = true; });
    await page.getByRole("link", { name: "Chemistry paper", exact: true }).click();
    await page.getByRole("alert").filter({ hasText: "无法打开链接" }).waitFor();
    assert.equal(page.url(), initialURL);
    assert.equal(await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().length), 1);
    assert.deepEqual(fs.readFileSync(path.join(historyRoot(project, data), "sessions/links.json")), original, "Existing transcripts must remain unchanged");
    assert.deepEqual(errors, []);
    console.log("Link E2E passed: saved Sources links, HTTP/HTTPS, bare URLs, mouse/keyboard/middle click, opener failures, protocol validation and no app navigation");
  } finally { if (app) await app.close().catch(() => {}); }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
