const { _electron: electron } = require("playwright");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const assert = require("node:assert/strict");
const { atomicJSON, projectKey } = require("../electron/catalog.cjs");
const { defaultWorkspace } = require("../electron/workspace.cjs");
const root = path.resolve(__dirname, "..");
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "codeclaw-workspace-e2e-"));
const data = path.join(fixture, "profile"), config = path.join(fixture, "config");
const custom = path.join(fixture, "自定义默认目录"), explicit = path.join(fixture, "explicit-project");
for (const dir of [data, config, explicit]) fs.mkdirSync(dir);
const workbench = path.join(data, "workbench.json");
// No projects, workspace selection, or preconfigured default directory.
atomicJSON(workbench, { projects: [], activeConversation: null });
atomicJSON(path.join(config, "config.json"), { model_provider: "openai", openai_model: "", openai_api_key: "", openai_base_url: "http://127.0.0.1:9/v1" });
const env = { ...process.env, CODECLAW_USER_DATA: data, CODECLAW_CONFIG_DIR: config, CODECLAW_TEST_HEADLESS: "1",
  LOCALAPPDATA: path.join(fixture, "os-data"), XDG_DATA_HOME: path.join(fixture, "xdg-data") };
delete env.ELECTRON_RUN_AS_NODE; delete env.CODECLAW_DEV_URL;
for (const key of ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "CODECLAW_MODEL_PROVIDER"]) delete env[key];
const options = { executablePath: process.env.CODECLAW_TEST_EXECUTABLE,
  args: process.env.CODECLAW_TEST_EXECUTABLE ? [] : [root], env, timeout: 30000 };
let app, page;
const errors = [];
const assertPath = (actual, expected) => assert.equal(projectKey(actual), projectKey(expected));
const readSettings = () => JSON.parse(fs.readFileSync(workbench, "utf8"));
async function launch() {
  app = await electron.launch(options); page = await app.firstWindow(); page.setDefaultTimeout(45000);
  page.on("pageerror", (error) => errors.push(error.message));
  await page.waitForFunction(() => !document.querySelector(".new-task")?.disabled);
}
async function main() {
  try {
    await launch();
    // Stub only the OS home for macOS, keeping all created files in the fixture.
    if (process.platform === "darwin") await app.evaluate(({ app }, home) => app.setPath("home", home), fixture);
    const home = await app.evaluate(({ app }) => app.getPath("home"));
    const defaultDirectory = defaultWorkspace({ home, env });
    assertPath((await page.evaluate(() => window.codeclaw.invoke("workspace.get"))).defaultDirectory, defaultDirectory);
    assert(!fs.existsSync(defaultDirectory), "Startup does not create an unsolicited project");
    // New session must never request an OS folder chooser.
    await app.evaluate(({ dialog }) => { dialog.showOpenDialog = async () => { throw new Error("Unexpected folder chooser"); }; });
    await page.locator(".new-task").click();
    await page.locator(".session-item.active").waitFor();
    assertPath(await page.locator(".project-context span").getAttribute("title"), fs.realpathSync(defaultDirectory));
    const defaultId = await page.locator(".session-item.active").getAttribute("data-session-id");
    assert(fs.existsSync(path.join(defaultDirectory, ".codeclaw/sessions", defaultId.split(":")[1] + ".json")));
    await page.getByRole("button", { name: "设置与模型", exact: true }).click();
    assertPath(await page.getByLabel("默认工作目录", { exact: true }).inputValue(), defaultDirectory);
    // Directory configuration works independently of the model configuration.
    await page.getByLabel("模型 ID", { exact: true }).fill("");
    assert(await page.getByRole("button", { name: "保存配置", exact: true }).isDisabled());
    await page.getByLabel("默认工作目录", { exact: true }).fill("relative-directory");
    await page.getByRole("button", { name: "保存默认目录", exact: true }).click();
    await page.getByRole("alert").filter({ hasText: "绝对路径" }).waitFor();
    assert.equal(readSettings().defaultDirectory, undefined);
    const originalConfig = fs.readFileSync(path.join(config, "config.json"), "utf8");
    await page.getByLabel("默认工作目录", { exact: true }).fill(custom);
    await page.getByRole("button", { name: "保存默认目录", exact: true }).click();
    await page.getByRole("status").filter({ hasText: "默认目录已保存" }).waitFor();
    assertPath(readSettings().defaultDirectory, fs.realpathSync(custom));
    assert.equal(fs.readFileSync(path.join(config, "config.json"), "utf8"), originalConfig);
    const screenshot = await app.evaluate(async ({ BrowserWindow }) =>
      (await BrowserWindow.getAllWindows()[0].webContents.capturePage(undefined, { stayHidden: true, stayAwake: true })).toPNG().toString("base64"));
    fs.mkdirSync(path.join(root, "test-results"), { recursive: true });
    fs.writeFileSync(path.join(root, "test-results/workspace-settings.png"), Buffer.from(screenshot, "base64"));
    await page.getByRole("button", { name: "关闭设置", exact: true }).click();
    // Changing the default does not redirect a conversation in a selected project.
    await page.locator(".new-task").click();
    await page.waitForFunction(() => !document.querySelector(".new-task").disabled);
    assert.equal(await page.locator(".session-item.active").getAttribute("data-session-id"), defaultId);
    await app.evaluate(({ dialog }, directory) => {
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [directory] });
    }, explicit);
    await page.getByRole("button", { name: "打开项目", exact: true }).click();
    await page.locator(`[data-project="${explicit.replace(/\\/g, "\\\\")}"] .session-item.active`).waitFor();
    await page.locator(".new-task").click();
    await page.waitForFunction(() => !document.querySelector(".new-task").disabled);
    assertPath(await page.locator(".project-context span").getAttribute("title"), fs.realpathSync(explicit));
    for (const directory of [defaultDirectory, explicit]) await page.evaluate((directory) => window.codeclaw.invoke("project.remove", { path: directory, confirm: true }), directory);
    // Restore renderer after backend-only project removal; this also checks persistent empty selection.
    await app.close(); await launch();
    assert.equal(await page.locator(".session-item").count(), 0);
    await page.getByRole("button", { name: "开始新会话", exact: true }).click();
    await page.locator(".session-item.active").waitFor();
    assertPath(await page.locator(".project-context span").getAttribute("title"), fs.realpathSync(custom));
    assert(fs.existsSync(path.join(defaultDirectory, ".codeclaw/sessions", defaultId.split(":")[1] + ".json")), "Default directory changes preserve old histories");

    // Seed one archived row through the real backend, keeping the current session active.
    const archived = "archived-fixture";
    atomicJSON(path.join(custom, ".codeclaw/sessions", archived + ".json"), { messages: [{ role: "user", content: "归档测试记录" }], metadata: {} });
    const snapshot = await page.evaluate(() => window.codeclaw.invoke("catalog.refresh"));
    const archivedRow = snapshot.projects.flatMap((p) => p.sessions).find((row) => row.sessionId === archived);
    await page.evaluate((id) => window.codeclaw.invoke("session.update", { id, patch: { archived: true } }), archivedRow.id);
    let checkbox = page.getByLabel("显示已归档", { exact: true });
    // Delay replies outside the main-process write queue to force reversed acknowledgements.
    await app.evaluate(({ ipcMain }) => {
      const original = ipcMain._invokeHandlers.get("workbench:invoke");
      let calls = 0;
      ipcMain.removeHandler("workbench:invoke");
      ipcMain.handle("workbench:invoke", async (...args) => {
        const ordinal = args[1] === "catalog.view" ? ++calls : 0;
        const result = await original(...args);
        if (ordinal) await new Promise((resolve) => setTimeout(resolve, ordinal % 2 ? 900 : 100));
        return result;
      });
    });
    await checkbox.check(); await checkbox.uncheck();
    await page.getByRole("button", { name: "刷新会话列表", exact: true }).click();
    await page.waitForTimeout(1200);
    assert.equal(await checkbox.isChecked(), false);
    assert.equal(await page.getByText("归档测试记录", { exact: true }).count(), 0);
    for (let i = 0; i < 5; i++) {
      await page.locator(".archive-toggle").click();
      assert.equal(await checkbox.isChecked(), true);
      await checkbox.focus(); await page.keyboard.press("Space");
      assert.equal(await checkbox.isChecked(), false);
    }
    await page.waitForTimeout(1200);
    assert.equal(await checkbox.isChecked(), false);
    assert.equal(readSettings().sidebar.showArchived, false);
    await checkbox.check(); await page.waitForTimeout(1200);
    await page.getByText("归档测试记录", { exact: true }).waitFor();
    await app.close(); await launch();
    checkbox = page.getByLabel("显示已归档", { exact: true });
    assert.equal(await checkbox.isChecked(), true);
    await checkbox.uncheck(); await page.waitForTimeout(150);
    assert.equal(readSettings().sidebar.showArchived, false);
    await app.close(); await launch();
    checkbox = page.getByLabel("显示已归档", { exact: true });
    assert.equal(await checkbox.isChecked(), false);
    assert.equal(await page.getByText("归档测试记录", { exact: true }).count(), 0);
    assertPath(await page.locator(".project-context span").getAttribute("title"), fs.realpathSync(custom));
    assert.deepEqual(errors, []);
    console.log("Workspace E2E passed: per-user default, no folder chooser, independent save, explicit project, restart, rapid archive toggles and reversed replies");
  } finally { if (app) await app.close().catch(() => {}); }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
