const { historyRoot } = require("../electron/history-storage.cjs");
const { _electron: electron } = require("playwright");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const assert = require("node:assert/strict");
const { atomicJSON, projectKey, sessionKey } = require("../electron/catalog.cjs");
const root = path.resolve(__dirname, "..");
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "xuloom-archives-e2e-"));
const project = path.join(fixture, "当前项目"), archivedProject = path.join(fixture, "归档项目");
const third = path.join(fixture, "另一个归档项目"), missing = path.join(fixture, "已移动目录");
const data = path.join(fixture, "profile"), config = path.join(fixture, "config");
for (const directory of [project, archivedProject, third, data, config]) fs.mkdirSync(directory);
const legacyHistory = (directory, id) => path.join(directory, ".xuloom/sessions", id + ".json");
const history = (directory, id) => path.join(historyRoot(directory, data), "sessions", id + ".json");
for (const [directory, id, title] of [[project, "active", "当前会话"], [project, "one", "同名归档"], [project, "two", "同名归档"],
  [archivedProject, "child", "项目内归档会话"], [archivedProject, "visible", "项目内普通会话"], [third, "ordinary", "另一个项目会话"]])
  atomicJSON(legacyHistory(directory, id), { messages: [{ role: "user", content: title }, { role: "assistant", content: "保留的历史回复" }], metadata: {} });
const indexPath = (directory) => path.join(historyRoot(directory, data), "index.json");
atomicJSON(path.join(project, ".xuloom/session-index.json"), { version: 1, sessions: { active: { draft: "未发送的内容" }, one: { archived: true, pinned: true, draft: "归档草稿" }, two: { archived: true } } });
atomicJSON(path.join(archivedProject, ".xuloom/session-index.json"), { version: 1, sessions: { child: { archived: true } } });
const workbench = path.join(data, "workbench.json");
atomicJSON(workbench, { projects: [project, archivedProject, third, missing], activeConversation: { project, resumeId: "active" },
  sidebar: { showArchived: true, sort: "recent" }, projectMeta: Object.fromEntries([archivedProject, third, missing].map((directory) => [projectKey(directory), { archived: true, open: false, collapsed: true }])) });
atomicJSON(path.join(config, "config.json"), { model_provider: "openai", openai_model: "archive-fixture", openai_api_key: "fixture-key", openai_base_url: "http://127.0.0.1:9/v1" });
const env = { ...process.env, USERPROFILE: path.join(fixture, "home"), HOME: path.join(fixture, "home"), XULOOM_USER_DATA: data, XULOOM_CONFIG_DIR: config, XULOOM_TEST_HEADLESS: "1" };
for (const key of ["ELECTRON_RUN_AS_NODE", "XULOOM_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "XULOOM_MODEL_PROVIDER"]) delete env[key];
const options = { executablePath: process.env.XULOOM_TEST_EXECUTABLE, args: process.env.XULOOM_TEST_EXECUTABLE ? [] : [root], env, timeout: 30000 };
let app, page;
const errors = [];
async function launch() {
  app = await electron.launch(options); page = await app.firstWindow(); page.setDefaultTimeout(45000);
  page.on("pageerror", (error) => errors.push(error.message));
  await page.waitForFunction(() => document.querySelector(".new-task")?.disabled === false);
}
async function openArchives() {
  await page.getByRole("button", { name: "设置与模型", exact: true }).click();
  await page.getByRole("tab", { name: "已归档会话", exact: true }).click();
  await page.waitForFunction(() => document.querySelector(".archive-manager")?.getAttribute("aria-busy") === "false");
}
const archiveRow = (directory, id) => page.locator(`[data-archived-session="${sessionKey(directory, id)}"]`);
const sidebarRow = (directory, id) => page.locator(`[data-session-id="${sessionKey(directory, id)}"]`);
const readIndex = (directory) => JSON.parse(fs.readFileSync(indexPath(directory), "utf8")).sessions;
async function main() {
  try {
    await launch();
    assert.equal(await page.getByLabel("显示已归档", { exact: true }).count(), 0);
    assert.equal(await page.locator(".session-item").count(), 1, "Legacy visibility flag cannot expose archives");
    assert.equal(await page.getByRole("textbox", { name: "任务输入" }).inputValue(), "未发送的内容");
    const original = fs.readFileSync(history(project, "one"));
    const refused = await page.evaluate(async (id) => {
      try { await window.xuloom.invoke("session.select", { id }); return false; } catch { return true; }
    }, sessionKey(project, "one"));
    assert(refused, "Archived conversations cannot be opened before restoring");
    await openArchives();
    assert.equal(await page.locator("[data-archived-session]").count(), 3);
    assert.equal(await page.locator("[data-archived-project]").count(), 3);
    await page.getByLabel("搜索归档记录", { exact: true }).fill("同名归档");
    assert.equal(await page.locator("[data-archived-session]").count(), 2);
    await page.getByLabel("搜索归档记录", { exact: true }).fill("不存在的名称");
    await page.getByText("没有匹配的归档记录", { exact: true }).waitFor();
    await page.getByLabel("搜索归档记录", { exact: true }).fill("");
    // A failed restore remains visible and retryable. Slow replies cannot cause duplicate writes.
    await app.evaluate(({ ipcMain }, failingId) => {
      const original = ipcMain._invokeHandlers.get("workbench:invoke");
      let failed = false; globalThis.archiveRestoreCalls = [];
      ipcMain.removeHandler("workbench:invoke");
      ipcMain.handle("workbench:invoke", async (...args) => {
        if (args[1] === "archive.restore") {
          globalThis.archiveRestoreCalls.push(args[2]);
          if (!failed && args[2].id === failingId) { failed = true; throw new Error("模拟恢复失败，请重试"); }
          const result = await original(...args);
          await new Promise((resolve) => setTimeout(resolve, 700)); return result;
        }
        return original(...args);
      });
    }, sessionKey(project, "one"));
    await archiveRow(project, "one").getByRole("button", { name: "恢复会话 同名归档", exact: true }).click();
    await page.getByRole("alert").filter({ hasText: "模拟恢复失败" }).waitFor();
    assert(readIndex(project).one.archived);
    assert.equal(await sidebarRow(project, "one").count(), 0);
    await archiveRow(project, "one").getByRole("button", { name: "恢复会话 同名归档", exact: true }).click();
    await page.getByRole("status").filter({ hasText: "已恢复会话：同名归档" }).waitFor();
    assert(!readIndex(project).one.archived && readIndex(project).two.archived);
    assert(readIndex(project).one.pinned);
    assert.equal(readIndex(project).one.draft, "归档草稿");
    assert.deepEqual(fs.readFileSync(history(project, "one")), original);
    assert.equal(await page.locator(".session-item.active").getAttribute("data-session-id"), sessionKey(project, "active"));
    assert.equal(await page.getByRole("textbox", { name: "任务输入" }).inputValue(), "未发送的内容");
    // A missing project reports an error without silently restoring its metadata.
    await page.getByRole("button", { name: "恢复项目 已移动目录", exact: true }).click();
    await page.getByRole("alert").waitFor();
    assert(JSON.parse(fs.readFileSync(workbench, "utf8")).projectMeta[projectKey(missing)].archived);
    await archiveRow(archivedProject, "child").getByRole("button", { name: "恢复会话 项目内归档会话", exact: true }).click();
    await page.getByRole("status").filter({ hasText: "已恢复会话：项目内归档会话" }).waitFor();
    assert(!readIndex(archivedProject).child.archived);
    assert(!JSON.parse(fs.readFileSync(workbench, "utf8")).projectMeta[projectKey(archivedProject)].archived);
    assert.equal(await sidebarRow(archivedProject, "child").count(), 1);
    await page.getByRole("button", { name: "恢复项目 另一个归档项目", exact: true }).click();
    await page.getByRole("status").filter({ hasText: "已恢复项目" }).waitFor();
    assert.equal(await sidebarRow(third, "ordinary").count(), 1);
    await page.setViewportSize({ width: 1050, height: 700 });
    const bounds = await page.getByRole("dialog", { name: "工作台设置" }).boundingBox();
    assert(bounds.x >= 0 && bounds.y >= 0 && bounds.x + bounds.width <= 1050 && bounds.y + bounds.height <= 700);
    const screenshot = await app.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage(undefined, { stayHidden: true, stayAwake: true })).toPNG().toString("base64"));
    fs.mkdirSync(path.join(root, "test-results"), { recursive: true });
    fs.writeFileSync(path.join(root, "test-results/archives-settings.png"), Buffer.from(screenshot, "base64"));
    await page.keyboard.press("Escape");
    await page.getByRole("dialog", { name: "工作台设置" }).waitFor({ state: "hidden" });
    for (let i = 0; i < 3; i++) {
      await openArchives();
      assert.equal(await archiveRow(project, "one").count(), 0);
      assert.equal(await archiveRow(project, "two").count(), 1);
      await page.getByRole("button", { name: "关闭设置", exact: true }).click();
    }
    await app.close(); await launch();
    assert.equal(await sidebarRow(project, "two").count(), 0);
    assert.equal(await sidebarRow(project, "one").count(), 1);
    await openArchives();
    await page.getByLabel("搜索归档记录", { exact: true }).fill("同名归档");
    assert.equal(await page.locator("[data-archived-session]").count(), 1);
    await archiveRow(project, "two").getByRole("button", { name: "恢复会话 同名归档", exact: true }).click();
    await page.getByRole("status").filter({ hasText: "已恢复会话" }).waitFor();
    await page.getByLabel("搜索归档记录", { exact: true }).fill("");
    assert.equal(await page.locator("[data-archived-session]").count(), 0);
    await page.getByRole("button", { name: "关闭设置", exact: true }).click();
    // Archive the current conversation through its real menu, then restore without opening it.
    await page.getByRole("button", { name: "会话操作 当前会话", exact: true }).click();
    await page.getByRole("menuitem", { name: "归档会话", exact: true }).click();
    await sidebarRow(project, "active").waitFor({ state: "hidden" });
    await openArchives();
    await archiveRow(project, "active").getByRole("button", { name: "恢复会话 当前会话", exact: true }).click();
    await page.getByRole("status").filter({ hasText: "已恢复会话：当前会话" }).waitFor();
    await page.getByRole("button", { name: "关闭设置", exact: true }).click();
    await sidebarRow(project, "active").locator(".session-select").click();
    await page.waitForFunction(() => !document.querySelector(".new-task").disabled);
    assert.equal(await page.getByRole("textbox", { name: "任务输入" }).inputValue(), "未发送的内容");
    assert.deepEqual(errors, []);
    console.log("Archive E2E passed: settings-only archive list, legacy flag ignored, search, duplicate titles, retry, missing directory, project restore, drafts, repeated closing and restart");
  } finally { if (app) await app.close().catch(() => {}); }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
