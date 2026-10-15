const { _electron: electron } = require("playwright");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), assert = require("node:assert/strict");
const { SessionCatalog, atomicJSON } = require("../electron/catalog.cjs");
const root = path.resolve(__dirname, "..");
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "codeclaw-appearance-"));
const data = path.join(fixture, "data"), config = path.join(fixture, "config"), project = path.join(fixture, "research-workspace");
for (const directory of [data, config, project]) fs.mkdirSync(directory, { recursive: true });
let settings = { projects: [project], projectMeta: { [project.toLowerCase()]: { name: "Research workspace" } } };
const catalog = new SessionCatalog(() => settings, (value) => { settings = value; }, data);
const row = catalog.createStandalone(path.join(fixture));
catalog.update(row, { title: "让工作台变得更清晰" });
const file = path.join(row.project, "README.md");
fs.writeFileSync(file, "# 工作台设计\n\n清晰的正文、柔和的层次、专注的工作流。\n", "utf8");
const answer = "可以先从三个地方调整，让注意力回到正在处理的任务。\n\n**聊天区域**\n\n统一正文和辅助信息的字号，用轻量气泡区分你的消息；回复保持干净，执行过程收在上方。\n\n**工具面板**\n\nSkills、子代理和文件保持在同一个侧栏，按需展开。减少重复边框，让名称、说明和操作各有层次。";
const fileChange = { path: file, operation: "edit", added: 1, removed: 1, statsAvailable: true, updatedAt: Date.now(), diff: "@@ -1,3 +1,3 @@\n # 工作台设计\n \n-旧版界面\n+清晰的正文、柔和的层次、专注的工作流。\n" };
atomicJSON(catalog.file(row.project, row.sessionId), { messages: [
  { role: "user", _uuid: "u1", content: [{ type: "text", text: "帮我看看这个工作台，怎么让它更清晰、更舒服？" }] },
  { role: "assistant", content: [{ type: "text", text: answer }] },
  { role: "user", _uuid: "u2", content: [{ type: "text", text: "很好，把这些设计原则整理到 README 里。" }] },
  { role: "assistant", content: [{ type: "text", text: "已整理到 [README.md](README.md)，包含排版、配色与面板层次三部分。后续调整可以按同一套规范保持一致。" }] }
], metadata: { file_changes: [fileChange], turn_traces: [
  { id: "t1", userMessageId: "u1", status: "completed", elapsedMs: 18000, startedAt: Date.now() - 18000,
    steps: [], answer },
  { id: "t2", userMessageId: "u2", status: "completed", elapsedMs: 7000, startedAt: Date.now() - 7000,
    steps: [{ kind: "text", text: "我会把配色与排版整理为可复用的设计规范。" },
      { kind: "tool", tool: { id: "edit-readme", name: "file_edit_tool", status: "completed", input: { file_path: file }, result: "README updated" } }],
    files: [fileChange], answer: "已整理到 [README.md](README.md)，包含排版、配色与面板层次三部分。后续调整可以按同一套规范保持一致。" }
] } });
settings.activeConversation = { project: row.project, resumeId: row.sessionId, scope: "standalone" };
atomicJSON(path.join(data, "workbench.json"), settings);
atomicJSON(path.join(config, "config.json"), { model_provider: "openai", openai_model: "design-preview", openai_api_key: "fixture", openai_base_url: "http://127.0.0.1:9/v1" });
const errors = [];
let app, page;
async function capture(name) {
  const destination = path.join(root, "test-results", name + ".png");
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  // Wake the hidden window's compositor before capturing its updated frame.
  const png = await app.evaluate(async ({ BrowserWindow }) => {
    const web = BrowserWindow.getAllWindows()[0].webContents;
    await web.capturePage(undefined, { stayHidden: true, stayAwake: true });
    await new Promise((resolve) => setTimeout(resolve, 100));
    return (await web.capturePage(undefined, { stayHidden: true, stayAwake: true })).toPNG().toString("base64");
  });
  fs.writeFileSync(destination, Buffer.from(png, "base64"));
}
async function checkLayout() {
  const result = await page.evaluate(() => {
    const rect = (selector) => document.querySelector(selector).getBoundingClientRect().toJSON();
    return { viewport: innerWidth, scrollWidth: document.documentElement.scrollWidth,
      input: rect(".composer"), panel: rect(".inspector"),
      tabs: [...document.querySelectorAll(".panel-tabs button")].map((element) => element.getBoundingClientRect().toJSON()),
      body: getComputedStyle(document.querySelector(".message.assistant .markdown")).fontSize };
  });
  assert(result.scrollWidth <= result.viewport, "App must not overflow horizontally");
  assert(result.input.right <= result.panel.left + 1, "Composer must stay within conversation");
  assert(result.tabs.every((tab) => tab.left >= result.panel.left && tab.right <= result.panel.right), "All panel tabs must fit");
  assert.equal(result.body, "15px");
}
(async () => {
  try {
    const env = { ...process.env, CODECLAW_DATA_DIR: data, CODECLAW_USER_DATA: data, CODECLAW_CONFIG_DIR: config,
      HOME: path.join(fixture, "home"), USERPROFILE: path.join(fixture, "home"), CODECLAW_TEST_HEADLESS: "1" };
    for (const name of ["ELECTRON_RUN_AS_NODE", "CODECLAW_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "CODECLAW_MODEL_PROVIDER"]) delete env[name];
    app = await electron.launch({ executablePath: process.env.CODECLAW_TEST_EXECUTABLE, args: process.env.CODECLAW_TEST_EXECUTABLE ? [] : [root], env });
    page = await app.firstWindow(); page.setDefaultTimeout(30000); page.on("pageerror", (error) => errors.push(error.message));
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.waitForFunction(() => document.querySelector(".new-task")?.disabled === false);
    await page.getByText("聊天区域", { exact: true }).waitFor();
    await checkLayout();
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("DOM.enable"); await cdp.send("CSS.enable");
    const doc = await cdp.send("DOM.getDocument");
    const { nodeId } = await cdp.send("DOM.querySelector", { nodeId: doc.root.nodeId, selector: ".message.assistant .markdown p" });
    console.log("Rendered fonts:", JSON.stringify(await cdp.send("CSS.getPlatformFontsForNode", { nodeId })));
    await page.evaluate(() => document.querySelector(".conversation").scrollTop = 0);
    await capture("appearance-chat-wide");
    await page.getByRole("textbox", { name: "任务输入" }).fill("$");
    await page.getByRole("listbox", { name: "可用技能" }).waitFor();
    await page.getByRole("listbox", { name: "可用技能" }).getByRole("option").first().waitFor();
    await capture("appearance-skill-picker");
    await page.keyboard.press("Escape");
    await page.getByRole("textbox", { name: "任务输入" }).fill("");
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1100, 850));
    await checkLayout();
    await capture("appearance-chat-compact");
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1440, 940));
    await page.getByRole("button", { name: "设置与模型", exact: true }).click();
    await page.getByRole("dialog").waitFor();
    await capture("appearance-settings");
    await page.getByRole("button", { name: "关闭设置", exact: true }).click();
    await page.locator(".new-task").click();
    await page.getByRole("heading", { name: "开始一个新会话。", exact: true }).waitFor();
    await capture("appearance-welcome");
    assert.deepEqual(errors, []);
    console.log("Appearance E2E passed: 1440/1100 layouts, readable type, five panel tabs, Skill picker, settings and welcome; screenshots saved");
  } finally { if (app) await app.close().catch(() => {}); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
