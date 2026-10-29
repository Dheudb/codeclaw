const { _electron: electron } = require("playwright");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), assert = require("node:assert/strict");
const { SessionCatalog, atomicJSON } = require("../electron/catalog.cjs");
const root = path.resolve(__dirname, "..");
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "xuloom-message-actions-"));
const data = path.join(fixture, "data"), config = path.join(fixture, "config"), project = path.join(fixture, "project");
for (const dir of [data, config, project]) fs.mkdirSync(dir, { recursive: true });
let settings = { projects: [project] };
const catalog = new SessionCatalog(() => settings, (value) => { settings = value; }, data);
const row = catalog.create(project); catalog.update(row, { title: "Branch origin" });
const prompt = "USER_PROMPT_SELECTABLE_TEXT **literal**\nSecond line with `notes.md`.";
const code = 'def bubble_sort(arr):\n    """冒泡排序 < & >"""\n    return sorted(arr)\n\nprint(bubble_sort([3, 1, 2]))\n';
const plainCode = 'notes.md\n';
const formula = String.raw`\boxed{e^{i\pi} + 1 = 0}`;
const inlineFormula = String.raw`\theta = \pi`;
const answer = "ANSWER_ONE **bold** and `code`.\n\n```python\n" + code + "```\n\n```\n" + plainCode + "```\n\n$$" + formula + "$$\n\n当 $" + inlineFormula + "$ 时成立。\n\n$$\n" + String.raw`\begin{pmatrix}1 & 2\\3 & 4\end{pmatrix}` + "\n$$\n\n错误公式 $\\frac{$ 不应破坏回复。\n\n```text\n$do_not_render$\n```";
const trace = (i, text) => ({ id: "t" + i, userMessageId: "u" + i, status: "completed", elapsedMs: 1000, steps: [], answer: text });
const original = { messages: [
  { role: "user", _uuid: "u1", content: prompt },
  { role: "assistant", content: [{ type: "text", text: "Inspecting" }, { type: "tool_use", id: "read1", name: "file_read_tool", input: { file_path: "notes.md" } }] },
  { role: "user", content: [{ type: "tool_result", tool_use_id: "read1", content: "evidence" }] },
  { role: "assistant", content: answer },
  { role: "user", _uuid: "u2", content: "SECOND_PROMPT" }, { role: "assistant", content: "ANSWER_TWO" },
  { role: "user", _uuid: "u3", content: "FUTURE_PROMPT" }, { role: "assistant", content: "FUTURE_REPLY" },
], metadata: { turn_traces: [trace(1, answer), trace(2, "ANSWER_TWO"), trace(3, "FUTURE_REPLY")], collaboration_mode: "ultra", subagents: [] } };
const originalPath = catalog.file(project, row.sessionId); atomicJSON(originalPath, original);
settings.activeConversation = { project, resumeId: row.sessionId, scope: "project" };
atomicJSON(path.join(data, "workbench.json"), settings);
// All actions must work without an API connection.
atomicJSON(path.join(config, "config.json"), { model_provider: "openai", openai_model: "", openai_api_key: "", openai_base_url: "http://127.0.0.1:9/v1" });
let app, page;
const errors = [];
async function launch() {
  const env = { ...process.env, XULOOM_DATA_DIR: data, XULOOM_USER_DATA: data, XULOOM_CONFIG_DIR: config,
    USERPROFILE: path.join(fixture, "home"), HOME: path.join(fixture, "home"), XULOOM_TEST_HEADLESS: "1" };
  for (const key of ["ELECTRON_RUN_AS_NODE", "XULOOM_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "XULOOM_MODEL_PROVIDER"]) delete env[key];
  app = await electron.launch({ executablePath: process.env.XULOOM_TEST_EXECUTABLE, args: process.env.XULOOM_TEST_EXECUTABLE ? [] : [root], env });
  page = await app.firstWindow(); page.setDefaultTimeout(30000); page.on("pageerror", (error) => errors.push(error.message));
  await page.waitForFunction(() => document.querySelector(".new-task")?.disabled === false);
  // Exercise the real renderer -> IPC -> Electron clipboard boundary without
  // overwriting or reading the user's clipboard in an automated test.
  await app.evaluate(({ clipboard }) => { globalThis.testCopiedText = []; clipboard.writeText = (text) => {
    if (globalThis.failTestCopy) throw new Error("Fixture clipboard unavailable");
    globalThis.testCopiedText.push(text);
  }; });
}
async function selectWithMouse(start, end) {
  const points = await page.locator(".message.user .user-prompt-text").first().evaluate((el, [start, end]) => {
    const position = (offset) => { const range = document.createRange(); range.setStart(el.firstChild, offset); range.setEnd(el.firstChild, offset + 1); const rect = range.getBoundingClientRect(); return { x: rect.left + .5, y: rect.top + rect.height / 2 }; };
    return [position(start), position(end)];
  }, [start, end]);
  await page.mouse.move(points[0].x, points[0].y); await page.mouse.down();
  await page.mouse.move(points[1].x, points[1].y, { steps: 12 }); await page.mouse.up();
  return page.evaluate(() => window.getSelection()?.toString() || "");
}
(async () => {
  try {
    await launch();
    assert.equal(await page.getByRole("button", { name: "复制回复", exact: true }).count(), 3);
    assert.equal(await page.getByRole("button", { name: "复制提示词", exact: true }).count(), 3);
    const user = page.getByRole("article", { name: "你的消息", exact: true }).first();
    await user.scrollIntoViewIfNeeded(); await user.hover();
    assert.equal(await user.locator(".user-prompt-text").innerText(), prompt, "Prompts preserve literal Markdown and line breaks");
    await user.getByRole("button", { name: "复制提示词", exact: true }).click();
    const reply = page.getByRole("article", { name: "Xuloom 回复", exact: true }).first();
    await reply.getByRole("button", { name: "复制回复", exact: true }).click();
    assert.deepEqual(await app.evaluate(() => globalThis.testCopiedText), [prompt, answer]);
    const blocks = reply.locator(".markdown-code-block");
    assert.equal(await blocks.count(), 3);
    assert.equal(await blocks.first().locator(".code-block-language").innerText(), "python");
    assert.equal(await blocks.first().locator("pre").textContent(), code);
    assert.equal(await blocks.nth(1).locator("pre button").count(), 0, "A filename inside a code block remains code, not a file link");
    assert.equal(await blocks.last().locator("pre").textContent(), "$do_not_render$\n");
    assert.equal(await blocks.last().locator(".math-formula").count(), 0);
    assert.equal(await reply.locator(".math-display .katex").count(), 2);
    assert.equal(await reply.locator(".math-inline .katex").count(), 1);
    assert.equal(await reply.locator(".math-fallback").textContent(), String.raw`\frac{`);
    await blocks.first().getByRole("button", { name: "复制代码", exact: true }).click();
    await blocks.first().getByRole("status").filter({ hasText: "已复制" }).waitFor();
    await blocks.nth(1).getByRole("button", { name: "复制代码", exact: true }).focus();
    await page.keyboard.press("Enter");
    await page.waitForFunction(() => document.querySelectorAll(".code-block-toolbar [role=status]").length === 2);
    assert.deepEqual(await app.evaluate(() => globalThis.testCopiedText), [prompt, answer, code, plainCode], "Code copy preserves indentation, Unicode and line breaks, without Markdown fences or toolbar text");
    await app.evaluate(() => { globalThis.failTestCopy = true; });
    await blocks.first().getByRole("button", { name: "复制代码", exact: true }).click();
    await blocks.first().getByRole("alert").filter({ hasText: "复制失败" }).waitFor();
    await app.evaluate(() => { globalThis.failTestCopy = false; });
    await blocks.first().getByRole("button", { name: "复制代码", exact: true }).click();
    await blocks.first().getByRole("alert").waitFor({ state: "detached" });
    assert.equal((await app.evaluate(() => globalThis.testCopiedText)).at(-1), code);
    const formulas = reply.locator(".math-formula");
    await formulas.first().getByRole("button", { name: "复制公式 LaTeX" }).click();
    await formulas.first().getByRole("status").waitFor();
    assert.equal((await app.evaluate(() => globalThis.testCopiedText)).at(-1), formula);
    await formulas.nth(1).getByRole("button", { name: "复制公式 LaTeX" }).focus();
    await page.keyboard.press("Enter");
    await formulas.nth(1).getByRole("status").waitFor();
    assert.equal((await app.evaluate(() => globalThis.testCopiedText)).at(-1), inlineFormula);
    await app.evaluate(() => { globalThis.failTestCopy = true; });
    await formulas.first().getByRole("button", { name: "复制公式 LaTeX" }).click();
    await formulas.first().getByRole("alert").waitFor();
    await app.evaluate(() => { globalThis.failTestCopy = false; });
    await formulas.first().getByRole("button", { name: "复制公式 LaTeX" }).click();
    await formulas.first().getByRole("alert").waitFor({ state: "detached" });
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.evaluate(() => document.fonts.check('16px KaTeX_Main')), true);
    fs.mkdirSync(path.join(root, "test-results"), { recursive: true });
    await page.screenshot({ path: path.join(root, "test-results/code-copy.png") });
    await user.scrollIntoViewIfNeeded();
    const firstSelection = await selectWithMouse(0, 10);
    assert(firstSelection.includes("USER"), firstSelection);
    const secondSelection = await selectWithMouse(2, 14);
    assert(secondSelection.length >= 8, "Dragging inside a prior selection starts a fresh text selection");
    const rejectedDrag = await user.locator(".message-text").evaluate((el) => !el.dispatchEvent(new DragEvent("dragstart", { bubbles: true, cancelable: true, dataTransfer: new DataTransfer() })));
    assert.equal(rejectedDrag, true);
    assert.equal(await page.locator(".session-item.active").getAttribute("draggable"), "true", "Sidebar session drag remains available");
    await page.getByRole("textbox", { name: "任务输入" }).fill("ORIGINAL_DRAFT_KEEP");
    await reply.getByRole("button", { name: "从这里分支", exact: true }).click();
    await page.waitForFunction(() => document.querySelectorAll(".message.assistant").length === 1 && document.querySelector(".new-task")?.disabled === false);
    assert.equal(await page.locator(".message.user").count(), 1);
    assert(!(await page.locator(".transcript").innerText()).includes("FUTURE"));
    assert.equal(await page.getByLabel("协作档位", { exact: true }).inputValue(), "ultra");
    const state = await page.evaluate(() => window.xuloom.invoke("catalog.refresh"));
    const rows = state.projects.flatMap((p) => p.sessions);
    const branch = rows.find((item) => item.id !== row.id);
    assert(branch, "Branch must appear as a separate conversation");
    const saved = JSON.parse(fs.readFileSync(catalog.file(project, branch.sessionId)));
    assert.equal(saved.messages.length, 4, "Full tool call/result pair is retained at the selected reply boundary");
    assert.equal(saved.metadata.turn_traces.length, 1);
    assert.equal(saved.metadata.subagents, undefined);
    assert.deepEqual(JSON.parse(fs.readFileSync(originalPath)), original, "The original checkpoint is unchanged");
    assert.equal(rows.find((item) => item.id === row.id).draft, "ORIGINAL_DRAFT_KEEP");
    await app.close(); app = null;
    await launch();
    assert.equal(await page.locator(".message.assistant").count(), 1, "Reply boundary survives restart");
    await page.locator(".session-item").filter({ hasText: "Branch origin" }).filter({ hasNotText: "分支" }).click();
    await page.waitForFunction(() => document.querySelectorAll(".message.assistant").length === 3);
    assert.equal(await page.getByRole("textbox", { name: "任务输入" }).inputValue(), "ORIGINAL_DRAFT_KEEP");
    assert.deepEqual(errors, []);
    console.log("Message actions E2E passed: exact reply/prompt and fenced code copy through IPC, Unicode/indentation, plain code, keyboard, clipboard failure/retry, native mouse selection/reselection, no text dragging, retained sidebar dragging, reply-boundary fork, intact source/draft, tool pairs, no API and restart");
  } finally { if (app) await app.close().catch(() => {}); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
