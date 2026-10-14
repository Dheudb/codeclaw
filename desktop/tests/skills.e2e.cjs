const { _electron: electron } = require("playwright");
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), http = require("node:http"), assert = require("node:assert/strict");
const { atomicJSON } = require("../electron/catalog.cjs");
const root = path.resolve(__dirname, "..");
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "codeclaw-skills-e2e-"));
const project = path.join(fixture, "project-a"), other = path.join(fixture, "project-b");
const data = path.join(fixture, "profile"), config = path.join(fixture, "config"), home = path.join(fixture, "home");
for (const directory of [project, other, data, config, home]) fs.mkdirSync(directory);
function skill(parent, name, description, body = "FIXTURE_" + name) {
  const directory = path.join(parent, name); fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(path.join(directory, "SKILL.md"), `---\nname: ${name}\ndescription: ${description}\n---\n\n${body}`, "utf8");
  return directory;
}
const local = path.join(project, ".codeclaw/skills");
skill(local, "project-check", "检查当前项目的真实目录技能", "PROJECT_BODY_PROOF");
skill(local, "code-review", "当前项目覆盖内置审查", "CUSTOM_REVIEW_BODY_PROOF");
const disabled = skill(local, "disabled-skill", "旧停用配置中的技能仍可直接选择");
const invalid = skill(local, "invalid-skill", "格式错误不应显示");
fs.writeFileSync(path.join(invalid, "SKILL.md"), "not frontmatter", "utf8");
const imported = skill(path.join(fixture, "mounted"), "imported-check", "挂载目录的技能");
skill(path.join(home, ".codeclaw/skills"), "global-check", "全局技能");
skill(path.join(other, ".agents/skills"), "other-project", "第二个项目独有技能");
atomicJSON(path.join(project, ".codeclaw/capabilities.json"), {
  skill_roots: [imported], disabled_skills: [process.platform === "win32" ? path.join(disabled, "SKILL.md").toLowerCase() : path.join(disabled, "SKILL.md")],
});
atomicJSON(path.join(data, "workbench.json"), { projects: [project, other], workspace: [], defaultDirectory: project });
const requests = [], errors = [];
let app, page;
const server = http.createServer((req, res) => {
  let raw = ""; req.on("data", (chunk) => raw += chunk);
  req.on("end", () => {
    const body = JSON.parse(raw);
    if (!body.stream) { res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({ choices: [{ message: { role: "assistant", content: "memory" } }] })); return; }
    requests.push(body);
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    for (const [delta, finish_reason] of [[{ role: "assistant", content: "SKILL_WIRING_VERIFIED" }, null], [{}, "stop"]]) {
      res.write("data: " + JSON.stringify({ id: "fixture", object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason }] }) + "\n\n");
    }
    res.end("data: [DONE]\n\n");
  });
});
async function launch() {
  const env = { ...process.env, USERPROFILE: path.join(fixture, "home"), HOME: path.join(fixture, "home"), CODECLAW_USER_DATA: data, CODECLAW_CONFIG_DIR: config, CODECLAW_TEST_HEADLESS: "1", USERPROFILE: home, HOME: home, CODECLAW_SKILLS_PATHS: "" };
  for (const key of ["ELECTRON_RUN_AS_NODE", "CODECLAW_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "CODECLAW_MODEL_PROVIDER"]) delete env[key];
  app = await electron.launch({ executablePath: process.env.CODECLAW_TEST_EXECUTABLE, args: process.env.CODECLAW_TEST_EXECUTABLE ? [] : [root], env, timeout: 30000 });
  page = await app.firstWindow(); page.setDefaultTimeout(30000);
  page.on("pageerror", (error) => errors.push(error.message));
  await page.waitForFunction(() => document.querySelector(".new-task")?.disabled === false);
}
async function picker(query = "$") {
  const input = page.getByRole("textbox", { name: "任务输入" });
  await input.fill(query); await input.focus();
  await page.getByRole("listbox", { name: "可用技能" }).waitFor();
  await page.waitForFunction(() => document.querySelector('[role="listbox"]')?.getAttribute("aria-busy") === "false");
  return input;
}
async function main() {
  try {
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    atomicJSON(path.join(config, "config.json"), { model_provider: "openai", openai_model: "skill-fixture", openai_api_key: "fixture-key", openai_base_url: `http://127.0.0.1:${server.address().port}/v1` });
    await launch(); await page.locator(".new-task").click(); await page.locator(".session-item.active").waitFor();
    const input = await picker();
    assert.equal(await page.locator("#skill-options").getByRole("option").count(), 7); // valid skills remain available despite legacy disabled flags
    assert.equal(await page.getByRole("switch", { name: /启用技能/ }).count(), 0);
    assert.equal(await page.locator("#skill-options").getByRole("option", { name: /disabled-skill/ }).count(), 1);
    assert.equal(await page.locator("#skill-options").getByRole("option", { name: /code-review/ }).count(), 1);
    for (const name of ["invalid-skill", "other-project"]) assert.equal(await page.locator("#skill-options").getByRole("option", { name: new RegExp(name) }).count(), 0);
    assert.match(await page.locator("#skill-options").getByRole("option", { name: /code-review/ }).innerText(), /项目/);
    await input.press("ArrowDown"); assert.equal(await page.locator("#skill-options").getByRole("option", { selected: true }).getAttribute("id"), "skill-option-1");
    await input.press("ArrowUp"); assert.equal(await page.locator("#skill-options").getByRole("option", { selected: true }).getAttribute("id"), "skill-option-0");
    // A native hidden capture avoids Playwright's visible-compositor screenshot wait.
    const shot = await app.evaluate(async ({ BrowserWindow }) => {
      const web = BrowserWindow.getAllWindows()[0].webContents;
      await web.capturePage(undefined, { stayHidden: true, stayAwake: true });
      await new Promise((resolve) => setTimeout(resolve, 100));
      return (await web.capturePage(undefined, { stayHidden: true, stayAwake: true })).toPNG().toString("base64");
    });
    fs.mkdirSync(path.join(root, "test-results"), { recursive: true });
    fs.writeFileSync(path.join(root, "test-results/skill-picker.png"), Buffer.from(shot, "base64"));
    await input.press("Escape"); assert.equal(await page.getByRole("listbox").count(), 0); assert.equal(await input.inputValue(), "$");
    await picker("$project-check"); await input.press("Enter");
    assert.equal(await input.inputValue(), ""); assert.equal(await page.locator(".selected-skills button").count(), 1); assert.equal(requests.length, 0);
    await picker("$code-review"); await input.press("Tab");
    assert.equal(await page.locator(".selected-skills button").count(), 2);
    await picker("$debug"); await page.locator("#skill-options").getByRole("option", { name: /debug-issue/ }).click();
    assert.equal(await page.locator(".selected-skills button").count(), 3);
    await picker("$project-check"); await input.press("Enter"); assert.equal(await page.locator(".selected-skills button").count(), 3, "Selecting twice must not duplicate a skill");
    // Search descriptions, unknown queries and amounts; Enter in an empty menu never submits a bogus task.
    await picker("$挂载"); assert.equal(await page.locator("#skill-options").getByRole("option").count(), 1);
    await picker("$does-not-exist"); assert.equal(await page.locator("#skill-options").getByRole("option").count(), 0);
    await input.press("Enter"); assert.equal(requests.length, 0);
    await input.fill("$100"); assert.equal(await page.getByRole("listbox").count(), 0);
    // Files copied after startup are found when the picker opens again.
    skill(local, "fresh-skill", "刚放进目录的技能");
    await picker("$fresh"); assert.equal(await page.locator("#skill-options").getByRole("option", { name: /fresh-skill/ }).count(), 1);
    // Selecting a mention before a task preserves the task rather than deleting everything.
    await input.fill("$imported-check 保留后面的任务");
    await input.press("Home"); for (let i = 0; i < 4; i++) await input.press("ArrowRight");
    await page.getByRole("listbox").waitFor();
    await page.waitForFunction(() => document.querySelector('[role="listbox"]')?.getAttribute("aria-busy") === "false");
    await page.locator("#skill-options").getByRole("option", { name: /imported-check/ }).click();
    assert.equal(await input.inputValue(), "保留后面的任务");
    await page.locator(".selected-skills button").filter({ hasText: "imported-check" }).click();
    await input.fill("验证技能挂载"); await input.press("Enter");
    await page.getByText("SKILL_WIRING_VERIFIED", { exact: true }).waitFor();
    await page.waitForFunction(() => !document.querySelector(".send-button.stop"));
    assert.equal(requests.length, 1);
    const prompt = requests[0].messages.find((message) => message.role === "user").content;
    for (const proof of ["PROJECT_BODY_PROOF", "CUSTOM_REVIEW_BODY_PROOF", "Skill: debug-issue"]) assert(prompt.includes(proof), "Missing full instructions: " + proof);
    const builtinBody = fs.readFileSync(path.join(root, "../codeclaw/skills/builtin/debug-issue/SKILL.md"), "utf8").split("---")[2].trim();
    assert(prompt.includes(builtinBody), "Bundled skill must inject its actual body");
    assert(!prompt.includes("Skill: imported-check"));
    const displayed = await page.locator(".message.user").innerText();
    assert(displayed.includes("验证技能挂载"));
    assert(!displayed.includes("BODY_PROOF") && !displayed.includes("<selected-skills>"), "Instructions must not leak into the displayed user text");
    await picker("$project-check");
    await page.getByRole("button", { name: "在项目 project-b 中新建会话", exact: true }).click();
    await page.waitForFunction((other) => document.querySelector(".project-context span")?.getAttribute("title")?.toLowerCase() === other.toLowerCase(), fs.realpathSync(other));
    assert.equal(await page.locator(".selected-skills button").count(), 0);
    await picker();
    assert.equal(await page.locator("#skill-options").getByRole("option", { name: /project-check/ }).count(), 0);
    assert.equal(await page.locator("#skill-options").getByRole("option", { name: /other-project/ }).count(), 1);
    assert.match(await page.locator("#skill-options").getByRole("option", { name: /code-review/ }).innerText(), /内置/);
    await app.close(); await launch();
    await page.locator(".session-item.active").waitFor(); await picker("$other");
    assert.equal(await page.locator("#skill-options").getByRole("option", { name: /other-project/ }).count(), 1);
    assert.deepEqual(errors, []);
    console.log("Skill E2E passed: actual directories, default availability with no switches, legacy flag compatibility, shadowing/invalid filters, keyboard/mouse/multiple selection, fresh files, caret/task preservation, full instructions reaching model, project isolation and restart");
  } finally { if (app) await app.close().catch(() => {}); server.closeAllConnections(); server.close(); }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
