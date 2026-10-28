const { _electron: electron } = require("playwright");
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), assert = require("node:assert/strict");
const { atomicJSON } = require("../electron/catalog.cjs");
const root = path.resolve(__dirname, ".."), fixture = fs.mkdtempSync(path.join(os.tmpdir(), "xuloom-html-e2e-"));
const project = path.join(fixture, "project"), data = path.join(fixture, "profile"), config = path.join(fixture, "config"), home = path.join(fixture, "home");
for (const directory of [project, data, config, home, path.join(project, "pages"), path.join(project, "assets")]) fs.mkdirSync(directory);
const document = (heading) => `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HTML preview fixture</title>
<link rel="stylesheet" href="../assets/style.css"><body><h1>${heading}</h1><p>SVG 动画与本地资源</p>
<svg viewBox="0 0 500 260" width="100%" aria-label="自行车动画"><rect width="500" height="260" fill="#e4f0f0"/>
<path d="M0 230H500" stroke="#9eaf78" stroke-width="8"/><g id="wheel"><circle cx="140" cy="170" r="55" fill="none" stroke="#40575e" stroke-width="5"/><path d="M85 170H195M140 115V225" stroke="#40575e" stroke-width="3"/></g>
<circle cx="360" cy="170" r="55" fill="none" stroke="#40575e" stroke-width="5"/><path d="M140 170L220 100L280 170Z M220 100H320L360 170M280 170L320 100" fill="none" stroke="#d1764b" stroke-width="6"/>
<ellipse cx="255" cy="63" rx="42" ry="30" fill="#fffdf3"/><path d="M270 48Q300 4 325 43L385 52L322 63" fill="#e4b25f"/><circle cx="322" cy="36" r="4"/></svg>
<img id="local-image" src="../assets/icon.svg" width="36" height="36"><button id="increment">点击测试交互</button><output id="counter">0</output><p id="data"></p>
<script type="module" src="../assets/app.mjs"></script></body></html>`;
const pagePath = path.join(project, "pages/pelican.html"); fs.writeFileSync(pagePath, document("鹈鹕骑自行车"));
fs.writeFileSync(path.join(project, "assets/style.css"), "body{margin:0;padding:24px;background:#fcf9ef;color:#304952;font:16px sans-serif}h1{color:rgb(24, 98, 75)}svg{border-radius:18px}button{margin:24px 12px;padding:12px;border:0;border-radius:8px;background:#dfa96c}#wheel{transform-origin:140px 170px;animation:spin 2s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}");
fs.writeFileSync(path.join(project, "assets/icon.svg"), '<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36"><circle cx="18" cy="18" r="16" fill="#568d70"/></svg>');
fs.writeFileSync(path.join(project, "assets/app.mjs"), 'import { label } from "./value.mjs";document.querySelector("#increment").onclick=()=>document.querySelector("#counter").textContent=String(Number(document.querySelector("#counter").textContent)+1);fetch("../assets/data.json").then(r=>r.json()).then(v=>document.querySelector("#data").textContent=label+v.text);');
fs.writeFileSync(path.join(project, "assets/value.mjs"), 'export const label="本地数据：";'); fs.writeFileSync(path.join(project, "assets/data.json"), '{"text":"资源加载正常"}');
atomicJSON(path.join(project, ".xuloom/sessions/html.json"), { messages: [{ role: "user", content: "制作动画" }, { role: "assistant", content: "已写入：`pages/pelican.html`。" }], metadata: {} });
atomicJSON(path.join(data, "workbench.json"), { projects: [project], workspace: [{ project, resumeId: "html", active: true }] });
let app;
async function main() {
  try {
    const env = { ...process.env, USERPROFILE: path.join(fixture, "home"), HOME: path.join(fixture, "home"), XULOOM_USER_DATA: data, XULOOM_CONFIG_DIR: config, XULOOM_TEST_HEADLESS: "1", USERPROFILE: home, HOME: home };
    for (const key of ["ELECTRON_RUN_AS_NODE", "XULOOM_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "XULOOM_MODEL_PROVIDER"]) delete env[key];
    app = await electron.launch({ executablePath: process.env.XULOOM_TEST_EXECUTABLE, args: process.env.XULOOM_TEST_EXECUTABLE ? [] : [root], env, timeout: 30000 });
    const page = await app.firstWindow(); page.setDefaultTimeout(20000);
    const errors = []; page.on("pageerror", (error) => errors.push(error.message));
    await page.locator(".inline-file-link").click();
    const preview = page.getByRole("region", { name: "文件预览", exact: true }), frame = page.frameLocator('iframe[title="HTML 网页预览"]');
    await frame.getByRole("heading", { name: "鹈鹕骑自行车", exact: true }).waitFor();
    assert.equal(await preview.getByRole("tab", { name: "预览", exact: true }).getAttribute("aria-selected"), "true");
    await frame.getByText("本地数据：资源加载正常", { exact: true }).waitFor();
    assert.equal(await frame.locator("h1").evaluate((node) => getComputedStyle(node).color), "rgb(24, 98, 75)");
    assert.equal(await frame.locator("#wheel").evaluate((node) => getComputedStyle(node).animationName), "spin");
    assert.equal(await frame.locator("#local-image").evaluate((node) => node.naturalWidth), 36);
    await frame.getByRole("button", { name: "点击测试交互", exact: true }).click(); assert.equal(await frame.locator("#counter").innerText(), "1");
    const isolation = await frame.locator("body").evaluate(() => {
      let parentBlocked = false; try { void parent.document.body; } catch { parentBlocked = true; }
      return { parentBlocked, bridge: typeof window.xuloom, require: typeof require, node: typeof process };
    });
    assert.deepEqual(isolation, { parentBlocked: true, bridge: "undefined", require: "undefined", node: "undefined" });
    const firstURL = await preview.locator("iframe").getAttribute("src"), mainURL = page.url();
    fs.writeFileSync(pagePath, document("刷新后的动画"));
    await preview.getByRole("button", { name: "刷新文件内容", exact: true }).click();
    await frame.getByRole("heading", { name: "刷新后的动画", exact: true }).waitFor();
    assert.equal(await frame.locator("#counter").innerText(), "0");
    const image = await app.evaluate(async ({ BrowserWindow }) => {
      const web = BrowserWindow.getAllWindows()[0].webContents; await web.capturePage(undefined, { stayHidden: true, stayAwake: true });
      await new Promise((resolve) => setTimeout(resolve, 100)); return (await web.capturePage(undefined, { stayHidden: true, stayAwake: true })).toPNG().toString("base64");
    });
    fs.mkdirSync(path.join(root, "test-results"), { recursive: true }); fs.writeFileSync(path.join(root, "test-results/html-preview.png"), Buffer.from(image, "base64"));
    await preview.getByRole("tab", { name: "源码", exact: true }).click(); await preview.locator(".file-source").waitFor();
    assert.equal(await preview.locator("iframe").count(), 0);
    await preview.getByRole("tab", { name: "预览", exact: true }).click(); await frame.getByRole("heading", { name: "刷新后的动画", exact: true }).waitFor();
    assert.notEqual(await preview.locator("iframe").getAttribute("src"), firstURL);
    const id = await page.locator(".session-item.active").getAttribute("data-session-id");
    assert.equal(await page.evaluate(async(id)=>{try{await window.xuloom.invoke("preview.open",{session:id,path:"../outside.html"});return true;}catch{return false;}},id), false);
    await preview.getByRole("button", { name: "关闭文件预览", exact: true }).click(); assert.equal(await page.locator("iframe").count(), 0);
    assert.equal(page.url(), mainURL); assert.equal(await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().length), 1);
    assert.deepEqual(errors, []); console.log("HTML E2E passed: embedded preview, CSS/SVG animation, relative images/scripts/modules/JSON, interaction, latest disk refresh, source switching, origin/IPC/Node isolation and workspace bounds");
  } finally { if (app) await app.close().catch(() => {}); }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
