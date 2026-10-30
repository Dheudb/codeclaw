const { _electron: electron } = require("playwright");
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), http = require("node:http"), assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { atomicJSON, sessionKey } = require("../electron/catalog.cjs");
const { historyRoot } = require("../electron/history-storage.cjs");
const root = path.resolve(__dirname, "..");
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "xuloom-markdown-images-"));
const project = path.join(fixture, "project"), data = path.join(fixture, "profile"), config = path.join(fixture, "config");
for (const dir of [project, data, config, path.join(project, "docs/figures")]) fs.mkdirSync(dir, { recursive: true });
const python = process.env.XULOOM_TEST_EXECUTABLE ? path.join(path.dirname(process.env.XULOOM_TEST_EXECUTABLE), "resources/python/python.exe") : path.join(root, "../.venv-mcp/Scripts/python.exe");
const picture = path.join(project, "docs/figures/plot.png");
execFileSync(python, ["-I", "-c", "from PIL import Image,ImageDraw;import sys;im=Image.new('RGB',(640,360),'#e9f3fa');d=ImageDraw.Draw(im);d.ellipse((470,30,580,140),fill='#f4cf76');d.polygon([(0,350),(190,110),(450,360)],fill='#89b5c8');im.save(sys.argv[1])", picture], { windowsHide: true });
const bytes = fs.readFileSync(picture);
fs.writeFileSync(path.join(project, "docs/report.md"), "# 图片报告\n![文档图](figures/plot.png)\n");
fs.writeFileSync(path.join(fixture, "private.png"), bytes);
const requests = [], errors = [];
const server = http.createServer((req, res) => {
  requests.push({ path: req.url, referer: req.headers.referer });
  if (req.url === "/figure") res.writeHead(200, { "Content-Type": "image/png" }).end(bytes);
  else res.writeHead(404).end("missing");
});
let app, page, original;
async function launch() {
  const env = { ...process.env, HOME: path.join(fixture, "home"), USERPROFILE: path.join(fixture, "home"), XULOOM_DATA_DIR: data, XULOOM_USER_DATA: data, XULOOM_CONFIG_DIR: config, XULOOM_TEST_HEADLESS: "1" };
  for (const key of ["ELECTRON_RUN_AS_NODE", "XULOOM_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "XULOOM_MODEL_PROVIDER"]) delete env[key];
  app = await electron.launch({ executablePath: process.env.XULOOM_TEST_EXECUTABLE, args: process.env.XULOOM_TEST_EXECUTABLE ? [] : [root], env });
  page = await app.firstWindow(); page.setDefaultTimeout(30000); page.on("pageerror", e => errors.push(e.message));
  await page.locator(".session-item.active").waitFor();
}
async function loaded(name, scope = page) {
  const button = scope.getByRole("button", { name: "放大图片 " + name, exact: true });
  await button.scrollIntoViewIfNeeded();
  await button.locator("img").waitFor();
  await page.waitForFunction(name => [...document.querySelectorAll("img")].some(im => im.alt === name && im.naturalWidth > 0), name);
  return button;
}
async function main() {
  try {
    await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
    const endpoint = `http://127.0.0.1:${server.address().port}`;
    const text = `**![](zzc)**\n\n![本地图](docs/figures/plot.png)\n\n![网络图](${endpoint}/figure)\n\n![坏链接](${endpoint}/missing)\n\n![不允许](../private.png)\n\n![危险](javascript:alert(1))\n\n[打开报告](docs/report.md)`;
    const history = path.join(project, ".xuloom/sessions/images.json");
    atomicJSON(history, { messages: [{ role: "user", content: "查看图像" }, { role: "assistant", content: text }], metadata: {} });
    original = fs.readFileSync(history);
    atomicJSON(path.join(data, "workbench.json"), { projects: [project], workspace: [{ project, resumeId: "images", active: true }] });
    await launch();
    const localReceipt = await page.evaluate(async ({ session, path }) => {
      try { const value = await window.xuloom.invoke("files.image", { session, path }); return { width: value.width }; }
      catch (error) { return { error: String(error) }; }
    }, { session: sessionKey(project, "images"), path: "docs/figures/plot.png" });
    assert.equal(localReceipt.width, 640, JSON.stringify(localReceipt));
    assert(!(await page.locator(".message.assistant").innerText()).includes("zzc"));
    assert(!(await page.locator(".message.assistant").innerText()).includes("javascript:alert"));
    const local = await loaded("本地图");
    await local.click(); await page.getByRole("dialog", { name: "图片附件预览" }).waitFor();
    assert.equal(await page.getByRole("dialog").locator("img").evaluate(im => im.naturalWidth), 640);
    await page.keyboard.press("Escape"); assert.equal(await page.getByRole("dialog").count(), 0);
    await (await loaded("网络图")).click();
    await page.getByRole("dialog", { name: "图片附件预览" }).waitFor();
    assert.equal(await page.getByRole("dialog").locator("img").getAttribute("referrerpolicy"), "no-referrer");
    await page.getByRole("button", { name: "关闭图片预览", exact: true }).click();
    await page.evaluate(() => document.querySelector('img[alt="坏链接"]')?.scrollIntoView());
    for (let i = 0; i < 100 && !requests.some(r => r.path === "/missing"); i++) await new Promise(resolve => setTimeout(resolve, 20));
    assert(requests.some(r => r.path === "/missing"));
    await page.waitForFunction(() => !document.querySelector('img[alt="坏链接"]'));
    assert.equal(await page.locator(".markdown-image-error").count(), 0);
    assert(!(await page.locator(".message.assistant").innerText()).includes("图片无法显示"));
    assert.equal(await page.getByRole("button", { name: "放大图片 不允许", exact: true }).count(), 0);
    assert(requests.some(r => r.path === "/figure"));
    assert(requests.every(r => !r.referer), "images must not send local conversation referrers");
    const initialURL = page.url();
    await page.getByRole("button", { name: "打开报告", exact: true }).click();
    await loaded("文档图", page.getByRole("region", { name: "文件预览" }));
    assert.equal(page.url(), initialURL);
    fs.mkdirSync(path.join(root, "test-results"), { recursive: true });
    const shot = await app.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage(undefined, { stayHidden: true, stayAwake: true })).toPNG().toString("base64"));
    fs.writeFileSync(path.join(root, "test-results/markdown-images.png"), Buffer.from(shot, "base64"));
    await app.close(); await launch();
    await loaded("本地图");
    assert.deepEqual(fs.readFileSync(path.join(historyRoot(project, data), "sessions/images.json")), original);
    assert.deepEqual(errors, []);
    console.log("Markdown image E2E passed: real local/remote images, enlargement, relative document image; invalid, unsafe and missing images omitted without placeholders; no referrer, restart and unchanged history");
  } finally { if (app) await app.close().catch(() => {}); server.closeAllConnections(); server.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
