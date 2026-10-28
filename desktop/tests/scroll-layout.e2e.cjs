const { _electron: electron } = require('playwright');
const fs = require('node:fs'), path = require('node:path'), os = require('node:os');
const http = require('node:http'), assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'xuloom-scroll-layout-'));
const data = path.join(fixture, 'data'), config = path.join(fixture, 'config');
for (const directory of [data, config]) fs.mkdirSync(directory);
fs.writeFileSync(path.join(data, 'workbench.json'), JSON.stringify({ projects: [], activeConversation: null }));
let app, page, finishStream, holdStream = false;
const errors = [];
const paragraph = index => `第 ${index} 段：这是用于验证滚动位置的长回复。用户阅读历史内容时，新的回答内容不应把视图拉回底部。\n\n`;
const server = http.createServer((req, res) => {
  let raw = ''; req.on('data', chunk => raw += chunk);
  req.on('end', () => {
    const body = JSON.parse(raw);
    if (!body.stream) { res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'fixture memory' } }] })); return; }
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    const chunk = (content, finish_reason = null) => res.write('data: ' + JSON.stringify({ id: 'fixture', object: 'chat.completion.chunk', choices: [{ index: 0, delta: content ? { role: 'assistant', content } : {}, finish_reason }] }) + '\n\n');
    chunk(Array.from({ length: 90 }, (_, index) => paragraph(index)).join(''));
    let index = 90;
    const timer = setInterval(() => { if (!holdStream) chunk(paragraph(index++)); }, 180);
    finishStream = () => { clearInterval(timer); chunk('', 'stop'); res.end('data: [DONE]\n\n'); };
    res.on('close', () => clearInterval(timer));
  });
});
async function launch() {
  const env = { ...process.env, XULOOM_DATA_DIR: data, XULOOM_USER_DATA: data, XULOOM_CONFIG_DIR: config,
    XULOOM_TEST_HEADLESS: '1', USERPROFILE: path.join(fixture, 'home'), HOME: path.join(fixture, 'home'), XULOOM_SKILLS_PATHS: '' };
  for (const name of ['ELECTRON_RUN_AS_NODE', 'XULOOM_DEV_URL', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL', 'XULOOM_MODEL_PROVIDER']) delete env[name];
  app = await electron.launch({ executablePath: process.env.XULOOM_TEST_EXECUTABLE, args: process.env.XULOOM_TEST_EXECUTABLE ? [] : [root], env, timeout: 90000 });
  page = await app.firstWindow(); page.setDefaultTimeout(90000);
  page.on('pageerror', error => errors.push(error.message));
  await page.waitForFunction(() => document.querySelector('.new-task')?.disabled === false);
  await page.getByRole("button", { name: "展开能力面板", exact: true }).click();
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setContentSize(1600, 1000));
}
const metrics = () => page.locator('.conversation').evaluate(element => ({ top: element.scrollTop, height: element.clientHeight, total: element.scrollHeight }));
async function wheel(delta) {
  const box = await page.locator('.conversation').boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel(0, delta);
}
async function settledScroll() {
  await page.evaluate(() => { delete window.__scrollLayoutSample; });
  await page.waitForFunction(() => {
    const top = document.querySelector('.conversation').scrollTop;
    const previous = window.__scrollLayoutSample;
    if (!previous || Math.abs(top - previous.top) > 0.01) {
      window.__scrollLayoutSample = { top, at: performance.now() }; return false;
    }
    return performance.now() - previous.at >= 200;
  }, undefined, { timeout: 10000 });
}
async function dragSidebar(target) {
  const separator = page.getByRole('separator', { name: '调整历史栏宽度' });
  const before = Number(await separator.getAttribute('aria-valuenow'));
  const box = await separator.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + 150); await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + target - before, box.y + 150, { steps: 10 }); await page.mouse.up();
  await page.waitForFunction(value => Number(document.querySelector('.sidebar-resizer').getAttribute('aria-valuenow')) === value, target);
  await page.waitForFunction(value => Math.round(document.querySelector('.sidebar').getBoundingClientRect().width) === value, target);
}
(async () => {
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    fs.writeFileSync(path.join(config, 'config.json'), JSON.stringify({ model_provider: 'openai', openai_model: 'scroll-fixture', openai_api_key: 'fixture-key', openai_base_url: `http://127.0.0.1:${server.address().port}/v1` }));
    await launch(); await page.locator('.new-task').click(); await page.locator('.session-item.active').waitFor();
    await page.getByRole('textbox', { name: '任务输入' }).fill('SCROLL_FIXTURE');
    await page.getByRole('button', { name: '发送任务', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('.conversation').scrollHeight > 3000);
    const unobstructed = await page.locator('.conversation').evaluate(element => {
      const box = element.getBoundingClientRect();
      return !document.elementFromPoint(box.right - 3, box.y + box.height / 2)?.closest('.panel-resizer');
    });
    assert(unobstructed, 'Inspector resizer must not intercept the conversation scrollbar');
    await page.waitForFunction(() => { const e = document.querySelector('.conversation'); return e.scrollHeight - e.clientHeight - e.scrollTop <= 2; });
    // Keep the stream open, but freeze its content while measuring a small gesture.
    // Otherwise a new chunk can move the tail between measuring and dispatching input.
    holdStream = true;
    await new Promise(resolve => setTimeout(resolve, 250));
    const tail = await metrics();
    await wheel(-24);
    await page.waitForFunction(top => document.querySelector('.conversation').scrollTop < top - 15, tail.top, { timeout: 10000 });
    await settledScroll();
    const smallScroll = await metrics();
    holdStream = false;
    await page.waitForFunction(total => document.querySelector('.conversation').scrollHeight > total + 40, smallScroll.total);
    assert(Math.abs((await metrics()).top - smallScroll.top) < 15, 'Small wheel/trackpad movements must also pause following');
    const beforeLargeScroll = await metrics();
    await wheel(-1800);
    await page.waitForFunction(top => document.querySelector('.conversation').scrollTop < top - 500, beforeLargeScroll.top, { timeout: 10000 });
    await settledScroll();
    const reading = await metrics();
    await page.waitForFunction(total => document.querySelector('.conversation').scrollHeight > total + 80, reading.total);
    const afterUpdates = await metrics();
    assert(Math.abs(afterUpdates.top - reading.top) < 60, 'Stream updates must preserve the reader position: ' + JSON.stringify({ reading, afterUpdates }));
    finishStream(); finishStream = undefined;
    await page.waitForFunction(() => !document.querySelector('.send-button.stop'));
    // Drag the actual native scrollbar thumb after the final layout has settled.
    const box = await page.locator('.conversation').boundingBox();
    const dragStart = await metrics();
    const gutter = await page.locator('.conversation').evaluate(element => element.offsetWidth - element.clientWidth);
    const track = dragStart.height - 2 * gutter;
    const thumbHeight = Math.max(20, track * dragStart.height / dragStart.total);
    const thumb = gutter + dragStart.top / (dragStart.total - dragStart.height) * (track - thumbHeight);
    const x = box.x + box.width - gutter / 2, y = box.y + thumb + thumbHeight / 2;
    await page.mouse.move(x, y); await page.mouse.down();
    await page.mouse.move(x, Math.min(box.y + box.height - gutter - 10, y + 120), { steps: 12 }); await page.mouse.up();
    await page.waitForFunction(top => document.querySelector('.conversation').scrollTop > top + 100, dragStart.top, { timeout: 10000 });
    const stopped = await metrics(); await wheel(-650);
    await page.waitForFunction(top => document.querySelector('.conversation').scrollTop < top - 100, stopped.top);
    await settledScroll();
    const up = await metrics(); await wheel(400);
    await page.waitForFunction(top => document.querySelector('.conversation').scrollTop > top + 100, up.top);
    await settledScroll();
    let sidebar = page.getByRole('separator', { name: '调整历史栏宽度' });
    await dragSidebar(380);
    await sidebar.focus(); await page.keyboard.press('ArrowRight'); assert.equal(await sidebar.getAttribute('aria-valuenow'), '404');
    await page.getByRole('separator', { name: '调整右侧面板宽度' }).focus(); await page.keyboard.press('ArrowLeft');
    assert.equal(JSON.parse(fs.readFileSync(path.join(data, 'workbench.json'))).layout.sidebarWidth, 404);
    assert.equal(JSON.parse(fs.readFileSync(path.join(data, 'workbench.json'))).layout.inspectorWidth, 342);
    await page.getByRole('button', { name: '收起能力面板' }).click();
    assert.equal(await sidebar.getAttribute('aria-valuenow'), '404');
    await page.getByRole('button', { name: '展开能力面板' }).click();
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setContentSize(1050, 820));
    await page.waitForFunction(() => {
      const main = document.querySelector('.main-panel').getBoundingClientRect();
      return innerWidth <= 1060 && main.width >= 399 && document.documentElement.scrollWidth <= innerWidth;
    });
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setContentSize(1600, 1000));
    await page.waitForFunction(() => document.querySelector('.sidebar').getBoundingClientRect().width === 404);
    await app.close(); await launch(); await page.locator('.session-item.active').waitFor();
    sidebar = page.getByRole('separator', { name: '调整历史栏宽度' });
    assert.equal(await sidebar.getAttribute('aria-valuenow'), '404');
    assert.equal(await page.getByRole('separator', { name: '调整右侧面板宽度' }).getAttribute('aria-valuenow'), '342');
    await sidebar.dblclick(); assert.equal(await sidebar.getAttribute('aria-valuenow'), '244');
    assert.deepEqual(errors, []);
    console.log('PASS: native wheel and scrollbar drag, reading during streaming, left/right resize, independent saved widths, narrow-window bounds, restart and reset.');
  } catch (error) {
    if (page && !page.isClosed()) console.error('Scroll/layout diagnostics:', await page.evaluate(() => {
      const e = document.querySelector('.conversation');
      return { top:e.scrollTop, total:e.scrollHeight, height:e.clientHeight, gutter:e.offsetWidth-e.clientWidth,
        resizing:document.body.className, sidebar:document.querySelector('.sidebar').getBoundingClientRect().width };
    }).catch(() => ({})));
    throw error;
  } finally { finishStream?.(); if (app) await app.close().catch(() => {}); server.closeAllConnections(); server.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
