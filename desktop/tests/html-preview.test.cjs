const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs/promises"), path = require("node:path"), os = require("node:os");
const { HtmlPreviews } = require("../electron/html-preview.cjs");

test("HTML preview serves current workspace assets and rejects hidden files, traversal, writes and expired pages", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "xuloom-html-unit-"));
  const project = path.join(root, "project"); await fs.mkdir(path.join(project, "nested"), { recursive: true });
  await fs.writeFile(path.join(project, "nested/page.html"), "<h1>preview</h1>");
  await fs.writeFile(path.join(project, "style.css"), "body {color:red}");
  await fs.writeFile(path.join(project, "config.py"), "must not expose");
  await fs.writeFile(path.join(project, ".secret.json"), "{}");
  await fs.writeFile(path.join(root, "outside.html"), "outside");
  const pages = new HtmlPreviews(), view = await pages.create("session-a", project, path.join(project, "nested/page.html"));
  const request = (relative, method = "GET") => pages.respond(new Request(new URL(relative, view.url), { method }));
  assert.match(view.url, /^xuloom-preview:\/\/[a-z0-9-]+\/nested\/page.html$/);
  assert.match(await (await request(view.url)).text(), /preview/);
  const css = await request("../style.css"); assert.equal(css.headers.get("content-type"), "text/css; charset=utf-8");
  assert.equal(css.headers.get("cache-control"), "no-store");
  await fs.writeFile(path.join(project, "style.css"), "body {color:blue}");
  assert.match(await (await request("../style.css")).text(), /blue/);
  assert.equal((await request("../.secret.json")).status, 403);
  assert.equal((await request("../config.py")).status, 415);
  assert.notEqual((await request("../../outside.html")).status, 200);
  assert.equal((await request(view.url, "POST")).status, 405);
  assert.equal((await request("..%2f..%2foutside.html")).status, 403);
  pages.revoke(view.token, "other-session"); assert.equal((await request(view.url)).status, 200);
  pages.revokeSession("session-a"); assert.equal((await request(view.url)).status, 404);
});

test("explicitly modified external HTML does not grant access to its neighboring files", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "xuloom-html-outside-"));
  const project = path.join(root, "project"); await fs.mkdir(project);
  await fs.writeFile(path.join(root, "page.htm"), "outside preview"); await fs.writeFile(path.join(root, "data.json"), "{}");
  const pages = new HtmlPreviews(), view = await pages.create("a", project, path.join(root, "page.htm"));
  assert.equal((await pages.respond(new Request(view.url))).status, 200);
  assert.equal((await pages.respond(new Request(new URL("data.json", view.url)))).status, 403);
});
