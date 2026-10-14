const fs = require("node:fs/promises");
const path = require("node:path");
const { randomUUID } = require("node:crypto");

const SCHEME = "codeclaw-preview";
const TYPES = {
  ".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json", ".txt": "text/plain; charset=utf-8", ".csv": "text/csv; charset=utf-8",
  ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
  ".gif": "image/gif", ".webp": "image/webp", ".avif": "image/avif", ".ico": "image/x-icon",
  ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf", ".otf": "font/otf",
  ".mp4": "video/mp4", ".webm": "video/webm", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
  ".wasm": "application/wasm",
};
function inside(root, filename) {
  const relative = path.relative(root, filename);
  return relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative);
}

class HtmlPreviews {
  constructor() { this.pages = new Map(); }
  async create(session, project, filename) {
    const file = await fs.realpath(filename), workspace = await fs.realpath(project);
    if (!/\.html?$/i.test(file) || !(await fs.stat(file)).isFile()) throw new Error("请选择 HTML 文件");
    // An explicitly modified file outside the project may be previewed alone.
    const root = inside(workspace, file) ? workspace : path.dirname(file);
    const token = randomUUID();
    this.pages.set(token, { session, root, file, assets: inside(workspace, file) });
    return { token, url: `${SCHEME}://${token}/${path.relative(root, file).split(path.sep).map(encodeURIComponent).join("/")}` };
  }
  revoke(token, session) { if (this.pages.get(token)?.session === session) this.pages.delete(token); }
  revokeSession(session) { for (const [token, page] of this.pages) if (page.session === session) this.pages.delete(token); }
  async respond(request) {
    const denied = (status, message) => new Response(message, { status, headers: { "content-type": "text/plain; charset=utf-8" } });
    try {
      const url = new URL(request.url), page = this.pages.get(url.hostname);
      if (url.protocol !== SCHEME + ":" || !page) return denied(404, "预览已关闭，请重新打开文件。");
      if (!["GET", "HEAD"].includes(request.method)) return denied(405, "预览仅支持读取文件。");
      const relative = decodeURIComponent(url.pathname).replace(/^\/+/, "");
      if (/[\0\\:]/.test(relative)) return denied(403, "无法读取此资源。");
      const candidate = path.resolve(page.root, relative);
      if (!inside(page.root, candidate)) return denied(403, "资源超出工作目录。");
      const file = await fs.realpath(candidate);
      if (!inside(page.root, file) || (!page.assets && file !== page.file) ||
          path.relative(page.root, file).split(path.sep).some((part) => part.startsWith("."))) return denied(403, "无法读取此资源。");
      const type = TYPES[path.extname(file).toLowerCase()];
      if (!type) return denied(415, "此资源类型不支持网页预览。");
      const stat = await fs.stat(file);
      if (!stat.isFile()) return denied(404, "资源不存在。");
      if (stat.size > 30 * 1024 * 1024) return denied(413, "资源超过预览大小限制（30 MB）。");
      return new Response(request.method === "HEAD" ? null : await fs.readFile(file), { headers: {
        "content-type": type, "cache-control": "no-store", "x-content-type-options": "nosniff",
        "content-security-policy": "default-src 'self' data: blob: https: http:; script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: https: http:; style-src 'self' 'unsafe-inline' https: http:; object-src 'none'; frame-src 'none'; base-uri 'self'; form-action 'none'",
      } });
    } catch { return denied(404, "文件已移动、删除，或资源不可读取。"); }
  }
}
module.exports = { HtmlPreviews, SCHEME };
