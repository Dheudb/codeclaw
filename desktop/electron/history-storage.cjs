const fs = require("node:fs"), path = require("node:path"), crypto = require("node:crypto");
const { dataDirectory } = require("./workspace.cjs");
const os = require("node:os");
const key = (project) => crypto.createHash("sha256").update(process.platform === "win32" ? path.resolve(project).toLowerCase() : path.resolve(project)).digest("hex").slice(0, 16);
const historyRoot = (project, data = dataDirectory({ home: os.homedir() })) => path.join(data, "history", key(project));
function rebase(value, from, to) {
  if (typeof value === "string") {
    const converted = value.split(from).join(to);
    return from.includes("\\") ? converted.split(from.replaceAll("\\", "/")).join(to.replaceAll("\\", "/")) : converted;
  }
  if (Array.isArray(value)) return value.map((item) => rebase(item, from, to));
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([name, item]) => [name, rebase(item, from, to)]));
  return value;
}
function migrateProject(project, data, atomicJSON) {
  const target = historyRoot(project, data), legacy = path.join(project, ".codeclaw");
  const sessions = path.join(legacy, "sessions"), index = path.join(legacy, "session-index.json"), artifacts = path.join(legacy, "tool-results");
  const runtimeKinds = ["browser", "shell-tasks", "vcr", "notebooks"];
  if (![sessions, index, artifacts, ...runtimeKinds.map((name) => path.join(legacy, name))].some((filename) => fs.existsSync(filename))) return target;
  if (fs.lstatSync(legacy).isSymbolicLink()) throw new Error("不能迁移链接形式的历史目录");
  const journalPath = path.join(target, "migration.json");
  const journal = fs.existsSync(journalPath) ? JSON.parse(fs.readFileSync(journalPath, "utf8")) : {};
  function transfer(source, destination, transform = false) {
    if (!fs.existsSync(source)) return;
    const stat = fs.lstatSync(source);
    if (stat.isSymbolicLink()) throw new Error("不能迁移链接形式的会话资源");
    if (stat.isDirectory()) {
      for (const name of fs.readdirSync(source)) transfer(path.join(source, name), path.join(destination, name), transform);
      try { fs.rmdirSync(source); } catch (error) { if (!["ENOTEMPTY", "EACCES", "EPERM"].includes(error.code)) throw error; }
      return;
    }
    const original = fs.readFileSync(source), hash = crypto.createHash("sha256").update(original).digest("hex");
    let content = original;
    if (transform && source.endsWith(".json")) {
      try {
        let value = JSON.parse(original);
        for (const name of ["tool-results", ...runtimeKinds]) {
          value = rebase(value, path.join(legacy, name), path.join(target, name));
          value = rebase(value, path.join(".codeclaw", name), path.join(target, name));
        }
        content = Buffer.from(JSON.stringify(value, null, 2));
      } catch { /* Preserve malformed records for recovery. */ }
    }
    if (journal[source] !== hash && fs.existsSync(destination)) {
      if (!fs.readFileSync(destination).equals(content)) throw new Error("旧会话与集中存储冲突，已保留双方记录：" + source);
    } else if (journal[source] !== hash) {
      fs.mkdirSync(path.dirname(destination), { recursive: true }); fs.writeFileSync(destination, content, { flag: "wx" });
      fs.utimesSync(destination, stat.atime, stat.mtime);
    }
    journal[source] = hash; atomicJSON(journalPath, journal);
    if (fs.readFileSync(source).equals(original)) { try { fs.unlinkSync(source); } catch (error) { if (!["EACCES", "EPERM"].includes(error.code)) throw error; } }
    else throw new Error("迁移过程中旧会话被其他程序修改，原文件保留");
  }
  transfer(artifacts, path.join(target, "tool-results"));
  for (const name of runtimeKinds) transfer(path.join(legacy, name), path.join(target, name));
  transfer(sessions, path.join(target, "sessions"), true);
  const targetIndex = path.join(target, "index.json");
  if (fs.existsSync(index) && fs.existsSync(targetIndex)) {
    const old = JSON.parse(fs.readFileSync(index, "utf8")), current = JSON.parse(fs.readFileSync(targetIndex, "utf8"));
    if (old.version !== 1 || current.version !== 1) throw new Error("旧会话索引格式不正确");
    atomicJSON(targetIndex, { version: 1, sessions: { ...old.sessions, ...current.sessions } });
    try { fs.unlinkSync(index); } catch (error) { if (!["EACCES", "EPERM"].includes(error.code)) throw error; }
  } else transfer(index, targetIndex);
  atomicJSON(path.join(target, "project.json"), { version: 1, project: path.resolve(project) });
  return target;
}
module.exports = { historyRoot, migrateProject, rebase };
