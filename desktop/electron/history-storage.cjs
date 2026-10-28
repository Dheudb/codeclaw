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
  const target = historyRoot(project, data), legacy = path.join(project, ".xuloom");
  const kinds = ["sessions", "session-index.json", "tool-results", "browser", "shell-tasks", "vcr", "notebooks"];
  if (kinds.some((name) => fs.existsSync(path.join(legacy, name))))
    require("./storage-helper.cjs").mutateSync({ dataRoot: data }, { action: "migrate", project });
  return target;
}
module.exports = { historyRoot, migrateProject, rebase };
