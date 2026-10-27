const fs = require("node:fs"), path = require("node:path");
const { projectKey, sessionKey } = require("./catalog.cjs");
const { mutateSync } = require("./storage-helper.cjs");

function prepareMove(catalog, row, directory, scope = "project") {
  if (typeof directory !== "string" || !path.isAbsolute(directory)) throw new Error("请选择目标项目");
  const target = fs.realpathSync(directory);
  if (!fs.statSync(target).isDirectory()) throw new Error("目标项目目录不存在");
  const project = catalog.snapshot().projects.find((entry) => projectKey(entry.path) === projectKey(target));
  if (scope !== "standalone" && (!project || project.archived)) throw new Error("目标项目不存在或已归档");
  if (row.error || row.archived && scope !== "standalone") throw new Error("请先修复或恢复此会话，再移动");
  return { target, sameDirectory: projectKey(row.project) === projectKey(target) };
}

function moveSpec(row, target, scope = "project") {
  return { action: "move", project: row.project, session_id: row.sessionId, target, scope, title: row.title, created: row.created };
}

function moveSession(catalog, row, directory, scope = "project") {
  const { target } = prepareMove(catalog, row, directory, scope);
  mutateSync(catalog, moveSpec(row, target, scope));
  return catalog.find(sessionKey(target, row.sessionId));
}
module.exports = { prepareMove, moveSpec, moveSession };
