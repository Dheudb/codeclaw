const fs = require("node:fs"), path = require("node:path");
const { mutateSync } = require("./storage-helper.cjs");

function readSettings(dataRoot, legacy) {
  const target = path.join(dataRoot, "workbench.json");
  if (legacy && target !== legacy && !fs.existsSync(target) && fs.existsSync(legacy))
    mutateSync({ dataRoot }, { action: "workbench", before: {}, after: {}, legacy });
  return fs.existsSync(target) ? JSON.parse(fs.readFileSync(target, "utf8").replace(/^\uFEFF/, "")) : {};
}
function saveSettings(dataRoot, after, before) {
  if (!before) throw new Error("Desktop preferences must include their original snapshot");
  return mutateSync({ dataRoot }, { action: "workbench", before, after }).settings;
}
module.exports = { readSettings, saveSettings };
