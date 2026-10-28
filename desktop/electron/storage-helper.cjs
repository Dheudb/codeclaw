// Offline catalog callers use the same storage actor as the desktop's async RPC.
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { resolveEnginePython } = require("./runtime.cjs");
function mutateSync(catalog, spec) {
  const projectRoot = path.resolve(__dirname, "../.."), resourcesPath = process.resourcesPath;
  const packaged = !!process.versions.electron && !process.defaultApp;
  const python = resolveEnginePython({ packaged, resourcesPath, projectRoot });
  const args = [...(packaged ? ["-I"] : []), "-m", "xuloom.core.session_mutation"];
  const options = { windowsHide: true, cwd: projectRoot, input: JSON.stringify(spec), encoding: "utf8",
    stdio: ["pipe", "pipe", "pipe"], env: { ...process.env, XULOOM_DATA_DIR: catalog.dataRoot,
      PYTHONIOENCODING: "utf-8", PYTHONPATH: projectRoot } };
  let output;
  try { output = execFileSync(python, args, options); }
  catch (error) {
    let detail;
    try { detail = JSON.parse(error.stdout).error; } catch {}
    throw new Error(detail || "Session storage actor failed: " + String(error.stderr || error.message));
  }
  return JSON.parse(output);
}
module.exports = { mutateSync };
