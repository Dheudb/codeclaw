const fs = require("node:fs");
const path = require("node:path");

function resolveEnginePython({
  packaged,
  resourcesPath,
  projectRoot,
  env = process.env,
}) {
  if (packaged) {
    const executable = path.join(resourcesPath, "python", "python.exe");
    if (!fs.existsSync(executable))
      throw new Error("应用内置运行时缺失，请重新解压完整的 Xuloom 程序包。");
    return executable;
  }
  if (env.XULOOM_PYTHON) return env.XULOOM_PYTHON;
  for (const directory of [".venv", ".venv-mcp"]) {
    const executable = path.join(
      projectRoot,
      directory,
      process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
    );
    if (fs.existsSync(executable)) return executable;
  }
  return "python";
}
module.exports = { resolveEnginePython };
