const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
if (process.platform !== "win32")
  throw new Error("Portable packaging currently targets Windows.");
const root = path.resolve(__dirname, "..");
const repository = path.resolve(root, "..");
const candidates = [
  process.env.CODECLAW_BUILD_PYTHON,
  process.env.CODECLAW_PYTHON,
  path.join(repository, ".venv/Scripts/python.exe"),
  path.join(repository, ".venv-mcp/Scripts/python.exe"),
].filter(Boolean);
const python = candidates.find((value) => fs.existsSync(value)) || "python";
execFileSync(python, [path.join(__dirname, "prepare-runtime.py")], {
  cwd: repository,
  stdio: "inherit",
  windowsHide: true,
});
const release = path.join(root, "release");
const output = path.join(release, "CodeClaw-win32-" + process.arch);
fs.mkdirSync(release, { recursive: true });
if (
  path.dirname(output) !== release ||
  (fs.existsSync(output) &&
    (fs.lstatSync(output).isSymbolicLink() ||
      path.dirname(fs.realpathSync(output)) !== fs.realpathSync(release)))
)
  throw new Error("Unsafe package output path");
fs.rmSync(output, { recursive: true, force: true });
fs.cpSync(path.dirname(require("electron")), output, { recursive: true });
const resources = path.join(output, "resources");
const application = path.join(resources, "app");
fs.mkdirSync(application, { recursive: true });
for (const name of ["dist", "electron"])
  fs.cpSync(path.join(root, name), path.join(application, name), {
    recursive: true,
  });
const metadata = JSON.parse(
  fs.readFileSync(path.join(root, "package.json"), "utf8"),
);
fs.writeFileSync(
  path.join(application, "package.json"),
  JSON.stringify(
    {
      name: metadata.name,
      productName: "CodeClaw",
      version: metadata.version,
      main: "electron/main.cjs",
    },
    null,
    2,
  ),
);
fs.cpSync(path.join(root, "build/python"), path.join(resources, "python"), {
  recursive: true,
});
fs.renameSync(
  path.join(output, "electron.exe"),
  path.join(output, "CodeClaw.exe"),
);
fs.copyFileSync(
  path.join(repository, "LICENSE"),
  path.join(output, "LICENSE-CodeClaw.txt"),
);
fs.writeFileSync(
  path.join(output, "START-HERE.txt"),
  "CodeClaw Desktop\r\n\r\nExtract the complete folder, then run CodeClaw.exe.\r\nPython and MCP dependencies are included. No Python installation or interpreter selection is required.\r\nOpen a project and configure a model endpoint to begin.\r\nThird-party MCP programs may have their own runtime requirements.\r\nPython's license and dependency metadata are under resources/python.\r\n",
);
console.log("Self-contained desktop: " + path.join(output, "CodeClaw.exe"));
