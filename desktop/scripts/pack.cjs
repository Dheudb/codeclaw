const fs = require("node:fs");
const path = require("node:path");
if (process.platform !== "win32")
  throw new Error("This portable packaging script currently targets Windows.");
const root = path.resolve(__dirname, "..");
const output = path.join(root, "release", "CodeClaw-win32-" + process.arch);
const runtime = path.dirname(require("electron"));
fs.mkdirSync(output, { recursive: true });
fs.cpSync(runtime, output, { recursive: true });
const resources = path.join(output, "resources");
const application = path.join(resources, "app");
fs.mkdirSync(application, { recursive: true });
for (const name of ["dist", "electron"])
  fs.cpSync(path.join(root, name), path.join(application, name), {
    recursive: true,
  });
fs.writeFileSync(
  path.join(application, "package.json"),
  JSON.stringify(
    {
      name: "codeclaw-desktop",
      productName: "CodeClaw",
      version: "0.1.0",
      main: "electron/main.cjs",
    },
    null,
    2,
  ),
);
fs.cpSync(
  path.join(root, "../codeclaw"),
  path.join(resources, "engine/codeclaw"),
  {
    recursive: true,
    filter: (source) =>
      !source.includes("__pycache__") && !source.endsWith(".pyc"),
  },
);
fs.renameSync(
  path.join(output, "electron.exe"),
  path.join(output, "CodeClaw.exe"),
);
fs.writeFileSync(
  path.join(output, "START-HERE.txt"),
  "CodeClaw Desktop Preview\r\n\r\nRun CodeClaw.exe.\r\nA Python interpreter with CodeClaw dependencies is required.\r\nChoose the interpreter in Settings if python is not on PATH.\r\nThis preview does not bundle a Python runtime.\r\n",
);
console.log("Portable desktop: " + path.join(output, "CodeClaw.exe"));
