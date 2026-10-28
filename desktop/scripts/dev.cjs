const { spawn } = require("node:child_process");
const path = require("node:path");
const http = require("node:http");
const root = path.resolve(__dirname, "..");
const vite = spawn(
  process.execPath,
  [path.join(root, "node_modules/vite/bin/vite.js")],
  { cwd: root, stdio: "inherit", windowsHide: true },
);
let electron;
let closing = false;
function close() {
  if (closing) return;
  closing = true;
  vite.kill();
  electron?.kill();
}
async function ready() {
  for (let i = 0; i < 100; i++) {
    if (vite.exitCode !== null) throw new Error("Vite failed to start");
    if (
      await new Promise((resolve) => {
        const request = http.get("http://127.0.0.1:5178", (response) => {
          response.resume();
          resolve(response.statusCode === 200);
        });
        request.on("error", () => resolve(false));
        request.setTimeout(500, () => {
          request.destroy();
          resolve(false);
        });
      })
    )
      return;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error("Vite startup timed out");
}
ready()
  .then(() => {
    const env = { ...process.env, XULOOM_DEV_URL: "http://127.0.0.1:5178" };
    delete env.ELECTRON_RUN_AS_NODE;
    electron = spawn(require("electron"), [root], {
      cwd: root,
      stdio: "inherit",
      env,
      windowsHide: true,
    });
    electron.on("exit", (code) => {
      close();
      process.exitCode = code || 0;
    });
  })
  .catch((error) => {
    console.error(error);
    close();
    process.exitCode = 1;
  });
process.on("SIGINT", close);
process.on("SIGTERM", close);
process.on("exit", close);
