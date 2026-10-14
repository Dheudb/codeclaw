const path = require("node:path");

// Resolve per-user data on the running OS; never store developer paths in a package.
function defaultWorkspace({ platform = process.platform, home, env = process.env }) {
  const paths = platform === "win32" ? path.win32 : path.posix;
  const absolute = (value) => typeof value === "string" && paths.isAbsolute(value);
  let data;
  if (platform === "win32") data = absolute(env.LOCALAPPDATA) ? env.LOCALAPPDATA : paths.join(home, "AppData", "Local");
  else if (platform === "darwin") data = paths.join(home, "Library", "Application Support");
  else data = absolute(env.XDG_DATA_HOME) ? env.XDG_DATA_HOME : paths.join(home, ".local", "share");
  return paths.join(data, platform === "linux" ? "codeclaw" : "CodeClaw", "workspaces", "default");
}

module.exports = { defaultWorkspace };
