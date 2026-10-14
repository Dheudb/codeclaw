const test = require("node:test");
const assert = require("node:assert/strict");
const { defaultWorkspace } = require("../electron/workspace.cjs");

test("Windows uses each user's local app data including redirected locations", () => {
  assert.equal(defaultWorkspace({ platform: "win32", home: "C:\\Users\\Alice", env: {} }),
    "C:\\Users\\Alice\\AppData\\Local\\CodeClaw\\workspaces\\default");
  assert.equal(defaultWorkspace({ platform: "win32", home: "C:\\Users\\Bob", env: { LOCALAPPDATA: "D:\\用户数据" } }),
    "D:\\用户数据\\CodeClaw\\workspaces\\default");
});
test("Linux honors absolute XDG data home and ignores relative values", () => {
  assert.equal(defaultWorkspace({ platform: "linux", home: "/home/alice", env: {} }),
    "/home/alice/.local/share/codeclaw/workspaces/default");
  assert.equal(defaultWorkspace({ platform: "linux", home: "/home/bob", env: { XDG_DATA_HOME: "/mnt/data" } }),
    "/mnt/data/codeclaw/workspaces/default");
  assert.equal(defaultWorkspace({ platform: "linux", home: "/home/bob", env: { XDG_DATA_HOME: "relative" } }),
    "/home/bob/.local/share/codeclaw/workspaces/default");
});
test("macOS uses the current user's Application Support directory", () => {
  assert.equal(defaultWorkspace({ platform: "darwin", home: "/Users/陈", env: { LOCALAPPDATA: "C:\\ignored", XDG_DATA_HOME: "/ignored" } }),
    "/Users/陈/Library/Application Support/CodeClaw/workspaces/default");
});
