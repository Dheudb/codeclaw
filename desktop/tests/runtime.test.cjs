const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { resolveEnginePython } = require("../electron/runtime.cjs");

test("portable app uses its runtime and ignores developer/system overrides", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "codeclaw-runtime-test-"));
  try {
    const executable = path.join(root, "python/python.exe");
    fs.mkdirSync(path.dirname(executable), { recursive: true });
    fs.writeFileSync(executable, "fixture");
    assert.equal(
      resolveEnginePython({
        packaged: true,
        resourcesPath: root,
        projectRoot: "missing",
        env: { CODECLAW_PYTHON: "missing-system-python" },
      }),
      executable,
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("missing portable runtime reports a damaged package without a system fallback", () => {
  assert.throws(
    () =>
      resolveEnginePython({
        packaged: true,
        resourcesPath: path.join(os.tmpdir(), "codeclaw-missing-runtime"),
        env: { CODECLAW_PYTHON: "python" },
      }),
    /内置运行时缺失/,
  );
});

test("source development retains an explicit interpreter override", () => {
  assert.equal(
    resolveEnginePython({
      packaged: false,
      projectRoot: "/unused",
      env: { CODECLAW_PYTHON: "developer-python" },
    }),
    "developer-python",
  );
});
