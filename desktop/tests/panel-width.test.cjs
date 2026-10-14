const test = require("node:test");
const assert = require("node:assert/strict");
test("panel width fits the viewport while leaving room for chat", async () => {
  const { clampPanelWidth } = await import("../src/panel-width.ts");
  assert.equal(clampPanelWidth(560, 1440), 560);
  assert.equal(clampPanelWidth(760, 1050), 430);
  assert.equal(clampPanelWidth(-20, 1440), 280);
  assert.equal(clampPanelWidth(2000, 2200), 760);
  assert.equal(clampPanelWidth(NaN, 1440), 318);
});
