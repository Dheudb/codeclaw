const test = require("node:test");
const assert = require("node:assert/strict");
test("Markdown image references distinguish real images from unsafe/placeholder sources", async () => {
  const { imageReference: image } = await import("../src/image-reference.ts");
  assert.equal(image("https://example.org/photo?id=1").kind, "remote");
  assert.equal(image("http://example.org/p.png").kind, "remote");
  assert.equal(image("data:image/png;base64,aGVsbG8=").kind, "data");
  assert.deepEqual(image("figures/my%20plot.png"), { kind: "local", source: "figures/my plot.png" });
  assert.equal(image("C:\\workspace\\plot.png").kind, "local");
  assert.equal(image("/workspace/plot.jpg").kind, "local");
  for (const src of ["zzc", "", "javascript:alert(1)", "https://user:password@example.org/p.png", "file:///C:/Windows/p.png", "data:text/html,test", "data:image/svg+xml;base64,AAAA", "x\u0000.png"])
    assert.equal(image(src).kind, "invalid", src);
});
