const test = require("node:test");
const assert = require("node:assert/strict");
const preferences = import("../src/catalog-preferences.ts");
const snapshot = (revision, sort = "recent") => ({ revision, projects: [], view: { sort } });
test("refresh and old acknowledgements cannot undo a later sort", async () => {
  const { CatalogPreferences } = await preferences;
  const state = new CatalogPreferences(snapshot(1));
  const first = state.begin({ sort: "name" });
  const last = state.begin({ sort: "oldest" });
  assert.equal(state.receive(snapshot(2, "name")).view.sort, "oldest");
  assert.equal(state.finish(last.id, snapshot(3, "oldest")).view.sort, "oldest");
  assert.equal(state.finish(first.id, snapshot(2, "name")).view.sort, "oldest");
});
test("failed sort writes roll back independently", async () => {
  const { CatalogPreferences } = await preferences;
  const state = new CatalogPreferences(snapshot(1));
  const first = state.begin({ sort: "name" });
  const last = state.begin({ sort: "oldest" });
  assert.equal(state.fail(first.id).view.sort, "oldest");
  assert.equal(state.fail(last.id).view.sort, "recent");
});
