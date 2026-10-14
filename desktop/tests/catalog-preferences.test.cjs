const test = require("node:test");
const assert = require("node:assert/strict");
const preferences = import("../src/catalog-preferences.ts");
const snapshot = (revision, showArchived, sort = "recent") => ({ revision, projects: [], view: { showArchived, sort } });

test("refresh and old acknowledgements cannot undo a later archive click", async () => {
  const { CatalogPreferences } = await preferences;
  const state = new CatalogPreferences(snapshot(1, false));
  const on = state.begin({ showArchived: true });
  const off = state.begin({ showArchived: false });
  assert.equal(state.receive(snapshot(2, true)).view.showArchived, false);
  assert.equal(state.finish(on.id, snapshot(2, true)).view.showArchived, false);
  assert.equal(state.finish(off.id, snapshot(3, false)).view.showArchived, false);
  assert.equal(state.receive(snapshot(2, true)).view.showArchived, false);
});
test("out of order acknowledgements do not revive superseded pending fields", async () => {
  const { CatalogPreferences } = await preferences;
  const state = new CatalogPreferences(snapshot(1, false));
  const on = state.begin({ showArchived: true });
  const off = state.begin({ showArchived: false });
  assert.equal(state.finish(off.id, snapshot(3, false)).view.showArchived, false);
  assert.equal(state.finish(on.id, snapshot(2, true)).view.showArchived, false);
});
test("a failed older write preserves the latest click and independent sort", async () => {
  const { CatalogPreferences } = await preferences;
  const state = new CatalogPreferences(snapshot(1, false));
  const on = state.begin({ showArchived: true });
  const sort = state.begin({ sort: "name" });
  const off = state.begin({ showArchived: false });
  assert.deepEqual(state.fail(on.id).view, { showArchived: false, sort: "name" });
  state.finish(off.id, snapshot(2, false));
  assert.equal(state.visible().view.sort, "name");
  assert.equal(state.finish(sort.id, snapshot(3, false, "name")).view.sort, "name");
});
test("a failed latest write rolls back to the acknowledged preference", async () => {
  const { CatalogPreferences } = await preferences;
  const state = new CatalogPreferences(snapshot(1, false));
  const on = state.begin({ showArchived: true });
  state.receive(snapshot(2, true)); state.finish(on.id, snapshot(2, true));
  const off = state.begin({ showArchived: false });
  assert.equal(state.fail(off.id).view.showArchived, true);
});
