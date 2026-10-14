import type { SessionCatalog } from "./types.ts";

type ViewPatch = Partial<SessionCatalog["view"]>;

// Keep the user's latest intent visible until its own write is acknowledged.
// Refreshes and older IPC replies must not undo a newer checkbox click.
export class CatalogPreferences {
  private server: SessionCatalog;
  private pending = new Map<number, ViewPatch>();
  private sequence = 0;

  constructor(initial: SessionCatalog) { this.server = initial; }
  visible(): SessionCatalog {
    const view = { ...this.server.view };
    for (const patch of this.pending.values()) Object.assign(view, patch);
    return { ...this.server, view };
  }
  receive(value: SessionCatalog): SessionCatalog {
    if ((value.revision || 0) >= (this.server.revision || 0)) this.server = value;
    return this.visible();
  }
  begin(patch: ViewPatch) {
    const id = ++this.sequence;
    // Only the latest write owns a field, even if acknowledgements arrive out of order.
    for (const pending of this.pending.values()) {
      for (const key of Object.keys(patch) as (keyof ViewPatch)[]) delete pending[key];
    }
    this.pending.set(id, { ...patch });
    return { id, catalog: this.visible() };
  }
  finish(id: number, value: SessionCatalog): SessionCatalog {
    this.receive(value);
    this.pending.delete(id);
    return this.visible();
  }
  fail(id: number): SessionCatalog {
    this.pending.delete(id);
    return this.visible();
  }
}
