export function imageReference(value?: string): { kind: "remote" | "data" | "local" | "invalid"; source: string } {
  const source = (value || "").trim();
  if (!source || source.length > 28_000_000) return { kind: "invalid", source };
  try {
    const url = new URL(source);
    if (["http:", "https:"].includes(url.protocol) && !url.username && !url.password) return { kind: "remote", source: url.href };
  } catch { /* Relative local paths are resolved by the session, never the renderer. */ }
  if (/^data:image\/(?:png|jpeg|gif|webp|bmp);base64,[a-z0-9+/=\r\n]+$/i.test(source)) return { kind: "data", source };
  let path: string;
  try { path = decodeURIComponent(source); } catch { return { kind: "invalid", source }; }
  if (!/[\0\r\n<>]/.test(path) && /\.(?:png|jpe?g|gif|webp|bmp)$/i.test(path) &&
      (!/^[a-z][\w+.-]*:/i.test(path) || /^[a-z]:[\\/]/i.test(path))) return { kind: "local", source: path };
  return { kind: "invalid", source };
}
