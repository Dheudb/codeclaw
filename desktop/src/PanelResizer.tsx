import { useEffect, useRef, useState } from "react";
import { clampPanelWidth, DEFAULT_PANEL_WIDTH, panelLimits } from "./panel-width";

export default function PanelResizer({ width, viewport, change, save }: {
  width: number; viewport: number; change: (value: number) => void; save: (value: number) => void;
}) {
  const drag = useRef<{ x: number; width: number; current: number } | null>(null);
  const [resizing, setResizing] = useState(false);
  const limits = panelLimits(viewport);
  useEffect(() => {
    if (resizing) document.body.classList.add("panel-resizing");
    return () => document.body.classList.remove("panel-resizing");
  }, [resizing]);
  function finish() {
    if (!drag.current) return;
    save(drag.current.current); drag.current = null; setResizing(false);
  }
  return <div className={"panel-resizer " + (resizing ? "dragging" : "")} role="separator"
    aria-label="调整右侧面板宽度" aria-orientation="vertical" aria-valuemin={limits.min} aria-valuemax={limits.max} aria-valuenow={width}
    title="拖动调整宽度，双击恢复默认宽度；方向键也可调整" tabIndex={0}
    onPointerDown={(event) => {
      if (event.button !== 0) return;
      event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
      drag.current = { x: event.clientX, width, current: width }; setResizing(true);
    }}
    onPointerMove={(event) => {
      if (!drag.current) return;
      const value = clampPanelWidth(drag.current.width + drag.current.x - event.clientX, viewport);
      drag.current.current = value; change(value);
    }} onPointerUp={finish} onPointerCancel={finish} onLostPointerCapture={finish}
    onDoubleClick={() => { const value = clampPanelWidth(DEFAULT_PANEL_WIDTH, viewport); change(value); save(value); }}
    onKeyDown={(event) => {
      const value = event.key === "ArrowLeft" ? width + 24 : event.key === "ArrowRight" ? width - 24 :
        event.key === "Home" ? limits.min : event.key === "End" ? limits.max : null;
      if (value === null) return;
      event.preventDefault(); const next = clampPanelWidth(value, viewport); change(next); save(next);
    }} />;
}
