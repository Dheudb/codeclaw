import { useEffect, useState } from "react";
import { ChevronDown, Terminal } from "lucide-react";
import type { TurnTrace } from "./types";
import Markdown from "./Markdown";
import { names, statusText } from "./ActivityPanel";

export function durationText(ms: number) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(total / 3600), minutes = Math.floor(total % 3600 / 60), seconds = total % 60;
  return `${hours ? hours + "小时" : ""}${minutes ? minutes + "分" : ""}${seconds}秒`;
}

export default function TurnProcess({ trace }: { trace: TurnTrace }) {
  const running = trace.status === "running";
  const [open, setOpen] = useState(running);
  const [now, setNow] = useState(Date.now());
  useEffect(() => { setOpen(running); }, [running]);
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running]);
  const elapsed = trace.elapsedMs ?? (trace.startedAt && (trace.finishedAt || (running ? now : 0))
    ? Math.max(0, (trace.finishedAt || now) - trace.startedAt) : undefined);
  const state = running ? "执行中" : trace.status === "cancelled" ? "已停止" : ["error", "interrupted"].includes(trace.status) ? "已中断" : "";
  return <details className="turn-process" open={open} onToggle={(event) => setOpen(event.currentTarget.open)} data-turn-status={trace.status}>
    <summary aria-label="查看本轮执行过程">
      <span>{state}{state && elapsed !== undefined ? " · " : ""}{elapsed !== undefined ? `${trace.approximate ? "约" : ""}用时 ${durationText(elapsed)}` : state ? "" : "执行过程"}</span>
      <ChevronDown size={14} /><span className="turn-step-count">{trace.steps.length ? `${trace.steps.length} 项过程` : ""}</span>
    </summary>
    <div className="turn-process-content">
      {!trace.steps.length && <span className="muted">{running ? "正在等待模型返回…" : "本轮没有工具操作或中间过程消息。"}</span>}
      {trace.steps.map((step, index) => step.kind === "text"
        ? <div className="turn-commentary markdown" key={index}><Markdown text={step.text} /></div>
        : <details className="turn-tool" key={index}>
          <summary><Terminal size={14} /><span>{names[step.tool.name] || step.tool.name}</span><span className="turn-tool-brief">{String(step.tool.input?.description || step.tool.input?.command || step.tool.input?.file_path || step.tool.input?.query || "")}</span><span className="turn-tool-status">{statusText[step.tool.status] || step.tool.status}</span></summary>
          <div><pre>{JSON.stringify(step.tool.input || {}, null, 2)}</pre><pre>{step.tool.result || "尚未返回结果。"}</pre></div>
        </details>)}
    </div>
  </details>;
}
