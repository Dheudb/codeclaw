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
  const elapsed = trace.elapsedMs ?? (trace.resumedAt && running ? (trace.elapsedBeforeResumeMs || 0) + Math.max(0, now - trace.resumedAt) : trace.startedAt && (trace.finishedAt || (running ? now : 0))
    ? Math.max(0, (trace.finishedAt || now) - trace.startedAt) : undefined);
  const state = running ? "执行中" : trace.status === "cancelled" ? "已停止" : ["error", "interrupted"].includes(trace.status) ? "已中断" : "";
  const stream = trace.modelStreams?.at(-1);
  const phase = stream && running ? ({ waiting: "等待接口响应", connected: "等待模型输出", reasoning: "模型正在推理", finishing: "正在结束接收" } as Record<string, string>)[stream.status] : "";
  const timing = stream ? `本轮 ${trace.modelStreams?.length} 次模型请求；最近一次：${stream.firstTextMs !== undefined ? `首个文本 ${(stream.firstTextMs / 1000).toFixed(1)} 秒` : "尚未收到文本"}；已收到 ${stream.textChars || 0} 字符；推理数据 ${stream.reasoningChunks || 0} 段${stream.finishReason ? `；结束标记 ${stream.finishReason}` : ""}${stream.error ? `；${stream.error}` : ""}` : undefined;
  const label = <span title={timing}>{trace.status === "cancelled" && elapsed !== undefined
    ? `你在 ${durationText(elapsed)} 后停止了`
    : <>{state}{state && elapsed !== undefined ? " · " : ""}{elapsed !== undefined ? `${trace.approximate ? "约" : ""}用时 ${durationText(elapsed)}` : state ? "" : "执行过程"}{phase ? ` · ${phase}` : ""}</>}</span>;
  if (!trace.steps.some((step) => step.kind === "tool")) {
    return elapsed === undefined && !state ? null : <div className="turn-process turn-process-time" data-turn-status={trace.status}>{label}</div>;
  }
  return <details className="turn-process" open={open} onToggle={(event) => setOpen(event.currentTarget.open)} data-turn-status={trace.status}>
    <summary aria-label="查看本轮执行过程">
      {label}
      <ChevronDown size={14} /><span className="turn-step-count">{trace.steps.length ? `${trace.steps.length} 项过程` : ""}</span>
    </summary>
    <div className="turn-process-content">
      {trace.steps.map((step, index) => step.kind === "text"
        ? <div className="turn-commentary markdown" key={index}><Markdown text={step.text} /></div>
        : <details className="turn-tool" key={index}>
          <summary><Terminal size={14} /><span>{names[step.tool.name] || step.tool.name}</span><span className="turn-tool-brief">{String(step.tool.input?.description || step.tool.input?.command || step.tool.input?.file_path || step.tool.input?.query || "")}</span><span className="turn-tool-status">{statusText[step.tool.status] || step.tool.status}</span></summary>
          <div><pre>{JSON.stringify(step.tool.input || {}, null, 2)}</pre><pre>{step.tool.result || "尚未返回结果。"}</pre></div>
        </details>)}
    </div>
  </details>;
}
