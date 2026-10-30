import { useEffect, useState } from "react";
import { ChevronDown, LoaderCircle } from "lucide-react";
import type { TurnTrace, ToolActivity } from "./types";
import Markdown from "./Markdown";
import { statusText } from "./ActivityPanel";
import { toolPresentation } from "./tool-presentation";
import ToolIcon from "./ToolIcon";
import ToolImages from "./ToolImages";

export function durationText(ms: number) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(total / 3600), minutes = Math.floor(total % 3600 / 60), seconds = total % 60;
  return `${hours ? hours + "小时" : ""}${minutes ? minutes + "分" : ""}${seconds}秒`;
}

function ProcessTool({ tool, sessionId }: { tool: ToolActivity; sessionId: string }) {
  const view = toolPresentation(tool), pending = ["running", "queued"].includes(tool.status);
  const elapsed = tool.startedAt && tool.finishedAt && tool.finishedAt - tool.startedAt >= 1000 ? durationText(tool.finishedAt - tool.startedAt) : "";
  return <div className="turn-tool-entry">
    <details className={"turn-tool " + tool.status}>
      <summary title={view.brief}>
        <ToolIcon kind={view.kind} /><span className="turn-tool-title">{view.title}</span>
        <span className="turn-tool-brief">{view.brief}</span>
        {pending && <LoaderCircle size={12} className="spin turn-tool-spinner" aria-label="正在执行" />}
        {tool.status === "waiting" && <span className="turn-tool-status">{statusText[tool.status]}</span>}
        {elapsed && <span className="turn-tool-duration">{elapsed}</span>}<ChevronDown size={12} className="turn-tool-chevron" />
      </summary>
      <div className="turn-tool-detail"><small>调用参数</small><pre>{JSON.stringify(tool.input || {}, null, 2)}</pre><small>执行结果</small><pre>{tool.result || "尚未返回结果。"}</pre></div>
    </details>
    <ToolImages media={tool.media} sessionId={sessionId} />
  </div>;
}

export default function TurnProcess({ trace, sessionId }: { trace: TurnTrace; sessionId: string }) {
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
        : <ProcessTool tool={step.tool} sessionId={sessionId} key={step.tool.id || index} />)}
    </div>
  </details>;
}
